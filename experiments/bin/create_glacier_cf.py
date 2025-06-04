#!/usr/bin/env python
# coding: utf-8
import logging
import os
import pickle as pkl
from argparse import ArgumentParser
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from aeon.classification.deep_learning import InceptionTimeClassifier
from keras.layers import Reshape
from loguru import logger

# from keras_models import *
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import to_categorical
from wildboar.datasets import load_dataset

from experiments.competitors.glacier.src._guided import get_global_weights
from experiments.competitors.glacier.src.help_functions import (
    ResultWriter,
    conditional_pad,
    evaluate,
    find_best_lr,
    fit_evaluation_models,
    remove_paddings,
    reset_seeds,
    time_series_normalize,
    upsample_minority,
)
from experiments.data.data import UNI_DATASETS, get_cf_data, get_data
from experiments.models.ae import Autoencoder
from experiments.models.classifier import GradientInceptionTimeClassifier

TASK_ID = int(os.getenv("TASK_ID", "0"))

os.environ["TF_DETERMINISTIC_OPS"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

tf.random.set_seed(1)


def main() -> None:

    w_type = "uniform"
    pred_margin_weight = 0.5  # W_value
    tau_value = 0.5

    RANDOM_STATE = 123

    data_name = UNI_DATASETS[TASK_ID]

    ae_path = Path(f"experiments/out/models/ae/{data_name}/{data_name}.h5")
    model_path = Path(
        f"experiments/out/models/classifier/{data_name}/{tf.__version__}/model_padding"
    )
    out_path = Path(f"experiments/out/cf/glacier/{data_name}")
    out_path.mkdir(exist_ok=True, parents=True)

    classifier = GradientInceptionTimeClassifier.load(str(model_path))
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    classifier.compile(
        optimizer=optimizer, loss="binary_crossentropy", metrics=["accuracy"]
    )

    autoencoder = load_model(ae_path, compile=False)
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    autoencoder.compile(optimizer=optimizer, loss="mse")

    # 1. Load data
    X_train, y_train, X_test, y_test = get_data(data_name)

    # X_train, X_test = X_train.swapaxes(1, 2), X_test.swapaxes(1, 2)

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.125,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )

    nb_classes = len(np.unique(y_train))
    y_train_classes, y_val_classes, y_test_classes = (
        y_train.copy(),
        y_val.copy(),
        y_test.copy(),
    )

    pos_label, neg_label = 1, 0
    classes = np.unique(y_train)

    label_mapper = np.vectorize(lambda x: {classes[0]: 0, classes[1]: 1}[x])

    y_train = label_mapper(y_train)
    y_test = label_mapper(y_test)
    y_val = label_mapper(y_val)

    y_train, y_val, y_test = (
        to_categorical(y_train, nb_classes),
        to_categorical(y_val, nb_classes),
        to_categorical(y_test, nb_classes),
    )

    # ### 1.1 Normalization - fit scaler using training data
    n_training, _, n_timesteps = X_train.shape
    n_features = 1

    X_train_processed, trained_scaler = time_series_normalize(
        data=X_train, n_timesteps=n_timesteps
    )
    X_val_processed, _ = time_series_normalize(
        data=X_val, n_timesteps=n_timesteps, scaler=trained_scaler
    )
    X_test_processed, _ = time_series_normalize(
        data=X_test, n_timesteps=n_timesteps, scaler=trained_scaler
    )

    # add extra padding zeros if n_timesteps cannot be divided by 4, required for 1dCNN autoencoder structure
    X_train_processed_padded, padding_size = conditional_pad(X_train_processed)
    X_val_processed_padded, _ = conditional_pad(X_val_processed)
    X_test_processed_padded, _ = conditional_pad(X_test_processed)
    n_timesteps_padded = X_train_processed_padded.shape[2]

    y_train_pred, y_test_pred = classifier.predict_cls(
        X_train_processed_padded, permute_dim=True
    ), classifier.predict_cls(X_test_processed_padded, permute_dim=True)

    # Get 50 samples for CF evaluation if test size larger than 50
    rand_X_test, rand_y_pred = get_cf_data(
        X_test_processed_padded, y_test_pred, seed=RANDOM_STATE
    )

    logger.info(
        f"Data pre-processed, original #timesteps={n_timesteps}, padded #timesteps={n_timesteps_padded}."
    )

    # ### 1.2 Evaluation models
    # n_neighbors_lof = int(np.cbrt(X_train_processed.shape[0]))
    # lof_estimator_pos, nn_model_pos = fit_evaluation_models(
    #     n_neighbors_lof=n_neighbors_lof,
    #     n_neighbors_nn=1,
    #     training_data=np.squeeze(
    #         X_train_processed[y_train_classes == pos_label]
    #     ),
    # )
    # lof_estimator_neg, nn_model_neg = fit_evaluation_models(
    #     n_neighbors_lof=n_neighbors_lof,
    #     n_neighbors_nn=1,
    #     training_data=np.squeeze(
    #         X_train_processed[y_train_classes == neg_label]
    #     ),
    # )
    logger.info(f"LOF and NN estimators trained for dataset: [[{data_name}]].")

    # ## 2. LatentCF models
    # reset seeds for numpy, tensorflow, python random package and python environment seed
    reset_seeds()

    y_pred = classifier.predict_cls(X_test_processed_padded, permute_dim=True)
    y_pred_classes = y_pred

    # ### 2.0.1 Get `step_weights` based on the input argument
    if w_type == "global":
        step_weights = get_global_weights(
            X_train_processed_padded,
            y_train_classes,
            classifier,
            random_state=RANDOM_STATE,
        )
    elif w_type == "uniform":
        step_weights = np.ones((1, n_timesteps_padded, n_features))
    elif w_type.lower() == "local":
        step_weights = "local"
    elif w_type == "unconstrained":
        step_weights = np.zeros((1, n_timesteps_padded, n_features))
    else:
        raise NotImplementedError(
            "A.w_type not implemented, please choose 'local', 'global', 'uniform', or 'unconstrained'."
        )

    ###############################################
    # ## 2.1 CF search with 1dCNN autoencoder
    ###############################################
    # ### 1dCNN autoencoder

    # Get these instances for CF evaluation; class abnormal (0) VS normal class (1)

    lr_list = [0.001, 0.0001]
    best_lr, best_cf_model, best_cf_samples, _ = find_best_lr(
        classifier,
        X_samples=rand_X_test,
        pred_labels=rand_y_pred,
        autoencoder=autoencoder,
        lr_list=lr_list,
        pred_margin_weight=pred_margin_weight,
        step_weights=step_weights,
        random_state=RANDOM_STATE,
        padding_size=padding_size,
        target_prob=tau_value,
    )
    logger.info(f"The best learning rate found is {best_lr}.")

    # predicted probabilities of CFs
    z_pred = classifier.predict(best_cf_samples)
    cf_pred_labels = np.argmax(z_pred, axis=1)

    # remove extra paddings after counterfactual generation in 1dCNN autoencoder
    best_cf_samples = remove_paddings(best_cf_samples, padding_size)
    best_cf_samples = trained_scaler.inverse_transform(best_cf_samples)
    # use the unpadded X_test for evaluation
    rand_X_test_original = np.squeeze(rand_X_test)

    # evaluate_res = evaluate(
    #     rand_X_test_original,
    #     best_cf_samples,
    #     rand_y_pred,
    #     cf_pred_labels,
    #     lof_estimator_pos,
    #     lof_estimator_neg,
    #     nn_model_pos,
    #     nn_model_neg,
    # )

    with open(out_path / "cf.pkl", "wb") as f:
        pkl.dump(best_cf_samples, f)
    # with open(out_path / "glacier_eval.pkl", "wb") as f:
    # pkl.dump(evaluate_res, f)

    logger.info("Done")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        level=logging.DEBUG,
    )
    main()
