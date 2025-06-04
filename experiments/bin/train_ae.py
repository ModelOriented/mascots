#!/usr/bin/env python
# coding: utf-8
import os
import pickle as pkl
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import tensorflow as tf
from aeon.classification.deep_learning import InceptionTimeClassifier
from loguru import logger
from sklearn.metrics import balanced_accuracy_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from tensorflow import keras
from tensorflow.keras.utils import to_categorical
from wildboar.datasets import load_dataset

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
from experiments.data.data import MULTI_DATASETS, UNI_DATASETS, get_data
from experiments.models.ae import Autoencoder

os.environ["TF_DETERMINISTIC_OPS"] = "1"
config = tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)

TASK_ID = int(os.getenv("TASK_ID", "0"))

tf.random.set_seed(1)


def train_ae(name: str) -> None:

    X_train, y_train, X_test, y_test = get_data(name)
    X_train = X_train.reshape((X_train.shape[0], X_train.shape[2]))

    X_train, X_val, y_train, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.125,
        random_state=32,
        stratify=y_train,
    )

    # Upsample the minority class
    y_train_copy = y_train.copy()
    pos_label, neg_label = 1, 0
    # X_train, y_train = upsample_minority(
    #     X_train, y_train, pos_label=pos_label, neg_label=neg_label
    # )

    # ### 1.1 Normalization - fit scaler using training data
    n_training, n_timesteps = X_train.shape
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
    # X_test_processed_padded, _ = conditional_pad(X_test_processed)
    n_timesteps_padded = X_train_processed_padded.shape[1]

    autoencoder = Autoencoder(n_timesteps_padded, n_features)
    optimizer = keras.optimizers.Adam(learning_rate=0.0005)
    autoencoder.compile(optimizer=optimizer, loss="mse")

    # Define the early stopping criteria
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_loss",
        min_delta=0.0001,
        patience=5,
        restore_best_weights=True,
    )
    # Train the model
    reset_seeds()
    logger.info("Training log for 1dCNN autoencoder:")

    autoencoder_history = autoencoder.fit(
        X_train_processed_padded,
        X_train_processed_padded,
        epochs=1000,
        batch_size=32,
        shuffle=True,
        verbose=True,
        validation_data=(X_val_processed_padded, X_val_processed_padded),
        callbacks=[early_stopping],
    )

    out_path = Path(f"experiments/out/models/ae/{name}/{name}.h5")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    autoencoder.save(out_path)

    with open(out_path.parent / "history.pkl", "wb") as f:
        pkl.dump(autoencoder_history, f)


def main() -> None:
    names = UNI_DATASETS
    name = names[TASK_ID]
    train_ae(name)


if __name__ == "__main__":
    main()
