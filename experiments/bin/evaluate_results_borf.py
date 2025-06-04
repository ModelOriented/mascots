import json
import pickle as pkl
from pathlib import Path

import numpy as np
import tensorflow as tf
from loguru import logger
from sklearn.ensemble import IsolationForest

from experiments.data.data import (
    MULTI_DATASETS,
    UNI_DATASETS,
    get_cf_data,
    get_data,
)
from experiments.data.metrics import compactness, euclidean_distance, validity
from experiments.data.utils import read_borf_res, read_glacier_res
from experiments.models.classifier import GradientInceptionTimeClassifier


def main(results_name: str) -> None:
    out_dir = Path("experiments/out/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    res = {}
    for data_name in UNI_DATASETS + MULTI_DATASETS:

        try:

            logger.info(f"Start {data_name}")

            data_res = {}
            res[data_name] = data_res

            model_path = Path(
                f"experiments/out/models/classifier/{data_name}/{tf.__version__}/model_no_padding"
            )
            model = GradientInceptionTimeClassifier.load(model_path)
            X_train, _, X_test, y_test = get_data(data_name)
            X_cf, _ = get_cf_data(X_test, y_test)

            obs, borf_cfs = read_borf_res(
                f"experiments/out/cf/{results_name}/{data_name}/assets/out.pkl",
                only_first=True,
            )
            obs = np.vstack(obs)
            obs[np.isnan(obs)] = 0

            borf_cfs[np.isnan(borf_cfs)] = 0

            y_pred = model.predict_cls(borf_cfs)
            # y_pred = y_pred.reshape(-1, 5)
            # idxs = np.argmax(y_pred != model.predict_cls(obs), axis=1)
            # borf_cfs = borf_cfs.reshape(
            # -1, 5, borf_cfs.shape[1], borf_cfs.shape[2]
            # )
            # obs = borf_cfs.reshape(-1, 5, borf_cfs.shape[1], borf_cfs.shape[2])
            # borf_cfs = borf_cfs[:, idxs, :, :]
            # borf_cfs = borf_cfs[:, 0, :, :]

            # obs = obs[:, idxs, :, :]
            # obs = obs[:, 0, :, :]

            with open(
                f"experiments/out/cf/{results_name}/{data_name}/assets/build_res.pkl",
                "rb",
            ) as f:
                build_res = pkl.load(f)
            logger.info(build_res)

            y_pred = model.predict_cls(X_test)
            data_res["model_acc"] = (y_pred == y_test).mean()

            valid_idx = model.predict_cls(X_cf) == model.predict_cls(borf_cfs)

            iforest = IsolationForest(contamination=0.01)
            n_feats = X_train.shape[1] * X_train.shape[2]
            iforest.fit(X_train.reshape(-1, n_feats))

            data_res["borf"] = {
                "validity": validity(obs, borf_cfs, model.predict_cls),
                "proximity-all": euclidean_distance(
                    obs, borf_cfs, normalize=True
                ),
                "proximity": euclidean_distance(
                    obs[valid_idx], borf_cfs[valid_idx], normalize=True
                ),
                "compactness-all": compactness(obs, borf_cfs),
                "compactness": compactness(
                    obs[valid_idx], borf_cfs[valid_idx]
                ),
                "outlier-factor-all": (
                    iforest.predict(borf_cfs.reshape((-1, n_feats))) == 1
                ).mean(),
                "outlier-factor": (
                    iforest.predict(borf_cfs[valid_idx].reshape((-1, n_feats)))
                    == 1
                ).mean(),
            }

            build_res["borf_build_out"]["accuracy"] = float(
                build_res["borf_build_out"]["accuracy"]
            )
            build_res["borf_build_out"]["cross-entropy"] = float(
                build_res["borf_build_out"]["cross-entropy"]
            )
            if "mse" in build_res["borf_build_out"].keys():
                build_res["borf_build_out"]["mse"] = float(
                    build_res["borf_build_out"]["mse"]
                )
            data_res["build_res"] = build_res

        except Exception as e:
            logger.error(data_name)

    logger.info(res)
    with open(out_dir / f"{results_name}_results.json", "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    main("borf")
    main("borf-scalar-C=0.0")
