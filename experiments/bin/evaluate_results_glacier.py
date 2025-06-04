import json
from pathlib import Path

import tensorflow as tf
from loguru import logger
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler

from experiments.competitors.glacier.src.help_functions import (
    conditional_pad,
    time_series_normalize,
)
from experiments.data.data import (
    MULTI_DATASETS,
    UNI_DATASETS,
    get_cf_data,
    get_data,
)
from experiments.data.metrics import compactness, euclidean_distance, validity
from experiments.data.utils import read_borf_res, read_glacier_res
from experiments.models.classifier import GradientInceptionTimeClassifier


def main() -> None:
    out_dir = Path("experiments/out/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    res = {}
    for data_name in UNI_DATASETS:

        logger.info(f"Start {data_name}")

        data_res = {}
        res[data_name] = data_res

        model_path = Path(
            f"experiments/out/models/classifier/{data_name}/{tf.__version__}/model_padding"
        )
        model = GradientInceptionTimeClassifier.load(model_path)
        X_train, _, X_test, y_test = get_data(data_name)
        X_cf_org, _ = get_cf_data(X_test, y_test)
        n_timesteps = X_train.shape[2]
        scaler = MinMaxScaler()
        scaler.fit(X_train.reshape(X_train.shape[0], -1))

        X_train_processed, trained_scaler = time_series_normalize(
            data=X_train, n_timesteps=n_timesteps
        )
        X_test_processed, _ = time_series_normalize(
            data=X_test, n_timesteps=n_timesteps, scaler=trained_scaler
        )
        X_test, _ = conditional_pad(X_test_processed)

        X_cf, _ = get_cf_data(X_test_processed, y_test)

        # X_cf, _ = time_series_normalize(
        # data=X_cf, n_timesteps=n_timesteps, scaler=trained_scaler
        # )

        X_cf, _ = conditional_pad(X_cf)

        glacier_cfs = read_glacier_res(
            f"experiments/out/cf/glacier/{data_name}/cf.pkl"
        )

        glacier_cfs, _ = time_series_normalize(
            data=glacier_cfs,
            n_timesteps=glacier_cfs.shape[-1],
            scaler=trained_scaler,
        )

        glacier_cfs_rescaled = scaler.inverse_transform(
            glacier_cfs.reshape(glacier_cfs.shape[0], -1)
        )

        glacier_cfs, _ = conditional_pad(glacier_cfs)

        # print(glacier_cfs.shape)
        # raise Exception

        # X_test, X_cf = X_test, X_cf.swapaxes(1, 2)
        # glacier_cfs = glacier_cfs.swapaxes(1, 2)

        y_pred = model.predict_cls(X_test)
        data_res["model_acc"] = (y_pred == y_test).mean()

        valid_idx = model.predict_cls(X_cf) == model.predict_cls(glacier_cfs)

        iforest = IsolationForest(contamination=0.01)
        n_feats = X_train.shape[1] * X_train.shape[2]
        iforest.fit(X_train.reshape(-1, n_feats))

        data_res["glacier"] = {
            "validity": validity(X_cf, glacier_cfs, model.predict_cls),
            "proximity-all": euclidean_distance(
                X_cf_org, glacier_cfs_rescaled, normalize=True
            ),
            "proximity": euclidean_distance(
                X_cf_org[valid_idx],
                glacier_cfs_rescaled[valid_idx],
                normalize=True,
            ),
            "compactness-all": compactness(X_cf_org, glacier_cfs_rescaled),
            "compactness": compactness(
                X_cf_org[valid_idx], glacier_cfs_rescaled[valid_idx]
            ),
            "outlier-factor-all": (
                iforest.predict(glacier_cfs_rescaled.reshape((-1, n_feats)))
                == 1
            ).mean(),
            "outlier-factor": (
                iforest.predict(
                    glacier_cfs_rescaled[valid_idx].reshape((-1, n_feats))
                )
                == 1
            ).mean(),
        }

    with open(out_dir / "glacier_results.json", "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    main()
