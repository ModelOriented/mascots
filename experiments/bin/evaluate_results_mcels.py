import json
from pathlib import Path

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
from experiments.data.utils import (
    read_borf_res,
    read_glacier_res,
    read_mcels_res,
)
from experiments.models.classifier import GradientInceptionTimeClassifier


def main() -> None:
    out_dir = Path("experiments/out/results")
    out_dir.mkdir(parents=True, exist_ok=True)

    res = {}
    for data_name in UNI_DATASETS + MULTI_DATASETS:

        logger.info(f"Start {data_name}")

        data_res = {}
        res[data_name] = data_res

        model_path = Path(
            f"experiments/out/models/classifier/{data_name}/{tf.__version__}/model_no_padding"
        )
        model = GradientInceptionTimeClassifier.load(model_path)
        X_train, _, X_test, y_test = get_data(data_name)
        X_cf, _ = get_cf_data(X_test, y_test)

        mcels_cfs = read_mcels_res(
            f"experiments/out/cf/mcels/{data_name}/saliency_cf.npy"
        )
        logger.info(f"{mcels_cfs.shape=}")

        valid_idx = model.predict_cls(X_cf) == model.predict_cls(mcels_cfs)

        iforest = IsolationForest(contamination=0.01)
        n_feats = X_train.shape[1] * X_train.shape[2]
        iforest.fit(X_train.reshape(-1, n_feats))

        y_pred = model.predict_cls(X_test)
        data_res["model_acc"] = (y_pred == y_test).mean()
        data_res["mcels"] = {
            "validity": validity(X_cf, mcels_cfs, model.predict_cls),
            "proximity-all": euclidean_distance(
                X_cf, mcels_cfs, normalize=True
            ),
            "proximity": euclidean_distance(
                X_cf[valid_idx], mcels_cfs[valid_idx], normalize=True
            ),
            "compactness-all": compactness(X_cf, mcels_cfs),
            "compactness": compactness(X_cf[valid_idx], mcels_cfs[valid_idx]),
            "outlier-factor-all": (
                iforest.predict(mcels_cfs.reshape((-1, n_feats))) == 1
            ).mean(),
            "outlier-factor": (
                iforest.predict(mcels_cfs[valid_idx].reshape((-1, n_feats)))
                == 1
            ).mean(),
        }

    with open(out_dir / "mcels_results.json", "w") as f:
        json.dump(res, f, indent=4)


if __name__ == "__main__":
    main()
