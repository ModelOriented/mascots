import json
import os
import shutil
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from loguru import logger
from numpy.typing import NDArray

from bin.utils.configs import (
    alternative_borf_explainer_build_args,
    alternative_borf_explainer_counterfactual_args,
    alternative_borf_explainer_init_args,
    default_borf_explainer_build_args,
    default_borf_explainer_counterfactual_args,
    default_borf_explainer_init_args,
)
from mascots.explainer.pipeline import get_borf_config
from mascots.metrics.evaluation import Evaluator
from mascots.metrics.utils import get_current_timestamp
from experiments.data.data import (
    MULTI_DATASETS,
    UNI_DATASETS,
    get_cf_data,
    get_data,
)
from experiments.models.classifier import GradientInceptionTimeClassifier

N_SAMPLES = 50
N_COUNTERFACTUALS = 1

TASK_ID = int(os.getenv("TASK_ID", "0"))
SWAP_METHOD = os.getenv("SWAP_METHOD", "scalar")
C_VAL = float(os.getenv("C", "0.1"))

logger.info(f'{C_VAL=}')
logger.info(f'{TASK_ID=}')
logger.info(f'{SWAP_METHOD=}')


data_names = UNI_DATASETS + MULTI_DATASETS
data_name = data_names[TASK_ID]
OUT_DIR = f"experiments/out/cf/borf-{SWAP_METHOD}-C={C_VAL}/{data_name}"


def __get_obj_name(obj: Any) -> str:
    if isinstance(obj, (str, int, float)):
        cls_name = str(obj)
    elif isinstance(obj, partial):
        cls_name = f"{obj.__getattribute__('func').__name__};"
        for key, val in obj.__getattribute__("keywords").items():
            cls_name += f"{key}={val};"
    elif isinstance(obj, type):
        cls_name = obj.__name__
    else:
        cls_name = type(obj).__name__

    if cls_name[-1] == ";":
        cls_name = cls_name[:-1]

    return cls_name


def __dict_vals_to_repr(d: dict) -> dict:

    new_d = deepcopy(d)

    def dict_vals_to_repr(d: dict) -> None:
        for key, val in d.items():
            if not isinstance(val, dict):
                d[key] = __get_obj_name(val)
            else:
                dict_vals_to_repr(d[key])

    dict_vals_to_repr(new_d)
    return new_d


def create_evaluation(
    exp_config_idx: int,
    model: Any,
    X_train: NDArray[np.float64],
    y_train: NDArray[np.int64],
    X_cnt: NDArray[np.float64],
    y_cnt: NDArray[np.float64],
    y_target_cnt: NDArray[np.int64],
    n_counterfactuals: int,
) -> None:

    exp_config = configs[exp_config_idx]
    exp_config["borf_explainer_init_args"]["borf_config"] = get_borf_config(
        X_train.shape
    )

    logger.info(f"Data size: {X_train.shape}")

    res_dir = Path(OUT_DIR)

    evaluator = Evaluator(
        exp_config["borf_explainer_init_args"]
        | {
            "prediction_fn": model.predict_cls,
            "prediction_fn_proba": model.predict,
        },
        X_train,
        y_train,
    )

    logger.info("Build evaluator")

    evaluator.build(exp_config["borf_explainer_build_args"])

    logger.info("Evaluate")
    exp_config["borf_explainer_counterfactual_args"][
        "swap_method"
    ] = SWAP_METHOD
    exp_config['borf_explainer_counterfactual_args']['C'] =C_VAL

    evaluator.evaluate(
        X_cnt,
        y_cnt,
        y_target_cnt,
        n_counterfactuals,
        exp_config["borf_explainer_counterfactual_args"],
    )

    evaluator.save_results(res_dir)

    with open(res_dir / "result.json", "w") as f:
        json.dump(__dict_vals_to_repr(exp_config), f)


def generate_configs() -> tuple[list[dict[dict[str, Any]]], list[str]]:
    base_config = {
        "borf_explainer_init_args": default_borf_explainer_init_args,
        "borf_explainer_build_args": default_borf_explainer_build_args,
        "borf_explainer_counterfactual_args": default_borf_explainer_counterfactual_args,
    }

    configs = [base_config]
    names = ["all=default"]

    for key, val_list in alternative_borf_explainer_init_args.items():
        for val in val_list:
            new_config = deepcopy(base_config)
            new_config["borf_explainer_init_args"][key] = deepcopy(val)
            configs.append(new_config)
            names.append(f"{key}={__get_obj_name(val)}")

    for key, val_list in alternative_borf_explainer_build_args.items():
        for val in val_list:
            new_config = deepcopy(base_config)
            new_config["borf_explainer_build_args"][key] = deepcopy(val)
            configs.append(new_config)
            names.append(f"{key}={__get_obj_name(val)}")

    for (
        key,
        val_list,
    ) in alternative_borf_explainer_counterfactual_args.items():
        for val in val_list:
            new_config = deepcopy(base_config)
            new_config["borf_explainer_counterfactual_args"][key] = deepcopy(
                val
            )
            configs.append(new_config)
            names.append(f"{key}={__get_obj_name(val)}")

    return configs, names


def main(exp_config_idx: int, exp_config_name: str) -> None:

    X_train, y_train, X_test, y_test = get_data(data_name)
    X_cf, y_cf = get_cf_data(X_test, y_test, n_samples=N_SAMPLES)

    model_path = Path(
        f"experiments/out/models/classifier/{data_name}/{tf.__version__}/model_no_padding"
    )
    out_path = Path(f"experiments/out/cf/borf/{data_name}")
    out_path.mkdir(exist_ok=True, parents=True)

    model = GradientInceptionTimeClassifier.load(str(model_path))
    y_cf_pred = model.call(X_cf)
    y_cf_target = np.argsort(-y_cf_pred, axis=1)[:, 1]

    logger.info(f"Running experiment: {exp_config_name}")
    create_evaluation(
        exp_config_idx,
        model=model,
        X_train=X_train,
        y_train=y_train,
        X_cnt=X_cf,
        y_cnt=model.predict_cls(X_cf),
        y_target_cnt=y_cf_target,
        n_counterfactuals=N_COUNTERFACTUALS,
    )

    # shutil.move("temp_res", f"results/{exp_config_name}")


if __name__ == "__main__":

    logger.info("Generate configs")
    configs, names = generate_configs()

    main(0, names[0])
