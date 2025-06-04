import json
import shutil
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Callable

import numpy as np
from loguru import logger
from numpy.typing import NDArray
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import FunctionTransformer

from bin.utils.configs import (
    alternative_borf_explainer_build_args,
    alternative_borf_explainer_counterfactual_args,
    alternative_borf_explainer_init_args,
    default_borf_explainer_build_args,
    default_borf_explainer_counterfactual_args,
    default_borf_explainer_init_args,
)
from bin.utils.data import get_data
from mascots.metrics.evaluation import Evaluator, GroupEvaluator
from mascots.metrics.utils import get_current_timestamp

OUT_DIR = "experiments/out/cf/borf"
N_SAMPLES = 50
N_COUNTERFACTUALS = 5


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
    prediction_fn: Callable[[NDArray[np.float64]], np.int64],
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_samples: int,
    n_counterfactuals: int,
) -> None:

    exp_config = configs[exp_config_idx]

    timestamp = get_current_timestamp()
    res_dir = Path(OUT_DIR) / timestamp

    evaluator = Evaluator(
        exp_config["borf_explainer_init_args"]
        | {"prediction_fn": prediction_fn},
        X,
        y,
    )

    logger.info("Build evaluator")

    evaluator.build(exp_config["borf_explainer_build_args"])

    logger.info("Evaluate")

    evaluator.evaluate(
        n_samples,
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

    logger.info(f"Running experiment: {exp_config_name}")
    create_evaluation(
        exp_config_idx,
        prediction_fn=deepcopy(model).predict,
        X=X_test,
        y=y_test,
        n_samples=N_SAMPLES,
        n_counterfactuals=N_COUNTERFACTUALS,
    )

    shutil.move("temp_res", f"results/{exp_config_name}")


logger.info("Get data")
X_train, y_train, X_test, y_test = get_data("CBF")
logger.info("Train model")
model = make_pipeline(
    FunctionTransformer(lambda x: x[:, 0, :]), RandomForestClassifier()
)
model.fit(X_train, y_train)
logger.info("Generate configs")
configs, names = generate_configs()

params: list[tuple[int, str]] = []
for idx, name in enumerate(names):
    main(idx, name)

group_evaluator = GroupEvaluator(root_path=OUT_DIR)
group_evaluator.compare("results_group/hyperparameters")
