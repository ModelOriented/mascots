import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from experiments.data.data import MULTI_DATASETS, UNI_DATASETS, get_data

OUT_PATH = Path("experiments/out/results")

res_paths = {
    "experiments/out/results/borf_results.json": "borf",
    "experiments/out/results/borf-scalar-C=0.0_results.json": "borf-no-lambda",
    "experiments/out/results/glacier_results.json": "glacier",
    "experiments/out/results/mcels_results.json": "mcels",
}


def res2row(name: str, el: dict[str, Any], method_name: str) -> pd.DataFrame:
    dataname = name
    model_acc = el["model_acc"]
    validity = el[method_name]["validity"]
    proximity = el[method_name]["proximity"]
    compactness = el[method_name]["compactness"]
    outlier_factor = el[method_name]["outlier-factor"]

    if method_name == "borf":
        surrogate_acc = el["build_res"]["borf_build_out"]["accuracy"]
        surrogate_ce = el["build_res"]["borf_build_out"]["cross-entropy"]
    else:
        surrogate_acc = None
        surrogate_ce = None

    row = pd.DataFrame(
        {
            "data-name": [dataname],
            "model-accuracy": [model_acc],
            "validity": [validity],
            "proximity": [proximity],
            "compactness": [compactness],
            "surrogte-accuracy": [surrogate_acc],
            "surrogates-cross-entropy": [surrogate_ce],
            "outlier-factor": outlier_factor,
        }
    )

    return row


def get_main_summary() -> pd.DataFrame:
    dfs = []

    for path, method_name in res_paths.items():

        with open(path, "r") as f:
            obj = json.load(f)

        rows = []
        for key, item in obj.items():
            if len(list(item.keys())) == 0:
                continue
            print(key)
            row = res2row(key, item, method_name)
            rows.append(row)

        df_partial = pd.concat(rows).reset_index(drop=True)
        df_partial["method"] = method_name
        dfs.append(df_partial)

    return pd.concat(dfs)


def enrich_with_ranks(df: pd.DataFrame) -> pd.DataFrame:
    ranks = df[["validity", "proximity", "compactness", "data-name"]].copy()
    ranks["proximity"] = -ranks["proximity"]
    ranks = ranks.groupby(["data-name"]).rank(ascending=False)

    df["validity-rank"] = ranks["validity"]
    df["proximity-rank"] = ranks["proximity"]
    df["compactness-rank"] = ranks["compactness"]
    return df


def get_data_info() -> pd.DataFrame:
    res = {
        "data_name": [],
        "n_classes": [],
        "n_channels": [],
        "n_timestamps": [],
        "n_instances": [],
    }
    for data_name in UNI_DATASETS + MULTI_DATASETS:
        X_train, y_train, _, _ = get_data(data_name)
        res["data_name"].append(data_name)
        res["n_classes"].append(np.unique(y_train).shape[0])
        res["n_channels"].append(X_train.shape[1])
        res["n_timestamps"].append(X_train.shape[2])
        res["n_instances"].append(X_train.shape[0])

    return pd.DataFrame(res)


def main() -> None:
    df = get_main_summary()
    df = enrich_with_ranks(df)
    df.to_csv(OUT_PATH / "metrics.csv", index=False)

    df_data = get_data_info()
    df_data.to_csv(OUT_PATH / "data.csv", index=False)


if __name__ == "__main__":
    main()
