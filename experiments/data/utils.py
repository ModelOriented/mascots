import pickle as pkl
from pathlib import Path
from typing import Tuple

import numpy as np
from loguru import logger
from numpy.typing import NDArray

from experiments.data.data import get_data


def read_mcels_res(
    path: str,
) -> NDArray[np.float64]:
    data_name = Path(path).parent.stem
    X, _, _, _ = get_data(data_name)
    cfs = np.load(path)
    return cfs.reshape((-1, X.shape[1], X.shape[2]))


def read_glacier_res(
    path: str,
) -> NDArray[np.float64]:
    with open(path, "rb") as f:
        cfs = pkl.load(f)
    return cfs


def read_borf_res(
    path: str, only_first: bool = True
) -> Tuple[NDArray[np.float64], NDArray[np.float64]]:
    with open(path, "rb") as f:
        obj = pkl.load(f)
    cfs = []
    obs = []

    for _, item in obj.items():
        obs.append(item["observation"])
        if only_first:
            cfs.append(item["counterfactuals"][0][np.newaxis, :, :])
        else:
            cfs.append(item["counterfactuals"])

    if not only_first:
        obs = np.repeat(obs, cfs[0].shape[0], axis=0)
    cfs = np.concatenate(cfs)

    return obs, cfs
