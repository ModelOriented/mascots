from typing import Callable

import numpy as np
from numpy.typing import NDArray


def validity(
    obs: NDArray[np.float64], cfs: NDArray[np.float64], predict_fn: Callable
) -> float:
    y_pred_org = predict_fn(obs)
    y_pred_cfs = predict_fn(cfs)
    return (y_pred_org != y_pred_cfs).mean()


def euclidean_distance(
    obs: NDArray[np.float64], cfs: NDArray[np.float64], normalize: bool = True
) -> float:
    if normalize:
        mean, std = obs.mean(), obs.std()
        obs = (obs - mean) / std
        cfs = (cfs - mean) / std

    diff = obs - cfs
    diff[np.isnan(diff)] = 0
    paired_distances = np.linalg.norm(diff, axis=1)
    return np.mean(paired_distances)


def compactness(obs: NDArray[np.float64], cfs: NDArray[np.float64]) -> float:
    return (np.abs(obs - cfs) < 1e-5).mean()
