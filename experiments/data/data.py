import warnings
from typing import Generator, List, Tuple

import numpy as np
from aeon.datasets import load_classification
from numpy.typing import NDArray
from sklearn.model_selection import train_test_split

UNI_DATASETS: List[str] = [
    "TwoLeadECG",  # 0
    "GunPoint",  # 1
    "Earthquakes",  # 2
    "Coffee",  # 3
    "Wine",  # 4
    "ItalyPowerDemand",  # 5
    "CBF",  # 6
    "Herring",  # 7
    "Fish",  # 8
]

MULTI_DATASETS: List[str] = [
    "ArticularyWordRecognition",  # 9
    "BasicMotions",  # 10
    "Cricket",  # 11
    "Epilepsy",  # 12
    "ERing",  # 13
    "NATOPS",  # 14
    "RacketSports",  # 15
    "ECG200",  # 16
    "Handwriting",  # 17
]


def get_data(
    name: str,
) -> Tuple[
    NDArray[np.float64],
    NDArray[np.int64],
    NDArray[np.float64],
    NDArray[np.int64],
]:
    X_train, y_train = load_classification(
        name, split="train", extract_path="data/multi"
    )
    X_test, y_test = load_classification(
        name, split="test", extract_path="data/multi"
    )

    y_unique = np.unique(y_train)

    y_train = np.searchsorted(y_unique, y_train)
    y_test = np.searchsorted(y_unique, y_test)

    return X_train, y_train, X_test, y_test


def get_cf_data(
    X: NDArray[np.float64],
    y: NDArray[np.int64],
    n_samples: int = 50,
    seed: int = 123,
) -> Tuple[NDArray[np.float64], NDArray[np.int64]]:
    if n_samples <= y.shape[0]:
        X_cf, _, y_cf, _ = train_test_split(
            X, y, stratify=y, random_state=seed, train_size=n_samples
        )
    else:
        X_cf, y_cf = X, y
    return X_cf, y_cf


def iter_over_datasets(only_univariate: bool) -> Generator[
    Tuple[
        str,
        Tuple[
            NDArray[np.float64],
            NDArray[np.int64],
            NDArray[np.float64],
            NDArray[np.int64],
        ],
    ],
    None,
    None,
]:
    names = UNI_DATASETS
    if not only_univariate:
        names + MULTI_DATASETS

    for name in names:
        yield name, get_data(name)
