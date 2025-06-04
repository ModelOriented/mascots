import os
import warnings
from pathlib import Path
from typing import Any, List, Union

import aeon
import numpy as np
import tensorflow as tf
from aeon.classification.deep_learning import InceptionTimeClassifier
from numpy.typing import NDArray
from tensorflow.keras.models import load_model


class GradientInceptionTimeClassifier(tf.keras.Model):
    def __init__(self, classifiers: List[Any]) -> None:
        super().__init__()
        self.classifiers = classifiers

    @classmethod
    def fit(
        cls,
        X: Union[tf.Tensor, NDArray[np.float64]],
        y: Union[tf.Tensor, NDArray[np.float64]],
    ) -> Any:

        internal_model = InceptionTimeClassifier(
            n_epochs=1000, batch_size=4, random_state=123
        )

        internal_model.fit(X, y)
        classifiers = []

        if int(aeon.__version__[0]) >= 1:  # typo in previous versions
            models = internal_model.classifiers_
        else:
            models = internal_model.classifers_

        for i in range(len(models)):
            if int(aeon.__version__[0]) >= 1:  # typo in previous versions
                classifiers.append(models[i].model_)
            else:
                classifiers.append(models[i].model_)

        return cls(classifiers)

    def save(self, path: str) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        for idx, model in enumerate(self.classifiers):
            model.save(path / f"model-{idx}.h5")

    @classmethod
    def load(cls, path: str) -> Any:
        path = Path(path)
        models = []
        for file in path.glob("*.h5"):
            models.append(load_model(file))

        if len(models) == 0:
            raise ValueError(f"Directory {path} does not contain models")

        return cls(models)

    def call(
        self,
        X: Union[tf.Tensor, NDArray[np.float64]],
        permute_dim: bool = True,
    ) -> Union[tf.Tensor, NDArray[np.float64]]:

        input_type = type(X)
        if isinstance(X, np.ndarray):
            X = tf.convert_to_tensor(X)
        elif not isinstance(X, (np.ndarray, tf.Tensor)):
            raise ValueError("X needs to be either tf.Tensor or np.ndarray")

        if len(X.shape) == 2:
            if self.classifiers[0].input_shape[0] != X.shape[0]:
                X = tf.reshape(X, [1, X.shape[0], X.shape[1]])
            else:
                X = tf.reshape(X, [X.shape[0], 1, X.shape[1]])

        if permute_dim:
            X = tf.reshape(X, [X.shape[0], X.shape[2], X.shape[1]])

        if X.shape[1] > X.shape[2]:
            warnings.warn(
                f"There are more features ({X.shape[1]}) than timepoints ({X.shape[2]})"
            )

        outs = list(
            map(
                lambda m: m(X)[tf.newaxis, :, :],
                self.classifiers,
            )
        )
        y_pred = tf.stack(outs)

        y_pred = tf.reduce_sum(y_pred, axis=0)
        y_pred = y_pred / len(self.classifiers)

        if input_type == np.ndarray:
            return y_pred[0, :, :].numpy()
        else:
            return y_pred[0, :, :]

    def predict(
        self,
        X: Union[tf.Tensor, NDArray[np.float64]],
        permute_dim: bool = True,
    ) -> Union[tf.Tensor, NDArray[np.float64]]:
        return self.call(X, permute_dim)

    def predict_cls(
        self,
        X: Union[tf.Tensor, NDArray[np.float64]],
        permute_dim: bool = True,
    ) -> Union[tf.Tensor, NDArray[np.float64]]:
        pred = self.call(X, permute_dim)
        if isinstance(pred, np.ndarray):
            return np.argmax(pred, axis=1)
        else:
            return tf.argmax(pred, axis=1)
