from abc import ABC, abstractmethod
from typing import Literal

import numpy as np
import torch
from numpy.typing import NDArray
from sklearn.pipeline import Pipeline

from mascots.attributions.utils import shap_mapping


class AttributionMixIn(ABC):

    def __init__(
        self,
        borf_pipeline: Pipeline,
        X: NDArray[np.float64],
    ) -> None:
        super().__init__()
        self.borf_pipeline = borf_pipeline
        self.X = X

        self.is_built = False

    def build(self) -> None:
        self._build()
        self.is_built = True

    @abstractmethod
    def _build(self) -> None:
        pass

    def explain(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if not self.is_built:
            ValueError("Explainer is not built.")

        exp = self._explain(X)
        return exp

    @abstractmethod
    def _explain(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        pass


class ShapAttribution(AttributionMixIn):

    def __init__(
        self,
        borf_pipeline: Pipeline,
        X: NDArray[np.float64],
        scope: Literal["local", "global"] = "local",
        mode: Literal[
            "normal", "linear", "tree", "kernel", "gradient", "deep"
        ] = "normal",
    ):
        super().__init__(borf_pipeline, X)
        self.scope = scope
        self.mode = mode

    def _build(self) -> None:
        if self.mode == "deep":
            self.X = torch.Tensor(self.X)
        exp = shap_mapping[self.mode](self.borf_pipeline, self.X)
        self.attr_method = exp

        if self.scope == "global":
            importance = (
                exp.shap_values(self.X[0]).swapaxes(1, 2).swapaxes(0, 1)
            )
            self._F = np.abs(importance).mean(axis=1)

        # del self.X

    def _explain(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
        if self.scope == "global":
            return np.tile(self._F[:, np.newaxis, :], (1, X.shape[0], 1))
        else:
            if self.mode == "deep":
                X = torch.Tensor(X.toarray())
                X = X.to(self.X.device)
            else:
                X = X.toarray()

            F: NDArray[np.float64] = (
                self.attr_method.shap_values(X).swapaxes(1, 2).swapaxes(0, 1)
            )
            return F


# class SklearnFeatureImportance(AttributionMixIn):

#     def _build(self) -> None:
#         pass

#     def _explain(self, X: NDArray[np.float64]) -> NDArray[np.float64]:
#         return self.borf_pipeline[-1].feature_importances_


attribution_mapping: dict[str, type] = {
    "shap": ShapAttribution,
    # "feature_importance": SklearnFeatureImportance,
}
