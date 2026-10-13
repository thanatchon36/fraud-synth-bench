"""LightGBM with unsupervised anomaly scores as extra features (RQ4 mitigation).

Isolation Forest and the autoencoder are fit on **all** rows before the training
cutoff, labelled or not, so they also learn from recent transactions whose labels
haven't arrived yet. Their scores are appended to the feature matrix of a tuned
LightGBM. This is the "unsupervised + supervised" combination of
``carcillo2021combining``.
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np

from fraudlab.models.anomaly import AutoEncoder, IForest
from fraudlab.models.base import ModelData
from fraudlab.models.classical import LightGBMTuned


class LightGBMSemi(LightGBMTuned):
    name, family = "lgbm_semi", "classical"

    def _augment(self, data: ModelData) -> ModelData:
        all_rows = np.arange(len(data.t))
        extra = np.stack([d.predict_proba(data, all_rows) for d in self.detectors], axis=1)
        return replace(
            data,
            X=np.hstack([data.X, extra.astype(np.float32)]),
            feature_names=(*data.feature_names, "iforest_score", "autoencoder_score"),
        )

    def fit(self, data: ModelData, train: np.ndarray, valid: np.ndarray) -> LightGBMSemi:
        unlabelled = np.flatnonzero(data.t < data.cutoff)
        self.detectors = [
            IForest(self.seed).fit(data, unlabelled, valid),
            AutoEncoder(self.seed).fit(data, unlabelled, valid),
        ]
        self._aug = self._augment(data)
        return super().fit(self._aug, train, valid)

    def predict_proba(self, data: ModelData, idx: np.ndarray) -> np.ndarray:
        aug = (
            self._aug
            if data.X.shape[0] == self._aug.X.shape[0]
            and np.shares_memory(data.y_true, self._aug.y_true)
            else self._augment(data)
        )
        return super().predict_proba(aug, idx)

    def explain(self, data: ModelData, idx: np.ndarray, top: int = 10, **kw) -> dict[str, float]:
        return super().explain(self._aug, idx, top)
