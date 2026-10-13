"""TabPFN v2+ (``hollmann2025tabpfnv2``). Optional: needs the ``tabpfn`` extra and a token.

TabPFN's weights require a one-time license acceptance (env ``TABPFN_TOKEN``), so
the model is skipped, and the skip is recorded, when they're unavailable
(DECISIONS D019). It runs CPU-only on a ≤10k-row stratified temporal subsample
(DECISIONS D008).
"""

from __future__ import annotations

import importlib.util
import os

import numpy as np

from fraudlab.models.base import FraudModel, ModelData

MAX_ROWS = 10_000


def tabpfn_available() -> tuple[bool, str]:
    if importlib.util.find_spec("tabpfn") is None:
        return False, "package not installed (uv sync --extra tabpfn)"
    if not os.environ.get("TABPFN_TOKEN"):
        return False, "TABPFN_TOKEN not set (weights need license acceptance)"
    return True, ""


def stratified_temporal_subsample(
    y: np.ndarray, idx: np.ndarray, max_rows: int, rng: np.random.Generator
) -> np.ndarray:
    """All positives (≤ half the cap) plus negatives evenly spread over time."""
    pos = idx[y[idx] == 1]
    if len(pos) > max_rows // 2:
        pos = np.sort(rng.choice(pos, max_rows // 2, replace=False))
    neg = idx[y[idx] == 0]  # idx is time-ordered
    n_neg = min(len(neg), max_rows - len(pos))
    take = np.linspace(0, len(neg) - 1, n_neg).round().astype(int)
    return np.sort(np.concatenate([pos, neg[take]]))


class TabPFNModel(FraudModel):
    name, family = "tabpfn", "modern"

    def fit(self, data: ModelData, train: np.ndarray, valid: np.ndarray) -> TabPFNModel:
        from tabpfn import TabPFNClassifier

        rng = np.random.default_rng(self.seed)
        sub = stratified_temporal_subsample(data.y_train, train, MAX_ROWS, rng)
        self.model = TabPFNClassifier(
            device="cpu",
            ignore_pretraining_limits=True,
            random_state=self.seed,
            categorical_features_indices=data.categorical_idx,
        ).fit(data.X[sub], data.y_train[sub])
        self.info = {
            "train_rows": len(sub),
            "train_pos": int(data.y_train[sub].sum()),
            "train_rows_available": len(train),
        }
        return self

    def predict_proba(self, data: ModelData, idx: np.ndarray) -> np.ndarray:
        parts = [
            self.model.predict_proba(data.X[c])[:, 1]
            for c in np.array_split(idx, max(1, len(idx) // 5000))
        ]
        return np.concatenate(parts)
