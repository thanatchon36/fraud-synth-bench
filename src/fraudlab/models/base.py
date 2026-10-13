"""Common model interface and the shared data container."""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, replace
from typing import Any

import numpy as np
import polars as pl
from sklearn.metrics import average_precision_score
from sklearn.preprocessing import OneHotEncoder, QuantileTransformer

from fraudlab.features import CATEGORICAL, GRAPH, TABULAR, LabelView, build_features
from fraudlab.features import training_labels as _training_labels
from fraudlab.synth import SyntheticDataset

FEATURE_SETS: dict[str, tuple[str, ...]] = {
    "tabular": TABULAR,
    "tabular+graph": TABULAR + GRAPH,
}


@dataclass
class ModelData:
    """All rows of one dataset, aligned by row index (sorted by time).

    Models receive index arrays (``train``, ``valid``, ``test`` rows) and read
    what they need: the feature matrix ``X``, training labels ``y_train`` (only
    labels known before the cutoff), or past rows for sequences and graphs.
    ``y_true`` is the evaluation target and must never be used for fitting.
    """

    ds: SyntheticDataset
    frame: pl.DataFrame
    feature_names: tuple[str, ...]
    X: np.ndarray  # float32 [n, d]; NaN allowed
    y_train: np.ndarray  # int8, labels usable for fitting
    y_true: np.ndarray  # int8, ground truth for evaluation only
    t: np.ndarray  # int64 seconds since dataset start
    day: np.ndarray
    amount: np.ndarray
    account: np.ndarray
    cutoff: int  # seconds; labels known before this
    #: Every feature column (tabular + graph), so ablations need no rebuild.
    all_cols: tuple[str, ...] = ()
    X_all: np.ndarray | None = None
    cache: dict[str, Any] = field(default_factory=dict)

    @property
    def categorical_idx(self) -> list[int]:
        return [i for i, c in enumerate(self.feature_names) if c in CATEGORICAL]

    def with_features(self, names: tuple[str, ...]) -> ModelData:
        """Same rows, different feature set (shares sequence/graph caches)."""
        assert self.X_all is not None
        cols = [self.all_cols.index(c) for c in names]
        return replace(self, feature_names=names, X=self.X_all[:, cols])


def build_model_data(ds: SyntheticDataset, view: LabelView, features: str = "tabular") -> ModelData:
    frame = build_features(ds, view)
    labels = ds.labels.join(_training_labels(ds, view), on="tx_id")
    frame = frame.join(labels.select("tx_id", "is_fraud", "y_train", "scenario"), on="tx_id").sort(
        "timestamp", "tx_id"
    )
    all_cols = list(TABULAR) + list(GRAPH)
    x_all = frame.select(all_cols).to_numpy().astype(np.float32)
    start = frame["timestamp"].min().replace(hour=0, minute=0, second=0, microsecond=0)
    t = ((frame["timestamp"] - start).dt.total_milliseconds() // 1000).to_numpy()
    names = FEATURE_SETS[features]
    md = ModelData(
        ds=ds,
        frame=frame,
        feature_names=names,
        X=x_all[:, [all_cols.index(c) for c in names]],
        y_train=frame["y_train"].to_numpy().astype(np.int8),
        y_true=frame["is_fraud"].to_numpy().astype(np.int8),
        t=t.astype(np.int64),
        day=(t // 86_400).astype(np.int64),
        amount=frame["amount"].to_numpy(),
        account=frame["account_id"].to_numpy(),
        cutoff=int((view.cutoff - start).total_seconds()),
        all_cols=tuple(all_cols),
        X_all=x_all,
    )
    return md


class Preprocessor:
    """Quantile-normal numeric features + one-hot categoricals (for LR, IF, neural nets)."""

    def __init__(self, cat_idx: list[int], seed: int = 0) -> None:
        self.cat_idx = cat_idx
        self.seed = seed

    def fit(self, X: np.ndarray) -> Preprocessor:
        n = X.shape[1]
        self.num_idx = [i for i in range(n) if i not in self.cat_idx]
        self.qt = QuantileTransformer(
            n_quantiles=200,
            output_distribution="normal",
            subsample=100_000,
            random_state=self.seed,
        ).fit(np.nan_to_num(X[:, self.num_idx], nan=-1.0))
        self.ohe = (
            OneHotEncoder(handle_unknown="ignore", sparse_output=False).fit(X[:, self.cat_idx])
            if self.cat_idx
            else None
        )
        return self

    def numeric(self, X: np.ndarray) -> np.ndarray:
        return self.qt.transform(np.nan_to_num(X[:, self.num_idx], nan=-1.0)).astype(np.float32)

    def transform(self, X: np.ndarray) -> np.ndarray:
        if self.ohe is None:
            return self.numeric(X)
        return np.hstack([self.numeric(X), self.ohe.transform(X[:, self.cat_idx])]).astype(
            np.float32
        )


class FraudModel(ABC):
    """``fit / predict_proba / explain`` over row indices of a :class:`ModelData`."""

    name: str = ""
    family: str = ""
    #: False for anomaly scores, which are rankings rather than probabilities.
    probabilistic: bool = True
    #: Whether the model can use the graph feature group (else uses tabular only).
    uses_feature_matrix: bool = True

    def __init__(self, seed: int = 0, **params: Any) -> None:
        self.seed = seed
        self.params = params
        self.info: dict[str, Any] = {}

    @abstractmethod
    def fit(self, data: ModelData, train: np.ndarray, valid: np.ndarray) -> FraudModel: ...

    @abstractmethod
    def predict_proba(self, data: ModelData, idx: np.ndarray) -> np.ndarray:
        """Fraud score for rows ``idx`` (higher = more suspicious)."""

    def explain(
        self, data: ModelData, idx: np.ndarray, top: int = 10, max_rows: int = 5_000
    ) -> dict[str, float]:
        """Global importance: drop in AUPRC when a feature column is permuted."""
        rng = np.random.default_rng(self.seed)
        idx = idx if len(idx) <= max_rows else np.sort(rng.choice(idx, max_rows, replace=False))
        y = data.y_true[idx]
        if y.sum() == 0:
            return {}
        base = average_precision_score(y, self.predict_proba(data, idx))
        out = {}
        for j, name in enumerate(data.feature_names):
            Xp = data.X.copy()
            Xp[idx, j] = Xp[rng.permutation(idx), j]
            perm = replace(data, X=Xp, cache={})
            out[name] = base - average_precision_score(y, self.predict_proba(perm, idx))
        return dict(sorted(out.items(), key=lambda kv: -kv[1])[:top])

    @staticmethod
    def valid_auprc(y: np.ndarray, p: np.ndarray) -> float:
        return float(average_precision_score(y, p)) if y.sum() else 0.0


def timed(fn, *args, **kwargs) -> tuple[Any, float]:
    t = time.perf_counter()
    out = fn(*args, **kwargs)
    return out, time.perf_counter() - t
