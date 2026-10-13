"""Unsupervised detectors: Isolation Forest and an MLP autoencoder.

Both are fit on *all* training-window rows without labels, so they don't depend
on label availability (relevant to RQ2 and RQ4). Scores are rankings, not
probabilities, so ECE is not reported for them.
"""

from __future__ import annotations

import numpy as np
import torch
from sklearn.ensemble import IsolationForest

from fraudlab.models.base import FraudModel, ModelData, Preprocessor
from fraudlab.models.torch_utils import seed_everything, torch_device_for


class IForest(FraudModel):
    """Isolation Forest (``liu2008iforest``), 200 trees, on preprocessed features."""

    name, family, probabilistic = "iforest", "anomaly", False

    def fit(self, data: ModelData, train: np.ndarray, valid: np.ndarray) -> IForest:
        self.prep = Preprocessor(data.categorical_idx, self.seed).fit(data.X[train])
        self.model = IsolationForest(
            n_estimators=200, max_samples=4096, random_state=self.seed, n_jobs=5
        ).fit(self.prep.transform(data.X[train]))
        return self

    def predict_proba(self, data: ModelData, idx: np.ndarray) -> np.ndarray:
        return -self.model.score_samples(self.prep.transform(data.X[idx]))


class AutoEncoder(FraudModel):
    """MLP autoencoder d→64→16→64→d; score = reconstruction MSE."""

    name, family, probabilistic = "autoencoder", "anomaly", False

    def fit(self, data: ModelData, train: np.ndarray, valid: np.ndarray) -> AutoEncoder:
        seed_everything(self.seed)
        self.device = torch_device_for(self.name)
        self.prep = Preprocessor(data.categorical_idx, self.seed).fit(data.X[train])
        x = torch.tensor(self.prep.transform(data.X[train]))
        d = x.shape[1]
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, 16),
            torch.nn.ReLU(),
            torch.nn.Linear(16, 64),
            torch.nn.ReLU(),
            torch.nn.Linear(64, d),
        ).to(self.device)
        opt = torch.optim.Adam(self.net.parameters(), lr=1e-3)
        g = torch.Generator().manual_seed(self.seed)
        for _ in range(int(self.params.get("epochs", 8))):
            for b in torch.randperm(len(x), generator=g).split(1024):
                xb = x[b].to(self.device)
                loss = ((self.net(xb) - xb) ** 2).mean()
                opt.zero_grad()
                loss.backward()
                opt.step()
        return self

    @torch.no_grad()
    def predict_proba(self, data: ModelData, idx: np.ndarray) -> np.ndarray:
        self.net.eval()
        x = torch.tensor(self.prep.transform(data.X[idx]))
        out = []
        for xb in x.split(8192):
            xb = xb.to(self.device)
            out.append(((self.net(xb) - xb) ** 2).mean(dim=1).cpu())
        return torch.cat(out).numpy()
