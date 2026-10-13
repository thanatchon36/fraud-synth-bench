"""GRU over each account's transaction history.

Follows ``jurgovsky2018sequence`` and ``branco2020interleaved``.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fraudlab.features import SEQUENCE_STEP
from fraudlab.models.base import FraudModel, ModelData, Preprocessor
from fraudlab.models.torch_utils import seed_everything, torch_device_for, train_early_stopped

SEQ_LEN = 16


def previous_rows(data: ModelData, length: int = SEQ_LEN) -> np.ndarray:
    """``[n, length]`` row indices of each row's previous transactions on the same account.

    Column ``length - 1`` is the most recent previous transaction; ``-1`` pads.
    Rows are already in time order, so a stable sort by account keeps time order
    within each account.
    """
    key = f"prev_rows_{length}"
    if key in data.cache:
        return data.cache[key]
    order = np.argsort(data.account, kind="stable")
    acc_sorted = data.account[order]
    start = np.r_[0, np.flatnonzero(np.diff(acc_sorted)) + 1]
    pos = np.arange(len(order)) - np.repeat(start, np.diff(np.r_[start, len(order)]))
    prev = np.full((len(order), length), -1, np.int64)
    for j in range(1, length + 1):
        ok = pos >= j
        src = np.flatnonzero(ok)
        prev[order[src], length - j] = order[src - j]
    data.cache[key] = prev
    return prev


class _GRUNet(nn.Module):
    def __init__(self, n_step_num: int, n_mcc: int, n_ch: int, n_cur: int, hidden: int = 64):
        super().__init__()
        self.mcc = nn.Embedding(n_mcc + 1, 8)  # last index = padding
        self.ch = nn.Embedding(n_ch + 1, 3)
        self.step = nn.Sequential(nn.Linear(n_step_num + 8 + 3 + 1, 32), nn.ReLU())
        self.gru = nn.GRU(32, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_cur, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    def forward(self, num, mcc, ch, pad, cur):  # [B,L,k], [B,L], [B,L], [B,L], [B,c]
        z = torch.cat([num, self.mcc(mcc), self.ch(ch), pad.unsqueeze(-1)], dim=-1)
        _, h = self.gru(self.step(z))
        return self.head(torch.cat([h[-1], cur], dim=-1)).squeeze(-1)


class GRUModel(FraudModel):
    """GRU over the previous 16 transactions plus the current row's feature vector."""

    name, family = "gru", "sequence"

    def _tensors(self, data: ModelData, idx: np.ndarray):
        prev = previous_rows(data)[idx]
        pad = prev < 0
        rows = np.where(pad, 0, prev)
        step = self.step_num[rows]  # [B,L,k]
        step[pad] = 0.0
        mcc = np.where(pad, self.n_mcc, self.step_mcc[rows])
        ch = np.where(pad, self.n_ch, self.step_ch[rows])
        cur = self.prep.transform(data.X[idx])
        return (
            torch.tensor(step),
            torch.tensor(mcc),
            torch.tensor(ch),
            torch.tensor(pad.astype(np.float32)),
            torch.tensor(cur),
        )

    def fit(self, data: ModelData, train: np.ndarray, valid: np.ndarray) -> GRUModel:
        seed_everything(self.seed)
        dev = self.device = torch_device_for(self.name)
        cols = list(data.all_cols)
        num_cols = [c for c in SEQUENCE_STEP if c not in ("mcc", "channel")]
        step_raw = data.X_all[:, [cols.index(c) for c in num_cols]]
        step_prep = Preprocessor([], self.seed).fit(step_raw[train])
        self.step_num = step_prep.numeric(step_raw)
        self.step_mcc = data.X_all[:, cols.index("mcc")].astype(np.int64)
        self.step_ch = data.X_all[:, cols.index("channel")].astype(np.int64)
        self.n_mcc, self.n_ch = int(self.step_mcc.max()) + 1, int(self.step_ch.max()) + 1
        self.prep = Preprocessor(data.categorical_idx, self.seed).fit(data.X[train])
        n_cur = self.prep.transform(data.X[train[:10]]).shape[1]
        self.net = _GRUNet(len(num_cols), self.n_mcc, self.n_ch, n_cur).to(dev)
        y = torch.tensor(data.y_train.astype(np.float32))
        g = torch.Generator().manual_seed(self.seed)
        bce = nn.BCEWithLogitsLoss()

        def batches(_epoch: int):
            for b in torch.randperm(len(train), generator=g).split(1024):
                idx = train[b.numpy()]
                yield (*[t.to(dev) for t in self._tensors(data, idx)], y[idx].to(dev))

        def loss_fn(*args):
            *x, yb = args
            return bce(self.net(*x), yb)

        self.info = train_early_stopped(
            self.net,
            batches,
            lambda: self.predict_proba(data, valid),
            data.y_train[valid],
            loss_fn,
            lr=1e-3,
            max_epochs=int(self.params.get("max_epochs", 8)),
        )
        return self

    @torch.no_grad()
    def predict_proba(self, data: ModelData, idx: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = []
        for chunk in np.array_split(idx, max(1, len(idx) // 8192)):
            x = [t.to(self.device) for t in self._tensors(data, chunk)]
            out.append(torch.sigmoid(self.net(*x)).cpu())
        return torch.cat(out).numpy()
