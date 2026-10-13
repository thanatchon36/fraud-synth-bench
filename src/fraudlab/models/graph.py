"""GraphSAGE with time-respecting neighbour sampling (``hamilton2017graphsage``).

Implemented in plain PyTorch because PyG's neighbour samplers need ``pyg-lib`` or
``torch-sparse``, which have no PyPI wheels for macOS arm64 (DECISIONS D019).

Graph: account, merchant and device nodes. Edges: account–merchant (payment),
account–account (transfer) and account–device (use), each stamped with the time
it was first seen. To score a transaction at time ``T``, we sample neighbours
uniformly among edges first seen *before* ``T`` (2 hops, fan-out 10 then 5), so
no future structure leaks in (``lu2022bright``). The node inputs are the node type,
log-degree at ``T``, and whether the node was involved in fraud whose label was
known before ``min(T, cutoff)``.

The transaction score combines the row's tabular features with mean-aggregated
GraphSAGE embeddings of its account, its merchant or counterparty, and its device.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import timedelta

import numpy as np
import polars as pl
import torch
from torch import nn

from fraudlab.features import LabelView, known_labels
from fraudlab.models.base import FraudModel, ModelData, Preprocessor
from fraudlab.models.torch_utils import (
    correct_undersampling,
    seed_everything,
    torch_device_for,
    train_early_stopped,
    undersampled_epoch,
)

FANOUT = (10, 5)
NODE_FEATS = 5  # type one-hot (3), known-fraud flag, log1p(degree)
_BIG = np.int64(1) << 40


@dataclass
class TemporalGraph:
    ptr: np.ndarray  # CSR row pointer [n_nodes + 1]
    nbr: np.ndarray  # neighbour ids, sorted by (node, time)
    key: np.ndarray  # node * BIG + time, for searchsorted
    node_type: np.ndarray  # 0 account, 1 merchant, 2 device
    fraud_known_at: np.ndarray  # seconds; earliest known fraud involving the node (inf if none)
    seeds: np.ndarray  # [n_rows, 3] node ids: account, merchant/counterparty, device (-1 none)

    def sample(self, nodes: np.ndarray, t: np.ndarray, k: int, rng: np.random.Generator):
        """``k`` uniform neighbours of each node among edges first seen before ``t``."""
        valid = nodes >= 0
        safe = np.where(valid, nodes, 0)
        lo = self.ptr[safe]
        cnt = np.searchsorted(self.key, safe.astype(np.int64) * _BIG + t, "left") - lo
        cnt = np.where(valid, cnt, 0)
        off = (rng.random((len(nodes), k)) * np.maximum(cnt, 1)[:, None]).astype(np.int64)
        nb = self.nbr[np.minimum(lo[:, None] + off, len(self.nbr) - 1)]
        nb = np.where(cnt[:, None] > 0, nb, -1)
        return nb, cnt

    def degree(self, nodes: np.ndarray, t: np.ndarray) -> np.ndarray:
        safe = np.where(nodes >= 0, nodes, 0)
        cnt = np.searchsorted(self.key, safe.astype(np.int64) * _BIG + t, "left") - self.ptr[safe]
        return np.where(nodes >= 0, cnt, 0)

    def features(self, nodes: np.ndarray, t: np.ndarray, q: np.ndarray, deg: np.ndarray):
        """Node input features at query time (zeros for padding ``-1``)."""
        valid = nodes >= 0
        safe = np.where(valid, nodes, 0)
        f = np.zeros((*nodes.shape, NODE_FEATS), np.float32)
        f[..., 0:3] = np.eye(3, dtype=np.float32)[self.node_type[safe]]
        f[..., 3] = self.fraud_known_at[safe] < q
        f[..., 4] = np.log1p(deg)
        f[~valid] = 0.0
        return f


def build_temporal_graph(data: ModelData) -> TemporalGraph:
    if "tgraph" in data.cache:
        return data.cache["tgraph"]
    ds = data.ds
    n_acc, n_mer = ds.accounts.height, ds.merchants.height
    tx = data.frame.select("tx_id", "account_id").join(
        ds.transactions.select("tx_id", "merchant_id", "counterparty_id", "device_id"),
        on="tx_id",
        how="left",
        maintain_order="left",
    )
    t = data.t
    acc = tx["account_id"].to_numpy().astype(np.int64)
    mer = tx["merchant_id"].to_numpy(allow_copy=True)
    cp = tx["counterparty_id"].to_numpy(allow_copy=True)
    dev = tx["device_id"].to_numpy(allow_copy=True)
    mer_node = np.where(np.isnan(mer.astype(float)), -1, n_acc + np.nan_to_num(mer.astype(float)))
    cp_node = np.where(np.isnan(cp.astype(float)), -1, np.nan_to_num(cp.astype(float)))
    dev_node = np.where(
        np.isnan(dev.astype(float)), -1, n_acc + n_mer + np.nan_to_num(dev.astype(float))
    )
    other = np.where(mer_node >= 0, mer_node, cp_node).astype(np.int64)
    dev_node = dev_node.astype(np.int64)
    n_nodes = n_acc + n_mer + int(np.nanmax(np.r_[dev.astype(float), 0])) + 1

    # Undirected edges, first-seen time per (u, v) pair.
    src = np.concatenate([acc[other >= 0], acc[dev_node >= 0]])
    dst = np.concatenate([other[other >= 0], dev_node[dev_node >= 0]])
    et = np.concatenate([t[other >= 0], t[dev_node >= 0]])
    e = pl.DataFrame({"u": np.r_[src, dst], "v": np.r_[dst, src], "t": np.r_[et, et]})
    e = e.group_by("u", "v").agg(pl.col("t").min()).sort("u", "t")
    u, v, et = e["u"].to_numpy(), e["v"].to_numpy(), e["t"].to_numpy()
    ptr = np.zeros(n_nodes + 1, np.int64)
    np.add.at(ptr, u + 1, 1)
    ptr = np.cumsum(ptr)

    node_type = np.full(n_nodes, 2, np.int8)
    node_type[:n_acc] = 0
    node_type[n_acc : n_acc + n_mer] = 1

    # Earliest *realistically known* fraud per node (same rule as the tabular features).
    start = data.frame["timestamp"].min().replace(hour=0, minute=0, second=0, microsecond=0)
    cutoff = start + timedelta(seconds=int(data.cutoff))
    known = known_labels(ds, LabelView(cutoff, "delayed")).with_columns(
        ((pl.col("known_at") - start).dt.total_seconds()).alias("ks")
    )
    row_of = {tid: i for i, tid in enumerate(data.frame["tx_id"].to_list())}
    rows = np.array([row_of[x] for x in known["tx_id"].to_list()], dtype=np.int64)
    ks = known["ks"].to_numpy().astype(np.float64)
    fk = np.full(n_nodes, np.inf)
    for nodes in (acc[rows], other[rows], dev_node[rows]):
        ok = nodes >= 0
        np.minimum.at(fk, nodes[ok], ks[ok])

    g = TemporalGraph(
        ptr=ptr,
        nbr=v.astype(np.int64),
        key=u.astype(np.int64) * _BIG + et.astype(np.int64),
        node_type=node_type,
        fraud_known_at=fk,
        seeds=np.stack([acc, other, dev_node], axis=1),
    )
    data.cache["tgraph"] = g
    return g


class _SAGE(nn.Module):
    """Two mean-aggregator GraphSAGE layers shared by all seed types + transaction head."""

    def __init__(self, n_tab: int, hidden: int = 64):
        super().__init__()
        self.self1, self.nb1 = nn.Linear(NODE_FEATS, hidden), nn.Linear(NODE_FEATS, hidden)
        self.self2, self.nb2 = nn.Linear(hidden, hidden), nn.Linear(hidden, hidden)
        self.head = nn.Sequential(
            nn.Linear(n_tab + 3 * hidden, 64), nn.ReLU(), nn.Dropout(0.1), nn.Linear(64, 1)
        )

    @staticmethod
    def _mean(x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        m = mask.unsqueeze(-1)
        return (x * m).sum(-2) / m.sum(-2).clamp(min=1)

    def embed(self, x0, x1, m1, x2, m2):
        # x0 [N,F] seeds; x1 [N,k1,F] hop-1; x2 [N,k1,k2,F] hop-2
        h1_seed = torch.relu(self.self1(x0) + self.nb1(self._mean(x1, m1)))
        h1_hop1 = torch.relu(self.self1(x1) + self.nb1(self._mean(x2, m2)))
        return self.self2(h1_seed) + self.nb2(self._mean(h1_hop1, m1))

    def forward(self, tab, x0, x1, m1, x2, m2, present):
        b = tab.shape[0]
        h = self.embed(x0, x1, m1, x2, m2) * present.unsqueeze(-1)  # [B*3, H]
        return self.head(torch.cat([tab, h.view(b, -1)], dim=-1)).squeeze(-1)


class GraphSAGEModel(FraudModel):
    name, family = "graphsage", "graph"

    def _batch(self, data: ModelData, idx: np.ndarray, rng: np.random.Generator):
        g = self.g
        seeds = g.seeds[idx].reshape(-1)  # [B*3]
        t = np.repeat(data.t[idx], 3)
        q = np.minimum(t, data.cutoff)
        k1, k2 = FANOUT
        h1, _ = g.sample(seeds, t, k1, rng)  # [N,k1]
        t1 = np.repeat(t, k1)
        h2, _ = g.sample(h1.reshape(-1), t1, k2, rng)  # [N*k1,k2]
        x0 = g.features(seeds, t, q, g.degree(seeds, t))
        x1 = g.features(h1, t[:, None], q[:, None], g.degree(h1.reshape(-1), t1).reshape(h1.shape))
        t2 = np.repeat(t1, k2)
        h2r = h2.reshape(len(seeds), k1, k2)
        x2 = g.features(
            h2r, t[:, None, None], q[:, None, None], g.degree(h2.reshape(-1), t2).reshape(h2r.shape)
        )
        tab = self.prep.transform(data.X[idx])
        return [
            torch.tensor(a)
            for a in (
                tab,
                x0,
                x1,
                (h1 >= 0).astype(np.float32),
                x2,
                (h2r >= 0).astype(np.float32),
                (seeds >= 0).astype(np.float32),
            )
        ]

    def fit(self, data: ModelData, train: np.ndarray, valid: np.ndarray) -> GraphSAGEModel:
        seed_everything(self.seed)
        dev = self.device = (
            torch.device("cpu")
            if torch_device_for(self.name).type == "mps"
            else torch_device_for(self.name)
        )  # GNNs run on CPU (MISSION §0)
        self.g = build_temporal_graph(data)
        self.prep = Preprocessor(data.categorical_idx, self.seed).fit(data.X[train])
        n_tab = self.prep.transform(data.X[train[:10]]).shape[1]
        self.net = _SAGE(n_tab).to(dev)
        y = torch.tensor(data.y_train.astype(np.float32))
        bce = nn.BCEWithLogitsLoss()

        self.neg_frac = float(self.params.get("neg_frac", 0.1))

        def batches(epoch: int):
            rng = np.random.default_rng([self.seed, epoch])
            rows = undersampled_epoch(train, data.y_train, epoch, self.seed, self.neg_frac)
            for idx in np.array_split(rows, max(1, len(rows) // 1024)):
                yield (*[a.to(dev) for a in self._batch(data, idx, rng)], y[idx].to(dev))

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
            max_epochs=int(self.params.get("max_epochs", 20)),
            patience=3,
        )
        return self

    @torch.no_grad()
    def predict_proba(self, data: ModelData, idx: np.ndarray) -> np.ndarray:
        self.net.eval()
        rng = np.random.default_rng([self.seed, 10_000])  # fixed sampling at inference
        out = []
        for chunk in np.array_split(idx, max(1, len(idx) // 4096)):
            x = [a.to(self.device) for a in self._batch(data, chunk, rng)]
            out.append(torch.sigmoid(self.net(*x)).cpu())
        return correct_undersampling(torch.cat(out).numpy(), self.neg_frac)
