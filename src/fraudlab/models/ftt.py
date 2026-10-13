"""FT-Transformer (``gorishniy2021ftt``), implemented compactly in PyTorch.

Feature tokenizer (one learned affine token per numeric feature, one embedding per
categorical value) plus a [CLS] token, three pre-norm transformer blocks, and a
head on [CLS]. Defaults follow the paper's recommended configuration, scaled down
(d=64, 3 blocks, 8 heads) to fit the compute budget.
"""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from fraudlab.models.base import FraudModel, ModelData, Preprocessor
from fraudlab.models.torch_utils import seed_everything, torch_device_for, train_early_stopped


class _FTT(nn.Module):
    def __init__(
        self, n_num: int, cat_cards: list[int], d: int = 64, blocks: int = 3, heads: int = 8
    ):
        super().__init__()
        self.w = nn.Parameter(torch.randn(n_num, d) / d**0.5)
        self.b = nn.Parameter(torch.zeros(n_num, d))
        self.cats = nn.ModuleList(nn.Embedding(c + 1, d) for c in cat_cards)
        self.cls = nn.Parameter(torch.randn(1, 1, d) / d**0.5)
        layer = nn.TransformerEncoderLayer(
            d,
            heads,
            dim_feedforward=2 * d,
            dropout=0.1,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, blocks, enable_nested_tensor=False)
        self.head = nn.Sequential(nn.LayerNorm(d), nn.ReLU(), nn.Linear(d, 1))

    def forward(self, num: torch.Tensor, cat: torch.Tensor) -> torch.Tensor:
        tok = num.unsqueeze(-1) * self.w + self.b  # [B, n_num, d]
        cat_tok = [emb(cat[:, i]) for i, emb in enumerate(self.cats)]
        x = torch.cat([self.cls.expand(len(num), -1, -1), tok, torch.stack(cat_tok, 1)], 1)
        return self.head(self.encoder(x)[:, 0]).squeeze(-1)


class FTTransformer(FraudModel):
    name, family = "ft_transformer", "modern"

    def _tensors(self, data: ModelData, idx: np.ndarray):
        num = self.prep.numeric(data.X[idx])
        cat = np.clip(data.X[idx][:, self.cat_idx].astype(np.int64), 0, self.cards)
        return torch.tensor(num), torch.tensor(cat)

    def fit(self, data: ModelData, train: np.ndarray, valid: np.ndarray) -> FTTransformer:
        seed_everything(self.seed)
        dev = self.device = torch_device_for(self.name)
        self.cat_idx = data.categorical_idx
        self.prep = Preprocessor(self.cat_idx, self.seed).fit(data.X[train])
        self.cards = data.X[train][:, self.cat_idx].max(axis=0).astype(np.int64) + 1
        self.net = _FTT(len(self.prep.num_idx), self.cards.tolist()).to(dev)
        y = torch.tensor(data.y_train.astype(np.float32))
        g = torch.Generator().manual_seed(self.seed)
        bce = nn.BCEWithLogitsLoss()

        def batches(_epoch: int):
            for b in torch.randperm(len(train), generator=g).split(1024):
                idx = train[b.numpy()]
                num, cat = self._tensors(data, idx)
                yield num.to(dev), cat.to(dev), y[idx].to(dev)

        self.info = train_early_stopped(
            self.net,
            batches,
            lambda: self.predict_proba(data, valid),
            data.y_train[valid],
            lambda num, cat, yb: bce(self.net(num, cat), yb),
            lr=3e-4,
            weight_decay=1e-5,
            max_epochs=int(self.params.get("max_epochs", 6)),
        )
        return self

    @torch.no_grad()
    def predict_proba(self, data: ModelData, idx: np.ndarray) -> np.ndarray:
        self.net.eval()
        out = []
        for chunk in np.array_split(idx, max(1, len(idx) // 4096)):
            num, cat = self._tensors(data, chunk)
            out.append(torch.sigmoid(self.net(num.to(self.device), cat.to(self.device))).cpu())
        return torch.cat(out).numpy()
