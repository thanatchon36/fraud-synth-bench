"""Shared PyTorch helpers: seeding, device choice, early-stopped training loop."""

from __future__ import annotations

import copy
import os
import random
from collections.abc import Callable, Iterable

import numpy as np
import torch
from sklearn.metrics import average_precision_score

from fraudlab.device import torch_device

#: Per-model device choice, set from the benchmark in DECISIONS D022. Models not
#: listed use ``torch_device()`` (MPS if available, else CPU). The environment
#: variable FRAUDLAB_DEVICE overrides everything.
DEVICE_OVERRIDES: dict[str, str] = {"autoencoder": "cpu"}


def torch_device_for(model_name: str) -> torch.device:
    env = os.environ.get("FRAUDLAB_DEVICE")
    if env:
        return torch.device(env)
    if model_name in DEVICE_OVERRIDES:
        return torch.device(DEVICE_OVERRIDES[model_name])
    return torch_device()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def train_early_stopped(
    net: torch.nn.Module,
    batches: Callable[[int], Iterable[tuple[torch.Tensor, ...]]],
    score_valid: Callable[[], np.ndarray],
    y_valid: np.ndarray,
    loss_fn: Callable[..., torch.Tensor],
    lr: float = 1e-3,
    weight_decay: float = 1e-5,
    max_epochs: int = 10,
    patience: int = 2,
) -> dict:
    """Train with AdamW, keep the epoch with the best validation AUPRC."""
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    best, best_state, history, bad = -1.0, None, [], 0
    for epoch in range(max_epochs):
        net.train()
        for batch in batches(epoch):
            loss = loss_fn(*batch)
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            auprc = float(average_precision_score(y_valid, score_valid())) if y_valid.sum() else 0
        history.append(auprc)
        if auprc > best:
            best, best_state, bad = auprc, copy.deepcopy(net.state_dict()), 0
        else:
            bad += 1
            if bad > patience:
                break
    if best_state is not None:
        net.load_state_dict(best_state)
    return {"valid_auprc": best, "epochs": len(history), "valid_history": history}
