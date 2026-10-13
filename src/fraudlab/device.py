"""Compute-device selection (MPS → CPU) and the hardware probe."""

from __future__ import annotations

import json
import os
import platform
import time
from pathlib import Path
from typing import Any

import psutil
import torch

# Unsupported MPS ops fall back to CPU instead of raising.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

# Three OpenMP runtimes end up in one process (LightGBM's libomp, scikit-learn's and
# PyTorch's). Multi-threaded torch CPU ops then segfault or deadlock alongside them
# on macOS, so torch runs its CPU kernels single-threaded. LightGBM and scikit-learn
# keep their threads. See DECISIONS D023.
torch.set_num_threads(1)


def torch_device(prefer_mps: bool = True) -> torch.device:
    """``mps`` when available and preferred, else ``cpu``. Never CUDA-specific."""
    if prefer_mps and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _mps_op_support() -> dict[str, bool]:
    """Try the ops our models use on MPS; False means they would fall back to CPU."""
    if not torch.backends.mps.is_available():
        return {}
    dev = torch.device("mps")
    checks = {
        "matmul": lambda: torch.randn(64, 64, device=dev) @ torch.randn(64, 64, device=dev),
        "gru": lambda: torch.nn.GRU(8, 16, batch_first=True).to(dev)(
            torch.randn(4, 10, 8, device=dev)
        ),
        "multihead_attention": lambda: torch.nn.MultiheadAttention(16, 4, batch_first=True).to(dev)(
            *(torch.randn(2, 5, 16, device=dev),) * 3
        ),
        "scatter_reduce": lambda: torch.zeros(5, device=dev).scatter_reduce(
            0, torch.tensor([0, 1, 1], device=dev), torch.ones(3, device=dev), "mean"
        ),
        "embedding_bag": lambda: torch.nn.EmbeddingBag(10, 4, mode="mean").to(dev)(
            torch.tensor([[1, 2], [3, 4]], device=dev)
        ),
    }
    out = {}
    for name, fn in checks.items():
        try:
            fn()
            torch.mps.synchronize()
            out[name] = True
        except Exception:
            out[name] = False
    return out


def _matmul_gflops(dev: torch.device, n: int = 1024, reps: int = 10) -> float:
    a, b = torch.randn(n, n, device=dev), torch.randn(n, n, device=dev)
    (a @ b).sum().item()
    t = time.perf_counter()
    for _ in range(reps):
        c = a @ b
    c.sum().item()
    return round(2 * n**3 * reps / (time.perf_counter() - t) / 1e9, 1)


def probe(out: str | Path = "results/hardware.json") -> dict[str, Any]:
    vm = psutil.virtual_memory()
    info: dict[str, Any] = {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor": platform.processor(),
        "cpu_count_logical": psutil.cpu_count(),
        "cpu_count_physical": psutil.cpu_count(logical=False),
        "memory_total_gb": round(vm.total / 2**30, 1),
        "memory_available_gb": round(vm.available / 2**30, 1),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "mps_available": torch.backends.mps.is_available(),
        "cuda_available": torch.cuda.is_available(),
        "selected_device": str(torch_device()),
        "mps_op_support": _mps_op_support(),
        "cpu_matmul_gflops": _matmul_gflops(torch.device("cpu")),
    }
    if info["mps_available"]:
        info["mps_matmul_gflops"] = _matmul_gflops(torch.device("mps"))
    path = Path(out)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(info, indent=2))
    return info
