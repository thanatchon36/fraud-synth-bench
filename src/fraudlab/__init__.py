"""fraudlab: reproducible fraud-analytics research on synthetic data."""

# LightGBM links an external libomp, and PyTorch ships its own. On macOS, loading
# torch first makes LightGBM segfault during training, so LightGBM's OpenMP
# runtime must be loaded before anything imports torch (DECISIONS D023).
import lightgbm as _lightgbm  # noqa: F401  (import order matters)

__version__ = "0.1.0"
