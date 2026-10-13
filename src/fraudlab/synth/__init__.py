"""Seeded, parameterised synthetic fraud data generator.

>>> from fraudlab.synth import generate, preset
>>> ds = generate(preset("tiny", seed=0))
"""

from fraudlab.synth.config import (
    ALL_SCENARIOS,
    DRIFT_RQ2,
    PRESETS,
    RING_SCENARIOS,
    DriftSegment,
    GeneratorConfig,
    LabelDelay,
    preset,
)
from fraudlab.synth.generator import SyntheticDataset, content_hash, generate, load, write

__all__ = [
    "ALL_SCENARIOS",
    "DRIFT_RQ2",
    "PRESETS",
    "RING_SCENARIOS",
    "DriftSegment",
    "GeneratorConfig",
    "LabelDelay",
    "SyntheticDataset",
    "content_hash",
    "generate",
    "load",
    "preset",
    "write",
]
