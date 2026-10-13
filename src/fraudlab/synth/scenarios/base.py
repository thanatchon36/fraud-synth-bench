"""Base class, shared helpers and registry for fraud scenario injectors.

To add a scenario:

1. Subclass :class:`Scenario`, set ``name``, and implement :meth:`Scenario.event`,
   which returns the transactions of **one** fraud event (one compromised card,
   one ring, ...). Cite the pattern you imitate in the docstring.
2. Decorate the class with :func:`register`.
3. Add the name to ``ALL_SCENARIOS`` in :mod:`fraudlab.synth.config` and a
   signature test in ``tests/test_synth.py``.

The generator calls :meth:`Scenario.inject` with a transaction budget and a time
window. ``inject`` repeats ``event`` until the budget is met and trims the
surplus from the *time-ordered* last event, as if it had been interrupted.
Realised fraud rates therefore match the target exactly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

import numpy as np

from fraudlab.synth import catalog as C
from fraudlab.synth.batch import TxBatch
from fraudlab.synth.config import ALL_SCENARIOS
from fraudlab.synth.world import DAY, World, sample_time_of_day


@dataclass
class Window:
    """Half-open time window ``[start, end)`` in seconds since simulation start."""

    start: int
    end: int

    def sample(self, rng: np.random.Generator, margin: int = 0) -> int:
        """A start time leaving ``margin`` seconds before the window end when possible."""
        hi = max(self.start + 1, self.end - margin)
        return int(rng.integers(self.start, hi))


class Scenario(ABC):
    """A pluggable fraud pattern."""

    name: str = ""

    def __init__(self) -> None:
        self.code = ALL_SCENARIOS.index(self.name) + 1
        self._next_event = 0
        self._device_pool: np.ndarray | None = None

    # -- to implement -------------------------------------------------------
    @abstractmethod
    def event(self, world: World, rng: np.random.Generator, window: Window) -> TxBatch:
        """Transactions of one fraud event inside ``window``."""

    # -- driver -------------------------------------------------------------
    def inject(
        self,
        world: World,
        rng: np.random.Generator,
        window: Window,
        budget: int,
        event_offset: int,
    ) -> TxBatch:
        """Generate events until ``budget`` transactions exist, then trim the surplus."""
        out: list[TxBatch] = []
        total = 0
        self._next_event = event_offset
        while total < budget:
            b = self.event(world, rng, window)
            if len(b) == 0:
                continue
            b.scenario[:] = self.code
            b.event[:] = self._next_event
            self._next_event += 1
            # Time-order the event so that trimming keeps a consistent prefix.
            keep = np.flatnonzero(b.t < world.horizon)
            b = b.take(keep[np.argsort(b.t[keep], kind="stable")])
            out.append(b)
            total += len(b)
        batch = TxBatch.concat(out)
        return batch.take(np.arange(budget)) if len(batch) > budget else batch

    @property
    def n_events(self) -> int:
        return self._next_event

    # -- shared helpers -----------------------------------------------------
    @staticmethod
    def victims(world: World, rng: np.random.Generator, k: int, before: int) -> np.ndarray:
        """``k`` distinct customer accounts opened before ``before``, weighted by activity."""
        ok = (world.acc_role == 0) & (world.acc_opened < before) & (world.acc_rate > 0)
        cand = np.flatnonzero(ok)
        w = world.acc_rate[cand] / world.acc_rate[cand].sum()
        return rng.choice(cand, size=min(k, len(cand)), replace=False, p=w).astype(np.int32)

    @staticmethod
    def merchants(
        world: World,
        rng: np.random.Generator,
        mccs: tuple[int, ...],
        k: int,
        at: int,
        weighted: bool = True,
    ) -> np.ndarray:
        """``k`` existing (legitimate) merchants of the given categories.

        Drawn by popularity, or uniformly when ``weighted`` is False (attackers
        who spread over many small merchants).
        """
        ok = np.isin(world.mer_mcc, mccs) & (world.mer_pop > 0) & (world.mer_opened <= at)
        cand = np.flatnonzero(ok)
        w = world.mer_pop[cand] / world.mer_pop[cand].sum() if weighted else None
        return rng.choice(cand, size=k, p=w).astype(np.int32)

    def attacker_devices(self, world: World, rng: np.random.Generator, pool: int) -> int:
        """A device from a scenario-wide attacker pool, so devices recur across victims."""
        if self._device_pool is None:
            self._device_pool = world.new_device(pool)
        return int(rng.choice(self._device_pool))

    @staticmethod
    def times(
        rng: np.random.Generator, start: int, n: int, spread: float, sort: bool = True
    ) -> np.ndarray:
        """``n`` timestamps after ``start`` with gaps ~ Exponential(``spread``/n) seconds."""
        gaps = rng.exponential(spread / max(n, 1), n)
        t = start + np.cumsum(gaps)
        return np.sort(t).astype(np.int64) if sort else t.astype(np.int64)

    @staticmethod
    def day_times(
        rng: np.random.Generator, world: World, acc: np.ndarray, days: np.ndarray
    ) -> np.ndarray:
        """Timestamps on given day starts at each account's usual time of day."""
        return (days // DAY) * DAY + sample_time_of_day(rng, world.acc_hour[acc])

    @staticmethod
    def channel_for(mcc: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        return np.where(rng.random(len(mcc)) < C.MCC_ONLINE_SHARE[mcc], C.CH_ONLINE, C.CH_POS)


REGISTRY: dict[str, type[Scenario]] = {}


def register(cls: type[Scenario]) -> type[Scenario]:
    if cls.name not in ALL_SCENARIOS:
        raise ValueError(f"{cls.name!r} must be listed in config.ALL_SCENARIOS")
    REGISTRY[cls.name] = cls
    return cls
