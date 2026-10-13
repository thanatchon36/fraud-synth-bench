"""Column-oriented transaction batches used inside the generator."""

from __future__ import annotations

from dataclasses import dataclass, fields

import numpy as np

NO_ID = -1


@dataclass
class TxBatch:
    """A set of transactions as parallel numpy arrays.

    ``t`` is seconds since the simulation start. ``merchant`` is ``NO_ID`` for
    P2P transfers. ``counterparty`` is ``NO_ID`` for card payments. ``event``
    groups the transactions of one fraud event (``NO_ID`` for legitimate ones).
    """

    t: np.ndarray
    account: np.ndarray
    merchant: np.ndarray
    counterparty: np.ndarray
    device: np.ndarray
    amount: np.ndarray
    mcc: np.ndarray
    channel: np.ndarray
    country: np.ndarray
    scenario: np.ndarray  # int code, 0 = legit, i+1 = ALL_SCENARIOS[i]
    event: np.ndarray

    def __len__(self) -> int:
        return len(self.t)

    @classmethod
    def build(
        cls,
        *,
        t: np.ndarray,
        account: np.ndarray | int,
        device: np.ndarray | int,
        amount: np.ndarray,
        mcc: np.ndarray | int,
        channel: np.ndarray | int,
        country: np.ndarray | int,
        merchant: np.ndarray | int = NO_ID,
        counterparty: np.ndarray | int = NO_ID,
        scenario: int = 0,
        event: int = NO_ID,
    ) -> TxBatch:
        """Build a batch, broadcasting scalar columns to the length of ``t``."""
        n = len(t)

        def col(x: np.ndarray | int, dtype: type) -> np.ndarray:
            return np.broadcast_to(np.asarray(x, dtype=dtype), (n,)).copy()

        return cls(
            t=np.asarray(t, dtype=np.int64),
            account=col(account, np.int32),
            merchant=col(merchant, np.int32),
            counterparty=col(counterparty, np.int32),
            device=col(device, np.int32),
            amount=np.round(np.asarray(amount, dtype=np.float64), 2),
            mcc=col(mcc, np.int8),
            channel=col(channel, np.int8),
            country=col(country, np.int8),
            scenario=col(scenario, np.int8),
            event=col(event, np.int32),
        )

    @classmethod
    def concat(cls, batches: list[TxBatch]) -> TxBatch:
        if not batches:
            return cls.empty()
        return cls(
            **{f.name: np.concatenate([getattr(b, f.name) for b in batches]) for f in fields(cls)}
        )

    @classmethod
    def empty(cls) -> TxBatch:
        return cls.build(
            t=np.zeros(0, np.int64),
            account=0,
            device=0,
            amount=np.zeros(0),
            mcc=0,
            channel=0,
            country=0,
        )

    def take(self, idx: np.ndarray) -> TxBatch:
        return TxBatch(**{f.name: getattr(self, f.name)[idx] for f in fields(self)})
