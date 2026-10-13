"""Strict temporal splits. Never random."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta


@dataclass(frozen=True)
class TemporalSplit:
    """Half-open train, validation and test windows.

    ``[start, valid_start)``, ``[valid_start, test_start)``, ``[test_start, end)``.
    """

    start: datetime
    valid_start: datetime
    test_start: datetime
    end: datetime

    @classmethod
    def from_fractions(
        cls, start: datetime, n_days: int, train: float = 0.60, valid: float = 0.15
    ) -> TemporalSplit:
        """Default protocol: 60 / 15 / 25 % of the timeline, cut at day boundaries."""
        d_valid = round(n_days * train)
        d_test = round(n_days * (train + valid))
        return cls(
            start,
            start + timedelta(days=d_valid),
            start + timedelta(days=d_test),
            start + timedelta(days=n_days),
        )

    @classmethod
    def from_days(
        cls, start: datetime, valid_day: float, test_day: float, end_day: float
    ) -> TemporalSplit:
        return cls(
            start,
            start + timedelta(days=valid_day),
            start + timedelta(days=test_day),
            start + timedelta(days=end_day),
        )

    @property
    def label_cutoff(self) -> datetime:
        """Deployment time. Train and validation labels must be known before it,
        and label-derived features use only labels known before it (DECISIONS D024)."""
        return self.test_start
