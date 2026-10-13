"""Generator configuration: parameters, drift schedule, label-delay model, presets."""

from __future__ import annotations

import itertools
from dataclasses import asdict, dataclass, field, replace
from typing import Any

#: Scenarios whose fraud is defined by relational structure (shared devices,
#: mule rings, colluding merchants). Used to set the "ring share" for RQ3.
RING_SCENARIOS: tuple[str, ...] = ("mule_network", "merchant_collusion", "synthetic_identity")

#: Scenarios that hijack or abuse an individual card/account.
INDIVIDUAL_SCENARIOS: tuple[str, ...] = (
    "card_testing",
    "account_takeover",
    "velocity_abuse",
    "friendly_fraud",
    "bust_out",
)

ALL_SCENARIOS: tuple[str, ...] = INDIVIDUAL_SCENARIOS + RING_SCENARIOS


@dataclass(frozen=True)
class DriftSegment:
    """Fraud scenario mix active on ``[start_frac, end_frac)`` of the timeline.

    ``weights`` maps scenario name to relative share of the fraud budget in
    this segment. Weights need not sum to one.
    """

    start_frac: float
    end_frac: float
    weights: dict[str, float]


@dataclass(frozen=True)
class LabelDelay:
    """Delay between a fraud transaction and its label becoming available.

    Delay in days ~ Uniform(min_days, max_days), drawn once per fraud event,
    plus up to ``jitter_days`` of per-transaction jitter. Friendly fraud
    (chargeback abuse) is multiplied by ``friendly_multiplier`` because
    disputes are filed later than issuer-detected fraud.
    """

    min_days: float = 7.0
    max_days: float = 30.0
    jitter_days: float = 2.0
    friendly_multiplier: float = 1.5


@dataclass(frozen=True)
class GeneratorConfig:
    """All knobs of the synthetic generator. Fully determines the output with ``seed``."""

    seed: int = 0
    n_accounts: int = 20_000
    n_days: int = 30
    #: Mean legitimate transactions per account per day (heterogeneous across accounts).
    tx_per_account_day: float = 0.333
    #: Target share of all transactions that are fraudulent.
    fraud_rate: float = 0.005
    drift: tuple[DriftSegment, ...] = (DriftSegment(0.0, 1.0, {s: 1.0 for s in ALL_SCENARIOS}),)
    label_delay: LabelDelay = field(default_factory=LabelDelay)
    #: Probability that a fraud event is ever detected (observed_label = 1).
    detection_prob: float = 0.95
    #: 0 = legitimate behaviour is perfectly regular; 1 = many legitimate
    #: look-alikes (travel, new devices, large purchases, shared devices).
    noise: float = 0.4
    merchants_per_account: float = 0.05
    #: Share of legitimate transactions that are P2P transfers.
    p2p_share: float = 0.05
    start_date: str = "2025-01-01"
    preset: str = "custom"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    def with_ring_share(self, ring_share: float) -> GeneratorConfig:
        """Uniform scenario mix where ``ring_share`` of fraud comes from RING_SCENARIOS."""
        if not 0.0 <= ring_share <= 1.0:
            raise ValueError("ring_share must be in [0, 1]")
        w = {s: (1 - ring_share) / len(INDIVIDUAL_SCENARIOS) for s in INDIVIDUAL_SCENARIOS}
        w |= {s: ring_share / len(RING_SCENARIOS) for s in RING_SCENARIOS}
        return replace(self, drift=(DriftSegment(0.0, 1.0, w),))

    def validate(self) -> None:
        if not 0 < self.fraud_rate < 0.5:
            raise ValueError("fraud_rate must be in (0, 0.5)")
        if self.n_accounts < 10 or self.n_days < 1:
            raise ValueError("need at least 10 accounts and 1 day")
        segs = sorted(self.drift, key=lambda s: s.start_frac)
        if segs[0].start_frac != 0.0 or segs[-1].end_frac != 1.0:
            raise ValueError("drift segments must cover [0, 1]")
        for a, b in itertools.pairwise(segs):
            if abs(a.end_frac - b.start_frac) > 1e-9:
                raise ValueError("drift segments must be contiguous")
        for seg in segs:
            unknown = set(seg.weights) - set(ALL_SCENARIOS)
            if unknown:
                raise ValueError(f"unknown scenarios in drift schedule: {sorted(unknown)}")
            if sum(seg.weights.values()) <= 0:
                raise ValueError("each drift segment needs a positive total weight")


#: Drift schedule for RQ2: individual-card attacks before 60% of the timeline,
#: bust-out / synthetic identity / mule rings after.
DRIFT_RQ2: tuple[DriftSegment, ...] = (
    DriftSegment(0.0, 0.6, {"card_testing": 1, "account_takeover": 1, "velocity_abuse": 1}),
    DriftSegment(0.6, 1.0, {"bust_out": 1, "synthetic_identity": 1, "mule_network": 1}),
)

PRESETS: dict[str, GeneratorConfig] = {
    # For unit tests and `make quick`.
    "tiny": GeneratorConfig(n_accounts=2_000, n_days=30, fraud_rate=0.01, preset="tiny"),
    # About 200k transactions, the main experimental preset (RQ1, RQ3, RQ5).
    "small": GeneratorConfig(n_accounts=20_000, n_days=30, preset="small"),
    # About 200k transactions over 90 days (RQ2, RQ4). DECISIONS D007.
    "small_long": GeneratorConfig(
        n_accounts=8_000, n_days=90, tx_per_account_day=0.278, preset="small_long"
    ),
    # About 2M transactions. Documented and tested, not used in the paper.
    "full": GeneratorConfig(n_accounts=200_000, n_days=30, preset="full"),
}


def preset(name: str, **overrides: Any) -> GeneratorConfig:
    """Return a preset config with optional field overrides."""
    if name not in PRESETS:
        raise KeyError(f"unknown preset {name!r}; choose from {sorted(PRESETS)}")
    return replace(PRESETS[name], **overrides)
