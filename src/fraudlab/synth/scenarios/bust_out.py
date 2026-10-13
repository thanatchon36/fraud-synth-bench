"""Bust-out: an established account suddenly maxes out spending, then goes dark."""

from __future__ import annotations

import numpy as np

from fraudlab.synth import catalog as C
from fraudlab.synth.batch import NO_ID, TxBatch
from fraudlab.synth.scenarios.base import Scenario, Window, register
from fraudlab.synth.world import DAY, World


@register
class BustOut(Scenario):
    """First-party bust-out.

    An account with normal history (its legitimate transactions before the bust
    are kept and stay labelled legitimate) spends 8–25 times over 1–3 days at
    3–10× its usual amounts, mostly on resellable goods, and is never used again
    (all of its later legitimate activity is removed). This is a standard
    credit-risk/fraud hybrid pattern. It is *sequence* fraud: each transaction
    is plausible alone, and only the change relative to the account's own
    history is anomalous, which is the motivation for per-account sequence
    models (``jurgovsky2018sequence``, ``branco2020interleaved``).

    Signature: spend in the bust window ≫ the account's prior daily average;
    no activity afterwards.
    """

    name = "bust_out"

    def event(self, world: World, rng: np.random.Generator, window: Window) -> TxBatch:
        acc = int(self.victims(world, rng, 1, before=window.start - 7 * DAY)[0])
        if acc in world.terminated:
            return TxBatch.empty()
        start = window.sample(rng, margin=3 * DAY)
        n = int(rng.integers(8, 26))
        days = start + rng.uniform(0, rng.uniform(1, 3) * DAY, n).astype(np.int64)
        t = np.sort(self.day_times(rng, world, np.full(n, acc), days))
        merchant = self.merchants(world, rng, (*C.HIGH_RISK_MCC, C.MCC_INDEX["retail"]), n, start)
        mcc = world.mer_mcc[merchant]
        amount = np.exp(world.acc_mu[acc] + C.MCC_LOG_OFFSET[mcc] + np.log(rng.uniform(3, 10, n)))
        channel = self.channel_for(mcc, rng)
        device = np.where(channel == C.CH_ONLINE, world.acc_device[acc], NO_ID)
        world.terminated[acc] = int(t[-1])
        return TxBatch.build(
            t=t,
            account=acc,
            merchant=merchant,
            device=device,
            amount=amount,
            mcc=mcc,
            channel=channel,
            country=world.acc_country[acc],
        )
