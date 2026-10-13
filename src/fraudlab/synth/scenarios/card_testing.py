"""Card testing: a burst of tiny online charges to check that a stolen card works."""

from __future__ import annotations

import numpy as np

from fraudlab.synth import catalog as C
from fraudlab.synth.batch import TxBatch
from fraudlab.synth.scenarios.base import Scenario, Window, register
from fraudlab.synth.world import World


@register
class CardTesting(Scenario):
    """Enumeration-style card testing.

    A fraudster validates a compromised card with 5–15 micro-charges ($0.50–$3)
    at different online-only merchants (chosen uniformly, not by
    popularity) within ~10 minutes, from a device in a
    small attacker pool. Half the time, 1–3 large purchases follow within a day.
    This is a widely described industry pattern. It is the canonical target of
    short-window velocity features (Correa Bahnsen et al. 2016, ``bahnsen2016feature``).

    Signature: very small amounts, sub-minute gaps, many distinct merchants per burst.
    """

    name = "card_testing"

    def event(self, world: World, rng: np.random.Generator, window: Window) -> TxBatch:
        victim = int(self.victims(world, rng, 1, before=window.start)[0])
        start = window.sample(rng, margin=3600)
        n = int(rng.integers(5, 16))
        t = self.times(rng, start, n, spread=600)
        online_only = (C.MCC_INDEX["digital_goods"], C.MCC_INDEX["online_marketplace"])
        merchant = self.merchants(world, rng, online_only, n, start, weighted=False)
        mcc = world.mer_mcc[merchant]
        amount = rng.uniform(0.5, 3.0, n)
        device = self.attacker_devices(world, rng, pool=40)
        if rng.random() < 0.5:
            k = int(rng.integers(1, 4))
            t2 = self.times(rng, int(t[-1]), k, spread=86_400 / 2)
            mcc2 = rng.choice(C.HIGH_RISK_MCC, k).astype(np.int8)
            t = np.concatenate([t, t2])
            mcc = np.concatenate([mcc, mcc2])
            merchant = np.concatenate(
                [merchant, self.merchants(world, rng, C.HIGH_RISK_MCC, k, start)]
            )
            amount = np.concatenate(
                [amount, np.exp(world.acc_mu[victim] + 1.5 + rng.normal(0, 0.5, k))]
            )
        return TxBatch.build(
            t=t,
            account=victim,
            merchant=merchant,
            device=device,
            amount=amount,
            mcc=mcc,
            channel=C.CH_ONLINE,
            country=world.mer_country[merchant],
        )
