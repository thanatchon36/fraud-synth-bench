"""Friendly fraud: the genuine customer makes a purchase and later disputes it."""

from __future__ import annotations

import numpy as np

from fraudlab.synth import catalog as C
from fraudlab.synth.batch import TxBatch
from fraudlab.synth.scenarios.base import Scenario, Window, register
from fraudlab.synth.world import World


@register
class FriendlyFraud(Scenario):
    """Chargeback abuse by the legitimate cardholder.

    The customer buys online from their own device, at their usual time of day,
    at slightly elevated amounts, then files a "not received / not authorised"
    dispute. Because the behaviour is the customer's own, it is close to
    indistinguishable from legitimate traffic at transaction time. It acts as
    near-irreducible label noise, a realistic limit on achievable recall.
    Labels arrive later than for other fraud (``LabelDelay.friendly_multiplier``),
    consistent with dispute-driven verification latency (``dalpozzolo2018realistic``).

    Signature: known device, usual hours, amounts within ~2× the account's norm.
    """

    name = "friendly_fraud"

    def event(self, world: World, rng: np.random.Generator, window: Window) -> TxBatch:
        acc = int(self.victims(world, rng, 1, before=window.start)[0])
        n = int(rng.integers(1, 3))
        starts = np.array([window.sample(rng) for _ in range(n)])
        t = self.day_times(rng, world, np.full(n, acc), starts)
        cats = (
            C.MCC_INDEX["online_marketplace"],
            C.MCC_INDEX["electronics"],
            C.MCC_INDEX["digital_goods"],
        )
        merchant = self.merchants(world, rng, cats, n, int(starts.min()))
        mcc = world.mer_mcc[merchant]
        amount = np.exp(world.acc_mu[acc] + C.MCC_LOG_OFFSET[mcc] + rng.normal(0.4, 0.5, n))
        return TxBatch.build(
            t=np.sort(t),
            account=acc,
            merchant=merchant,
            device=world.acc_device[acc],
            amount=amount,
            mcc=mcc,
            channel=C.CH_ONLINE,
            country=world.mer_country[merchant],
        )
