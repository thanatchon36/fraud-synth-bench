"""Velocity abuse: many medium-value purchases in a short burst at one merchant type."""

from __future__ import annotations

import numpy as np

from fraudlab.synth import catalog as C
from fraudlab.synth.batch import NO_ID, TxBatch
from fraudlab.synth.scenarios.base import Scenario, Window, register
from fraudlab.synth.world import World


@register
class VelocityAbuse(Scenario):
    """Rapid repeated spending on a compromised card.

    8–25 medium-value ($20–$150) purchases within 1–2 hours at 1–3 merchants in a
    single resellable category (gift cards, digital goods), either online from an
    attacker device or card-present with a cloned card. This is the pattern that
    count- and sum-over-window aggregation features target
    (``bahnsen2016feature``, ``dalpozzolo2014lessons``).

    Signature: high transaction count per account-hour, repeated merchants.
    """

    name = "velocity_abuse"

    def event(self, world: World, rng: np.random.Generator, window: Window) -> TxBatch:
        victim = int(self.victims(world, rng, 1, before=window.start)[0])
        start = window.sample(rng, margin=3 * 3600)
        n = int(rng.integers(8, 26))
        t = self.times(rng, start, n, spread=rng.uniform(3600, 7200))
        cat = C.MCC_INDEX["gift_cards"] if rng.random() < 0.5 else C.MCC_INDEX["digital_goods"]
        shops = self.merchants(world, rng, (cat,), int(rng.integers(1, 4)), start)
        merchant = rng.choice(shops, n)
        online = rng.random() < 0.6
        device = self.attacker_devices(world, rng, pool=40) if online else NO_ID
        return TxBatch.build(
            t=t,
            account=victim,
            merchant=merchant,
            device=device,
            amount=rng.uniform(20, 150, n),
            mcc=cat,
            channel=C.CH_ONLINE if online else C.CH_POS,
            country=world.acc_country[victim],
        )
