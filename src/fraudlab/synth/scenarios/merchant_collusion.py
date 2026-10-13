"""Merchant collusion: a fraudster-controlled merchant bills stolen cards."""

from __future__ import annotations

import numpy as np

from fraudlab.synth import catalog as C
from fraudlab.synth.batch import TxBatch
from fraudlab.synth.scenarios.base import Scenario, Window, register
from fraudlab.synth.world import DAY, World


@register
class MerchantCollusion(Scenario):
    """Collusive / fake merchant.

    A merchant opened during the window charges 5–20 compromised cards 1–3 times
    each, usually with round amounts ($100, $200, $250, $500), over several days.
    It has no legitimate customers. This mirrors the "compromised terminal"
    scenario of the ULB handbook (``leborgne2022handbook``), where fraud concentrates
    on specific terminals, and the merchant–card bipartite structure used by
    heterogeneous fraud GNNs (``rao2022xfraud``).

    Signature: new merchant, every transaction on it is fraud, round amounts.
    """

    name = "merchant_collusion"

    def event(self, world: World, rng: np.random.Generator, window: Window) -> TxBatch:
        start = window.sample(rng, margin=5 * DAY)
        cat = int(
            rng.choice(
                (
                    C.MCC_INDEX["electronics"],
                    C.MCC_INDEX["online_marketplace"],
                    C.MCC_INDEX["digital_goods"],
                )
            )
        )
        merchant = world.add_merchant(cat, opened=start - int(rng.uniform(0, 7) * DAY))
        k = int(rng.integers(5, 21))
        cards = self.victims(world, rng, k, before=window.start)
        reps = rng.integers(1, 4, k)
        acc = np.repeat(cards, reps)
        n = len(acc)
        t = np.sort(start + rng.uniform(0, 5 * DAY, n).astype(np.int64))
        round_amt = rng.random(n) < 0.8
        amount = np.where(round_amt, rng.choice([100, 200, 250, 500], n), rng.uniform(50, 600, n))
        online = rng.random() < C.MCC_ONLINE_SHARE[cat]
        return TxBatch.build(
            t=t,
            account=acc,
            merchant=merchant,
            device=world.acc_device[acc] if online else -1,
            amount=amount,
            mcc=cat,
            channel=C.CH_ONLINE if online else C.CH_POS,
            country=world.acc_country[acc],
        )
