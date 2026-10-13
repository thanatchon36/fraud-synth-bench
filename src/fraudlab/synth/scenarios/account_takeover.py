"""Account takeover (ATO): stolen credentials used from an attacker device."""

from __future__ import annotations

import numpy as np

from fraudlab.synth import catalog as C
from fraudlab.synth.batch import TxBatch
from fraudlab.synth.scenarios.base import Scenario, Window, register
from fraudlab.synth.world import DAY, World


@register
class AccountTakeover(Scenario):
    """Account takeover followed by cash-out.

    Modelled on the PaySim fraud agent (Lopez-Rojas et al. 2016, ``lopezrojas2016paysim``),
    which takes control of a customer account and empties it, and on the
    "compromised customer" scenario of the ULB handbook (``leborgne2022handbook``),
    where a compromised card's amounts are inflated. The attacker logs in from a
    device in a shared pool (so devices recur across victims), often from abroad
    and at night, and makes 2–6 high-value purchases or transfers within two days.

    Signature: device never seen on the account, amount ≫ account median,
    night hours, foreign country.
    """

    name = "account_takeover"

    def event(self, world: World, rng: np.random.Generator, window: Window) -> TxBatch:
        victim = int(self.victims(world, rng, 1, before=window.start)[0])
        start = window.sample(rng, margin=2 * DAY)
        n = int(rng.integers(2, 7))
        t = self.times(rng, start, n, spread=2 * DAY)
        night = rng.random(n) < 0.6
        hour = np.where(night, rng.uniform(0, 5, n), rng.uniform(0, 24, n))
        t = (t // DAY) * DAY + (hour * 3600).astype(np.int64)
        mcc = rng.choice((*C.HIGH_RISK_MCC, C.MCC_INDEX["travel"]), n).astype(np.int8)
        merchant = self.merchants(world, rng, tuple(np.unique(mcc)), n, start)
        mcc = world.mer_mcc[merchant]
        amount = np.exp(world.acc_mu[victim] + rng.uniform(1.2, 2.3, n))
        abroad = rng.random() < 0.7
        country = int(rng.integers(1, C.N_COUNTRIES)) if abroad else int(world.acc_country[victim])
        return TxBatch.build(
            t=np.sort(t),
            account=victim,
            merchant=merchant,
            device=self.attacker_devices(world, rng, pool=60),
            amount=amount,
            mcc=mcc,
            channel=C.CH_ONLINE,
            country=country,
        )
