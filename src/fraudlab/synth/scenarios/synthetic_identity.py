"""Synthetic identity: clusters of fabricated accounts sharing devices."""

from __future__ import annotations

import numpy as np

from fraudlab.synth import catalog as C
from fraudlab.synth.batch import TxBatch
from fraudlab.synth.scenarios.base import Scenario, Window, register
from fraudlab.synth.world import DAY, World


@register
class SyntheticIdentity(Scenario):
    """Synthetic-identity ring.

    A fraudster opens 3–8 accounts with fabricated identities during the window
    and operates them from 1–2 shared devices. After a short dormancy, each
    account makes 3–10 purchases of resellable goods at above-average amounts
    within about a week. Account-opening fraud of this kind is the subject of the
    BAF datasets (Jesus et al. 2022, ``jesus2022baf``). The device-sharing
    structure is the relational signal that GNN-based detectors exploit
    (``wang2019semignn``, ``liu2021pcgnn``).

    Signature: accounts opened inside the window, ≥3 accounts per device,
    all activity fraudulent.
    """

    name = "synthetic_identity"

    def event(self, world: World, rng: np.random.Generator, window: Window) -> TxBatch:
        start = window.sample(rng, margin=7 * DAY)
        k = int(rng.integers(3, 9))
        opened = start - rng.uniform(0, 5, k) * DAY
        accs = world.add_accounts(k, "synthetic", opened.astype(np.int64), rng)
        world.acc_device[accs] = rng.choice(world.new_device(int(rng.integers(1, 3))), k)
        n_each = rng.integers(3, 11, k)
        acc = np.repeat(accs, n_each)
        n = len(acc)
        days = start + rng.uniform(1, 8, n) * DAY
        t = self.day_times(rng, world, acc, days.astype(np.int64))
        merchant = self.merchants(world, rng, C.HIGH_RISK_MCC, n, start)
        mcc = world.mer_mcc[merchant]
        amount = np.exp(world.acc_mu[acc] + C.MCC_LOG_OFFSET[mcc] + rng.normal(0.8, 0.5, n))
        order = np.argsort(t, kind="stable")
        return TxBatch.build(
            t=t[order],
            account=acc[order],
            merchant=merchant[order],
            device=world.acc_device[acc[order]],
            amount=amount[order],
            mcc=mcc[order],
            channel=C.CH_ONLINE,
            country=0,
        )
