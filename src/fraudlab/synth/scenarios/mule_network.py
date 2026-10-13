"""Mule network: stolen funds routed through a ring of mule accounts to a collector."""

from __future__ import annotations

import numpy as np

from fraudlab.synth import catalog as C
from fraudlab.synth.batch import TxBatch
from fraudlab.synth.scenarios.base import Scenario, Window, register
from fraudlab.synth.world import DAY, World


@register
class MuleNetwork(Scenario):
    """Money-mule ring (fan-in / fan-out laundering).

    Follows the laundering typologies of AMLworld (Altman et al. 2023,
    ``altman2023amlworld``): compromised victims transfer funds to 5–12 recently
    opened mule accounts (fan-out), which forward 90–98% of the money to a single
    collector account within hours (fan-in). Mules are operated from 1–3 shared
    devices, the shared-entity structure that graph methods exploit
    (``weber2019elliptic``, ``dou2020caregnn``).

    Signature: P2P transfers, young accounts, many accounts per device,
    fan-in to a collector.
    """

    name = "mule_network"

    def event(self, world: World, rng: np.random.Generator, window: Window) -> TxBatch:
        start = window.sample(rng, margin=3 * DAY)
        k = int(rng.integers(5, 13))
        opened = start - rng.uniform(1, 45, k) * DAY
        mules = world.add_accounts(k, "mule", opened.astype(np.int64), rng)
        collector = world.add_accounts(1, "collector", np.array([start - 60 * DAY]), rng)[0]
        ring_devices = world.new_device(int(rng.integers(1, 4)))
        world.acc_device[mules] = rng.choice(ring_devices, k)
        victims = self.victims(world, rng, k, before=window.start)

        t_in = start + rng.uniform(0, 2 * DAY, k).astype(np.int64)
        amt_in = rng.uniform(300, 3000, k)
        inflow = TxBatch.build(
            t=t_in,
            account=victims,
            counterparty=mules,
            device=world.acc_device[victims],
            amount=amt_in,
            mcc=C.P2P_MCC,
            channel=C.CH_TRANSFER,
            country=world.acc_country[victims],
        )
        t_out = t_in + rng.uniform(600, 6 * 3600, k).astype(np.int64)
        outflow = TxBatch.build(
            t=t_out,
            account=mules,
            counterparty=collector,
            device=world.acc_device[mules],
            amount=amt_in * rng.uniform(0.90, 0.98, k),
            mcc=C.P2P_MCC,
            channel=C.CH_TRANSFER,
            country=0,
        )
        return TxBatch.concat([inflow, outflow])
