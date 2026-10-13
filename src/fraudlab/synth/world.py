"""The simulated world: accounts, merchants, devices, and legitimate behaviour.

Legitimate behaviour follows the stylised facts used by agent-based payment
simulators (Lopez-Rojas et al. 2016; Le Borgne et al. 2022):

* heterogeneous activity: per-account rate ~ Gamma(shape=1.5);
* log-normal amounts with an account-level mean (segment) and a category offset;
* time-of-day from a von Mises distribution around each account's preferred hour,
  mixed with 10% uniform activity;
* habitual merchants: 65% of card payments go to one of five favourites;
* a small share of P2P transfers to a fixed set of contacts.

The ``noise`` knob adds legitimate look-alikes of fraud signals (travel abroad,
one-off and switched devices, unusually large purchases, devices shared within a
household), so that no single raw field separates fraud from legitimate traffic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from fraudlab.synth import catalog as C
from fraudlab.synth.batch import NO_ID, TxBatch
from fraudlab.synth.config import GeneratorConfig

DAY = 86_400
N_FAVOURITES = 5
N_CONTACTS = 3


@dataclass
class World:
    """Mutable entity tables. Fraud injectors append accounts, merchants and devices."""

    n_days: int
    # accounts
    acc_opened: np.ndarray  # seconds relative to start (negative = before start)
    acc_country: np.ndarray
    acc_segment: np.ndarray
    acc_role: np.ndarray
    acc_mu: np.ndarray  # log-amount location
    acc_hour: np.ndarray  # preferred hour of day
    acc_rate: np.ndarray  # legitimate tx per day
    acc_device: np.ndarray  # primary device
    # merchants
    mer_mcc: np.ndarray
    mer_country: np.ndarray
    mer_opened: np.ndarray
    mer_pop: np.ndarray  # popularity weight
    next_device: int
    #: account -> time (s) after which the account goes dark (bust-out).
    terminated: dict[int, int] = field(default_factory=dict)

    @property
    def n_accounts(self) -> int:
        return len(self.acc_opened)

    @property
    def horizon(self) -> int:
        return self.n_days * DAY

    def new_device(self, k: int = 1) -> np.ndarray:
        ids = np.arange(self.next_device, self.next_device + k, dtype=np.int32)
        self.next_device += k
        return ids

    def add_accounts(
        self, k: int, role: str, opened: np.ndarray, rng: np.random.Generator
    ) -> np.ndarray:
        """Append ``k`` accounts with role ``role``; returns their ids."""
        start = self.n_accounts
        seg = rng.choice(len(C.SEGMENTS), size=k, p=C.SEGMENT_P)
        self.acc_opened = np.concatenate([self.acc_opened, opened.astype(np.int64)])
        self.acc_country = np.concatenate([self.acc_country, np.zeros(k, np.int8)])
        self.acc_segment = np.concatenate([self.acc_segment, seg.astype(np.int8)])
        role_code = np.full(k, C.ACCOUNT_ROLES.index(role), np.int8)
        self.acc_role = np.concatenate([self.acc_role, role_code])
        self.acc_mu = np.concatenate([self.acc_mu, C.SEGMENT_LOG_AMOUNT[seg]])
        self.acc_hour = np.concatenate([self.acc_hour, rng.normal(14, 3.5, k)])
        self.acc_rate = np.concatenate([self.acc_rate, np.zeros(k)])
        self.acc_device = np.concatenate([self.acc_device, self.new_device(k)])
        return np.arange(start, start + k, dtype=np.int32)

    def add_merchant(self, mcc: int, opened: int) -> int:
        self.mer_mcc = np.append(self.mer_mcc, np.int8(mcc))
        self.mer_country = np.append(self.mer_country, np.int8(0))
        self.mer_opened = np.append(self.mer_opened, np.int64(opened))
        self.mer_pop = np.append(self.mer_pop, 0.0)  # never chosen by legit customers
        return len(self.mer_mcc) - 1


def build_world(cfg: GeneratorConfig, rng: np.random.Generator) -> World:
    n = cfg.n_accounts
    seg = rng.choice(len(C.SEGMENTS), size=n, p=C.SEGMENT_P)
    # Accounts opened up to 5 years before the start; 3% open during the window.
    opened = -rng.uniform(0, 5 * 365, n) * DAY
    late = rng.random(n) < 0.03
    opened[late] = rng.uniform(0, cfg.n_days * 0.8, late.sum()) * DAY
    rate = rng.gamma(1.5, cfg.tx_per_account_day / 1.5, n)

    n_mer = max(20, int(n * cfg.merchants_per_account))
    mer_mcc = rng.choice(len(C.MCC_TABLE), size=n_mer, p=C.MCC_POPULARITY)
    mer_country = np.where(rng.random(n_mer) < 0.85, 0, rng.integers(1, C.N_COUNTRIES, n_mer))
    mer_pop = rng.zipf(1.6, n_mer).astype(np.float64).clip(max=200)

    return World(
        n_days=cfg.n_days,
        acc_opened=opened.astype(np.int64),
        acc_country=rng.choice(C.N_COUNTRIES, size=n, p=C.HOME_COUNTRY_P).astype(np.int8),
        acc_segment=seg.astype(np.int8),
        acc_role=np.zeros(n, np.int8),
        acc_mu=C.SEGMENT_LOG_AMOUNT[seg] + rng.normal(0, 0.3, n),
        acc_hour=rng.normal(14, 3.5, n),
        acc_rate=rate,
        acc_device=np.arange(n, dtype=np.int32),
        mer_mcc=mer_mcc.astype(np.int8),
        mer_country=mer_country.astype(np.int8),
        mer_opened=np.full(n_mer, -10 * 365 * DAY, np.int64),
        mer_pop=mer_pop,
        next_device=n,
    )


def sample_time_of_day(
    rng: np.random.Generator, centre: np.ndarray, kappa: float = 2.5, uniform_share: float = 0.1
) -> np.ndarray:
    """Seconds into the day around ``centre`` hours (von Mises, plus uniform mixture)."""
    hour = centre + rng.vonmises(0.0, kappa, len(centre)) * 24 / (2 * np.pi)
    uni = rng.random(len(centre)) < uniform_share
    hour[uni] = rng.uniform(0, 24, uni.sum())
    return (np.mod(hour, 24) * 3600).astype(np.int64)


def generate_legit(world: World, cfg: GeneratorConfig, rng: np.random.Generator) -> TxBatch:
    """Vectorised legitimate transactions for every customer account."""
    n = world.n_accounts
    horizon = world.horizon
    noise = cfg.noise
    active_from = np.clip(world.acc_opened, 0, horizon)
    active_days = (horizon - active_from) / DAY
    counts = rng.poisson(world.acc_rate * active_days)
    acc = np.repeat(np.arange(n, dtype=np.int32), counts)
    m = len(acc)

    day_start = active_from[acc] + rng.uniform(0, 1, m) * (horizon - active_from[acc])
    day_floor = (day_start // DAY) * DAY
    t = day_floor + sample_time_of_day(rng, world.acc_hour[acc])
    t = np.clip(t, active_from[acc], horizon - 1)

    # Merchants: favourites (popularity-weighted) or a popularity-weighted draw.
    n_mer = len(world.mer_mcc)
    p_mer = world.mer_pop / world.mer_pop.sum()
    favourites = rng.choice(n_mer, size=(n, N_FAVOURITES), p=p_mer)
    use_fav = rng.random(m) < 0.65
    merchant = np.where(
        use_fav,
        favourites[acc, rng.integers(0, N_FAVOURITES, m)],
        rng.choice(n_mer, size=m, p=p_mer),
    ).astype(np.int32)
    mcc = world.mer_mcc[merchant].astype(np.int8)

    sigma = 0.8
    amount = np.exp(world.acc_mu[acc] + C.MCC_LOG_OFFSET[mcc] + sigma * rng.standard_normal(m))

    # Noise: unusually large legitimate purchases.
    big = rng.random(m) < 0.01 * noise
    amount[big] *= rng.uniform(4, 15, big.sum())

    channel = np.where(rng.random(m) < C.MCC_ONLINE_SHARE[mcc], C.CH_ONLINE, C.CH_POS)
    country = world.acc_country[acc].copy()
    online = channel == C.CH_ONLINE
    country[online] = world.mer_country[merchant[online]]

    # P2P transfers to fixed contacts.
    contacts = rng.integers(0, n, size=(n, N_CONTACTS), dtype=np.int32)
    p2p = rng.random(m) < cfg.p2p_share
    counterparty = np.full(m, NO_ID, np.int32)
    counterparty[p2p] = contacts[acc[p2p], rng.integers(0, N_CONTACTS, p2p.sum())]
    merchant[p2p] = NO_ID
    mcc[p2p] = C.P2P_MCC
    channel[p2p] = C.CH_TRANSFER
    country[p2p] = world.acc_country[acc[p2p]]
    amount[p2p] = np.exp(
        world.acc_mu[acc[p2p]]
        + C.MCC_LOG_OFFSET[C.P2P_MCC]
        + sigma * rng.standard_normal(p2p.sum())
    )
    # People send round sums: half of transfers are rounded to 10, a fifth to 50.
    r = rng.random(m)
    to10, to50 = p2p & (r < 0.5), p2p & (r >= 0.5) & (r < 0.7)
    amount[to10] = np.maximum(10, np.round(amount[to10] / 10) * 10)
    amount[to50] = np.maximum(50, np.round(amount[to50] / 50) * 50)

    # Devices: card-present payments carry no customer device (set at the end).
    # Noise: households share one device between two accounts.
    shared = rng.random(n) < 0.15 * noise
    partner = rng.integers(0, n, n)
    acc_dev = world.acc_device.copy()
    acc_dev[shared] = world.acc_device[partner[shared]]
    device = acc_dev[acc]
    # Noise: some accounts switch to a new phone mid-window.
    switch = rng.random(n) < 0.10 * noise
    switch_t = rng.uniform(0, horizon, n).astype(np.int64)
    switch_dev = np.full(n, NO_ID, np.int32)
    switch_dev[switch] = world.new_device(int(switch.sum()))
    after = switch[acc] & (t >= switch_t[acc])
    device[after] = switch_dev[acc[after]]
    # Noise: one-off logins from a fresh device.
    fresh = online & (rng.random(m) < 0.02 * noise)
    device[fresh] = world.new_device(int(fresh.sum()))
    device[channel == C.CH_POS] = NO_ID

    # Noise: trips abroad (all transactions in the trip window are foreign).
    travel = rng.random(n) < 0.10 * noise
    trip_start = rng.uniform(0, horizon, n)
    trip_len = rng.uniform(2, 7, n) * DAY
    trip_country = rng.integers(1, C.N_COUNTRIES, n).astype(np.int8)
    on_trip = travel[acc] & (t >= trip_start[acc]) & (t < trip_start[acc] + trip_len[acc])
    country[on_trip & ~p2p] = trip_country[acc[on_trip & ~p2p]]

    world.acc_device = acc_dev  # households persist for injectors' device lookups
    return TxBatch.build(
        t=t,
        account=acc,
        merchant=merchant,
        counterparty=counterparty,
        device=device,
        amount=amount,
        mcc=mcc,
        channel=channel,
        country=country,
    )
