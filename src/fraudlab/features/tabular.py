"""Past-only per-transaction features.

Every feature for a transaction at time ``T`` uses only transactions strictly
before ``T`` and labels that were *known* before ``min(T, label_cutoff)``.
``tests/test_features.py`` checks this by perturbing future rows.

Features are in two groups:

* ``TABULAR``: the transaction itself plus its own account's history and static
  attributes (velocity, amount ratios, device novelty, account and merchant age).
  This is what a classic feature-engineered fraud model uses
  (``bahnsen2016feature``).
* ``GRAPH``: hand-crafted relational features about *other* entities (accounts
  per device, counterparty fan-in, merchant customer base, known-fraud
  neighbours, connected-component size, PageRank). This is the RQ3 ablation.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Literal

import numpy as np
import polars as pl
import scipy.sparse as sp
from scipy.sparse.csgraph import connected_components

from fraudlab.synth import SyntheticDataset

LabelMode = Literal["oracle", "delayed"]

TABULAR: tuple[str, ...] = (
    "log_amount",
    "mcc",
    "channel",
    "is_foreign",
    "is_round",
    "hour_sin",
    "hour_cos",
    "segment",
    "account_age_days",
    "merchant_age_days",
    "acc_hist_cnt",
    "acc_cnt_1h",
    "acc_cnt_24h",
    "acc_cnt_7d",
    "acc_sum_24h",
    "acc_mean_amt_30d",
    "amt_ratio_30d",
    "log_secs_since_last",
    "acc_same_merchant_24h",
    "device_missing",
    "device_new_for_account",
    "acc_n_devices_past",
    "acc_fanin_24h",
    "acc_known_fraud_cnt",
)

GRAPH: tuple[str, ...] = (
    "dev_n_accounts_past",
    "dev_known_fraud_cnt",
    "cp_age_days",
    "cp_n_senders_past",
    "cp_fanin_24h",
    "cp_known_fraud_cnt",
    "merch_n_accounts_past",
    "merch_cnt_24h",
    "merch_known_fraud_cnt",
    "acc_component_size",
    "acc_pagerank",
)

CATEGORICAL: tuple[str, ...] = ("mcc", "channel", "segment")

#: Per-step inputs of the sequence model (individual information only).
SEQUENCE_STEP: tuple[str, ...] = (
    "log_amount",
    "mcc",
    "channel",
    "is_foreign",
    "is_round",
    "hour_sin",
    "hour_cos",
    "device_missing",
    "device_new_for_account",
    "log_secs_since_last",
)

DAY_MS = 86_400_000


@dataclass(frozen=True)
class LabelView:
    """Which fraud labels may be used, and when.

    ``mode`` controls the **training targets** only: under ``oracle`` every
    detected fraud in the training window is labelled (the classic benchmark
    assumption, as if training waits until all labels arrive); under ``delayed``
    only labels with ``label_available_at < cutoff`` are positive (RQ4).

    Label-derived **features** (known-fraud counts) always use realistic
    availability (``label_available_at``) before ``min(T, cutoff)``, whatever the
    mode. Otherwise training rows would see instantly known labels from earlier
    in the same fraud event, which test rows can never see (a train/test feature
    shift). See DECISIONS D020.
    """

    cutoff: datetime
    mode: LabelMode = "oracle"


def known_labels(ds: SyntheticDataset, view: LabelView) -> pl.DataFrame:
    """``tx_id, known_at, known_fraud``: labels known before the cutoff under ``view.mode``."""
    lab = ds.labels.join(ds.transactions.select("tx_id", "timestamp"), on="tx_id")
    known_at = pl.col("timestamp") if view.mode == "oracle" else pl.col("label_available_at")
    return (
        lab.filter(pl.col("observed_label") == 1)
        .with_columns(known_at.alias("known_at"))
        .filter(pl.col("known_at") < view.cutoff)
        .select("tx_id", "known_at", pl.lit(1).cast(pl.Int32).alias("known_fraud"))
    )


def training_labels(ds: SyntheticDataset, view: LabelView) -> pl.DataFrame:
    """``tx_id, y_train``: label a model may train on (1 only if known before cutoff)."""
    k = known_labels(ds, view)
    return (
        ds.labels.select("tx_id")
        .join(k, on="tx_id", how="left")
        .select("tx_id", pl.col("known_fraud").fill_null(0).alias("y_train"))
    )


def _distinct_before(
    tx: pl.DataFrame, key: str, other: str, name: str, query: str = "timestamp"
) -> pl.DataFrame:
    """For each row: number of distinct ``other`` seen with ``key`` strictly before ``query``."""
    pairs = (
        tx.filter(pl.col(key).is_not_null() & pl.col(other).is_not_null())
        .group_by(key, other)
        .agg(pl.col("timestamp").min().alias("first_ts"))
        .sort("first_ts")
        .with_columns(pl.int_range(1, pl.len() + 1).over(key).alias(name))
        .select(key, "first_ts", name)
    )
    q = tx.select("tx_id", key, query).filter(pl.col(key).is_not_null()).sort(query)
    out = q.join_asof(
        pairs,
        left_on=query,
        right_on="first_ts",
        by=key,
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    return out.select("tx_id", pl.col(name).fill_null(0))


def _known_before(
    tx: pl.DataFrame,
    known: pl.DataFrame,
    key: str,
    name: str,
    cutoff: datetime,
    event_key: str | None = None,
) -> pl.DataFrame:
    """Known-fraud transactions involving entity ``key`` known before ``min(T, cutoff)``.

    ``event_key`` names the column of the *labelled* transaction that identifies
    the entity (defaults to ``key``).
    """
    ek = event_key or key
    events = (
        tx.select("tx_id", pl.col(ek).alias(key))
        .join(known, on="tx_id")
        .filter(pl.col(key).is_not_null())
        .sort("known_at")
        .with_columns(pl.int_range(1, pl.len() + 1).over(key).alias(name))
        .select(key, "known_at", name)
    )
    q = (
        tx.select("tx_id", key, pl.min_horizontal("timestamp", pl.lit(cutoff)).alias("q"))
        .filter(pl.col(key).is_not_null())
        .sort("q")
    )
    out = q.join_asof(
        events,
        left_on="q",
        right_on="known_at",
        by=key,
        strategy="backward",
        allow_exact_matches=False,
        check_sortedness=False,
    )
    return out.select("tx_id", pl.col(name).fill_null(0))


def _rolling_count(col_key: str, period: str, name: str, value: str = "one") -> pl.Expr:
    return (
        pl.col(value)
        .rolling_sum_by("timestamp", window_size=period, closed="left")
        .over(col_key)
        .alias(name)
    )


def _snapshot_graph_features(tx: pl.DataFrame, n_accounts: int) -> pl.DataFrame:
    """Component size and PageRank of the account in the start-of-day snapshot.

    The snapshot for day ``d`` contains account–device and account–account
    (transfer) edges first seen before day ``d`` starts. Merchants are left out:
    they connect almost everyone and would merge all components.
    """
    t0 = tx["timestamp"].min().replace(hour=0, minute=0, second=0, microsecond=0)
    dev = (
        tx.filter(pl.col("device_id").is_not_null())
        .group_by("account_id", "device_id")
        .agg(pl.col("timestamp").min().alias("first_ts"))
        .with_columns((pl.col("device_id") + n_accounts).alias("dst"))
    )
    trf = (
        tx.filter(pl.col("counterparty_id").is_not_null())
        .group_by("account_id", "counterparty_id")
        .agg(pl.col("timestamp").min().alias("first_ts"))
        .rename({"counterparty_id": "dst"})
    )
    edges = pl.concat(
        [dev.select("account_id", "dst", "first_ts"), trf.select("account_id", "dst", "first_ts")]
    )
    edges = edges.with_columns(
        ((pl.col("first_ts") - t0).dt.total_milliseconds() // DAY_MS).alias("day")
    )
    src_all = edges["account_id"].to_numpy().astype(np.int64)
    dst_all = edges["dst"].to_numpy().astype(np.int64)
    day_all = edges["day"].to_numpy()
    n_days = int((tx["timestamp"].max() - t0).total_seconds() // 86400) + 1
    rows = []
    for d in range(n_days):
        m = day_all < d  # strictly before the start of day d
        # Only nodes present in the snapshot exist, so nothing depends on the future.
        nodes, inv = np.unique(np.concatenate([src_all[m], dst_all[m]]), return_inverse=True)
        k = len(nodes)
        size = np.ones(n_accounts, np.float32)
        rank = np.ones(n_accounts, np.float32)  # PageRank x n: 1.0 = average node
        if k:
            e = int(m.sum())
            a = sp.coo_matrix((np.ones(e), (inv[:e], inv[e:])), shape=(k, k)).tocsr()
            a = ((a + a.T) > 0).astype(np.float64)
            _, comp = connected_components(a, directed=False)
            is_acc = nodes < n_accounts
            size[nodes[is_acc]] = np.bincount(comp)[comp[is_acc]]
            rank[nodes[is_acc]] = _pagerank(a)[is_acc] * k
        rows.append(
            pl.DataFrame(
                {
                    "day": np.full(n_accounts, d, np.int64),
                    "account_id": np.arange(n_accounts, dtype=np.int32),
                    "acc_component_size": size,
                    "acc_pagerank": rank,
                }
            )
        )
    return pl.concat(rows)


def _pagerank(a: sp.csr_matrix, alpha: float = 0.85, iters: int = 30) -> np.ndarray:
    n = a.shape[0]
    deg = np.asarray(a.sum(axis=1)).ravel()
    inv = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)
    p = np.full(n, 1.0 / n)
    at = a.T.tocsr()
    for _ in range(iters):
        dangling = p[deg == 0].sum()
        p = alpha * (at @ (p * inv) + dangling / n) + (1 - alpha) / n
    return p


def build_features(ds: SyntheticDataset, view: LabelView) -> pl.DataFrame:
    """One row per transaction: ``tx_id``, ``timestamp``, ``amount`` and all features."""
    tx = (
        ds.transactions.join(
            ds.accounts.select("account_id", "opened_at", "home_country", "segment"),
            on="account_id",
        )
        .join(
            ds.merchants.select("merchant_id", pl.col("opened_at").alias("merchant_opened")),
            on="merchant_id",
            how="left",
        )
        .join(
            ds.accounts.select(
                pl.col("account_id").alias("counterparty_id"),
                pl.col("opened_at").alias("cp_opened"),
            ),
            on="counterparty_id",
            how="left",
        )
        .sort("timestamp", "tx_id")
        .with_columns(pl.lit(1.0).alias("one"))
    )

    hour = tx["timestamp"].dt.hour() + tx["timestamp"].dt.minute() / 60
    days = lambda a, b: (pl.col(a) - pl.col(b)).dt.total_milliseconds() / DAY_MS  # noqa: E731

    f = tx.with_columns(
        pl.col("amount").log1p().alias("log_amount"),
        pl.col("mcc").to_physical().cast(pl.Int32).alias("mcc"),
        pl.col("channel").to_physical().cast(pl.Int32).alias("channel"),
        pl.col("segment").to_physical().cast(pl.Int32).alias("segment"),
        (pl.col("country").cast(pl.String) != pl.col("home_country").cast(pl.String))
        .cast(pl.Int8)
        .alias("is_foreign"),
        ((pl.col("amount") % 50) == 0).cast(pl.Int8).alias("is_round"),
        pl.Series("hour_sin", np.sin(2 * np.pi * hour.to_numpy() / 24)),
        pl.Series("hour_cos", np.cos(2 * np.pi * hour.to_numpy() / 24)),
        days("timestamp", "opened_at").alias("account_age_days"),
        days("timestamp", "merchant_opened").fill_null(-1).alias("merchant_age_days"),
        days("timestamp", "cp_opened").fill_null(-1).alias("cp_age_days"),
        pl.col("device_id").is_null().cast(pl.Int8).alias("device_missing"),
        (pl.int_range(pl.len()).over("account_id")).alias("acc_hist_cnt"),
        _rolling_count("account_id", "1h", "acc_cnt_1h"),
        _rolling_count("account_id", "24h", "acc_cnt_24h"),
        _rolling_count("account_id", "7d", "acc_cnt_7d"),
        pl.col("amount")
        .rolling_sum_by("timestamp", window_size="24h", closed="left")
        .over("account_id")
        .alias("acc_sum_24h"),
        pl.col("amount")
        .rolling_mean_by("timestamp", window_size="30d", closed="left")
        .over("account_id")
        .alias("acc_mean_amt_30d"),
        (pl.col("timestamp") - pl.col("timestamp").shift(1).over("account_id"))
        .dt.total_seconds()
        .alias("secs_since_last"),
    ).with_columns(
        (pl.col("amount") / pl.col("acc_mean_amt_30d")).alias("amt_ratio_30d"),
        pl.col("secs_since_last").log1p().fill_null(-1).alias("log_secs_since_last"),
    )

    # Same-merchant velocity, merchant activity, transfers received.
    f = f.with_columns(
        pl.when(pl.col("merchant_id").is_not_null())
        .then(
            pl.col("one")
            .rolling_sum_by("timestamp", window_size="24h", closed="left")
            .over("account_id", "merchant_id")
        )
        .otherwise(0)
        .alias("acc_same_merchant_24h"),
        pl.when(pl.col("merchant_id").is_not_null())
        .then(
            pl.col("one")
            .rolling_sum_by("timestamp", window_size="24h", closed="left")
            .over("merchant_id")
        )
        .otherwise(0)
        .alias("merch_cnt_24h"),
    )
    received = f.filter(pl.col("counterparty_id").is_not_null()).select(
        pl.col("counterparty_id").alias("acct"), "timestamp"
    )
    f = f.with_columns(
        _fanin(f, received, "account_id").alias("acc_fanin_24h"),
        _fanin(f, received, "counterparty_id").alias("cp_fanin_24h"),
    )

    # Distinct-before counts. Label features always use realistic availability.
    known = known_labels(ds, LabelView(view.cutoff, "delayed"))
    parts = [
        _distinct_before(f, "device_id", "account_id", "dev_n_accounts_past"),
        _distinct_before(f, "account_id", "device_id", "acc_n_devices_past"),
        _distinct_before(
            f.with_columns(pl.col("account_id").alias("sender")),
            "counterparty_id",
            "sender",
            "cp_n_senders_past",
        ),
        _distinct_before(f, "merchant_id", "account_id", "merch_n_accounts_past"),
        _known_before(f, known, "account_id", "acc_known_fraud_cnt", view.cutoff),
        _known_before(f, known, "device_id", "dev_known_fraud_cnt", view.cutoff),
        _known_before(f, known, "merchant_id", "merch_known_fraud_cnt", view.cutoff),
        # fraud involving the counterparty, as sender or as receiver
        _known_before(f, known, "counterparty_id", "cp_known_recv", view.cutoff),
        _known_before(
            f, known, "counterparty_id", "cp_known_sent", view.cutoff, event_key="account_id"
        ),
    ]
    for p in parts:
        f = f.join(p, on="tx_id", how="left")
    first_use = (
        f.filter(pl.col("device_id").is_not_null())
        .group_by("account_id", "device_id")
        .agg(pl.col("tx_id").min().alias("first_tx"))
    )
    f = f.join(first_use, on=["account_id", "device_id"], how="left").with_columns(
        (pl.col("tx_id") == pl.col("first_tx"))
        .fill_null(False)
        .cast(pl.Int8)
        .alias("device_new_for_account"),
        (pl.col("cp_known_recv").fill_null(0) + pl.col("cp_known_sent").fill_null(0)).alias(
            "cp_known_fraud_cnt"
        ),
    )

    snap = _snapshot_graph_features(f, ds.accounts.height)
    t0 = f["timestamp"].min().replace(hour=0, minute=0, second=0, microsecond=0)
    f = f.with_columns(
        ((pl.col("timestamp") - t0).dt.total_milliseconds() // DAY_MS).alias("day")
    ).join(snap, on=["day", "account_id"], how="left")

    cols = list(TABULAR) + list(GRAPH)
    return f.sort("timestamp", "tx_id").select(
        "tx_id",
        "timestamp",
        "account_id",
        "day",
        "amount",
        *[
            pl.col(c).cast(pl.Float32).fill_null(0.0)
            if c not in ("amt_ratio_30d", "acc_mean_amt_30d")
            else pl.col(c).cast(pl.Float32)
            for c in cols
        ],
    )


def _fanin(f: pl.DataFrame, received: pl.DataFrame, key: str) -> pl.Series:
    """Transfers received by entity ``key`` in the 24h strictly before each row."""
    ev = received.sort("timestamp").with_columns(
        pl.int_range(1, pl.len() + 1).over("acct").alias("cum")
    )
    q = f.select("tx_id", pl.col(key).alias("acct"), "timestamp").with_columns(
        (pl.col("timestamp") - pl.duration(hours=24)).alias("t_lo")
    )
    hi = (
        q.sort("timestamp")
        .join_asof(
            ev.select("acct", "timestamp", "cum"),
            on="timestamp",
            by="acct",
            strategy="backward",
            allow_exact_matches=False,
            check_sortedness=False,
        )
        .select("tx_id", pl.col("cum").fill_null(0).alias("hi"))
    )
    lo = (
        q.sort("t_lo")
        .join_asof(
            ev.select("acct", pl.col("timestamp").alias("t_lo"), "cum"),
            on="t_lo",
            by="acct",
            strategy="backward",
            allow_exact_matches=True,
            check_sortedness=False,
        )
        .select("tx_id", pl.col("cum").fill_null(0).alias("lo"))
    )
    out = (
        f.select("tx_id")
        .join(hi, on="tx_id", how="left", maintain_order="left")
        .join(lo, on="tx_id", how="left", maintain_order="left")
    )
    return (out["hi"] - out["lo"]).fill_null(0).cast(pl.Float32)
