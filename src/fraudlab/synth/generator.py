"""Top-level generator: legitimate traffic + scenario injection + labels + outputs."""

from __future__ import annotations

import hashlib
import json
import platform
import zlib
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import fraudlab
from fraudlab.synth import catalog as C
from fraudlab.synth.batch import NO_ID, TxBatch
from fraudlab.synth.config import ALL_SCENARIOS, GeneratorConfig
from fraudlab.synth.scenarios import REGISTRY, Window
from fraudlab.synth.world import DAY, World, build_world, generate_legit

TABLES = ("transactions", "labels", "accounts", "merchants", "edges")


def _rng(seed: int, stream: str) -> np.random.Generator:
    """Independent, named RNG stream, so adding a stream never shifts another."""
    return np.random.default_rng([seed, zlib.crc32(stream.encode())])


@dataclass
class SyntheticDataset:
    config: GeneratorConfig
    transactions: pl.DataFrame
    labels: pl.DataFrame
    accounts: pl.DataFrame
    merchants: pl.DataFrame
    edges: pl.DataFrame
    stats: dict[str, Any]

    def tables(self) -> dict[str, pl.DataFrame]:
        return {name: getattr(self, name) for name in TABLES}

    def content_hash(self) -> str:
        return content_hash(self.tables())


def _allocate(total: int, weights: dict[str, float]) -> dict[str, int]:
    """Split ``total`` across weights by largest remainder (deterministic)."""
    names = [s for s in ALL_SCENARIOS if weights.get(s, 0) > 0]
    w = np.array([weights[s] for s in names], dtype=float)
    raw = total * w / w.sum()
    base = np.floor(raw).astype(int)
    order = np.argsort(-(raw - base), kind="stable")
    base[order[: total - base.sum()]] += 1
    return dict(zip(names, base.tolist(), strict=True))


def generate(cfg: GeneratorConfig) -> SyntheticDataset:
    """Generate a full synthetic dataset from ``cfg``. Deterministic in ``cfg``."""
    cfg.validate()
    world = build_world(cfg, _rng(cfg.seed, "world"))
    legit = generate_legit(world, cfg, _rng(cfg.seed, "legit"))
    n_legit_planned = len(legit)

    scenarios = {name: cls() for name, cls in REGISTRY.items()}
    fraud: list[TxBatch] = []
    budgets: dict[str, int] = {s: 0 for s in ALL_SCENARIOS}
    event_offset = 0
    for i, seg in enumerate(sorted(cfg.drift, key=lambda s: s.start_frac)):
        window = Window(int(seg.start_frac * world.horizon), int(seg.end_frac * world.horizon))
        in_seg = (legit.t >= window.start) & (legit.t < window.end)
        target = round(cfg.fraud_rate * in_seg.sum() / (1 - cfg.fraud_rate))
        for name, budget in _allocate(target, seg.weights).items():
            if budget == 0:
                continue
            rng = _rng(cfg.seed, f"scenario/{name}/{i}")
            fraud.append(scenarios[name].inject(world, rng, window, budget, event_offset))
            event_offset = scenarios[name].n_events
            budgets[name] += budget

    # Bust-out accounts go dark after the bust.
    if world.terminated:
        dark = np.zeros(len(legit), bool)
        for acc, t_end in world.terminated.items():
            dark |= (legit.account == acc) & (legit.t > t_end)
        legit = legit.take(np.flatnonzero(~dark))

    tx = TxBatch.concat([legit, *fraud])
    order = np.lexsort((tx.account, tx.t))  # stable, deterministic ordering
    tx = tx.take(order)
    return _assemble(cfg, world, tx, n_legit_planned, budgets)


def _labels(cfg: GeneratorConfig, tx: TxBatch, rng: np.random.Generator) -> dict[str, np.ndarray]:
    is_fraud = tx.scenario > 0
    n_events = int(tx.event.max()) + 1 if is_fraud.any() else 0
    ld = cfg.label_delay
    event_delay = rng.uniform(ld.min_days, ld.max_days, n_events) * DAY
    event_detected = rng.random(n_events) < cfg.detection_prob
    delay = np.zeros(len(tx))
    ev = tx.event[is_fraud]
    jitter = rng.uniform(0, ld.jitter_days, is_fraud.sum()) * DAY
    friendly = tx.scenario[is_fraud] == ALL_SCENARIOS.index("friendly_fraud") + 1
    delay[is_fraud] = (event_delay[ev] + jitter) * np.where(friendly, ld.friendly_multiplier, 1.0)
    observed = np.zeros(len(tx), np.int8)
    observed[is_fraud] = event_detected[ev]
    available = np.where(observed == 1, tx.t + delay.astype(np.int64), NO_ID)
    return {"is_fraud": is_fraud.astype(np.int8), "observed": observed, "available": available}


def _ts(start: datetime, seconds: np.ndarray) -> pl.Series:
    base_ms = int(np.datetime64(start, "ms").astype(np.int64))
    return pl.Series(base_ms + seconds.astype(np.int64) * 1000).cast(pl.Datetime("ms"))


def _null_sentinel(*cols: str) -> list[pl.Expr]:
    """Replace the ``NO_ID`` sentinel with null in integer id columns."""
    return [pl.when(pl.col(c) != NO_ID).then(pl.col(c)).alias(c) for c in cols]


def _assemble(
    cfg: GeneratorConfig,
    world: World,
    tx: TxBatch,
    n_legit_planned: int,
    budgets: dict[str, int],
) -> SyntheticDataset:
    start = datetime.fromisoformat(cfg.start_date)
    lab = _labels(cfg, tx, _rng(cfg.seed, "labels"))
    n = len(tx)
    tx_id = np.arange(n, dtype=np.int64)
    scen_names = ["legit", *ALL_SCENARIOS]

    transactions = pl.DataFrame(
        {
            "tx_id": tx_id,
            "timestamp": _ts(start, tx.t),
            "account_id": tx.account,
            "merchant_id": tx.merchant,
            "counterparty_id": tx.counterparty,
            "device_id": tx.device,
            "amount": tx.amount,
            "mcc": pl.Series(np.array(C.MCC_NAMES)[tx.mcc]).cast(pl.Enum(C.MCC_NAMES)),
            "channel": pl.Series(np.array(C.CHANNELS)[tx.channel]).cast(pl.Enum(C.CHANNELS)),
            "country": pl.Series(np.array(C.COUNTRIES)[tx.country]).cast(pl.Enum(C.COUNTRIES)),
        }
    ).with_columns(*_null_sentinel("merchant_id", "counterparty_id", "device_id"))
    available = _ts(start, np.where(lab["available"] == NO_ID, 0, lab["available"]))
    labels = pl.DataFrame(
        {
            "tx_id": tx_id,
            "is_fraud": lab["is_fraud"],
            "observed_label": lab["observed"],
            "label_available_at": available,
            "scenario": pl.Series(np.array(scen_names)[tx.scenario]).cast(pl.Enum(scen_names)),
            "fraud_event_id": tx.event,
        }
    ).with_columns(
        *_null_sentinel("fraud_event_id"),
        pl.when(pl.col("observed_label") == 1)
        .then(pl.col("label_available_at"))
        .otherwise(None)
        .alias("label_available_at"),
    )
    accounts = pl.DataFrame(
        {
            "account_id": np.arange(world.n_accounts, dtype=np.int32),
            "opened_at": _ts(start, world.acc_opened),
            "home_country": pl.Series(np.array(C.COUNTRIES)[world.acc_country]).cast(
                pl.Enum(C.COUNTRIES)
            ),
            "segment": pl.Series(np.array(C.SEGMENTS)[world.acc_segment]).cast(pl.Enum(C.SEGMENTS)),
            "account_role": pl.Series(np.array(C.ACCOUNT_ROLES)[world.acc_role]).cast(
                pl.Enum(C.ACCOUNT_ROLES)
            ),
        }
    )
    merchants = pl.DataFrame(
        {
            "merchant_id": np.arange(len(world.mer_mcc), dtype=np.int32),
            "mcc": pl.Series(np.array(C.MCC_NAMES)[world.mer_mcc]).cast(pl.Enum(C.MCC_NAMES)),
            "country": pl.Series(np.array(C.COUNTRIES)[world.mer_country]).cast(
                pl.Enum(C.COUNTRIES)
            ),
            "opened_at": _ts(start, np.maximum(world.mer_opened, -20 * 365 * DAY)),
        }
    )
    edges = _edges(transactions)

    n_fraud = int(lab["is_fraud"].sum())
    stats = {
        "n_transactions": n,
        "n_fraud": n_fraud,
        "fraud_rate": n_fraud / n,
        "n_legit_planned": n_legit_planned,
        "n_accounts": world.n_accounts,
        "n_merchants": len(world.mer_mcc),
        "n_devices": world.next_device,
        "n_edges": edges.height,
        "fraud_budget_by_scenario": {k: v for k, v in budgets.items() if v},
        "n_bust_out_accounts_terminated": len(world.terminated),
    }
    return SyntheticDataset(cfg, transactions, labels, accounts, merchants, edges, stats)


def _edges(transactions: pl.DataFrame) -> pl.DataFrame:
    """Aggregated typed edge list: account→merchant, account→account, account→device."""
    specs = (
        ("pays", "merchant", "merchant_id"),
        ("transfers", "account", "counterparty_id"),
        ("uses_device", "device", "device_id"),
    )
    parts = []
    for etype, dst_type, col in specs:
        parts.append(
            transactions.filter(pl.col(col).is_not_null())
            .group_by("account_id", col)
            .agg(
                pl.col("timestamp").min().alias("first_seen"),
                pl.col("timestamp").max().alias("last_seen"),
                pl.len().cast(pl.Int32).alias("n_tx"),
            )
            .select(
                pl.lit("account").alias("src_type"),
                pl.col("account_id").alias("src_id"),
                pl.lit(dst_type).alias("dst_type"),
                pl.col(col).cast(pl.Int32).alias("dst_id"),
                pl.lit(etype).alias("edge_type"),
                "first_seen",
                "last_seen",
                "n_tx",
            )
        )
    kinds = ("account", "merchant", "device")
    return (
        pl.concat(parts)
        .with_columns(
            pl.col("src_type").cast(pl.Enum(kinds)),
            pl.col("dst_type").cast(pl.Enum(kinds)),
            pl.col("edge_type").cast(pl.Enum([s[0] for s in specs])),
        )
        .sort("edge_type", "src_id", "dst_id")
    )


def content_hash(tables: dict[str, pl.DataFrame]) -> str:
    """SHA-256 over column names, dtypes and values of every table, in a fixed order.

    Independent of parquet writer metadata, so it identifies *content*.
    """
    h = hashlib.sha256()
    for name in TABLES:
        df = tables[name]
        h.update(name.encode())
        for col in df.columns:
            s = df[col]
            h.update(f"{col}:{s.dtype}".encode())
            if isinstance(s.dtype, pl.Enum):
                s = s.to_physical()
            elif s.dtype == pl.Datetime("ms"):
                s = s.cast(pl.Int64)
            h.update(s.is_null().to_numpy().tobytes())
            h.update(s.fill_null(0).to_numpy().tobytes())
    return h.hexdigest()


def write(ds: SyntheticDataset, out_dir: str | Path) -> dict[str, Any]:
    """Write all tables as parquet plus ``manifest.json``; returns the manifest."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for name, df in ds.tables().items():
        df.write_parquet(out / f"{name}.parquet", compression="zstd", statistics=False)
    manifest = {
        "generator": "fraudlab.synth",
        "generator_version": fraudlab.__version__,
        "preset": ds.config.preset,
        "seed": ds.config.seed,
        "params": ds.config.to_dict(),
        "content_hash": ds.content_hash(),
        "stats": ds.stats,
        "environment": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "polars": pl.__version__,
            "platform": platform.platform(),
        },
    }
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    return manifest


def load(data_dir: str | Path) -> SyntheticDataset:
    """Read a dataset written by :func:`write`."""
    d = Path(data_dir)
    manifest = json.loads((d / "manifest.json").read_text())
    from fraudlab.synth.config import DriftSegment, LabelDelay

    p = dict(manifest["params"])
    p["drift"] = tuple(DriftSegment(**s) for s in p["drift"])
    p["label_delay"] = LabelDelay(**p["label_delay"])
    cfg = GeneratorConfig(**p)
    tables = {name: pl.read_parquet(d / f"{name}.parquet") for name in TABLES}
    return SyntheticDataset(cfg, **tables, stats=manifest["stats"])
