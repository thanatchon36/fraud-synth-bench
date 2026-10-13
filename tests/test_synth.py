"""Tests for the synthetic generator: determinism, rates, schedules, scenario signatures."""

from __future__ import annotations

import json
from dataclasses import replace
from datetime import datetime
from functools import cache

import polars as pl
import pytest

from fraudlab.synth import (
    ALL_SCENARIOS,
    DRIFT_RQ2,
    RING_SCENARIOS,
    DriftSegment,
    SyntheticDataset,
    generate,
    load,
    preset,
    write,
)
from fraudlab.synth.config import LabelDelay

DAY_MS = 86_400_000


@cache
def single(name: str, seed: int = 0) -> SyntheticDataset:
    """A tiny dataset whose fraud comes only from scenario ``name``."""
    cfg = preset("tiny", seed=seed, fraud_rate=0.03, drift=(DriftSegment(0, 1, {name: 1.0}),))
    return generate(cfg)


def joined(ds: SyntheticDataset) -> pl.DataFrame:
    return ds.transactions.join(ds.labels, on="tx_id")


def split(ds: SyntheticDataset) -> tuple[pl.DataFrame, pl.DataFrame]:
    df = joined(ds)
    return df.filter(pl.col("is_fraud") == 1), df.filter(pl.col("is_fraud") == 0)


# ---------------------------------------------------------------- determinism


def test_same_seed_same_hash() -> None:
    assert (
        generate(preset("tiny", seed=7)).content_hash()
        == generate(preset("tiny", seed=7)).content_hash()
    )


def test_different_seed_different_hash() -> None:
    assert (
        generate(preset("tiny", seed=1)).content_hash()
        != generate(preset("tiny", seed=2)).content_hash()
    )


def test_write_load_roundtrip(tmp_path) -> None:
    ds = generate(preset("tiny", seed=3))
    manifest = write(ds, tmp_path)
    assert manifest["content_hash"] == ds.content_hash()
    for key in ("generator_version", "params", "seed", "preset", "stats", "environment"):
        assert key in manifest
    back = load(tmp_path)
    assert back.content_hash() == ds.content_hash()
    assert back.config == ds.config
    on_disk = json.loads((tmp_path / "manifest.json").read_text())
    assert on_disk["content_hash"] == ds.content_hash()


# ---------------------------------------------------------------- rates & schema


@pytest.mark.parametrize("rate", [0.001, 0.005, 0.01])
def test_fraud_rate_tolerance(rate: float) -> None:
    ds = generate(preset("tiny", seed=0, fraud_rate=rate))
    realised = ds.labels["is_fraud"].mean()
    assert abs(realised - rate) / rate < 0.05, realised


def test_small_preset_size() -> None:
    ds = generate(preset("small", seed=0))
    assert 170_000 < ds.transactions.height < 230_000
    assert ds.config.preset == "small"


def test_schema_and_ordering() -> None:
    ds = generate(preset("tiny", seed=0))
    tx = ds.transactions
    assert tx.columns == [
        "tx_id",
        "timestamp",
        "account_id",
        "merchant_id",
        "counterparty_id",
        "device_id",
        "amount",
        "mcc",
        "channel",
        "country",
    ]
    assert tx["tx_id"].is_unique().all()
    assert tx["timestamp"].is_sorted()
    assert (tx["amount"] > 0).all()
    assert ds.labels.height == tx.height
    start = pl.datetime(2025, 1, 1)
    in_range = tx.select(
        (
            (pl.col("timestamp") >= start) & (pl.col("timestamp") < start + pl.duration(days=30))
        ).all()
    ).item()
    assert in_range
    # Card-present payments carry no device; transfers carry a counterparty, not a merchant.
    pos = tx.filter(pl.col("channel") == "pos")
    assert pos["device_id"].null_count() == pos.height
    transfers = tx.filter(pl.col("channel") == "transfer")
    assert transfers["merchant_id"].null_count() == transfers.height
    assert transfers["counterparty_id"].null_count() == 0
    # edges reference every typed relation
    assert set(ds.edges["edge_type"].unique().to_list()) == {"pays", "transfers", "uses_device"}


def test_legit_amounts_lognormal_like() -> None:
    _, legit = split(generate(preset("tiny", seed=0)))
    a = legit.filter(pl.col("channel") != "transfer")["amount"]
    # Heavy right tail: mean well above median, and log-amounts roughly symmetric.
    assert a.mean() > 1.3 * a.median()
    logs = a.log()
    assert abs(logs.mean() - logs.median()) < 0.3


# ---------------------------------------------------------------- labels


def test_label_delay_bounds_and_detection() -> None:
    ld = LabelDelay(min_days=7, max_days=30, jitter_days=2, friendly_multiplier=1.5)
    ds = generate(preset("tiny", seed=0, fraud_rate=0.02, label_delay=ld))
    df = joined(ds)
    legit = df.filter(pl.col("is_fraud") == 0)
    assert legit["observed_label"].sum() == 0
    assert legit["label_available_at"].null_count() == legit.height
    fraud = df.filter(pl.col("is_fraud") == 1)
    observed = fraud.filter(pl.col("observed_label") == 1)
    assert observed["label_available_at"].null_count() == 0
    unobserved = fraud.filter(pl.col("observed_label") == 0)
    assert unobserved["label_available_at"].null_count() == unobserved.height
    delay = observed.select(
        (
            (pl.col("label_available_at") - pl.col("timestamp")).dt.total_milliseconds() / DAY_MS
        ).alias("d"),
        "scenario",
    )
    assert delay["d"].min() >= 7
    assert delay["d"].max() <= (30 + 2) * 1.5
    friendly = delay.filter(pl.col("scenario") == "friendly_fraud")["d"].median()
    other = delay.filter(pl.col("scenario") != "friendly_fraud")["d"].median()
    assert friendly > 1.2 * other
    # Detection is per event, ~95%.
    ev = fraud.group_by("fraud_event_id").agg(pl.col("observed_label").max())
    assert 0.85 < ev["observed_label"].mean() <= 1.0


# ---------------------------------------------------------------- schedules


def test_drift_schedule_switches_scenarios() -> None:
    cfg = preset("tiny", seed=0, n_days=90, fraud_rate=0.01, drift=DRIFT_RQ2)
    fraud, _ = split(generate(cfg))
    cut = pl.datetime(2025, 1, 1) + pl.duration(days=54)
    before = set(fraud.filter(pl.col("timestamp") < cut)["scenario"].unique().to_list())
    after = set(
        fraud.filter(pl.col("timestamp") >= cut + pl.duration(days=3))["scenario"]
        .unique()
        .to_list()
    )
    assert before <= {"card_testing", "account_takeover", "velocity_abuse"}
    assert after <= {"bust_out", "synthetic_identity", "mule_network"}
    assert len(before) == 3 and len(after) == 3


@pytest.mark.parametrize("share", [0.0, 1.0])
def test_ring_share(share: float) -> None:
    ds = generate(preset("tiny", seed=0).with_ring_share(share))
    scen = set(split(ds)[0]["scenario"].unique().to_list())
    if share == 0.0:
        assert scen.isdisjoint(RING_SCENARIOS)
    else:
        assert scen == set(RING_SCENARIOS)


def test_invalid_drift_rejected() -> None:
    bad = (DriftSegment(0.0, 0.5, {"card_testing": 1}),)
    with pytest.raises(ValueError):
        generate(preset("tiny", drift=bad))
    with pytest.raises(ValueError):
        generate(preset("tiny", drift=(DriftSegment(0, 1, {"nope": 1}),)))


def test_noise_knob_adds_lookalikes() -> None:
    def lookalikes(noise: float) -> dict[str, float]:
        ds = generate(replace(preset("tiny", seed=0), noise=noise))
        legit = split(ds)[1]
        pos = legit.filter(pl.col("channel") == "pos").join(ds.accounts, on="account_id")
        online = legit.filter(pl.col("device_id").is_not_null())
        return {
            # card-present abroad only happens on trips
            "pos_abroad": (
                pos["country"].cast(pl.String) != pos["home_country"].cast(pl.String)
            ).mean(),
            "devices_per_acc": online.group_by("account_id")
            .agg(pl.col("device_id").n_unique())["device_id"]
            .mean(),
            "accounts_per_device": online.group_by("device_id")
            .agg(pl.col("account_id").n_unique())["account_id"]
            .mean(),
        }

    calm, noisy = lookalikes(0.0), lookalikes(1.0)
    assert calm["pos_abroad"] == 0.0 < noisy["pos_abroad"]
    assert calm["accounts_per_device"] == 1.0 < noisy["accounts_per_device"]
    assert noisy["devices_per_acc"] > calm["devices_per_acc"]


# ---------------------------------------------------------------- scenario signatures


def test_every_scenario_registered() -> None:
    from fraudlab.synth.scenarios import REGISTRY

    assert set(REGISTRY) == set(ALL_SCENARIOS)
    for cls in REGISTRY.values():
        assert cls.__doc__ and len(cls.__doc__) > 100, f"{cls.__name__} needs a docstring"


def test_card_testing_signature() -> None:
    fraud, legit = split(single("card_testing"))
    burst = fraud.filter(pl.col("amount") <= 3)
    assert burst.height > 0.6 * fraud.height
    assert burst["amount"].median() < 0.1 * legit["amount"].median()
    gaps = (
        burst.sort("timestamp")
        .group_by("fraud_event_id")
        .agg(pl.col("timestamp").diff().dt.total_seconds().median().alias("gap"))
    )
    assert gaps["gap"].median() < 120
    per_event = burst.group_by("fraud_event_id").agg(pl.col("merchant_id").n_unique())
    assert per_event["merchant_id"].median() >= 4


def test_account_takeover_signature() -> None:
    ds = single("account_takeover")
    fraud, legit = split(ds)
    known = legit.filter(pl.col("device_id").is_not_null()).select("account_id", "device_id")
    known = known.unique().with_columns(pl.lit(True).alias("known"))
    f = fraud.join(known, on=["account_id", "device_id"], how="left")
    assert f["known"].is_null().mean() > 0.9  # new device for the account
    med = legit.group_by("account_id").agg(pl.col("amount").median().alias("med"))
    ratio = fraud.join(med, on="account_id").select(pl.col("amount") / pl.col("med"))
    assert ratio.to_series().median() > 3
    night = lambda df: (df["timestamp"].dt.hour() < 5).mean()  # noqa: E731
    assert night(fraud) > 0.4 > 2 * night(legit)


def test_velocity_abuse_signature() -> None:
    fraud, legit = split(single("velocity_abuse"))
    per_hour = lambda df: (  # noqa: E731
        df.group_by("account_id", pl.col("timestamp").dt.truncate("1h")).len()["len"].max()
    )
    per_event = fraud.group_by("fraud_event_id").agg(
        pl.len().alias("n"),
        (pl.col("timestamp").max() - pl.col("timestamp").min()).dt.total_hours().alias("h"),
        pl.col("mcc").n_unique().alias("mccs"),
    )
    assert per_event["n"].median() >= 8
    assert per_event["h"].max() <= 3
    assert (per_event["mccs"] == 1).all()
    assert per_hour(fraud) > 2 * per_hour(legit)


def test_friendly_fraud_signature() -> None:
    ds = single("friendly_fraud")
    fraud, legit = split(ds)
    main = (
        legit.filter(pl.col("device_id").is_not_null())
        .group_by("account_id")
        .agg(pl.col("device_id").mode().first().alias("main"))
    )
    f = fraud.join(main, on="account_id")
    assert (f["device_id"] == f["main"]).mean() > 0.8  # customer's own device
    med = legit.group_by("account_id").agg(pl.col("amount").median().alias("med"))
    ratio = fraud.join(med, on="account_id").select(pl.col("amount") / pl.col("med"))
    assert 0.8 < ratio.to_series().median() < 5  # elevated but plausible


def test_bust_out_signature() -> None:
    ds = single("bust_out")
    fraud, legit = split(ds)
    for acc in fraud["account_id"].unique().to_list()[:10]:
        f = fraud.filter(pl.col("account_id") == acc)
        lg = legit.filter(pl.col("account_id") == acc)
        assert lg.filter(pl.col("timestamp") > f["timestamp"].max()).height == 0  # goes dark
        prior = lg.filter(pl.col("timestamp") < f["timestamp"].min())
        if prior.height < 3:
            continue
        span_days = 30
        daily = prior["amount"].sum() / span_days
        bust_days = max(1.0, (f["timestamp"].max() - f["timestamp"].min()).total_seconds() / 86400)
        assert f["amount"].sum() / bust_days > 3 * daily


def test_mule_network_signature() -> None:
    ds = single("mule_network")
    fraud, _ = split(ds)
    assert (fraud["channel"] == "transfer").all()
    mules = ds.accounts.filter(pl.col("account_role") == "mule")["account_id"]
    mule_tx = fraud.filter(pl.col("account_id").is_in(mules.implode()))
    per_device = mule_tx.group_by("device_id").agg(pl.col("account_id").n_unique())
    assert per_device["account_id"].mean() >= 2  # devices shared within the ring
    collectors = ds.accounts.filter(pl.col("account_role") == "collector")["account_id"]
    fan_in = (
        fraud.filter(pl.col("counterparty_id").is_in(collectors.implode()))
        .group_by("counterparty_id")
        .agg(pl.col("account_id").n_unique())
    )
    assert fan_in["account_id"].median() >= 4
    # mules are young accounts
    opened = ds.accounts.filter(pl.col("account_role") == "mule")["opened_at"]
    first = mule_tx["timestamp"].min()
    assert ((first - opened).dt.total_days() < 60).mean() > 0.9


def test_merchant_collusion_signature() -> None:
    ds = single("merchant_collusion")
    df = joined(ds)
    fraud = df.filter(pl.col("is_fraud") == 1)
    shops = fraud["merchant_id"].unique()
    on_shops = df.filter(pl.col("merchant_id").is_in(shops.implode()))
    assert on_shops["is_fraud"].mean() == 1.0  # no legitimate customers
    assert fraud["amount"].is_in([100.0, 200.0, 250.0, 500.0]).mean() > 0.6
    opened = ds.merchants.filter(pl.col("merchant_id").is_in(shops.implode()))["opened_at"]
    assert (opened >= datetime(2024, 12, 20)).all()


def test_synthetic_identity_signature() -> None:
    ds = single("synthetic_identity")
    df = joined(ds)
    synth = ds.accounts.filter(pl.col("account_role") == "synthetic")
    tx = df.filter(pl.col("account_id").is_in(synth["account_id"].implode()))
    assert tx["is_fraud"].mean() == 1.0
    per_device = tx.group_by("device_id").agg(pl.col("account_id").n_unique())
    assert per_device["account_id"].mean() >= 2.5
    assert (synth["opened_at"] >= datetime(2024, 12, 25)).mean() > 0.9


# ---------------------------------------------------------------- full preset


@pytest.mark.slow
def test_full_preset_generates() -> None:
    ds = generate(preset("full", seed=0))
    assert 1_700_000 < ds.transactions.height < 2_300_000
    assert abs(ds.labels["is_fraud"].mean() - 0.005) / 0.005 < 0.05
