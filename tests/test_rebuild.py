"""Rebuild tests (§8.7). No network: every response comes from a stored bundle."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from costco_gas.config import load_config
from costco_gas.rebuild import rebuild
from costco_gas.schema import read_rows_csv_gz
from costco_gas.store import open_store
from helpers_rebuild import (
    checkout_with_config,
    drop_au_e10,
    make_bundle,
    restore_config,
    seed_month,
)

NOW = datetime(2026, 10, 2, 4, 41, tzinfo=UTC)


def _daily(store, tag, day, tmp_path, name="d.csv.gz"):
    return read_rows_csv_gz(store.download(tag, f"costco-gas-{day}.csv.gz", tmp_path / name))


def test_rebuild_applies_the_checkouts_grades_to_stored_responses(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha-one")
    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    bundle = make_bundle(
        tmp_path / "bundles",
        capture_id="2026-09-01T1817Z",
        config_dir=checkout / "config",
    )
    seed_month(
        store,
        "2026-09",
        bundles={"2026-09-01T1817Z": bundle},
        work=tmp_path / "seed",
    )
    drop_au_e10(checkout)

    result = rebuild(store, load_config(checkout), scope="month", value="2026-09", now=NOW)

    assert result.months == ["2026-09"]
    assert result.captures == 1
    assert result.resumed is False
    rows = _daily(store, "data-2026-09", "2026-09-01", tmp_path, "before.csv.gz")
    assert rows.height == 3
    e10 = rows.filter(pl.col("grade_raw") == "E10").row(0, named=True)
    assert e10["station_key"] == "AU-109"
    assert e10["price"] == 2.127
    assert e10["grade"] == "other"

    restore_config(checkout)
    rebuild(store, load_config(checkout), scope="month", value="2026-09", now=NOW)

    rows = _daily(store, "data-2026-09", "2026-09-01", tmp_path, "after.csv.gz")
    e10 = rows.filter(pl.col("grade_raw") == "E10").row(0, named=True)
    assert e10["grade"] == "regular"
    assert e10["price_local_per_litre"] == 2.127
    assert e10["currency"] == "AUD"
