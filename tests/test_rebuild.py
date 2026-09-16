"""Rebuild tests (§8.7). No network: every response comes from a stored bundle."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl
import pytest

from costco_gas.config import load_config
from costco_gas.rebuild import rebuild
from costco_gas.rollup import read_month_manifest
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


class RecordingStore:
    def __init__(self, inner):
        self.inner = inner
        self.log: list[str] = []

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            if name in ("replace_atomic", "upload_new"):
                self.log.append(f"{name}:{args[0]}:{args[2]}")
            elif name == "update_release":
                self.log.append(f"update_release:{args[0]}:prerelease={kwargs.get('prerelease')}")
            return attr(*args, **kwargs)

        return call


def test_rebuild_reopens_a_closed_month_before_rewriting_anything(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha-two")
    store = RecordingStore(open_store(f"local:{tmp_path / 'releases'}"))
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
        prerelease=False,
    )
    store.log.clear()

    rebuild(store, load_config(checkout), scope="month", value="2026-09", now=NOW)

    assert store.get_release("data-2026-09").prerelease is True
    reopened = store.log.index("update_release:data-2026-09:prerelease=True")
    first_write = min(i for i, entry in enumerate(store.log) if entry.startswith("replace_atomic:"))
    assert reopened < first_write


def test_rebuild_stamps_the_manifest_with_the_git_sha_and_config_shas(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha-three")
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

    rebuild(store, load_config(checkout), scope="month", value="2026-09", now=NOW)

    manifest = read_month_manifest(store, "data-2026-09")
    entry = manifest["captures"]["2026-09-01T1817Z"]
    assert entry["rebuild_git_sha"] == "sha-three"
    assert set(entry["config_sha256"]) == {
        "config/grades.csv",
        "config/station_links.csv",
        "config/countries.toml",
    }
    assert entry["rows_by_capture_date"] == {"2026-09-01": 3}
    assert entry["daily_files"]["2026-09-01"]["rows"] == 3


def _two_day_month(store, tmp_path, checkout):
    bundles = {
        "2026-09-01T1817Z": make_bundle(
            tmp_path / "bundles",
            capture_id="2026-09-01T1817Z",
            config_dir=checkout / "config",
            price_e10="$2.127",
        ),
        "2026-09-02T1817Z": make_bundle(
            tmp_path / "bundles",
            capture_id="2026-09-02T1817Z",
            config_dir=checkout / "config",
            price_e10="$2.187",
        ),
    }
    return seed_month(store, "2026-09", bundles=bundles, work=tmp_path / "seed")


def _put_state(store, tag, state, tmp_path):
    path = tmp_path / "rebuild-state.json"
    path.write_text(json.dumps(state, indent=1, sort_keys=True), "utf-8")
    store.replace_atomic(tag, path, "rebuild-state.json", "run-test")


def test_rebuild_resumes_after_the_last_completed_day_when_everything_matches(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GITHUB_SHA", "sha-four")
    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    _two_day_month(store, tmp_path, checkout)
    cfg = load_config(checkout)
    from costco_gas.rebuild import interp_config_sha256

    _put_state(
        store,
        "data-2026-09",
        {
            "schema_version": 1,
            "scope": "month",
            "value": "2026-09",
            "git_sha": "sha-four",
            "config_sha256": interp_config_sha256(cfg),
            "last_completed_day": "2026-09-01",
            "complete": False,
        },
        tmp_path,
    )

    result = rebuild(store, cfg, scope="month", value="2026-09", now=NOW)

    assert result.resumed is True
    assert result.captures == 1
    day_one = _daily(store, "data-2026-09", "2026-09-01", tmp_path, "one.csv.gz")
    assert day_one.height == 0
    day_two = _daily(store, "data-2026-09", "2026-09-02", tmp_path, "two.csv.gz")
    assert day_two.height == 3
    state = json.loads(
        store.download("data-2026-09", "rebuild-state.json", tmp_path / "state.json").read_text()
    )
    assert state["complete"] is True
    assert state["last_completed_day"] == "2026-09-02"


def test_rebuild_restarts_when_the_git_sha_differs(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha-new")
    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    _two_day_month(store, tmp_path, checkout)
    cfg = load_config(checkout)
    from costco_gas.rebuild import interp_config_sha256

    _put_state(
        store,
        "data-2026-09",
        {
            "schema_version": 1,
            "scope": "month",
            "value": "2026-09",
            "git_sha": "sha-old",
            "config_sha256": interp_config_sha256(cfg),
            "last_completed_day": "2026-09-01",
            "complete": False,
        },
        tmp_path,
    )

    result = rebuild(store, cfg, scope="month", value="2026-09", now=NOW)

    assert result.resumed is False
    assert result.captures == 2
    assert _daily(store, "data-2026-09", "2026-09-01", tmp_path, "r1.csv.gz").height == 3


def test_a_second_rebuild_after_a_grades_change_reprocesses_every_day(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha-five")
    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    _two_day_month(store, tmp_path, checkout)
    drop_au_e10(checkout)
    first = rebuild(store, load_config(checkout), scope="month", value="2026-09", now=NOW)
    assert first.captures == 2

    restore_config(checkout)
    second = rebuild(store, load_config(checkout), scope="month", value="2026-09", now=NOW)

    assert second.resumed is False
    assert second.captures == 2
    for day in ("2026-09-01", "2026-09-02"):
        rows = _daily(store, "data-2026-09", day, tmp_path, f"{day}.csv.gz")
        assert rows.filter(pl.col("grade_raw") == "E10")["grade"].to_list() == ["regular"]


def test_rebuild_of_an_open_month_then_refreshes_current(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha-six")
    from costco_gas.rollup import rebuild_current

    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    _two_day_month(store, tmp_path, checkout)
    cfg = load_config(checkout)
    rebuild(store, cfg, scope="month", value="2026-09", now=NOW)

    rebuild_current(store, cfg, now=NOW)

    stations = pl.read_csv(store.download("current", "stations.csv", tmp_path / "stations.csv"))
    assert "AU-109" in stations["station_key"].to_list()
    captures = pl.read_parquet(
        store.download("current", "costco-gas-all-captures.parquet", tmp_path / "all.parquet")
    )
    assert captures.height == 6
    fx = pl.read_csv(store.download("current", "fx.csv", tmp_path / "fx.csv"))
    assert fx["currency"].to_list() == ["AUD", "AUD"]
    manifest = json.loads(
        store.download("current", "manifest.json", tmp_path / "m.json").read_text()
    )
    assert manifest["newest_capture_by_country"] == {"AU": "2026-09-02T1817Z"}


def test_rebuild_interrupted_mid_month_lets_close_periods_close_cleanly_after_resume(
    tmp_path, monkeypatch
):
    """A crash between two days must never leave the manifest describing a day
    differently from the daily file `rebuild` already published for it: that
    disagreement used to make `close_periods`'s row-count cross-check raise and
    abort the whole close, including every other month in the same invocation
    (spec review, Task 16 round 2)."""
    monkeypatch.setenv("GITHUB_SHA", "sha-seven")
    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    _two_day_month(store, tmp_path, checkout)
    cfg = load_config(checkout)

    import costco_gas.rebuild as rebuild_module

    real_rebuild_capture = rebuild_module._rebuild_capture
    calls = {"n": 0}

    def flaky(store_, cfg_, tag, capture_id, work):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash mid-month")
        return real_rebuild_capture(store_, cfg_, tag, capture_id, work)

    monkeypatch.setattr(rebuild_module, "_rebuild_capture", flaky)

    with pytest.raises(RuntimeError, match="simulated crash mid-month"):
        rebuild(store, cfg, scope="month", value="2026-09", now=NOW)

    # Day one is already durable: its daily file and its manifest entry agree,
    # even though the "crash" happened while day two was only just starting.
    manifest = read_month_manifest(store, "data-2026-09")
    entry_one = manifest["captures"]["2026-09-01T1817Z"]
    assert entry_one["rows_by_capture_date"] == {"2026-09-01": 3}
    day_one = _daily(store, "data-2026-09", "2026-09-01", tmp_path, "mid.csv.gz")
    assert day_one.height == entry_one["daily_files"]["2026-09-01"]["rows"] == 3

    monkeypatch.setattr(rebuild_module, "_rebuild_capture", real_rebuild_capture)
    result = rebuild(store, cfg, scope="month", value="2026-09", now=NOW)
    assert result.resumed is True
    assert result.captures == 1

    from costco_gas.rollup import close_periods

    close_result = close_periods(store, cfg, now=datetime(2026, 11, 2, tzinfo=UTC))

    assert close_result.closed_months == ["2026-09"]
    assert close_result.blocked_months == []
