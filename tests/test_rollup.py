"""Roll-up tests. LocalReleaseStore only, no network."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import polars as pl
import pytest

from costco_gas.rollup import close_periods, daily_grain
from costco_gas.store import open_store
from helpers_rollup import (
    RecordingIssues,
    RecordingStore,
    price_row,
    rows_frame,
    seed_month,
    stub_config,
)

CURRENT_DATA_ASSET_NAMES = {
    "costco-gas-all.parquet",
    "costco-gas-all.csv.gz",
    "costco-gas-all-captures.parquet",
    "costco-gas-latest.csv",
    "stations.csv",
    "fx.csv",
}
CURRENT_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "closed_months",
    "closed_years",
    "newest_capture_by_country",
    "assets",
}


def test_daily_grain_keeps_the_last_capture_of_the_day():
    rows = rows_frame(
        [
            price_row(
                capture_id="2026-09-15T0017Z",
                station_key="US-1364",
                grade_raw="regular",
                grade="regular",
                price=3.899,
            ),
            price_row(
                capture_id="2026-09-15T1817Z",
                station_key="US-1364",
                grade_raw="regular",
                grade="regular",
                price=3.999,
            ),
            price_row(
                capture_id="2026-09-15T1817Z",
                station_key="US-1364",
                grade_raw="premium",
                grade="premium",
                price=4.629,
            ),
            price_row(
                capture_id="2026-09-16T0017Z",
                station_key="US-1364",
                grade_raw="regular",
                grade="regular",
                price=4.099,
            ),
        ]
    )

    out = daily_grain(rows)

    assert out.height == 3
    day15 = out.filter(
        (pl.col("capture_date") == date(2026, 9, 15)) & (pl.col("grade_raw") == "regular")
    ).row(0, named=True)
    assert day15["capture_id"] == "2026-09-15T1817Z"
    assert day15["price"] == 3.999
    assert day15["n_captures"] == 2
    assert day15["price_min"] == 3.899
    assert day15["price_max"] == 3.999
    premium = out.filter(pl.col("grade_raw") == "premium").row(0, named=True)
    assert premium["n_captures"] == 1
    assert premium["price_min"] == 4.629
    assert premium["price_max"] == 4.629
    day16 = out.filter(pl.col("capture_date") == date(2026, 9, 16)).row(0, named=True)
    assert day16["price"] == 4.099
    assert day16["n_captures"] == 1


def _us_rows(capture_id: str, price: float = 3.999) -> pl.DataFrame:
    return rows_frame(
        [
            price_row(
                capture_id=capture_id,
                station_key="US-1364",
                grade_raw="regular",
                grade="regular",
                price=price,
            ),
            price_row(
                capture_id=capture_id,
                station_key="US-1364",
                grade_raw="premium",
                grade="premium",
                price=price + 0.63,
            ),
        ]
    )


def _status(capture_id: str) -> dict:
    return {
        "schema_version": 1,
        "capture_id": capture_id,
        "countries": {"US": {"status": "ok", "rows": 2}},
    }


def test_close_periods_blocks_a_month_that_has_an_orphan_bundle(tmp_path):
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    seed_month(
        store,
        "2026-08",
        {"2026-08-31": _us_rows("2026-08-31T1817Z")},
        captures={
            "2026-08-31T1817Z": {
                "status": _status("2026-08-31T1817Z"),
                "rows_by_capture_date": {"2026-08-31": 2},
            }
        },
        orphan_bundles=["capture-2026-08-31T2317Z.tar.gz"],
        work=tmp_path / "seed-08",
    )
    issues = RecordingIssues()

    result = close_periods(store, cfg, now=datetime(2026, 9, 15, 18, 17, tzinfo=UTC), issues=issues)

    assert result.blocked_months == ["2026-08"]
    assert result.closed_months == []
    assert store.get_release("data-2026-08").prerelease is True
    assert issues.opened[0][0] == "Period close blocked: 2026-08"
    assert "capture-2026-08-31T2317Z.tar.gz" in issues.opened[0][1]


def _seed_august(store, tmp_path, *, fx=None):
    return seed_month(
        store,
        "2026-08",
        {
            "2026-08-30": _us_rows("2026-08-30T1817Z", 3.899),
            "2026-08-31": _us_rows("2026-08-31T1817Z", 3.999),
        },
        captures={
            "2026-08-30T1817Z": {
                "status": _status("2026-08-30T1817Z"),
                "rows_by_capture_date": {"2026-08-30": 2},
                "fx": fx or [],
            },
            "2026-08-31T1817Z": {
                "status": _status("2026-08-31T1817Z"),
                "rows_by_capture_date": {"2026-08-31": 2},
                "fx": fx or [],
            },
        },
        work=tmp_path / "seed-08",
    )


def test_month_close_writes_both_grains_and_clears_prerelease_last(tmp_path):
    store = RecordingStore(open_store(f"local:{tmp_path / 'releases'}"))
    cfg = stub_config(tmp_path)
    _seed_august(store, tmp_path)

    result = close_periods(
        store,
        cfg,
        now=datetime(2026, 9, 1, 3, 17, tzinfo=UTC),
        issues=RecordingIssues(),
    )

    assert result.closed_months == ["2026-08"]
    names = {a.name for a in store.list_assets("data-2026-08")}
    assert {
        "costco-gas-2026-08.parquet",
        "costco-gas-2026-08.csv.gz",
        "costco-gas-2026-08-captures.parquet",
    } <= names
    captures = pl.read_parquet(
        store.download(
            "data-2026-08",
            "costco-gas-2026-08-captures.parquet",
            tmp_path / "caps.parquet",
        )
    )
    assert captures.height == 4
    grain = pl.read_parquet(
        store.download("data-2026-08", "costco-gas-2026-08.parquet", tmp_path / "grain.parquet")
    )
    assert grain.height == 4
    assert set(grain["n_captures"].to_list()) == {1}
    assert store.get_release("data-2026-08").prerelease is False
    cleared = store.log.index("update_release:data-2026-08:prerelease=False")
    assert cleared > store.log.index(
        "replace_atomic:data-2026-08:costco-gas-2026-08-captures.parquet"
    )
    assert cleared > store.log.index("replace_atomic:current:manifest.json")


def test_month_close_is_safe_to_repeat(tmp_path):
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    _seed_august(store, tmp_path)
    now = datetime(2026, 9, 1, 3, 17, tzinfo=UTC)
    close_periods(store, cfg, now=now, issues=RecordingIssues())
    before = {a.name: a.size for a in store.list_assets("data-2026-08")}

    again = close_periods(store, cfg, now=now, issues=RecordingIssues())

    assert again.closed_months == []
    assert {a.name: a.size for a in store.list_assets("data-2026-08")} == before


def test_month_close_refuses_when_a_daily_file_disagrees_with_the_manifest(tmp_path):
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    manifest = _seed_august(store, tmp_path)
    manifest["captures"]["2026-08-31T1817Z"]["daily_files"]["2026-08-31"]["rows"] = 99
    path = tmp_path / "manifest-2026-08.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), "utf-8")
    store.replace_atomic("data-2026-08", path, "manifest-2026-08.json", "run-test")

    with pytest.raises(ValueError, match="manifest records 99"):
        close_periods(
            store,
            cfg,
            now=datetime(2026, 9, 1, 3, 17, tzinfo=UTC),
            issues=RecordingIssues(),
        )
