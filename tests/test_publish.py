"""Tests for costco_gas.publish (spec 8.4 recovery, 8.5 publish)."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from costco_gas import rollup, schema

REPO_ROOT = Path(__file__).resolve().parents[1]

# The 2026-09-15 Taiwan prices, per litre in TWD, with the Frankfurter rate of
# 31.709 TWD per USD from the same day.
TWD_PER_USD = 31.709
USD_PER_TWD = 1.0 / TWD_PER_USD


def price_row(capture_id, captured_at, station, grade_raw, grade, price):
    return {
        "capture_id": capture_id,
        "capture_date": captured_at.date(),
        "captured_at_utc": captured_at,
        "local_date": (captured_at + timedelta(hours=8)).date(),
        "country": "TW",
        "station_key": f"TW-{station}",
        "source_station_id": station,
        "source": "costco-occ",
        "name": station,
        "name_local": None,
        "address": None,
        "city": None,
        "region": "桃園市",
        "postcode": "320",
        "lat": 24.9573,
        "lon": 121.2196,
        "timezone": "Asia/Taipei",
        "grade_raw": grade_raw,
        "grade": grade,
        "price_raw": f"${price}",
        "price": price,
        "price_unit": "TWD/L",
        "currency": "TWD",
        "price_local_per_litre": price,
        "fx_usd_per_unit": USD_PER_TWD,
        "fx_rate_date": date(2026, 9, 14),
        "fx_source": "frankfurter-v2",
        "fx_fetched_at_utc": captured_at,
        "price_usd_per_litre": round(price * USD_PER_TWD, 4),
        "price_usd_per_gallon": round(price * USD_PER_TWD * 3.785411784, 4),
    }


def test_daily_grain_keeps_the_last_capture_of_the_day():
    early = datetime(2026, 9, 15, 0, 17, tzinfo=timezone.utc)
    late = datetime(2026, 9, 15, 18, 17, tzinfo=timezone.utc)
    rows = pl.DataFrame(
        [
            price_row("2026-09-15T0017Z", early, "Chungli", "95", "regular", 30.0),
            price_row("2026-09-15T1817Z", late, "Chungli", "95", "regular", 30.5),
            price_row("2026-09-15T1817Z", late, "Chungli", "98", "premium", 31.5),
        ],
        schema=schema.ROW_SCHEMA,
    )

    daily = rollup.daily_grain(rows)

    assert daily.height == 2
    regular = daily.filter(pl.col("grade_raw") == "95").to_dicts()[0]
    assert regular["capture_id"] == "2026-09-15T1817Z"
    assert regular["price"] == 30.5
    assert regular["n_captures"] == 2
    assert regular["price_min"] == 30.0
    assert regular["price_max"] == 30.5
    premium = daily.filter(pl.col("grade_raw") == "98").to_dicts()[0]
    assert premium["n_captures"] == 1
    assert premium["price_min"] == premium["price_max"] == 31.5
    assert list(daily.columns) == list(rollup.DAILY_SCHEMA)


def test_daily_grain_of_an_empty_frame_keeps_the_schema():
    empty = rollup.daily_grain(pl.DataFrame(schema=schema.ROW_SCHEMA))
    assert empty.height == 0
    assert dict(empty.schema) == dict(rollup.DAILY_SCHEMA)


from costco_gas.config import load_config
from costco_gas.publish import publish
from costco_gas.store import AssetNotFound, LocalReleaseStore

NOW = datetime(2026, 9, 15, 18, 20, tzinfo=timezone.utc)

STATION_COLUMNS = list(schema.STATION_SCHEMA)


@pytest.fixture(autouse=True)
def no_github_env(monkeypatch):
    """Keep the issue helper in its dry mode: these tests never touch the network."""
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)


def station_row(station, captured_at, grades, status="active"):
    return {
        "station_key": f"TW-{station}",
        "country": "TW",
        "source_station_id": station,
        "alt_id": "560",
        "name": station,
        "name_local": None,
        "address": None,
        "city": None,
        "region": "桃園市",
        "postcode": "320",
        "lat": 24.9573,
        "lon": 121.2196,
        "timezone": "Asia/Taipei",
        "grades_seen": "|".join(sorted(grades)),
        "first_seen_utc": captured_at,
        "last_seen_utc": captured_at,
        "status": status,
        "superseded_by": None,
    }


def make_capture_dir(
    root: Path, capture_id: str, captured_at: datetime, *, prices, tw_status="ok"
) -> Path:
    """Write a capture directory shaped like `costco-gas capture --out` does."""
    directory = root / capture_id
    directory.mkdir(parents=True, exist_ok=True)
    rows = pl.DataFrame(
        [
            price_row(capture_id, captured_at, station, grade_raw, grade, price)
            for station, grade_raw, grade, price in prices
        ],
        schema=schema.ROW_SCHEMA,
    )
    schema.write_rows_csv_gz(rows, directory / "rows.csv.gz")
    by_station: dict[str, set[str]] = {}
    for station, grade_raw, _, _ in prices:
        by_station.setdefault(station, set()).add(grade_raw)
    pl.DataFrame(
        [station_row(s, captured_at, g) for s, g in sorted(by_station.items())],
        schema=schema.STATION_SCHEMA,
    ).write_csv(directory / "stations.csv")
    (directory / "fx.json").write_text(
        json.dumps(
            [
                {
                    "currency": "TWD",
                    "units_per_usd": TWD_PER_USD,
                    "fx_rate_date": "2026-09-14",
                    "fx_source": "frankfurter-v2",
                    "fx_fetched_at_utc": captured_at.strftime("%Y-%m-%dT%H:%M:%SZ"),
                }
            ],
            indent=1,
        ),
        encoding="utf-8",
    )
    status = {
        "schema_version": 1,
        "capture_id": capture_id,
        "countries": {
            "TW": {"status": tw_status, "stations": len(by_station), "rows": rows.height},
            "JP": {"status": "failed", "stations": 0, "rows": 0},
        },
    }
    (directory / "status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    (directory / "capture.json").write_text(
        json.dumps({"capture_id": capture_id, "capture_date": captured_at.date().isoformat()}),
        encoding="utf-8",
    )
    with tarfile.open(directory / "bundle.tar.gz", "w:gz") as tar:
        for name in ("capture.json", "rows.csv.gz", "stations.csv", "fx.json", "status.json"):
            tar.add(directory / name, arcname=name)
    return directory


@pytest.fixture()
def cfg(tmp_path: Path):
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    return load_config(tmp_path)


def asset_digests(store, tag, scratch: Path) -> dict[str, str]:
    out = {}
    for asset in store.list_assets(tag):
        dest = scratch / f"{tag}--{asset.name}"
        store.download(tag, asset.name, dest)
        out[asset.name] = hashlib.sha256(dest.read_bytes()).hexdigest()
    return out


DAY1 = datetime(2026, 9, 15, 18, 17, tzinfo=timezone.utc)
PRICES = [
    ("Chungli", "Diesel", "diesel", 28.6),
    ("Chungli", "95", "regular", 30.0),
    ("Chungli", "98", "premium", 31.5),
    ("Xinzhuang", "95", "regular", 30.0),
    ("Xinzhuang", "98", "premium", 31.5),
]


def test_first_publish_creates_current_and_the_month_release(tmp_path: Path, cfg):
    store = LocalReleaseStore(tmp_path / "releases")
    capture_dir = make_capture_dir(tmp_path / "captures", "2026-09-15T1817Z", DAY1, prices=PRICES)

    result = publish(store, capture_dir, cfg, now=NOW)

    assert result.capture_id == "2026-09-15T1817Z"
    month = {a.name for a in store.list_assets("data-2026-09")}
    assert month == {
        "costco-gas-2026-09-15.csv.gz",
        "capture-2026-09-15T1817Z.tar.gz",
        "manifest-2026-09.json",
    }
    current = {a.name for a in store.list_assets("current")}
    assert current == {
        "costco-gas-all.parquet",
        "costco-gas-all.csv.gz",
        "costco-gas-all-captures.parquet",
        "costco-gas-latest.csv",
        "stations.csv",
        "fx.csv",
        "manifest.json",
    }
    assert store.get_release("data-2026-09").prerelease is True
    assert store.get_release("current").prerelease is False

    daily = store.download("data-2026-09", "costco-gas-2026-09-15.csv.gz", tmp_path / "d.csv.gz")
    assert schema.read_rows_csv_gz(daily).height == 5

    manifest = json.loads(
        store.download("data-2026-09", "manifest-2026-09.json", tmp_path / "m.json").read_text()
    )
    entry = manifest["captures"]["2026-09-15T1817Z"]
    assert entry["rows_by_capture_date"] == {"2026-09-15": 5}
    assert entry["daily_files"]["2026-09-15"]["rows"] == 5
    assert [row["currency"] for row in entry["fx"]] == ["TWD"]

    latest = pl.read_csv(
        store.download("current", "costco-gas-latest.csv", tmp_path / "l.csv"),
        schema=schema.ROW_SCHEMA,
    )
    assert latest.height == 5
    fx = pl.read_csv(
        store.download("current", "fx.csv", tmp_path / "fx.csv"), schema=schema.FX_SCHEMA
    )
    assert fx.to_dicts()[0]["units_per_usd"] == TWD_PER_USD
    current_manifest = json.loads(
        store.download("current", "manifest.json", tmp_path / "cm.json").read_text()
    )
    assert current_manifest["status"]["capture_id"] == "2026-09-15T1817Z"
    assert current_manifest["newest_capture_by_country"]["TW"] == "2026-09-15T1817Z"
    # JP failed, so it must not claim this capture as its newest.
    assert "JP" not in current_manifest["newest_capture_by_country"]


def test_republishing_the_same_directory_changes_no_asset_content(tmp_path: Path, cfg):
    store = LocalReleaseStore(tmp_path / "releases")
    capture_dir = make_capture_dir(tmp_path / "captures", "2026-09-15T1817Z", DAY1, prices=PRICES)

    publish(store, capture_dir, cfg, now=NOW)
    before_month = asset_digests(store, "data-2026-09", tmp_path / "a")
    before_current = asset_digests(store, "current", tmp_path / "a")

    result = publish(store, capture_dir, cfg, now=NOW)

    assert result.warnings == []
    assert asset_digests(store, "data-2026-09", tmp_path / "b") == before_month
    assert asset_digests(store, "current", tmp_path / "b") == before_current


def test_a_second_capture_the_same_day_keeps_the_first(tmp_path: Path, cfg):
    store = LocalReleaseStore(tmp_path / "releases")
    first = make_capture_dir(
        tmp_path / "captures",
        "2026-09-15T0017Z",
        datetime(2026, 9, 15, 0, 17, tzinfo=timezone.utc),
        prices=PRICES,
    )
    second = make_capture_dir(
        tmp_path / "captures",
        "2026-09-15T1817Z",
        DAY1,
        prices=[("Chungli", "95", "regular", 30.5)],
    )

    publish(store, first, cfg, now=NOW)
    publish(store, second, cfg, now=NOW)

    daily = schema.read_rows_csv_gz(
        store.download("data-2026-09", "costco-gas-2026-09-15.csv.gz", tmp_path / "d.csv.gz")
    )
    assert sorted(daily["capture_id"].unique().to_list()) == [
        "2026-09-15T0017Z",
        "2026-09-15T1817Z",
    ]
    assert daily.height == 6

    manifest = json.loads(
        store.download("data-2026-09", "manifest-2026-09.json", tmp_path / "m.json").read_text()
    )
    assert set(manifest["captures"]) == {"2026-09-15T0017Z", "2026-09-15T1817Z"}

    # latest: Chungli 95 moves to the newer capture, the other rows are untouched.
    latest = pl.read_csv(
        store.download("current", "costco-gas-latest.csv", tmp_path / "l.csv"),
        schema=schema.ROW_SCHEMA,
    )
    chungli = latest.filter(pl.col("station_key") == "TW-Chungli")
    assert chungli["grade_raw"].to_list() == ["95"]
    assert chungli["price"].to_list() == [30.5]
    xin = latest.filter(pl.col("station_key") == "TW-Xinzhuang")
    assert xin.height == 2
    assert xin["capture_id"].unique().to_list() == ["2026-09-15T0017Z"]

    daily_all = pl.read_parquet(
        store.download("current", "costco-gas-all.parquet", tmp_path / "all.parquet")
    )
    regular = daily_all.filter(
        (pl.col("station_key") == "TW-Chungli") & (pl.col("grade_raw") == "95")
    ).to_dicts()[0]
    assert regular["n_captures"] == 2
    assert regular["price_min"] == 30.0
    assert regular["price_max"] == 30.5


def test_an_older_capture_does_not_move_the_stored_status(tmp_path: Path, cfg):
    store = LocalReleaseStore(tmp_path / "releases")
    newer = make_capture_dir(tmp_path / "captures", "2026-09-15T1817Z", DAY1, prices=PRICES)
    older = make_capture_dir(
        tmp_path / "captures",
        "2026-09-15T0017Z",
        datetime(2026, 9, 15, 0, 17, tzinfo=timezone.utc),
        prices=[("Chungli", "95", "regular", 29.0)],
    )

    publish(store, newer, cfg, now=NOW)
    publish(store, older, cfg, now=NOW)

    manifest = json.loads(
        store.download("current", "manifest.json", tmp_path / "cm.json").read_text()
    )
    assert manifest["status"]["capture_id"] == "2026-09-15T1817Z"
    assert manifest["newest_capture_by_country"]["TW"] == "2026-09-15T1817Z"
    stations = pl.read_csv(
        store.download("current", "stations.csv", tmp_path / "s.csv"),
        schema=schema.STATION_SCHEMA,
    )
    # Xinzhuang has no rows in the older capture, but its status must not flip.
    assert set(stations["status"].to_list()) == {"active"}


DAY2 = datetime(2026, 9, 16, 0, 17, tzinfo=timezone.utc)


def test_two_crashes_rebuild_the_manifest_and_reconcile_the_orphan(tmp_path: Path, cfg):
    store = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"

    # Capture A published normally.
    a_dir = make_capture_dir(
        captures,
        "2026-09-15T0017Z",
        datetime(2026, 9, 15, 0, 17, tzinfo=timezone.utc),
        prices=PRICES,
    )
    publish(store, a_dir, cfg, now=NOW)

    # Crash 1: the manifest upload never landed.
    manifest_asset = next(
        a for a in store.list_assets("data-2026-09") if a.name == "manifest-2026-09.json"
    )
    store.delete("data-2026-09", manifest_asset.id)

    # Crash 2: capture B uploaded its bundle, then died before the daily file.
    b_dir = make_capture_dir(
        captures,
        "2026-09-15T1217Z",
        datetime(2026, 9, 15, 12, 17, tzinfo=timezone.utc),
        prices=[("Chungli", "95", "regular", 30.2)],
    )
    store.upload_new("data-2026-09", b_dir / "bundle.tar.gz", "capture-2026-09-15T1217Z.tar.gz")

    # The next capture, on a new date.
    c_dir = make_capture_dir(
        captures, "2026-09-16T0017Z", DAY2, prices=[("Chungli", "95", "regular", 30.9)]
    )
    result = publish(store, c_dir, cfg, now=NOW)

    assert "manifest_rebuilt" in result.warnings
    assert "reconciled:2026-09-15T1217Z" in result.warnings

    manifest = json.loads(
        store.download("data-2026-09", "manifest-2026-09.json", tmp_path / "m.json").read_text()
    )
    assert set(manifest["captures"]) == {
        "2026-09-15T0017Z",
        "2026-09-15T1217Z",
        "2026-09-16T0017Z",
    }

    day15 = schema.read_rows_csv_gz(
        store.download("data-2026-09", "costco-gas-2026-09-15.csv.gz", tmp_path / "d15.csv.gz")
    )
    assert sorted(day15["capture_id"].unique().to_list()) == [
        "2026-09-15T0017Z",
        "2026-09-15T1217Z",
    ]
    day16 = schema.read_rows_csv_gz(
        store.download("data-2026-09", "costco-gas-2026-09-16.csv.gz", tmp_path / "d16.csv.gz")
    )
    assert day16["capture_id"].unique().to_list() == ["2026-09-16T0017Z"]

    captures_all = pl.read_parquet(
        store.download("current", "costco-gas-all-captures.parquet", tmp_path / "ac.parquet")
    )
    assert sorted(captures_all["capture_id"].unique().to_list()) == [
        "2026-09-15T0017Z",
        "2026-09-15T1217Z",
        "2026-09-16T0017Z",
    ]
    fx = pl.read_csv(
        store.download("current", "fx.csv", tmp_path / "fx.csv"), schema=schema.FX_SCHEMA
    )
    assert sorted(fx["capture_id"].to_list()) == [
        "2026-09-15T0017Z",
        "2026-09-15T1217Z",
        "2026-09-16T0017Z",
    ]
    # Reconciling the orphan must not resurrect it as the newest status.
    current_manifest = json.loads(
        store.download("current", "manifest.json", tmp_path / "cm.json").read_text()
    )
    assert current_manifest["status"]["capture_id"] == "2026-09-16T0017Z"
    assert current_manifest["newest_capture_by_country"]["TW"] == "2026-09-16T0017Z"


def test_a_missing_daily_file_that_the_manifest_records_raises(tmp_path: Path, cfg):
    from costco_gas.store import StorageError

    store = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"
    a_dir = make_capture_dir(
        captures,
        "2026-09-15T0017Z",
        datetime(2026, 9, 15, 0, 17, tzinfo=timezone.utc),
        prices=PRICES,
    )
    publish(store, a_dir, cfg, now=NOW)

    daily = next(
        a for a in store.list_assets("data-2026-09") if a.name == "costco-gas-2026-09-15.csv.gz"
    )
    store.delete("data-2026-09", daily.id)

    b_dir = make_capture_dir(
        captures, "2026-09-15T1817Z", DAY1, prices=[("Chungli", "95", "regular", 30.2)]
    )
    with pytest.raises(StorageError, match="costco-gas-2026-09-15.csv.gz is missing"):
        publish(store, b_dir, cfg, now=NOW)


from dataclasses import replace as dataclass_replace


class StateOverrideStore:
    """LocalReleaseStore with one asset forced to a non-uploaded state."""

    def __init__(self, inner, asset_name: str, state: str):
        self._inner = inner
        self._asset_name = asset_name
        self._state = state

    def list_assets(self, tag):
        return [
            dataclass_replace(a, state=self._state) if a.name == self._asset_name else a
            for a in self._inner.list_assets(tag)
        ]

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_an_incomplete_orphan_bundle_is_deleted(tmp_path: Path, cfg):
    inner = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"
    a_dir = make_capture_dir(
        captures,
        "2026-09-15T0017Z",
        datetime(2026, 9, 15, 0, 17, tzinfo=timezone.utc),
        prices=PRICES,
    )
    publish(inner, a_dir, cfg, now=NOW)

    b_dir = make_capture_dir(
        captures,
        "2026-09-15T1217Z",
        datetime(2026, 9, 15, 12, 17, tzinfo=timezone.utc),
        prices=[("Chungli", "95", "regular", 30.2)],
    )
    inner.upload_new("data-2026-09", b_dir / "bundle.tar.gz", "capture-2026-09-15T1217Z.tar.gz")
    store = StateOverrideStore(inner, "capture-2026-09-15T1217Z.tar.gz", "starter")

    c_dir = make_capture_dir(
        captures, "2026-09-16T0017Z", DAY2, prices=[("Chungli", "95", "regular", 30.9)]
    )
    result = publish(store, c_dir, cfg, now=NOW)

    assert "discarded_incomplete_bundle:2026-09-15T1217Z" in result.warnings
    assert "capture-2026-09-15T1217Z.tar.gz" not in {
        a.name for a in inner.list_assets("data-2026-09")
    }
    day15 = schema.read_rows_csv_gz(
        inner.download("data-2026-09", "costco-gas-2026-09-15.csv.gz", tmp_path / "d15.csv.gz")
    )
    assert day15["capture_id"].unique().to_list() == ["2026-09-15T0017Z"]


def test_an_invalid_orphan_bundle_is_renamed_and_reported(tmp_path: Path, cfg, capsys):
    store = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"
    a_dir = make_capture_dir(
        captures,
        "2026-09-15T0017Z",
        datetime(2026, 9, 15, 0, 17, tzinfo=timezone.utc),
        prices=PRICES,
    )
    publish(store, a_dir, cfg, now=NOW)

    broken = tmp_path / "broken.tar.gz"
    broken.write_bytes(b"not a tar archive at all")
    store.upload_new("data-2026-09", broken, "capture-2026-09-15T1217Z.tar.gz")

    c_dir = make_capture_dir(
        captures, "2026-09-16T0017Z", DAY2, prices=[("Chungli", "95", "regular", 30.9)]
    )
    result = publish(store, c_dir, cfg, now=NOW)

    names = {a.name for a in store.list_assets("data-2026-09")}
    assert "capture-2026-09-15T1217Z.tar.gz.invalid" in names
    assert "reconciled:2026-09-15T1217Z" not in result.warnings
    # The no_github_env fixture clears GITHUB_REPOSITORY/GITHUB_TOKEN, so the
    # issue helper stays in its dry mode and only prints. No network call.
    assert "Invalid capture bundle: 2026-09-15T1217Z" in capsys.readouterr().out


def test_a_capture_id_collision_fails(tmp_path: Path, cfg):
    from costco_gas.store import StorageError

    store = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"
    a_dir = make_capture_dir(captures, "2026-09-15T1817Z", DAY1, prices=PRICES)
    publish(store, a_dir, cfg, now=NOW)

    different = make_capture_dir(
        tmp_path / "other",
        "2026-09-15T1817Z",
        DAY1,
        prices=[("Chungli", "95", "regular", 99.0)],
    )
    with pytest.raises(StorageError, match="capture_id_collision"):
        publish(store, different, cfg, now=NOW)
