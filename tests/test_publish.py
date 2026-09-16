"""Tests for costco_gas.publish (spec 8.4 recovery, 8.5 publish)."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import tarfile
from dataclasses import replace as dataclass_replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from costco_gas import rollup, schema
from costco_gas.config import load_config
from costco_gas.publish import publish
from costco_gas.store import LocalReleaseStore, StorageError, sha256_file

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
    early = datetime(2026, 9, 15, 0, 17, tzinfo=UTC)
    late = datetime(2026, 9, 15, 18, 17, tzinfo=UTC)
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


NOW = datetime(2026, 9, 15, 18, 20, tzinfo=UTC)

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


DAY1 = datetime(2026, 9, 15, 18, 17, tzinfo=UTC)
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
        datetime(2026, 9, 15, 0, 17, tzinfo=UTC),
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
        datetime(2026, 9, 15, 0, 17, tzinfo=UTC),
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


DAY2 = datetime(2026, 9, 16, 0, 17, tzinfo=UTC)


def test_two_crashes_rebuild_the_manifest_and_reconcile_the_orphan(tmp_path: Path, cfg):
    store = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"

    # Capture A published normally.
    a_dir = make_capture_dir(
        captures,
        "2026-09-15T0017Z",
        datetime(2026, 9, 15, 0, 17, tzinfo=UTC),
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
        datetime(2026, 9, 15, 12, 17, tzinfo=UTC),
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
    store = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"
    a_dir = make_capture_dir(
        captures,
        "2026-09-15T0017Z",
        datetime(2026, 9, 15, 0, 17, tzinfo=UTC),
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
    with pytest.raises(StorageError, match=re.escape("costco-gas-2026-09-15.csv.gz is missing")):
        publish(store, b_dir, cfg, now=NOW)


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
        datetime(2026, 9, 15, 0, 17, tzinfo=UTC),
        prices=PRICES,
    )
    publish(inner, a_dir, cfg, now=NOW)

    b_dir = make_capture_dir(
        captures,
        "2026-09-15T1217Z",
        datetime(2026, 9, 15, 12, 17, tzinfo=UTC),
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
        datetime(2026, 9, 15, 0, 17, tzinfo=UTC),
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


def test_a_late_publish_reopens_a_closed_month(tmp_path: Path, cfg):
    store = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"
    a_dir = make_capture_dir(
        captures,
        "2026-09-15T0017Z",
        datetime(2026, 9, 15, 0, 17, tzinfo=UTC),
        prices=PRICES,
    )
    publish(store, a_dir, cfg, now=NOW)
    store.update_release("data-2026-09", prerelease=False)

    b_dir = make_capture_dir(
        captures, "2026-09-15T1817Z", DAY1, prices=[("Chungli", "95", "regular", 30.2)]
    )
    publish(store, b_dir, cfg, now=NOW)

    assert store.get_release("data-2026-09").prerelease is True


def test_recovery_promotes_a_verified_next_asset(tmp_path: Path, cfg):
    from costco_gas.store import recover_temporaries

    store = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"
    a_dir = make_capture_dir(
        captures,
        "2026-09-15T0017Z",
        datetime(2026, 9, 15, 0, 17, tzinfo=UTC),
        prices=PRICES,
    )
    publish(store, a_dir, cfg, now=NOW)

    # Simulate a crash between steps 3 and 4 of replace_atomic: <name> was renamed
    # to .old-<token> and the verified upload is still called .next-<token>-1.
    original = store.download("current", "stations.csv", tmp_path / "stations.csv")
    digest = sha256_file(original)
    asset = next(a for a in store.list_assets("current") if a.name == "stations.csv")
    store.rename("current", asset.id, "stations.csv.old-2026-09-15T1817Z")
    store.upload_new(
        "current",
        original,
        "stations.csv.next-2026-09-15T1817Z-1",
        label=f"sha256:{digest}",
    )

    actions = recover_temporaries(store, "current")

    assert actions, "store.py reports what recovery did"
    names = {a.name for a in store.list_assets("current")}
    assert "stations.csv" in names
    assert not any(".next-" in n or ".old-" in n for n in names)
    promoted = next(a for a in store.list_assets("current") if a.name == "stations.csv")
    assert not promoted.label
    restored = store.download("current", "stations.csv", tmp_path / "promoted.csv")
    assert restored.read_bytes() == original.read_bytes()


def test_recovery_rolls_back_to_old_when_no_next_verifies(tmp_path: Path, cfg):
    from costco_gas.store import recover_temporaries

    store = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"
    a_dir = make_capture_dir(
        captures,
        "2026-09-15T0017Z",
        datetime(2026, 9, 15, 0, 17, tzinfo=UTC),
        prices=PRICES,
    )
    publish(store, a_dir, cfg, now=NOW)

    original = store.download("current", "fx.csv", tmp_path / "fx.csv")
    asset = next(a for a in store.list_assets("current") if a.name == "fx.csv")
    store.rename("current", asset.id, "fx.csv.old-2026-09-15T1817Z")
    corrupt = tmp_path / "corrupt.csv"
    corrupt.write_text("garbage\n", encoding="utf-8")
    store.upload_new(
        "current", corrupt, "fx.csv.next-2026-09-15T1817Z-1", label="sha256:" + "0" * 64
    )

    assert recover_temporaries(store, "current")
    restored = store.download("current", "fx.csv", tmp_path / "fx-restored.csv")
    assert restored.read_bytes() == original.read_bytes()
    assert not any(".next-" in a.name or ".old-" in a.name for a in store.list_assets("current"))


# -- spec review round 1 findings -----------------------------------------------


class CrashOnceStore:
    """LocalReleaseStore that raises once, simulating a process crash right

    before one specific `replace_atomic` call -- nothing about that call
    reaches the underlying store, so no temporary asset is left behind for
    Step 0 recovery to clean up. That is the point: this reproduces a crash
    inside `_update_current` itself, not an interrupted `replace_atomic`.
    """

    def __init__(self, inner, tag: str, name: str):
        self._inner = inner
        self._tag = tag
        self._name = name
        self._armed = True

    def replace_atomic(self, tag, path, name, token):
        if self._armed and tag == self._tag and name == self._name:
            self._armed = False
            raise RuntimeError("simulated crash")
        return self._inner.replace_atomic(tag, path, name, token)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_a_crash_inside_update_current_leaves_the_capture_reconcilable(tmp_path: Path, cfg):
    """Finding 1: a crash between `current` and the month-manifest entry must

    not orphan the capture forever. Before the fix, the manifest entry was
    written before `_update_current`, so a crash here left the capture
    "done" in the manifest while `current` never saw it, and `_reconcile`
    would skip its bundle on every later run because
    `capture_id in manifest["captures"]` was already true.
    """
    inner = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"

    a_dir = make_capture_dir(
        captures, "2026-09-15T0017Z", datetime(2026, 9, 15, 0, 17, tzinfo=UTC), prices=PRICES
    )
    publish(inner, a_dir, cfg, now=NOW)

    # Capture B crashes partway through updating `current`: several of the six
    # data assets land, `fx.csv` (uploaded last, before manifest.json) never does.
    crashy = CrashOnceStore(inner, "current", "fx.csv")
    b_dir = make_capture_dir(
        captures,
        "2026-09-15T1217Z",
        datetime(2026, 9, 15, 12, 17, tzinfo=UTC),
        prices=[("Chungli", "95", "regular", 30.2)],
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        publish(crashy, b_dir, cfg, now=NOW)

    # The daily file for 2026-09-15 already has B's rows (merge_capture writes
    # it before calling `_update_current`), but the month manifest must not
    # have recorded B as done -- that is the commit marker `_reconcile` reads.
    day15 = schema.read_rows_csv_gz(
        inner.download(
            "data-2026-09", "costco-gas-2026-09-15.csv.gz", tmp_path / "d15-after-crash.csv.gz"
        )
    )
    assert "2026-09-15T1217Z" in day15["capture_id"].unique().to_list()
    manifest = json.loads(
        inner.download(
            "data-2026-09", "manifest-2026-09.json", tmp_path / "m-after-crash.json"
        ).read_text()
    )
    assert "2026-09-15T1217Z" not in manifest["captures"]

    # A later capture, on a new date, must reconcile B rather than skip it.
    c_dir = make_capture_dir(
        captures, "2026-09-16T0017Z", DAY2, prices=[("Chungli", "95", "regular", 30.9)]
    )
    result = publish(inner, c_dir, cfg, now=NOW)

    assert "reconciled:2026-09-15T1217Z" in result.warnings
    captures_all = pl.read_parquet(
        inner.download("current", "costco-gas-all-captures.parquet", tmp_path / "ac.parquet")
    )
    assert sorted(captures_all["capture_id"].unique().to_list()) == [
        "2026-09-15T0017Z",
        "2026-09-15T1217Z",
        "2026-09-16T0017Z",
    ]
    fx = pl.read_csv(
        inner.download("current", "fx.csv", tmp_path / "fx-final.csv"), schema=schema.FX_SCHEMA
    )
    assert sorted(fx["capture_id"].to_list()) == [
        "2026-09-15T0017Z",
        "2026-09-15T1217Z",
        "2026-09-16T0017Z",
    ]


class FlakyDownloadStore:
    """LocalReleaseStore whose download() raises for one (tag, name), always."""

    def __init__(self, inner, tag: str, asset_name: str):
        self._inner = inner
        self._tag = tag
        self._asset_name = asset_name
        self.attempts = 0

    def download(self, tag, name, dest):
        if tag == self._tag and name == self._asset_name:
            self.attempts += 1
            raise StorageError("simulated transient read failure")
        return self._inner.download(tag, name, dest)

    def __getattr__(self, name):
        return getattr(self._inner, name)


def test_a_transient_orphan_download_failure_leaves_the_bundle_in_place(tmp_path: Path, cfg):
    """Finding 2: a StorageError from `download()` means "could not read it this

    run", not "the bundle is bad" -- it must not be renamed to `.invalid` or
    reported as an issue, only retried on a later run.
    """
    inner = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"
    a_dir = make_capture_dir(
        captures, "2026-09-15T0017Z", datetime(2026, 9, 15, 0, 17, tzinfo=UTC), prices=PRICES
    )
    publish(inner, a_dir, cfg, now=NOW)

    b_dir = make_capture_dir(
        captures,
        "2026-09-15T1217Z",
        datetime(2026, 9, 15, 12, 17, tzinfo=UTC),
        prices=[("Chungli", "95", "regular", 30.2)],
    )
    inner.upload_new("data-2026-09", b_dir / "bundle.tar.gz", "capture-2026-09-15T1217Z.tar.gz")

    flaky = FlakyDownloadStore(inner, "data-2026-09", "capture-2026-09-15T1217Z.tar.gz")
    c_dir = make_capture_dir(
        captures, "2026-09-16T0017Z", DAY2, prices=[("Chungli", "95", "regular", 30.9)]
    )
    result = publish(flaky, c_dir, cfg, now=NOW)

    assert flaky.attempts == 1
    names = {a.name for a in inner.list_assets("data-2026-09")}
    assert "capture-2026-09-15T1217Z.tar.gz" in names
    assert "capture-2026-09-15T1217Z.tar.gz.invalid" not in names
    assert "reconciled:2026-09-15T1217Z" not in result.warnings
    assert "orphan_bundle_download_failed:2026-09-15T1217Z" in result.warnings

    # A later, un-flaky publish retries and reconciles it.
    d_dir = make_capture_dir(
        captures,
        "2026-09-17T0017Z",
        datetime(2026, 9, 17, 0, 17, tzinfo=UTC),
        prices=[("Chungli", "95", "regular", 31.0)],
    )
    result2 = publish(inner, d_dir, cfg, now=NOW)
    assert "reconciled:2026-09-15T1217Z" in result2.warnings


def test_current_missing_only_manifest_json_raises(tmp_path: Path, cfg):
    """Finding 3: first-publish means the whole of `current` is absent. A lone

    missing manifest.json alongside real data in the other six assets is
    corruption and must raise, not silently rebuild everything from empty.
    """
    store = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"
    a_dir = make_capture_dir(
        captures, "2026-09-15T0017Z", datetime(2026, 9, 15, 0, 17, tzinfo=UTC), prices=PRICES
    )
    publish(store, a_dir, cfg, now=NOW)

    manifest_asset = next(a for a in store.list_assets("current") if a.name == "manifest.json")
    store.delete("current", manifest_asset.id)

    b_dir = make_capture_dir(
        captures, "2026-09-16T0017Z", DAY2, prices=[("Chungli", "95", "regular", 30.9)]
    )
    with pytest.raises(StorageError, match="current is incomplete"):
        publish(store, b_dir, cfg, now=NOW)

    # Nothing about `current` was touched by the failed attempt.
    captures_all = pl.read_parquet(
        store.download("current", "costco-gas-all-captures.parquet", tmp_path / "ac.parquet")
    )
    assert captures_all.height == 5


def test_publish_recovers_an_interrupted_replace_before_merging(tmp_path: Path, cfg):
    """Finding 4a: publish() must run Step 0 recovery itself, not rely on a

    caller to have run it already. Deleting the `recovery_tags`/
    `recover_temporaries` loop in `publish()` leaves this test failing with an
    uncaught StorageError, because `_update_current` would otherwise try to
    read `stations.csv` while it is mid-replace (renamed to `.old-...`, with
    a verified `.next-...` still pending promotion).
    """
    store = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"
    a_dir = make_capture_dir(
        captures, "2026-09-15T0017Z", datetime(2026, 9, 15, 0, 17, tzinfo=UTC), prices=PRICES
    )
    publish(store, a_dir, cfg, now=NOW)

    # Simulate a crash between steps 3 and 4 of replace_atomic on `current`,
    # exactly as in test_recovery_promotes_a_verified_next_asset, but this
    # time leave it for publish() itself to find and fix.
    original = store.download("current", "stations.csv", tmp_path / "stations.csv")
    digest = sha256_file(original)
    asset = next(a for a in store.list_assets("current") if a.name == "stations.csv")
    store.rename("current", asset.id, "stations.csv.old-2026-09-15T1817Z")
    store.upload_new(
        "current",
        original,
        "stations.csv.next-2026-09-15T1817Z-1",
        label=f"sha256:{digest}",
    )

    b_dir = make_capture_dir(
        captures, "2026-09-16T0017Z", DAY2, prices=[("Chungli", "95", "regular", 30.9)]
    )
    publish(store, b_dir, cfg, now=NOW)

    names = {a.name for a in store.list_assets("current")}
    assert "stations.csv" in names
    assert not any(".next-" in n or ".old-" in n for n in names)


def test_a_row_count_mismatch_against_the_manifest_raises(tmp_path: Path, cfg):
    """Finding 4b: the recorded-row-count cross-check in merge_capture must

    actually run. Deleting it leaves this test failing, because publishing B
    would otherwise silently accept a daily file whose row count for A no
    longer matches what the month manifest records for A.
    """
    store = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"
    a_dir = make_capture_dir(
        captures, "2026-09-15T0017Z", datetime(2026, 9, 15, 0, 17, tzinfo=UTC), prices=PRICES
    )
    publish(store, a_dir, cfg, now=NOW)

    # Corrupt the daily file directly: drop one of A's rows without touching
    # the manifest, so its recorded row count for A no longer matches.
    daily_asset = next(
        a for a in store.list_assets("data-2026-09") if a.name == "costco-gas-2026-09-15.csv.gz"
    )
    local = store.download(
        "data-2026-09", "costco-gas-2026-09-15.csv.gz", tmp_path / "corrupt.csv.gz"
    )
    truncated = schema.read_rows_csv_gz(local).head(4)
    corrupted_path = tmp_path / "truncated.csv.gz"
    schema.write_rows_csv_gz(truncated, corrupted_path)
    store.delete("data-2026-09", daily_asset.id)
    store.upload_new("data-2026-09", corrupted_path, "costco-gas-2026-09-15.csv.gz")

    b_dir = make_capture_dir(
        captures, "2026-09-15T1817Z", DAY1, prices=[("Chungli", "95", "regular", 30.2)]
    )
    with pytest.raises(StorageError, match="the manifest records"):
        publish(store, b_dir, cfg, now=NOW)


def test_a_months_first_crash_is_not_fabricated_done_by_the_rebuild(tmp_path: Path, cfg):
    """Spec review round 2: read_or_rebuild_manifest must not fabricate a "done"

    entry for a capture whose rows never reached `current`, even though its
    rows are in the daily file and its bundle is uploaded. Being in the daily
    file only means merge_capture wrote that file before crashing inside
    `_update_current` (Finding 1's fix moved the manifest-entry write after
    `_update_current`, so when a capture is the FIRST of a brand new month,
    that crash means `manifest-YYYY-MM.json` was never created at all, and
    the rebuild path -- untouched by round 1's fix -- is what runs next).
    """
    inner = LocalReleaseStore(tmp_path / "releases")
    captures = tmp_path / "captures"

    # Establish `current` from an earlier month, so this exercises the
    # "current already has history" branch, not the bootstrap one.
    zero_dir = make_capture_dir(
        captures, "2026-08-15T0017Z", datetime(2026, 8, 15, 0, 17, tzinfo=UTC), prices=PRICES
    )
    publish(inner, zero_dir, cfg, now=NOW)

    # September's first capture crashes inside `_update_current` -- before
    # `costco-gas-all-captures.parquet` itself is written, so `current` never
    # records this capture at all (unlike a later crash point, which could
    # legitimately already have it there).
    crashy = CrashOnceStore(inner, "current", "costco-gas-all-captures.parquet")
    a_dir = make_capture_dir(
        captures, "2026-09-15T0017Z", datetime(2026, 9, 15, 0, 17, tzinfo=UTC), prices=PRICES
    )
    with pytest.raises(RuntimeError, match="simulated crash"):
        publish(crashy, a_dir, cfg, now=NOW)

    # Confirm the exact starting shape: no month manifest was ever written,
    # even though the daily file and bundle both exist.
    assert not any(a.name == "manifest-2026-09.json" for a in inner.list_assets("data-2026-09"))
    day15 = schema.read_rows_csv_gz(
        inner.download("data-2026-09", "costco-gas-2026-09-15.csv.gz", tmp_path / "d15.csv.gz")
    )
    assert "2026-09-15T0017Z" in day15["capture_id"].unique().to_list()

    # September's second capture must trigger a rebuild that does NOT mark
    # the crashed capture done, and must reconcile its bundle instead.
    b_dir = make_capture_dir(
        captures,
        "2026-09-16T0017Z",
        datetime(2026, 9, 16, 0, 17, tzinfo=UTC),
        prices=[("Chungli", "95", "regular", 30.9)],
    )
    result = publish(inner, b_dir, cfg, now=NOW)

    assert "manifest_rebuilt" in result.warnings
    assert "reconciled:2026-09-15T0017Z" in result.warnings

    all_ids = {"2026-08-15T0017Z", "2026-09-15T0017Z", "2026-09-16T0017Z"}
    captures_all = pl.read_parquet(
        inner.download("current", "costco-gas-all-captures.parquet", tmp_path / "ac.parquet")
    )
    assert set(captures_all["capture_id"].to_list()) == all_ids
    # No duplicate (capture_id, station_key, grade_raw) rows: the crashed
    # capture's rows were merged exactly once, not fabricated as "done" and
    # then re-merged on top of themselves.
    key_cols = captures_all.select(["capture_id", "station_key", "grade_raw"])
    assert key_cols.is_duplicated().sum() == 0

    fx = pl.read_csv(
        inner.download("current", "fx.csv", tmp_path / "fx-final.csv"), schema=schema.FX_SCHEMA
    )
    assert sorted(fx["capture_id"].to_list()) == sorted(all_ids)

    manifest = json.loads(
        inner.download(
            "data-2026-09", "manifest-2026-09.json", tmp_path / "m-final.json"
        ).read_text()
    )
    assert set(manifest["captures"]) == {"2026-09-15T0017Z", "2026-09-16T0017Z"}
