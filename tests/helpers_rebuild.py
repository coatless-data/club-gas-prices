"""Builds real capture bundles and a seeded month release for the rebuild tests."""

from __future__ import annotations

import hashlib
import json
import shutil
import tarfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from costco_gas.capture import US_ID_SET_SCHEMA
from costco_gas.schema import FX_SCHEMA, ROW_SCHEMA, STATION_SCHEMA, write_rows_csv_gz

REPO_ROOT = Path(__file__).resolve().parents[1]
REPO_CONFIG = REPO_ROOT / "config"
FIXTURE = Path(__file__).parent / "fixtures" / "rebuild" / "au_stores_marsden.json"
AU_URL = (
    "https://www.costco.com.au/rest/v2/australia/stores"
    "?fields=FULL&pageSize=100&lang=en_AU&curr=AUD"
)


def au_body(price_e10: str = "$2.127") -> bytes:
    data = json.loads(FIXTURE.read_text("utf-8"))
    for gas in data["stores"][0]["gasTypes"]:
        if gas["name"] == "E10":
            gas["price"] = price_e10
    return json.dumps(data, ensure_ascii=False).encode("utf-8")


def fx_rows(capture_id: str) -> list[dict]:
    """The bundle's fx.json: a JSON array of non-USD rows (§6.4).

    1.399 AUD per USD is the real Frankfurter v2 rate for 2026-09-14.
    """
    stamp = datetime.strptime(capture_id, "%Y-%m-%dT%H%MZ").replace(tzinfo=UTC)
    return [
        {
            "currency": "AUD",
            "units_per_usd": 1.399,
            "fx_rate_date": "2026-09-14",
            "fx_source": "frankfurter-v2",
            "fx_fetched_at_utc": stamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
    ]


def previous_stations(capture_id: str) -> pl.DataFrame:
    """`current/stations.csv` as the capture read it, for `inputs/stations_used.csv`."""
    seen = datetime.strptime(capture_id, "%Y-%m-%dT%H%MZ").replace(tzinfo=UTC) - timedelta(days=1)
    return pl.DataFrame(
        [
            {
                "station_key": "AU-109",
                "country": "AU",
                "source_station_id": "109",
                "alt_id": None,
                "name": "Marsden Park",
                "name_local": None,
                "address": "8 Elara Boulevard",
                "city": "Marsden Park",
                "region": "NSW",
                "postcode": "2765",
                "lat": -33.6926,
                "lon": 150.8407,
                "timezone": "Australia/Sydney",
                "grades_seen": "E10|Premium Unleaded|Unleaded",
                "first_seen_utc": seen,
                "last_seen_utc": seen,
                "status": "active",
                "superseded_by": None,
            }
        ],
        schema=STATION_SCHEMA,
    )


def previous_fx(capture_id: str) -> pl.DataFrame:
    """`current/fx.csv` as the capture read it, for `inputs/fx_used.csv`."""
    stamp = datetime.strptime(capture_id, "%Y-%m-%dT%H%MZ").replace(tzinfo=UTC) - timedelta(days=1)
    return pl.DataFrame(
        [
            {
                "capture_id": stamp.strftime("%Y-%m-%dT%H%MZ"),
                "currency": "AUD",
                "units_per_usd": 1.4012,
                "fx_usd_per_unit": 1.0 / 1.4012,
                "fx_rate_date": stamp.date(),
                "fx_source": "frankfurter-v2",
                "fx_fetched_at_utc": stamp,
            }
        ],
        schema=FX_SCHEMA,
    )


def capture_status(capture_id: str) -> dict:
    return {
        "schema_version": 1,
        "capture_id": capture_id,
        "countries": {"AU": {"status": "ok"}},
        "fx": {"status": "ok", "source": "frankfurter-v2", "rate_date": "2026-09-14"},
    }


def make_bundle(
    dest: Path, *, capture_id: str, config_dir: Path, price_e10: str = "$2.127"
) -> Path:
    """A §8.2 capture bundle holding one stored AU response."""
    stamp = datetime.strptime(capture_id, "%Y-%m-%dT%H%MZ").replace(tzinfo=UTC)
    iso = stamp.strftime("%Y-%m-%dT%H:%M:%SZ")
    root = dest / f"build-{capture_id}"
    if root.exists():
        shutil.rmtree(root)
    (root / "responses" / "AU").mkdir(parents=True)
    (root / "inputs").mkdir(parents=True)
    shutil.copytree(config_dir, root / "config")
    (root / "responses" / "AU" / "01-stores.body").write_bytes(au_body(price_e10))
    (root / "responses" / "AU" / "01-stores.meta.json").write_text(
        json.dumps(
            {
                "key": "AU/01-stores",
                "url": AU_URL,
                "status": 200,
                "headers": {"Content-Type": "application/json"},
                "received_at_utc": iso,
                "elapsed_ms": 412,
                "error": None,
            }
        ),
        "utf-8",
    )
    (root / "capture.json").write_text(
        json.dumps(
            {
                "capture_id": capture_id,
                "capture_date": stamp.date().isoformat(),
                "started_at_utc": iso,
                "git_sha": "seed",
                "run_id": 1,
                "config_sha256": {},
            }
        ),
        "utf-8",
    )
    (root / "fx.json").write_text(
        json.dumps(fx_rows(capture_id), indent=1, sort_keys=True), "utf-8"
    )
    (root / "status.json").write_text(
        json.dumps(capture_status(capture_id), indent=1, sort_keys=True), "utf-8"
    )
    (root / "inputs" / "status_previous.json").write_text("null", "utf-8")
    # Written the way `capture._write_bundle` writes them: real frames through a
    # bare `DataFrame.write_csv`, whose datetime text is Polars' own
    # `2026-09-14T18:17:40.000000+0000`, not `schema.CSV_DATETIME_FORMAT`. Empty
    # strings here would leave `rebuild`'s reader untested against what the
    # producer actually emits.
    previous_stations(capture_id).write_csv(root / "inputs" / "stations_used.csv")
    previous_fx(capture_id).write_csv(root / "inputs" / "fx_used.csv")
    pl.DataFrame(schema=US_ID_SET_SCHEMA).write_csv(root / "inputs" / "us_id_set.csv")
    pl.DataFrame(schema=STATION_SCHEMA).write_csv(root / "stations.csv")
    write_rows_csv_gz(pl.DataFrame(schema=ROW_SCHEMA), root / "rows.csv.gz")
    archive = dest / f"capture-{capture_id}.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for path in sorted(root.rglob("*")):
            tar.add(path, arcname=str(path.relative_to(root)))
    return archive


def seed_month(
    store, month: str, *, bundles: dict[str, Path], work: Path, prerelease: bool = True
) -> dict:
    """Create `data-<month>` with one empty daily file per capture day and a manifest.

    The daily files start empty on purpose, so a test that finds rows in them
    afterwards has proved that `rebuild` rewrote them. The manifest carries the same
    fx array the bundle does, exactly as `publish` records it.
    """
    tag = f"data-{month}"
    work.mkdir(parents=True, exist_ok=True)
    store.ensure_release(tag, f"Data {month}", "", prerelease, "false")
    manifest = {"schema_version": 1, "month": month, "captures": {}}
    for capture_id, archive in bundles.items():
        day = capture_id[:10]
        store.upload_new(tag, archive, f"capture-{capture_id}.tar.gz")
        daily = work / f"costco-gas-{day}.csv.gz"
        if not daily.exists():
            write_rows_csv_gz(pl.DataFrame(schema=ROW_SCHEMA), daily)
            store.upload_new(tag, daily, daily.name)
        digest = hashlib.sha256(daily.read_bytes()).hexdigest()
        manifest["captures"][capture_id] = {
            "capture_id": capture_id,
            "status": capture_status(capture_id),
            "rows_by_capture_date": {day: 0},
            "fx": fx_rows(capture_id),
            "bundle": {
                "sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
                "rows": 0,
            },
            "daily_files": {day: {"sha256": digest, "rows": 0}},
            "rebuild_git_sha": None,
            "config_sha256": {},
        }
    path = work / f"manifest-{month}.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), "utf-8")
    store.upload_new(tag, path, path.name)
    return manifest


def checkout_with_config(tmp_path: Path) -> Path:
    """A fake repository checkout holding a copy of the live config/ directory."""
    checkout = tmp_path / "checkout"
    checkout.mkdir(parents=True, exist_ok=True)
    shutil.copytree(REPO_CONFIG, checkout / "config", dirs_exist_ok=True)
    return checkout


def restore_config(checkout: Path) -> None:
    """Put the unmodified live config/ back into a checkout."""
    target = checkout / "config"
    if target.exists():
        shutil.rmtree(target)
    shutil.copytree(REPO_CONFIG, target)


def drop_au_e10(checkout: Path) -> None:
    path = checkout / "config" / "grades.csv"
    kept = [
        line for line in path.read_text("utf-8").splitlines(True) if not line.startswith("AU,E10,")
    ]
    path.write_text("".join(kept), "utf-8")
