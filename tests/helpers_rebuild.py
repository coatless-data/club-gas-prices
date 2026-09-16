"""Builds real capture bundles and a seeded month release for the rebuild tests."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from costco_gas.schema import ROW_SCHEMA, write_rows_csv_gz

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
    for name in ("stations_used.csv", "fx_used.csv", "us_id_set.csv"):
        (root / "inputs" / name).write_text("", "utf-8")
    (root / "stations.csv").write_text("", "utf-8")
    with gzip.open(root / "rows.csv.gz", "wb") as fh:
        fh.write(b"")
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
