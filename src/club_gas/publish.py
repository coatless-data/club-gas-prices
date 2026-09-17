"""Publish one capture into the release store (spec 8.4, 8.5).

`merge_capture` writes the month manifest's per-capture entry LAST, after
`_update_current` has already succeeded -- not in the spec 8.5 listing order,
which puts the manifest before `current`. That entry is the commit marker
`_reconcile` uses to decide a capture's bundle is already handled
(`capture_id in manifest["captures"]`). If the entry were written before
`current` reflected the capture, a crash inside `_update_current` would leave
the capture marked done while `current` never saw its rows, and `_reconcile`
would skip that bundle forever once the day rolled over (spec review round 1,
Finding 1). Writing the manifest entry only once `current` is truly caught up
costs nothing: the daily file's own cross-check only ever asserts that capture
ids the manifest *already* records are still present with their recorded row
count, which a manifest that is momentarily behind never violates, and a
retry after any crash in this window is idempotent either way. Do not "tidy"
this back to manifest-before-current.
"""

from __future__ import annotations

import gzip
import json
import os
import re
import shutil
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl

from . import rollup, schema, sitedata
from .config import Config
from .issues import Issues
from .store import (
    AssetNotFound,
    ReleaseStore,
    StorageError,
    recover_temporaries,
    recovery_tags,
    sha256_file,
)

BUNDLE_RE = re.compile(r"^capture-(?P<capture_id>.+)\.tar\.gz$")

# The seven `current` assets publish reads back before rewriting them (spec 8.2).
# `club-gas-all.csv.gz` belongs here too: this function writes it, so leaving it out
# would let a half-replaced copy slip past the completeness guard below.
CURRENT_INPUT_ASSETS = (
    "club-gas-all.parquet",
    "club-gas-all.csv.gz",
    "club-gas-all-captures.parquet",
    "club-gas-latest.csv",
    "stations.csv",
    "fx.csv",
    "manifest.json",
)
ROW_KEY = ["capture_id", "station_key", "grade_raw"]
NOTICE = (
    "Unofficial. Not affiliated with, endorsed by, or connected to Costco Wholesale "
    "Corporation. Prices are collected from Costco's public websites and may differ "
    "from the price at the pump."
)


@dataclass
class PublishResult:
    capture_id: str
    warnings: list[str] = field(default_factory=list)
    assets_written: list[str] = field(default_factory=list)


@dataclass
class CaptureInput:
    capture_id: str
    capture_date: str  # "YYYY-MM-DD"
    rows: pl.DataFrame
    stations: pl.DataFrame
    fx: list[dict]
    status: dict
    bundle_sha256: str


def _empty_manifest(month: str) -> dict:
    return {"schema_version": 1, "month": month, "captures": {}}


def read_or_rebuild_manifest(
    store: ReleaseStore, tag: str, scratch: Path, warnings: list[str]
) -> dict:
    """Read `manifest-YYYY-MM.json`, rebuilding it from the daily files if it is gone."""
    month = tag[len("data-") :]
    name = f"manifest-{month}.json"
    try:
        path = store.download(tag, name, scratch / f"{tag}-{name}")
        return json.loads(path.read_text(encoding="utf-8"))
    except AssetNotFound:
        pass

    assets = store.list_assets(tag)
    daily_names = sorted(
        a.name for a in assets if re.fullmatch(r"club-gas-\d{4}-\d{2}-\d{2}\.csv\.gz", a.name)
    )
    if not daily_names:
        return _empty_manifest(month)

    warnings.append("manifest_rebuilt")
    bundles = {}
    for asset in assets:
        match = BUNDLE_RE.match(asset.name)
        if match and asset.state == "uploaded":
            bundles[match.group("capture_id")] = asset

    # A capture id may only get a rebuilt entry once `current/manifest.json`
    # itself lists it in `merged_captures` -- being in the daily file alone
    # only means merge_capture reached that far, not that `current` was ever
    # fully updated for it. Checking a data file (spec review round 2 checked
    # club-gas-all-captures.parquet) is not enough: that file is only the
    # third of seven `current` assets `_update_current` writes, so a crash
    # after it but before, say, fx.csv would still fabricate a "done" entry
    # (spec review round 3). `merged_captures` is written last, as part of
    # manifest.json, so it is a true commit marker: if a capture id is in it,
    # every other `current` asset already holds that capture's data. When
    # `current/manifest.json` itself is absent (the genuine first publish
    # ever, nothing has completed `_update_current` even once), no entries
    # are fabricated at all and every bundle in these daily files is left
    # for `_reconcile`.
    try:
        current_manifest_path = store.download(
            "current", "manifest.json", scratch / f"{tag}-current-manifest.json"
        )
        merged_captures = set(
            json.loads(current_manifest_path.read_text(encoding="utf-8")).get("merged_captures", [])
        )
    except AssetNotFound:
        merged_captures = None

    manifest = _empty_manifest(month)
    for daily_name in daily_names:
        day = daily_name[len("club-gas-") : -len(".csv.gz")]
        local = store.download(tag, daily_name, scratch / f"{tag}-{daily_name}")
        rows = schema.read_rows_csv_gz(local)
        digest = sha256_file(local)
        for capture_id, count in rows.group_by("capture_id").len().iter_rows():
            asset = bundles.get(capture_id)
            if asset is None:
                raise StorageError(
                    f"{daily_name} holds capture {capture_id} but no uploaded bundle exists"
                )
            if merged_captures is None or capture_id not in merged_captures:
                # Not (yet) fully merged into `current`: leave this capture out
                # of the rebuilt manifest so `_reconcile` re-merges its bundle.
                # The merge is idempotent -- it removes this capture id's rows
                # before re-appending them -- so re-running it is always safe.
                continue
            bundle_path = store.download(tag, asset.name, scratch / f"{tag}-{asset.name}")
            captured = read_bundle(bundle_path, capture_id)
            if captured is None:
                raise StorageError(f"bundle for {capture_id} is unreadable")
            entry = manifest["captures"].setdefault(
                capture_id,
                {
                    "status": captured.status,
                    "fx": captured.fx,
                    "rows_by_capture_date": {},
                    "bundle": {
                        "sha256": captured.bundle_sha256,
                        "rows": captured.rows.height,
                    },
                    "daily_files": {},
                },
            )
            entry["rows_by_capture_date"][day] = int(count)
            entry["daily_files"][day] = {"sha256": digest, "rows": rows.height}
    return manifest


def _read_rows_dir(directory: Path) -> pl.DataFrame:
    return schema.read_rows_csv_gz(directory / "rows.csv.gz")


def load_capture_dir(directory: Path) -> CaptureInput:
    directory = Path(directory)
    status = json.loads((directory / "status.json").read_text(encoding="utf-8"))
    meta = json.loads((directory / "capture.json").read_text(encoding="utf-8"))
    return CaptureInput(
        capture_id=status["capture_id"],
        capture_date=meta["capture_date"],
        rows=_read_rows_dir(directory),
        stations=pl.read_csv(directory / "stations.csv", schema=schema.STATION_SCHEMA),
        # fx.json is the JSON array capture.py wrote from FxRates.to_json().
        fx=json.loads((directory / "fx.json").read_text(encoding="utf-8")),
        status=status,
        bundle_sha256=sha256_file(directory / "bundle.tar.gz"),
    )


def read_bundle(path: Path, capture_id: str) -> CaptureInput | None:
    """Validate and read an uploaded bundle; None means invalid (spec 8.5 step 0)."""
    required = ("capture.json", "rows.csv.gz", "stations.csv", "fx.json", "status.json")
    try:
        with tempfile.TemporaryDirectory(prefix="costco-bundle-") as temp:
            work = Path(temp)
            with tarfile.open(path, "r:gz") as tar:
                names = set(tar.getnames())
                if not set(required) <= names:
                    return None
                tar.extractall(work, filter="data")
            meta = json.loads((work / "capture.json").read_text(encoding="utf-8"))
            if meta.get("capture_id") != capture_id:
                return None
            return CaptureInput(
                capture_id=capture_id,
                capture_date=meta["capture_date"],
                rows=_read_rows_dir(work),
                stations=pl.read_csv(work / "stations.csv", schema=schema.STATION_SCHEMA),
                fx=json.loads((work / "fx.json").read_text(encoding="utf-8")),
                status=json.loads((work / "status.json").read_text(encoding="utf-8")),
                bundle_sha256=sha256_file(path),
            )
    except (tarfile.TarError, OSError, ValueError, KeyError, schema.SchemaError):
        return None


def _check_unique(frame: pl.DataFrame, keys: list[str], label: str) -> None:
    if frame.height and frame.select(pl.struct(keys).n_unique()).item() != frame.height:
        raise StorageError(f"{label}: duplicate rows for {keys}")


def _write_csv_gz(frame: pl.DataFrame, path: Path) -> None:
    """Gzip a frame that carries extra roll-up columns, so validate_rows is skipped.

    `gzip.open` has no `mtime` parameter; a fixed `mtime=0` (as schema.py's own
    `_write_gzip` uses) keeps the bytes identical for identical content, so
    re-publishing a capture never changes this asset's digest.
    """
    with (
        path.open("wb") as handle,
        gzip.GzipFile(filename="", mode="wb", fileobj=handle, mtime=0) as gz,
    ):
        gz.write(frame.write_csv().encode("utf-8"))


def _month_body(month: str) -> str:
    return f"Costco gas prices captured during {month}.\n\n{NOTICE}\n"


def ensure_current(store: ReleaseStore) -> None:
    store.ensure_release("current", "Current", f"All-time files.\n\n{NOTICE}\n", False, "true")


def ensure_releases(store: ReleaseStore, month: str) -> None:
    """Spec 8.5 step 1, including reopening a month closed by an earlier run."""
    ensure_current(store)
    tag = f"data-{month}"
    existing = store.get_release(tag)
    store.ensure_release(tag, f"Data {month}", _month_body(month), True, "false")
    if existing is not None and not existing.prerelease:
        store.update_release(tag, prerelease=True)


def upload_bundle(
    store: ReleaseStore, tag: str, path: Path, capture_id: str, local_sha: str, scratch: Path
) -> str:
    """Upload the bundle once; spec 8.5 step 2 covers the HTTP 422 duplicate-name path."""
    name = f"capture-{capture_id}.tar.gz"
    for attempt in (1, 2):
        existing = next((a for a in store.list_assets(tag) if a.name == name), None)
        if existing is None:
            try:
                store.upload_new(tag, path, name)
                return name
            except StorageError:
                # GitHub answers 422 when the name was taken between the list and
                # the upload; loop once more and inspect whatever is there now.
                if attempt == 2:
                    raise
                continue
        if existing.state != "uploaded":
            store.delete(tag, existing.id)
            if attempt == 2:
                raise StorageError(f"{name} never reached state uploaded")
            continue
        remote = existing.digest
        if not remote:
            dest = scratch / f"{name}.remote"
            store.download(tag, name, dest)
            remote = f"sha256:{sha256_file(dest)}"
        if remote != f"sha256:{local_sha}":
            raise StorageError(f"capture_id_collision: {name} exists with a different digest")
        return name
    raise StorageError(f"could not upload {name}")


def update_latest(previous: pl.DataFrame, incoming: pl.DataFrame) -> pl.DataFrame:
    """Spec 8.5 step 5.4: replace a station's rows only with a newer capture."""
    if incoming.height == 0:
        return previous
    if previous.height == 0:
        return incoming.sort(["station_key", "grade_raw"])
    new_ts = incoming.group_by("station_key").agg(pl.col("captured_at_utc").max().alias("new_ts"))
    old_ts = previous.group_by("station_key").agg(pl.col("captured_at_utc").max().alias("old_ts"))
    winners = (
        new_ts.join(old_ts, on="station_key", how="left")
        .filter(pl.col("old_ts").is_null() | (pl.col("new_ts") > pl.col("old_ts")))["station_key"]
        .to_list()
    )
    kept = previous.filter(~pl.col("station_key").is_in(winners))
    taken = incoming.filter(pl.col("station_key").is_in(winners))
    return pl.concat([kept, taken], how="vertical").sort(["station_key", "grade_raw"])


def _split_grades(value: str | None) -> set[str]:
    return {part for part in (value or "").split("|") if part}


def capture_time(capture_id: str) -> datetime | None:
    """The capture id is its own UTC timestamp, to the minute (spec 6.1)."""
    try:
        return datetime.strptime(capture_id, "%Y-%m-%dT%H%MZ").replace(tzinfo=UTC)
    except ValueError:
        return None


def station_status(last_seen, at: datetime | None, closed_after_days: int | None) -> str:
    """What to call a station that its country's feed no longer lists.

    A station is never dropped -- once seen it stays in the record forever -- so
    the only question is what to call it. Absent for one capture is a feed
    hiccup, a partial failure or a delisting over a refurbishment; absent for
    weeks is a closure. Calling the first one closed would be wrong far more
    often than right, so `missing` is the near term and `closed` is the durable
    statement. Neither is terminal: a station that comes back reads active
    again, because Costco does reopen them.
    """
    if not closed_after_days or at is None or last_seen is None:
        return "missing"
    gone = at - (last_seen if last_seen.tzinfo else last_seen.replace(tzinfo=UTC))
    return "closed" if gone > timedelta(days=closed_after_days) else "missing"


def upsert_stations(
    previous: pl.DataFrame,
    incoming: pl.DataFrame,
    status: dict,
    links: pl.DataFrame,
    newest_by_country: dict[str, str],
    capture_id: str,
    closed_after_days: dict[str, int] | None = None,
) -> pl.DataFrame:
    """Spec 6.3 incremental upsert.

    A capture older than its country's newest merged capture touches only
    first_seen_utc, last_seen_utc and grades_seen; metadata, alt_id and status
    stay as they are. Capture ids sort chronologically as plain strings.

    Stations are only ever added. One that disappears from its feed keeps its
    row and its history and is relabelled; see `station_status`.
    """
    metadata = [
        "source_station_id",
        "alt_id",
        "name",
        "name_local",
        "address",
        "city",
        "region",
        "postcode",
        "lat",
        "lon",
        "timezone",
    ]
    rows: dict[str, dict] = {r["station_key"]: dict(r) for r in previous.to_dicts()}
    succeeded = {
        code
        for code, block in (status.get("countries") or {}).items()
        if block.get("status") in ("ok", "degraded")
    }
    incoming_keys = set(incoming["station_key"].to_list())

    for record in incoming.to_dicts():
        key = record["station_key"]
        newest = capture_id > (newest_by_country.get(record["country"]) or "")
        current = rows.get(key)
        if current is None:
            rows[key] = dict(record)
            continue
        if newest:
            for column in metadata:
                current[column] = record[column]
            current["status"] = "active"
        seen = [current["first_seen_utc"], record["first_seen_utc"]]
        current["first_seen_utc"] = min(v for v in seen if v is not None)
        seen = [current["last_seen_utc"], record["last_seen_utc"]]
        current["last_seen_utc"] = max(v for v in seen if v is not None)
        current["grades_seen"] = "|".join(
            sorted(_split_grades(current["grades_seen"]) | _split_grades(record["grades_seen"]))
        )

    at = capture_time(capture_id)
    thresholds = closed_after_days or {}
    for key, record in rows.items():
        if key in incoming_keys:
            continue
        if record["country"] in succeeded and capture_id > (
            newest_by_country.get(record["country"]) or ""
        ):
            record["status"] = station_status(
                record["last_seen_utc"], at, thresholds.get(record["country"])
            )

    link_map: dict[str, str] = {}
    if links.height:
        link_map = dict(
            zip(
                links["old_station_key"].to_list(),
                links["new_station_key"].to_list(),
                strict=True,
            )
        )
    for key, record in rows.items():
        record["superseded_by"] = link_map.get(key)

    return pl.DataFrame(list(rows.values()), schema=schema.STATION_SCHEMA).sort("station_key")


def upsert_fx(previous: pl.DataFrame, capture_id: str, fx_rows: list[dict]) -> pl.DataFrame:
    """Spec 6.4: fx.csv always comes from the capture's fx.json, never from price rows."""
    if not fx_rows:
        return previous
    incoming = pl.DataFrame(
        [
            {
                "capture_id": capture_id,
                "currency": row["currency"],
                "units_per_usd": float(row["units_per_usd"]),
                "fx_usd_per_unit": 1.0 / float(row["units_per_usd"]),
                "fx_rate_date": date.fromisoformat(row["fx_rate_date"]),
                "fx_source": row["fx_source"],
                "fx_fetched_at_utc": datetime.fromisoformat(
                    row["fx_fetched_at_utc"].replace("Z", "+00:00")
                ),
            }
            for row in fx_rows
        ],
        schema=schema.FX_SCHEMA,
    )
    kept = previous.filter(pl.col("capture_id") != capture_id)
    merged = pl.concat([kept, incoming], how="vertical").sort(["capture_id", "currency"])
    # §6.1: the rates are stored to 10 significant digits. `schema.write_csv`
    # rounds on the way out, but this frame is also compared against the stored
    # one (the "would lose rows" guard) and against `rollup._fx_frame`'s, which
    # rounds here too -- so round here as well and keep the two writers equal.
    return schema.round_fx_columns(merged)


def merge_capture(
    store: ReleaseStore,
    cfg: Config,
    captured: CaptureInput,
    scratch: Path,
    warnings: list[str],
    assets_written: list[str],
    *,
    now: datetime,
) -> None:
    """Spec 8.5 steps 3 to 5 for one capture."""
    month = captured.capture_date[:7]
    tag = f"data-{month}"
    manifest = read_or_rebuild_manifest(store, tag, scratch, warnings)
    day = captured.capture_date
    daily_name = f"club-gas-{day}.csv.gz"

    recorded = {
        capture_id: entry["rows_by_capture_date"][day]
        for capture_id, entry in manifest["captures"].items()
        if day in entry.get("rows_by_capture_date", {})
    }
    local_daily = scratch / f"{tag}-in-{daily_name}"
    try:
        store.download(tag, daily_name, local_daily)
        daily = schema.read_rows_csv_gz(local_daily)
    except AssetNotFound:
        if recorded:
            raise StorageError(
                f"{daily_name} is missing but the manifest records {sorted(recorded)}"
            ) from None
        daily = pl.DataFrame(schema=schema.ROW_SCHEMA)

    daily = daily.filter(pl.col("capture_id") != captured.capture_id)
    merged = pl.concat([daily, captured.rows], how="vertical").sort(
        ["capture_id", "country", "station_key", "grade_raw"]
    )
    _check_unique(merged, ROW_KEY, daily_name)
    counts = {cid: int(n) for cid, n in merged.group_by("capture_id").len().iter_rows()}
    for capture_id, expected in recorded.items():
        if capture_id == captured.capture_id:
            continue
        if counts.get(capture_id) != expected:
            raise StorageError(
                f"{daily_name}: capture {capture_id} has {counts.get(capture_id)} rows, "
                f"the manifest records {expected}"
            )

    out_daily = scratch / f"{tag}-out-{daily_name}"
    schema.write_rows_csv_gz(merged, out_daily)
    store.replace_atomic(tag, out_daily, daily_name, captured.capture_id)
    assets_written.append(f"{tag}/{daily_name}")

    # `current` is updated BEFORE the month manifest records this capture: see the
    # module docstring for why that order, not the daily-file-then-manifest order
    # spec 8.5 lists, is the one that keeps a crash recoverable.
    _update_current(store, cfg, captured, merged, scratch, assets_written, now=now)

    entry = manifest["captures"].setdefault(captured.capture_id, {})
    entry["status"] = captured.status
    entry["fx"] = captured.fx
    entry["rows_by_capture_date"] = {
        str(d): int(n)
        for d, n in captured.rows.group_by(pl.col("capture_date").dt.to_string("%Y-%m-%d"))
        .len()
        .iter_rows()
    }
    entry["bundle"] = {"sha256": captured.bundle_sha256, "rows": captured.rows.height}
    entry.setdefault("daily_files", {})[day] = {
        "sha256": sha256_file(out_daily),
        "rows": merged.height,
    }
    manifest_name = f"manifest-{month}.json"
    manifest_path = scratch / f"{tag}-out-{manifest_name}"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    store.replace_atomic(tag, manifest_path, manifest_name, captured.capture_id)
    assets_written.append(f"{tag}/{manifest_name}")


def _update_current(
    store: ReleaseStore,
    cfg: Config,
    captured: CaptureInput,
    merged_daily: pl.DataFrame,
    scratch: Path,
    assets_written: list[str],
    *,
    now: datetime,
) -> None:
    ensure_current(store)
    local: dict[str, Path] = {}
    missing: list[str] = []
    for name in CURRENT_INPUT_ASSETS:
        try:
            local[name] = store.download("current", name, scratch / f"current-in-{name}")
        except AssetNotFound:
            missing.append(name)
    # First-publish means the whole of `current` is absent, not just manifest.json:
    # a lone missing manifest.json alongside real data in the other six assets is
    # corruption, not a fresh start, and rebuilding from empty would silently
    # discard that history (spec review Finding 3). Task 15's full `current`
    # rebuild is the repair path for that state, not this function.
    if len(missing) == len(CURRENT_INPUT_ASSETS):
        first = True
    elif missing:
        raise StorageError("current is incomplete: missing " + ", ".join(sorted(missing)))
    else:
        first = False

    manifest = {
        "schema_version": 1,
        "status": None,
        "closed_months": [],
        "closed_years": [],
        "newest_capture_by_country": {},
        "assets": {},
        "merged_captures": [],
    }
    if not first:
        manifest.update(json.loads(local["manifest.json"].read_text(encoding="utf-8")))
        manifest.setdefault("newest_capture_by_country", {})
        manifest.setdefault("assets", {})
        manifest.setdefault("merged_captures", [])

    all_caps = (
        pl.read_parquet(local["club-gas-all-captures.parquet"])
        if not first
        else pl.DataFrame(schema=schema.ROW_SCHEMA)
    )
    all_daily = (
        pl.read_parquet(local["club-gas-all.parquet"])
        if not first
        else pl.DataFrame(schema=rollup.DAILY_SCHEMA)
    )
    latest = (
        pl.read_csv(local["club-gas-latest.csv"], schema=schema.ROW_SCHEMA)
        if not first
        else pl.DataFrame(schema=schema.ROW_SCHEMA)
    )
    stations = (
        pl.read_csv(local["stations.csv"], schema=schema.STATION_SCHEMA)
        if not first
        else pl.DataFrame(schema=schema.STATION_SCHEMA)
    )
    fx = (
        pl.read_csv(local["fx.csv"], schema=schema.FX_SCHEMA)
        if not first
        else pl.DataFrame(schema=schema.FX_SCHEMA)
    )

    day = date.fromisoformat(captured.capture_date)
    old_dates = set(all_caps["capture_date"].to_list()) | set(all_daily["capture_date"].to_list())
    new_caps = pl.concat(
        [all_caps.filter(pl.col("capture_date") != day), merged_daily], how="vertical"
    )
    new_all = pl.concat(
        [all_daily.filter(pl.col("capture_date") != day), rollup.daily_grain(merged_daily)],
        how="vertical",
    )
    new_latest = update_latest(latest, captured.rows)
    new_stations = upsert_stations(
        stations,
        captured.stations,
        captured.status,
        cfg.station_links,
        manifest["newest_capture_by_country"],
        captured.capture_id,
        {code: c.closed_after_days for code, c in cfg.countries.items()},
    )
    new_fx = upsert_fx(fx, captured.capture_id, captured.fx)

    for frame, label in ((new_caps, "all-captures"), (new_all, "all")):
        lost = old_dates - set(frame["capture_date"].to_list())
        if lost:
            raise StorageError(f"{label} would lose capture dates {sorted(lost)}")
    if new_stations.height < stations.height:
        raise StorageError("stations.csv would lose rows")
    if new_fx.height < fx.height:
        raise StorageError("fx.csv would lose rows")
    _check_unique(new_caps, ROW_KEY, "club-gas-all-captures.parquet")
    _check_unique(new_all, rollup.DAILY_KEYS, "club-gas-all.parquet")
    _check_unique(new_latest, ["station_key", "grade_raw"], "club-gas-latest.csv")
    _check_unique(new_stations, ["station_key"], "stations.csv")
    _check_unique(new_fx, ["capture_id", "currency"], "fx.csv")

    daily_sort = ["capture_date", "country", "station_key", "grade_raw"]
    capture_sort = ["station_key", "grade_raw", "capture_id"]
    outputs: dict[str, Path] = {}

    path = scratch / "out-club-gas-all.parquet"
    schema.write_parquet(new_all, path, sort_by=daily_sort)
    outputs["club-gas-all.parquet"] = path

    path = scratch / "out-club-gas-all.csv.gz"
    _write_csv_gz(new_all.sort(daily_sort), path)
    outputs["club-gas-all.csv.gz"] = path

    path = scratch / "out-club-gas-all-captures.parquet"
    schema.write_parquet(new_caps, path, sort_by=capture_sort)
    outputs["club-gas-all-captures.parquet"] = path

    path = scratch / "out-club-gas-latest.csv"
    new_latest.write_csv(path)
    outputs["club-gas-latest.csv"] = path

    # Both assets go out through `schema.write_csv`, which validates them against
    # their schema, re-checks the key, rounds the rate columns and writes the
    # canonical CSV date formats -- the same guarantees the row files get, rather
    # than ones each call site has to remember (final review, finding 3).
    path = scratch / "out-stations.csv"
    schema.write_csv(new_stations, path, schema=schema.STATION_SCHEMA, sort_by=schema.STATION_SORT)
    outputs["stations.csv"] = path

    path = scratch / "out-fx.csv"
    schema.write_csv(new_fx, path, schema=schema.FX_SCHEMA, sort_by=schema.FX_SORT)
    outputs["fx.csv"] = path

    # Every other `current` asset is durable at this point: all six
    # replace_atomic calls above have already succeeded. Only now is it safe
    # to record this capture as merged -- merged_captures is written below as
    # part of manifest.json, the LAST `current` write, making it a true
    # commit marker (spec review round 3). Recording it any earlier, or
    # deriving completeness from one of the data files instead, would let a
    # crash between two of those six writes fabricate a "done" capture that
    # `current` does not actually fully reflect.
    for name, path in outputs.items():
        store.replace_atomic("current", path, name, captured.capture_id)
        assets_written.append(f"current/{name}")
        manifest["assets"][name] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }

    stored = (manifest.get("status") or {}).get("capture_id") or ""
    if captured.capture_id > stored:
        manifest["status"] = captured.status
    # newest_capture_by_country drives the spec 6.3 late-capture rule, so it is
    # part of current/manifest.json and the full rebuild recomputes it too.
    for code, block in (captured.status.get("countries") or {}).items():
        if block.get("status") in ("ok", "degraded") and captured.capture_id > (
            manifest["newest_capture_by_country"].get(code) or ""
        ):
            manifest["newest_capture_by_country"][code] = captured.capture_id
    manifest["merged_captures"] = sorted(
        {*manifest.get("merged_captures", []), captured.capture_id}
    )

    # The dashboard is a separate repository with no Python in it: it downloads
    # these five files and renders. They are built here, from the same frames the
    # six assets above were written from, and uploaded inside the same
    # transaction -- which is what stops the site from ever showing one capture's
    # map over another capture's history. They are built after `status` and
    # `merged_captures` are settled because meta.json reads both.
    for name, path in _build_site_assets(cfg, scratch, outputs, manifest, now=now).items():
        store.replace_atomic("current", path, name, captured.capture_id)
        assets_written.append(f"current/{name}")
        manifest["assets"][name] = {
            "sha256": sha256_file(path),
            "size": path.stat().st_size,
        }

    manifest_path = scratch / "out-manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=1, sort_keys=True), encoding="utf-8")
    store.replace_atomic("current", manifest_path, "manifest.json", captured.capture_id)
    assets_written.append("current/manifest.json")


def _build_site_assets(
    cfg: Config,
    scratch: Path,
    outputs: dict[str, Path],
    manifest: dict,
    *,
    now: datetime,
) -> dict[str, Path]:
    """Build the dashboard's files from the `current` assets about to be uploaded.

    `build_site_data` reads a directory laid out like a downloaded `current`, so
    the freshly written assets are linked into one under their canonical names,
    together with the manifest as it will be written a moment later.
    """
    staged = scratch / "site-input"
    staged.mkdir(parents=True, exist_ok=True)
    for name, path in outputs.items():
        target = staged / name
        target.unlink(missing_ok=True)
        try:
            os.link(path, target)
        except OSError:
            shutil.copyfile(path, target)
    (staged / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")

    out_dir = scratch / "site-data"
    sitedata.build_site_data(staged, out_dir, cfg, now=now)
    built = {}
    for name, asset in sitedata.SITE_ASSETS.items():
        path = out_dir / name
        if not path.exists():
            raise StorageError(f"site data is missing {name}")
        built[asset] = path
    return built


def _dedupe(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        if value not in seen:
            seen.add(value)
            out.append(value)
    return out


def publish(store: ReleaseStore, capture_dir: Path, cfg: Config, *, now: datetime) -> PublishResult:
    """Publish one capture directory into the release store (spec 8.5)."""
    captured = load_capture_dir(Path(capture_dir))
    warnings: list[str] = []
    assets_written: list[str] = []
    issues = Issues(os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN"))
    month = captured.capture_date[:7]

    with tempfile.TemporaryDirectory(prefix="costco-publish-") as temp:
        scratch = Path(temp)
        # Spec 8.4 recovery, implemented once in store.py.
        for tag in recovery_tags(store):
            recover_temporaries(store, tag)
        ensure_releases(store, month)
        _reconcile(
            store, cfg, scratch, captured.capture_id, warnings, assets_written, issues, now=now
        )
        upload_bundle(
            store,
            f"data-{month}",
            Path(capture_dir) / "bundle.tar.gz",
            captured.capture_id,
            captured.bundle_sha256,
            scratch,
        )
        assets_written.append(f"data-{month}/capture-{captured.capture_id}.tar.gz")
        merge_capture(store, cfg, captured, scratch, warnings, assets_written, now=now)

    return PublishResult(
        capture_id=captured.capture_id,
        warnings=_dedupe(warnings),
        assets_written=assets_written,
    )


def _reconcile(
    store: ReleaseStore,
    cfg: Config,
    scratch: Path,
    own_capture_id: str,
    warnings: list[str],
    assets_written: list[str],
    issues: Issues,
    *,
    now: datetime,
) -> None:
    """Spec 8.5 step 0: bundles uploaded by a crashed publish are merged or removed."""
    for release in store.list_releases():
        tag = release.tag
        if not tag.startswith("data-") or not release.prerelease:
            continue
        if len(tag) != len("data-YYYY-MM"):
            continue  # data-YYYY year releases hold no bundles
        manifest = read_or_rebuild_manifest(store, tag, scratch, warnings)
        for asset in sorted(store.list_assets(tag), key=lambda a: a.name):
            match = BUNDLE_RE.match(asset.name)
            if not match:
                continue
            capture_id = match.group("capture_id")
            if capture_id == own_capture_id or capture_id in manifest["captures"]:
                continue
            if asset.state != "uploaded":
                store.delete(tag, asset.id)
                warnings.append(f"discarded_incomplete_bundle:{capture_id}")
                continue
            dest = scratch / f"orphan-{capture_id}.tar.gz"
            try:
                store.download(tag, asset.name, dest)
            except (AssetNotFound, StorageError):
                # Could not read it this run -- not proof the bundle is bad. Leave
                # it in place and let the next publish retry it (spec review
                # Finding 2); renaming it here on a transient failure would
                # destroy a perfectly good bundle.
                warnings.append(f"orphan_bundle_download_failed:{capture_id}")
                continue
            orphan = read_bundle(dest, capture_id)
            if orphan is None:
                store.rename(tag, asset.id, f"{asset.name}.invalid")
                issues.ensure_open(
                    f"Invalid capture bundle: {capture_id}",
                    f"`{asset.name}` in `{tag}` did not parse and was renamed to "
                    f"`{asset.name}.invalid`. The workflow artifact for this capture "
                    "may still be available for 30 days.",
                    ["capture-failure"],
                )
                continue
            merge_capture(store, cfg, orphan, scratch, warnings, assets_written, now=now)
            warnings.append(f"reconciled:{capture_id}")
