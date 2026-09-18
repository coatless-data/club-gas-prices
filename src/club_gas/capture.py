"""Capture orchestration (spec 5.3)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.parse import urlencode

import polars as pl

from . import checks, schema
from .config import Config
from .fx import FxRates, fetch_rates
from .http import BudgetExceeded, Client
from .issues import run_url
from .normalize import normalize
from .sources import us as us_source
from .sources.base import (
    SOURCES,
    CaptureContext,
    Error,
    FetchResult,
    RawResponse,
    Warning,
    feeds_for,
)
from .store import ReleaseStore, sha256_file

# Paths relative to the repo checkout the CLI runs in.
CONFIG_DIR = Path("config")
STATUS_PATH = Path("status/latest.json")

# Shared warehouse locator for US and CA (spec 4.2, 5.3). The client-identifier
# is a public id from Costco's front end; configurable via countries.toml.
ECOM_API_URL = "https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json"
ECOM_API_PARAMS = {"latitude": "0", "longitude": "0", "limit": "5000"}
ECOM_CLIENT_IDENTIFIER = "7c71124c-7bf1-44db-bc9d-498584cd66e5"

DROP_SCHEMA = {
    "source_station_id": pl.String,
    "reason": pl.String,
    "detail": pl.String,
}
US_ID_SET_SCHEMA = {
    "source_station_id": pl.String,
    "id_origin": pl.String,
    "ecom_state": pl.String,
}

# Countries that depend on the previous `current/stations.csv`.
PREVIOUS_STATE_COUNTRIES = ("US", "CA")


@dataclass
class CaptureResult:
    status: dict
    all_failed: bool
    capture_id: str
    out: Path


def capture_id_for(now: datetime) -> str:
    """The capture id: UTC start time truncated to the minute (spec 6.1)."""
    return now.astimezone(UTC).strftime("%Y-%m-%dT%H%MZ")


def github_output(key: str, value: str) -> None:
    """Append a step output when running under Actions; a no-op otherwise."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def _utc_text(value: datetime) -> str:
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_response(directory: Path, resp: RawResponse, name: str) -> None:
    """Write one raw response as `<name>.body` plus `<name>.meta.json`."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.body").write_bytes(resp.body)
    meta = {
        "key": resp.key,
        "url": resp.url,
        "status": resp.status,
        "headers": dict(resp.headers),
        "received_at_utc": _utc_text(resp.received_at_utc),
        "elapsed_ms": resp.elapsed_ms,
        "error": resp.error,
        "body_bytes": len(resp.body),
        "body_sha256": hashlib.sha256(resp.body).hexdigest(),
    }
    (directory / f"{name}.meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )


def _read_previous_state(
    store: ReleaseStore, scratch: Path
) -> tuple[pl.DataFrame, pl.DataFrame, bool]:
    """Read `stations.csv` and `fx.csv` from `current` (spec 5.3 step 1).

    Uses `read_resolved` to handle partially-written assets from crashed publishes.
    """
    scratch.mkdir(parents=True, exist_ok=True)
    complete = True
    frames: list[pl.DataFrame] = []
    for name in ("stations.csv", "fx.csv"):
        try:
            path = store.read_resolved("current", name, scratch / name)
            frames.append(pl.read_csv(path))
        except Exception:
            frames.append(pl.DataFrame())
            complete = False
    return frames[0], frames[1], complete


def _read_previous_status() -> tuple[dict | None, list[Warning]]:
    if not STATUS_PATH.exists():
        return None, []
    try:
        return json.loads(STATUS_PATH.read_text(encoding="utf-8")), []
    except (json.JSONDecodeError, OSError):
        return None, [Warning(code="previous_status_unreadable")]


def _ecom_request(cfg: Config) -> tuple[str, str]:
    """The warehouse-locator URL and client-identifier (spec 4.2 step 1).

    These are non-query settings, so they are their own fields on
    `CountryConfig` under `[countries.US]`, never entries in `params`.
    """
    country = cfg.countries.get("US")
    base = getattr(country, "ecom_url", None) or ECOM_API_URL
    query = dict(getattr(country, "ecom_params", None) or ECOM_API_PARAMS)
    identifier = getattr(country, "ecom_client_identifier", None) or ECOM_CLIENT_IDENTIFIER
    url = f"{base}?{urlencode(query)}" if query else base
    return url, identifier


def _fetch_ecom(client: Client, cfg: Config) -> tuple[RawResponse | None, dict]:
    """Fetch the shared warehouse locator once (spec 5.3 step 3)."""
    url, identifier = _ecom_request(cfg)
    try:
        with client.budget("ecom-api"):
            resp = client.request("shared/ecom-api", url, headers={"client-identifier": identifier})
    except BudgetExceeded:
        return None, {"attempted": True, "http_status": None}
    return resp, {"attempted": True, "http_status": resp.status}


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def run_capture(
    cfg: Config,
    store: ReleaseStore,
    out: Path,
    *,
    countries: list[str],
    force_fallback: set[str],
    now: datetime,
    client: Client | None = None,
) -> CaptureResult:
    """Collect every selected country into `out` (spec 5.3)."""
    out = Path(out)
    started = now.astimezone(UTC)
    capture_id = capture_id_for(started)
    github_output("capture_id", capture_id)

    monotonic_start = time.monotonic()
    capture_out = out / "capture"
    capture_out.mkdir(parents=True, exist_ok=True)
    scratch = out / "state"

    warnings: list[Warning] = []
    prev_stations, prev_fx, complete = _read_previous_state(store, scratch)
    if not complete:
        warnings.append(Warning(code="previous_state_unavailable"))
    prev_status, status_warnings = _read_previous_status()
    warnings.extend(status_warnings)

    if client is None:
        client = Client(cfg.http)

    ctx = CaptureContext(
        capture_id=capture_id,
        capture_date=started.date(),
        fetch_config=cfg.fetch_view(),
        interp_config=cfg.interp_view(),
        previous_stations=prev_stations,
        previous_fx=prev_fx,
        previous_state_complete=complete,
        previous_status=prev_status,
        shared={},
        force_fallback=set(force_fallback),
    )

    fx = fetch_rates(client, ctx)

    ecom_api: dict = {"attempted": False, "http_status": None}
    if {"US", "CA"} & set(countries):
        resp, ecom_api = _fetch_ecom(client, cfg)
        if resp is not None:
            ctx.shared["ecom-api"] = resp
            write_response(out / "shared", resp, "ecom-api")

    # Disabled feeds are excluded from the run.
    configured = getattr(cfg, "feeds", {}) or {}
    feeds = [fid for fid in feeds_for(countries) if getattr(configured.get(fid), "enabled", True)]
    blocks, collected = _run_feeds(feeds, client, ctx, fx, out, now, warnings)

    rows = _concat(collected, "rows", schema.ROW_SCHEMA).sort(schema.ROW_SORT)
    stations = _concat(collected, "stations", schema.STATION_SCHEMA).sort(schema.STATION_SORT)
    schema.write_rows_csv_gz(rows, capture_out / "rows.csv.gz")
    stations.write_csv(capture_out / "stations.csv")
    # fx.json: rate rows as JSON (spec 4.5), read by publish and rebuild.
    (capture_out / "fx.json").write_text(
        json.dumps(fx.to_json(), indent=1, sort_keys=True), encoding="utf-8"
    )

    finished = started + timedelta(seconds=time.monotonic() - monotonic_start)
    run = {
        "run_id": _env_int("GITHUB_RUN_ID"),
        "run_attempt": _env_int("GITHUB_RUN_ATTEMPT"),
        "run_url": run_url(),
        "git_sha": os.environ.get("GITHUB_SHA"),
        "started_at_utc": _utc_text(started),
        "finished_at_utc": _utc_text(finished),
        "warnings": warnings,
    }
    status = checks.build_status(ctx, blocks, fx, ecom_api, now, run)
    (capture_out / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    try:
        _write_bundle(out, capture_out, ctx, run, collected)
    except Exception as exc:
        # Bundle failure is recorded, not fatal — status.json is already durable.
        detail = f"{type(exc).__name__}: {exc}"
        status["warnings"].append({"code": "bundle_failed", "detail": detail})
        (capture_out / "status.json").write_text(
            json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    all_failed = not any(
        block.get("status") in ("ok", "degraded") for block in status.get("feeds", {}).values()
    )
    github_output("all_failed", "true" if all_failed else "false")
    return CaptureResult(
        status=status, all_failed=all_failed, capture_id=capture_id, out=capture_out
    )


def _concat(collected: dict[str, dict], key: str, frame_schema: dict) -> pl.DataFrame:
    frames = [collected[cc][key] for cc in sorted(collected)]
    frames = [f for f in frames if f is not None]
    if not frames:
        return pl.DataFrame(schema=frame_schema)
    return pl.concat(frames, how="vertical")


def run_feed(
    fid: str,
    client: Client,
    ctx: CaptureContext,
    fx: FxRates,
    out: Path,
    now: datetime,
) -> dict:
    """Fetch, parse, normalize, check and write one feed (spec 5.3 step 4).

    Each feed runs in isolation: an exception marks only this feed failed.
    The US polled-id frame is computed here inside the same guard.
    """
    country = SOURCES[fid].country
    directory = out / "feeds" / fid
    responses: list[RawResponse] = []
    result: FetchResult | None = None
    normalized = None
    us_id_set: pl.DataFrame | None = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        source = SOURCES[fid]
        extra: list[Warning] = []
        # Per-feed budget override from feeds.toml.
        declared = (getattr(ctx.fetch_config, "feeds", None) or {}).get(fid)
        seconds = getattr(declared, "budget_s", None)
        try:
            with client.budget(f"country-{country}", seconds, key="country"):
                responses = source.fetch(client, ctx)
        except BudgetExceeded:
            # A source that catches this itself returns its partial responses;
            # catching it here covers the ones that do not.
            extra.append(Warning(code="deadline_exceeded", detail=fid))
        for resp in responses:
            write_response(directory / "responses", resp, resp.key.split("/")[-1])
        result = source.parse(responses, ctx)
        result.warnings.extend(extra)
        normalized = normalize(result, fx, ctx)
        if fid == "US-COSTCO":
            us_id_set = us_source.polled_id_frame(ctx)
    except Exception as exc:  # isolation is the point
        detail = f"{type(exc).__name__}: {exc}"
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "traceback.txt").write_text(traceback.format_exc(), encoding="utf-8")
        result = FetchResult(
            country=country,
            source="error",
            captured_at_utc=now,
            stations=[],
            responses=list(responses),
            requests=len(responses),
            warnings=[],
            errors=[Error(code="exception", host=None, http_status=None, detail=detail)],
        )
        normalized = None
        us_id_set = None

    if country in PREVIOUS_STATE_COUNTRIES and not ctx.previous_state_complete:
        # Repeated on the feed's own warnings (the capture-level one isn't visible
        # to evaluate_feed). Appended after try/except to survive the failure path.
        result.warnings.append(Warning(code="previous_state_unavailable"))

    block = checks.evaluate_feed(fid, result, normalized, ctx, now)

    rows = normalized.rows if normalized else pl.DataFrame(schema=schema.ROW_SCHEMA)
    stations = normalized.stations if normalized else pl.DataFrame(schema=schema.STATION_SCHEMA)
    drops = normalized.drops if normalized else []
    schema.write_rows_csv_gz(rows, directory / "rows.csv.gz")
    stations.write_csv(directory / "stations.csv")
    pl.DataFrame(
        {
            "source_station_id": [d.source_station_id for d in drops],
            "reason": [d.reason for d in drops],
            "detail": [d.detail for d in drops],
        },
        schema=DROP_SCHEMA,
    ).write_csv(directory / "drops.csv")
    (directory / "country.json").write_text(
        json.dumps(block, indent=2, sort_keys=True), encoding="utf-8"
    )
    return {
        "block": block,
        "rows": rows,
        "stations": stations,
        "result": result,
        "us_id_set": us_id_set,
    }


def _run_feeds(
    feeds: list[str],
    client: Client,
    ctx: CaptureContext,
    fx: FxRates,
    out: Path,
    now: datetime,
    warnings: list[Warning],
) -> tuple[dict[str, dict], dict[str, dict]]:
    blocks: dict[str, dict] = {}
    collected: dict[str, dict] = {}
    if not feeds:
        return blocks, collected
    try:
        with (
            client.budget("capture"),
            ThreadPoolExecutor(max_workers=max(1, len(feeds))) as pool,
        ):
            futures = {pool.submit(run_feed, fid, client, ctx, fx, out, now): fid for fid in feeds}
            for future in as_completed(futures):
                fid = futures[future]
                collected[fid] = future.result()
                blocks[fid] = collected[fid]["block"]
    except BudgetExceeded:
        if not any(item.code == "deadline_exceeded" for item in warnings):
            warnings.append(Warning(code="deadline_exceeded"))
    return blocks, collected


def _us_id_set(collected: dict[str, dict]) -> pl.DataFrame:
    """The set of US ids that were polled (not just priced), for discovery.

    Reads the frame computed in `run_feed`'s guarded path — must never recompute.
    """
    us = collected.get("US-COSTCO")
    frame = us.get("us_id_set") if us else None
    if frame is None:
        return pl.DataFrame(schema=US_ID_SET_SCHEMA)
    return frame


def _write_bundle(
    out: Path,
    capture_out: Path,
    ctx: CaptureContext,
    run: dict,
    collected: dict[str, dict],
) -> Path:
    """Build `capture/bundle.tar.gz` with the spec 8.2 contents."""
    stage = out / "bundle"
    if stage.exists():
        shutil.rmtree(stage)
    (stage / "inputs").mkdir(parents=True)
    (stage / "config").mkdir(parents=True)

    config_sha: dict[str, str] = {}
    for path in sorted(CONFIG_DIR.glob("*")):
        if path.is_file():
            shutil.copy2(path, stage / "config" / path.name)
            config_sha[path.name] = sha256_file(path)

    # Written to both the bundle and the capture directory (publish reads it
    # from the latter).
    capture_meta = json.dumps(
        {
            "capture_id": ctx.capture_id,
            "capture_date": ctx.capture_date.isoformat(),
            "started_at_utc": run["started_at_utc"],
            "git_sha": run["git_sha"],
            "run_id": run["run_id"],
            "config_sha256": config_sha,
        },
        indent=2,
        sort_keys=True,
    )
    (stage / "capture.json").write_text(capture_meta, encoding="utf-8")
    (out / "capture" / "capture.json").write_text(capture_meta, encoding="utf-8")
    ctx.previous_stations.write_csv(stage / "inputs" / "stations_used.csv")
    ctx.previous_fx.write_csv(stage / "inputs" / "fx_used.csv")
    (stage / "inputs" / "status_previous.json").write_text(
        json.dumps(ctx.previous_status, indent=2, sort_keys=True), encoding="utf-8"
    )
    _us_id_set(collected).write_csv(stage / "inputs" / "us_id_set.csv")

    countries_dir = out / "feeds"
    if countries_dir.is_dir():
        for country_dir in sorted(countries_dir.iterdir()):
            responses = country_dir / "responses"
            if responses.is_dir():
                shutil.copytree(responses, stage / "responses" / country_dir.name)
    shared = out / "shared"
    if shared.is_dir():
        shutil.copytree(shared, stage / "responses" / "shared")

    for name in ("fx.json", "status.json", "rows.csv.gz", "stations.csv"):
        shutil.copy2(capture_out / name, stage / name)

    bundle = capture_out / "bundle.tar.gz"
    with tarfile.open(bundle, "w:gz") as archive:
        for path in sorted(stage.rglob("*")):
            archive.add(path, arcname=str(path.relative_to(stage)), recursive=False)
    return bundle
