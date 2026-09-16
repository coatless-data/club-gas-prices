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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import polars as pl

from . import checks, schema
from .config import Config
from .fx import FxRates, fetch_rates
from .http import BudgetExceeded, Client
from .normalize import normalize
from .sources import us as us_source
from .sources.base import SOURCES, CaptureContext, RawResponse, Warning
from .store import ReleaseStore, sha256_file

# Paths are relative to the repository checkout the CLI runs in (capture.yml
# checks out the default branch, then runs from its root).
CONFIG_DIR = Path("config")
STATUS_PATH = Path("status/latest.json")

# The warehouse locator is shared by US and CA, so capture fetches it once
# (spec 4.2 step 1, 5.3 step 3). The client-identifier is a public id embedded
# in Costco's own front end; it lives in config/countries.toml under
# [countries.US] so it can be rotated without a code change, and these
# constants are only the built-in default.
ECOM_API_URL = "https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json"
ECOM_API_PARAMS = {"latitude": "0", "longitude": "0", "limit": "5000"}
ECOM_CLIENT_IDENTIFIER = "7c71124c-7bf1-44db-bc9d-498584cd66e5"

ECOM_BUDGET_S = 90.0
COUNTRY_BUDGET_S = 480.0
CAPTURE_BUDGET_S = 720.0

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


@dataclass
class CaptureResult:
    status: dict
    all_failed: bool
    capture_id: str
    out: Path


def capture_id_for(now: datetime) -> str:
    """The capture id: UTC start time truncated to the minute (spec 6.1)."""
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H%MZ")


def github_output(key: str, value: str) -> None:
    """Append a step output when running under Actions; a no-op otherwise."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


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

    `read_resolved` is used instead of `download` because capture runs before
    the §8.4 recovery that publish performs: a crashed publish can leave only a
    `stations.csv.next-<token>-1` behind, and read_resolved resolves it without
    modifying the release.
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
    identifier = (
        getattr(country, "ecom_client_identifier", None) or ECOM_CLIENT_IDENTIFIER
    )
    url = f"{base}?{urlencode(query)}" if query else base
    return url, identifier


def _fetch_ecom(client: Client, cfg: Config) -> tuple[RawResponse | None, dict]:
    """Fetch the shared warehouse locator once (spec 5.3 step 3)."""
    url, identifier = _ecom_request(cfg)
    try:
        with client.budget("ecom-api", ECOM_BUDGET_S):
            resp = client.request(
                "shared/ecom-api", url, headers={"client-identifier": identifier}
            )
    except BudgetExceeded:
        return None, {"attempted": True, "http_status": None}
    return resp, {"attempted": True, "http_status": resp.status}


def _env_int(name: str) -> int | None:
    raw = os.environ.get(name)
    try:
        return int(raw) if raw else None
    except ValueError:
        return None


def _run_url() -> str | None:
    server = os.environ.get("GITHUB_SERVER_URL") or "https://github.com"
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not repo or not run_id:
        return None
    return f"{server}/{repo}/actions/runs/{run_id}"


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
    started = now.astimezone(timezone.utc)
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

    blocks, collected = _run_countries(countries, client, ctx, fx, out, now, warnings)

    rows = _concat(collected, "rows", schema.ROW_SCHEMA).sort(
        ["capture_id", "country", "station_key", "grade_raw"]
    )
    stations = _concat(collected, "stations", schema.STATION_SCHEMA).sort("station_key")
    schema.write_rows_csv_gz(rows, capture_out / "rows.csv.gz")
    stations.write_csv(capture_out / "stations.csv")
    # fx.json is a JSON array of rate rows (spec 4.5). `publish` and `rebuild`
    # both read that array, so the writer is FxRates.to_json() and nothing else.
    (capture_out / "fx.json").write_text(
        json.dumps(fx.to_json(), indent=1, sort_keys=True), encoding="utf-8"
    )

    finished = started + timedelta(seconds=time.monotonic() - monotonic_start)
    run = {
        "run_id": _env_int("GITHUB_RUN_ID"),
        "run_attempt": _env_int("GITHUB_RUN_ATTEMPT"),
        "run_url": _run_url(),
        "git_sha": os.environ.get("GITHUB_SHA"),
        "started_at_utc": _utc_text(started),
        "finished_at_utc": _utc_text(finished),
        "warnings": warnings,
    }
    status = checks.build_status(ctx, blocks, fx, ecom_api, now, run)
    (capture_out / "status.json").write_text(
        json.dumps(status, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    _write_bundle(out, capture_out, ctx, run, collected)

    all_failed = not any(
        block.get("status") in ("ok", "degraded")
        for block in status.get("countries", {}).values()
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


def run_country(
    country: str,
    client: Client,
    ctx: CaptureContext,
    fx: FxRates,
    out: Path,
    now: datetime,
) -> dict:
    """Fetch, parse, normalize, check and write one country (spec 5.3 step 4).

    Everything is inside one try block so an exception marks only this country
    failed (spec 10.1), and the outputs land in `out/countries/<CC>/` as soon
    as the country finishes, before the other threads are done.
    """
    directory = out / "countries" / country
    directory.mkdir(parents=True, exist_ok=True)
    source = SOURCES[country]
    responses: list[RawResponse] = []
    result = None
    normalized = None
    failure: str | None = None
    try:
        extra: list[Warning] = []
        try:
            with client.budget(f"country-{country}", COUNTRY_BUDGET_S):
                responses = source.fetch(client, ctx)
        except BudgetExceeded:
            # A source that catches this itself returns its partial responses;
            # catching it here covers the ones that do not.
            extra.append(Warning(code="deadline_exceeded", detail=country))
        for resp in responses:
            write_response(directory / "responses", resp, resp.key.split("/")[-1])
        result = source.parse(responses, ctx)
        result.warnings.extend(extra)
        normalized = normalize(result, fx, ctx)
    except Exception as exc:  # noqa: BLE001 - isolation is the point
        failure = f"{type(exc).__name__}: {exc}"
        (directory / "traceback.txt").write_text(
            traceback.format_exc(), encoding="utf-8"
        )
        result = None
        normalized = None

    block = checks.evaluate_country(country, result, normalized, ctx, now)
    if failure is not None:
        block.setdefault("errors", []).append(
            {"code": "exception", "host": None, "http_status": None, "detail": failure}
        )

    rows = normalized.rows if normalized else pl.DataFrame(schema=schema.ROW_SCHEMA)
    stations = (
        normalized.stations if normalized else pl.DataFrame(schema=schema.STATION_SCHEMA)
    )
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
    return {"block": block, "rows": rows, "stations": stations, "result": result}


def _run_countries(
    countries: list[str],
    client: Client,
    ctx: CaptureContext,
    fx: FxRates,
    out: Path,
    now: datetime,
    warnings: list[Warning],
) -> tuple[dict[str, dict], dict[str, dict]]:
    blocks: dict[str, dict] = {}
    collected: dict[str, dict] = {}
    if not countries:
        return blocks, collected
    try:
        with client.budget("capture", CAPTURE_BUDGET_S):
            with ThreadPoolExecutor(max_workers=len(countries)) as pool:
                futures = {
                    pool.submit(run_country, cc, client, ctx, fx, out, now): cc
                    for cc in countries
                }
                for future in as_completed(futures):
                    country = futures[future]
                    collected[country] = future.result()
                    blocks[country] = collected[country]["block"]
    except BudgetExceeded:
        if not any(item.code == "deadline_exceeded" for item in warnings):
            warnings.append(Warning(code="deadline_exceeded"))
    return blocks, collected


def _us_id_set(ctx: CaptureContext, captured: list[str]) -> pl.DataFrame:
    """``inputs/us_id_set.csv``: every id the US fetcher POLLED, not every id it priced.

    `discover` (spec 10.4) subtracts this file from its candidate range, so it has to
    hold the polled set. Deriving it from `FetchResult.stations` instead would omit
    every id that answered `{}`, and discovery would re-sweep those ids every month.
    """
    if "US" not in captured:
        return pl.DataFrame(schema=US_ID_SET_SCHEMA)
    return us_source.polled_id_frame(ctx)


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

    (stage / "capture.json").write_text(
        json.dumps(
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
        ),
        encoding="utf-8",
    )
    ctx.previous_stations.write_csv(stage / "inputs" / "stations_used.csv")
    ctx.previous_fx.write_csv(stage / "inputs" / "fx_used.csv")
    (stage / "inputs" / "status_previous.json").write_text(
        json.dumps(ctx.previous_status, indent=2, sort_keys=True), encoding="utf-8"
    )
    _us_id_set(ctx, list(collected)).write_csv(stage / "inputs" / "us_id_set.csv")

    countries_dir = out / "countries"
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
