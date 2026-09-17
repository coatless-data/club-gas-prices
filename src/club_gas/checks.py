"""Per-country status, counters and the capture status document (spec 6.5)."""

from __future__ import annotations

import copy
import hashlib
from dataclasses import asdict, is_dataclass
from datetime import UTC, datetime, timedelta

import polars as pl

from club_gas.fx import FxRates
from club_gas.normalize import NormalizedCountry
from club_gas.sources.base import SOURCES, CaptureContext, FetchResult

SCHEMA_VERSION = 1
CAPTURE_ID_FORMAT = "%Y-%m-%dT%H%MZ"

# Rows carrying one of these sources came from a country's fallback path.
FALLBACK_SOURCES = {"costco-ca-lookup-us", "costco-ca-gasprices"}

# Warning codes that make a country degraded on their own (spec 6.5).
DEGRADING_WARNINGS = {
    "metadata_from_cache",
    "previous_state_unavailable",
    "fallback_used",
    "unknown_grade",
}

OUT_OF_BOUNDS_SHARE = 0.05
MAX_RECENT_ERRORS = 3


def parse_capture_id(capture_id: str) -> datetime:
    return datetime.strptime(capture_id, CAPTURE_ID_FORMAT).replace(tzinfo=UTC)


def price_fingerprint(rows: pl.DataFrame) -> str:
    """SHA-256 of the sorted (station_key, grade_raw, price_raw) tuples."""
    tuples = sorted(
        zip(
            rows["station_key"].to_list(),
            rows["grade_raw"].to_list(),
            rows["price_raw"].to_list(),
            strict=True,
        )
    )
    digest = hashlib.sha256()
    for station_key, grade_raw, price_raw in tuples:
        digest.update(f"{station_key}\t{grade_raw}\t{price_raw}\n".encode())
    return "sha256:" + digest.hexdigest()


def all_failed(status: dict) -> bool:
    """Spec 5.3: true when no country is ok or degraded. Skipped is not success."""
    blocks = (status.get("countries") or {}).values()
    return not any(block.get("status") in ("ok", "degraded") for block in blocks)


def _as_dict(item) -> dict:
    return asdict(item) if is_dataclass(item) else dict(item)


def _iso(value) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _previous_block(ctx: CaptureContext, fid: str) -> dict:
    previous = ctx.previous_status or {}
    return (previous.get("feeds") or {}).get(fid) or {}


def _carried(previous: dict) -> dict:
    return {
        "price_fingerprint": previous.get("price_fingerprint"),
        "unchanged_since_capture_id": previous.get("unchanged_since_capture_id"),
        "consecutive_failures": int(previous.get("consecutive_failures") or 0),
        "last_success_capture_id": previous.get("last_success_capture_id"),
        "recent_errors": copy.deepcopy(previous.get("recent_errors") or []),
    }


def _duration_s(result: FetchResult) -> float:
    """Wall clock from the first request leaving to the last response arriving."""
    if not result.responses:
        return 0.0
    starts = [
        response.received_at_utc - timedelta(milliseconds=response.elapsed_ms)
        for response in result.responses
    ]
    ends = [response.received_at_utc for response in result.responses]
    return round((max(ends) - min(starts)).total_seconds(), 1)


def _is_stale(unchanged_since: str | None, now: datetime, stale_after_days: int) -> bool:
    if not unchanged_since:
        return False
    try:
        since = parse_capture_id(unchanged_since)
    except ValueError:
        return False
    return now - since > timedelta(days=stale_after_days)


def evaluate_feed(
    fid: str,
    result: FetchResult | None,
    normalized: NormalizedCountry | None,
    ctx: CaptureContext,
    now: datetime,
) -> dict:
    """The spec 6.5 per-feed block.

    `result is None` means the feed was not selected by --countries, which is
    `skipped`. A feed whose fetch raised is passed a FetchResult carrying the
    error and no stations, which makes it `failed`.

    The unit is the feed, not the country: two chains in one country have their
    own floors, their own fingerprints and their own failure streaks, and a
    block saying only "US degraded" would not say which chain to go and look at.
    """
    country = SOURCES[fid].country if fid in SOURCES else fid
    previous = _previous_block(ctx, fid)

    if result is None:
        block = {
            "status": "skipped",
            "source": previous.get("source"),
            "stations": 0,
            "rows": 0,
            "requests": 0,
            "duration_s": 0.0,
            "dropped": {},
            "warnings": [],
            "errors": [],
        }
        block.update(_carried(previous))
        return block

    warnings = [_as_dict(item) for item in result.warnings]
    errors = [_as_dict(item) for item in result.errors]
    dropped: dict[str, int] = {}
    rows = None
    if normalized is not None:
        warnings += [_as_dict(item) for item in normalized.warnings]
        for drop in normalized.drops:
            dropped[drop.reason] = dropped.get(drop.reason, 0) + 1
        rows = normalized.rows

    n_rows = 0 if rows is None else rows.height
    n_stations = 0 if rows is None or n_rows == 0 else rows["station_key"].n_unique()

    block = {
        "status": "failed",
        "source": result.source,
        "stations": n_stations,
        "rows": n_rows,
        "requests": result.requests,
        "duration_s": _duration_s(result),
        "dropped": dropped,
        "warnings": warnings,
        "errors": errors,
    }

    if n_rows == 0:
        block.update(_carried(previous))
        block["consecutive_failures"] = int(previous.get("consecutive_failures") or 0) + 1
        recent = [
            {"capture_id": ctx.capture_id, "run_url": None, "errors": errors},
            *block["recent_errors"],
        ]
        block["recent_errors"] = recent[:MAX_RECENT_ERRORS]
        return block

    fingerprint = price_fingerprint(rows)
    if previous.get("price_fingerprint") == fingerprint and previous.get(
        "unchanged_since_capture_id"
    ):
        unchanged_since = previous["unchanged_since_capture_id"]
    else:
        unchanged_since = ctx.capture_id

    block["price_fingerprint"] = fingerprint
    block["unchanged_since_capture_id"] = unchanged_since
    block["consecutive_failures"] = 0
    block["last_success_capture_id"] = ctx.capture_id
    block["recent_errors"] = copy.deepcopy(previous.get("recent_errors") or [])

    country_cfg = ctx.interp_config.countries[country]
    out_of_bounds = dropped.get("out_of_bounds", 0)
    parsed = n_rows + out_of_bounds
    warning_codes = {item["code"] for item in warnings}
    sources = set(rows["source"].unique().to_list())

    degraded = (
        n_stations < country_cfg.floor
        or bool(sources & FALLBACK_SOURCES)
        or bool(warning_codes & DEGRADING_WARNINGS)
        or (parsed > 0 and out_of_bounds / parsed > OUT_OF_BOUNDS_SHARE)
        or _is_stale(unchanged_since, now, country_cfg.stale_after_days)
    )
    block["status"] = "degraded" if degraded else "ok"
    return block


# Worst-first, so a country reads as badly as its unhappiest feed.
_STATUS_RANK = {"failed": 0, "degraded": 1, "ok": 2, "skipped": 3}


def _country_rollup(feeds: dict[str, dict]) -> dict:
    """One block per country, derived from its feeds.

    The dashboard's freshness notice and the alert bodies are country-shaped,
    and a reader asking "is Mexico current?" does not care that the United
    States happens to have two chains. The per-feed blocks stay authoritative.
    """
    out: dict[str, dict] = {}
    for fid, block in feeds.items():
        country = SOURCES[fid].country if fid in SOURCES else fid
        roll = out.setdefault(
            country,
            {
                "status": "skipped",
                "rows": 0,
                "stations": 0,
                "feeds": [],
                "last_success_capture_id": None,
            },
        )
        roll["feeds"].append(fid)
        roll["rows"] += int(block.get("rows") or 0)
        roll["stations"] += int(block.get("stations") or 0)
        if _STATUS_RANK.get(block.get("status"), 3) < _STATUS_RANK.get(roll["status"], 3):
            roll["status"] = block.get("status")
        newest = block.get("last_success_capture_id")
        if newest and (roll["last_success_capture_id"] or "") < newest:
            roll["last_success_capture_id"] = newest
    for roll in out.values():
        roll["feeds"].sort()
    return out


def build_status(
    ctx: CaptureContext,
    countries: dict[str, dict],
    fx: FxRates,
    ecom_api: dict,
    now: datetime,
    run: dict,
) -> dict:
    """Assemble status.json. capture writes everything except publish.outcome
    and close.outcome, which alerts fills in later (spec 6.5)."""
    previous = ctx.previous_status or {}
    previous_publish = previous.get("publish") or {}

    blocks = copy.deepcopy(dict(countries))
    for fid in SOURCES:
        if fid not in blocks:
            blocks[fid] = evaluate_feed(fid, None, None, ctx, now)

    run_url = run.get("run_url")
    for block in blocks.values():
        for entry in block.get("recent_errors") or []:
            if entry.get("capture_id") == ctx.capture_id and not entry.get("run_url"):
                entry["run_url"] = run_url

    fx_row = fx.rows[0] if fx.rows else None
    return {
        "schema_version": SCHEMA_VERSION,
        "capture_id": ctx.capture_id,
        "run_id": run.get("run_id"),
        "run_attempt": run.get("run_attempt"),
        "run_url": run_url,
        "started_at_utc": _iso(run.get("started_at_utc")),
        "finished_at_utc": _iso(now),
        "git_sha": run.get("git_sha"),
        "fx": {
            "status": fx.status,
            "source": None if fx_row is None else fx_row.fx_source,
            "rate_date": None if fx_row is None else fx_row.fx_rate_date.isoformat(),
        },
        "ecom_api": dict(ecom_api),
        "publish": {
            "outcome": None,
            "consecutive_failures": int(previous_publish.get("consecutive_failures") or 0),
            "unpublished": copy.deepcopy(previous_publish.get("unpublished") or []),
        },
        "close": {"outcome": None},
        "warnings": [_as_dict(item) for item in run.get("warnings") or []],
        "feeds": blocks,
        "countries": _country_rollup(blocks),
    }
