"""Canada: the costco.ca warehouse lookup, with the price service as fallback."""

from __future__ import annotations

import json
import re
from datetime import date, datetime, timedelta, timezone
from typing import Any
from urllib.parse import urlencode

import polars as pl

from ..config import CountryConfig
from .base import CaptureContext, RawPrice, RawStation

LOOKUP_SOURCE = "costco-ca-lookup"
PRICE_SOURCE = "costco-ca-gasprices"
LOOKUP_KEY = "CA/01-lookup"
PRICE_KEY_PREFIX = "CA/02-gasprices-"

# config/countries.toml supplies these three as fields of CountryConfig
# (fallback_url, batch_size, seen_within_days). The constants below are the same
# values, used only when a Config view has blanked the field to None.
PRICE_SERVICE_URL = "https://www.costco.ca/AjaxGetGasPricesService"
DEFAULT_BATCH_SIZE = 10
DEFAULT_SEEN_WITHIN_DAYS = 30

# Every key of gasPrices except these two is a grade label (§4.3).
NON_GRADE_KEYS = frozenset({"warehouseid", "oid"})

MONTHS = {
    name: number
    for number, name in enumerate(
        ("jan", "feb", "mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"),
        start=1,
    )
}
OPEN_DATE = re.compile(r"^\s*([A-Za-z]{3})[a-z]*\.?\s+(\d{1,2}),?\s+(\d{4})\s*$")


def lookup_url(cc: CountryConfig) -> str:
    """The lookup endpoint with its query string.

    ``cc.params`` holds query parameters and nothing else, so it is urlencoded
    whole: hasGas, populateWarehouseDetails and countryCode.
    """
    return f"{cc.url}?{urlencode(cc.params)}"


def batch_url(cc: CountryConfig, ids: list[str]) -> str:
    """One price-service request for up to ``cc.batch_size`` warehouse ids."""
    base = cc.fallback_url or PRICE_SERVICE_URL
    return f"{base}?warehouseid={'_'.join(ids)}"


def parse_open_date(value: Any) -> date | None:
    """Parse "Aug 23, 1995".

    ``strptime("%b")`` reads month names from the process locale, so a runner
    with a non-English LC_TIME would silently fail; this table does not.
    """
    if not isinstance(value, str):
        return None
    match = OPEN_DATE.match(value)
    if match is None:
        return None
    month = MONTHS.get(match.group(1).lower())
    if month is None:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(2)))
    except ValueError:
        return None


def parse_lookup_body(body: bytes) -> list[dict[str, Any]]:
    """Warehouse objects from the lookup body.

    The body starts with a CRLF and its element 0 is the boolean ``false``;
    the warehouses are elements 1..n. Raises ValueError when the body is not
    the expected JSON array, which is one of the fallback triggers. The
    response's Content-Type is ``text/html`` even when the body is valid JSON,
    so it can never be used to tell a block from a good answer.
    """
    payload = json.loads(body.decode("utf-8", errors="strict"))
    if not isinstance(payload, list):
        raise ValueError("lookup body is not a JSON array")
    return [e for e in payload if isinstance(e, dict) and "stlocID" in e]


def source_filter_reason(station: RawStation, capture_date: date) -> str | None:
    """The §7.0 step-1 source filter that removes this station, if any.

    Nothing here deletes a station: ``normalize()`` applies these reasons and
    counts them into status.json, and this module only needs the verdict to
    decide whether the lookup produced anything usable.
    """
    if station.opening_date is not None and station.opening_date > capture_date:
        return "not_open"
    if station.has_hours is False:
        return "no_hours"
    if not station.prices:
        return "no_price"
    if not station.timezone:
        return "no_timezone"
    return None


def passes_source_filters(station: RawStation, capture_date: date) -> bool:
    return source_filter_reason(station, capture_date) is None


def cached_stations(
    previous: pl.DataFrame,
    capture_date: date,
    seen_within_days: int = DEFAULT_SEEN_WITHIN_DAYS,
) -> dict[str, dict[str, Any]]:
    """CA rows of stations.csv last seen inside the window, keyed by station id.

    Callers pass ``cc.seen_within_days`` (30 for CA in config/countries.toml);
    the default matches it so a blanked field cannot change the window.
    """
    if previous is None or previous.is_empty() or "station_key" not in previous.columns:
        return {}
    cutoff = capture_date - timedelta(days=seen_within_days)
    out: dict[str, dict[str, Any]] = {}
    for row in previous.filter(pl.col("country") == "CA").to_dicts():
        last_seen = _as_date(row.get("last_seen_utc"))
        station_id = row.get("source_station_id")
        if station_id is None or last_seen is None or last_seen < cutoff:
            continue
        out[str(station_id)] = row
    return out


def _prices(grades: Any) -> tuple[RawPrice, ...]:
    if not isinstance(grades, dict):
        return ()
    return tuple(
        RawPrice(grade_raw=key, price_raw=value)
        for key, value in grades.items()
        if key not in NON_GRADE_KEYS and isinstance(value, str)
    )


def _regular_in_bounds(station: RawStation, cc: CountryConfig) -> bool:
    """True when the station quotes a regular price inside CA's sanity bounds.

    ``cc.bounds`` is keyed by price unit and every value is a ``Bounds``, so the
    lookup is always ``cc.bounds[cc.price_unit].for_grade(grade_raw)``. CA has no
    per-grade override, so ``for_grade("regular")`` returns (1.0, 3.5) CAD/L.
    """
    if cc.price_unit not in (cc.bounds or {}):
        return False
    low, high = cc.bounds[cc.price_unit].for_grade("regular")
    for price in station.prices:
        if price.grade_raw.lower() != "regular":
            continue
        value = _price_value(price.price_raw)
        return value is not None and low <= value <= high
    return False


def _price_value(raw: str) -> float | None:
    text = re.sub(r"[^0-9.]", "", raw or "")
    if text.startswith("."):
        text = "0" + text
    try:
        return float(text)
    except ValueError:
        return None


def _clean(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return value.strip() or None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str) and len(value) >= 10:
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _id_sort_key(station_id: str) -> tuple[int, Any]:
    return (0, int(station_id)) if station_id.isdigit() else (1, station_id)


def _capture_time(ctx: CaptureContext) -> datetime:
    return datetime.strptime(ctx.capture_id, "%Y-%m-%dT%H%MZ").replace(tzinfo=timezone.utc)
