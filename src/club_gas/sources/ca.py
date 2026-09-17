"""Canada: the costco.ca warehouse lookup, with the price service as fallback."""

from __future__ import annotations

import json
import re
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import urlencode, urlsplit

import polars as pl

from club_gas.sources import costco_lookup

from ..config import CountryConfig
from ..http import BudgetExceeded, Client
from .base import (
    CaptureContext,
    Error,
    FetchResult,
    RawPrice,
    RawResponse,
    RawStation,
    Warning,
)

LOOKUP_SOURCE = "costco-ca-lookup"
PRICE_SOURCE = "costco-ca-gasprices"
FEED = "CA-COSTCO"
LOOKUP_KEY = f"{FEED}/01-lookup"
PRICE_KEY_PREFIX = f"{FEED}/02-gasprices-"

# config/countries.toml supplies these three as fields of CountryConfig
# (fallback_url, batch_size, seen_within_days). The constants below are the same
# values, used only when a Config view has blanked the field to None.
PRICE_SERVICE_URL = "https://www.costco.ca/AjaxGetGasPricesService"
DEFAULT_BATCH_SIZE = 10
DEFAULT_SEEN_WITHIN_DAYS = 30

# Every key of gasPrices except these two is a grade label (§4.3).
NON_GRADE_KEYS = costco_lookup.NON_GRADE_KEYS

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
    """Parse "Aug 23, 1995", in any casing."""
    return costco_lookup.open_date(value)


def parse_lookup_body(body: bytes) -> list[dict[str, Any]]:
    """Warehouse objects from the lookup body."""
    return costco_lookup.warehouses(body)


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


def fallback_trigger(response: RawResponse | None, ctx: CaptureContext) -> str | None:
    """Why the price-service fallback is needed, or None.

    Recomputed from the stored responses in parse(), so a rebuild years later
    reaches the same conclusion the live capture did.
    """
    if response is None:
        return "forced"
    if response.error or response.status != 200:
        return "request_failed"
    try:
        entries = parse_lookup_body(response.body)
    except (ValueError, UnicodeDecodeError):
        return "unparseable_body"
    cc = ctx.interp_config.countries["CA"]
    ecom = ecom_index(ctx.shared)
    cached = cached_stations(
        ctx.previous_stations,
        ctx.capture_date,
        cc.seen_within_days or DEFAULT_SEEN_WITHIN_DAYS,
    )
    for entry in entries:
        station, _ = _lookup_station(entry, ecom, cached, cc)
        if passes_source_filters(station, ctx.capture_date):
            return None
    return "zero_stations"


class CaSource:
    country = "CA"

    def fetch(self, client: Client, ctx: CaptureContext) -> list[RawResponse]:
        cc = ctx.fetch_config.countries["CA"]
        responses: list[RawResponse] = []
        url = lookup_url(cc)
        trigger = "forced" if "CA" in ctx.force_fallback else None

        if trigger is None:
            if client.abandoned(url):
                return responses
            try:
                response = client.request(LOOKUP_KEY, url, expect_json=True)
            except BudgetExceeded:
                return responses
            responses.append(response)
            trigger = fallback_trigger(response, ctx)
            if trigger is None:
                return responses

        # batch_size and seen_within_days are country settings, not query
        # parameters: config/countries.toml gives CA 10 and 30.
        batch_size = cc.batch_size or DEFAULT_BATCH_SIZE
        ids = sorted(
            cached_stations(
                ctx.previous_stations,
                ctx.capture_date,
                cc.seen_within_days or DEFAULT_SEEN_WITHIN_DAYS,
            ),
            key=_id_sort_key,
        )
        for number, start in enumerate(range(0, len(ids), batch_size), start=1):
            batch = batch_url(cc, ids[start : start + batch_size])
            if client.abandoned(batch):
                break
            try:
                responses.append(
                    client.request(f"{PRICE_KEY_PREFIX}{number:02d}", batch, expect_json=True)
                )
            except BudgetExceeded:
                break
        return responses

    def parse(self, responses: list[RawResponse], ctx: CaptureContext) -> FetchResult:
        lookup = next((r for r in responses if r.key == LOOKUP_KEY), None)
        prices = sorted(
            (r for r in responses if r.key.startswith(PRICE_KEY_PREFIX)), key=lambda r: r.key
        )
        if prices:
            return self._parse_fallback(lookup, prices, responses, ctx)
        return self._parse_lookup(lookup, responses, ctx)

    def _parse_lookup(
        self, lookup: RawResponse | None, responses: list[RawResponse], ctx: CaptureContext
    ) -> FetchResult:
        cc = ctx.interp_config.countries["CA"]
        warnings: list[Warning] = []
        errors: list[Error] = []
        stations: list[RawStation] = []
        captured = lookup.received_at_utc if lookup else _capture_time(ctx)

        if lookup is None:
            errors.append(
                Error(
                    code="host_abandoned",
                    host=urlsplit(lookup_url(ctx.fetch_config.countries["CA"])).netloc,
                    detail="no lookup response",
                )
            )
        elif lookup.error or lookup.status != 200:
            errors.append(
                Error(
                    code="request_failed" if lookup.status is None else "http_error",
                    host=urlsplit(lookup.url).netloc,
                    http_status=lookup.status,
                    detail=lookup.error,
                )
            )
        else:
            try:
                entries = parse_lookup_body(lookup.body)
            except (ValueError, UnicodeDecodeError) as exc:
                errors.append(
                    Error(
                        code="unparseable_body",
                        host=urlsplit(lookup.url).netloc,
                        http_status=lookup.status,
                        detail=str(exc),
                    )
                )
                entries = []
            ecom = ecom_index(ctx.shared)
            cached = cached_stations(
                ctx.previous_stations,
                ctx.capture_date,
                cc.seen_within_days or DEFAULT_SEEN_WITHIN_DAYS,
            )
            for entry in entries:
                station, tz_origin = _lookup_station(entry, ecom, cached, cc)
                if tz_origin == "region":
                    warnings.append(
                        Warning(code="timezone_from_region", detail=station.source_station_id)
                    )
                # `priced_before_open` is normalize()'s to emit, not this
                # module's: it applies the same rule to the same stations, and
                # checks.evaluate_country concatenates both warning lists, so
                # emitting here double-counted every affected station.
                stations.append(station)

        return FetchResult(
            country="CA",
            source=LOOKUP_SOURCE,
            captured_at_utc=captured,
            stations=stations,
            responses=list(responses),
            requests=len([r for r in responses if r.key.startswith(f"{FEED}/")]),
            warnings=warnings,
            errors=errors,
        )

    def _parse_fallback(
        self,
        lookup: RawResponse | None,
        prices: list[RawResponse],
        responses: list[RawResponse],
        ctx: CaptureContext,
    ) -> FetchResult:
        cc = ctx.interp_config.countries["CA"]
        warnings = [
            Warning(code="fallback_used", detail=fallback_trigger(lookup, ctx) or "forced"),
            Warning(code="metadata_from_cache"),
        ]
        errors: list[Error] = []
        stations: list[RawStation] = []
        cached = cached_stations(
            ctx.previous_stations,
            ctx.capture_date,
            cc.seen_within_days or DEFAULT_SEEN_WITHIN_DAYS,
        )
        captured: datetime | None = None

        for response in prices:
            if response.error or response.status != 200:
                errors.append(
                    Error(
                        code="request_failed" if response.status is None else "http_error",
                        host=urlsplit(response.url).netloc,
                        http_status=response.status,
                        detail=response.error,
                    )
                )
                continue
            try:
                payload = json.loads(response.body)
            except (ValueError, UnicodeDecodeError):
                errors.append(
                    Error(
                        code="unparseable_body",
                        host=urlsplit(response.url).netloc,
                        http_status=response.status,
                        detail=response.key,
                    )
                )
                continue
            if not isinstance(payload, dict) or "errorMessage" in payload:
                # A non-numeric id fails the whole request with HTTP 200 and an
                # errorMessage object instead of prices.
                errors.append(
                    Error(
                        code="price_service_error",
                        host=urlsplit(response.url).netloc,
                        http_status=response.status,
                        detail=str(payload)[:200],
                    )
                )
                continue
            captured = (
                response.received_at_utc
                if captured is None
                else max(captured, response.received_at_utc)
            )
            for station_id, grades in payload.items():
                row = cached.get(str(station_id))
                if row is None:
                    warnings.append(Warning(code="no_metadata", detail=str(station_id)))
                    continue
                station, tz_origin = _cached_station(str(station_id), row, grades, cc)
                if tz_origin == "region":
                    warnings.append(
                        Warning(code="timezone_from_region", detail=station.source_station_id)
                    )
                stations.append(station)

        return FetchResult(
            country="CA",
            source=PRICE_SOURCE,
            captured_at_utc=captured or _capture_time(ctx),
            stations=stations,
            responses=list(responses),
            requests=len([r for r in responses if r.key.startswith(f"{FEED}/")]),
            warnings=warnings,
            errors=errors,
        )


def ecom_index(shared: dict[str, RawResponse]) -> dict[str, dict[str, Any]]:
    """warehouseId -> warehouse, from this capture's ecom-api response."""
    response = shared.get("ecom-api")
    if response is None or response.error or response.status != 200:
        return {}
    try:
        payload = json.loads(response.body)
    except (ValueError, UnicodeDecodeError):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for warehouse in (payload.get("warehouses") if isinstance(payload, dict) else None) or []:
        warehouse_id = warehouse.get("warehouseId")
        if warehouse_id is not None:
            out[str(warehouse_id)] = warehouse
    return out


def _lookup_station(
    entry: dict[str, Any],
    ecom: dict[str, dict[str, Any]],
    cached: dict[str, dict[str, Any]],
    cc: CountryConfig | None,
) -> tuple[RawStation, str | None]:
    """The station, and where its timezone came from: ecom, cache or region."""
    station_id = str(entry.get("stlocID"))
    region = _clean(entry.get("state"))
    lat = _as_float(entry.get("latitude"))
    lon = _as_float(entry.get("longitude"))
    tz, tz_origin = None, None

    warehouse = ecom.get(station_id)
    if warehouse is not None:
        address = warehouse.get("address") or {}
        if address.get("latitude") is not None:
            lat = _as_float(address.get("latitude"))
        if address.get("longitude") is not None:
            lon = _as_float(address.get("longitude"))
        tz = _clean(warehouse.get("timeZone"))
        tz_origin = "ecom" if tz else None
    if tz is None:
        tz = _cached_timezone(cached, station_id)
        tz_origin = "cache" if tz else None
    if tz is None and cc is not None:
        # The province table, through CountryConfig; CA has no "*" default row,
        # so an unknown or missing province stays None and is dropped later.
        tz = cc.timezone_for_region(region)
        tz_origin = "region" if tz else None

    station = RawStation(
        source_station_id=station_id,
        alt_id=None,
        id_origin="lookup",
        ecom_state=None,
        name=_clean(entry.get("locationName")) or station_id,
        name_local=None,
        address=_clean(entry.get("address1")),
        city=_clean(entry.get("city")),
        region=region,
        postcode=_clean(entry.get("zipCode")),
        lat=lat,
        lon=lon,
        timezone=tz,
        opening_date=parse_open_date(entry.get("openDate")),
        has_hours=bool(entry.get("gasStationHours")),
        prices=_prices(entry.get("gasPrices")),
    )
    return station, tz_origin


def _cached_station(
    station_id: str, row: dict[str, Any], grades: Any, cc: CountryConfig
) -> tuple[RawStation, str | None]:
    region = _clean(row.get("region"))
    tz = _clean(row.get("timezone"))
    tz_origin = "cache" if tz else None
    if tz is None:
        tz = cc.timezone_for_region(region)
        tz_origin = "region" if tz else None
    station = RawStation(
        source_station_id=station_id,
        alt_id=_clean(row.get("alt_id")),
        id_origin="cache",
        ecom_state=None,
        name=_clean(row.get("name")) or station_id,
        name_local=_clean(row.get("name_local")),
        address=_clean(row.get("address")),
        city=_clean(row.get("city")),
        region=region,
        postcode=_clean(row.get("postcode")),
        lat=_as_float(row.get("lat")),
        lon=_as_float(row.get("lon")),
        timezone=tz,
        opening_date=None,
        has_hours=None,
        prices=_prices(grades),
    )
    return station, tz_origin


def _cached_timezone(cached: dict[str, dict[str, Any]], station_id: str) -> str | None:
    row = cached.get(station_id)
    return _clean(row.get("timezone")) if row else None


def _prices(grades: Any) -> tuple[RawPrice, ...]:
    if not isinstance(grades, dict):
        return ()
    return tuple(
        RawPrice(grade_raw=label, price_raw=text)
        for label, text in costco_lookup.prices(grades).items()
    )


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
    return datetime.strptime(ctx.capture_id, "%Y-%m-%dT%H%MZ").replace(tzinfo=UTC)
