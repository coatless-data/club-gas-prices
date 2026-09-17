"""United States source: ecom-api metadata + AjaxGetGasPricesService, costco.ca US fallback."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import Any
from urllib.parse import urlencode

import polars as pl

from club_gas.http import BudgetExceeded
from club_gas.sources import costco_lookup
from club_gas.sources.base import (
    CaptureContext,
    Error,
    FetchResult,
    RawPrice,
    RawResponse,
    RawStation,
    Warning,
)

COUNTRY = "US"
DEFAULT_BATCH_SIZE = 10
DEFAULT_SEEN_WINDOW_DAYS = 30
ECOM_SHARED_KEY = "ecom-api"
PRICE_SOURCE = "costco-us-gasprices"
LOOKUP_SOURCE = "costco-ca-lookup-us"

DEFAULT_PRICE_URL = "https://www.costco.com/AjaxGetGasPricesService"
DEFAULT_LOOKUP_URL = "https://www.costco.ca/AjaxWarehouseBrowseLookupView"
DEFAULT_LOOKUP_PARAMS = {
    "hasGas": "true",
    "populateWarehouseDetails": "true",
    "countryCode": "US",
}

# The response key's prefix is the feed id, which is also the bundle directory
# the rebuild groups on. It is not the country: the US has two chains.
FEED = "US-COSTCO"
KEY_PRICE = f"{FEED}/02-gasprices-{{n:03d}}"
KEY_LOOKUP = f"{FEED}/03-lookup-us"
KEY_TOPUP = f"{FEED}/04-gasprices-fb-{{n:03d}}"

NON_GRADE_KEYS = costco_lookup.NON_GRADE_KEYS


# --------------------------------------------------------------------------- helpers


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _iso_date(value: Any) -> date | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def lookup_open_date(value: Any) -> date | None:
    """Parse the costco.ca lookup's ``"Aug 23, 1995"`` opening dates."""
    return costco_lookup.open_date(value)


def _as_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    return _iso_date(value)


def _id_sort(warehouse_id: str) -> tuple[int, int, str]:
    if warehouse_id.isdigit():
        return (0, int(warehouse_id), "")
    return (1, 0, warehouse_id)


def _capture_start(ctx: CaptureContext) -> datetime:
    """The capture start, recovered from ``capture_id`` so no module reads the clock."""
    try:
        return datetime.strptime(ctx.capture_id, "%Y-%m-%dT%H%MZ").replace(tzinfo=UTC)
    except ValueError:
        return datetime.combine(ctx.capture_date, datetime.min.time(), tzinfo=UTC)


def _country_cfg(cfg: Any) -> Any:
    countries = getattr(cfg, "countries", None) or {}
    return countries.get(COUNTRY)


def _str_params(params: Any) -> dict[str, str]:
    if not isinstance(params, dict):
        return {}
    return {str(k): str(v) for k, v in params.items() if v is not None}


def _with_query(url: str, params: dict[str, str]) -> str:
    if not params:
        return url
    return url + ("&" if "?" in url else "?") + urlencode(params)


def _positive_int(value: Any, default: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return number if number > 0 else default


def price_url(ctx: CaptureContext) -> str:
    """``CountryConfig.url``: the AjaxGetGasPricesService endpoint, without a query."""
    cfg = _country_cfg(ctx.fetch_config)
    return _clean(getattr(cfg, "url", None)) or DEFAULT_PRICE_URL


def price_params(ctx: CaptureContext) -> dict[str, str]:
    """``CountryConfig.params``: query parameters only, flattened (US ships none)."""
    cfg = _country_cfg(ctx.fetch_config)
    return _str_params(getattr(cfg, "params", None))


def batch_url(ctx: CaptureContext, batch: list[str]) -> str:
    """The price-service URL for one batch: ``warehouseid`` first, config params after."""
    return _with_query(price_url(ctx), {"warehouseid": "_".join(batch), **price_params(ctx)})


def lookup_url(ctx: CaptureContext) -> str:
    """``CountryConfig.fallback_url`` + ``fallback_params``: the costco.ca US lookup."""
    cfg = _country_cfg(ctx.fetch_config)
    base = _clean(getattr(cfg, "fallback_url", None))
    if base is None:
        return _with_query(DEFAULT_LOOKUP_URL, DEFAULT_LOOKUP_PARAMS)
    return _with_query(base, _str_params(getattr(cfg, "fallback_params", None)))


def batch_size(ctx: CaptureContext) -> int:
    """``CountryConfig.batch_size`` (10), the protocol limit of the price service."""
    cfg = _country_cfg(ctx.fetch_config)
    return _positive_int(getattr(cfg, "batch_size", None), DEFAULT_BATCH_SIZE)


def seen_window_days(ctx: CaptureContext) -> int:
    """``CountryConfig.seen_within_days`` (30): how far back a cached id is still polled."""
    cfg = _country_cfg(ctx.fetch_config)
    return _positive_int(getattr(cfg, "seen_within_days", None), DEFAULT_SEEN_WINDOW_DAYS)


def _timezone_for_region(ctx: CaptureContext, region: str | None) -> str | None:
    """The country's region table, always through ``CountryConfig.timezone_for_region``."""
    cfg = _country_cfg(ctx.interp_config)
    resolver = getattr(cfg, "timezone_for_region", None)
    if resolver is None:
        return None
    return _clean(resolver(region))


def batches(ids: list[str], size: int = DEFAULT_BATCH_SIZE) -> list[list[str]]:
    """Split the polled IDs into request batches.

    The service processes ONLY the first 10 IDs of a request (verified on
    2026-09-15: 654 IDs in one call still returned 10 entries), so the batch
    size is a hard protocol limit, not a politeness knob.
    """
    return [ids[i : i + size] for i in range(0, len(ids), size)]


# --------------------------------------------------------------------------- responses


def parse_price_batch(response: RawResponse | None) -> dict[str, dict[str, str]] | None:
    """Parse one AjaxGetGasPricesService body, or ``None`` when the batch failed.

    The endpoint serves valid JSON with ``Content-Type: text/html;charset=UTF-8``,
    so the content type is never consulted here or anywhere else.
    """
    if response is None or response.error or response.status != 200:
        return None
    text = response.body.decode("utf-8", "replace").lstrip("\ufeff \t\r\n")
    if not text.startswith("{"):
        return None
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, dict) or "errorMessage" in payload:
        return None
    parsed: dict[str, dict[str, str]] = {}
    for warehouse_id, grades in payload.items():
        if not isinstance(grades, dict):
            continue
        parsed[str(warehouse_id)] = {
            str(k): str(v)
            for k, v in grades.items()
            if costco_lookup.is_grade_key(k) and isinstance(v, (str, int, float))
        }
    return parsed


def ids_in_url(url: str) -> list[str]:
    _, _, query = url.partition("warehouseid=")
    query = query.split("&", 1)[0]
    return [part for part in query.split("_") if part]


# --------------------------------------------------------------------------- ecom-api


@dataclass(frozen=True)
class EcomWarehouse:
    warehouse_id: str
    country: str
    name: str
    sub_type: str | None
    line1: str | None
    city: str | None
    territory: str | None
    postal_code: str | None
    lat: float | None
    lon: float | None
    timezone: str | None
    opening_date: date | None
    has_gas: bool


def parse_ecom(response: RawResponse | None) -> dict[str, EcomWarehouse] | None:
    """Return the warehouse index, or ``None`` when step 1 is unusable."""
    if response is None or response.error or response.status != 200:
        return None
    try:
        payload = json.loads(response.body.decode("utf-8", "replace"))
        warehouses = payload["warehouses"]
    except (json.JSONDecodeError, KeyError, TypeError, UnicodeError):
        return None
    if not isinstance(warehouses, list):
        return None
    index: dict[str, EcomWarehouse] = {}
    for item in warehouses:
        if not isinstance(item, dict):
            continue
        warehouse_id = _clean(item.get("warehouseId"))
        if warehouse_id is None:
            continue
        address = item.get("address") or {}
        names = item.get("name") or []
        services = item.get("services") or []
        index[warehouse_id] = EcomWarehouse(
            warehouse_id=warehouse_id,
            country=_clean(address.get("countryName")) or "",
            name=_clean(names[0].get("value")) if names else warehouse_id,
            sub_type=_clean((item.get("subType") or {}).get("code")),
            line1=_clean(address.get("line1")),
            city=_clean(address.get("city")),
            territory=_clean(address.get("territory")),
            postal_code=_clean(address.get("postalCode")),
            lat=_as_float(address.get("latitude")),
            lon=_as_float(address.get("longitude")),
            timezone=_clean(item.get("timeZone")),
            opening_date=_iso_date(item.get("openingDate")),
            has_gas=any(_clean(s.get("code")) == "gas" for s in services if isinstance(s, dict)),
        )
    return index


def ecom_state(index: dict[str, EcomWarehouse] | None, warehouse_id: str) -> str:
    if index is None:
        return "unavailable"
    warehouse = index.get(warehouse_id)
    if warehouse is None:
        return "absent"
    if warehouse.country == COUNTRY and warehouse.has_gas:
        return "gas"
    return "no_gas"


# --------------------------------------------------------------------------- id set


@dataclass(frozen=True)
class PolledId:
    source_station_id: str
    id_origin: str
    ecom_state: str


def _extra_rows(ctx: CaptureContext) -> dict[str, dict[str, Any]]:
    frame = getattr(ctx.fetch_config, "us_extra_ids", None)
    if frame is None or not isinstance(frame, pl.DataFrame) or frame.height == 0:
        return {}
    id_column = next(
        (c for c in ("source_station_id", "warehouse_id", "id") if c in frame.columns),
        None,
    )
    if id_column is None:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in frame.iter_rows(named=True):
        warehouse_id = _clean(row.get(id_column))
        if warehouse_id is not None:
            rows[warehouse_id] = row
    return rows


def _previous_rows(ctx: CaptureContext) -> dict[str, dict[str, Any]]:
    frame = ctx.previous_stations
    if frame is None or not isinstance(frame, pl.DataFrame) or frame.height == 0:
        return {}
    if "country" not in frame.columns:
        return {}
    rows: dict[str, dict[str, Any]] = {}
    for row in frame.filter(pl.col("country") == COUNTRY).iter_rows(named=True):
        warehouse_id = _clean(row.get("source_station_id"))
        if warehouse_id is None:
            key = _clean(row.get("station_key")) or ""
            warehouse_id = key.rsplit("-", 1)[1] if "-" in key else None
        if warehouse_id is not None:
            rows[warehouse_id] = row
    return rows


def _seen_ids(ctx: CaptureContext) -> list[str]:
    cutoff = ctx.capture_date - timedelta(days=seen_window_days(ctx))
    fresh = []
    for warehouse_id, row in _previous_rows(ctx).items():
        last_seen = _as_date(row.get("last_seen_utc"))
        if last_seen is not None and last_seen >= cutoff:
            fresh.append(warehouse_id)
    return sorted(fresh, key=_id_sort)


def polled_id_set(ctx: CaptureContext) -> list[PolledId]:
    """The US IDs polled this capture, each tagged with its origin (spec 4.2)."""
    index = parse_ecom(ctx.shared.get(ECOM_SHARED_KEY))
    extras = _extra_rows(ctx)
    seen = _seen_ids(ctx)

    polled: list[PolledId] = []
    taken: set[str] = set()

    def add(warehouse_id: str, origin: str) -> None:
        if warehouse_id in taken:
            return
        taken.add(warehouse_id)
        polled.append(PolledId(warehouse_id, origin, ecom_state(index, warehouse_id)))

    if index is None:
        for warehouse_id in seen:
            add(warehouse_id, "cache")
        for warehouse_id in extras:
            add(warehouse_id, "extra")
        return polled

    gas_ids = sorted(
        (w.warehouse_id for w in index.values() if w.country == COUNTRY and w.has_gas),
        key=_id_sort,
    )
    for warehouse_id in gas_ids:
        add(warehouse_id, "ecom")
    for warehouse_id in extras:
        add(warehouse_id, "extra")
    for warehouse_id in seen:
        if ecom_state(index, warehouse_id) != "gas":
            add(warehouse_id, "seen")
    return polled


def polled_id_frame(ctx: CaptureContext) -> pl.DataFrame:
    """``inputs/us_id_set.csv`` for the capture bundle (spec 8.2)."""
    polled = polled_id_set(ctx)
    return pl.DataFrame(
        {
            "source_station_id": [p.source_station_id for p in polled],
            "id_origin": [p.id_origin for p in polled],
            "ecom_state": [p.ecom_state for p in polled],
        },
        schema={
            "source_station_id": pl.Utf8,
            "id_origin": pl.Utf8,
            "ecom_state": pl.Utf8,
        },
    )


def parse_lookup(response: RawResponse | None) -> dict[str, dict[str, Any]] | None:
    """Parse the costco.ca US lookup body, or ``None`` when it failed."""
    if response is None or response.error or response.status != 200:
        return None
    text = response.body.decode("utf-8", "replace").lstrip("\ufeff \t\r\n")
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return None
    if not isinstance(payload, list) or len(payload) < 2:
        return None
    rows: dict[str, dict[str, Any]] = {}
    for item in payload[1:]:
        if not isinstance(item, dict):
            continue
        warehouse_id = _clean(item.get("stlocID"))
        if warehouse_id is not None:
            rows[warehouse_id] = item
    return rows


def lookup_prices(row: dict[str, Any]) -> dict[str, str]:
    gas_prices = row.get("gasPrices") or {}
    if not isinstance(gas_prices, dict):
        return {}
    return costco_lookup.prices(gas_prices)


# --------------------------------------------------------------------------- metadata


@dataclass
class Meta:
    name: str | None = None
    address: str | None = None
    city: str | None = None
    region: str | None = None
    postcode: str | None = None
    lat: float | None = None
    lon: float | None = None
    timezone: str | None = None
    opening_date: date | None = None
    alt_id: str | None = None


def _meta_from_ecom(warehouse: EcomWarehouse) -> Meta:
    return Meta(
        name=warehouse.name,
        address=warehouse.line1,
        city=warehouse.city,
        region=warehouse.territory,
        postcode=warehouse.postal_code,
        lat=warehouse.lat,
        lon=warehouse.lon,
        timezone=warehouse.timezone,
        opening_date=warehouse.opening_date,
    )


def _meta_from_row(row: dict[str, Any]) -> Meta:
    return Meta(
        name=_clean(row.get("name")),
        address=_clean(row.get("address")),
        city=_clean(row.get("city")),
        region=_clean(row.get("region")),
        postcode=_clean(row.get("postcode")),
        lat=_as_float(row.get("lat")),
        lon=_as_float(row.get("lon")),
        timezone=_clean(row.get("timezone")),
        alt_id=_clean(row.get("alt_id")),
    )


def _meta_from_lookup(row: dict[str, Any]) -> Meta:
    return Meta(
        name=_clean(row.get("locationName")) or _clean(row.get("displayName")),
        address=_clean(row.get("address1")),
        city=_clean(row.get("city")),
        region=_clean(row.get("state")),
        postcode=_clean(row.get("zipCode")),
        lat=_as_float(row.get("latitude")),
        lon=_as_float(row.get("longitude")),
        opening_date=lookup_open_date(row.get("openDate")),
    )


def _merge(chain: list[Meta]) -> Meta:
    merged = Meta()
    for field_name in Meta.__dataclass_fields__:
        for candidate in chain:
            value = getattr(candidate, field_name)
            if value is not None:
                setattr(merged, field_name, value)
                break
    return merged


# --------------------------------------------------------------------------- source


def parse_us(responses: list[RawResponse], ctx: CaptureContext) -> FetchResult:
    polled = polled_id_set(ctx)
    origins = {p.source_station_id: p.id_origin for p in polled}
    states = {p.source_station_id: p.ecom_state for p in polled}
    index = parse_ecom(ctx.shared.get(ECOM_SHARED_KEY))
    extras = _extra_rows(ctx)
    previous = _previous_rows(ctx)

    warnings: list[Warning] = []
    errors: list[Error] = []
    if index is None and previous:
        warnings.append(Warning(code="metadata_from_cache", detail=COUNTRY))

    grades_by_id: dict[str, dict[str, str]] = {}
    origin_by_id: dict[str, str] = {}
    lookup_rows: dict[str, dict[str, Any]] = {}
    received: list[datetime] = []

    for response in sorted(responses, key=lambda r: r.key):
        if response.error == "deadline_exceeded":
            warnings.append(Warning(code="deadline_exceeded", detail=response.key))
            continue
        if "gasprices" in response.key:
            parsed = parse_price_batch(response)
            if parsed is None:
                errors.append(
                    Error(
                        code="price_batch_failed",
                        host="www.costco.com",
                        http_status=response.status,
                        detail=response.key,
                    )
                )
                continue
            received.append(response.received_at_utc)
            for warehouse_id, grades in parsed.items():
                if grades and warehouse_id not in grades_by_id:
                    grades_by_id[warehouse_id] = grades
                    origin_by_id[warehouse_id] = origins.get(warehouse_id, "seen")
        elif "lookup" in response.key:
            rows = parse_lookup(response)
            if rows is None:
                errors.append(
                    Error(
                        code="lookup_failed",
                        host="www.costco.ca",
                        http_status=response.status,
                        detail=response.key,
                    )
                )
                continue
            received.append(response.received_at_utc)
            lookup_rows = rows
            for warehouse_id, row in rows.items():
                if warehouse_id not in origins:
                    continue
                grades = lookup_prices(row)
                if grades and warehouse_id not in grades_by_id:
                    grades_by_id[warehouse_id] = grades
                    origin_by_id[warehouse_id] = "lookup"

    use_all_lookup = index is None and not previous
    emitted: list[str] = list(origins)
    if use_all_lookup:
        for warehouse_id, row in lookup_rows.items():
            if warehouse_id in origins:
                continue
            grades = lookup_prices(row)
            if not grades:
                continue
            grades_by_id[warehouse_id] = grades
            origin_by_id[warehouse_id] = "lookup"
            emitted.append(warehouse_id)

    stations: list[RawStation] = []
    for warehouse_id in emitted:
        grades = grades_by_id.get(warehouse_id, {})
        origin = origin_by_id.get(warehouse_id, origins.get(warehouse_id, "lookup"))
        state = states.get(warehouse_id, "unavailable")
        is_extra = warehouse_id in extras
        lookup_row = lookup_rows.get(warehouse_id)

        chain: list[Meta] = []
        if state == "gas" and index is not None and warehouse_id in index:
            chain.append(_meta_from_ecom(index[warehouse_id]))
        if is_extra:
            chain.append(_meta_from_row(extras[warehouse_id]))
        if warehouse_id in previous:
            chain.append(_meta_from_row(previous[warehouse_id]))
        if lookup_row is not None:
            chain.append(_meta_from_lookup(lookup_row))
        meta = _merge(chain)

        if state == "absent" and grades:
            warnings.append(Warning(code="not_in_ecom_api", detail=warehouse_id))

        tz = meta.timezone
        if tz is None:
            tz = _timezone_for_region(ctx, meta.region)
            if tz is not None:
                warnings.append(Warning(code="timezone_from_region", detail=warehouse_id))

        opening_date = None if is_extra else meta.opening_date
        has_hours = None
        if origin == "lookup" and lookup_row is not None and not is_extra:
            has_hours = bool(lookup_row.get("gasStationHours"))

        stations.append(
            RawStation(
                source_station_id=warehouse_id,
                alt_id=meta.alt_id,
                id_origin=origin,
                ecom_state=state,
                name=meta.name or warehouse_id,
                name_local=None,
                address=meta.address,
                city=meta.city,
                region=meta.region,
                postcode=meta.postcode,
                lat=meta.lat,
                lon=meta.lon,
                timezone=tz,
                opening_date=opening_date,
                has_hours=has_hours,
                prices=tuple(RawPrice(grade_raw=g, price_raw=p) for g, p in grades.items()),
            )
        )

    used_lookup = any(s.id_origin == "lookup" for s in stations)
    return FetchResult(
        country=COUNTRY,
        source=LOOKUP_SOURCE if used_lookup else PRICE_SOURCE,
        captured_at_utc=max(received) if received else _capture_start(ctx),
        stations=stations,
        responses=list(responses),
        requests=len(responses),
        warnings=warnings,
        errors=errors,
    )


def _deadline_response(ctx: CaptureContext, key: str, url: str) -> RawResponse:
    return RawResponse(
        key=key,
        url=url,
        status=None,
        headers={},
        received_at_utc=_capture_start(ctx),
        elapsed_ms=0,
        body=b"",
        error="deadline_exceeded",
    )


def fetch_us(client, ctx: CaptureContext) -> list[RawResponse]:
    responses: list[RawResponse] = []
    ids = [p.source_station_id for p in polled_id_set(ctx)]
    planned = batches(ids, batch_size(ctx))
    prices = price_url(ctx)
    forced = COUNTRY in (ctx.force_fallback or set())
    step1_failed = parse_ecom(ctx.shared.get(ECOM_SHARED_KEY)) is None
    no_cached_rows = not _seen_ids(ctx)

    attempted = 0
    failed = 0
    priced: set[str] = set()

    if not forced:
        for number, batch in enumerate(planned, start=1):
            if client.abandoned(prices):
                break
            url = batch_url(ctx, batch)
            try:
                response = client.request(KEY_PRICE.format(n=number), url)
            except BudgetExceeded:
                responses.append(_deadline_response(ctx, KEY_PRICE.format(n=number), url))
                break
            responses.append(response)
            attempted += 1
            parsed = parse_price_batch(response)
            if parsed is None:
                failed += 1
            else:
                priced.update(k for k, v in parsed.items() if v)

    use_fallback = (
        forced
        or (bool(planned) and failed * 2 > len(planned))
        or attempted < len(planned)
        or (step1_failed and no_cached_rows)
    )
    if not use_fallback:
        return responses

    lookup = lookup_url(ctx)
    if not client.abandoned(lookup):
        try:
            response = client.request(KEY_LOOKUP, lookup, profile="bulk")
        except BudgetExceeded:
            response = _deadline_response(ctx, KEY_LOOKUP, lookup)
        responses.append(response)
        rows = parse_lookup(response) or {}
        priced.update(
            warehouse_id
            for warehouse_id, row in rows.items()
            if warehouse_id in set(ids) and lookup_prices(row)
        )

    remaining = [warehouse_id for warehouse_id in ids if warehouse_id not in priced]
    for number, batch in enumerate(batches(remaining, batch_size(ctx)), start=1):
        if client.abandoned(prices):
            break
        url = batch_url(ctx, batch)
        try:
            responses.append(client.request(KEY_TOPUP.format(n=number), url))
        except BudgetExceeded:
            responses.append(_deadline_response(ctx, KEY_TOPUP.format(n=number), url))
            break
    return responses


class UsSource:
    country = COUNTRY

    def fetch(self, client, ctx: CaptureContext) -> list[RawResponse]:
        return fetch_us(client, ctx)

    def parse(self, responses: list[RawResponse], ctx: CaptureContext) -> FetchResult:
        return parse_us(responses, ctx)
