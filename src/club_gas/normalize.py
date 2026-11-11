"""Turn a country's FetchResult into price rows and station rows (spec section 7)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl

from club_gas.fx import FxRates
from club_gas.schema import HELD_SCHEMA, ROW_SCHEMA, STATION_SCHEMA, SchemaError, validate_held
from club_gas.sources.base import CaptureContext, FetchResult, RawStation, Warning

LITRES_PER_US_GALLON = 3.785411784

CURRENCY_BY_UNIT = {
    "USD/gal": "USD",
    "USD/L": "USD",
    "CAD/L": "CAD",
    "MXN/L": "MXN",
    "GBp/L": "GBP",
    "AUD/L": "AUD",
    "JPY/L": "JPY",
    "TWD/L": "TWD",
}

# Longest first, so "NT$" is removed before "$".
_CURRENCY_SYMBOLS = ("NT$", "$", "¥", "£", "€")


@dataclass(frozen=True)
class Drop:
    source_station_id: str
    reason: str
    detail: str = ""


@dataclass
class NormalizedCountry:
    rows: pl.DataFrame
    stations: pl.DataFrame
    drops: list[Drop] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)
    # Rows of stations recorded but not published, each with its hold_reason.
    held: pl.DataFrame = field(default_factory=lambda: pl.DataFrame(schema=HELD_SCHEMA))


def parse_price(price_raw: str) -> float | None:
    """Spec 7.2: strip whitespace, currency symbols and separators, then parse."""
    text = price_raw.strip()
    for symbol in _CURRENCY_SYMBOLS:
        text = text.replace(symbol, "")
    # NBSP and the ideographic space (U+3000) pad OCC storefront fields; ruff's
    # formatter unescapes them to literal characters, hence the RUF001 waiver.
    text = text.replace(",", "").replace(" ", "").replace("　", "").strip()  # noqa: RUF001
    if text.startswith("."):
        text = "0" + text
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def round_significant(value: float, digits: int) -> float:
    if value == 0.0 or not math.isfinite(value):
        return value
    return round(value, -math.floor(math.log10(abs(value))) + (digits - 1))


def local_per_litre(price: float, price_unit: str) -> float:
    """Spec 7.3."""
    if price_unit == "USD/gal":
        return price / LITRES_PER_US_GALLON
    if price_unit == "GBp/L":
        return price / 100.0
    return price


def row_source(country: str, id_origin: str, result_source: str) -> str:
    """Spec 6.1: the US costco.ca fallback and the CA price-service fallback
    label their rows differently from the country's primary source."""
    if country == "US" and id_origin == "lookup":
        return "costco-ca-lookup-us"
    if country == "CA" and id_origin == "cache":
        return "costco-ca-gasprices"
    return result_source


def _valid_zone(name: str | None) -> bool:
    if not name:
        return False
    try:
        ZoneInfo(name)
    except (ZoneInfoNotFoundError, ValueError):
        return False
    return True


def _price_unit(station: RawStation, country_cfg) -> str:
    unit = country_cfg.price_unit
    overrides = getattr(country_cfg, "unit_overrides", None) or {}
    if station.region and station.region in overrides:
        return overrides[station.region]
    return unit


def _bounds_for(country_cfg, grade_raw: str, price_unit: str) -> tuple[float, float] | None:
    """Spec 7.4. `country_cfg.bounds` is `dict[str, Bounds]` keyed by price unit,
    and each `Bounds` carries its own per-grade overrides, so the lookup is always
    `bounds[price_unit].for_grade(grade_raw)`. A unit with no configured `Bounds`
    means no bounds check."""
    entry = country_cfg.bounds.get(price_unit)
    if entry is None:
        return None
    return entry.for_grade(grade_raw)


def _gate(station: RawStation, capture_date: date) -> str | None:
    """Spec 7.0 step 1's station-level gates, in the listed order.

    A station a gate stops is held, not discarded, when it has prices to record.
    """
    if station.id_origin == "extra":
        # Extras price live before their listed opening date and carry no hours
        # (spec 4.2), so the gates are skipped for them.
        return None
    if station.ecom_state == "no_gas":
        return "no_gas_service"
    if station.opening_date is not None and station.opening_date > capture_date:
        return "not_open"
    if station.has_hours is False:
        return "no_hours"
    return None


def _approvals(interp, feed: str) -> dict[str, date]:
    """The stations watch.csv turns on for this feed, each with its publish_from.

    Read from the interpretation view, so a rebuild applies today's decisions
    to captures recorded before they were made.
    """
    frame = getattr(interp, "watch", None)
    if not isinstance(frame, pl.DataFrame) or frame.is_empty():
        return {}
    out: dict[str, date] = {}
    on = frame.filter((pl.col("feed") == feed) & (pl.col("status") == "on"))
    for row in on.iter_rows(named=True):
        if row["source_station_id"] and row["publish_from"]:
            out[str(row["source_station_id"])] = date.fromisoformat(row["publish_from"])
    return out


def _priced_before_open(station: RawStation, country: str, brand: str, country_cfg, grades) -> bool:
    """Spec 4.3: a station dropped as not_open/no_hours that still shows an
    in-bounds regular price is worth a warning."""
    unit = _price_unit(station, country_cfg)
    for raw in station.prices:
        entry = grades.map(country, brand, raw.grade_raw)
        if entry is None or entry.grade != "regular":
            continue
        value = parse_price(raw.price_raw)
        if value is None:
            continue
        bounds = _bounds_for(country_cfg, raw.grade_raw, unit)
        if bounds is None or bounds[0] <= value <= bounds[1]:
            return True
    return False


def _surviving_prices(
    station: RawStation,
    station_key: str,
    country: str,
    brand: str,
    unit: str,
    country_cfg,
    grades,
    unknown_labels: set[str],
) -> tuple[list[dict], list[Drop], list[Warning]]:
    """Steps 2 to 4 for one station: parse, map grades, apply the bounds."""
    sid = station.source_station_id
    drops: list[Drop] = []
    warnings: list[Warning] = []

    # Step 2: price parsing. Step 3: grade mapping.
    working: list[dict] = []
    for raw in station.prices:
        value = parse_price(raw.price_raw)
        if value is None:
            drops.append(Drop(sid, "unparseable_price", f"{raw.grade_raw}={raw.price_raw}"))
            continue
        entry = grades.map(country, brand, raw.grade_raw)
        if entry is None:
            grade, priority = "other", 0
            if raw.grade_raw not in unknown_labels:
                unknown_labels.add(raw.grade_raw)
                warnings.append(Warning(code="unknown_grade", detail=raw.grade_raw))
        else:
            grade, priority = entry.grade, entry.priority
        working.append(
            {
                "grade_raw": raw.grade_raw,
                "price_raw": raw.price_raw,
                "price": value,
                "grade": grade,
                "priority": priority,
            }
        )

    # Step 3 conflict rule: one non-`other` grade per station, highest
    # priority wins, source order breaks a tie.
    by_grade: dict[str, list[int]] = {}
    for index, item in enumerate(working):
        if item["grade"] != "other":
            by_grade.setdefault(item["grade"], []).append(index)
    for grade, indexes in by_grade.items():
        if len(indexes) < 2:
            continue
        winner = max(indexes, key=lambda i: (working[i]["priority"], -i))
        for index in indexes:
            if index == winner:
                continue
            warnings.append(
                Warning(
                    code="grade_conflict",
                    detail=f"{station_key}:{grade}:{working[index]['grade_raw']}",
                )
            )
            working[index]["grade"] = "other"

    # Step 4: units are already fixed per station; apply the bounds.
    surviving: list[dict] = []
    for item in working:
        bounds = _bounds_for(country_cfg, item["grade_raw"], unit)
        if bounds is not None and not (bounds[0] <= item["price"] <= bounds[1]):
            drops.append(
                Drop(
                    sid,
                    "out_of_bounds",
                    f"{item['grade_raw']}={item['price_raw']} {unit}",
                )
            )
            continue
        surviving.append(item)
    return surviving, drops, warnings


def _price_records(
    station: RawStation,
    station_key: str,
    country: str,
    brand: str,
    unit: str,
    surviving: list[dict],
    result: FetchResult,
    fx: FxRates,
    ctx: CaptureContext,
) -> list[dict]:
    """Step 6 for one station: FX, the USD columns and local_date, one row per price."""
    currency = CURRENCY_BY_UNIT[unit]
    if currency == "USD":
        fx_usd_per_unit = 1.0
        fx_rate_date = ctx.capture_date
        fx_source = "identity"
        fx_fetched_at = result.captured_at_utc
    else:
        fx_row = fx.for_currency(currency)
        if fx_row is None:
            fx_usd_per_unit = None
            fx_rate_date = None
            fx_source = None
            fx_fetched_at = None
        else:
            fx_usd_per_unit = fx_row.fx_usd_per_unit
            fx_rate_date = fx_row.fx_rate_date
            fx_source = fx_row.fx_source
            fx_fetched_at = fx_row.fx_fetched_at_utc

    local_date = result.captured_at_utc.astimezone(ZoneInfo(station.timezone)).date()
    source = row_source(country, station.id_origin, result.source)

    records: list[dict] = []
    for item in surviving:
        litre = local_per_litre(item["price"], unit)
        if fx_usd_per_unit is None:
            usd_litre = None
            usd_gallon = None
        else:
            usd_litre = litre * fx_usd_per_unit
            usd_gallon = item["price"] if unit == "USD/gal" else usd_litre * LITRES_PER_US_GALLON
        records.append(
            {
                "capture_id": ctx.capture_id,
                "capture_date": ctx.capture_date,
                "captured_at_utc": result.captured_at_utc,
                "local_date": local_date,
                "country": country,
                "brand": brand,
                "station_key": station_key,
                "source_station_id": station.source_station_id,
                "source": source,
                "name": station.name,
                "name_local": station.name_local,
                "address": station.address,
                "city": station.city,
                "region": station.region,
                "postcode": station.postcode,
                "lat": station.lat,
                "lon": station.lon,
                "timezone": station.timezone,
                "grade_raw": item["grade_raw"],
                "grade": item["grade"],
                "price_raw": item["price_raw"],
                "price": item["price"],
                "price_unit": unit,
                "currency": currency,
                # Unrounded intermediates: schema.py's round_price_columns and
                # round_fx_columns are the only place rounding happens (spec 6.1).
                "price_local_per_litre": litre,
                "fx_usd_per_unit": fx_usd_per_unit,
                "fx_rate_date": fx_rate_date,
                "fx_source": fx_source,
                "fx_fetched_at_utc": fx_fetched_at,
                "price_usd_per_litre": usd_litre,
                "price_usd_per_gallon": usd_gallon,
            }
        )
    return records


def normalize(result: FetchResult, fx: FxRates, ctx: CaptureContext) -> NormalizedCountry:
    country = result.country
    brand = result.brand
    interp = ctx.interp_config
    country_cfg = interp.countries[country]
    grades = interp.grades
    turned_on = _approvals(interp, f"{country}-{brand}")

    drops: list[Drop] = []
    warnings: list[Warning] = []
    unknown_labels: set[str] = set()
    records: list[dict] = []
    held_records: list[dict] = []
    station_records: list[dict] = []

    held_keys: set[str] = set()

    def hold(station: RawStation, station_key: str, reason: str) -> None:
        """Record a station's rows without publishing them.

        Its price-level drops and warnings are discarded: these stations were
        dropped before price parsing until held stations existed, and the
        status checks must keep seeing the counts they saw then.

        Its rows are checked here, one station at a time, so that a held
        station with bad data -- no name, say -- is skipped with a warning
        instead of failing the feed's published rows along with it.
        """
        unit = _price_unit(station, country_cfg)
        surviving, _, _ = _surviving_prices(
            station, station_key, country, brand, unit, country_cfg, grades, set()
        )
        records = [
            {**record, "hold_reason": reason}
            for record in _price_records(
                station, station_key, country, brand, unit, surviving, result, fx, ctx
            )
        ]
        if not records:
            return
        sid = station.source_station_id
        if station_key in held_keys:
            warnings.append(Warning(code="held_rows_invalid", detail=f"{sid}: held twice"))
            return
        try:
            validate_held(pl.DataFrame(records, schema=HELD_SCHEMA))
        except SchemaError as exc:
            warnings.append(Warning(code="held_rows_invalid", detail=f"{sid}: {exc}"))
            return
        held_keys.add(station_key)
        held_records.extend(records)

    for station in result.stations:
        sid = station.source_station_id
        station_key = f"{country}-{brand}-{sid}"
        publish_from = turned_on.get(sid)
        # capture_date is the capture's UTC date, the column publish_from names.
        is_on = publish_from is not None and publish_from <= ctx.capture_date

        # Step 1: the source filter. A station watch.csv has turned on skips the
        # gates, exactly as an extra does.
        gate = None if is_on else _gate(station, ctx.capture_date)
        if gate is not None:
            if gate in ("not_open", "no_hours") and _priced_before_open(
                station, country, brand, country_cfg, grades
            ):
                warnings.append(Warning(code="priced_before_open", detail=sid))
            drops.append(Drop(sid, gate, ""))
            if station.prices and _valid_zone(station.timezone):
                hold(station, station_key, gate)
            continue
        if not station.prices:
            drops.append(Drop(sid, "no_price", ""))
            continue
        if not _valid_zone(station.timezone):
            drops.append(Drop(sid, "no_timezone", ""))
            continue
        if station.id_origin == "watch" and not is_on:
            # Polled only because watch.csv names it: recorded until turned on.
            drops.append(Drop(sid, "watch", ""))
            hold(station, station_key, "watch")
            continue

        unit = _price_unit(station, country_cfg)
        surviving, price_drops, price_warnings = _surviving_prices(
            station, station_key, country, brand, unit, country_cfg, grades, unknown_labels
        )
        drops.extend(price_drops)
        warnings.extend(price_warnings)
        if not surviving:
            continue

        # Step 5: the station check.
        if not any(item["grade"] == "regular" for item in surviving):
            if country == "US" and station.id_origin == "seen" and station.ecom_state == "absent":
                drops.append(Drop(sid, "no_regular", ""))
                continue
            warnings.append(Warning(code="no_regular", detail=sid))

        records.extend(
            _price_records(station, station_key, country, brand, unit, surviving, result, fx, ctx)
        )
        station_records.append(
            {
                "station_key": station_key,
                "country": country,
                "brand": brand,
                "source_station_id": sid,
                "alt_id": station.alt_id,
                "name": station.name,
                "name_local": station.name_local,
                "address": station.address,
                "city": station.city,
                "region": station.region,
                "postcode": station.postcode,
                "lat": station.lat,
                "lon": station.lon,
                "timezone": station.timezone,
                "grades_seen": "|".join(sorted({item["grade_raw"] for item in surviving})),
                "first_seen_utc": result.captured_at_utc,
                "last_seen_utc": result.captured_at_utc,
                "status": "active",
                "superseded_by": None,
            }
        )

    return NormalizedCountry(
        rows=pl.DataFrame(records, schema=ROW_SCHEMA),
        stations=pl.DataFrame(station_records, schema=STATION_SCHEMA),
        drops=drops,
        warnings=warnings,
        held=pl.DataFrame(held_records, schema=HELD_SCHEMA),
    )
