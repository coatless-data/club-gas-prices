"""Tests for the spec section 7 normalization pipeline."""

from __future__ import annotations

import csv
from datetime import date, datetime, timezone
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from costco_gas.config import Bounds
from costco_gas.fx import FxRates, FxRow
from costco_gas.normalize import LITRES_PER_US_GALLON, parse_price, round_significant
from costco_gas.schema import ROW_SCHEMA, STATION_SCHEMA
from costco_gas.sources.base import CaptureContext, FetchResult, RawPrice, RawStation

FIXTURES = Path(__file__).parent / "fixtures"

CAPTURE_ID = "2026-09-15T1817Z"
CAPTURE_DATE = date(2026, 9, 15)
CAPTURED_AT = datetime(2026, 9, 15, 18, 17, 40, tzinfo=timezone.utc)
FX_FETCHED_AT = datetime(2026, 9, 15, 18, 17, 45, tzinfo=timezone.utc)

# config/grades.csv as of 2026-09-15: (country, grade_raw) -> (grade, priority).
GRADE_ROWS = {
    ("US", "regular"): ("regular", 0),
    ("US", "premium"): ("premium", 0),
    ("US", "diesel"): ("diesel", 0),
    ("US", "clear"): ("other", 0),
    ("CA", "regular"): ("regular", 0),
    ("CA", "premium"): ("premium", 0),
    ("CA", "diesel"): ("diesel", 0),
    ("MX", "Regular"): ("regular", 0),
    ("MX", "Premium"): ("premium", 0),
    ("GB", "5301"): ("regular", 0),
    ("GB", "5302"): ("premium", 0),
    ("GB", "5303"): ("diesel", 0),
    ("AU", "Unleaded 91"): ("regular", 2),
    ("AU", "E10"): ("regular", 1),
    ("AU", "Premium 98"): ("premium", 0),
    ("AU", "Diesel"): ("diesel", 0),
    ("JP", "Regular"): ("regular", 0),
    ("JP", "Premium"): ("premium", 0),
    ("JP", "Diesel"): ("diesel", 0),
    ("JP", "Kerosene"): ("other", 0),
    ("TW", "95"): ("regular", 0),
    ("TW", "98"): ("premium", 0),
    ("TW", "Diesel"): ("diesel", 0),
}

# config/countries.toml as of 2026-09-15, in the shape config.py emits:
# dict[price_unit, Bounds], with per-grade overrides nested in the unit.
BOUNDS = {
    "US": {
        "USD/gal": Bounds(
            min=2.0, max=11.0, grade_overrides={"clear": (2.0, 11.0)}
        ),
        "USD/L": Bounds(min=0.5, max=2.5, grade_overrides={}),
    },
    "CA": {"CAD/L": Bounds(min=1.0, max=3.5, grade_overrides={})},
    "MX": {"MXN/L": Bounds(min=15.0, max=40.0, grade_overrides={})},
    "GB": {"GBp/L": Bounds(min=100.0, max=250.0, grade_overrides={})},
    "AU": {"AUD/L": Bounds(min=1.2, max=3.4, grade_overrides={})},
    "JP": {
        "JPY/L": Bounds(
            min=90.0, max=250.0, grade_overrides={"Kerosene": (60.0, 250.0)}
        )
    },
    "TW": {"TWD/L": Bounds(min=20.0, max=45.0, grade_overrides={})},
}

UNITS = {
    "US": "USD/gal",
    "CA": "CAD/L",
    "MX": "MXN/L",
    "GB": "GBp/L",
    "AU": "AUD/L",
    "JP": "JPY/L",
    "TW": "TWD/L",
}

# The "*" entry of each countries.toml timezone table.
DEFAULT_TIMEZONES = {
    "US": "America/Chicago",
    "CA": "America/Toronto",
    "MX": "America/Mexico_City",
    "GB": "Europe/London",
    "AU": "Australia/Sydney",
    "JP": "Asia/Tokyo",
    "TW": "Asia/Taipei",
}


class FakeGradeTable:
    def map(self, country: str, grade_raw: str):
        found = GRADE_ROWS.get((country, grade_raw))
        if found is None:
            return None
        return SimpleNamespace(
            grade=found[0],
            priority=found[1],
            label=grade_raw,
            spec="",
            spec_source="",
            spec_source_url="",
        )


class FakeTimezoneLookup:
    """Mirrors CountryConfig.timezone_for_region: the country default lives under
    the "*" key, and an unknown or missing region falls back to it. normalize()
    never calls this — RawStation.timezone already carries a resolved IANA name —
    but the stub offers the same accessor as the real CountryConfig so that no
    later edit can reintroduce a direct `.timezones` lookup here."""

    def __init__(self, default: str) -> None:
        self.table = {"*": default}

    def __call__(self, region: str | None) -> str | None:
        if region is not None and region in self.table:
            return self.table[region]
        return self.table.get("*")


def interp_config() -> SimpleNamespace:
    """A stand-in for Config.interp_view(). normalize() only reads
    .grades, .countries[cc].price_unit, .unit_overrides and .bounds."""
    countries = {}
    for code, unit in UNITS.items():
        countries[code] = SimpleNamespace(
            code=code,
            price_unit=unit,
            unit_overrides={"PR": "USD/L"} if code == "US" else {},
            bounds=BOUNDS[code],
            floor=0,
            stale_after_days=3,
            timezone_for_region=FakeTimezoneLookup(DEFAULT_TIMEZONES[code]),
        )
    return SimpleNamespace(countries=countries, grades=FakeGradeTable())


def context(**overrides) -> CaptureContext:
    kwargs = {
        "capture_id": CAPTURE_ID,
        "capture_date": CAPTURE_DATE,
        "fetch_config": SimpleNamespace(),
        "interp_config": interp_config(),
        "previous_stations": pl.DataFrame(schema=STATION_SCHEMA),
        "previous_fx": pl.DataFrame(),
        "previous_status": None,
        "shared": {},
        "force_fallback": set(),
    }
    kwargs.update(overrides)
    return CaptureContext(**kwargs)


def station(
    source_station_id: str,
    prices: list[tuple[str, str]],
    *,
    timezone_name: str = "America/Chicago",
    id_origin: str = "ecom",
    ecom_state: str | None = None,
    region: str | None = None,
    opening_date: date | None = None,
    has_hours: bool | None = None,
    name: str = "Test Warehouse",
    name_local: str | None = None,
    alt_id: str | None = None,
) -> RawStation:
    return RawStation(
        source_station_id=source_station_id,
        alt_id=alt_id,
        id_origin=id_origin,
        ecom_state=ecom_state,
        name=name,
        name_local=name_local,
        address="1 Test Way",
        city="Testville",
        region=region,
        postcode="00000",
        lat=41.9,
        lon=-87.6,
        timezone=timezone_name,
        opening_date=opening_date,
        has_hours=has_hours,
        prices=tuple(RawPrice(grade_raw=g, price_raw=p) for g, p in prices),
    )


def fetch_result(country: str, source: str, stations: list[RawStation]) -> FetchResult:
    return FetchResult(
        country=country,
        source=source,
        captured_at_utc=CAPTURED_AT,
        stations=stations,
        responses=[],
        requests=1,
        warnings=[],
        errors=[],
    )


def fx_rates() -> FxRates:
    """The Frankfurter v2 rates of 2026-09-14, as re-fetched on 2026-09-15."""
    rates = {
        "AUD": 1.399,
        "CAD": 1.3871,
        "GBP": 0.73996,
        "JPY": 154.24,
        "MXN": 17.0477,
        "TWD": 31.709,
    }
    return FxRates(
        status="ok",
        rows=[
            FxRow(
                currency=code,
                units_per_usd=rate,
                fx_usd_per_unit=1.0 / rate,
                fx_rate_date=date(2026, 9, 14),
                fx_source="frankfurter-v2",
                fx_fetched_at_utc=FX_FETCHED_AT,
            )
            for code, rate in rates.items()
        ],
    )


def codes(items) -> list[str]:
    return [item.code for item in items]


def reasons(drops) -> list[str]:
    return [drop.reason for drop in drops]


# --- price parsing (spec 7.2) -------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("3.999", 3.999),
        (".959", 0.959),
        ("¥149", 149.0),
        ("$2.127", 2.127),
        ("$22.99", 22.99),
        ("160.9", 160.9),
        ("$30.0", 30.0),
        ("NT$30.0", 30.0),
        ("  1.097  ", 1.097),
        ("1,234.5", 1234.5),
        ("£1.899", 1.899),
    ],
)
def test_parse_price_handles_every_published_format(raw, expected):
    assert parse_price(raw) == pytest.approx(expected)


@pytest.mark.parametrize("raw", ["", "   ", "N/A", "--", "$", "nan"])
def test_parse_price_rejects_non_numeric_strings(raw):
    assert parse_price(raw) is None


def test_round_significant_keeps_ten_digits():
    assert round_significant(1.0 / 154.24, 10) == 0.00648340249


def test_litres_per_us_gallon_is_the_exact_constant():
    assert LITRES_PER_US_GALLON == 3.785411784


def test_every_fixture_price_sits_at_least_20_percent_inside_its_maximum():
    with (FIXTURES / "prices_2026-09-15.csv").open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) >= 40
    for row in rows:
        unit = row["price_unit"]
        low, high = BOUNDS[row["country"]][unit].for_grade(row["grade_raw"])
        value = parse_price(row["price_raw"])
        assert value is not None, row
        assert value <= 0.8 * high, row
        assert low > 0
