"""Tests for the spec section 7 normalization pipeline."""

from __future__ import annotations

import csv
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from costco_gas.config import Bounds
from costco_gas.fx import FxRates, FxRow
from costco_gas.normalize import (
    LITRES_PER_US_GALLON,
    normalize,
    parse_price,
    round_significant,
)
from costco_gas.schema import ROW_SCHEMA, STATION_SCHEMA
from costco_gas.sources.base import CaptureContext, FetchResult, RawPrice, RawStation

FIXTURES = Path(__file__).parent / "fixtures"

CAPTURE_ID = "2026-09-15T1817Z"
CAPTURE_DATE = date(2026, 9, 15)
CAPTURED_AT = datetime(2026, 9, 15, 18, 17, 40, tzinfo=UTC)
FX_FETCHED_AT = datetime(2026, 9, 15, 18, 17, 45, tzinfo=UTC)

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
        "USD/gal": Bounds(min=2.0, max=11.0, grade_overrides={"clear": (2.0, 11.0)}),
        "USD/L": Bounds(min=0.5, max=2.5, grade_overrides={}),
    },
    "CA": {"CAD/L": Bounds(min=1.0, max=3.5, grade_overrides={})},
    "MX": {"MXN/L": Bounds(min=15.0, max=40.0, grade_overrides={})},
    "GB": {"GBp/L": Bounds(min=100.0, max=250.0, grade_overrides={})},
    "AU": {"AUD/L": Bounds(min=1.2, max=3.4, grade_overrides={})},
    "JP": {"JPY/L": Bounds(min=90.0, max=250.0, grade_overrides={"Kerosene": (60.0, 250.0)})},
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


# --- units, rounding and FX ---------------------------------------------------


def test_us_gallon_row_keeps_its_published_value_per_gallon():
    result = fetch_result("US", "costco-us-gasprices", [station("1364", [("regular", "3.999")])])
    out = normalize(result, fx_rates(), context())
    row = out.rows.row(0, named=True)
    assert row["price"] == 3.999
    assert row["price_unit"] == "USD/gal"
    assert row["currency"] == "USD"
    assert row["price_local_per_litre"] == pytest.approx(1.0564)
    assert row["fx_usd_per_unit"] == 1.0
    assert row["fx_source"] == "identity"
    assert row["fx_rate_date"] == CAPTURE_DATE
    assert row["fx_fetched_at_utc"] == CAPTURED_AT
    assert row["price_usd_per_gallon"] == 3.999
    assert row["price_usd_per_litre"] == pytest.approx(1.0564)


def test_puerto_rico_uses_the_per_litre_region_override():
    result = fetch_result(
        "US",
        "costco-us-gasprices",
        [
            station(
                "335",
                [("regular", "1.097"), ("premium", "1.267")],
                region="PR",
                timezone_name="America/Puerto_Rico",
            )
        ],
    )
    out = normalize(result, fx_rates(), context())
    row = out.rows.filter(pl.col("grade") == "regular").row(0, named=True)
    assert row["price_unit"] == "USD/L"
    assert row["price_local_per_litre"] == 1.097
    assert row["price_usd_per_gallon"] == pytest.approx(4.1526)


def test_gb_pence_become_major_units_per_litre():
    result = fetch_result(
        "GB",
        "costco-occ",
        [station("Coventry", [("5301", "160.9")], timezone_name="Europe/London")],
    )
    out = normalize(result, fx_rates(), context())
    row = out.rows.row(0, named=True)
    assert row["price"] == 160.9
    assert row["price_unit"] == "GBp/L"
    assert row["currency"] == "GBP"
    assert row["price_local_per_litre"] == 1.609
    assert row["fx_usd_per_unit"] == 1.351424401
    assert row["price_usd_per_litre"] == pytest.approx(2.1744)
    assert row["price_usd_per_gallon"] == pytest.approx(8.2312)


def test_jp_keeps_ten_significant_fx_digits_and_a_next_day_local_date():
    result = fetch_result(
        "JP",
        "costco-occ",
        [
            station(
                "Tomiya",
                [("Kerosene", "¥124"), ("Regular", "¥149")],
                timezone_name="Asia/Tokyo",
                name="Tomiya",
                name_local="富谷",
            )
        ],
    )
    out = normalize(result, fx_rates(), context())
    row = out.rows.filter(pl.col("grade") == "regular").row(0, named=True)
    assert row["local_date"] == date(2026, 9, 16)
    assert row["capture_date"] == date(2026, 9, 15)
    assert row["fx_usd_per_unit"] == 0.00648340249
    assert row["fx_source"] == "frankfurter-v2"
    assert row["fx_rate_date"] == date(2026, 9, 14)
    assert row["price_usd_per_litre"] == pytest.approx(0.966)


def test_missing_fx_row_leaves_the_usd_columns_null():
    result = fetch_result(
        "TW",
        "costco-occ",
        [station("Chungli", [("95", "$30.0")], timezone_name="Asia/Taipei")],
    )
    out = normalize(result, FxRates(status="failed", rows=[]), context())
    row = out.rows.row(0, named=True)
    assert row["price_local_per_litre"] == 30.0
    assert row["fx_usd_per_unit"] is None
    assert row["fx_source"] is None
    assert row["price_usd_per_litre"] is None
    assert row["price_usd_per_gallon"] is None


# --- grade mapping (spec 7.1) -------------------------------------------------


def test_grades_map_by_label_not_by_position():
    result = fetch_result(
        "TW",
        "costco-occ",
        [
            station(
                "Chungli",
                [("Diesel", "$28.6"), ("95", "$30.0"), ("98", "$31.5")],
                timezone_name="Asia/Taipei",
            )
        ],
    )
    out = normalize(result, fx_rates(), context())
    got = dict(zip(out.rows["grade_raw"], out.rows["grade"], strict=True))
    assert got == {"Diesel": "diesel", "95": "regular", "98": "premium"}


def test_unknown_label_becomes_other_with_one_warning_per_label():
    result = fetch_result(
        "AU",
        "costco-occ",
        [
            station(
                "109",
                [("Unleaded 91", "$2.147"), ("LPG", "$1.399")],
                timezone_name="Australia/Sydney",
            ),
            station(
                "103",
                [("Unleaded 91", "$2.147"), ("LPG", "$1.409")],
                timezone_name="Australia/Sydney",
            ),
        ],
    )
    out = normalize(result, fx_rates(), context())
    lpg = out.rows.filter(pl.col("grade_raw") == "LPG")
    assert lpg["grade"].to_list() == ["other", "other"]
    unknown = [w for w in out.warnings if w.code == "unknown_grade"]
    assert len(unknown) == 1
    assert unknown[0].detail == "LPG"


def test_grade_conflict_keeps_the_higher_priority_label():
    result = fetch_result(
        "AU",
        "costco-occ",
        [
            station(
                "109",
                [("E10", "$2.127"), ("Unleaded 91", "$2.147")],
                timezone_name="Australia/Sydney",
            )
        ],
    )
    out = normalize(result, fx_rates(), context())
    got = dict(zip(out.rows["grade_raw"], out.rows["grade"], strict=True))
    assert got == {"Unleaded 91": "regular", "E10": "other"}
    conflicts = [w for w in out.warnings if w.code == "grade_conflict"]
    assert len(conflicts) == 1
    assert conflicts[0].detail == "AU-109:regular:E10"


# --- source filters (spec 7.0 step 1) ----------------------------------------


def test_not_open_drops_a_pre_opening_placeholder():
    result = fetch_result(
        "US",
        "costco-us-gasprices",
        [
            station(
                "1838",
                [("regular", "3.999"), ("premium", "5.999")],
                opening_date=date(2026, 10, 2),
            )
        ],
    )
    out = normalize(result, fx_rates(), context())
    assert out.rows.height == 0
    assert reasons(out.drops) == ["not_open"]
    assert codes(out.warnings) == ["priced_before_open"]
    assert out.warnings[0].detail == "1838"


def test_no_hours_drops_a_lookup_row_and_flags_a_live_price():
    result = fetch_result(
        "CA",
        "costco-ca-lookup",
        [
            station(
                "1813",
                [("regular", "1.549")],
                id_origin="lookup",
                has_hours=False,
                timezone_name="America/Edmonton",
            )
        ],
    )
    out = normalize(result, fx_rates(), context())
    assert reasons(out.drops) == ["no_hours"]
    assert codes(out.warnings) == ["priced_before_open"]


def test_extras_skip_not_open_and_no_hours():
    result = fetch_result(
        "US",
        "costco-us-gasprices",
        [
            station(
                "1680",
                [("regular", "5.799")],
                id_origin="extra",
                opening_date=date(2026, 10, 30),
                has_hours=False,
                timezone_name="America/Los_Angeles",
            )
        ],
    )
    out = normalize(result, fx_rates(), context())
    assert out.drops == []
    assert out.rows["station_key"].to_list() == ["US-1680"]


def test_no_gas_service_drops_a_seen_id_that_lost_its_pumps():
    result = fetch_result(
        "US",
        "costco-us-gasprices",
        [
            station("120", [("premium", "3.829")], id_origin="seen", ecom_state="no_gas"),
            station("1364", [("regular", "3.999")], ecom_state="gas"),
        ],
    )
    out = normalize(result, fx_rates(), context())
    assert reasons(out.drops) == ["no_gas_service"]
    assert out.rows["station_key"].to_list() == ["US-1364"]


def test_no_price_and_no_timezone_drop_their_stations():
    result = fetch_result(
        "US",
        "costco-us-gasprices",
        [
            station("1121", []),
            station("1122", [("regular", "3.999")], timezone_name=None),
            station("1123", [("regular", "3.999")], timezone_name="Mars/Olympus"),
        ],
    )
    out = normalize(result, fx_rates(), context())
    assert reasons(out.drops) == ["no_price", "no_timezone", "no_timezone"]
    assert out.rows.height == 0


# --- bounds (spec 7.4) and the station check (spec 7.0 step 5) ----------------


def test_canadian_placeholder_is_rejected_by_the_minimum():
    result = fetch_result(
        "CA",
        "costco-ca-gasprices",
        [
            station(
                "56",
                [("regular", ".959"), ("premium", ".959")],
                id_origin="cache",
                timezone_name="America/Toronto",
            )
        ],
    )
    out = normalize(result, fx_rates(), context())
    assert out.rows.height == 0
    assert reasons(out.drops) == ["out_of_bounds", "out_of_bounds"]
    assert out.drops[0].detail == "regular=.959 CAD/L"


def test_per_grade_override_admits_japanese_kerosene():
    result = fetch_result(
        "JP",
        "costco-occ",
        [
            station(
                "Maebashi",
                [("Kerosene", "¥123"), ("Regular", "¥150")],
                timezone_name="Asia/Tokyo",
            )
        ],
    )
    out = normalize(result, fx_rates(), context())
    assert sorted(out.rows["grade_raw"].to_list()) == ["Kerosene", "Regular"]
    assert out.drops == []


def test_a_grade_without_an_override_uses_the_units_own_bounds():
    """JP Kerosene has a 60 floor; JP Regular keeps the unit's 90 floor."""
    result = fetch_result(
        "JP",
        "costco-occ",
        [
            station(
                "Zama",
                [("Kerosene", "¥62"), ("Regular", "¥80")],
                timezone_name="Asia/Tokyo",
            )
        ],
    )
    out = normalize(result, fx_rates(), context())
    assert out.rows["grade_raw"].to_list() == ["Kerosene"]
    assert reasons(out.drops) == ["out_of_bounds"]
    assert out.drops[0].detail == "Regular=¥80 JPY/L"


def test_sub_minimum_gallon_price_is_dropped_and_leaves_a_no_regular_warning():
    result = fetch_result(
        "US",
        "costco-us-gasprices",
        [station("1090", [("regular", "1.929"), ("premium", "2.129")])],
    )
    out = normalize(result, fx_rates(), context())
    assert reasons(out.drops) == ["out_of_bounds"]
    assert out.rows["grade_raw"].to_list() == ["premium"]
    assert codes(out.warnings) == ["no_regular"]
    assert out.warnings[0].detail == "1090"


def test_no_regular_drops_only_an_absent_seen_us_id():
    result = fetch_result(
        "US",
        "costco-us-gasprices",
        [
            station("1601", [("premium", "4.279")], id_origin="seen", ecom_state="absent"),
            station(
                "1602",
                [("premium", "4.279")],
                id_origin="cache",
                ecom_state="unavailable",
            ),
        ],
    )
    out = normalize(result, fx_rates(), context())
    assert reasons(out.drops) == ["no_regular"]
    assert out.drops[0].source_station_id == "1601"
    assert out.rows["station_key"].to_list() == ["US-1602"]
    assert codes(out.warnings) == ["no_regular"]
    assert out.warnings[0].detail == "1602"


# --- per-row source, frames --------------------------------------------------


def test_fallback_rows_carry_the_fallback_source_name():
    us = normalize(
        fetch_result(
            "US",
            "costco-us-gasprices",
            [
                station("1793", [("regular", "3.499")], id_origin="extra"),
                station("140", [("regular", "4.899")], id_origin="lookup"),
            ],
        ),
        fx_rates(),
        context(),
    )
    assert us.rows["source"].to_list() == ["costco-us-gasprices", "costco-ca-lookup-us"]

    ca = normalize(
        fetch_result(
            "CA",
            "costco-ca-lookup",
            [
                station(
                    "530",
                    [("regular", "1.739")],
                    id_origin="lookup",
                    timezone_name="America/Toronto",
                ),
                station(
                    "524",
                    [("regular", "1.739")],
                    id_origin="cache",
                    timezone_name="America/Toronto",
                ),
            ],
        ),
        fx_rates(),
        context(),
    )
    assert ca.rows["source"].to_list() == ["costco-ca-lookup", "costco-ca-gasprices"]


def test_rows_and_stations_match_the_declared_schemas():
    result = fetch_result(
        "JP",
        "costco-occ",
        [
            station(
                "Tomiya",
                [("Regular", "¥149"), ("Diesel", "¥135")],
                timezone_name="Asia/Tokyo",
                name="Tomiya",
                name_local="富谷",
                alt_id="costcoJapanTomiyaWarehouse",
                id_origin="occ",
            )
        ],
    )
    out = normalize(result, fx_rates(), context())
    assert out.rows.columns == list(ROW_SCHEMA)
    assert dict(out.rows.schema) == ROW_SCHEMA
    assert out.stations.columns == list(STATION_SCHEMA)
    assert dict(out.stations.schema) == STATION_SCHEMA
    st = out.stations.row(0, named=True)
    assert st["station_key"] == "JP-Tomiya"
    assert st["alt_id"] == "costcoJapanTomiyaWarehouse"
    assert st["name_local"] == "富谷"
    assert st["grades_seen"] == "Diesel|Regular"
    assert st["first_seen_utc"] == CAPTURED_AT
    assert st["last_seen_utc"] == CAPTURED_AT
    assert st["status"] == "active"
    assert st["superseded_by"] is None


def test_an_empty_country_still_returns_typed_frames():
    out = normalize(fetch_result("MX", "costco-occ", []), fx_rates(), context())
    assert out.rows.height == 0
    assert dict(out.rows.schema) == ROW_SCHEMA
    assert out.stations.height == 0
    assert out.drops == []
