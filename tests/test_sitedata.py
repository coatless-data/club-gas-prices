import json
from datetime import UTC, date, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl
import pytest

from costco_gas.sitedata import dedupe_daily

NOW = datetime(2026, 9, 16, 2, 0, tzinfo=UTC)

# country, grade_raw -> (grade, priority), from config/grades.csv (spec 7.1).
GRADE_TABLE = {
    ("US", "regular"): ("regular", 0),
    ("US", "premium"): ("premium", 0),
    ("US", "clear"): ("other", 0),
    ("GB", "5301"): ("regular", 0),
    ("AU", "Unleaded 91"): ("regular", 2),
    ("AU", "E10"): ("regular", 1),
}

NOTICE = (
    "Unofficial. Not affiliated with, endorsed by, or connected to Costco Wholesale "
    "Corporation. Prices are collected from Costco's public websites and may differ "
    "from the price at the pump."
)


class StubGrades:
    """The GradeTable surface sitedata uses: map() and rows()."""

    def map(self, country, grade_raw):
        found = GRADE_TABLE.get((country, grade_raw))
        if found is None:
            return None
        return SimpleNamespace(grade=found[0], priority=found[1], label=grade_raw)

    def rows(self):
        return [
            {
                "country": country,
                "grade_raw": grade_raw,
                "grade": grade,
                "priority": priority,
                "label": grade_raw,
                "spec": "",
                "spec_source": "",
                "spec_source_url": "",
            }
            for (country, grade_raw), (grade, priority) in GRADE_TABLE.items()
        ]


def _cfg() -> SimpleNamespace:
    """A stand-in Config whose site block uses the key names config/site.toml uses."""
    return SimpleNamespace(
        grades=StubGrades(),
        site=SimpleNamespace(
            notice=NOTICE,
            release_base_url="https://github.com/coatless-dashboard/costco-gas-prices/releases",
            basemap_key_env="CARTO_BASEMAP_KEY",
            basemaps={
                "carto": {
                    "light_url": "https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png?key={key}",
                    "dark_url": "https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png?key={key}",
                    "subdomains": "abcd",
                    "max_zoom": 20,
                    "dark_filter": False,
                    "attribution": "© OpenStreetMap contributors © CARTO",
                },
                "osm": {
                    "light_url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                    "dark_url": "https://tile.openstreetmap.org/{z}/{x}/{y}.png",
                    "subdomains": "",
                    "max_zoom": 19,
                    "dark_filter": True,
                    "attribution": '© <a href="https://www.openstreetmap.org/copyright">OpenStreetMap contributors</a>',
                },
            },
        ),
    )


def _daily_row(**overrides) -> dict:
    row = {
        "capture_date": date(2026, 9, 15),
        "station_key": "US-1364",
        "country": "US",
        "region": "FL",
        "grade": "regular",
        "grade_raw": "regular",
        "captured_at_utc": datetime(2026, 9, 15, 18, 17),
        "price_local_per_litre": 1.0,
        "price_usd_per_litre": 1.0,
        "currency": "USD",
        "n_captures": 4,
    }
    row.update(overrides)
    return row


def _daily_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "capture_date": pl.Date,
            "station_key": pl.String,
            "country": pl.String,
            "region": pl.String,
            "grade": pl.String,
            "grade_raw": pl.String,
            "captured_at_utc": pl.Datetime("us"),
            "price_local_per_litre": pl.Float64,
            "price_usd_per_litre": pl.Float64,
            "currency": pl.String,
            "n_captures": pl.Int64,
        },
    )


def test_other_grades_are_dropped():
    frame = _daily_frame([_daily_row(), _daily_row(grade="other", grade_raw="clear")])
    assert dedupe_daily(frame, _cfg())["grade"].to_list() == ["regular"]


def test_an_intraday_relabel_keeps_the_later_capture():
    frame = _daily_frame(
        [
            _daily_row(
                station_key="AU-109",
                country="AU",
                region="NSW",
                grade_raw="E10",
                captured_at_utc=datetime(2026, 9, 15, 6, 17),
                price_local_per_litre=1.90,
                n_captures=3,
            ),
            _daily_row(
                station_key="AU-109",
                country="AU",
                region="NSW",
                grade_raw="Unleaded 91",
                captured_at_utc=datetime(2026, 9, 15, 18, 17),
                price_local_per_litre=1.95,
                n_captures=1,
            ),
        ]
    )
    out = dedupe_daily(frame, _cfg())
    assert out.height == 1
    assert out["grade_raw"].to_list() == ["Unleaded 91"]
    assert out["n_captures"].to_list() == [1]


def test_a_tie_on_time_is_broken_by_priority():
    same_time = datetime(2026, 9, 15, 18, 17)
    frame = _daily_frame(
        [
            _daily_row(
                station_key="AU-109", country="AU", region="NSW",
                grade_raw="E10", captured_at_utc=same_time, price_local_per_litre=1.90,
            ),
            _daily_row(
                station_key="AU-109", country="AU", region="NSW",
                grade_raw="Unleaded 91", captured_at_utc=same_time, price_local_per_litre=1.95,
            ),
        ]
    )
    out = dedupe_daily(frame, _cfg())
    assert out["grade_raw"].to_list() == ["Unleaded 91"]  # priority 2 beats priority 1


def test_the_key_is_unique_per_day_station_and_grade():
    frame = _daily_frame(
        [
            _daily_row(),
            _daily_row(capture_date=date(2026, 9, 14), price_local_per_litre=1.1),
            _daily_row(grade="premium", grade_raw="premium", price_local_per_litre=1.5),
        ]
    )
    out = dedupe_daily(frame, _cfg())
    assert out.height == 3
    assert out.select("capture_date", "station_key", "grade").is_unique().all()


from costco_gas.sitedata import build_site_data  # noqa: E402

LATEST_ROWS = [
    # US-1364: two grades from the same capture.
    {
        "station_key": "US-1364", "country": "US", "name": "Bradenton", "name_local": None,
        "city": "BRADENTON", "region": "FL", "lat": 27.49462445, "lon": -82.47014406,
        "captured_at_utc": "2026-09-15T18:17:00Z", "grade": "regular", "grade_raw": "regular",
        "price_raw": "3.999", "price": 3.999, "price_unit": "USD/gal", "currency": "USD",
        "price_local_per_litre": 1.0564, "price_usd_per_litre": 1.0564,
        "price_usd_per_gallon": 3.999, "fx_usd_per_unit": 1.0,
        "fx_rate_date": "2026-09-15", "fx_source": "identity",
    },
    {
        "station_key": "US-1364", "country": "US", "name": "Bradenton", "name_local": None,
        "city": "BRADENTON", "region": "FL", "lat": 27.49462445, "lon": -82.47014406,
        "captured_at_utc": "2026-09-15T18:17:00Z", "grade": "premium", "grade_raw": "premium",
        "price_raw": "4.629", "price": 4.629, "price_unit": "USD/gal", "currency": "USD",
        "price_local_per_litre": 1.2229, "price_usd_per_litre": 1.2229,
        "price_usd_per_gallon": 4.629, "fx_usd_per_unit": 1.0,
        "fx_rate_date": "2026-09-15", "fx_source": "identity",
    },
    # US-140 sells the "clear" grade, which maps to other and lands under other.
    {
        "station_key": "US-140", "country": "US", "name": "Seattle", "name_local": None,
        "city": "SEATTLE", "region": "WA", "lat": 47.6, "lon": -122.3,
        "captured_at_utc": "2026-09-15T18:17:00Z", "grade": "other", "grade_raw": "clear",
        "price_raw": "5.699", "price": 5.699, "price_unit": "USD/gal", "currency": "USD",
        "price_local_per_litre": 1.5055, "price_usd_per_litre": 1.5055,
        "price_usd_per_gallon": 5.699, "fx_usd_per_unit": 1.0,
        "fx_rate_date": "2026-09-15", "fx_source": "identity",
    },
    # JP-Tomiya carries a native-script name and a real exchange rate.
    {
        "station_key": "JP-Tomiya", "country": "JP", "name": "Tomiya", "name_local": "富谷倉庫店",
        "city": "富谷市", "region": "宮城県", "lat": 38.39, "lon": 140.89,
        "captured_at_utc": "2026-09-15T18:19:00Z", "grade": "regular", "grade_raw": "Regular",
        "price_raw": "¥149", "price": 149.0, "price_unit": "JPY/L", "currency": "JPY",
        "price_local_per_litre": 149.0, "price_usd_per_litre": 0.966,
        "price_usd_per_gallon": 3.6566, "fx_usd_per_unit": 0.006483402489,
        "fx_rate_date": "2026-09-15", "fx_source": "frankfurter-v2",
    },
    # No coordinates, so this station must not reach latest.json.
    {
        "station_key": "GB-Reading", "country": "GB", "name": "Reading", "name_local": None,
        "city": "Reading", "region": None, "lat": None, "lon": None,
        "captured_at_utc": "2026-09-15T18:18:00Z", "grade": "regular", "grade_raw": "5301",
        "price_raw": "160.9", "price": 160.9, "price_unit": "GBp/L", "currency": "GBP",
        "price_local_per_litre": 1.609, "price_usd_per_litre": 2.1793,
        "price_usd_per_gallon": 8.2487, "fx_usd_per_unit": 1.354,
        "fx_rate_date": "2026-09-15", "fx_source": "frankfurter-v2",
    },
]

STATION_ROWS = [
    {
        "station_key": "US-1364", "country": "US", "source_station_id": "1364", "alt_id": None,
        "name": "Bradenton", "name_local": None, "address": "805 LIGHTHOUSE DRIVE",
        "city": "BRADENTON", "region": "FL", "postcode": "34212",
        "lat": 27.49462445, "lon": -82.47014406, "timezone": "America/New_York",
        "grades_seen": "premium|regular", "first_seen_utc": "2026-09-01T18:17:00Z",
        "last_seen_utc": "2026-09-15T18:17:00Z", "status": "active", "superseded_by": None,
    },
    {
        "station_key": "US-140", "country": "US", "source_station_id": "140", "alt_id": None,
        "name": "Seattle", "name_local": None, "address": "4401 4TH AVE S",
        "city": "SEATTLE", "region": "WA", "postcode": "98134",
        "lat": 47.6, "lon": -122.3, "timezone": "America/Los_Angeles",
        "grades_seen": "clear|diesel|premium|regular", "first_seen_utc": "2026-09-01T18:17:00Z",
        "last_seen_utc": "2026-09-15T18:17:00Z", "status": "active", "superseded_by": None,
    },
    {
        "station_key": "JP-Tomiya", "country": "JP", "source_station_id": "Tomiya", "alt_id": "676",
        "name": "Tomiya", "name_local": "富谷倉庫店", "address": "宮城県富谷市成田9-1-1",
        "city": "富谷市", "region": "宮城県", "postcode": "981-3341",
        "lat": 38.39, "lon": 140.89, "timezone": "Asia/Tokyo",
        "grades_seen": "Diesel|Kerosene|Premium|Regular", "first_seen_utc": "2026-09-01T18:19:00Z",
        "last_seen_utc": "2026-09-15T18:19:00Z", "status": "active", "superseded_by": None,
    },
    {
        "station_key": "GB-Reading", "country": "GB", "source_station_id": "Reading", "alt_id": "5241",
        "name": "Reading", "name_local": None, "address": "1 Jenner Way",
        "city": "Reading", "region": None, "postcode": "RG2 0TF",
        "lat": None, "lon": None, "timezone": "Europe/London",
        "grades_seen": "5301|5302|5303", "first_seen_utc": "2026-09-01T18:18:00Z",
        "last_seen_utc": "2026-09-15T18:18:00Z", "status": "missing", "superseded_by": "GB-Reading2",
    },
]

MANIFEST = {
    "status": {
        "capture_id": "2026-09-15T1817Z",
        "countries": {
            "US": {"status": "ok", "last_success_capture_id": "2026-09-15T1817Z"},
            "JP": {"status": "degraded", "last_success_capture_id": "2026-09-15T1817Z"},
            "GB": {"status": "failed", "last_success_capture_id": "2026-09-14T1817Z"},
            "AU": {"status": "ok", "last_success_capture_id": "2026-09-15T1817Z"},
        },
    },
    "closed_months": ["2026-08"],
    "closed_years": [],
}

DAILY_ROWS = [
    # Four US stations on 2026-09-15: local 1, 2, 3, 4 with one null USD value.
    _daily_row(station_key="US-1", region="FL", price_local_per_litre=1.0, price_usd_per_litre=1.0),
    _daily_row(station_key="US-2", region="FL", price_local_per_litre=2.0, price_usd_per_litre=None),
    _daily_row(station_key="US-3", region="WA", price_local_per_litre=3.0, price_usd_per_litre=3.0),
    _daily_row(station_key="US-4", region="WA", price_local_per_litre=4.0, price_usd_per_litre=4.0),
    _daily_row(station_key="US-1", grade="premium", grade_raw="premium",
               price_local_per_litre=1.5, price_usd_per_litre=1.5),
    _daily_row(station_key="US-1", grade="other", grade_raw="clear",
               price_local_per_litre=1.6, price_usd_per_litre=1.6),
    # The UK has no regions, so it contributes country rows only.
    _daily_row(station_key="GB-Reading", country="GB", region=None, grade_raw="5301",
               price_local_per_litre=1.609, price_usd_per_litre=2.1793, currency="GBP"),
    # The Australian relabel, plus the previous day for the history sort order.
    _daily_row(station_key="AU-109", country="AU", region="NSW", grade_raw="E10",
               captured_at_utc=datetime(2026, 9, 15, 6, 17), price_local_per_litre=1.90,
               price_usd_per_litre=1.25, currency="AUD", n_captures=3),
    _daily_row(station_key="AU-109", country="AU", region="NSW", grade_raw="Unleaded 91",
               captured_at_utc=datetime(2026, 9, 15, 18, 17), price_local_per_litre=1.95,
               price_usd_per_litre=1.28, currency="AUD", n_captures=1),
    _daily_row(capture_date=date(2026, 9, 14), station_key="AU-109", country="AU", region="NSW",
               grade_raw="E10", price_local_per_litre=1.88, price_usd_per_litre=1.20,
               currency="AUD", n_captures=4),
]


@pytest.fixture
def current_dir(tmp_path) -> Path:
    directory = tmp_path / "current"
    directory.mkdir()
    _daily_frame(DAILY_ROWS).write_parquet(directory / "costco-gas-all.parquet")
    pl.DataFrame(LATEST_ROWS).write_csv(directory / "costco-gas-latest.csv")
    pl.DataFrame(STATION_ROWS).write_csv(directory / "stations.csv")
    (directory / "manifest.json").write_text(json.dumps(MANIFEST), encoding="utf-8")
    return directory


def test_latest_json_holds_every_displayed_field(current_dir, tmp_path):
    out = tmp_path / "data"
    build_site_data(current_dir, out, _cfg(), now=NOW)
    records = {r["station_key"]: r for r in json.loads((out / "latest.json").read_text(encoding="utf-8"))}

    assert set(records) == {"US-1364", "US-140", "JP-Tomiya"}  # GB-Reading has no coordinates

    bradenton = records["US-1364"]
    assert bradenton["country"] == "US"
    assert bradenton["name"] == "Bradenton"
    assert bradenton["city"] == "BRADENTON"
    assert bradenton["region"] == "FL"
    assert bradenton["lat"] == 27.49462445
    assert bradenton["status"] == "active"
    assert bradenton["first_seen_utc"] == "2026-09-01T18:17:00Z"
    assert bradenton["captured_at_utc"] == "2026-09-15T18:17:00Z"
    assert set(bradenton["grades"]) == {"regular", "premium"}
    assert bradenton["grades"]["regular"] == {
        "grade_raw": "regular", "price_raw": "3.999", "price": 3.999,
        "price_unit": "USD/gal", "currency": "USD", "price_local_per_litre": 1.0564,
        "price_usd_per_litre": 1.0564, "price_usd_per_gallon": 3.999,
        "fx_usd_per_unit": 1.0, "fx_rate_date": "2026-09-15", "fx_source": "identity",
    }
    assert bradenton["other"] == {}

    assert records["US-140"]["grades"] == {}
    assert records["US-140"]["other"]["clear"]["price_raw"] == "5.699"

    tokyo = records["JP-Tomiya"]
    assert tokyo["name_local"] == "富谷倉庫店"
    assert tokyo["grades"]["regular"]["price_raw"] == "¥149"
    assert tokyo["grades"]["regular"]["fx_usd_per_unit"] == 0.006483402489


def test_latest_json_rejects_a_duplicate_grade(current_dir, tmp_path):
    duplicate = LATEST_ROWS + [dict(LATEST_ROWS[0], price_raw="4.199", price=4.199)]
    pl.DataFrame(duplicate).write_csv(current_dir / "costco-gas-latest.csv")
    with pytest.raises(ValueError, match="US-1364"):
        build_site_data(current_dir, tmp_path / "data", _cfg(), now=NOW)


def test_stations_json_holds_every_search_field(current_dir, tmp_path):
    out = tmp_path / "data"
    build_site_data(current_dir, out, _cfg(), now=NOW)
    records = {r["station_key"]: r for r in json.loads((out / "stations.json").read_text(encoding="utf-8"))}

    assert set(records) == {"US-1364", "US-140", "JP-Tomiya", "GB-Reading"}
    reading = records["GB-Reading"]
    assert reading == {
        "station_key": "GB-Reading", "country": "GB", "name": "Reading", "name_local": None,
        "city": "Reading", "region": None, "lat": None, "lon": None, "status": "missing",
        "first_seen_utc": "2026-09-01T18:18:00Z", "last_seen_utc": "2026-09-15T18:18:00Z",
        "superseded_by": "GB-Reading2",
    }
