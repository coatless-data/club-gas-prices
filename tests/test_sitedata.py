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
