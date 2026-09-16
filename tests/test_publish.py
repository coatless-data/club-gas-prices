"""Tests for costco_gas.publish (spec 8.4 recovery, 8.5 publish)."""

from __future__ import annotations

import gzip
import hashlib
import json
import shutil
import tarfile
from datetime import date, datetime, timedelta, timezone
from pathlib import Path

import polars as pl
import pytest

from costco_gas import rollup, schema

REPO_ROOT = Path(__file__).resolve().parents[1]

# The 2026-09-15 Taiwan prices, per litre in TWD, with the Frankfurter rate of
# 31.709 TWD per USD from the same day.
TWD_PER_USD = 31.709
USD_PER_TWD = 1.0 / TWD_PER_USD


def price_row(capture_id, captured_at, station, grade_raw, grade, price):
    return {
        "capture_id": capture_id,
        "capture_date": captured_at.date(),
        "captured_at_utc": captured_at,
        "local_date": (captured_at + timedelta(hours=8)).date(),
        "country": "TW",
        "station_key": f"TW-{station}",
        "source_station_id": station,
        "source": "costco-occ",
        "name": station,
        "name_local": None,
        "address": None,
        "city": None,
        "region": "桃園市",
        "postcode": "320",
        "lat": 24.9573,
        "lon": 121.2196,
        "timezone": "Asia/Taipei",
        "grade_raw": grade_raw,
        "grade": grade,
        "price_raw": f"${price}",
        "price": price,
        "price_unit": "TWD/L",
        "currency": "TWD",
        "price_local_per_litre": price,
        "fx_usd_per_unit": USD_PER_TWD,
        "fx_rate_date": date(2026, 9, 14),
        "fx_source": "frankfurter-v2",
        "fx_fetched_at_utc": captured_at,
        "price_usd_per_litre": round(price * USD_PER_TWD, 4),
        "price_usd_per_gallon": round(price * USD_PER_TWD * 3.785411784, 4),
    }


def test_daily_grain_keeps_the_last_capture_of_the_day():
    early = datetime(2026, 9, 15, 0, 17, tzinfo=timezone.utc)
    late = datetime(2026, 9, 15, 18, 17, tzinfo=timezone.utc)
    rows = pl.DataFrame(
        [
            price_row("2026-09-15T0017Z", early, "Chungli", "95", "regular", 30.0),
            price_row("2026-09-15T1817Z", late, "Chungli", "95", "regular", 30.5),
            price_row("2026-09-15T1817Z", late, "Chungli", "98", "premium", 31.5),
        ],
        schema=schema.ROW_SCHEMA,
    )

    daily = rollup.daily_grain(rows)

    assert daily.height == 2
    regular = daily.filter(pl.col("grade_raw") == "95").to_dicts()[0]
    assert regular["capture_id"] == "2026-09-15T1817Z"
    assert regular["price"] == 30.5
    assert regular["n_captures"] == 2
    assert regular["price_min"] == 30.0
    assert regular["price_max"] == 30.5
    premium = daily.filter(pl.col("grade_raw") == "98").to_dicts()[0]
    assert premium["n_captures"] == 1
    assert premium["price_min"] == premium["price_max"] == 31.5
    assert list(daily.columns) == list(rollup.DAILY_SCHEMA)


def test_daily_grain_of_an_empty_frame_keeps_the_schema():
    empty = rollup.daily_grain(pl.DataFrame(schema=schema.ROW_SCHEMA))
    assert empty.height == 0
    assert dict(empty.schema) == dict(rollup.DAILY_SCHEMA)
