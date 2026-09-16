import datetime as dt

import polars as pl
import pytest

from costco_gas.schema import (
    FX_SCHEMA,
    ROW_SCHEMA,
    ROW_SORT,
    STATION_SCHEMA,
    SchemaError,
    empty_frame,
    validate_rows,
)

CAPTURE_ID = "2026-09-15T1817Z"
CAPTURED_AT = dt.datetime(2026, 9, 15, 18, 19, 2, tzinfo=dt.UTC)
# Frankfurter v2 returned USD->JPY 154.24 for rate date 2026-09-14.
JPY_PER_USD = 154.24
USD_PER_JPY = 1.0 / JPY_PER_USD
GALLON_L = 3.785411784


def row(**overrides: object) -> dict[str, object]:
    """One real JP row: Tomiya Regular at 149 JPY/L on 2026-09-15."""
    base: dict[str, object] = {
        "capture_id": CAPTURE_ID,
        "capture_date": dt.date(2026, 9, 15),
        "captured_at_utc": CAPTURED_AT,
        "local_date": dt.date(2026, 9, 16),
        "country": "JP",
        "station_key": "JP-Tomiya",
        "source_station_id": "Tomiya",
        "source": "costco-occ",
        "name": "Tomiya",
        "name_local": "富谷",
        "address": "宮城県富谷市高屋敷26",
        "city": "富谷市",
        "region": "宮城県",
        "postcode": "981-3313",
        "lat": None,
        "lon": None,
        "timezone": "Asia/Tokyo",
        "grade_raw": "Regular",
        "grade": "regular",
        "price_raw": "¥149",
        "price": 149.0,
        "price_unit": "JPY/L",
        "currency": "JPY",
        "price_local_per_litre": 149.0,
        "fx_usd_per_unit": USD_PER_JPY,
        "fx_rate_date": dt.date(2026, 9, 14),
        "fx_source": "frankfurter-v2",
        "fx_fetched_at_utc": CAPTURED_AT,
        "price_usd_per_litre": 149.0 * USD_PER_JPY,
        "price_usd_per_gallon": 149.0 * USD_PER_JPY * GALLON_L,
    }
    base.update(overrides)
    return base


def frame(*rows: dict[str, object]) -> pl.DataFrame:
    return pl.DataFrame(list(rows), schema=dict(ROW_SCHEMA), orient="row")


def test_row_schema_column_order_and_dtypes():
    names = list(ROW_SCHEMA)
    assert names[:5] == ["capture_id", "capture_date", "captured_at_utc", "local_date", "country"]
    assert names[-2:] == ["price_usd_per_litre", "price_usd_per_gallon"]
    assert len(names) == 30
    assert ROW_SCHEMA["captured_at_utc"] == pl.Datetime("us", "UTC")
    assert ROW_SCHEMA["lat"] == pl.Float64()
    assert next(iter(STATION_SCHEMA)) == "station_key"
    assert list(FX_SCHEMA) == [
        "capture_id",
        "currency",
        "units_per_usd",
        "fx_usd_per_unit",
        "fx_rate_date",
        "fx_source",
        "fx_fetched_at_utc",
    ]


def test_validate_rows_accepts_a_good_frame_and_an_empty_one():
    validate_rows(frame(row()))
    validate_rows(empty_frame(ROW_SCHEMA))


def test_validate_rows_rejects_a_missing_column():
    with pytest.raises(SchemaError, match="missing columns: price_unit"):
        validate_rows(frame(row()).drop("price_unit"))


def test_validate_rows_rejects_an_unexpected_column():
    df = frame(row()).with_columns(pl.lit(1).alias("extra"))
    with pytest.raises(SchemaError, match="unexpected columns: extra"):
        validate_rows(df)


def test_validate_rows_rejects_a_wrong_dtype():
    df = frame(row()).with_columns(pl.col("price").cast(pl.Float32))
    with pytest.raises(SchemaError, match="wrong dtypes: price"):
        validate_rows(df)


def test_validate_rows_rejects_a_null_in_a_required_column():
    df = frame(row()).with_columns(pl.lit(None, dtype=pl.String).alias("timezone"))
    with pytest.raises(SchemaError, match="null values in required column timezone"):
        validate_rows(df)


def test_validate_rows_allows_nulls_in_the_nullable_columns():
    validate_rows(
        frame(
            row(
                name_local=None,
                address=None,
                city=None,
                region=None,
                postcode=None,
                fx_usd_per_unit=None,
                fx_rate_date=None,
                fx_source=None,
                fx_fetched_at_utc=None,
                price_usd_per_litre=None,
                price_usd_per_gallon=None,
            )
        )
    )


def test_validate_rows_rejects_a_duplicate_key():
    with pytest.raises(SchemaError, match="duplicate key"):
        validate_rows(frame(row(), row(name="Tomiya again")))


def test_two_grades_at_one_station_are_not_a_duplicate():
    validate_rows(frame(row(), row(grade_raw="Premium", grade="premium", price=159.0)))


def test_row_sort_is_the_daily_file_order():
    assert ROW_SORT == ["capture_id", "country", "station_key", "grade_raw"]
