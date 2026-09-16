import datetime as dt

import polars as pl
import pytest

from costco_gas.schema import (
    FX_SCHEMA,
    FX_SORT,
    ROW_SCHEMA,
    ROW_SORT,
    STATION_SCHEMA,
    STATION_SORT,
    SchemaError,
    cast_to_schema,
    empty_frame,
    read_csv,
    read_rows_csv_gz,
    round_fx_columns,
    round_price_columns,
    round_significant,
    validate_fx,
    validate_rows,
    validate_stations,
    write_csv,
    write_parquet,
    write_rows_csv_gz,
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


def us_row(**overrides: object) -> dict[str, object]:
    """One real US row: #1364 regular at 3.999 USD/gal."""
    litres = 3.999 / GALLON_L
    base = row(
        country="US",
        station_key="US-1364",
        source_station_id="1364",
        source="costco-us-gasprices",
        name="Bloomington",
        name_local=None,
        address=None,
        city="BLOOMINGTON",
        region="MN",
        postcode="55425",
        lat=44.855,
        lon=-93.239,
        timezone="America/Chicago",
        grade_raw="regular",
        grade="regular",
        price_raw="3.999",
        price=3.999,
        price_unit="USD/gal",
        currency="USD",
        price_local_per_litre=litres,
        fx_usd_per_unit=1.0,
        fx_source="identity",
        fx_rate_date=dt.date(2026, 9, 15),
        local_date=dt.date(2026, 9, 15),
        price_usd_per_litre=litres,
        price_usd_per_gallon=litres * GALLON_L,
    )
    base.update(overrides)
    return base


def fx_frame() -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "capture_id": CAPTURE_ID,
                "currency": "JPY",
                "units_per_usd": JPY_PER_USD,
                "fx_usd_per_unit": USD_PER_JPY,
                "fx_rate_date": dt.date(2026, 9, 14),
                "fx_source": "frankfurter-v2",
                "fx_fetched_at_utc": CAPTURED_AT,
            }
        ],
        schema=dict(FX_SCHEMA),
        orient="row",
    )


def test_round_significant_keeps_ten_digits_of_a_small_rate():
    # 4-decimal rounding would give 0.0065, an error of 0.26%.
    assert round_significant(USD_PER_JPY) == 0.00648340249
    assert round_significant(1 / 1.3871) == 0.720928556
    assert round_significant(None) is None
    assert round_significant(0.0) == 0.0


def test_round_price_columns_rounds_to_four_decimals():
    out = round_price_columns(frame(row()))
    assert out["price_usd_per_litre"].item() == 0.966
    assert out["price_usd_per_gallon"].item() == 3.6568


def test_a_usd_per_gallon_row_keeps_its_published_value_after_rounding():
    # The gallon value travels through litres, so it arrives as
    # 3.9990000000000006. Rounding at write time restores the published value.
    out = round_price_columns(frame(us_row()))
    assert out["price_usd_per_gallon"].item() == 3.999
    assert out["price_local_per_litre"].item() == 1.0564


def test_round_fx_columns_uses_significant_digits():
    out = round_fx_columns(fx_frame())
    assert out["units_per_usd"].item() == 154.24
    assert out["fx_usd_per_unit"].item() == 0.00648340249


def test_cast_to_schema_requires_every_column():
    df = pl.DataFrame({"currency": ["JPY"], "units_per_usd": ["154.24"]})
    with pytest.raises(SchemaError, match="missing columns"):
        cast_to_schema(df, FX_SCHEMA)


def test_cast_to_schema_reorders_and_casts_strings():
    df = pl.DataFrame({"country": ["JP"], "lat": ["38.4"], "station_key": ["JP-Tomiya"]})
    out = cast_to_schema(
        df, {"station_key": pl.String(), "country": pl.String(), "lat": pl.Float64()}
    )
    assert out.columns == ["station_key", "country", "lat"]
    assert out["lat"].item() == 38.4


def test_rows_csv_gz_round_trips_and_is_sorted(tmp_path):
    df = frame(us_row(), row())
    path = write_rows_csv_gz(df, tmp_path / "costco-gas-2026-09-15.csv.gz")
    back = read_rows_csv_gz(path)

    assert back.schema == pl.Schema(ROW_SCHEMA)
    # ROW_SORT is capture_id, country, station_key, grade_raw: JP before US.
    assert back["country"].to_list() == ["JP", "US"]
    assert back["name_local"].to_list() == ["富谷", None]
    assert back["captured_at_utc"].to_list() == [CAPTURED_AT, CAPTURED_AT]
    assert back["fx_rate_date"].to_list() == [dt.date(2026, 9, 14), dt.date(2026, 9, 15)]
    assert back["price_usd_per_gallon"].to_list() == [3.6568, 3.999]
    assert back["fx_usd_per_unit"].to_list() == [0.00648340249, 1.0]
    assert back["lat"].to_list() == [None, 44.855]


def test_rows_csv_gz_is_byte_identical_for_identical_content(tmp_path):
    df = frame(row())
    write_rows_csv_gz(df, tmp_path / "a.csv.gz")
    write_rows_csv_gz(df, tmp_path / "b.csv.gz")
    # Re-publishing a capture must not change any asset's SHA-256, so the gzip
    # header stores no timestamp and no original filename.
    assert (tmp_path / "a.csv.gz").read_bytes() == (tmp_path / "b.csv.gz").read_bytes()


def test_rows_csv_gz_refuses_an_invalid_frame(tmp_path):
    with pytest.raises(SchemaError, match="duplicate key"):
        write_rows_csv_gz(frame(row(), row()), tmp_path / "never-written.csv.gz")
    assert not (tmp_path / "never-written.csv.gz").exists()


def test_write_parquet_sorts_explicitly(tmp_path):
    df = frame(row(station_key="JP-Zama", source_station_id="Zama", name="Zama"), row())
    path = write_parquet(
        df,
        tmp_path / "captures.parquet",
        sort_by=["station_key", "grade_raw", "capture_id"],
        row_group_size=20000,
    )
    back = pl.read_parquet(path)
    assert back["station_key"].to_list() == ["JP-Tomiya", "JP-Zama"]
    assert back.schema == pl.Schema(ROW_SCHEMA)
    assert back["price_usd_per_gallon"].to_list() == [3.6568, 3.6568]


def test_stations_csv_round_trips(tmp_path):
    stations = pl.DataFrame(
        [
            {
                "station_key": "JP-Tomiya",
                "country": "JP",
                "source_station_id": "Tomiya",
                "alt_id": "costcoJapanTomiyaWarehouse",
                "name": "Tomiya",
                "name_local": "富谷",
                "address": "宮城県富谷市高屋敷26",
                "city": "富谷市",
                "region": "宮城県",
                "postcode": "981-3313",
                "lat": None,
                "lon": None,
                "timezone": "Asia/Tokyo",
                "grades_seen": "Diesel|Kerosene|Premium|Regular",
                "first_seen_utc": CAPTURED_AT,
                "last_seen_utc": CAPTURED_AT,
                "status": "active",
                "superseded_by": None,
            }
        ],
        schema=dict(STATION_SCHEMA),
        orient="row",
    )
    validate_stations(stations)
    path = write_csv(
        stations, tmp_path / "stations.csv", schema=STATION_SCHEMA, sort_by=STATION_SORT
    )
    back = read_csv(path, STATION_SCHEMA)
    assert back.schema == pl.Schema(STATION_SCHEMA)
    assert back["superseded_by"].to_list() == [None]
    assert back["grades_seen"].item() == "Diesel|Kerosene|Premium|Regular"
    assert back["first_seen_utc"].item() == CAPTURED_AT


def test_fx_csv_round_trips_with_ten_significant_digits(tmp_path):
    fx = fx_frame()
    validate_fx(fx)
    path = write_csv(fx, tmp_path / "fx.csv", schema=FX_SCHEMA, sort_by=FX_SORT)
    assert "0.00648340249" in path.read_text(encoding="utf-8")
    back = read_csv(path, FX_SCHEMA)
    assert back["fx_usd_per_unit"].item() == 0.00648340249
    assert back["units_per_usd"].item() == 154.24
