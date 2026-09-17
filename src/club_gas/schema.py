"""Column specifications, validators and file writers for every stored frame."""

from __future__ import annotations

import gzip
import io
from pathlib import Path

import polars as pl

UTC_DATETIME = pl.Datetime("us", "UTC")

# Written as "2026-09-15" and "2026-09-15T18:19:02Z" in every CSV we produce, so
# the files round-trip exactly and stay diffable.
CSV_DATE_FORMAT = "%Y-%m-%d"
CSV_DATETIME_FORMAT = "%Y-%m-%dT%H:%M:%SZ"

# Derived price columns are computed from unrounded intermediates and rounded
# only here, at write time.
PRICE_DECIMALS = 4
PRICE_COLUMNS = ("price_local_per_litre", "price_usd_per_litre", "price_usd_per_gallon")

# Rates are kept to significant digits, not decimals: JPY at 154.24 per USD is
# 0.006483 USD per yen, which 4-decimal rounding would turn into 0.0065, an
# error of 0.26%.
FX_SIGNIFICANT_DIGITS = 10
FX_RATE_COLUMNS = ("units_per_usd", "fx_usd_per_unit")

# One row per capture, station and grade. This dict is the file column order.
ROW_SCHEMA: dict[str, pl.DataType] = {
    "capture_id": pl.String(),
    "capture_date": pl.Date(),
    "captured_at_utc": UTC_DATETIME,
    "local_date": pl.Date(),
    "country": pl.String(),
    "station_key": pl.String(),
    "source_station_id": pl.String(),
    "source": pl.String(),
    "name": pl.String(),
    "name_local": pl.String(),
    "address": pl.String(),
    "city": pl.String(),
    "region": pl.String(),
    "postcode": pl.String(),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
    "timezone": pl.String(),
    "grade_raw": pl.String(),
    "grade": pl.String(),
    "price_raw": pl.String(),
    "price": pl.Float64(),
    "price_unit": pl.String(),
    "currency": pl.String(),
    "price_local_per_litre": pl.Float64(),
    "fx_usd_per_unit": pl.Float64(),
    "fx_rate_date": pl.Date(),
    "fx_source": pl.String(),
    "fx_fetched_at_utc": UTC_DATETIME,
    "price_usd_per_litre": pl.Float64(),
    "price_usd_per_gallon": pl.Float64(),
}

# Every column that must never hold a null. The rest are nullable by design:
# a station can lack a local name or coordinates, and every fx_* and USD column
# is null when the capture could not get an exchange rate.
REQUIRED_ROW_COLUMNS: tuple[str, ...] = (
    "capture_id",
    "capture_date",
    "captured_at_utc",
    "local_date",
    "country",
    "station_key",
    "source_station_id",
    "source",
    "name",
    "timezone",
    "grade_raw",
    "grade",
    "price_raw",
    "price",
    "price_unit",
    "currency",
    "price_local_per_litre",
)

ROW_KEY: tuple[str, ...] = ("capture_id", "station_key", "grade_raw")
ROW_SORT: list[str] = ["capture_id", "country", "station_key", "grade_raw"]

STATION_SCHEMA: dict[str, pl.DataType] = {
    "station_key": pl.String(),
    "country": pl.String(),
    "source_station_id": pl.String(),
    "alt_id": pl.String(),
    "name": pl.String(),
    "name_local": pl.String(),
    "address": pl.String(),
    "city": pl.String(),
    "region": pl.String(),
    "postcode": pl.String(),
    "lat": pl.Float64(),
    "lon": pl.Float64(),
    "timezone": pl.String(),
    "grades_seen": pl.String(),
    "first_seen_utc": UTC_DATETIME,
    "last_seen_utc": UTC_DATETIME,
    "status": pl.String(),
    "superseded_by": pl.String(),
}
STATION_KEY: tuple[str, ...] = ("station_key",)
STATION_SORT: list[str] = ["country", "station_key"]

FX_SCHEMA: dict[str, pl.DataType] = {
    "capture_id": pl.String(),
    "currency": pl.String(),
    "units_per_usd": pl.Float64(),
    "fx_usd_per_unit": pl.Float64(),
    "fx_rate_date": pl.Date(),
    "fx_source": pl.String(),
    "fx_fetched_at_utc": UTC_DATETIME,
}
FX_KEY: tuple[str, ...] = ("capture_id", "currency")
FX_SORT: list[str] = ["capture_id", "currency"]


class SchemaError(Exception):
    """A frame does not match the schema it is about to be written under."""


def empty_frame(schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """An empty frame with the right columns, for missing previous state."""
    return pl.DataFrame(schema=dict(schema))


def _check_schema(
    df: pl.DataFrame,
    schema: dict[str, pl.DataType],
    key: tuple[str, ...],
    required: tuple[str, ...],
) -> None:
    expected = set(schema)
    actual = set(df.columns)
    missing = sorted(expected - actual)
    if missing:
        raise SchemaError(f"missing columns: {', '.join(missing)}")
    unexpected = sorted(actual - expected)
    if unexpected:
        raise SchemaError(f"unexpected columns: {', '.join(unexpected)}")

    wrong = [
        f"{name}: expected {dtype}, got {df.schema[name]}"
        for name, dtype in schema.items()
        if df.schema[name] != dtype
    ]
    if wrong:
        raise SchemaError("wrong dtypes: " + "; ".join(wrong))

    for name in required:
        nulls = df[name].null_count()
        if nulls:
            raise SchemaError(f"null values in required column {name}: {nulls}")

    if df.height:
        key_cols = df.select(list(key))
        duplicated = key_cols.is_duplicated()
        if duplicated.any():
            duplicate = key_cols.filter(duplicated).head(1).row(0, named=True)
            raise SchemaError(f"duplicate key {tuple(key)}: {duplicate}")


def validate_rows(df: pl.DataFrame) -> None:
    """Raise SchemaError unless df can be written as price rows."""
    _check_schema(df, ROW_SCHEMA, ROW_KEY, REQUIRED_ROW_COLUMNS)


def validate_stations(df: pl.DataFrame) -> None:
    """Raise SchemaError unless df can be written as stations.csv."""
    _check_schema(df, STATION_SCHEMA, STATION_KEY, ("station_key", "country", "status"))


def validate_fx(df: pl.DataFrame) -> None:
    """Raise SchemaError unless df can be written as fx.csv."""
    _check_schema(df, FX_SCHEMA, FX_KEY, tuple(FX_SCHEMA))


def _validate_structure(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> None:
    """Validate columns and dtypes exist, without key or required-column checks."""
    expected = set(schema)
    actual = set(df.columns)
    missing = sorted(expected - actual)
    if missing:
        raise SchemaError(f"missing columns: {', '.join(missing)}")
    unexpected = sorted(actual - expected)
    if unexpected:
        raise SchemaError(f"unexpected columns: {', '.join(unexpected)}")

    wrong = [
        f"{name}: expected {dtype}, got {df.schema[name]}"
        for name, dtype in schema.items()
        if df.schema[name] != dtype
    ]
    if wrong:
        raise SchemaError("wrong dtypes: " + "; ".join(wrong))


def cast_to_schema(df: pl.DataFrame, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Select the schema's columns, in order, casting each to its dtype."""
    missing = [name for name in schema if name not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {', '.join(missing)}")
    return df.select(
        [pl.col(name).cast(dtype, strict=False).alias(name) for name, dtype in schema.items()]
    )


def round_significant(value: float | None, digits: int = FX_SIGNIFICANT_DIGITS) -> float | None:
    """Round one float to significant digits, exactly as the frame helper does."""
    if value is None:
        return None
    return pl.Series([float(value)]).round_sig_figs(digits).item()


def round_price_columns(df: pl.DataFrame, decimals: int = PRICE_DECIMALS) -> pl.DataFrame:
    """Round the derived price columns that are present."""
    present = [name for name in PRICE_COLUMNS if name in df.columns]
    if not present:
        return df
    return df.with_columns([pl.col(name).round(decimals) for name in present])


def round_fx_columns(df: pl.DataFrame, digits: int = FX_SIGNIFICANT_DIGITS) -> pl.DataFrame:
    """Round the exchange-rate columns that are present to significant digits."""
    present = [name for name in FX_RATE_COLUMNS if name in df.columns]
    if not present:
        return df
    return df.with_columns([pl.col(name).round_sig_figs(digits) for name in present])


def _prepare(df: pl.DataFrame, schema: dict[str, pl.DataType], sort_by: list[str]) -> pl.DataFrame:
    out = cast_to_schema(df, schema)
    out = round_price_columns(out)
    out = round_fx_columns(out)
    return out.sort(sort_by)


def _write_csv_bytes(df: pl.DataFrame) -> bytes:
    buf = io.BytesIO()
    df.write_csv(buf, date_format=CSV_DATE_FORMAT, datetime_format=CSV_DATETIME_FORMAT)
    return buf.getvalue()


def _read_csv_bytes(raw: bytes, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    # Every field is read as text and cast explicitly, so no column's type
    # depends on how the first rows happen to look.
    df = pl.read_csv(raw, infer_schema_length=0)
    missing = [name for name in schema if name not in df.columns]
    if missing:
        raise SchemaError(f"missing columns: {', '.join(missing)}")
    exprs = []
    for name, dtype in schema.items():
        col = pl.col(name)
        if dtype == pl.Date():
            exprs.append(col.str.to_date(CSV_DATE_FORMAT).alias(name))
        elif isinstance(dtype, pl.Datetime):
            exprs.append(
                col.str.to_datetime(CSV_DATETIME_FORMAT, time_unit="us")
                .dt.replace_time_zone("UTC")
                .alias(name)
            )
        elif dtype == pl.String():
            exprs.append(col.alias(name))
        else:
            exprs.append(col.cast(dtype).alias(name))
    return df.select(exprs)


def _write_gzip(raw: bytes, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    # mtime=0 and an empty stored filename keep the bytes identical for
    # identical content, so re-publishing a capture does not change any digest.
    with path.open("wb") as fh, gzip.GzipFile(filename="", mode="wb", fileobj=fh, mtime=0) as gz:
        gz.write(raw)
    return path


def write_rows_csv_gz(df: pl.DataFrame, path: Path) -> Path:
    """Validate, round, sort and write price rows as gzipped CSV."""
    validate_rows(df)
    return _write_gzip(_write_csv_bytes(_prepare(df, ROW_SCHEMA, ROW_SORT)), Path(path))


def read_rows_csv_gz(path: Path) -> pl.DataFrame:
    """Read a gzipped CSV of price rows back under ROW_SCHEMA."""
    with gzip.open(Path(path), "rb") as fh:
        raw = fh.read()
    return _read_csv_bytes(raw, ROW_SCHEMA)


def write_csv(
    df: pl.DataFrame, path: Path, *, schema: dict[str, pl.DataType], sort_by: list[str]
) -> Path:
    """Write a plain CSV asset such as stations.csv or fx.csv, validating before writing."""
    # Validate based on schema type
    if schema is ROW_SCHEMA:
        validate_rows(df)
    elif schema is STATION_SCHEMA:
        validate_stations(df)
    elif schema is FX_SCHEMA:
        validate_fx(df)
    else:
        _validate_structure(df, schema)

    out = _prepare(df, schema, sort_by)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(_write_csv_bytes(out))
    return path


def read_csv(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    """Read a plain CSV asset back under an explicit schema."""
    return _read_csv_bytes(Path(path).read_bytes(), schema)


def write_parquet(
    df: pl.DataFrame,
    path: Path,
    *,
    sort_by: list[str],
    row_group_size: int | None = None,
) -> Path:
    """Write a Parquet file, validating that sort_by columns exist before writing."""
    # Validate that all sort_by columns exist in the frame
    missing_sort_cols = [col for col in sort_by if col not in df.columns]
    if missing_sort_cols:
        raise SchemaError(f"sort_by columns not in frame: {', '.join(missing_sort_cols)}")

    out = round_fx_columns(round_price_columns(df)).sort(sort_by)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out.write_parquet(
        path,
        compression="zstd",
        statistics=True,
        row_group_size=row_group_size,
    )
    return path
