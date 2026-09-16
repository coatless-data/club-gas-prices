"""Roll-ups over price rows (spec 6.2, 8.6).

Only the daily grain lives here for now; the period-closing functions are
appended to this module by a later task, which keeps DAILY_KEYS, DAILY_SCHEMA
and daily_grain exactly as they are -- publish.py reads all three.
"""

from __future__ import annotations

import polars as pl

from . import schema

DAILY_KEYS = ["capture_date", "station_key", "grade_raw"]
DAILY_SCHEMA: dict[str, pl.DataType] = {
    **schema.ROW_SCHEMA,
    "n_captures": pl.Int64,
    "price_min": pl.Float64,
    "price_max": pl.Float64,
}


def daily_grain(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per (capture_date, station_key, grade_raw), spec 6.2.

    The kept row is the one with the greatest `captured_at_utc` that day;
    `n_captures`, `price_min` and `price_max` summarise the whole day.
    """
    if rows.height == 0:
        return pl.DataFrame(schema=DAILY_SCHEMA)
    summary = rows.group_by(DAILY_KEYS).agg(
        pl.col("capture_id").n_unique().cast(pl.Int64).alias("n_captures"),
        pl.col("price").min().alias("price_min"),
        pl.col("price").max().alias("price_max"),
    )
    newest = rows.sort(["captured_at_utc", "capture_id"]).group_by(DAILY_KEYS).last()
    return (
        newest.join(summary, on=DAILY_KEYS, how="left")
        .select(list(DAILY_SCHEMA))
        .sort(["capture_date", "country", "station_key", "grade_raw"])
    )
