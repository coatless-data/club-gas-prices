"""Build the files the Quarto dashboard reads (spec 9.1).

Render downloads the ``current`` release assets into one directory and calls
this command. The output is shipped inside the Pages site, because GitHub
release downloads send no CORS headers and the browser cannot fetch them.

Nothing here recomputes a USD value: every price row carries the exchange rate
that was in effect when it was collected, so a past day keeps its past USD
price. Rows with a null USD value are excluded from USD statistics and counted
in ``n_stations_usd``.
"""

from __future__ import annotations

from datetime import date, datetime

import polars as pl

DEDUPE_KEY = ["capture_date", "station_key", "grade"]


def dedupe_daily(rows: pl.DataFrame, cfg) -> pl.DataFrame:
    """One row per (capture_date, station_key, grade), ``other`` grades dropped.

    A station can relabel a grade between captures on the same day (an
    Australian station listing ``E10`` in the morning and ``Unleaded 91`` in
    the evening), which leaves two daily-grain rows mapping to ``regular``. The
    later capture wins; a tie is broken by the higher ``priority`` in
    ``config/grades.csv``.
    """
    rows = rows.filter(pl.col("grade") != "other")
    priorities = _priority_frame(rows, cfg)
    rows = rows.join(priorities, on=["country", "grade_raw"], how="left").with_columns(
        pl.col("_priority").fill_null(0)
    )
    return (
        rows.sort(
            ["capture_date", "station_key", "grade", "captured_at_utc", "_priority"],
            descending=[False, False, False, True, True],
        )
        .unique(subset=DEDUPE_KEY, keep="first", maintain_order=True)
        .drop("_priority")
    )


def _priority_frame(rows: pl.DataFrame, cfg) -> pl.DataFrame:
    pairs = rows.select("country", "grade_raw").unique().rows()
    table = getattr(cfg, "grades", None)
    priorities = []
    for country, grade_raw in pairs:
        entry = table.map(country, grade_raw) if table is not None else None
        priorities.append(int(getattr(entry, "priority", 0) or 0))
    return pl.DataFrame(
        {
            "country": [pair[0] for pair in pairs],
            "grade_raw": [pair[1] for pair in pairs],
            "_priority": priorities,
        },
        schema={"country": pl.String, "grade_raw": pl.String, "_priority": pl.Int64},
    )


def _json_default(value: object) -> object:
    if isinstance(value, datetime):
        text = value.replace(microsecond=0).isoformat()
        return text.replace("+00:00", "Z") if value.tzinfo else text + "Z"
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"cannot serialise {type(value).__name__} to JSON")
