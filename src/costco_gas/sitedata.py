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

import json
from datetime import date, datetime
from pathlib import Path

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


LATEST_NUMERIC = (
    "price",
    "price_local_per_litre",
    "price_usd_per_litre",
    "price_usd_per_gallon",
    "fx_usd_per_unit",
    "lat",
    "lon",
)
STATION_NUMERIC = ("lat", "lon")
GRADE_FIELDS = (
    "grade_raw",
    "price_raw",
    "price",
    "price_unit",
    "currency",
    "price_local_per_litre",
    "price_usd_per_litre",
    "price_usd_per_gallon",
    "fx_usd_per_unit",
    "fx_rate_date",
    "fx_source",
)
STATION_CORE = ("station_key", "country", "name", "name_local", "city", "region", "lat", "lon")
STATION_JSON_FIELDS = (
    *STATION_CORE,
    "status",
    "first_seen_utc",
    "last_seen_utc",
    "superseded_by",
)
CORE_GRADES = ("regular", "premium", "diesel")


def build_site_data(current_dir: Path, out_dir: Path, cfg, *, now: datetime) -> None:
    current_dir = Path(current_dir)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stations = _read_csv(current_dir / "stations.csv", STATION_NUMERIC)
    latest = _read_csv(current_dir / "costco-gas-latest.csv", LATEST_NUMERIC)

    _write_json(out_dir / "latest.json", _latest_records(latest, stations))
    _write_json(out_dir / "stations.json", _station_records(stations))


def _read_csv(path: Path, numeric: tuple[str, ...]) -> pl.DataFrame:
    frame = pl.read_csv(path, try_parse_dates=True)
    return frame.with_columns(
        [pl.col(name).cast(pl.Float64, strict=False) for name in numeric if name in frame.columns]
    )


def _latest_records(latest: pl.DataFrame, stations: pl.DataFrame) -> list[dict]:
    meta = {row["station_key"]: row for row in stations.to_dicts()}
    records: dict[str, dict] = {}
    seen: set[tuple[str, str]] = set()
    for row in latest.to_dicts():
        if row.get("lat") is None or row.get("lon") is None:
            continue
        key = row["station_key"]
        record = records.get(key)
        if record is None:
            info = meta.get(key, {})
            record = {field: row.get(field) for field in STATION_CORE}
            record["status"] = info.get("status")
            record["first_seen_utc"] = info.get("first_seen_utc")
            record["captured_at_utc"] = row.get("captured_at_utc")
            record["grades"] = {}
            record["other"] = {}
            records[key] = record
        price = {field: row.get(field) for field in GRADE_FIELDS}
        grade = row.get("grade")
        if grade in CORE_GRADES:
            if (key, grade) in seen:
                raise ValueError(
                    f"costco-gas-latest.csv has more than one {grade!r} row for {key}"
                )
            seen.add((key, grade))
            record["grades"][grade] = price
        else:
            record["other"][row.get("grade_raw")] = price
    return list(records.values())


def _station_records(stations: pl.DataFrame) -> list[dict]:
    return [
        {field: row.get(field) for field in STATION_JSON_FIELDS} for row in stations.to_dicts()
    ]


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), default=_json_default)
        handle.write("\n")
