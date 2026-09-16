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
import os
from datetime import date, datetime
from pathlib import Path

import polars as pl

from .schema import write_parquet

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

    deduped = dedupe_daily(pl.read_parquet(current_dir / "costco-gas-all.parquet"), cfg)
    write_parquet(summary_daily(deduped), out_dir / "summary_daily.parquet", sort_by=SUMMARY_SORT)
    write_parquet(
        history(deduped),
        out_dir / "history.parquet",
        sort_by=HISTORY_SORT,
        row_group_size=HISTORY_ROW_GROUP_SIZE,
    )

    _write_json(out_dir / "meta.json", _meta(current_dir, cfg, now=now))


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
                raise ValueError(f"costco-gas-latest.csv has more than one {grade!r} row for {key}")
            seen.add((key, grade))
            record["grades"][grade] = price
        else:
            record["other"][row.get("grade_raw")] = price
    return list(records.values())


def _station_records(stations: pl.DataFrame) -> list[dict]:
    return [{field: row.get(field) for field in STATION_JSON_FIELDS} for row in stations.to_dicts()]


def _write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, separators=(",", ":"), default=_json_default)
        handle.write("\n")


# The dashboard's DuckDB queries select these names, in this order.
SUMMARY_COLUMNS = [
    "capture_date",
    "country",
    "level",
    "region",
    "grade",
    "n_stations",
    "n_stations_usd",
    "median_local_per_litre",
    "p25_local_per_litre",
    "p75_local_per_litre",
    "median_usd_per_litre",
    "p25_usd_per_litre",
    "p75_usd_per_litre",
]
SUMMARY_KEY = ["capture_date", "country", "level", "region", "grade"]
SUMMARY_SORT = ["country", "level", "region", "grade", "capture_date"]
HISTORY_COLUMNS = [
    "capture_date",
    "station_key",
    "grade",
    "price_local_per_litre",
    "price_usd_per_litre",
    "currency",
    "n_captures",
]
HISTORY_SORT = ["station_key", "grade", "capture_date"]
HISTORY_ROW_GROUP_SIZE = 20000


def summary_daily(deduped: pl.DataFrame) -> pl.DataFrame:
    country = _aggregate(deduped, ["capture_date", "country", "grade"], "country")
    # Region rows use only the stations that have a region; the UK has none.
    regional = _aggregate(
        deduped.filter(pl.col("region").is_not_null()),
        ["capture_date", "country", "region", "grade"],
        "region",
    )
    frame = pl.concat([country, regional], how="vertical").sort(SUMMARY_SORT, nulls_last=True)
    _check_unique(frame)
    return frame


def _aggregate(frame: pl.DataFrame, keys: list[str], level: str) -> pl.DataFrame:
    # The counts are Int32 because that is the width the dashboard contract
    # declares for n_stations and n_stations_usd.
    out = frame.group_by(keys).agg(
        pl.col("station_key").n_unique().cast(pl.Int32).alias("n_stations"),
        pl.col("price_usd_per_litre").is_not_null().sum().cast(pl.Int32).alias("n_stations_usd"),
        pl.col("price_local_per_litre").median().alias("median_local_per_litre"),
        pl.col("price_local_per_litre")
        .quantile(0.25, interpolation="linear")
        .alias("p25_local_per_litre"),
        pl.col("price_local_per_litre")
        .quantile(0.75, interpolation="linear")
        .alias("p75_local_per_litre"),
        # USD statistics ignore the rows whose capture had no exchange rate.
        pl.col("price_usd_per_litre").drop_nulls().median().alias("median_usd_per_litre"),
        pl.col("price_usd_per_litre")
        .drop_nulls()
        .quantile(0.25, interpolation="linear")
        .alias("p25_usd_per_litre"),
        pl.col("price_usd_per_litre")
        .drop_nulls()
        .quantile(0.75, interpolation="linear")
        .alias("p75_usd_per_litre"),
    )
    out = out.with_columns(pl.lit(level, dtype=pl.String).alias("level"))
    if "region" not in out.columns:
        out = out.with_columns(pl.lit(None, dtype=pl.String).alias("region"))
    return out.select(SUMMARY_COLUMNS)


def _check_unique(frame: pl.DataFrame) -> None:
    duplicates = frame.group_by(SUMMARY_KEY).len().filter(pl.col("len") > 1)
    if duplicates.height:
        raise ValueError(
            f"summary_daily is not unique on {SUMMARY_KEY}: {duplicates.head(5).to_dicts()}"
        )


def history(deduped: pl.DataFrame) -> pl.DataFrame:
    return deduped.select(HISTORY_COLUMNS)


DEFAULT_RELEASE_BASE_URL = "https://github.com/coatless-dashboard/costco-gas-prices/releases"
DEFAULT_NOTICE = (
    "Unofficial. Not affiliated with, endorsed by, or connected to Costco Wholesale "
    "Corporation. Prices are collected from Costco's public websites and may differ "
    "from the price at the pump."
)
DEFAULT_BASEMAP_KEY_ENV = "CARTO_BASEMAP_KEY"
KEYED_PROVIDER = "carto"
FALLBACK_PROVIDER = "osm"
DEFAULT_MAX_ZOOM = 19
# The dashboard's freshness notice reads this instead of hardcoding 12 hours.
STALE_AFTER_HOURS = 12
CURRENT_ASSETS = {
    "all_parquet": "costco-gas-all.parquet",
    "all_csv_gz": "costco-gas-all.csv.gz",
    "all_captures_parquet": "costco-gas-all-captures.parquet",
    "latest_csv": "costco-gas-latest.csv",
    "stations_csv": "stations.csv",
    "fx_csv": "fx.csv",
}
GRADE_TABLE_FIELDS = (
    "country",
    "grade_raw",
    "grade",
    "priority",
    "label",
    "spec",
    "spec_source",
    "spec_source_url",
)


def _meta(current_dir: Path, cfg, *, now: datetime) -> dict:
    manifest = _read_json(current_dir / "manifest.json")
    status = _manifest_status(manifest)
    countries = {
        code: {
            "status": entry.get("status"),
            "last_success_capture_id": entry.get("last_success_capture_id"),
        }
        for code, entry in (status.get("countries") or {}).items()
    }
    base = str(_site_value(cfg, "release_base_url", DEFAULT_RELEASE_BASE_URL)).rstrip("/")
    return {
        "built_at_utc": _json_default(now),
        "capture_id": status.get("capture_id"),
        "countries": countries,
        "closed_months": manifest.get("closed_months") or [],
        "closed_years": manifest.get("closed_years") or [],
        "grades": _grade_table(cfg),
        "notice": _site_value(cfg, "notice", DEFAULT_NOTICE),
        "stale_after_hours": STALE_AFTER_HOURS,
        "releases": {
            "current": f"{base}/tag/current",
            "all": base,
            **{key: f"{base}/download/current/{name}" for key, name in CURRENT_ASSETS.items()},
        },
        "basemap": _basemap(cfg),
    }


def _read_json(path: Path) -> dict:
    if not path.exists():
        print(f"::warning::{path.name} is missing; meta.json will carry no capture status")
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except ValueError as exc:
        print(f"::warning::{path.name} is not valid JSON: {exc}")
        return {}
    return payload if isinstance(payload, dict) else {}


def _manifest_status(manifest: dict) -> dict:
    for key in ("status", "status_json"):
        if isinstance(manifest.get(key), dict):
            return manifest[key]
    return manifest if "countries" in manifest else {}


def _site_value(cfg, name: str, default):
    site = getattr(cfg, "site", None)
    value = getattr(site, name, None)
    if value is None and isinstance(site, dict):
        value = site.get(name)
    return default if value in (None, "") else value


def _grade_table(cfg) -> list[dict]:
    grades = getattr(cfg, "grades", None)
    rows = getattr(grades, "rows", None)
    if callable(rows):
        rows = rows()
    if rows is None:
        return []
    if isinstance(rows, pl.DataFrame):
        rows = rows.to_dicts()
    if isinstance(rows, dict):
        rows = list(rows.values())
    table = []
    for entry in rows:
        if isinstance(entry, dict):
            table.append({field: entry.get(field) for field in GRADE_TABLE_FIELDS})
        else:
            table.append({field: getattr(entry, field, None) for field in GRADE_TABLE_FIELDS})
    return table


def _basemap(cfg) -> dict:
    """The basemap block the dashboard reads (spec 9.2).

    CARTO's raster basemaps need an API key since 2026-08-14 and keyless tiles
    are watermarked, so the OSM fallback is used when no key is configured. The
    ``{key}`` placeholder is substituted here, because the browser only ever
    sees the finished URL.
    """
    env_name = str(_site_value(cfg, "basemap_key_env", DEFAULT_BASEMAP_KEY_ENV))
    key = (os.environ.get(env_name) or "").strip()
    providers = _site_value(cfg, "basemaps", {}) or {}
    name = KEYED_PROVIDER if key else FALLBACK_PROVIDER
    provider = dict(providers.get(name) or {})
    if not key:
        print(
            f"::warning::{env_name} is not set; the dashboard falls back to "
            "OpenStreetMap tiles, which are best effort and have no SLA"
        )
    return {
        "provider": name,
        "light_url": _with_key(provider.get("light_url"), key),
        "dark_url": _with_key(provider.get("dark_url"), key),
        "subdomains": str(provider.get("subdomains") or ""),
        "max_zoom": int(provider.get("max_zoom") or DEFAULT_MAX_ZOOM),
        "dark_filter": bool(provider.get("dark_filter", False)),
        "attribution": str(provider.get("attribution") or ""),
    }


def _with_key(url: object, key: str) -> str:
    return "" if not url else str(url).replace("{key}", key)
