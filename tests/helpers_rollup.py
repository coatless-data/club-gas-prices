"""Builders shared by the roll-up tests. Nothing here touches the network."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from club_gas.schema import ROW_SCHEMA, STATION_SCHEMA, write_rows_csv_gz

GALLON_LITRES = 3.785411784
NOTICE = (
    "Unofficial. Not affiliated with, endorsed by, or connected to Costco Wholesale "
    "Corporation. Prices are collected from Costco's public websites and may differ "
    "from the price at the pump."
)


def capture_dt(capture_id: str) -> datetime:
    return datetime.strptime(capture_id, "%Y-%m-%dT%H%MZ").replace(tzinfo=UTC)


def price_row(
    *,
    capture_id: str,
    station_key: str,
    grade_raw: str,
    grade: str,
    price: float,
    country: str = "US",
    price_unit: str = "USD/gal",
    currency: str = "USD",
    fx_usd_per_unit: float | None = 1.0,
    captured_at_utc: datetime | None = None,
    name: str = "Costco",
    tz: str = "America/Chicago",
    source: str = "costco-us-gasprices",
    region: str | None = "TX",
    lat: float | None = 32.7,
    lon: float | None = -97.1,
) -> dict:
    """One §6.1 price row as a plain dict, with the real derived-column formulas."""
    started = capture_dt(capture_id)
    at = captured_at_utc or started
    if price_unit == "USD/gal":
        local = price / GALLON_LITRES
    elif price_unit == "GBp/L":
        local = price / 100
    else:
        local = price
    usd_l = None if fx_usd_per_unit is None else local * fx_usd_per_unit
    if usd_l is None:
        usd_gal = None
    elif price_unit == "USD/gal":
        usd_gal = price
    else:
        usd_gal = usd_l * GALLON_LITRES
    return {
        "capture_id": capture_id,
        "capture_date": started.date(),
        "captured_at_utc": at,
        "local_date": at.date(),
        "country": country,
        "station_key": station_key,
        "source_station_id": station_key.split("-", 1)[1],
        "source": source,
        "name": name,
        "name_local": None,
        "address": None,
        "city": None,
        "region": region,
        "postcode": None,
        "lat": lat,
        "lon": lon,
        "timezone": tz,
        "grade_raw": grade_raw,
        "grade": grade,
        "price_raw": str(price),
        "price": price,
        "price_unit": price_unit,
        "currency": currency,
        "price_local_per_litre": round(local, 4),
        "fx_usd_per_unit": fx_usd_per_unit,
        "fx_rate_date": started.date(),
        "fx_source": "identity" if currency == "USD" else "frankfurter-v2",
        "fx_fetched_at_utc": at,
        "price_usd_per_litre": None if usd_l is None else round(usd_l, 4),
        "price_usd_per_gallon": None if usd_gal is None else round(usd_gal, 4),
    }


def rows_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=ROW_SCHEMA)


def stations_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(rows, schema=STATION_SCHEMA)


def write_csv_gz(df: pl.DataFrame, path: Path) -> None:
    # Delegates to the real writer rather than a bare `df.write_csv`: rollup's
    # `_read_daily_files` reads these fixtures back with `schema.read_rows_csv_gz`,
    # which parses strict `CSV_DATE_FORMAT`/`CSV_DATETIME_FORMAT` strings -- exactly
    # what `write_rows_csv_gz` produces and Polars' own default text form is not.
    write_rows_csv_gz(df, path)


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


@dataclass
class StubSite:
    notice: str = NOTICE


@dataclass(frozen=True)
class StubCountry:
    """Only what rollup asks a country for."""

    closed_after_days: int = 45


@dataclass
class StubConfig:
    """rollup reads `.root`, `.station_links`, `.site.notice` and, per country,
    `.closed_after_days`."""

    root: Path
    station_links: pl.DataFrame
    site: StubSite = field(default_factory=StubSite)
    countries: dict = field(
        default_factory=lambda: {
            code: StubCountry() for code in ("US", "CA", "MX", "GB", "AU", "JP", "TW")
        }
    )


def stub_config(tmp_path: Path, links: pl.DataFrame | None = None) -> StubConfig:
    empty = pl.DataFrame(
        [],
        schema={
            "old_station_key": pl.String,
            "new_station_key": pl.String,
            "effective_date": pl.String,
            "note": pl.String,
        },
    )
    return StubConfig(root=tmp_path, station_links=empty if links is None else links)


class RecordingIssues:
    def __init__(self) -> None:
        self.opened: list[tuple[str, str]] = []
        self.closed: list[tuple[str, str]] = []

    def ensure_open(self, title: str, body: str, labels: list[str] | None = None) -> None:
        self.opened.append((title, body))

    def close(self, title: str, comment: str) -> None:
        self.closed.append((title, comment))


class RecordingStore:
    """Delegates to a real store and logs writes, so tests can assert their order."""

    def __init__(self, inner) -> None:
        self.inner = inner
        self.log: list[str] = []

    def __getattr__(self, name: str):
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            if name in ("replace_atomic", "upload_new"):
                self.log.append(f"{name}:{args[0]}:{args[2]}")
            elif name == "update_release":
                self.log.append(f"update_release:{args[0]}:prerelease={kwargs.get('prerelease')}")
            elif name == "ensure_release":
                self.log.append(f"ensure_release:{args[0]}")
            return attr(*args, **kwargs)

        return call


def seed_month(
    store,
    month: str,
    days: dict[str, pl.DataFrame],
    *,
    captures: dict[str, dict],
    work: Path,
    prerelease: bool = True,
    orphan_bundles: list[str] | None = None,
) -> dict:
    """Create `data-<month>` with one daily file per day and a matching manifest.

    `captures` maps a capture_id to {"status": dict, "rows_by_capture_date": {day: n},
    "fx": [fx.json rows]}. The bundles are placeholder bytes: close-periods only ever
    compares bundle *names* against the manifest, it never opens the tar.
    """
    tag = f"data-{month}"
    work.mkdir(parents=True, exist_ok=True)
    store.ensure_release(tag, f"Data {month}", "", prerelease, "false")
    daily_meta: dict[str, dict] = {}
    for day, frame in days.items():
        path = work / f"club-gas-{day}.csv.gz"
        write_csv_gz(frame, path)
        store.upload_new(tag, path, path.name)
        daily_meta[day] = {"sha256": sha256_file(path), "rows": frame.height}
    manifest = {"schema_version": 1, "month": month, "captures": {}}
    for cid, info in captures.items():
        bundle = work / f"capture-{cid}.tar.gz"
        bundle.write_bytes(b"bundle-placeholder")
        store.upload_new(tag, bundle, bundle.name)
        by_date = info["rows_by_capture_date"]
        manifest["captures"][cid] = {
            "capture_id": cid,
            "status": info["status"],
            "rows_by_capture_date": by_date,
            "fx": info.get("fx", []),
            "bundle": {"sha256": sha256_file(bundle), "rows": sum(by_date.values())},
            "daily_files": {d: daily_meta[d] for d in by_date if d in daily_meta},
            "rebuild_git_sha": None,
            "config_sha256": {},
        }
    for name in orphan_bundles or []:
        extra = work / name
        extra.write_bytes(b"orphan-bundle")
        store.upload_new(tag, extra, name)
    path = work / f"manifest-{month}.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), "utf-8")
    store.upload_new(tag, path, path.name)
    return manifest
