"""Load and validate every file in config/ into typed objects.

The config splits into two halves. Fetch config (URLs, query parameters, HTTP
policy, us_extra_ids.csv) says what to request. Interpretation config (grades,
units, bounds, region-to-timezone tables, station floors, stale_after_days,
station_links.csv) says how to read a response that was stored earlier. The
rebuild command replays months-old responses with today's interpretation
config, which is why Config exposes fetch_view() and interp_view().
"""

from __future__ import annotations

import re
import tomllib
from dataclasses import dataclass, field, replace
from pathlib import Path

import polars as pl

GRADES = ("regular", "premium", "diesel", "other")
PRICE_UNITS = (
    "USD/gal",
    "USD/L",
    "CAD/L",
    "MXN/L",
    "GBp/L",
    "AUD/L",
    "JPY/L",
    "TWD/L",
)
SPEC_SOURCES = ("source", "reported", "")
DEFAULT_TIMEZONE_KEY = "*"

# Area/Location, or Area/Region/Location for zones such as
# America/Indiana/Indianapolis.
_IANA_RE = re.compile(r"[A-Za-z]+(?:/[A-Za-z0-9_+-]+){1,2}")

US_EXTRA_IDS_SCHEMA: dict[str, pl.DataType] = {
    "source_station_id": pl.String,
    "name": pl.String,
    "city": pl.String,
    "region": pl.String,
    "postcode": pl.String,
    "lat": pl.Float64,
    "lon": pl.Float64,
    "timezone": pl.String,
    "note": pl.String,
}

STATION_LINKS_SCHEMA: dict[str, pl.DataType] = {
    "old_station_key": pl.String,
    "new_station_key": pl.String,
    "effective_date": pl.String,
    "note": pl.String,
}


class ConfigError(Exception):
    """A config file is missing, malformed or internally inconsistent."""


@dataclass(frozen=True)
class TimeoutProfile:
    """connect and read are per-operation; total covers the whole request.

    read also covers the wait for response headers, so a server that takes
    30 s to start answering needs a profile whose read exceeds 30 s.
    """

    connect: float
    read: float
    total: float


def _default_profiles() -> dict[str, TimeoutProfile]:
    return {
        "default": TimeoutProfile(connect=10.0, read=20.0, total=60.0),
        "bulk": TimeoutProfile(connect=10.0, read=150.0, total=150.0),
    }


def _default_budgets() -> dict[str, float]:
    return {"fx": 90.0, "ecom-api": 90.0, "country": 480.0, "capture": 720.0}


CHROME_UA = (
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/128.0.0.0 Safari/537.36"
)
PROJECT_UA = "costco-gas-prices/0.1 (+https://github.com/coatless-dashboard/costco-gas-prices)"
PROJECT_URL = "https://github.com/coatless-dashboard/costco-gas-prices"


@dataclass(frozen=True)
class HttpConfig:
    """Everything the shared HTTP client needs. Defaults mirror http.toml."""

    costco_user_agent: str = CHROME_UA
    project_user_agent: str = PROJECT_UA
    x_project: str = PROJECT_URL
    costco_host_suffixes: tuple[str, ...] = (
        "costco.com",
        "costco.ca",
        "costco.com.mx",
        "costco.co.uk",
        "costco.com.au",
        "costco.co.jp",
        "costco.com.tw",
    )
    profiles: dict[str, TimeoutProfile] = field(default_factory=_default_profiles)
    max_attempts: int = 3
    backoff_seconds: tuple[float, ...] = (1.0, 2.0)
    backoff_jitter: float = 0.5
    min_interval_seconds: float = 1.0
    block_signals_before_abandon: int = 2
    budgets: dict[str, float] = field(default_factory=_default_budgets)


@dataclass(frozen=True)
class Bounds:
    """Sanity bounds for one price unit, with optional per-grade overrides.

    Every consumer reads bounds the same way:
    cfg.countries[cc].bounds[price_unit].for_grade(grade_raw).
    """

    min: float
    max: float
    grade_overrides: dict[str, tuple[float, float]] = field(default_factory=dict)

    def for_grade(self, grade_raw: str) -> tuple[float, float]:
        return self.grade_overrides.get(grade_raw, (self.min, self.max))


@dataclass(frozen=True)
class CountryConfig:
    """One country's fetch and interpretation settings.

    params holds query parameters only, flat and all strings, so a fetcher can
    urlencode it whole. Every non-query setting is its own field.
    """

    code: str
    url: str
    params: dict[str, str]
    price_unit: str
    unit_overrides: dict[str, str]
    bounds: dict[str, Bounds]
    floor: int
    stale_after_days: int
    timezones: dict[str, str]
    batch_size: int | None = None
    seen_within_days: int | None = None
    fallback_url: str | None = None
    fallback_params: dict[str, str] = field(default_factory=dict)
    ecom_url: str | None = None
    ecom_params: dict[str, str] = field(default_factory=dict)
    ecom_client_identifier: str | None = None

    def unit_for_region(self, region: str | None) -> str:
        if region is None:
            return self.price_unit
        return self.unit_overrides.get(region, self.price_unit)

    def timezone_for_region(self, region: str | None) -> str | None:
        """The region's zone, else the country default under the "*" key.

        Callers use this instead of indexing .timezones, so that the default
        key lives in exactly one place.
        """
        if region is not None and region in self.timezones:
            return self.timezones[region]
        return self.timezones.get(DEFAULT_TIMEZONE_KEY)


@dataclass(frozen=True)
class GradeEntry:
    grade: str
    priority: int
    label: str
    spec: str
    spec_source: str
    spec_source_url: str


@dataclass(frozen=True)
class GradeTable:
    entries: dict[tuple[str, str], GradeEntry]

    def map(self, country: str, grade_raw: str) -> GradeEntry | None:
        """The mapping is by exact label. An unmapped label returns None."""
        return self.entries.get((country, grade_raw))

    def rows(self) -> list[dict[str, object]]:
        """The whole table, for the dashboard's About page."""
        return [
            {
                "country": country,
                "grade_raw": grade_raw,
                "grade": entry.grade,
                "priority": entry.priority,
                "label": entry.label,
                "spec": entry.spec,
                "spec_source": entry.spec_source,
                "spec_source_url": entry.spec_source_url,
            }
            for (country, grade_raw), entry in self.entries.items()
        ]


@dataclass(frozen=True)
class SiteConfig:
    notice: str
    release_base_url: str
    basemap_key_env: str
    basemaps: dict[str, dict[str, object]]


@dataclass(frozen=True, eq=False)
class Config:
    root: Path
    http: HttpConfig
    countries: dict[str, CountryConfig]
    grades: GradeTable
    us_extra_ids: pl.DataFrame
    station_links: pl.DataFrame
    site: SiteConfig

    def fetch_view(self) -> Config:
        """URLs, parameters, HTTP policy and us_extra_ids; nothing else.

        A rebuild loads this view from the capture bundle, so the bundle's
        stale interpretation tables are blanked rather than used.
        """
        return replace(
            self,
            countries={
                code: replace(
                    country,
                    price_unit="",
                    unit_overrides={},
                    bounds={},
                    floor=0,
                    stale_after_days=0,
                    timezones={},
                )
                for code, country in self.countries.items()
            },
            grades=GradeTable(entries={}),
            station_links=pl.DataFrame(schema=STATION_LINKS_SCHEMA),
        )

    def interp_view(self) -> Config:
        """Grades, units, bounds, region tables, floors, staleness and links.

        A rebuild loads this view from the live checkout, and hashes those
        files through .root. .http is carried along for convenience; callers
        read HTTP policy from fetch_view().
        """
        return replace(
            self,
            countries={
                code: replace(
                    country,
                    url="",
                    params={},
                    batch_size=None,
                    seen_within_days=None,
                    fallback_url=None,
                    fallback_params={},
                    ecom_url=None,
                    ecom_params={},
                    ecom_client_identifier=None,
                )
                for code, country in self.countries.items()
            },
            us_extra_ids=pl.DataFrame(schema=US_EXTRA_IDS_SCHEMA),
        )


def load_config(root: Path) -> Config:
    """Read and validate root/config/*.

    root is kept on the result as Config.root, because rebuild hashes the live
    interpretation config by re-reading those files.
    """
    root = Path(root)
    cfg_dir = root / "config"
    if not cfg_dir.is_dir():
        raise ConfigError(f"no config directory at {cfg_dir}")
    cfg = Config(
        root=root,
        http=_load_http(cfg_dir / "http.toml"),
        countries=_load_countries(cfg_dir / "countries.toml"),
        grades=_load_grades(cfg_dir / "grades.csv"),
        us_extra_ids=_read_csv(cfg_dir / "us_extra_ids.csv", US_EXTRA_IDS_SCHEMA),
        station_links=_read_csv(cfg_dir / "station_links.csv", STATION_LINKS_SCHEMA),
        site=_load_site(cfg_dir / "site.toml"),
    )
    _validate(cfg)
    return cfg


def _read_toml(path: Path) -> dict:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    with path.open("rb") as handle:
        try:
            return tomllib.load(handle)
        except tomllib.TOMLDecodeError as exc:
            raise ConfigError(f"{path}: {exc}") from exc


def _read_csv(path: Path, schema: dict[str, pl.DataType]) -> pl.DataFrame:
    if not path.is_file():
        raise ConfigError(f"missing config file: {path}")
    frame = pl.read_csv(path, schema_overrides=schema)
    missing = [name for name in schema if name not in frame.columns]
    if missing:
        raise ConfigError(f"{path}: missing columns {missing}")
    return frame.select(list(schema))


def _load_http(path: Path) -> HttpConfig:
    raw = _read_toml(path)
    try:
        identity = raw["identity"]
        profiles = {
            name: TimeoutProfile(
                connect=float(spec["connect"]),
                read=float(spec["read"]),
                total=float(spec["total"]),
            )
            for name, spec in raw["profiles"].items()
        }
        if "default" not in profiles:
            raise ConfigError(f"{path}: no [profiles.default]")
        retry = raw.get("retry", {})
        return HttpConfig(
            costco_user_agent=str(identity["costco_user_agent"]),
            project_user_agent=str(identity["project_user_agent"]),
            x_project=str(identity["x_project"]),
            costco_host_suffixes=tuple(str(host) for host in identity["costco_host_suffixes"]),
            profiles=profiles,
            max_attempts=int(retry.get("max_attempts", 3)),
            backoff_seconds=tuple(
                float(value) for value in retry.get("backoff_seconds", (1.0, 2.0))
            ),
            backoff_jitter=float(retry.get("jitter", 0.5)),
            min_interval_seconds=float(raw.get("pacing", {}).get("min_interval_seconds", 1.0)),
            block_signals_before_abandon=int(
                raw.get("blocks", {}).get("signals_before_abandon", 2)
            ),
            budgets={str(k): float(v) for k, v in raw.get("budgets", {}).items()},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _load_bounds(raw: dict, where: str) -> dict[str, Bounds]:
    out: dict[str, Bounds] = {}
    for unit, spec in raw.items():
        overrides = {
            grade: (float(values["min"]), float(values["max"]))
            for grade, values in (spec.get("grades") or {}).items()
        }
        try:
            out[unit] = Bounds(
                min=float(spec["min"]),
                max=float(spec["max"]),
                grade_overrides=overrides,
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{where}: bounds for {unit!r}: {exc}") from exc
    return out


def _query_params(raw: object, where: str) -> dict[str, str]:
    """A query table: flat, strings only, ready for urlencode()."""
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(f"{where}: must be a table of query parameters")
    out: dict[str, str] = {}
    for key, value in raw.items():
        if not isinstance(value, str):
            raise ConfigError(f"{where}: query parameter {key!r} must be a string, got {value!r}")
        out[str(key)] = value
    return out


def _optional_int(value: object) -> int | None:
    return None if value is None else int(value)


def _optional_str(value: object) -> str | None:
    return None if value is None else str(value)


def _load_countries(path: Path) -> dict[str, CountryConfig]:
    raw = _read_toml(path)
    table = raw.get("countries")
    if not table:
        raise ConfigError(f"{path}: no [countries.*] tables")
    out: dict[str, CountryConfig] = {}
    for code, spec in table.items():
        where = f"{path}: countries.{code}"
        try:
            out[code] = CountryConfig(
                code=code,
                url=str(spec["url"]),
                params=_query_params(spec.get("params"), f"{where}.params"),
                price_unit=str(spec["price_unit"]),
                unit_overrides={str(k): str(v) for k, v in spec.get("unit_overrides", {}).items()},
                bounds=_load_bounds(spec.get("bounds", {}), where),
                floor=int(spec["floor"]),
                stale_after_days=int(spec["stale_after_days"]),
                timezones={str(k): str(v) for k, v in spec.get("timezones", {}).items()},
                batch_size=_optional_int(spec.get("batch_size")),
                seen_within_days=_optional_int(spec.get("seen_within_days")),
                fallback_url=_optional_str(spec.get("fallback_url")),
                fallback_params=_query_params(
                    spec.get("fallback_params"), f"{where}.fallback_params"
                ),
                ecom_url=_optional_str(spec.get("ecom_url")),
                ecom_params=_query_params(spec.get("ecom_params"), f"{where}.ecom_params"),
                ecom_client_identifier=_optional_str(spec.get("ecom_client_identifier")),
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise ConfigError(f"{where}: {exc}") from exc
    return out


def _text(value: object) -> str:
    """Polars reads an empty CSV field as null; the config treats it as ''."""
    return "" if value is None else str(value)


def _load_grades(path: Path) -> GradeTable:
    columns: dict[str, pl.DataType] = {
        "country": pl.String,
        "grade_raw": pl.String,
        "grade": pl.String,
        "priority": pl.Int64,
        "label": pl.String,
        "spec": pl.String,
        "spec_source": pl.String,
        "spec_source_url": pl.String,
    }
    frame = _read_csv(path, columns)
    entries: dict[tuple[str, str], GradeEntry] = {}
    for row in frame.iter_rows(named=True):
        key = (_text(row["country"]), _text(row["grade_raw"]))
        if key in entries:
            raise ConfigError(f"{path}: duplicate row for {key[0]} {key[1]!r}")
        if row["priority"] is None:
            raise ConfigError(f"{path}: {key[0]} {key[1]!r} has no priority")
        entries[key] = GradeEntry(
            grade=_text(row["grade"]),
            priority=int(row["priority"]),
            label=_text(row["label"]),
            spec=_text(row["spec"]),
            spec_source=_text(row["spec_source"]),
            spec_source_url=_text(row["spec_source_url"]),
        )
    return GradeTable(entries=entries)


def _load_site(path: Path) -> SiteConfig:
    raw = _read_toml(path)
    site = raw.get("site")
    if not site:
        raise ConfigError(f"{path}: no [site] table")
    try:
        return SiteConfig(
            notice=str(site["notice"]),
            release_base_url=str(site["release_base_url"]),
            basemap_key_env=str(site["basemap_key_env"]),
            basemaps={str(k): dict(v) for k, v in site.get("basemaps", {}).items()},
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise ConfigError(f"{path}: {exc}") from exc


def _validate(cfg: Config) -> None:
    for code, country in cfg.countries.items():
        if country.price_unit not in PRICE_UNITS:
            raise ConfigError(
                f"{code}: price_unit {country.price_unit!r} is not one of {', '.join(PRICE_UNITS)}"
            )
        units = {country.price_unit, *country.unit_overrides.values()}
        for unit in sorted(units):
            if unit not in PRICE_UNITS:
                raise ConfigError(
                    f"{code}: unit override {unit!r} is not one of {', '.join(PRICE_UNITS)}"
                )
            if unit not in country.bounds:
                raise ConfigError(f"{code}: no bounds for unit {unit!r}")
        for unit, bounds in country.bounds.items():
            if bounds.min >= bounds.max:
                raise ConfigError(
                    f"{code}: bounds for {unit!r} have min {bounds.min} >= max {bounds.max}"
                )
            for grade_raw, (low, high) in bounds.grade_overrides.items():
                if low >= high:
                    raise ConfigError(
                        f"{code}: bounds for {unit!r} grade {grade_raw!r} have "
                        f"min {low} >= max {high}"
                    )
        if country.floor < 1:
            raise ConfigError(f"{code}: floor must be at least 1")
        if country.stale_after_days < 1:
            raise ConfigError(f"{code}: stale_after_days must be at least 1")
        if not country.timezones:
            raise ConfigError(f"{code}: timezones table is empty")
        for region, zone in country.timezones.items():
            if not _IANA_RE.fullmatch(zone):
                raise ConfigError(
                    f"{code}: {zone!r} for region {region!r} is not an IANA timezone name"
                )
        for region in sorted(country.unit_overrides):
            if region not in country.timezones:
                raise ConfigError(f"{code}: no timezone for region {region!r}")

    for (country_code, grade_raw), entry in cfg.grades.entries.items():
        where = f"grades.csv: {country_code} {grade_raw!r}"
        if country_code not in cfg.countries:
            raise ConfigError(f"{where}: unknown country {country_code!r}")
        if entry.grade not in GRADES:
            raise ConfigError(
                f"{where}: maps to unknown grade {entry.grade!r}; expected one of "
                f"{', '.join(GRADES)}"
            )
        if entry.spec_source not in SPEC_SOURCES:
            raise ConfigError(
                f"{where}: spec_source {entry.spec_source!r} must be 'source', 'reported' or blank"
            )
        if entry.spec_source == "reported" and not entry.spec_source_url:
            raise ConfigError(f"{where}: a reported spec needs a spec_source_url")

    us = cfg.countries.get("US")
    if us is not None:
        for row in cfg.us_extra_ids.iter_rows(named=True):
            station_id = _text(row["source_station_id"])
            if not station_id:
                raise ConfigError("us_extra_ids.csv: a row has no source_station_id")
            region = _text(row["region"])
            if region and region not in us.timezones:
                raise ConfigError(
                    f"US: no timezone for region {region!r}, used by "
                    f"us_extra_ids.csv id {station_id}"
                )
            zone = _text(row["timezone"])
            if not zone:
                raise ConfigError(
                    f"us_extra_ids.csv id {station_id}: no timezone; ecom-api does "
                    "not list these ids, so normalize would drop the station"
                )
            if not _IANA_RE.fullmatch(zone):
                raise ConfigError(
                    f"us_extra_ids.csv id {station_id}: {zone!r} is not an IANA timezone name"
                )

    seen: set[str] = set()
    for row in cfg.station_links.iter_rows(named=True):
        old_key = _text(row["old_station_key"])
        if old_key in seen:
            raise ConfigError(f"station_links.csv: duplicate old_station_key {old_key!r}")
        seen.add(old_key)
