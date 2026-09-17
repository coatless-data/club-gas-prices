"""Shared record types for the country sources, and the capture bundle's responses.

`fetch()` makes requests and returns `RawResponse` objects; `parse()` is pure and
turns them into a `FetchResult`. The split exists so that a rebuild can re-run
`parse()` over the responses stored in an old capture bundle, with no network.

`club_gas.http` is imported only under `if TYPE_CHECKING`. That module imports
`RawResponse` from here at module scope, so the runtime dependency must stay
one-directional: http.py -> sources/base.py.

`Warning` and `Error` below are data records (the project's fixed interface), not
exceptions. This module never uses the builtin `Warning`.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Protocol

import polars as pl

if TYPE_CHECKING:
    from collections.abc import Iterable

    from club_gas.config import Config
    from club_gas.http import Client


@dataclass(frozen=True)
class RawResponse:
    """One HTTP response, exactly as received."""

    key: str
    url: str
    status: int | None
    headers: dict[str, str]
    received_at_utc: datetime
    elapsed_ms: int
    body: bytes
    error: str | None = None


@dataclass(frozen=True)
class RawPrice:
    grade_raw: str
    price_raw: str


@dataclass(frozen=True)
class RawStation:
    source_station_id: str
    alt_id: str | None
    id_origin: Literal["ecom", "extra", "seen", "lookup", "cache", "occ"]
    ecom_state: Literal["gas", "no_gas", "absent", "unavailable"] | None
    name: str
    name_local: str | None = None
    address: str | None = None
    city: str | None = None
    region: str | None = None
    postcode: str | None = None
    lat: float | None = None
    lon: float | None = None
    timezone: str | None = None
    opening_date: date | None = None
    has_hours: bool | None = None
    prices: tuple[RawPrice, ...] = ()


@dataclass(frozen=True)
class Warning:
    code: str
    detail: str | None = None


@dataclass(frozen=True)
class Error:
    code: str
    host: str | None = None
    http_status: int | None = None
    detail: str | None = None


@dataclass
class FetchResult:
    country: str
    source: str
    captured_at_utc: datetime
    # The chain whose feed this is. Part of the station key, because two chains
    # number their sites independently and collide on the bare number.
    brand: str = "COSTCO"
    stations: list[RawStation] = field(default_factory=list)
    responses: list[RawResponse] = field(default_factory=list)
    requests: int = 0
    warnings: list[Warning] = field(default_factory=list)
    errors: list[Error] = field(default_factory=list)


@dataclass
class CaptureContext:
    capture_id: str
    capture_date: date
    fetch_config: Config
    interp_config: Config
    previous_stations: pl.DataFrame = field(default_factory=pl.DataFrame)
    previous_fx: pl.DataFrame = field(default_factory=pl.DataFrame)
    # False when `current/stations.csv` or `current/fx.csv` could not be read, so
    # `previous_stations` and `previous_fx` above are empty stand-ins rather than
    # the real previous state. US and CA are the two countries that depend on
    # them (the `seen` ids, and CA's cached fallback metadata), and §6.5 makes
    # both degraded when this is False.
    previous_state_complete: bool = True
    previous_status: dict | None = None
    shared: dict[str, RawResponse] = field(default_factory=dict)
    force_fallback: set[str] = field(default_factory=set)


class Source(Protocol):
    country: str

    def fetch(self, client: Client, ctx: CaptureContext) -> list[RawResponse]: ...

    def parse(self, responses: list[RawResponse], ctx: CaptureContext) -> FetchResult: ...


@dataclass
class _LazySource:
    """A `Source` that imports its implementation module on first use.

    `sources/us.py`, `sources/ca.py` and `sources/occ.py` all import this module,
    so `base.py` cannot import them at module scope without a circular import.
    Delegating through this wrapper keeps `SOURCES` a plain, fully populated dict
    that can be listed and iterated before those modules are even written.
    """

    country: str
    module: str
    factory: str
    pass_country: bool = False
    _impl: Source | None = field(default=None, init=False, repr=False, compare=False)

    def _resolve(self) -> Source:
        if self._impl is None:
            import importlib

            module = importlib.import_module(self.module)
            factory = getattr(module, self.factory)
            self._impl = factory(self.country) if self.pass_country else factory()
        return self._impl

    def fetch(self, client: Client, ctx: CaptureContext) -> list[RawResponse]:
        return self._resolve().fetch(client, ctx)

    def parse(self, responses: list[RawResponse], ctx: CaptureContext) -> FetchResult:
        return self._resolve().parse(responses, ctx)


SOURCES: dict[str, Source] = {
    "US": _LazySource("US", "club_gas.sources.us", "UsSource"),
    "CA": _LazySource("CA", "club_gas.sources.ca", "CaSource"),
    "MX": _LazySource("MX", "club_gas.sources.occ", "OccSource", pass_country=True),
    "GB": _LazySource("GB", "club_gas.sources.occ", "OccSource", pass_country=True),
    "AU": _LazySource("AU", "club_gas.sources.occ", "OccSource", pass_country=True),
    "JP": _LazySource("JP", "club_gas.sources.occ", "OccSource", pass_country=True),
    "TW": _LazySource("TW", "club_gas.sources.occ", "OccSource", pass_country=True),
}


RESPONSES_DIRNAME = "responses"
_BODY_SUFFIX = ".body"
_META_SUFFIX = ".meta.json"


def _to_utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _iso_utc(value: datetime) -> str:
    return _to_utc(value).isoformat().replace("+00:00", "Z")


def _check_key(key: str) -> None:
    parts = key.split("/")
    unsafe = (
        not key
        or key.startswith("/")
        or "\\" in key
        or any(part in ("", ".", "..") for part in parts)
    )
    if unsafe:
        raise ValueError(f"unsafe response key: {key!r}")


def response_paths(root: Path, key: str) -> tuple[Path, Path]:
    """Return the (.body, .meta.json) paths for `key` under the bundle `root`."""
    _check_key(key)
    base = Path(root) / RESPONSES_DIRNAME / key
    return (
        base.parent / f"{base.name}{_BODY_SUFFIX}",
        base.parent / f"{base.name}{_META_SUFFIX}",
    )


def write_responses(responses: Iterable[RawResponse], root: Path) -> list[Path]:
    """Write each response to `<root>/responses/<key>.body` and `.meta.json`."""
    written: list[Path] = []
    for response in responses:
        body_path, meta_path = response_paths(root, response.key)
        body_path.parent.mkdir(parents=True, exist_ok=True)
        body_path.write_bytes(response.body)
        meta = {
            "key": response.key,
            "url": response.url,
            "status": response.status,
            "headers": dict(response.headers),
            "received_at_utc": _iso_utc(response.received_at_utc),
            "elapsed_ms": int(response.elapsed_ms),
            "error": response.error,
        }
        meta_path.write_text(
            json.dumps(meta, indent=2, sort_keys=True, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        written.extend([body_path, meta_path])
    return written


def read_responses(root: Path) -> list[RawResponse]:
    """Read back every response written by `write_responses`, sorted by key."""
    base = Path(root) / RESPONSES_DIRNAME
    if not base.is_dir():
        return []
    out: list[RawResponse] = []
    for meta_path in sorted(base.rglob(f"*{_META_SUFFIX}")):
        stem = meta_path.name[: -len(_META_SUFFIX)]
        key = meta_path.relative_to(base).as_posix()[: -len(_META_SUFFIX)]
        body_path = meta_path.parent / f"{stem}{_BODY_SUFFIX}"
        if not body_path.is_file():
            raise FileNotFoundError(f"missing response body for {key}: {body_path}")
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        received = str(meta["received_at_utc"]).replace("Z", "+00:00")
        out.append(
            RawResponse(
                key=str(meta.get("key") or key),
                url=str(meta["url"]),
                status=meta["status"],
                headers=dict(meta.get("headers") or {}),
                received_at_utc=_to_utc(datetime.fromisoformat(received)),
                elapsed_ms=int(meta["elapsed_ms"]),
                body=body_path.read_bytes(),
                error=meta.get("error"),
            )
        )
    return out
