"""Shared record types for the country sources, and the capture bundle's responses.

`fetch()` makes requests and returns `RawResponse` objects; `parse()` is pure and
turns them into a `FetchResult`. The split exists so that a rebuild can re-run
`parse()` over the responses stored in an old capture bundle, with no network.

`costco_gas.http` is imported only under `if TYPE_CHECKING`. That module imports
`RawResponse` from here at module scope, so the runtime dependency must stay
one-directional: http.py -> sources/base.py.

`Warning` and `Error` below are data records (the project's fixed interface), not
exceptions. This module never uses the builtin `Warning`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime
from typing import TYPE_CHECKING, Literal, Protocol

import polars as pl

if TYPE_CHECKING:
    from costco_gas.config import Config
    from costco_gas.http import Client


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
    previous_status: dict | None = None
    shared: dict[str, RawResponse] = field(default_factory=dict)
    force_fallback: set[str] = field(default_factory=set)


class Source(Protocol):
    country: str

    def fetch(self, client: Client, ctx: CaptureContext) -> list[RawResponse]: ...

    def parse(
        self, responses: list[RawResponse], ctx: CaptureContext
    ) -> FetchResult: ...


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
    "US": _LazySource("US", "costco_gas.sources.us", "UsSource"),
    "CA": _LazySource("CA", "costco_gas.sources.ca", "CaSource"),
    "MX": _LazySource("MX", "costco_gas.sources.occ", "OccSource", pass_country=True),
    "GB": _LazySource("GB", "costco_gas.sources.occ", "OccSource", pass_country=True),
    "AU": _LazySource("AU", "costco_gas.sources.occ", "OccSource", pass_country=True),
    "JP": _LazySource("JP", "costco_gas.sources.occ", "OccSource", pass_country=True),
    "TW": _LazySource("TW", "costco_gas.sources.occ", "OccSource", pass_country=True),
}
