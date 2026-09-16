"""Tests for the United States source (spec 4.2). No network: a fake Client serves fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from costco_gas.http import BudgetExceeded
from costco_gas.sources import us
from costco_gas.sources.base import SOURCES, CaptureContext, RawResponse

FIXTURES = Path(__file__).parent / "fixtures"
CAPTURE_ID = "2026-09-15T1910Z"
CAPTURE_DATE = date(2026, 9, 15)
CAPTURE_START = datetime(2026, 9, 15, 19, 10, tzinfo=UTC)

US_TIMEZONES = {
    "AZ": "America/Phoenix",
    "CA": "America/Los_Angeles",
    "DC": "America/New_York",
    "FL": "America/New_York",
    "HI": "Pacific/Honolulu",
    "MO": "America/Chicago",
    "PR": "America/Puerto_Rico",
    "TX": "America/Chicago",
    "WA": "America/Los_Angeles",
}

EXTRAS = pl.DataFrame(
    {
        "source_station_id": ["1680", "1765", "1772"],
        "name": ["Camarillo", "Chandler Business Center", "Mission Viejo"],
        "city": ["CAMARILLO", "CHANDLER", "MISSION VIEJO"],
        "region": ["CA", "AZ", "CA"],
        "postcode": ["93010-9122", "85226", "92691"],
        "lat": [None, None, None],
        "lon": [None, None, None],
        "timezone": ["America/Los_Angeles", "America/Phoenix", None],
        "note": ["prices live, absent from ecom-api", "business center", "reported"],
    },
    schema_overrides={"lat": pl.Float64, "lon": pl.Float64, "timezone": pl.Utf8},
)

STATION_COLUMNS = [
    "station_key",
    "country",
    "source_station_id",
    "alt_id",
    "name",
    "name_local",
    "address",
    "city",
    "region",
    "postcode",
    "lat",
    "lon",
    "timezone",
    "grades_seen",
    "first_seen_utc",
    "last_seen_utc",
    "status",
    "superseded_by",
]


# --------------------------------------------------------------------------- helpers


@dataclass
class StubCountry:
    """The `config.CountryConfig` fields this module reads, with the real US values."""

    code: str = "US"
    url: str = us.DEFAULT_PRICE_URL
    params: dict = field(default_factory=dict)
    batch_size: int | None = 10
    seen_within_days: int | None = 30
    fallback_url: str | None = "https://www.costco.ca/AjaxWarehouseBrowseLookupView"
    fallback_params: dict = field(
        default_factory=lambda: {
            "hasGas": "true",
            "populateWarehouseDetails": "true",
            "countryCode": "US",
        }
    )
    ecom_url: str | None = "https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json"
    ecom_params: dict = field(
        default_factory=lambda: {"latitude": "0", "longitude": "0", "limit": "5000"}
    )
    ecom_client_identifier: str | None = "7c71124c-7bf1-44db-bc9d-498584cd66e5"
    timezones: dict = field(default_factory=lambda: dict(US_TIMEZONES))

    def timezone_for_region(self, region: str | None) -> str | None:
        """Same fallback as the real one: the region key, then the `"*"` default."""
        if region is not None and region in self.timezones:
            return self.timezones[region]
        return self.timezones.get("*")


@dataclass
class StubConfig:
    countries: dict
    us_extra_ids: pl.DataFrame


def stations_frame(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            {c: [] for c in STATION_COLUMNS},
            schema={c: pl.Utf8 for c in STATION_COLUMNS},
        )
    filled = [{c: row.get(c) for c in STATION_COLUMNS} for row in rows]
    return pl.DataFrame(filled, schema={c: pl.Utf8 for c in STATION_COLUMNS})


def make_ctx(
    *,
    ecom: RawResponse | None = None,
    extras: pl.DataFrame | None = None,
    previous: pl.DataFrame | None = None,
    force_fallback: set[str] | None = None,
    country: StubCountry | None = None,
) -> CaptureContext:
    config = StubConfig(
        countries={"US": country or StubCountry()},
        us_extra_ids=EXTRAS if extras is None else extras,
    )
    return CaptureContext(
        capture_id=CAPTURE_ID,
        capture_date=CAPTURE_DATE,
        fetch_config=config,
        interp_config=config,
        previous_stations=stations_frame([]) if previous is None else previous,
        previous_fx=pl.DataFrame(),
        previous_status=None,
        shared={} if ecom is None else {"ecom-api": ecom},
        force_fallback=force_fallback or set(),
    )


def response(
    key: str,
    url: str,
    *,
    fixture: str | None = None,
    body: bytes = b"",
    status: int | None = 200,
    error: str | None = None,
    offset: int = 0,
) -> RawResponse:
    payload = (FIXTURES / fixture).read_bytes() if fixture else body
    return RawResponse(
        key=key,
        url=url,
        status=status,
        headers={"Content-Type": "text/html;charset=UTF-8"},
        received_at_utc=CAPTURE_START + timedelta(seconds=offset),
        elapsed_ms=120,
        body=payload,
        error=error,
    )


def ecom_ok() -> RawResponse:
    return response(
        "shared/01-ecom-api",
        "https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json",
        fixture="us_ecom_warehouses.json",
    )


def ecom_failed() -> RawResponse:
    return response(
        "shared/01-ecom-api",
        "https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json",
        body=b"",
        status=None,
        error="ConnectTimeout",
    )


class FakeClient:
    """Stands in for http.Client. The real one is covered by test_http.py."""

    def __init__(self, responder, *, abandon_hosts=(), abandon_after=None):
        self.responder = responder
        self.calls: list[tuple[str, str, str]] = []
        self.abandon_hosts = set(abandon_hosts)
        self.abandon_after = dict(abandon_after or {})
        self.counts: dict[str, int] = {}

    @staticmethod
    def host_of(url: str) -> str:
        return url.split("://", 1)[-1].split("/", 1)[0]

    def request(self, key, url, *, profile="default", headers=None, expect_json=True):
        host = self.host_of(url)
        self.calls.append((key, url, profile))
        self.counts[host] = self.counts.get(host, 0) + 1
        return self.responder(key, url)

    def abandoned(self, url: str) -> bool:
        host = self.host_of(url)
        if host in self.abandon_hosts:
            return True
        limit = self.abandon_after.get(host)
        return limit is not None and self.counts.get(host, 0) >= limit


TOPUP_BODY = (
    b'{"1793":{"premium":"4.199","regular":"3.499"},'
    b'"1765":{"premium":"5.299","regular":"4.999"},'
    b'"1772":{"premium":"6.149","regular":"5.849"}}'
)


def fallback_responder(key, url):
    if "lookup" in key:
        return response(key, url, fixture="us_lookup_us.json", offset=30)
    return response(key, url, body=TOPUP_BODY, offset=40)


def by_id(result) -> dict:
    return {s.source_station_id: s for s in result.stations}


def grades_of(station) -> dict[str, str]:
    return {p.grade_raw: p.price_raw for p in station.prices}


def warning_details(result, code: str) -> list[str]:
    return [w.detail for w in result.warnings if w.code == code]


# --------------------------------------------------------------------------- batching


def test_batches_split_by_ten():
    ids = [str(i) for i in range(1, 26)]
    assert us.batches(ids) == [ids[0:10], ids[10:20], ids[20:25]]


def test_batch_size_and_seen_window_come_from_the_country_config():
    assert us.batch_size(make_ctx()) == 10
    assert us.seen_window_days(make_ctx()) == 30
    small = make_ctx(country=StubCountry(batch_size=3, seen_within_days=7))
    assert us.batch_size(small) == 3
    assert us.seen_window_days(small) == 7
    ids = [str(i) for i in range(1, 8)]
    assert us.batches(ids, us.batch_size(small)) == [
        ["1", "2", "3"],
        ["4", "5", "6"],
        ["7"],
    ]


def test_price_and_lookup_urls_come_from_the_country_config():
    ctx = make_ctx()
    assert us.price_url(ctx) == "https://www.costco.com/AjaxGetGasPricesService"
    assert us.price_params(ctx) == {}
    assert us.lookup_url(ctx) == (
        "https://www.costco.ca/AjaxWarehouseBrowseLookupView"
        "?hasGas=true&populateWarehouseDetails=true&countryCode=US"
    )


def test_batch_url_puts_the_ids_first_and_appends_configured_params():
    assert us.batch_url(make_ctx(), ["1364", "140"]) == (
        "https://www.costco.com/AjaxGetGasPricesService?warehouseid=1364_140"
    )
    extra = make_ctx(country=StubCountry(params={"locale": "en-US"}))
    assert us.batch_url(extra, ["1364"]) == (
        "https://www.costco.com/AjaxGetGasPricesService?warehouseid=1364&locale=en-US"
    )
    assert us.ids_in_url(us.batch_url(extra, ["1364", "140"])) == ["1364", "140"]


def test_price_service_processes_only_the_first_ten_ids():
    raw = response(
        "US/02-gasprices-001",
        "https://www.costco.com/AjaxGetGasPricesService?warehouseid=1_2_3_4_6_8_9_10_11_13_14",
        fixture="us_gasprices_batch_cap.json",
    )
    requested = us.ids_in_url(raw.url)
    parsed = us.parse_price_batch(raw)
    assert len(requested) == 11
    assert set(parsed) == set(requested[:10])
    assert "14" not in parsed


def test_parse_price_batch_rejects_error_message_and_html():
    bad_id = response(
        "US/02-gasprices-001",
        "u",
        body=b'{"errorMessage":"warehouse id supplied, abc, is not a number"}',
    )
    blocked = response(
        "US/02-gasprices-001",
        "u",
        body=b"<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD></HTML>",
    )
    assert us.parse_price_batch(bad_id) is None
    assert us.parse_price_batch(blocked) is None


def test_parse_price_batch_accepts_json_served_as_text_html():
    raw = response("US/02-gasprices-001", "u", fixture="us_gasprices_batch_ua.json")
    assert raw.headers["Content-Type"] == "text/html;charset=UTF-8"
    parsed = us.parse_price_batch(raw)
    assert parsed["1364"] == {"premium": "4.629", "regular": "3.999"}
    assert parsed["120"] == {"premium": "3.829"}
    assert parsed["140"]["clear"] == "5.699"
