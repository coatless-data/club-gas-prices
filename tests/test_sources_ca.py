from __future__ import annotations

from datetime import date, datetime, timezone
from pathlib import Path

import httpx
import polars as pl
import pytest

from costco_gas.config import load_config
from costco_gas.http import Client
from costco_gas.sources.base import CaptureContext, RawResponse
from costco_gas.sources.ca import (
    cached_stations,
    parse_lookup_body,
    parse_open_date,
    source_filter_reason,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"
LOOKUP_BODY = (FIXTURES / "ca_lookup.body").read_bytes()
PRICES_BODY = (FIXTURES / "ca_gasprices.body").read_bytes()
ECOM_BODY = (FIXTURES / "ca_ecom_warehouses.json").read_bytes()


def ecom_response(body: bytes = ECOM_BODY, status: int = 200) -> RawResponse:
    return RawResponse(
        key="shared/ecom-api",
        url="https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json",
        status=status,
        headers={},
        received_at_utc=datetime(2026, 9, 15, 19, 10, tzinfo=timezone.utc),
        elapsed_ms=900,
        body=body,
        error=None,
    )


def stations_frame(rows: list[dict]) -> pl.DataFrame:
    return pl.DataFrame(
        rows,
        schema={
            "station_key": pl.Utf8,
            "country": pl.Utf8,
            "source_station_id": pl.Utf8,
            "alt_id": pl.Utf8,
            "name": pl.Utf8,
            "name_local": pl.Utf8,
            "address": pl.Utf8,
            "city": pl.Utf8,
            "region": pl.Utf8,
            "postcode": pl.Utf8,
            "lat": pl.Float64,
            "lon": pl.Float64,
            "timezone": pl.Utf8,
            "grades_seen": pl.Utf8,
            "first_seen_utc": pl.Datetime("us"),
            "last_seen_utc": pl.Datetime("us"),
            "status": pl.Utf8,
            "superseded_by": pl.Utf8,
        },
    )


def cached_row(station_id: str, **overrides) -> dict:
    row = {
        "station_key": f"CA-{station_id}",
        "country": "CA",
        "source_station_id": station_id,
        "alt_id": None,
        "name": "Vaudreuil",
        "name_local": None,
        "address": "22400 CH DUMBERRY",
        "city": "VAUDREUIL-DORION",
        "region": "QC",
        "postcode": "J7V 0M8",
        "lat": 45.415,
        "lon": -74.038,
        "timezone": "America/Toronto",
        "grades_seen": "diesel|premium|regular",
        "first_seen_utc": datetime(2026, 8, 1, 0, 17),
        "last_seen_utc": datetime(2026, 9, 14, 18, 17),
        "status": "active",
        "superseded_by": None,
    }
    row.update(overrides)
    return row


def make_ctx(cfg, *, shared=None, previous=None, force_fallback=()) -> CaptureContext:
    return CaptureContext(
        capture_id="2026-09-15T1911Z",
        capture_date=date(2026, 9, 15),
        fetch_config=cfg.fetch_view(),
        interp_config=cfg.interp_view(),
        previous_stations=previous if previous is not None else pl.DataFrame(),
        previous_fx=pl.DataFrame(),
        previous_status=None,
        shared=shared or {},
        force_fallback=set(force_fallback),
    )


def run(handler, *, shared=None, previous=None, force_fallback=()):
    cfg = load_config(ROOT)
    ctx = make_ctx(cfg, shared=shared, previous=previous, force_fallback=force_fallback)
    client = Client(cfg.http, transport=httpx.MockTransport(handler))
    source = CaSource()
    responses = source.fetch(client, ctx)
    return source.parse(responses, ctx), responses


def serve(body: bytes, status: int = 200):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status, content=body, headers={"content-type": "text/html;charset=UTF-8"}
        )

    return handler


def station(result, station_id):
    matches = [s for s in result.stations if s.source_station_id == station_id]
    assert matches, f"{station_id} not in {[s.source_station_id for s in result.stations]}"
    return matches[0]


def codes(items):
    return [(i.code, i.detail) for i in items]


def test_parse_lookup_body_skips_the_leading_crlf_and_the_false_element():
    assert LOOKUP_BODY.startswith(b"\r\n")

    entries = parse_lookup_body(LOOKUP_BODY)

    assert [e["stlocID"] for e in entries] == [1324, 1213, 530, 1790, 1813]
    with pytest.raises(ValueError):
        parse_lookup_body(b"\r\n<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD></HTML>")


def test_parse_open_date_is_locale_independent():
    assert parse_open_date("Aug 23, 1995") == date(1995, 8, 23)
    assert parse_open_date("Nov 19, 2026") == date(2026, 11, 19)
    assert parse_open_date("Apr 1, 2027") == date(2027, 4, 1)
    assert parse_open_date("") is None
    assert parse_open_date(None) is None
    assert parse_open_date("soon") is None


def test_cached_stations_keeps_only_recent_ca_rows():
    previous = stations_frame(
        [
            cached_row("1213"),
            cached_row("530", last_seen_utc=datetime(2026, 7, 1, 18, 17)),
            cached_row("1364", station_key="US-1364", country="US"),
        ]
    )

    cached = cached_stations(previous, date(2026, 9, 15))

    assert sorted(cached) == ["1213"]  # #530 is 76 days stale; #1364 is not Canadian
    assert cached["1213"]["city"] == "VAUDREUIL-DORION"
    # The window is the caller's: CountryConfig.seen_within_days, 30 for CA.
    assert sorted(cached_stations(previous, date(2026, 9, 15), 90)) == ["1213", "530"]
    assert cached_stations(pl.DataFrame(), date(2026, 9, 15)) == {}
