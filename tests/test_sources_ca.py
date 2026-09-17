from __future__ import annotations

from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from club_gas.config import load_config
from club_gas.http import Client
from club_gas.sources.base import CaptureContext, RawResponse
from club_gas.sources.ca import (
    CaSource,
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
        received_at_utc=datetime(2026, 9, 15, 19, 10, tzinfo=UTC),
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


def test_lookup_fields_and_grade_keys():
    result, responses = run(serve(LOOKUP_BODY))

    assert result.country == "CA"
    assert result.source == "costco-ca-lookup"
    assert [r.key for r in responses] == ["CA/01-lookup"]
    assert result.requests == 1
    assert result.errors == []

    vaudreuil = station(result, "1213")
    assert vaudreuil.name == "Vaudreuil"
    assert vaudreuil.city == "VAUDREUIL-DORION"
    assert vaudreuil.region == "QC"
    assert vaudreuil.postcode == "J7V 0M8"
    assert vaudreuil.address == "22400 CH DUMBERRY"
    assert vaudreuil.id_origin == "lookup"
    assert vaudreuil.opening_date == date(2015, 10, 16)
    assert vaudreuil.has_hours is True
    # warehouseid and oid are not grades; every other key is.
    assert [(p.grade_raw, p.price_raw) for p in vaudreuil.prices] == [
        ("diesel", "2.649"),
        ("regular", "1.799"),
        ("premium", "1.999"),
    ]


def test_not_open_no_hours_and_priced_before_open():
    result, _ = run(serve(LOOKUP_BODY))

    lloydminster = station(result, "1790")
    assert lloydminster.opening_date == date(2026, 11, 19)
    assert source_filter_reason(lloydminster, date(2026, 9, 15)) == "not_open"

    calgary = station(result, "1813")
    assert calgary.opening_date == date(2027, 4, 1)
    assert calgary.has_hours is False
    assert calgary.prices == ()
    assert source_filter_reason(calgary, date(2026, 9, 15)) == "not_open"

    # An already-open, priced, staffed station passes every source filter.
    assert source_filter_reason(station(result, "1324"), date(2026, 9, 15)) is None

    # #1790 is priced at 1.549 CAD/L before it opens, inside bounds["CAD/L"];
    # #1813 has no price at all.
    assert ("priced_before_open", "1790") in codes(result.warnings)
    assert ("priced_before_open", "1813") not in codes(result.warnings)


def test_ecom_api_supplies_coordinates_and_timezone():
    result, _ = run(serve(LOOKUP_BODY), shared={"ecom-api": ecom_response()})

    st_johns = station(result, "1324")
    assert st_johns.lat == pytest.approx(47.50731997)  # 8 decimals, not the lookup's 47.507
    assert st_johns.lon == pytest.approx(-52.83506698)
    assert st_johns.timezone == "America/St_Johns"
    assert station(result, "530").timezone == "America/Toronto"
    # #1213, #1790 and #1813 are not in ecom-api, so their provinces decide.
    assert codes(result.warnings) == [
        ("timezone_from_region", "1213"),
        ("timezone_from_region", "1790"),
        ("priced_before_open", "1790"),
        ("timezone_from_region", "1813"),
    ]

    # #1213 is not in this ecom-api response, so it keeps the lookup's 3 decimals.
    vaudreuil = station(result, "1213")
    assert vaudreuil.lat == pytest.approx(45.415)
    assert vaudreuil.timezone == "America/Toronto"  # QC, from the province table


def test_stations_csv_timezone_wins_over_the_province_table():
    previous = stations_frame([cached_row("1213", timezone="America/Montreal")])

    result, _ = run(serve(LOOKUP_BODY), previous=previous)

    assert station(result, "1213").timezone == "America/Montreal"
    assert ("timezone_from_region", "1213") not in codes(result.warnings)


def test_a_failed_lookup_falls_back_to_the_price_service():
    previous = stations_frame(
        [
            cached_row("1213"),
            cached_row("530", name="N London", city="LONDON", region="ON", timezone=None),
        ]
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        if "AjaxWarehouseBrowseLookupView" in str(request.url):
            return httpx.Response(
                403, content=b"<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD></HTML>"
            )
        return httpx.Response(
            200, content=PRICES_BODY, headers={"content-type": "text/html;charset=UTF-8"}
        )

    result, responses = run(handler, previous=previous)

    assert [r.key for r in responses] == ["CA/01-lookup", "CA/02-gasprices-01"]
    assert str(seen[1].url) == "https://www.costco.ca/AjaxGetGasPricesService?warehouseid=530_1213"
    assert result.source == "costco-ca-gasprices"
    assert codes(result.warnings)[:2] == [
        ("fallback_used", "request_failed"),
        ("metadata_from_cache", None),
    ]
    assert result.errors == []

    vaudreuil = station(result, "1213")
    assert vaudreuil.id_origin == "cache"
    assert vaudreuil.city == "VAUDREUIL-DORION"  # metadata from stations.csv
    assert [(p.grade_raw, p.price_raw) for p in vaudreuil.prices] == [
        ("diesel", "2.649"),
        ("premium", "1.999"),
        ("regular", "1.799"),
    ]
    # #530 has no cached timezone, so the province table fills it in.
    assert station(result, "530").timezone == "America/Toronto"
    assert ("timezone_from_region", "530") in codes(result.warnings)
    # Only cached CA ids are polled, so #56's ".959" placeholder is never requested,
    # and ids the response returns without a cached row are reported, not invented.
    assert "56" not in [s.source_station_id for s in result.stations]
    assert ("no_metadata", "56") in codes(result.warnings)


def test_an_unparseable_body_triggers_the_fallback():
    previous = stations_frame([cached_row("1213")])

    def handler(request: httpx.Request) -> httpx.Response:
        if "AjaxWarehouseBrowseLookupView" in str(request.url):
            # Valid JSON is served as text/html here, so Content-Type can never be
            # used to spot a block; only the body can.
            return httpx.Response(
                200,
                content=b"\r\n<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD></HTML>",
                headers={"content-type": "text/html;charset=UTF-8"},
            )
        return httpx.Response(200, content=PRICES_BODY)

    result, responses = run(handler, previous=previous)

    assert [r.key for r in responses] == ["CA/01-lookup", "CA/02-gasprices-01"]
    assert result.source == "costco-ca-gasprices"
    assert ("fallback_used", "unparseable_body") in codes(result.warnings)


def test_a_lookup_with_no_usable_station_triggers_the_fallback():
    previous = stations_frame([cached_row("1213")])
    # Both warehouses in this body open in the future and have no gas hours.
    empty = (
        b"\r\n[false,"
        b'{"stlocID":1790,"locationName":"Lloydminster","state":"AB",'
        b'"openDate":"Nov 19, 2026","gasStationHours":[],'
        b'"gasPrices":{"warehouseid":"1790","regular":"1.549"}},'
        b'{"stlocID":1813,"locationName":"N Calgary","state":"AB",'
        b'"openDate":"Apr 1, 2027","gasStationHours":[],'
        b'"gasPrices":{"warehouseid":"1813"}}]'
    )

    def handler(request: httpx.Request) -> httpx.Response:
        if "AjaxWarehouseBrowseLookupView" in str(request.url):
            return httpx.Response(200, content=empty)
        return httpx.Response(200, content=PRICES_BODY)

    result, responses = run(handler, previous=previous)

    assert [r.key for r in responses] == ["CA/01-lookup", "CA/02-gasprices-01"]
    assert ("fallback_used", "zero_stations") in codes(result.warnings)


def test_force_fallback_skips_the_lookup_entirely():
    previous = stations_frame([cached_row("1213")])
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=PRICES_BODY)

    result, responses = run(handler, previous=previous, force_fallback=("CA",))

    assert [str(r.url) for r in seen] == [
        "https://www.costco.ca/AjaxGetGasPricesService?warehouseid=1213"
    ]
    assert [r.key for r in responses] == ["CA/02-gasprices-01"]
    assert result.source == "costco-ca-gasprices"
    assert ("fallback_used", "forced") in codes(result.warnings)


def test_stale_cached_stations_are_not_polled():
    previous = stations_frame(
        [
            cached_row("1213", last_seen_utc=datetime(2026, 9, 14, 18, 17)),
            cached_row("530", last_seen_utc=datetime(2026, 7, 1, 18, 17)),
        ]
    )
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        return httpx.Response(200, content=PRICES_BODY)

    run(handler, previous=previous, force_fallback=("CA",))

    assert str(seen[0].url).endswith("warehouseid=1213")
