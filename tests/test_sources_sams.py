"""The Sam's Club source, against responses captured from the live endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import pytest

from club_gas.sources import sams
from club_gas.sources.base import RawResponse

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CAPTURED_AT = datetime(2026, 9, 17, 16, 45, tzinfo=UTC)


def body(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


def response(key: str, payload: bytes, status: int = 200, error: str | None = None) -> RawResponse:
    return RawResponse(
        key=key,
        url="https://www.samsclub.com/x",
        status=status,
        headers={},
        received_at_utc=CAPTURED_AT,
        elapsed_ms=1,
        body=payload,
        error=error,
    )


def roster_response() -> RawResponse:
    return response(sams.KEY_ROSTER, body("sams_clubfinder.json"))


def price_response(n: int = 1) -> RawResponse:
    return response(sams.KEY_PRICES.format(n=n), body("sams_fuel_6376.json"))


# --------------------------------------------------------------------- urls


def test_roster_url_saturates_both_caps():
    """Both are real, enforced caps; saturating them returns the whole country."""
    url = sams.roster_url(sams.DEFAULT_ROSTER_URL, "75001")
    q = parse_qs(urlsplit(url).query)
    assert q["singleLineAddr"] == ["75001"]
    assert q["nbrOfStores"] == ["2147483647"]
    assert q["distance"] == ["2147483647"]


def test_price_url_carries_one_club_and_the_document_hash():
    url = sams.price_url(sams.DEFAULT_PRICE_URL, "6376", "75244")
    assert sams.PRICE_QUERY_SHA256 in url
    variables = json.loads(parse_qs(urlsplit(url).query)["variables"][0])
    # $nodeId is typed String! -- an array is rejected by the schema, so this
    # endpoint is one club per request and no batching is possible.
    assert variables["nodeId"] == "6376"
    assert isinstance(variables["nodeId"], str)
    assert variables["pageId"] == "fuel-center"


def test_the_csrf_header_is_the_one_that_works():
    """Without it the gateway 400s; content-type: application/json does not do."""
    assert sams.PRICE_HEADERS["x-apollo-operation-name"] == "HyperLocalPagesTempo"


# ------------------------------------------------------------------- roster


def test_only_clubs_that_sell_fuel_are_polled():
    roster = json.loads(body("sams_clubfinder.json"))
    ids = sams.fuel_club_ids(roster)

    assert ids == ["6376", "8299", "8248", "4857"]
    # 4925 sells no fuel and 6225 is Puerto Rico, where no club does.
    assert "4925" not in ids and "6225" not in ids


def test_gasprices_beats_the_services_tag_in_both_directions():
    """Measured on the live roster, the two disagree both ways.

    Clubs 8132 (Springdale OH) and 4704 (Fresno CA) carry prices with no "gas"
    tag; club 7673 (Lebanon) tags "gas" and carries none. Filtering on the tag
    loses two real fuel clubs and polls one that has nothing.
    """
    untagged = {"id": 8132, "services": ["cafe"], "gasPrices": [{"name": "UNLEAD", "price": 3.1}]}
    tagged_dry = {"id": 7673, "services": ["gas"], "gasPrices": []}

    ids = sams.fuel_club_ids([untagged, tagged_dry])

    assert ids == ["8132"]


def test_a_roster_that_is_not_a_list_is_an_error_not_an_empty_capture():
    result = sams.SamsSource().parse([response(sams.KEY_ROSTER, b'{"statusCode":500}')], _ctx())
    assert [e.code for e in result.errors] == ["roster_unavailable"]
    assert result.stations == []


# ------------------------------------------------------------------- prices


def test_prices_come_from_the_tempo_module_never_from_vivaldi():
    """The whole point: vivaldi's gasPrices ran 31-43% below what members pay."""
    result = sams.SamsSource().parse([roster_response(), price_response()], _ctx())

    station = next(s for s in result.stations if s.source_station_id == "6376")
    prices = {p.grade_raw: float(p.price_raw) for p in station.prices}
    assert prices == {"UNLEAD": 3.699, "PREMIUM": 4.399, "DIESEL": 5.699, "MIDGRAD": 2.979}

    # The same club's stale vivaldi value, which must never reach a station.
    roster = json.loads(body("sams_clubfinder.json"))
    stale = {
        p["name"]: p["price"] for c in roster if c["id"] == 6376 for p in (c.get("gasPrices") or [])
    }
    assert stale["UNLEAD"] == 2.319
    assert prices["UNLEAD"] != stale["UNLEAD"]


def test_a_club_with_no_price_response_is_warned_and_dropped():
    """Better a short capture than a station carrying a stale roster price."""
    result = sams.SamsSource().parse([roster_response()], _ctx())

    assert result.stations == []
    assert sorted(w.detail for w in result.warnings) == ["4857", "6376", "8248", "8299"]
    assert {w.code for w in result.warnings} == {"no_current_price"}


def test_an_unknown_club_record_is_dropped():
    """A comma-joined nodeId answers 200 with id INVALID STORE and no prices."""
    payload = json.loads(body("sams_fuel_6376.json"))
    block = payload["data"]["contentLayout"]["modules"][0]["configs"]["storeFuelPrices"]
    block["id"] = "INVALID STORE"
    result = sams.SamsSource().parse(
        [roster_response(), response(sams.KEY_PRICES.format(n=1), json.dumps(payload).encode())],
        _ctx(),
    )
    assert result.stations == []


@pytest.mark.parametrize("status", [400, 403, 429, 500])
def test_a_failed_price_response_never_becomes_a_station(status):
    result = sams.SamsSource().parse(
        [roster_response(), response(sams.KEY_PRICES.format(n=1), b"", status=status)], _ctx()
    )
    assert result.stations == []


# ------------------------------------------------------------------ station


def test_a_station_carries_what_the_dashboard_needs():
    result = sams.SamsSource().parse([roster_response(), price_response()], _ctx())
    station = next(s for s in result.stations if s.source_station_id == "6376")

    assert station.name == "Addison Sam's Club"
    assert station.region == "TX"
    assert station.city == "Dallas"
    assert station.postcode == "75244"
    assert station.lat == pytest.approx(32.921881)
    assert station.lon == pytest.approx(-96.843062)
    # The fuel-centre page resolves from the id alone, with no city slug.
    assert station.alt_id == "6376"
    # timeZone is an abbreviation like "CST", not an IANA zone, so it is left
    # for downstream resolution rather than written in wrong.
    assert station.timezone is None


def test_the_result_is_branded_so_keys_cannot_collide():
    result = sams.SamsSource().parse([roster_response(), price_response()], _ctx())
    assert (result.country, result.brand, result.source) == ("US", "SAMS", "sams-clubfinder")


def _ctx():
    from types import SimpleNamespace

    return SimpleNamespace(fetch_config=SimpleNamespace(feeds={}), capture_date=None)
