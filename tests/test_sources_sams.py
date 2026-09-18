"""The Sam's Club source, against responses captured from the live endpoints."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

import httpx
import pytest

from club_gas.config import HttpConfig
from club_gas.http import BudgetExceeded, Client
from club_gas.sources import sams
from club_gas.sources.base import RawResponse

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CAPTURED_AT = datetime(2026, 9, 17, 16, 45, tzinfo=UTC)
# The roster request's answer on a GitHub-hosted runner, from the dry capture of
# 2026-09-17 (run 35281651770), byte for byte.
PERIMETERX = (FIXTURES / "blocks" / "perimeterx_412.json").read_bytes()


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


def tempo(n: int, club_id: str, prices: dict[str, float] | None = None) -> RawResponse:
    """The recorded Tempo answer, relabelled for another club or other prices."""
    payload = json.loads(body("sams_fuel_6376.json"))
    block = payload["data"]["contentLayout"]["modules"][0]["configs"]["storeFuelPrices"]
    block["id"] = f"CPF_FUELPRICE_PROD_{club_id}"
    if prices is not None:
        block["prices"] = [{"name": k, "price": v, "type": "fuel"} for k, v in prices.items()]
    return response(sams.KEY_PRICES.format(n=n), json.dumps(payload).encode())


class ScriptedClient:
    """Answers each request from a script; the script raises to end the budget."""

    def __init__(self, answer):
        self.answer = answer
        self.keys: list[str] = []

    def abandoned(self, url: str) -> bool:
        return False

    def request(self, key, url, *, headers=None, expect_json=True) -> RawResponse:
        self.keys.append(key)
        return self.answer(key)


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
    # The record also lists MIDGRAD at 2.979, below its own UNLEAD, which the
    # guard drops; see the test after this one.
    assert prices == {"UNLEAD": 3.699, "PREMIUM": 4.399, "DIESEL": 5.699}

    # The same club's stale vivaldi value, which must never reach a station.
    roster = json.loads(body("sams_clubfinder.json"))
    stale = {
        p["name"]: p["price"] for c in roster if c["id"] == 6376 for p in (c.get("gasPrices") or [])
    }
    assert stale["UNLEAD"] == 2.319
    assert prices["UNLEAD"] != stale["UNLEAD"]


def test_a_grade_priced_below_the_clubs_own_regular_is_dropped():
    """Mid-grade cannot be cheaper than regular, yet club 6376 lists MIDGRAD at
    2.979 against UNLEAD 3.699, and the API notes say to drop any grade below
    the club's own UNLEAD. A grade at or above it stays; the one dropped is
    named in a warning, so a real price that falls below regular still shows.
    """
    result = sams.SamsSource().parse(
        [
            roster_response(),
            price_response(1),
            tempo(2, "8299", {"UNLEAD": 3.499, "MIDGRAD": 3.499, "PREMIUM": 4.099}),
        ],
        _ctx(),
    )

    prices = {
        s.source_station_id: {p.grade_raw: p.price_raw for p in s.prices} for s in result.stations
    }
    assert prices["6376"] == {"UNLEAD": "3.699", "PREMIUM": "4.399", "DIESEL": "5.699"}
    assert prices["8299"] == {"UNLEAD": "3.499", "MIDGRAD": "3.499", "PREMIUM": "4.099"}
    assert [(w.code, w.detail) for w in result.warnings if w.code == "below_regular"] == [
        ("below_regular", "6376:MIDGRAD=2.979")
    ]


def test_with_no_regular_to_compare_against_every_grade_stays():
    result = sams.SamsSource().parse(
        [roster_response(), tempo(1, "6376", {"PREMIUM": 4.399, "MIDGRAD": 2.979})], _ctx()
    )

    station = next(s for s in result.stations if s.source_station_id == "6376")
    assert {p.grade_raw for p in station.prices} == {"PREMIUM", "MIDGRAD"}


def test_a_club_with_no_price_response_is_warned_and_dropped():
    """Better a short capture than a station carrying a stale roster price.

    A club that was never asked about is `not_reached`, not `no_current_price`:
    this capture knows nothing about it, and publish must not read its absence
    as Sam's Club no longer listing it. A club asked and answered without a
    price is the one that says something.
    """
    result = sams.SamsSource().parse([roster_response()], _ctx())

    assert result.stations == []
    assert sorted(w.detail for w in result.warnings) == ["4857", "6376", "8248", "8299"]
    assert {w.code for w in result.warnings} == {"not_reached"}

    asked = tempo(2, "8299", {})
    result = sams.SamsSource().parse([roster_response(), asked], _ctx())

    codes = {w.detail: w.code for w in result.warnings}
    assert codes == {
        "6376": "not_reached",
        "8299": "no_current_price",
        "8248": "not_reached",
        "4857": "not_reached",
    }


def test_a_sweep_its_budget_cut_short_says_so_and_names_every_club_it_never_asked():
    """The roster is sorted by distance from 75001, so a sweep cut short loses
    the same far-away clubs every time. It has to say so, and say which."""

    def answer(key: str) -> RawResponse:
        if key == sams.KEY_ROSTER:
            return roster_response()
        if key == sams.KEY_PRICES.format(n=1):
            return price_response(1)
        if key == sams.KEY_PRICES.format(n=2):
            return tempo(2, "8299")
        raise BudgetExceeded("country-US")

    client = ScriptedClient(answer)
    responses = sams.SamsSource().fetch(client, _ctx())
    result = sams.SamsSource().parse(responses, _ctx())

    assert client.keys == [sams.KEY_ROSTER, *(sams.KEY_PRICES.format(n=n) for n in (1, 2, 3))]
    # The marker for club 3 goes into the bundle, so a rebuild sees the same.
    assert (responses[-1].key, responses[-1].error) == (
        sams.KEY_PRICES.format(n=3),
        "deadline_exceeded",
    )
    assert result.requests == 3
    assert sorted(s.source_station_id for s in result.stations) == ["6376", "8299"]
    warnings = [(w.code, w.detail) for w in result.warnings if w.code != "below_regular"]
    assert warnings == [
        ("not_reached", "8248"),
        ("not_reached", "4857"),
        ("budget_exhausted", "2 of 4 fuel clubs not reached"),
    ]


def test_a_perimeterx_refusal_stops_the_sweep_and_is_named_in_the_errors():
    """Sam's Club refuses with HTTP 412 and a PerimeterX body. Two of those and
    the client leaves the host alone (http.toml signals_before_abandon), rather
    than sending the rest of ~531 requests into the challenge, and the failed
    feed says what refused it instead of recording no error at all."""
    sent: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request.url.path)
        if "clubfinder" in request.url.path:
            return httpx.Response(200, content=body("sams_clubfinder.json"))
        return httpx.Response(
            412, content=PERIMETERX, headers={"Content-Type": "application/json; charset=UTF-8"}
        )

    cfg = HttpConfig(backoff_seconds=(0.0, 0.0), min_interval_seconds=0.0)
    with Client(cfg, transport=httpx.MockTransport(handler)) as client:
        responses = sams.SamsSource().fetch(client, _ctx())
    result = sams.SamsSource().parse(responses, _ctx())

    assert len(sent) == 1 + cfg.block_signals_before_abandon
    assert result.stations == []
    assert [(e.code, e.http_status, e.detail) for e in result.errors] == [
        ("price_request_failed", 412, "2 of 2 price requests: PerimeterX challenge")
    ]


def test_a_refused_roster_names_perimeterx():
    result = sams.SamsSource().parse([response(sams.KEY_ROSTER, PERIMETERX, status=412)], _ctx())

    assert [(e.code, e.http_status, e.detail) for e in result.errors] == [
        ("roster_unavailable", 412, "PerimeterX challenge")
    ]


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
    # The feed says "CST" plus isDSTObserved; the pair resolves to a real zone.
    assert station.timezone == "America/Chicago"


def test_the_result_is_branded_so_keys_cannot_collide():
    result = sams.SamsSource().parse([roster_response(), price_response()], _ctx())
    assert (result.country, result.brand, result.source) == ("US", "SAMS", "sams-clubfinder")


def _ctx():
    from types import SimpleNamespace

    return SimpleNamespace(fetch_config=SimpleNamespace(feeds={}), capture_date=None)


# --------------------------------------------------------------- timezone


@pytest.mark.parametrize(
    ("abbreviation", "observes_dst", "expected"),
    [
        ("EST", True, "America/New_York"),
        ("CST", True, "America/Chicago"),
        ("MST", True, "America/Denver"),
        ("PST", True, "America/Los_Angeles"),
        ("MST", False, "America/Phoenix"),
        ("HST", False, "Pacific/Honolulu"),
    ],
)
def test_the_zone_needs_both_the_abbreviation_and_the_dst_flag(
    abbreviation, observes_dst, expected
):
    """These six pairs are every one that occurs across the 531 fuel clubs.

    MST is the case that matters: Denver where DST is observed, Phoenix where it
    is not. The abbreviation alone cannot tell them apart, and 13 Arizona clubs
    ride on it.
    """
    club = {"timeZone": abbreviation, "clubAttributes": {"isDSTObserved": observes_dst}}
    assert sams.timezone_of(club) == expected


def test_an_unknown_zone_pair_is_none_rather_than_a_guess():
    """normalize drops a station with no timezone, which is the right outcome:
    a wrong zone puts a price on the wrong local day."""
    assert sams.timezone_of({"timeZone": "XYZ", "clubAttributes": {"isDSTObserved": True}}) is None
    assert sams.timezone_of({"timeZone": None}) is None
    assert sams.timezone_of({}) is None


def test_every_resolved_zone_is_one_python_actually_knows():
    from zoneinfo import ZoneInfo

    for (abbreviation, dst), zone in sams.IANA_BY_ZONE.items():
        assert ZoneInfo(zone), (abbreviation, dst, zone)
