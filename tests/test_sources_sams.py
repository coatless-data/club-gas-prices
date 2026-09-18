"""The Sam's Club source, against its public server-rendered pages.

Sam's Club is read from the locator sitemap (the roster) and each club's
`/club/<id>/fuel-center` page, whose `__NEXT_DATA__` embeds the current fuel
record. A club with no fuel centre 307s and is skipped. These tests build those
pages compactly rather than carrying a 250 KB fixture per club.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from club_gas.config import HttpConfig
from club_gas.http import BudgetExceeded, Client
from club_gas.sources import sams
from club_gas.sources.base import RawResponse

FIXTURES = Path(__file__).resolve().parent / "fixtures"
CAPTURED_AT = datetime(2026, 9, 18, 16, 45, tzinfo=UTC)
# A PerimeterX 412 body, kept so the safety net can be exercised even though the
# HTML surface is not gated the way the JSON API is.
PERIMETERX = (FIXTURES / "blocks" / "perimeterx_412.json").read_bytes()


def sitemap(ids: list[str]) -> bytes:
    locs = "".join(f"<loc>https://www.samsclub.com/club/{i}-city-tx</loc>" for i in ids)
    return f'<?xml version="1.0"?><urlset>{locs}</urlset>'.encode()


def fuel_page(
    club_id: str,
    prices: dict[str, float],
    *,
    record_id: str | None = None,
    name: str = "Test Sam's Club",
    city: str = "Dallas",
    state: str = "TX",
    postcode: str = "75244",
    tz: str | None = "America/Chicago",
    lat: float = 32.921881,
    lon: float = -96.843062,
) -> bytes:
    """A fuel-centre page's bytes: the `__NEXT_DATA__` the parser reads, minimal.

    `record_id` overrides the storeFuelPrices id (e.g. "INVALID STORE"); it
    defaults to the club's own, as "CPF_FUELPRICE_PROD_<id>".
    """
    block = {
        "countryCode": "US",
        "id": record_id or f"CPF_FUELPRICE_PROD_{club_id}",
        "metadata": {"dateCreated": "2026-09-18T08:15:34.337Z", "createdBy": "CPF_STORAGE"},
        "prices": [{"name": k, "price": v, "type": "fuel"} for k, v in prices.items()],
    }
    store_details = {"capabilities": [{"timeZone": tz}, {"timeZone": tz}]}
    next_data = {
        "props": {
            "pageProps": {
                "initialTempoData": {
                    "contentLayout": {
                        "modules": [
                            {"configs": {"storeDetails": store_details}},
                            {"configs": {"storeFuelPrices": block}},
                        ]
                    }
                },
                "initialNodeDetail": {
                    "data": {
                        "nodeDetail": {
                            "id": club_id,
                            "name": name,
                            "displayName": name,
                            "address": {
                                "addressLineOne": "1 Main St",
                                "addressLineTwo": None,
                                "city": city,
                                "state": state,
                                "postalCode": postcode,
                                "country": "US",
                            },
                            "geoPoint": {"latitude": lat, "longitude": lon},
                            "operationalHours": [
                                {"day": "Monday", "start": "06:00", "end": "22:00", "closed": False}
                            ],
                        }
                    }
                },
            }
        }
    }
    script = json.dumps(next_data)
    return (
        f"<!doctype html><html><body>"
        f'<script id="__NEXT_DATA__" type="application/json">{script}</script>'
        f"</body></html>"
    ).encode()


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


def sitemap_response(ids: list[str]) -> RawResponse:
    return response(sams.KEY_SITEMAP, sitemap(ids))


def club_response(club_id: str, prices: dict[str, float], **kwargs) -> RawResponse:
    return response(sams.KEY_CLUB.format(club_id=club_id), fuel_page(club_id, prices, **kwargs))


def redirect_response(club_id: str) -> RawResponse:
    return response(sams.KEY_CLUB.format(club_id=club_id), b"", status=307)


class ScriptedClient:
    """Answers each request from a script; the script raises to end the budget."""

    def __init__(self, answer):
        self.answer = answer
        self.keys: list[str] = []
        self.follow_redirects: list[bool] = []

    def abandoned(self, url: str) -> bool:
        return False

    def request(self, key, url, *, headers=None, expect_json=True, follow_redirects=True):
        self.keys.append(key)
        self.follow_redirects.append(follow_redirects)
        return self.answer(key)


def _ctx():
    return SimpleNamespace(fetch_config=SimpleNamespace(feeds={}), capture_date=None)


# --------------------------------------------------------------------- roster


def test_club_ids_are_deduped_and_numerically_ordered():
    """The sitemap repeats some clubs and lists them by name; the roster is the
    set of ids in numeric order, so a sweep cut short stops on a stable prefix."""
    body = (
        b"<loc>https://www.samsclub.com/club/8248-dallas-tx</loc>"
        b"<loc>https://www.samsclub.com/club/6376-addison-tx</loc>"
        b"<loc>https://www.samsclub.com/club/6376-addison-tx</loc>"
        b"<loc>https://www.samsclub.com/club/10003-somewhere-nv</loc>"
    )
    assert sams.club_ids(body.decode()) == ["6376", "8248", "10003"]


def test_a_sitemap_with_no_clubs_is_an_error_not_an_empty_capture():
    result = sams.SamsSource().parse([response(sams.KEY_SITEMAP, b"<urlset></urlset>")], _ctx())
    assert [e.code for e in result.errors] == ["sitemap_unavailable"]
    assert result.stations == []


def test_a_refused_sitemap_names_perimeterx():
    """The HTML surface is not gated like the JSON API, but if PerimeterX ever
    does answer the sitemap, the failure says so instead of recording nothing."""
    result = sams.SamsSource().parse([response(sams.KEY_SITEMAP, PERIMETERX, status=412)], _ctx())
    assert [(e.code, e.http_status, e.detail) for e in result.errors] == [
        ("sitemap_unavailable", 412, "PerimeterX challenge")
    ]


# -------------------------------------------------------------------- sweep


def test_the_sweep_asks_the_sitemap_then_every_club_without_following_redirects():
    """Redirects are left unfollowed so a non-fuel club's 307 costs one empty
    response, not its 250 KB club home."""

    def answer(key):
        if key == sams.KEY_SITEMAP:
            return sitemap_response(["6376", "8248"])
        return club_response(key.split("-")[-1], {"UNLEAD": 3.5})

    client = ScriptedClient(answer)
    responses = sams.SamsSource().fetch(client, _ctx())

    assert client.keys == [
        sams.KEY_SITEMAP,
        sams.KEY_CLUB.format(club_id="6376"),
        sams.KEY_CLUB.format(club_id="8248"),
    ]
    # The sitemap may follow redirects; every club request must not.
    assert client.follow_redirects == [True, False, False]
    assert len(responses) == 3


def test_a_non_fuel_club_redirects_and_is_skipped_without_a_warning():
    result = sams.SamsSource().parse(
        [
            sitemap_response(["4866", "6376"]),
            redirect_response("4866"),
            club_response("6376", {"UNLEAD": 3.799, "PREMIUM": 4.499}),
        ],
        _ctx(),
    )
    assert [s.source_station_id for s in result.stations] == ["6376"]
    # 4866 sells no fuel: not a station, and not a warning either.
    assert result.warnings == []
    assert result.errors == []


def test_a_sweep_its_budget_cut_short_names_every_club_it_never_asked():
    """The roster is in numeric order, so a sweep cut short loses the same tail
    every time. It has to say so, and say which."""

    def answer(key):
        if key == sams.KEY_SITEMAP:
            return sitemap_response(["4857", "6376", "8248", "8299"])
        if key in (sams.KEY_CLUB.format(club_id="4857"), sams.KEY_CLUB.format(club_id="6376")):
            return club_response(key.split("-")[-1], {"UNLEAD": 3.5, "PREMIUM": 4.1})
        raise BudgetExceeded("country-US")

    client = ScriptedClient(answer)
    responses = sams.SamsSource().fetch(client, _ctx())
    result = sams.SamsSource().parse(responses, _ctx())

    # The marker for the club the budget stopped on goes into the bundle, so a
    # rebuild replaying it sees the same stop.
    assert (responses[-1].key, responses[-1].error) == (
        sams.KEY_CLUB.format(club_id="8248"),
        "deadline_exceeded",
    )
    assert sorted(s.source_station_id for s in result.stations) == ["4857", "6376"]
    warnings = [(w.code, w.detail) for w in result.warnings]
    assert warnings == [
        ("not_reached", "8248"),
        ("not_reached", "8299"),
        ("budget_exhausted", "2 of 4 clubs not reached"),
    ]


def test_a_sweep_the_client_abandoned_partway_says_so_like_a_budget_cut():
    """Bot protection can refuse late in a sweep too. Once the client gives up on
    the host the loop stops, and the unasked clubs read the same as a budget cut:
    degraded, naming how many were missed."""

    class AbandonsAfterTwo(ScriptedClient):
        def abandoned(self, url):
            return len([k for k in self.keys if k.startswith(sams._KEY_CLUB_PREFIX)]) >= 2

    def answer(key):
        if key == sams.KEY_SITEMAP:
            return sitemap_response(["4857", "6376", "8248"])
        return club_response(key.split("-")[-1], {"UNLEAD": 3.5})

    client = AbandonsAfterTwo(answer)
    responses = sams.SamsSource().fetch(client, _ctx())
    result = sams.SamsSource().parse(responses, _ctx())

    assert (responses[-1].key, responses[-1].error) == (
        sams.KEY_CLUB.format(club_id="8248"),
        "host_abandoned",
    )
    warnings = [(w.code, w.detail) for w in result.warnings]
    assert warnings == [
        ("not_reached", "8248"),
        ("sweep_abandoned", "1 of 3 clubs not reached"),
    ]


def test_a_perimeterx_refusal_stops_the_sweep_and_is_named_in_the_errors():
    """If the club pages ever answer 412, two of them and the client leaves the
    host alone, rather than sending the rest of ~604 requests into the challenge,
    and the failed feed says what refused it."""
    sent: list[str] = []

    def handler(request):
        sent.append(request.url.path)
        if "sitemap" in request.url.path:
            return httpx.Response(200, content=sitemap(["6376", "8248", "8299"]))
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
        ("club_request_failed", 412, "2 of 2 club requests: PerimeterX challenge")
    ]


# -------------------------------------------------------------------- prices


def test_prices_come_from_the_fuel_centre_pages_storefuelprices():
    result = sams.SamsSource().parse(
        [
            sitemap_response(["6376"]),
            club_response("6376", {"UNLEAD": 3.799, "PREMIUM": 4.499, "DIESEL": 5.899}),
        ],
        _ctx(),
    )
    station = next(s for s in result.stations if s.source_station_id == "6376")
    prices = {p.grade_raw: float(p.price_raw) for p in station.prices}
    assert prices == {"UNLEAD": 3.799, "PREMIUM": 4.499, "DIESEL": 5.899}


def test_a_grade_priced_below_the_clubs_own_regular_is_dropped():
    """Mid-grade cannot be cheaper than regular. A grade below the club's own
    UNLEAD is dropped and named in a warning, so a real price that dips below
    regular still shows."""
    result = sams.SamsSource().parse(
        [
            sitemap_response(["6376"]),
            club_response("6376", {"UNLEAD": 3.799, "MIDGRAD": 2.979, "PREMIUM": 4.499}),
        ],
        _ctx(),
    )
    station = next(s for s in result.stations if s.source_station_id == "6376")
    assert {p.grade_raw for p in station.prices} == {"UNLEAD", "PREMIUM"}
    assert [(w.code, w.detail) for w in result.warnings] == [
        ("below_regular", "6376:MIDGRAD=2.979")
    ]


def test_with_no_regular_to_compare_against_every_grade_stays():
    result = sams.SamsSource().parse(
        [sitemap_response(["6376"]), club_response("6376", {"PREMIUM": 4.499, "MIDGRAD": 2.979})],
        _ctx(),
    )
    station = next(s for s in result.stations if s.source_station_id == "6376")
    assert {p.grade_raw for p in station.prices} == {"PREMIUM", "MIDGRAD"}


def test_a_fuel_club_answering_without_a_price_is_no_current_price():
    """A fuel club that answers 200 but carries no price is `no_current_price` --
    which says something -- not `not_reached`, which says nothing."""
    result = sams.SamsSource().parse(
        [sitemap_response(["6376"]), club_response("6376", {})], _ctx()
    )
    assert result.stations == []
    assert [(w.code, w.detail) for w in result.warnings] == [("no_current_price", "6376")]


def test_an_unreached_club_is_only_warned_when_the_sweep_stopped():
    """A club with no response, when the sweep ran to the end, is simply absent
    (a non-fuel club never enters the roster's fuel set); `not_reached` is
    reserved for a sweep that stopped early, so publish does not read a complete
    sweep's silence as a closure."""
    result = sams.SamsSource().parse(
        [sitemap_response(["6376", "8248"]), club_response("6376", {"UNLEAD": 3.5})], _ctx()
    )
    # 8248 had no response but the sweep was not cut short, so no not_reached.
    assert [s.source_station_id for s in result.stations] == ["6376"]
    assert [w.code for w in result.warnings] == []


def test_an_invalid_store_record_is_skipped():
    """A 200 whose fuel record is INVALID STORE (a fuel-centre URL that resolved
    to the club home without a redirect) is non-fuel, not a failure."""
    result = sams.SamsSource().parse(
        [sitemap_response(["6376"]), club_response("6376", {}, record_id="INVALID STORE")],
        _ctx(),
    )
    assert result.stations == []
    assert result.warnings == []
    assert result.errors == []


@pytest.mark.parametrize("status", [400, 403, 429, 500])
def test_a_failed_club_response_never_becomes_a_station(status):
    result = sams.SamsSource().parse(
        [
            sitemap_response(["6376"]),
            response(sams.KEY_CLUB.format(club_id="6376"), b"", status=status),
        ],
        _ctx(),
    )
    assert result.stations == []
    assert [(e.code, e.http_status) for e in result.errors] == [("club_request_failed", status)]


def test_a_page_with_no_next_data_is_a_failure_not_a_crash():
    result = sams.SamsSource().parse(
        [
            sitemap_response(["6376"]),
            response(sams.KEY_CLUB.format(club_id="6376"), b"<html>x</html>"),
        ],
        _ctx(),
    )
    assert result.stations == []
    assert [e.code for e in result.errors] == ["club_request_failed"]


# ------------------------------------------------------------------ station


def test_a_station_carries_what_the_dashboard_needs():
    result = sams.SamsSource().parse(
        [sitemap_response(["6376"]), club_response("6376", {"UNLEAD": 3.799})], _ctx()
    )
    station = next(s for s in result.stations if s.source_station_id == "6376")

    assert station.name == "Test Sam's Club"
    assert station.region == "TX"
    assert station.city == "Dallas"
    assert station.postcode == "75244"
    assert station.lat == pytest.approx(32.921881)
    assert station.lon == pytest.approx(-96.843062)
    # The fuel-centre page resolves from the id alone, with no city slug.
    assert station.alt_id == "6376"
    # The page carries a real IANA zone, so no abbreviation disambiguation.
    assert station.timezone == "America/Chicago"
    assert station.has_hours is True


def test_the_result_is_branded_so_keys_cannot_collide():
    result = sams.SamsSource().parse(
        [sitemap_response(["6376"]), club_response("6376", {"UNLEAD": 3.5})], _ctx()
    )
    assert (result.country, result.brand, result.source) == ("US", "SAMS", "sams-fuel-center")


# -------------------------------------------------------------- extraction


def test_the_iana_timezone_is_read_straight_from_the_page():
    props = json.loads(
        fuel_page("6609", {"UNLEAD": 5.5}, tz="America/Los_Angeles")
        .split(b'application/json">')[1]
        .split(b"</script>")[0]
    )["props"]["pageProps"]
    assert sams.iana_timezone(props) == "America/Los_Angeles"


def test_a_missing_timezone_is_none_rather_than_a_guess():
    props = json.loads(
        fuel_page("6376", {"UNLEAD": 3.5}, tz=None)
        .split(b'application/json">')[1]
        .split(b"</script>")[0]
    )["props"]["pageProps"]
    assert sams.iana_timezone(props) is None
