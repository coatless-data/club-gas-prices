"""The shared HTTP client. No test here touches the network."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import httpx
import pytest

from club_gas import http
from club_gas.config import HttpConfig, TimeoutProfile
from club_gas.http import BudgetExceeded, Client

FIXTURES = Path(__file__).resolve().parent / "fixtures"
US_BATCH = FIXTURES / "us" / "gasprices_batch.json"
AKAMAI = FIXTURES / "blocks" / "akamai_access_denied.html"
PERIMETERX = FIXTURES / "blocks" / "perimeterx_412.json"

# Real policy, minus the waiting, so the suite stays fast.
FAST = HttpConfig(backoff_seconds=(0.0, 0.0), min_interval_seconds=0.0)

PRICE_URL = "https://www.costco.com/AjaxGetGasPricesService?warehouseid=1364"
ECOM_URL = "https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json"
FX_URL = "https://api.frankfurter.dev/v2/rates?base=USD"
SAMS_URL = "https://www.samsclub.com/orchestra/home/graphql/HyperLocalPagesTempo/x"


def json_ok(request: httpx.Request) -> httpx.Response:
    """Costco's legacy endpoints serve JSON as text/html. That is normal."""
    return httpx.Response(
        200,
        content=US_BATCH.read_bytes(),
        headers={"Content-Type": "text/html;charset=UTF-8"},
    )


def test_costco_hosts_get_the_browser_ua_and_x_project(monkeypatch):
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    seen: dict[str, dict[str, str]] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen[str(request.url)] = dict(request.headers)
        return json_ok(request)

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        client.request("US-COSTCO/01-gasprices", PRICE_URL)
        client.request("fx/01-frankfurter", FX_URL)

    costco = seen[PRICE_URL]
    assert costco["user-agent"] == FAST.costco_user_agent
    assert costco["user-agent"].endswith("Safari/537.36")
    # Appending a project token to the UA made this host reset the connection.
    assert "club-gas-prices" not in costco["user-agent"]
    assert costco["x-project"] == FAST.x_project
    assert costco["accept"] == "application/json"
    assert costco["accept-encoding"] == "gzip"
    assert "from" not in costco

    other = seen[FX_URL]
    assert other["user-agent"] == FAST.project_user_agent
    assert "x-project" not in other
    assert other["accept"] == "application/json"


def test_caller_headers_are_passed_through(monkeypatch):
    monkeypatch.delenv("CONTACT_EMAIL", raising=False)
    seen: dict[str, str] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        seen.update(dict(request.headers))
        return json_ok(request)

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        client.request(
            "shared/ecom-api",
            ECOM_URL,
            headers={"client-identifier": "7c71124c-7bf1-44db-bc9d-498584cd66e5"},
        )

    assert seen["client-identifier"] == "7c71124c-7bf1-44db-bc9d-498584cd66e5"
    # ecom-api.costco.com is a Costco host by suffix.
    assert seen["user-agent"] == FAST.costco_user_agent


def test_from_header_follows_contact_email(monkeypatch):
    seen: list[dict[str, str]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(dict(request.headers))
        return json_ok(request)

    monkeypatch.setenv("CONTACT_EMAIL", "ops@example.org")
    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        client.request("US-COSTCO/01", PRICE_URL)
    assert seen[-1]["from"] == "ops@example.org"

    monkeypatch.setenv("CONTACT_EMAIL", "")
    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        client.request("US-COSTCO/01", PRICE_URL)
    assert "from" not in seen[-1]


def test_retries_5xx_429_and_timeouts_but_not_other_4xx():
    calls: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        calls.append(host)
        if host == "five.test":
            return httpx.Response(503, content=b"unavailable")
        if host == "slow.test":
            raise httpx.ReadTimeout("read timeout", request=request)
        if host == "rate.test":
            return httpx.Response(429, content=b"slow down")
        return httpx.Response(404, content=b"missing")

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        five = client.request("k", "https://five.test/x")
        slow = client.request("k", "https://slow.test/x")
        rate = client.request("k", "https://rate.test/x")
        gone = client.request("k", "https://gone.test/x")

    assert five.status == 503
    assert calls.count("five.test") == 3
    assert slow.status is None
    assert slow.error is not None and "timeout" in slow.error
    assert calls.count("slow.test") == 3
    assert rate.status == 429
    assert calls.count("rate.test") == 3
    assert gone.status == 404
    assert calls.count("gone.test") == 1


@pytest.mark.real_sleep
def test_backoff_gaps_stay_within_the_jittered_bounds():
    """FAST zeroes backoff so the suite stays quick, but the real schedule
    and jitter formula (never exercised by FAST) must still hold: each gap
    is 50%-150% of its nominal step, in schedule order.

    The two steps are kept well apart (3x) so that an off-by-one in the
    schedule index reliably lands outside the other step's bounds instead
    of overlapping it by chance.
    """
    calls: list[float] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(time.monotonic())
        if len(calls) < 3:
            return httpx.Response(503, content=b"unavailable")
        return httpx.Response(200, content=b"[]")

    schedule = (0.08, 0.24)
    cfg = HttpConfig(backoff_seconds=schedule, backoff_jitter=0.5, min_interval_seconds=0.0)
    with Client(cfg, transport=httpx.MockTransport(handler)) as client:
        result = client.request("k", "https://backoff.test/x")

    assert result.status == 200
    assert len(calls) == 3
    gap_after_first_failure = calls[1] - calls[0]
    gap_after_second_failure = calls[2] - calls[1]
    # A wall-clock gap is sleep plus whatever else the machine did in between,
    # so overhead can only ever make it LONGER. The lower bound is therefore the
    # real assertion here and stays tight; the upper bound is generous, because
    # a busy runner has been seen to add over 70ms to an 80ms sleep, and a gate
    # that goes red when the machine is loaded is a gate people learn to ignore.
    # `test_backoff_sleeps_are_the_jittered_schedule` is what actually pins the
    # formula -- it reads the durations instead of timing them.
    for gap, nominal in (
        (gap_after_first_failure, schedule[0]),
        (gap_after_second_failure, schedule[1]),
    ):
        assert gap >= nominal * 0.5 - 0.005
        assert gap <= nominal * 1.5 + 0.5


@pytest.mark.real_sleep
def test_backoff_sleeps_are_the_jittered_schedule(monkeypatch):
    """The same property as above, read rather than timed.

    Recording what `_sleep_backoff` asks for removes the machine from the
    measurement entirely, so this holds under any load and can assert the bounds
    exactly instead of with an allowance.
    """
    slept: list[float] = []
    monkeypatch.setattr(http.time, "sleep", slept.append)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"unavailable")

    schedule = (0.08, 0.24)
    cfg = HttpConfig(backoff_seconds=schedule, backoff_jitter=0.5, min_interval_seconds=0.0)
    with Client(cfg, transport=httpx.MockTransport(handler)) as client:
        client.request("k", "https://backoff.test/x")

    # max_attempts is 3, so two backoffs, taken in schedule order.
    assert len(slept) == 2
    for requested, nominal in zip(slept, schedule, strict=True):
        assert nominal * 0.5 <= requested <= nominal * 1.5


@pytest.mark.real_sleep
def test_backoff_jitter_actually_varies_the_sleep(monkeypatch):
    """A jitter that silently collapsed to zero would satisfy the bounds above.

    Every retry across the whole fleet would then fire on the same tick, which
    is the one thing the jitter exists to prevent.
    """
    slept: list[float] = []
    monkeypatch.setattr(http.time, "sleep", slept.append)

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"unavailable")

    cfg = HttpConfig(backoff_seconds=(0.08,), backoff_jitter=0.5, min_interval_seconds=0.0)
    for _ in range(20):
        with Client(cfg, transport=httpx.MockTransport(handler)) as client:
            client.request("k", "https://backoff.test/x")

    assert len(slept) == 40
    assert len(set(slept)) > 1, "every backoff slept exactly the nominal step"


def test_connection_errors_are_retried_and_then_reported():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        raise httpx.ConnectError("connection reset", request=request)

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        result = client.request("US-COSTCO/01", PRICE_URL)

    assert len(calls) == 3
    assert result.status is None
    assert "ConnectError" in result.error
    assert result.body == b""
    # A reset connection is not a block signal; only 403/429, cpr_chlge,
    # timeouts and HTML-where-JSON-was-expected are.
    assert client.signals.get("www.costco.com", 0) == 0


def test_headers_that_arrive_after_30s_need_the_bulk_profile():
    """read covers the wait for response headers, so the profile decides."""

    def handler(request: httpx.Request) -> httpx.Response:
        read = request.extensions["timeout"]["read"]
        if read < 30.0:
            raise httpx.ReadTimeout("headers after 30 s", request=request)
        return httpx.Response(200, content=b"[]")

    url = "https://www.costco.ca/AjaxWarehouseBrowseLookupView?countryCode=US"
    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        bulk = client.request("US-COSTCO/00-lookup", url, profile="bulk")
        default = client.request("US-COSTCO/00-lookup", url)

    assert bulk.status == 200
    assert default.status is None
    assert "timeout" in default.error


def test_total_deadline_aborts_a_trickled_body():
    cfg = HttpConfig(
        backoff_seconds=(0.0, 0.0),
        min_interval_seconds=0.0,
        profiles={"default": TimeoutProfile(connect=1.0, read=5.0, total=0.3)},
    )

    def trickle():
        for _ in range(40):
            time.sleep(0.02)
            yield b"x"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=trickle())

    with Client(cfg, transport=httpx.MockTransport(handler)) as client:
        result = client.request("k", "https://trickle.test/x")

    assert result.status is None
    assert result.error == "total_timeout"


def test_budget_refuses_before_a_request_is_sent():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return json_ok(request)

    with (
        Client(FAST, transport=httpx.MockTransport(handler)) as client,
        pytest.raises(BudgetExceeded, match="ecom-api"),
        client.budget("ecom-api", 0.0),
    ):
        client.request("shared/ecom-api", ECOM_URL)

    assert calls == []


def test_budget_aborts_an_in_flight_request():
    def trickle():
        for _ in range(40):
            time.sleep(0.02)
            yield b"x"

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=trickle())

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        started = time.monotonic()
        with pytest.raises(BudgetExceeded, match="fx"), client.budget("fx", 0.15):
            client.request("fx/01", "https://fx.test/rates")
        assert time.monotonic() - started < 0.6


def test_the_wait_for_response_headers_is_bounded_by_the_budget():
    """A host that accepts the connection and never answers must not be
    bounded only by the profile's read timeout (up to 150s on bulk): the
    in-flight body-chunk checks never run until the first byte arrives, so
    connect/read must also be clamped to the active budget up front.

    One attempt only, so this isolates the pre-body clamp from the retry
    loop tested elsewhere.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        read = request.extensions["timeout"]["read"]
        if read > 1.0:
            # Not clamped: as if the stalled host eventually answered.
            return httpx.Response(200, content=b"[]")
        raise httpx.ReadTimeout("clamped short: bail out before headers", request=request)

    cfg = HttpConfig(
        backoff_seconds=(0.0, 0.0),
        min_interval_seconds=0.0,
        max_attempts=1,
        profiles={"bulk": TimeoutProfile(connect=10.0, read=150.0, total=150.0)},
    )
    with Client(cfg, transport=httpx.MockTransport(handler)) as client:
        started = time.monotonic()
        with client.budget("fx", 0.1):
            result = client.request("fx/01", "https://fx.test/rates", profile="bulk")
        elapsed = time.monotonic() - started

    # Nowhere near the profile's 150s read timeout.
    assert elapsed < 1.0
    assert result.status is None
    assert result.error is not None and "timeout" in result.error


def test_budget_ends_when_its_block_ends():
    with Client(FAST, transport=httpx.MockTransport(json_ok)) as client:
        with client.budget("fx", 0.0), pytest.raises(BudgetExceeded):
            client.request("fx/01", "https://fx.test/rates")
        # Outside the block the client works normally again.
        assert client.request("US-COSTCO/01", PRICE_URL).status == 200


def test_json_served_as_text_html_is_not_a_block_signal():
    with Client(FAST, transport=httpx.MockTransport(json_ok)) as client:
        result = client.request("US-COSTCO/01-gasprices", PRICE_URL)
        assert result.status == 200
        assert result.body.startswith(b'{"1090"')
        assert client.signals.get("www.costco.com", 0) == 0
        assert client.abandoned(PRICE_URL) is False


def test_a_json_array_after_leading_crlf_is_not_a_block_signal():
    """The costco.ca lookup body starts with \\r\\n before its JSON array."""

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=b'\r\n[false,{"stlocID":530}]',
            headers={"Content-Type": "text/html;charset=UTF-8"},
        )

    url = "https://www.costco.ca/AjaxWarehouseBrowseLookupView?countryCode=CA"
    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        assert client.request("CA-COSTCO/01-lookup", url).status == 200
        assert client.signals.get("www.costco.ca", 0) == 0


def test_two_block_signals_abandon_the_host_for_every_thread():
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return httpx.Response(
            403,
            content=AKAMAI.read_bytes(),
            headers={"Content-Type": "text/html"},
        )

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        first = client.request("US-COSTCO/01", PRICE_URL)
        assert first.status == 403
        assert client.signals["www.costco.com"] == 1
        assert client.abandoned(PRICE_URL) is False

        client.request("US-COSTCO/02", PRICE_URL)
        assert client.signals["www.costco.com"] == 2
        assert client.abandoned(PRICE_URL) is True

        result: dict[str, object] = {}

        def from_another_thread() -> None:
            result["response"] = client.request("US-COSTCO/03", PRICE_URL)

        thread = threading.Thread(target=from_another_thread)
        thread.start()
        thread.join()

    aborted = result["response"]
    assert aborted.status is None
    assert aborted.error == "host_abandoned"
    assert aborted.body == b""
    # 403 is not retried, and the abandoned request never reached the transport.
    assert len(calls) == 2
    # Another host is unaffected.
    assert client.abandoned("https://www.costco.ca/x") is False


def test_a_request_gives_at_most_one_block_signal():
    def handler(request: httpx.Request) -> httpx.Response:
        # 429 and cpr_chlge and a body starting with "<": still one signal.
        return httpx.Response(429, content=b'<x>{"cpr_chlge":"true"}</x>')

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        client.request("US-COSTCO/01", PRICE_URL)
        assert client.signals["www.costco.com"] == 1
        assert client.abandoned(PRICE_URL) is False


def test_cpr_chlge_in_a_200_body_is_a_block_signal():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b'{"cpr_chlge":"true"}')

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        client.request("US-COSTCO/01", PRICE_URL)
        assert client.signals["www.costco.com"] == 1


def test_a_403_or_a_429_is_a_block_signal_on_the_status_alone():
    """The status rule stands on its own, with nothing else to lean on.

    Every other test that reaches it trips a second rule as well -- the Akamai
    403 body is HTML, the 429 one carries `cpr_chlge` -- so deleting the status
    test would leave the suite green while a bare 403 stopped counting towards
    abandoning the host. These two bodies are ordinary JSON: not HTML, no
    challenge marker, no timeout, so only the status can make them signals.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        status = 403 if request.url.host == "forbidden.test" else 429
        return httpx.Response(status, content=b'{"message":"blocked"}')

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        forbidden = client.request("US-COSTCO/01", "https://forbidden.test/x")
        limited = client.request("US-COSTCO/02", "https://limited.test/x")

        assert forbidden.status == 403
        assert limited.status == 429
        assert client.signals["forbidden.test"] == 1
        assert client.signals["limited.test"] == 1
        assert [entry["block_signal"] for entry in client.log] == [True, True]


@pytest.mark.parametrize(
    "body",
    [
        PERIMETERX.read_bytes(),
        b'{"appId":"PXsLC3j22K","jsClientSrc":"/px/PXsLC3j22K/init.js"}',
        b'{"redirectUrl":"/are-you-human?url=Lw==&uuid=x"}',
    ],
    ids=["recorded", "app-id", "redirect"],
)
def test_a_perimeterx_412_is_a_block_signal(body):
    """Sam's Club refuses with 412 and a small JSON body, not 403 and HTML.

    The recorded body is the roster request's answer on a GitHub-hosted runner
    on 2026-09-17. Neither the status rule nor the HTML rule sees it, so a
    refused sweep kept sending its ~531 requests into the challenge.
    """

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            412, content=body, headers={"Content-Type": "application/json; charset=UTF-8"}
        )

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        client.request("US-SAMS/02-fuel-0001", SAMS_URL)
        assert client.signals["www.samsclub.com"] == 1
        assert client.abandoned(SAMS_URL) is False

        client.request("US-SAMS/02-fuel-0002", SAMS_URL)
        assert client.abandoned(SAMS_URL) is True


@pytest.mark.parametrize(
    "body", [b"", b'{"message":"Precondition Failed"}', b'{"appId":"not-perimeterx"}']
)
def test_a_412_without_the_perimeterx_body_is_not_a_block_signal(body):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(412, content=body)

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        client.request("US-SAMS/02-fuel-0001", SAMS_URL)
        assert client.signals.get("www.samsclub.com", 0) == 0


def test_html_is_not_a_signal_when_json_was_not_expected():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, content=b"<html>a page</html>")

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        client.request("US-COSTCO/01", PRICE_URL, expect_json=False)
        assert client.signals.get("www.costco.com", 0) == 0


@pytest.mark.real_sleep
def test_pacing_is_per_host_and_shared_across_threads():
    cfg = HttpConfig(backoff_seconds=(0.0, 0.0), min_interval_seconds=0.2)

    with Client(cfg, transport=httpx.MockTransport(json_ok)) as client:
        started = time.monotonic()
        threads = [
            threading.Thread(target=lambda: client.request("k", "https://paced.test/x"))
            for _ in range(2)
        ]
        for thread in threads:
            thread.start()
        client.request("k", "https://paced.test/x")
        for thread in threads:
            thread.join()
        assert time.monotonic() - started >= 0.4

        started = time.monotonic()
        client.request("k", "https://one.test/x")
        client.request("k", "https://two.test/x")
        assert time.monotonic() - started < 0.2


def test_raw_response_records_the_key_and_the_interesting_headers():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            content=US_BATCH.read_bytes(),
            headers={
                "Content-Type": "text/html;charset=UTF-8",
                "Date": "Tue, 15 Sep 2026 19:10:16 GMT",
                "Server-Timing": 'ak_p; desc="1789499415803_388408362";dur=1',
                "Set-Cookie": "bm_sz=0F6251C6; Domain=.costco.com; Path=/",
            },
        )

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        result = client.request("US-COSTCO/03-gasprices", PRICE_URL)

    assert result.key == "US-COSTCO/03-gasprices"
    assert result.url == PRICE_URL
    assert result.status == 200
    assert result.error is None
    assert result.headers["content-type"] == "text/html;charset=UTF-8"
    assert result.headers["date"] == "Tue, 15 Sep 2026 19:10:16 GMT"
    assert result.headers["server-timing"].startswith("ak_p;")
    # Cookies are not recorded: capture bundles are published publicly.
    assert "set-cookie" not in result.headers
    assert result.received_at_utc.tzinfo is not None
    assert result.elapsed_ms >= 0
    assert result.body == US_BATCH.read_bytes()


def test_the_client_logs_one_entry_per_logical_request():
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(503, content=b"unavailable")

    with Client(FAST, transport=httpx.MockTransport(handler)) as client:
        client.request("US-COSTCO/01", PRICE_URL)

    assert len(client.log) == 1
    entry = client.log[0]
    assert entry["key"] == "US-COSTCO/01"
    assert entry["host"] == "www.costco.com"
    assert entry["attempts"] == 3
    assert entry["status"] == 503
    assert entry["block_signal"] is False


def test_a_budget_with_no_explicit_length_comes_from_the_config_file():
    """`config/http.toml`'s [budgets] table is the only place the numbers live.

    The capture, fx and ecom budgets are named exactly as they are in the file;
    a country budget is named `country-<CC>` so a BudgetExceeded says which
    country ran out, and takes its length from the shared `country` entry.
    """
    calls: list[int] = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(1)
        return json_ok(request)

    cfg = HttpConfig(
        backoff_seconds=(0.0, 0.0),
        min_interval_seconds=0.0,
        budgets={"fx": 90.0, "ecom-api": 0.0, "country": 0.0, "capture": 720.0},
    )
    with Client(cfg, transport=httpx.MockTransport(handler)) as client:
        assert client.budget_seconds("ecom-api") == 0.0
        with pytest.raises(BudgetExceeded, match="ecom-api"), client.budget("ecom-api"):
            client.request("shared/ecom-api", ECOM_URL)
        with (
            pytest.raises(BudgetExceeded, match="country-AU"),
            client.budget("country-AU", key="country"),
        ):
            client.request("AU-COSTCO/01-stores", PRICE_URL)
        # An entry with room left does not abort anything.
        with client.budget("fx"):
            assert client.request("fx/01", FX_URL).status == 200

    assert calls == [1]


def test_a_budget_name_the_config_file_does_not_carry_falls_back_to_the_default():
    with Client(HttpConfig(budgets={}), transport=httpx.MockTransport(json_ok)) as client:
        assert client.budget_seconds("capture") == 720.0
