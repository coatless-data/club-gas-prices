"""Exchange rates: primary, fallbacks, carry-forward, identity and the budget."""

from __future__ import annotations

import contextlib
import json
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from costco_gas.config import HttpConfig, load_config
from costco_gas.fx import fetch_rates
from costco_gas.http import Client
from costco_gas.sources.base import CaptureContext

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures" / "fx"
CAPTURE_DATE = date(2026, 9, 15)


def read_fixture(name: str) -> bytes:
    return (FIXTURES / name).read_bytes()


@pytest.fixture
def make_client():
    """Build a real Client whose transport is a MockTransport (no network)."""
    cfg = load_config(REPO_ROOT)

    def _make(handler):
        return Client(cfg.http, transport=httpx.MockTransport(handler))

    return _make


def make_ctx(previous_fx: pl.DataFrame | None = None) -> CaptureContext:
    # fetch_config / interp_config are unused by fx.py and CaptureContext does no
    # runtime type checking, so None keeps this test free of config fixtures.
    return CaptureContext(
        capture_id="2026-09-15T1817Z",
        capture_date=CAPTURE_DATE,
        fetch_config=None,
        interp_config=None,
        previous_fx=pl.DataFrame() if previous_fx is None else previous_fx,
    )


def routing_handler(routes: dict[str, httpx.Response], calls: list[str]):
    """Answer by URL prefix; anything unrouted is a 404."""

    def handler(request: httpx.Request) -> httpx.Response:
        url = str(request.url)
        calls.append(url)
        for prefix, response in routes.items():
            if url.startswith(prefix):
                return httpx.Response(
                    response.status_code, content=response.content, headers=response.headers
                )
        return httpx.Response(404, content=b"no route")

    return handler


def test_frankfurter_primary_gives_status_ok(make_client):
    calls: list[str] = []
    handler = routing_handler(
        {
            "https://api.frankfurter.dev/": httpx.Response(
                200,
                content=read_fixture("frankfurter_v2.json"),
                headers={"Content-Type": "application/json"},
            )
        },
        calls,
    )

    fx = fetch_rates(make_client(handler), make_ctx())

    assert fx.status == "ok"
    assert [row.currency for row in fx.rows] == ["CAD", "MXN", "GBP", "AUD", "JPY", "TWD"]
    assert len(calls) == 1
    assert calls[0].startswith("https://api.frankfurter.dev/v2/rates?base=USD&quotes=")
    assert "CAD,MXN,GBP,AUD,JPY,TWD" in calls[0]

    jpy = fx.for_currency("JPY")
    assert jpy.units_per_usd == 154.24
    assert jpy.fx_usd_per_unit == pytest.approx(1 / 154.24, rel=1e-12)
    assert jpy.fx_source == "frankfurter-v2"
    # Frankfurter publishes the latest *business* day, so the rate date is taken
    # from the response and is not assumed to equal the capture date.
    assert jpy.fx_rate_date == date(2026, 9, 14)
    assert jpy.fx_fetched_at_utc is not None
    assert jpy.fx_fetched_at_utc.tzinfo is not None

    gbp = fx.for_currency("GBP")
    assert gbp.units_per_usd == 0.73996
    assert gbp.fx_usd_per_unit == pytest.approx(1.351424401, rel=1e-6)


def test_usd_is_always_the_identity_row(make_client):
    calls: list[str] = []
    handler = routing_handler(
        {
            "https://api.frankfurter.dev/": httpx.Response(
                200, content=read_fixture("frankfurter_v2.json")
            )
        },
        calls,
    )

    fx = fetch_rates(make_client(handler), make_ctx())

    usd = fx.for_currency("USD")
    assert usd.units_per_usd == 1.0
    assert usd.fx_usd_per_unit == 1.0
    assert usd.fx_source == "identity"
    assert usd.fx_rate_date == CAPTURE_DATE
    # normalize() fills this in with the country's captured_at_utc.
    assert usd.fx_fetched_at_utc is None
    # The identity row is never stored in fx.json / fx.csv, which hold non-USD only.
    assert "USD" not in [row.currency for row in fx.rows]
    assert fx.for_currency("usd").fx_source == "identity"


def test_fallback_uses_the_capture_date_then_the_previous_day(make_client):
    calls: list[str] = []
    handler = routing_handler(
        {
            # The real 403 Cloudflare "Error 1010" body captured on 2026-09-15,
            # which is what Frankfurter returns to a UA it does not accept.
            "https://api.frankfurter.dev/": httpx.Response(
                403, content=read_fixture("frankfurter_blocked.json")
            ),
            "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@2026-09-15/": (
                httpx.Response(404, content=b"Couldn't find the requested version")
            ),
            "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@2026-09-14/": (
                httpx.Response(200, content=read_fixture("fawaz_2026-09-14.json"))
            ),
        },
        calls,
    )

    fx = fetch_rates(make_client(handler), make_ctx())

    assert fx.status == "fallback"
    assert [row.currency for row in fx.rows] == ["CAD", "MXN", "GBP", "AUD", "JPY", "TWD"]
    twd = fx.for_currency("TWD")
    assert twd.units_per_usd == 31.7167273
    assert twd.fx_usd_per_unit == pytest.approx(1 / 31.7167273, rel=1e-12)
    assert twd.fx_source == "fawazahmed0-currency-api"
    assert twd.fx_rate_date == date(2026, 9, 14)

    assert len(calls) == 3
    assert calls[0].startswith("https://api.frankfurter.dev/")
    assert "@2026-09-15/v1/currencies/usd.json" in calls[1]
    assert "@2026-09-14/v1/currencies/usd.json" in calls[2]
    # @latest was a day stale on 2026-09-15 and must never be requested.
    assert not any("@latest" in url for url in calls)


def test_fallback_stops_at_the_capture_date_when_it_resolves(make_client):
    calls: list[str] = []
    handler = routing_handler(
        {
            "https://api.frankfurter.dev/": httpx.Response(
                403, content=read_fixture("frankfurter_blocked.json")
            ),
            "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@2026-09-15/": (
                httpx.Response(200, content=read_fixture("fawaz_2026-09-15.json"))
            ),
        },
        calls,
    )

    fx = fetch_rates(make_client(handler), make_ctx())

    assert fx.status == "fallback"
    assert fx.for_currency("CAD").units_per_usd == 1.39146472
    assert fx.for_currency("CAD").fx_rate_date == date(2026, 9, 15)
    assert len(calls) == 2
    assert not any("@2026-09-14" in url for url in calls)


def all_network_fails_handler(calls: list[str]):
    return routing_handler(
        {
            "https://api.frankfurter.dev/": httpx.Response(
                403, content=read_fixture("frankfurter_blocked.json")
            ),
            "https://cdn.jsdelivr.net/": httpx.Response(404, content=b"not found"),
        },
        calls,
    )


def previous_fx_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "capture_id": [
                "2026-09-13T1817Z",
                "2026-09-14T1817Z",
                "2026-09-15T0017Z",
                "2026-09-14T1817Z",
                "2026-09-14T1817Z",
            ],
            "currency": ["CAD", "CAD", "CAD", "JPY", "TWD"],
            "units_per_usd": [1.3800, 1.3871, 1.3900, 154.06565573, 31.7167273],
            "fx_usd_per_unit": [
                1 / 1.3800,
                1 / 1.3871,
                1 / 1.3900,
                1 / 154.06565573,
                1 / 31.7167273,
            ],
            "fx_rate_date": [
                date(2026, 9, 11),
                date(2026, 9, 12),
                date(2026, 9, 14),
                date(2026, 9, 1),
                date(2026, 9, 8),
            ],
            "fx_source": [
                "frankfurter-v2",
                "frankfurter-v2",
                "carried-forward:frankfurter-v2",
                "fawazahmed0-currency-api",
                "fawazahmed0-currency-api",
            ],
            "fx_fetched_at_utc": [
                datetime(2026, 9, 13, 18, 17, 41, tzinfo=UTC),
                datetime(2026, 9, 14, 18, 17, 42, tzinfo=UTC),
                datetime(2026, 9, 15, 0, 17, 43, tzinfo=UTC),
                datetime(2026, 9, 14, 18, 17, 42, tzinfo=UTC),
                datetime(2026, 9, 14, 18, 17, 42, tzinfo=UTC),
            ],
        }
    )


def test_carry_forward_copies_the_newest_uncarried_row_within_seven_days(make_client):
    calls: list[str] = []
    fx = fetch_rates(make_client(all_network_fails_handler(calls)), make_ctx(previous_fx_frame()))

    assert fx.status == "carried-forward"
    # JPY's stored rate date is 14 days before the capture, so it is not carried.
    assert [row.currency for row in fx.rows] == ["CAD", "TWD"]
    assert fx.for_currency("JPY") is None

    cad = fx.for_currency("CAD")
    # The newest CAD row is itself carried-forward and must be ignored, otherwise
    # a stale rate would be carried indefinitely.
    assert cad.fx_source == "carried-forward:frankfurter-v2"
    assert cad.units_per_usd == 1.3871
    assert cad.fx_usd_per_unit == pytest.approx(1 / 1.3871, rel=1e-12)
    assert cad.fx_rate_date == date(2026, 9, 12)
    assert cad.fx_fetched_at_utc == datetime(2026, 9, 14, 18, 17, 42, tzinfo=UTC)

    # Exactly 7 days old: still inside the limit.
    twd = fx.for_currency("TWD")
    assert twd.fx_source == "carried-forward:fawazahmed0-currency-api"
    assert twd.fx_rate_date == date(2026, 9, 8)

    assert fx.for_currency("USD").fx_source == "identity"


def test_carry_forward_accepts_string_dates(make_client):
    frame = pl.DataFrame(
        {
            "capture_id": ["2026-09-14T1817Z"],
            "currency": ["GBP"],
            "units_per_usd": [0.73996],
            "fx_usd_per_unit": [1 / 0.73996],
            "fx_rate_date": ["2026-09-14"],
            "fx_source": ["frankfurter-v2"],
            "fx_fetched_at_utc": ["2026-09-14T18:17:42Z"],
        }
    )
    calls: list[str] = []
    fx = fetch_rates(make_client(all_network_fails_handler(calls)), make_ctx(frame))

    gbp = fx.for_currency("GBP")
    assert gbp.fx_rate_date == date(2026, 9, 14)
    assert gbp.fx_fetched_at_utc == datetime(2026, 9, 14, 18, 17, 42, tzinfo=UTC)


def test_every_source_failing_gives_status_failed_and_an_empty_fx_json(make_client):
    calls: list[str] = []
    fx = fetch_rates(make_client(all_network_fails_handler(calls)), make_ctx())

    assert fx.status == "failed"
    assert fx.rows == []
    # fx.json is a JSON array, so "empty" is an empty list, not an empty object.
    assert fx.to_json() == []
    # normalize() turns a missing row into null USD columns; the capture still wins.
    assert fx.for_currency("CAD") is None
    assert fx.for_currency("USD").fx_usd_per_unit == 1.0


class BudgetStubClient:
    """A stand-in for http.Client whose first request exhausts the FX budget.

    A real 90-second budget cannot be exercised in a unit test, so this records
    the budget fx.py opens and raises BudgetExceeded from inside it. It resolves
    an unspecified length from `cfg.budgets` exactly as the real client does, so
    the recorded number is the one config/http.toml carries.
    """

    def __init__(self, cfg: HttpConfig | None = None) -> None:
        self.cfg = cfg if cfg is not None else HttpConfig()
        self.budgets: list[tuple[str, float]] = []
        self.requests: list[str] = []

    def budget_seconds(self, name: str) -> float:
        return self.cfg.budget_seconds(name)

    def budget(self, name: str, seconds: float | None = None, *, key: str | None = None):
        if seconds is None:
            seconds = self.budget_seconds(key or name)
        self.budgets.append((name, seconds))
        return contextlib.nullcontext()

    def abandoned(self, url: str) -> bool:
        return False

    def request(self, key, url, *, profile="default", headers=None, expect_json=True):
        from costco_gas.http import BudgetExceeded

        self.requests.append(url)
        raise BudgetExceeded(f"fx budget exhausted before {key}")


def test_fx_runs_inside_a_ninety_second_budget_and_still_carries_forward():
    client = BudgetStubClient()

    fx = fetch_rates(client, make_ctx(previous_fx_frame()))

    assert client.budgets == [("fx", 90.0)]
    assert len(client.requests) == 1
    assert fx.status == "carried-forward"
    assert fx.for_currency("CAD").units_per_usd == 1.3871


def test_the_fx_budget_is_the_one_the_config_file_carries():
    """`fetch_rates` opened a hardcoded 90 seconds, so [budgets] was dead config."""
    client = BudgetStubClient(HttpConfig(budgets={"fx": 12.5}))

    fetch_rates(client, make_ctx())

    assert client.budgets == [("fx", 12.5)]


def test_an_exhausted_budget_without_previous_rates_fails_softly():
    client = BudgetStubClient()

    fx = fetch_rates(client, make_ctx())

    assert fx.status == "failed"
    assert fx.rows == []
    assert client.budgets == [("fx", 90.0)]


def test_fx_rates_json_round_trip():
    from costco_gas.fx import FxRates

    original = FxRates(
        status="ok",
        rows=[
            FxRates(status="ok", rows=[]).for_currency("USD"),
        ],
        capture_date=CAPTURE_DATE,
    )
    assert original.rows[0].currency == "USD"

    # fx.json is a JSON array of rate rows and stores no status, so from_json
    # takes the array plus the status the caller read from status.json.
    rows = [
        {
            "currency": "JPY",
            "units_per_usd": 154.24,
            "fx_rate_date": "2026-09-14",
            "fx_source": "frankfurter-v2",
            "fx_fetched_at_utc": "2026-09-15T18:17:42Z",
        }
    ]
    restored = FxRates.from_json(rows, status="ok")
    assert restored.status == "ok"
    assert restored.capture_date is None
    assert restored.for_currency("JPY").fx_usd_per_unit == pytest.approx(1 / 154.24)
    assert restored.for_currency("JPY").fx_rate_date == date(2026, 9, 14)
    assert restored.for_currency("JPY").fx_fetched_at_utc == datetime(
        2026, 9, 15, 18, 17, 42, tzinfo=UTC
    )
    assert restored.to_json() == rows

    # capture_date is optional and only dates the synthetic USD identity row.
    with_date = FxRates.from_json(rows, status="fallback", capture_date=CAPTURE_DATE)
    assert with_date.status == "fallback"
    assert with_date.for_currency("USD").fx_rate_date == CAPTURE_DATE


def test_a_currency_frankfurter_never_returns_is_reported_missing_and_warned(make_client):
    """Reproduces the code-review finding: 5 of 6 currencies resolve (JPY is
    absent from Frankfurter's response), the dated fallback is unavailable and
    there is no carry-forward data. `status` still names how the rows that DID
    resolve were obtained ("ok", since every row that exists came from
    Frankfurter) -- spec §4.5 defines status as describing how the resolved
    rates were obtained, not how many resolved. Completeness is a separate,
    explicit signal: `missing` names the gap and `warnings` makes it visible to
    whatever renders the capture's status.json, instead of JPY silently having
    no USD conversion.
    """
    partial_frankfurter = [
        {"date": "2026-09-14", "base": "USD", "quote": "CAD", "rate": 1.3871},
        {"date": "2026-09-14", "base": "USD", "quote": "MXN", "rate": 17.0477},
        {"date": "2026-09-14", "base": "USD", "quote": "GBP", "rate": 0.73996},
        {"date": "2026-09-14", "base": "USD", "quote": "AUD", "rate": 1.399},
        {"date": "2026-09-14", "base": "USD", "quote": "TWD", "rate": 31.709},
        # JPY deliberately absent, as in the reviewer's repro.
    ]
    calls: list[str] = []
    handler = routing_handler(
        {
            "https://api.frankfurter.dev/": httpx.Response(
                200,
                content=json.dumps(partial_frankfurter).encode(),
                headers={"Content-Type": "application/json"},
            ),
            # The dated fallback is unavailable for either date it would try.
            "https://cdn.jsdelivr.net/": httpx.Response(404, content=b"not found"),
        },
        calls,
    )

    # No carry-forward data either.
    fx = fetch_rates(make_client(handler), make_ctx())

    assert fx.status == "ok"
    assert [row.currency for row in fx.rows] == ["CAD", "MXN", "GBP", "AUD", "TWD"]
    # JPY did not silently disappear: it is named, not just absent.
    assert fx.missing == ("JPY",)
    assert fx.for_currency("JPY") is None

    assert len(fx.warnings) == 1
    warning = fx.warnings[0]
    assert warning.code == "fx_missing"
    assert warning.detail == "JPY"


def test_missing_and_warnings_cannot_disagree_regardless_of_how_fxrates_is_built():
    """`missing` and `warnings` are derived together in FxRates.__post_init__, so
    they cannot drift apart no matter which path builds the FxRates: this checks
    the invariant directly (not just the fetch_rates repro above), and would fail
    immediately if the derivation were deleted or `missing`/`warnings` stopped
    being populated.
    """
    from costco_gas.fx import CURRENCIES, FxRates

    # An old bundle's fx.json that only ever recorded CAD.
    rows = [
        {
            "currency": "CAD",
            "units_per_usd": 1.3871,
            "fx_rate_date": "2026-09-14",
            "fx_source": "frankfurter-v2",
            "fx_fetched_at_utc": "2026-09-14T18:17:42Z",
        }
    ]
    restored = FxRates.from_json(rows, status="ok")

    expected_missing = tuple(sorted(set(CURRENCIES) - {"CAD"}))
    assert restored.missing == expected_missing
    assert restored.missing != ()

    # The structural guarantee: every missing currency has exactly one matching
    # warning, and there are no extra warnings for currencies that did resolve.
    assert {w.detail for w in restored.warnings} == set(restored.missing)
    assert len(restored.warnings) == len(restored.missing)
    assert all(w.code == "fx_missing" for w in restored.warnings)

    # The complete case: no gaps, no warnings.
    complete_rows = [
        {
            "currency": currency,
            "units_per_usd": 1.5,
            "fx_rate_date": "2026-09-14",
            "fx_source": "frankfurter-v2",
            "fx_fetched_at_utc": "2026-09-14T18:17:42Z",
        }
        for currency in CURRENCIES
    ]
    complete = FxRates.from_json(complete_rows, status="ok")
    assert complete.missing == ()
    assert complete.warnings == ()
