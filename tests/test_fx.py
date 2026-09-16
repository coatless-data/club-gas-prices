"""Exchange rates: primary, fallbacks, carry-forward, identity and the budget."""

from __future__ import annotations

import contextlib
from datetime import UTC, date, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from costco_gas.config import load_config
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
