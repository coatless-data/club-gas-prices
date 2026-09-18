"""Tests for the spec 6.5 status rules."""

from __future__ import annotations

from datetime import UTC, date, datetime, timedelta
from types import SimpleNamespace

import polars as pl
import pytest

from club_gas.checks import (
    all_failed,
    build_status,
    evaluate_feed,
    price_fingerprint,
)
from club_gas.fx import FxRates, FxRow
from club_gas.normalize import Drop, NormalizedCountry
from club_gas.schema import ROW_SCHEMA, STATION_SCHEMA
from club_gas.sources.base import (
    CaptureContext,
    Error,
    FetchResult,
    RawResponse,
    Warning,
)

CAPTURE_ID = "2026-09-15T1817Z"
CAPTURE_DATE = date(2026, 9, 15)
CAPTURED_AT = datetime(2026, 9, 15, 18, 17, 40, tzinfo=UTC)
NOW = datetime(2026, 9, 15, 18, 19, 2, tzinfo=UTC)

FLOORS = {"US": 585, "CA": 78, "MX": 18, "GB": 21, "AU": 14, "JP": 26, "TW": 3}
STALE_AFTER_DAYS = {"US": 3, "CA": 3, "MX": 3, "GB": 3, "AU": 3, "JP": 10, "TW": 14}


def interp_config() -> SimpleNamespace:
    """checks.py only reads .countries[cc].floor and .stale_after_days, and the
    .floor of a feed feeds.toml declares (Sam's is 460)."""
    return SimpleNamespace(
        countries={
            code: SimpleNamespace(code=code, floor=floor, stale_after_days=STALE_AFTER_DAYS[code])
            for code, floor in FLOORS.items()
        },
        feeds={"US-SAMS": SimpleNamespace(floor=460)},
    )


def context(previous_status: dict | None = None, capture_id: str = CAPTURE_ID) -> CaptureContext:
    return CaptureContext(
        capture_id=capture_id,
        capture_date=CAPTURE_DATE,
        fetch_config=SimpleNamespace(),
        interp_config=interp_config(),
        previous_stations=pl.DataFrame(schema=STATION_SCHEMA),
        previous_fx=pl.DataFrame(),
        previous_status=previous_status,
        shared={},
        force_fallback=set(),
    )


def rows(
    country: str,
    n_stations: int,
    *,
    source: str = "costco-occ",
    price_raw: str = "2.147",
    grade_raw: str = "regular",
) -> pl.DataFrame:
    records = [
        {
            "capture_id": CAPTURE_ID,
            "capture_date": CAPTURE_DATE,
            "captured_at_utc": CAPTURED_AT,
            "local_date": CAPTURE_DATE,
            "country": country,
            "station_key": f"{country}-{index}",
            "source_station_id": str(index),
            "source": source,
            "grade_raw": grade_raw,
            "grade": "regular",
            "price_raw": price_raw,
            "price": float(price_raw),
            "price_unit": "AUD/L",
            "currency": "AUD",
            "timezone": "Australia/Sydney",
        }
        for index in range(n_stations)
    ]
    return pl.DataFrame(records, schema=ROW_SCHEMA)


def normalized(
    country: str,
    n_stations: int,
    *,
    source: str = "costco-occ",
    price_raw: str = "2.147",
    drops: list[Drop] | None = None,
    warnings: list[Warning] | None = None,
) -> NormalizedCountry:
    return NormalizedCountry(
        rows=rows(country, n_stations, source=source, price_raw=price_raw),
        stations=pl.DataFrame(schema=STATION_SCHEMA),
        drops=drops or [],
        warnings=warnings or [],
    )


def empty_normalized() -> NormalizedCountry:
    return NormalizedCountry(
        rows=pl.DataFrame(schema=ROW_SCHEMA),
        stations=pl.DataFrame(schema=STATION_SCHEMA),
    )


def result(
    country: str,
    *,
    source: str = "costco-occ",
    requests: int = 1,
    warnings: list[Warning] | None = None,
    errors: list[Error] | None = None,
    responses: list[RawResponse] | None = None,
) -> FetchResult:
    return FetchResult(
        country=country,
        source=source,
        captured_at_utc=CAPTURED_AT,
        stations=[],
        responses=responses or [],
        requests=requests,
        warnings=warnings or [],
        errors=errors or [],
    )


def fx_rates(status: str = "ok") -> FxRates:
    return FxRates(
        status=status,
        rows=[
            FxRow(
                currency="AUD",
                units_per_usd=1.399,
                fx_usd_per_unit=1.0 / 1.399,
                fx_rate_date=date(2026, 9, 14),
                fx_source="frankfurter-v2",
                fx_fetched_at_utc=CAPTURED_AT,
            )
        ],
    )


def previous_status(**country_blocks) -> dict:
    """Keyed by feed. The kwargs stay country codes because an identifier
    cannot contain a hyphen; every one of them is that country's Costco feed."""
    return {
        "capture_id": "2026-09-15T1217Z",
        "feeds": {f"{code}-COSTCO": block for code, block in country_blocks.items()},
    }


# --- ok, degraded, failed, skipped -------------------------------------------


def test_a_healthy_country_is_ok():
    block = evaluate_feed("AU-COSTCO", result("AU"), normalized("AU", 14), context(), NOW)
    assert block["status"] == "ok"
    assert block["stations"] == 14
    assert block["rows"] == 14
    assert block["requests"] == 1
    assert block["consecutive_failures"] == 0
    assert block["last_success_capture_id"] == CAPTURE_ID
    assert block["unchanged_since_capture_id"] == CAPTURE_ID
    assert block["price_fingerprint"].startswith("sha256:")
    assert block["recent_errors"] == []
    assert block["dropped"] == {}


def test_a_station_count_below_the_floor_is_degraded():
    block = evaluate_feed("AU-COSTCO", result("AU"), normalized("AU", 13), context(), NOW)
    assert block["status"] == "degraded"


@pytest.mark.parametrize(("n_stations", "expected"), [(531, "ok"), (479, "ok"), (459, "degraded")])
def test_a_feed_is_judged_against_its_own_floor(n_stations, expected):
    """countries.US's floor of 585 counts Costco's stations. Against it a
    complete Sam's sweep of 531 clubs read degraded forever, and a short one
    looked no different. Sam's own floor is 460."""
    block = evaluate_feed(
        "US-SAMS",
        result("US", source="sams-clubfinder"),
        normalized("US", n_stations, source="sams-clubfinder"),
        context(),
        NOW,
    )
    assert block["status"] == expected


def test_costco_us_keeps_the_country_floor_beside_a_feed_with_its_own():
    block = evaluate_feed("US-COSTCO", result("US"), normalized("US", 531), context(), NOW)
    assert block["status"] == "degraded"


def test_a_sweep_cut_short_by_its_budget_is_degraded_whatever_the_floor_says():
    """479 of 531 clubs clears the floor of 460, so the floor alone reads a
    truncated sweep as ok. The source's own warning is what degrades it."""
    block = evaluate_feed(
        "US-SAMS",
        result(
            "US",
            source="sams-clubfinder",
            warnings=[Warning(code="budget_exhausted", detail="52 of 531 fuel clubs not reached")],
        ),
        normalized("US", 479, source="sams-clubfinder"),
        context(),
        NOW,
    )
    assert block["status"] == "degraded"


def test_a_fallback_source_is_degraded():
    block = evaluate_feed(
        "CA-COSTCO",
        result("CA", source="costco-ca-gasprices"),
        normalized("CA", 80, source="costco-ca-gasprices"),
        context(),
        NOW,
    )
    assert block["status"] == "degraded"


def test_a_us_costco_ca_lookup_fallback_source_is_degraded():
    # Well clear of the US floor (585) and otherwise healthy, so the only
    # thing that can make this degraded is the "costco-ca-lookup-us" marker
    # normalize.row_source() sets when the US falls back to the costco.ca
    # warehouse lookup.
    block = evaluate_feed(
        "US-COSTCO",
        result("US", source="costco-ca-lookup-us"),
        normalized("US", 600, source="costco-ca-lookup-us"),
        context(),
        NOW,
    )
    assert block["status"] == "degraded"


@pytest.mark.parametrize(
    "code", ["metadata_from_cache", "previous_state_unavailable", "unknown_grade"]
)
def test_degrading_warnings(code):
    block = evaluate_feed(
        "AU-COSTCO",
        result("AU", warnings=[Warning(code=code, detail="x")]),
        normalized("AU", 14),
        context(),
        NOW,
    )
    assert block["status"] == "degraded"
    assert {w["code"] for w in block["warnings"]} == {code}


def test_normalize_warnings_are_merged_into_the_block():
    block = evaluate_feed(
        "AU-COSTCO",
        result("AU", warnings=[Warning(code="deadline_exceeded", detail="AU")]),
        normalized("AU", 14, warnings=[Warning(code="no_regular", detail="109")]),
        context(),
        NOW,
    )
    assert block["status"] == "ok"
    assert {w["code"] for w in block["warnings"]} == {"deadline_exceeded", "no_regular"}


def test_more_than_five_percent_out_of_bounds_is_degraded():
    drops = [Drop("90", "out_of_bounds", "regular=1.929 USD/gal")]
    block = evaluate_feed(
        "AU-COSTCO", result("AU"), normalized("AU", 18, drops=drops), context(), NOW
    )
    assert block["dropped"] == {"out_of_bounds": 1}
    assert block["status"] == "degraded"  # 1 / 19 = 5.3%

    drops = [Drop("90", "out_of_bounds", "regular=1.929 USD/gal")]
    block = evaluate_feed(
        "AU-COSTCO", result("AU"), normalized("AU", 19, drops=drops), context(), NOW
    )
    assert block["status"] == "ok"  # 1 / 20 = 5.0%


def test_other_drop_reasons_do_not_degrade():
    drops = [Drop("1838", "not_open", ""), Drop("1121", "no_price", "")]
    block = evaluate_feed(
        "AU-COSTCO", result("AU"), normalized("AU", 14, drops=drops), context(), NOW
    )
    assert block["status"] == "ok"
    assert block["dropped"] == {"not_open": 1, "no_price": 1}


def test_no_rows_is_failed_and_bumps_the_counter():
    previous = previous_status(
        TW={
            "status": "ok",
            "consecutive_failures": 1,
            "last_success_capture_id": "2026-09-14T1817Z",
            "price_fingerprint": "sha256:old",
            "unchanged_since_capture_id": "2026-09-14T1817Z",
            "recent_errors": [{"capture_id": "2026-09-15T1217Z", "run_url": "u1", "errors": []}],
        }
    )
    errors = [Error(code="http_error", host="www.costco.com.tw", http_status=403, detail="")]
    block = evaluate_feed(
        "TW-COSTCO",
        result("TW", errors=errors),
        empty_normalized(),
        context(previous),
        NOW,
    )
    assert block["status"] == "failed"
    assert block["rows"] == 0
    assert block["stations"] == 0
    assert block["consecutive_failures"] == 2
    assert block["last_success_capture_id"] == "2026-09-14T1817Z"
    assert block["price_fingerprint"] == "sha256:old"
    assert block["errors"][0]["http_status"] == 403
    assert [entry["capture_id"] for entry in block["recent_errors"]] == [
        CAPTURE_ID,
        "2026-09-15T1217Z",
    ]
    assert block["recent_errors"][0]["errors"][0]["code"] == "http_error"


def test_recent_errors_keep_only_the_last_three():
    previous = previous_status(
        TW={
            "consecutive_failures": 3,
            "recent_errors": [
                {"capture_id": "2026-09-15T1217Z", "run_url": "u3", "errors": []},
                {"capture_id": "2026-09-15T0617Z", "run_url": "u2", "errors": []},
                {"capture_id": "2026-09-15T0017Z", "run_url": "u1", "errors": []},
            ],
        }
    )
    block = evaluate_feed("TW-COSTCO", result("TW"), empty_normalized(), context(previous), NOW)
    assert len(block["recent_errors"]) == 3
    assert [entry["capture_id"] for entry in block["recent_errors"]] == [
        CAPTURE_ID,
        "2026-09-15T1217Z",
        "2026-09-15T0617Z",
    ]
    assert block["consecutive_failures"] == 4


def test_a_success_resets_the_counter_but_keeps_recent_errors():
    previous = previous_status(
        AU={
            "consecutive_failures": 2,
            "last_success_capture_id": "2026-09-13T1817Z",
            "recent_errors": [{"capture_id": "2026-09-15T1217Z", "run_url": "u1", "errors": []}],
        }
    )
    block = evaluate_feed("AU-COSTCO", result("AU"), normalized("AU", 14), context(previous), NOW)
    assert block["status"] == "ok"
    assert block["consecutive_failures"] == 0
    assert block["last_success_capture_id"] == CAPTURE_ID
    assert len(block["recent_errors"]) == 1


def test_a_country_that_was_not_selected_is_skipped_and_changes_nothing():
    carried = {
        "status": "degraded",
        "source": "costco-occ",
        "stations": 27,
        "rows": 108,
        "consecutive_failures": 2,
        "last_success_capture_id": "2026-09-13T1817Z",
        "price_fingerprint": "sha256:abc",
        "unchanged_since_capture_id": "2026-09-10T1817Z",
        "recent_errors": [{"capture_id": "2026-09-15T1217Z", "run_url": "u1", "errors": []}],
    }
    block = evaluate_feed("JP-COSTCO", None, None, context(previous_status(JP=carried)), NOW)
    assert block["status"] == "skipped"
    assert block["consecutive_failures"] == 2
    assert block["last_success_capture_id"] == "2026-09-13T1817Z"
    assert block["price_fingerprint"] == "sha256:abc"
    assert block["unchanged_since_capture_id"] == "2026-09-10T1817Z"
    assert block["recent_errors"] == carried["recent_errors"]
    assert block["rows"] == 0
    assert block["stations"] == 0


# --- fingerprint and staleness ------------------------------------------------


def test_the_fingerprint_ignores_row_order():
    frame = rows("AU", 3)
    assert price_fingerprint(frame) == price_fingerprint(frame.reverse())


def test_an_unchanged_fingerprint_carries_its_capture_id_forward():
    first = evaluate_feed("AU-COSTCO", result("AU"), normalized("AU", 14), context(), NOW)
    ctx = context(previous_status(AU=first), capture_id="2026-09-16T0017Z")
    block = evaluate_feed("AU-COSTCO", result("AU"), normalized("AU", 14), ctx, NOW)
    assert block["price_fingerprint"] == first["price_fingerprint"]
    assert block["unchanged_since_capture_id"] == CAPTURE_ID
    assert block["last_success_capture_id"] == "2026-09-16T0017Z"


def test_a_changed_fingerprint_restarts_the_clock():
    first = evaluate_feed("AU-COSTCO", result("AU"), normalized("AU", 14), context(), NOW)
    ctx = context(previous_status(AU=first), capture_id="2026-09-16T0017Z")
    block = evaluate_feed(
        "AU-COSTCO", result("AU"), normalized("AU", 14, price_raw="2.157"), ctx, NOW
    )
    assert block["price_fingerprint"] != first["price_fingerprint"]
    assert block["unchanged_since_capture_id"] == "2026-09-16T0017Z"


@pytest.mark.parametrize(
    ("country", "n_stations", "days", "expected"),
    [
        ("AU", 14, 2, "ok"),
        ("AU", 14, 4, "degraded"),
        ("JP", 27, 4, "ok"),
        ("JP", 27, 11, "degraded"),
        ("TW", 3, 11, "ok"),
        ("TW", 3, 15, "degraded"),
    ],
)
def test_staleness_uses_each_countrys_stale_after_days(country, n_stations, days, expected):
    unchanged_since = "2026-09-01T1817Z"
    fingerprint = price_fingerprint(rows(country, n_stations))
    previous = previous_status(
        **{
            country: {
                "price_fingerprint": fingerprint,
                "unchanged_since_capture_id": unchanged_since,
            }
        }
    )
    now = datetime(2026, 9, 1, 18, 17, tzinfo=UTC) + timedelta(days=days, minutes=1)
    block = evaluate_feed(
        f"{country}-COSTCO",
        result(country),
        normalized(country, n_stations),
        context(previous),
        now,
    )
    assert block["unchanged_since_capture_id"] == unchanged_since
    assert block["status"] == expected


# --- duration -----------------------------------------------------------------


def test_duration_spans_the_first_request_and_the_last_response():
    responses = [
        RawResponse(
            key="AU-COSTCO/01-stores",
            url="https://www.costco.com.au/rest/v2/australia/stores",
            status=200,
            headers={},
            received_at_utc=datetime(2026, 9, 15, 18, 18, 0, tzinfo=UTC),
            elapsed_ms=1200,
            body=b"{}",
            error=None,
        ),
        RawResponse(
            key="AU-COSTCO/02-stores",
            url="https://www.costco.com.au/rest/v2/australia/stores",
            status=200,
            headers={},
            received_at_utc=datetime(2026, 9, 15, 18, 18, 9, tzinfo=UTC),
            elapsed_ms=800,
            body=b"{}",
            error=None,
        ),
    ]
    block = evaluate_feed(
        "AU-COSTCO",
        result("AU", responses=responses, requests=2),
        normalized("AU", 14),
        context(),
        NOW,
    )
    assert block["duration_s"] == 10.2


# --- all_failed ---------------------------------------------------------------


def test_all_failed_ignores_skipped_countries():
    assert all_failed({"countries": {"US": {"status": "failed"}, "CA": {"status": "skipped"}}})
    assert not all_failed({"countries": {"US": {"status": "failed"}, "CA": {"status": "degraded"}}})
    assert not all_failed({"countries": {"US": {"status": "ok"}}})
    assert all_failed({"countries": {}})


# --- build_status -------------------------------------------------------------


def test_build_status_assembles_the_whole_document():
    ctx = context(
        {
            "publish": {
                "outcome": "failure",
                "consecutive_failures": 1,
                "unpublished": [{"capture_id": "2026-09-15T1217Z", "run_id": 7, "run_url": "u"}],
            },
            "countries": {},
        }
    )
    au = evaluate_feed("AU-COSTCO", result("AU"), normalized("AU", 14), ctx, NOW)
    status = build_status(
        ctx,
        {"AU": au},
        fx_rates(),
        {"attempted": True, "http_status": 200},
        NOW,
        {
            "run_id": 123,
            "run_attempt": 1,
            "run_url": "https://github.com/x/y/actions/runs/123",
            "started_at_utc": CAPTURED_AT,
            "git_sha": "deadbeef",
            "warnings": [Warning(code="previous_state_unavailable", detail="")],
        },
    )
    assert status["schema_version"] == 1
    assert status["capture_id"] == CAPTURE_ID
    assert status["run_id"] == 123
    assert status["run_attempt"] == 1
    assert status["started_at_utc"] == "2026-09-15T18:17:40Z"
    assert status["finished_at_utc"] == "2026-09-15T18:19:02Z"
    assert status["git_sha"] == "deadbeef"
    assert status["fx"] == {
        "status": "ok",
        "source": "frankfurter-v2",
        "rate_date": "2026-09-14",
    }
    assert status["ecom_api"] == {"attempted": True, "http_status": 200}
    assert status["publish"] == {
        "outcome": None,
        "consecutive_failures": 1,
        "unpublished": [{"capture_id": "2026-09-15T1217Z", "run_id": 7, "run_url": "u"}],
    }
    assert status["close"] == {"outcome": None}
    assert status["warnings"] == [{"code": "previous_state_unavailable", "detail": ""}]
    assert sorted(status["countries"]) == ["AU", "CA", "GB", "JP", "MX", "TW", "US"]
    assert status["countries"]["AU"]["status"] == "ok"
    assert status["countries"]["US"]["status"] == "skipped"


def test_build_status_fills_the_run_url_into_this_captures_recent_errors():
    ctx = context()
    tw = evaluate_feed(
        "TW-COSTCO",
        result(
            "TW",
            errors=[
                Error(
                    code="timeout",
                    host="www.costco.com.tw",
                    http_status=None,
                    detail="",
                )
            ],
        ),
        empty_normalized(),
        ctx,
        NOW,
    )
    assert tw["recent_errors"][0]["run_url"] is None
    status = build_status(
        ctx,
        {"TW-COSTCO": tw},
        FxRates(status="failed", rows=[]),
        {"attempted": False, "http_status": None},
        NOW,
        {"run_id": 9, "run_attempt": 1, "run_url": "https://gh/runs/9"},
    )
    entry = status["feeds"]["TW-COSTCO"]["recent_errors"][0]
    assert entry["capture_id"] == CAPTURE_ID
    assert entry["run_url"] == "https://gh/runs/9"
    assert status["fx"] == {"status": "failed", "source": None, "rate_date": None}
    assert all_failed(status)
