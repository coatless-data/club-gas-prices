"""Record types, the source registry and the bundle response files."""

from __future__ import annotations

import gzip
import json
from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from costco_gas.sources.base import (
    CaptureContext,
    Error,
    FetchResult,
    RawPrice,
    RawResponse,
    RawStation,
    Warning,
)

FIXTURES = Path(__file__).resolve().parent / "fixtures"


def test_raw_station_holds_the_spec_fields_and_is_frozen():
    station = RawStation(
        source_station_id="1364",
        alt_id=None,
        id_origin="ecom",
        ecom_state="gas",
        name="Sunnyvale",
        prices=(
            RawPrice(grade_raw="regular", price_raw="3.999"),
            RawPrice(grade_raw="premium", price_raw="4.629"),
        ),
    )
    assert station.city is None
    assert station.lat is None
    assert station.opening_date is None
    assert station.has_hours is None
    assert station.prices[0].grade_raw == "regular"
    assert station.prices[1].price_raw == "4.629"
    with pytest.raises(FrozenInstanceError):
        station.name = "Renamed"


def test_raw_response_keeps_the_body_as_bytes():
    response = RawResponse(
        key="US/03-gasprices",
        url="https://www.costco.com/AjaxGetGasPricesService?warehouseid=1364",
        status=200,
        headers={"Content-Type": "text/html;charset=UTF-8"},
        received_at_utc=datetime(2026, 9, 15, 19, 10, 16, tzinfo=UTC),
        elapsed_ms=560,
        body=b'{"1364":{"premium":"4.629","regular":"3.999"}}',
    )
    assert response.error is None
    assert isinstance(response.body, bytes)
    with pytest.raises(FrozenInstanceError):
        response.status = 403


def test_fetch_result_collects_warnings_and_errors():
    result = FetchResult(
        country="US",
        source="costco-us-gasprices",
        captured_at_utc=datetime(2026, 9, 15, 19, 10, 16, tzinfo=UTC),
    )
    assert result.stations == []
    assert result.responses == []
    assert result.requests == 0
    result.warnings.append(Warning(code="not_in_ecom_api", detail="1772"))
    result.errors.append(
        Error(code="http_error", host="www.costco.com", http_status=403, detail="denied")
    )
    assert result.warnings[0].code == "not_in_ecom_api"
    assert result.warnings[0].detail == "1772"
    assert result.errors[0].host == "www.costco.com"
    assert Warning(code="no_regular").detail is None
    assert Error(code="timeout").http_status is None


def test_capture_context_defaults_are_empty_not_shared():
    first = CaptureContext(
        capture_id="2026-09-15T1817Z",
        capture_date=date(2026, 9, 15),
        fetch_config=None,
        interp_config=None,
    )
    second = CaptureContext(
        capture_id="2026-09-15T1817Z",
        capture_date=date(2026, 9, 15),
        fetch_config=None,
        interp_config=None,
    )
    assert first.previous_stations.height == 0
    assert first.previous_fx.height == 0
    assert first.previous_status is None
    assert first.shared == {}
    assert first.force_fallback == set()
    first.force_fallback.add("US")
    assert second.force_fallback == set()
    assert isinstance(first.previous_stations, pl.DataFrame)
