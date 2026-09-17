"""Record types, the source registry and the bundle response files."""

from __future__ import annotations

import gzip
import json
from dataclasses import FrozenInstanceError
from datetime import UTC, date, datetime
from pathlib import Path

import polars as pl
import pytest

from club_gas.sources.base import (
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
        key="US-COSTCO/03-gasprices",
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


def test_sources_registry_is_keyed_by_feed_not_country():
    """The United States has two chains, so the country cannot be the key."""
    from club_gas.sources.base import SOURCES, brand_of, country_of, feeds_for

    assert set(SOURCES) == {
        "US-COSTCO",
        "CA-COSTCO",
        "MX-COSTCO",
        "GB-COSTCO",
        "AU-COSTCO",
        "JP-COSTCO",
        "TW-COSTCO",
        "US-SAMS",
    }
    for fid, source in SOURCES.items():
        assert fid == f"{source.country}-{source.brand}"
        assert callable(source.fetch)
        assert callable(source.parse)

    assert feeds_for(["US"]) == ["US-COSTCO", "US-SAMS"]
    assert feeds_for(["JP"]) == ["JP-COSTCO"]
    assert (country_of("US-SAMS"), brand_of("US-SAMS")) == ("US", "SAMS")


def test_lazy_source_imports_its_module_only_when_called(monkeypatch):
    import sys
    import types

    from club_gas.sources.base import _LazySource

    calls: list[str] = []

    class FakeSource:
        def __init__(self, country: str) -> None:
            self.country = country

        def fetch(self, client, ctx):
            calls.append("fetch")
            return ["fetched"]

        def parse(self, responses, ctx):
            calls.append("parse")
            # A real FetchResult, because _LazySource stamps the registry's
            # brand onto whatever comes back.
            return FetchResult(
                country=self.country,
                source="fake",
                captured_at_utc=datetime(2026, 9, 15, 19, 10, tzinfo=UTC),
            )

    module = types.ModuleType("club_gas_fake_source")
    module.FakeSource = FakeSource
    monkeypatch.setitem(sys.modules, "club_gas_fake_source", module)

    source = _LazySource("ZZ", "club_gas_fake_source", "FakeSource", pass_country=True)
    assert source.country == "ZZ"
    assert calls == []
    assert source.fetch(None, None) == ["fetched"]
    parsed = source.parse([], None)
    # The registry stamps its own brand, so a source cannot disagree with the
    # key its rows will be built from.
    assert (parsed.country, parsed.brand) == ("ZZ", "COSTCO")
    assert calls == ["fetch", "parse"]


def test_bundle_responses_round_trip(tmp_path):
    from club_gas.sources.base import read_responses, response_paths, write_responses

    body = (FIXTURES / "bundle" / "us_gasprices_batch.body").read_bytes()
    assert len(body) == 463
    # A gzipped body proves the .body file is written and read as raw bytes and is
    # never decoded as text.
    shared_body = gzip.compress(b'{"warehouses":[{"warehouseId":"1364"}]}')

    prices = RawResponse(
        key="US-COSTCO/03-gasprices",
        url=(
            "https://www.costco.com/AjaxGetGasPricesService"
            "?warehouseid=1772_335_1680_1765_1793_140_1838_120_1090_1364"
        ),
        status=200,
        headers={
            "Content-Type": "text/html;charset=UTF-8",
            "Date": "Tue, 15 Sep 2026 19:10:16 GMT",
        },
        received_at_utc=datetime(2026, 9, 15, 19, 10, 16, tzinfo=UTC),
        elapsed_ms=560,
        body=body,
        error=None,
    )
    shared = RawResponse(
        key="shared/ecom-api",
        url=(
            "https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json"
            "?latitude=0&longitude=0&limit=5000"
        ),
        status=None,
        headers={},
        received_at_utc=datetime(2026, 9, 15, 19, 10, 45, tzinfo=UTC),
        elapsed_ms=20000,
        body=shared_body,
        error="ReadTimeout",
    )

    written = write_responses([prices, shared], tmp_path)
    assert len(written) == 4

    body_path, meta_path = response_paths(tmp_path, "US-COSTCO/03-gasprices")
    assert body_path == tmp_path / "responses" / "US-COSTCO" / "03-gasprices.body"
    assert meta_path == tmp_path / "responses" / "US-COSTCO" / "03-gasprices.meta.json"
    assert body_path.read_bytes() == body

    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["key"] == "US-COSTCO/03-gasprices"
    assert meta["status"] == 200
    assert meta["elapsed_ms"] == 560
    assert meta["error"] is None
    assert "body" not in meta

    back = {response.key: response for response in read_responses(tmp_path)}
    assert set(back) == {"US-COSTCO/03-gasprices", "shared/ecom-api"}
    assert back["US-COSTCO/03-gasprices"] == prices
    assert back["shared/ecom-api"] == shared
    assert back["shared/ecom-api"].status is None
    assert back["shared/ecom-api"].error == "ReadTimeout"


def test_read_responses_is_sorted_and_empty_without_a_responses_dir(tmp_path):
    from club_gas.sources.base import read_responses, write_responses

    assert read_responses(tmp_path) == []

    received = datetime(2026, 9, 15, 19, 11, 0, tzinfo=UTC)
    write_responses(
        [
            RawResponse("US-COSTCO/02", "https://example.test/2", 200, {}, received, 1, b"2"),
            RawResponse("US-COSTCO/01", "https://example.test/1", 200, {}, received, 1, b"1"),
            RawResponse("CA-COSTCO/01", "https://example.test/3", 200, {}, received, 1, b"3"),
        ],
        tmp_path,
    )
    assert [r.key for r in read_responses(tmp_path)] == [
        "CA-COSTCO/01",
        "US-COSTCO/01",
        "US-COSTCO/02",
    ]


@pytest.mark.parametrize(
    "key",
    ["", "/US/01", "US/../../etc/passwd", "US//01", "US\\01", "US/./01"],
)
def test_response_paths_rejects_unsafe_keys(tmp_path, key):
    from club_gas.sources.base import response_paths

    with pytest.raises(ValueError, match="unsafe response key"):
        response_paths(tmp_path, key)


def test_read_responses_requires_the_body_file(tmp_path):
    from club_gas.sources.base import read_responses, write_responses

    write_responses(
        [
            RawResponse(
                key="CA-COSTCO/01-lookup",
                url="https://www.costco.ca/AjaxWarehouseBrowseLookupView?countryCode=CA",
                status=200,
                headers={},
                received_at_utc=datetime(2026, 9, 15, 19, 11, 30, tzinfo=UTC),
                elapsed_ms=30600,
                body=b"[false]",
            )
        ],
        tmp_path,
    )
    (tmp_path / "responses" / "CA-COSTCO" / "01-lookup.body").unlink()
    with pytest.raises(FileNotFoundError, match="missing response body"):
        read_responses(tmp_path)
