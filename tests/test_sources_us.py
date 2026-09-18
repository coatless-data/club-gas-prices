"""Tests for the United States source (spec 4.2). No network: a fake Client serves fixtures."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from club_gas.http import BudgetExceeded
from club_gas.sources import us
from club_gas.sources.base import SOURCES, CaptureContext, RawResponse

FIXTURES = Path(__file__).parent / "fixtures"
CAPTURE_ID = "2026-09-15T1910Z"
CAPTURE_DATE = date(2026, 9, 15)
CAPTURE_START = datetime(2026, 9, 15, 19, 10, tzinfo=UTC)

US_TIMEZONES = {
    "AZ": "America/Phoenix",
    "CA": "America/Los_Angeles",
    "DC": "America/New_York",
    "FL": "America/New_York",
    "HI": "Pacific/Honolulu",
    "MO": "America/Chicago",
    "PR": "America/Puerto_Rico",
    "TX": "America/Chicago",
    "WA": "America/Los_Angeles",
}

EXTRAS = pl.DataFrame(
    {
        "source_station_id": ["1680", "1765", "1772"],
        "name": ["Camarillo", "Chandler Business Center", "Mission Viejo"],
        "city": ["CAMARILLO", "CHANDLER", "MISSION VIEJO"],
        "region": ["CA", "AZ", "CA"],
        "postcode": ["93010-9122", "85226", "92691"],
        "lat": [None, None, None],
        "lon": [None, None, None],
        "timezone": ["America/Los_Angeles", "America/Phoenix", None],
        "note": ["prices live, absent from ecom-api", "business center", "reported"],
    },
    schema_overrides={"lat": pl.Float64, "lon": pl.Float64, "timezone": pl.Utf8},
)

STATION_COLUMNS = [
    "station_key",
    "country",
    "source_station_id",
    "alt_id",
    "name",
    "name_local",
    "address",
    "city",
    "region",
    "postcode",
    "lat",
    "lon",
    "timezone",
    "grades_seen",
    "first_seen_utc",
    "last_seen_utc",
    "status",
    "superseded_by",
]


# --------------------------------------------------------------------------- helpers


@dataclass
class StubCountry:
    """The `config.CountryConfig` fields this module reads, with the real US values."""

    code: str = "US"
    url: str = us.DEFAULT_PRICE_URL
    params: dict = field(default_factory=dict)
    batch_size: int | None = 10
    seen_within_days: int | None = 30
    fallback_url: str | None = "https://www.costco.ca/AjaxWarehouseBrowseLookupView"
    fallback_params: dict = field(
        default_factory=lambda: {
            "hasGas": "true",
            "populateWarehouseDetails": "true",
            "countryCode": "US",
        }
    )
    ecom_url: str | None = "https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json"
    ecom_params: dict = field(
        default_factory=lambda: {"latitude": "0", "longitude": "0", "limit": "5000"}
    )
    ecom_client_identifier: str | None = "7c71124c-7bf1-44db-bc9d-498584cd66e5"
    timezones: dict = field(default_factory=lambda: dict(US_TIMEZONES))

    def timezone_for_region(self, region: str | None) -> str | None:
        """Same fallback as the real one: the region key, then the `"*"` default."""
        if region is not None and region in self.timezones:
            return self.timezones[region]
        return self.timezones.get("*")


@dataclass
class StubConfig:
    countries: dict
    us_extra_ids: pl.DataFrame


def stations_frame(rows: list[dict]) -> pl.DataFrame:
    if not rows:
        return pl.DataFrame(
            {c: [] for c in STATION_COLUMNS},
            schema={c: pl.Utf8 for c in STATION_COLUMNS},
        )
    filled = [{c: row.get(c) for c in STATION_COLUMNS} for row in rows]
    return pl.DataFrame(filled, schema={c: pl.Utf8 for c in STATION_COLUMNS})


def make_ctx(
    *,
    ecom: RawResponse | None = None,
    extras: pl.DataFrame | None = None,
    previous: pl.DataFrame | None = None,
    force_fallback: set[str] | None = None,
    country: StubCountry | None = None,
) -> CaptureContext:
    config = StubConfig(
        countries={"US": country or StubCountry()},
        us_extra_ids=EXTRAS if extras is None else extras,
    )
    return CaptureContext(
        capture_id=CAPTURE_ID,
        capture_date=CAPTURE_DATE,
        fetch_config=config,
        interp_config=config,
        previous_stations=stations_frame([]) if previous is None else previous,
        previous_fx=pl.DataFrame(),
        previous_status=None,
        shared={} if ecom is None else {"ecom-api": ecom},
        force_fallback=force_fallback or set(),
    )


def response(
    key: str,
    url: str,
    *,
    fixture: str | None = None,
    body: bytes = b"",
    status: int | None = 200,
    error: str | None = None,
    offset: int = 0,
) -> RawResponse:
    payload = (FIXTURES / fixture).read_bytes() if fixture else body
    return RawResponse(
        key=key,
        url=url,
        status=status,
        headers={"Content-Type": "text/html;charset=UTF-8"},
        received_at_utc=CAPTURE_START + timedelta(seconds=offset),
        elapsed_ms=120,
        body=payload,
        error=error,
    )


def ecom_ok() -> RawResponse:
    return response(
        "shared/01-ecom-api",
        "https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json",
        fixture="us_ecom_warehouses.json",
    )


def ecom_failed() -> RawResponse:
    return response(
        "shared/01-ecom-api",
        "https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json",
        body=b"",
        status=None,
        error="ConnectTimeout",
    )


class FakeClient:
    """Stands in for http.Client. The real one is covered by test_http.py."""

    def __init__(self, responder, *, abandon_hosts=(), abandon_after=None):
        self.responder = responder
        self.calls: list[tuple[str, str, str]] = []
        self.abandon_hosts = set(abandon_hosts)
        self.abandon_after = dict(abandon_after or {})
        self.counts: dict[str, int] = {}

    @staticmethod
    def host_of(url: str) -> str:
        return url.split("://", 1)[-1].split("/", 1)[0]

    def request(self, key, url, *, profile="default", headers=None, expect_json=True):
        host = self.host_of(url)
        self.calls.append((key, url, profile))
        self.counts[host] = self.counts.get(host, 0) + 1
        return self.responder(key, url)

    def abandoned(self, url: str) -> bool:
        host = self.host_of(url)
        if host in self.abandon_hosts:
            return True
        limit = self.abandon_after.get(host)
        return limit is not None and self.counts.get(host, 0) >= limit


TOPUP_BODY = (
    b'{"1793":{"premium":"4.199","regular":"3.499"},'
    b'"1765":{"premium":"5.299","regular":"4.999"},'
    b'"1772":{"premium":"6.149","regular":"5.849"}}'
)


def fallback_responder(key, url):
    if "lookup" in key:
        return response(key, url, fixture="us_lookup_us.json", offset=30)
    return response(key, url, body=TOPUP_BODY, offset=40)


def by_id(result) -> dict:
    return {s.source_station_id: s for s in result.stations}


def grades_of(station) -> dict[str, str]:
    return {p.grade_raw: p.price_raw for p in station.prices}


def warning_details(result, code: str) -> list[str]:
    return [w.detail for w in result.warnings if w.code == code]


# --------------------------------------------------------------------------- batching


def test_batches_split_by_ten():
    ids = [str(i) for i in range(1, 26)]
    assert us.batches(ids) == [ids[0:10], ids[10:20], ids[20:25]]


def test_batch_size_and_seen_window_come_from_the_country_config():
    assert us.batch_size(make_ctx()) == 10
    assert us.seen_window_days(make_ctx()) == 30
    small = make_ctx(country=StubCountry(batch_size=3, seen_within_days=7))
    assert us.batch_size(small) == 3
    assert us.seen_window_days(small) == 7
    ids = [str(i) for i in range(1, 8)]
    assert us.batches(ids, us.batch_size(small)) == [
        ["1", "2", "3"],
        ["4", "5", "6"],
        ["7"],
    ]


def test_price_and_lookup_urls_come_from_the_country_config():
    ctx = make_ctx()
    assert us.price_url(ctx) == "https://www.costco.com/AjaxGetGasPricesService"
    assert us.price_params(ctx) == {}
    assert us.lookup_url(ctx) == (
        "https://www.costco.ca/AjaxWarehouseBrowseLookupView"
        "?hasGas=true&populateWarehouseDetails=true&countryCode=US"
    )


def test_batch_url_puts_the_ids_first_and_appends_configured_params():
    assert us.batch_url(make_ctx(), ["1364", "140"]) == (
        "https://www.costco.com/AjaxGetGasPricesService?warehouseid=1364_140"
    )
    extra = make_ctx(country=StubCountry(params={"locale": "en-US"}))
    assert us.batch_url(extra, ["1364"]) == (
        "https://www.costco.com/AjaxGetGasPricesService?warehouseid=1364&locale=en-US"
    )
    assert us.ids_in_url(us.batch_url(extra, ["1364", "140"])) == ["1364", "140"]


def test_price_service_processes_only_the_first_ten_ids():
    raw = response(
        "US-COSTCO/02-gasprices-001",
        "https://www.costco.com/AjaxGetGasPricesService?warehouseid=1_2_3_4_6_8_9_10_11_13_14",
        fixture="us_gasprices_batch_cap.json",
    )
    requested = us.ids_in_url(raw.url)
    parsed = us.parse_price_batch(raw)
    assert len(requested) == 11
    assert set(parsed) == set(requested[:10])
    assert "14" not in parsed


def test_parse_price_batch_rejects_error_message_and_html():
    bad_id = response(
        "US-COSTCO/02-gasprices-001",
        "u",
        body=b'{"errorMessage":"warehouse id supplied, abc, is not a number"}',
    )
    blocked = response(
        "US-COSTCO/02-gasprices-001",
        "u",
        body=b"<HTML><HEAD><TITLE>Access Denied</TITLE></HEAD></HTML>",
    )
    assert us.parse_price_batch(bad_id) is None
    assert us.parse_price_batch(blocked) is None


def test_parse_price_batch_accepts_json_served_as_text_html():
    raw = response("US-COSTCO/02-gasprices-001", "u", fixture="us_gasprices_batch_ua.json")
    assert raw.headers["Content-Type"] == "text/html;charset=UTF-8"
    parsed = us.parse_price_batch(raw)
    assert parsed["1364"] == {"premium": "4.629", "regular": "3.999"}
    assert parsed["120"] == {"premium": "3.829"}
    assert parsed["140"]["clear"] == "5.699"


# --------------------------------------------------------------------------- ecom state


def test_ecom_state_classification():
    index = us.parse_ecom(ecom_ok())
    assert us.ecom_state(index, "1364") == "gas"
    assert us.ecom_state(index, "335") == "gas"
    assert us.ecom_state(index, "120") == "no_gas"
    assert us.ecom_state(index, "1324") == "no_gas"
    assert us.ecom_state(index, "1680") == "absent"
    assert us.ecom_state(None, "1364") == "unavailable"


def test_parse_ecom_reads_the_fields_the_spec_uses():
    index = us.parse_ecom(ecom_ok())
    kona = index["140"]
    assert kona.name == "Kona"
    assert kona.city == "KAILUA KONA"
    assert kona.territory == "HI"
    assert kona.postal_code == "96740-2630"
    assert kona.timezone == "Pacific/Honolulu"
    assert kona.lat == pytest.approx(19.68545677)
    assert index["1838"].opening_date == date(2026, 10, 2)
    assert index["729"].sub_type == "Business Center"
    assert index["1324"].country == "CA"


def test_parse_ecom_returns_none_for_a_failed_step_one():
    assert us.parse_ecom(ecom_failed()) is None
    assert us.parse_ecom(response("shared/01-ecom-api", "u", body=b"", status=401)) is None
    assert us.parse_ecom(response("shared/01-ecom-api", "u", body=b"<html>")) is None
    assert us.parse_ecom(None) is None


def test_polled_id_set_records_three_origins():
    previous = stations_frame(
        [
            {
                "station_key": "US-COSTCO-120",
                "country": "US",
                "source_station_id": "120",
                "last_seen_utc": "2026-09-14T18:17:00Z",
                "region": "HI",
                "timezone": "Pacific/Honolulu",
            },
            {
                "station_key": "US-COSTCO-9999",
                "country": "US",
                "source_station_id": "9999",
                "last_seen_utc": "2026-01-01T18:17:00Z",
                "region": "TX",
            },
            {
                "station_key": "US-COSTCO-1364",
                "country": "US",
                "source_station_id": "1364",
                "last_seen_utc": "2026-09-14T18:17:00Z",
                "region": "FL",
            },
        ]
    )
    ctx = make_ctx(ecom=ecom_ok(), previous=previous)
    polled = {p.source_station_id: p for p in us.polled_id_set(ctx)}
    assert polled["1364"].id_origin == "ecom"
    assert polled["1680"].id_origin == "extra"
    assert polled["120"].id_origin == "seen"
    assert polled["120"].ecom_state == "no_gas"
    assert "9999" not in polled
    frame = us.polled_id_frame(ctx)
    assert frame.columns == ["source_station_id", "id_origin", "ecom_state"]
    assert frame.height == len(polled)


def test_polled_id_set_uses_the_cache_when_step_one_fails():
    previous = stations_frame(
        [
            {
                "station_key": "US-COSTCO-1364",
                "country": "US",
                "source_station_id": "1364",
                "region": "FL",
                "last_seen_utc": "2026-09-14T18:17:00Z",
            }
        ]
    )
    ctx = make_ctx(ecom=ecom_failed(), previous=previous)
    polled = us.polled_id_set(ctx)
    assert [(p.source_station_id, p.id_origin) for p in polled] == [
        ("1364", "cache"),
        ("1680", "extra"),
        ("1765", "extra"),
        ("1772", "extra"),
    ]
    assert {p.ecom_state for p in polled} == {"unavailable"}


def test_a_sams_club_in_stations_csv_is_never_polled_at_costco():
    """stations.csv holds both US chains once Sam's has published. Its ~530
    club numbers would add some 53 batches to every capture, each asking
    Costco for warehouses it does not have."""
    previous = stations_frame(
        [
            {
                "station_key": "US-COSTCO-1364",
                "country": "US",
                "source_station_id": "1364",
                "region": "FL",
                "last_seen_utc": "2026-09-14T18:17:00Z",
            },
            {
                "station_key": "US-SAMS-6376",
                "country": "US",
                "source_station_id": "6376",
                "region": "TX",
                "last_seen_utc": "2026-09-14T18:17:00Z",
            },
        ]
    ).with_columns(pl.Series("brand", ["COSTCO", "SAMS"]))

    for ecom in (ecom_ok(), ecom_failed()):
        ctx = make_ctx(ecom=ecom, previous=previous)
        polled = {p.source_station_id for p in us.polled_id_set(ctx)}
        assert "1364" in polled
        assert "6376" not in polled

    # And no price batch that fetch_us sends carries the club number.
    ctx = make_ctx(ecom=ecom_failed(), previous=previous)
    client = FakeClient(lambda key, url: response(key, url, body=b"{}"))
    us.fetch_us(client, ctx)
    sent = [url for _, url, _ in client.calls if "AjaxGetGasPricesService" in url]
    assert sent
    assert not [url for url in sent if "6376" in url]


# --------------------------------------------------------------------------- lookup


def test_parse_lookup_skips_element_zero_and_leading_whitespace():
    raw = response("US-COSTCO/03-lookup-us", us.DEFAULT_LOOKUP_URL, fixture="us_lookup_us.json")
    assert raw.body.startswith(b"\r\n")
    rows = us.parse_lookup(raw)
    assert set(rows) == {"1", "140", "335", "651", "1364", "1680", "1838"}
    assert rows["1680"]["openDate"] == "Oct 30, 2026"
    assert rows["1680"]["gasStationHours"] == []
    assert rows["1838"]["gasStationHours"] != []


def test_lookup_prices_drop_the_two_non_grade_keys():
    rows = us.parse_lookup(
        response("US-COSTCO/03-lookup-us", us.DEFAULT_LOOKUP_URL, fixture="us_lookup_us.json")
    )
    assert us.lookup_prices(rows["140"]) == {
        "diesel": "6.899",
        "regular": "4.899",
        "premium": "5.699",
        "clear": "5.699",
    }
    assert us.lookup_prices(rows["1680"]) == {"regular": "5.799", "premium": "6.099"}


def test_lookup_open_date_parsing():
    assert us.lookup_open_date("Aug 23, 1995") == date(1995, 8, 23)
    assert us.lookup_open_date("Oct 30, 2026") == date(2026, 10, 30)
    assert us.lookup_open_date("") is None
    assert us.lookup_open_date(None) is None


def test_parse_lookup_rejects_a_failed_response():
    assert us.parse_lookup(response("US-COSTCO/03-lookup-us", "u", body=b"", status=403)) is None
    assert us.parse_lookup(response("US-COSTCO/03-lookup-us", "u", body=b"<html>")) is None


# --------------------------------------------------------------------------- happy path


def happy_ctx() -> CaptureContext:
    previous = stations_frame(
        [
            {
                "station_key": "US-COSTCO-120",
                "country": "US",
                "source_station_id": "120",
                "name": "Hawaii Kai",
                "region": "HI",
                "timezone": "Pacific/Honolulu",
                "last_seen_utc": "2026-09-14T18:17:00Z",
            },
        ]
    )
    return make_ctx(ecom=ecom_ok(), previous=previous)


def happy_result():
    ctx = happy_ctx()
    batch = response(
        "US-COSTCO/02-gasprices-001",
        "https://www.costco.com/AjaxGetGasPricesService?warehouseid="
        "1772_335_1680_1765_1793_140_1838_120_1090_1364",
        fixture="us_gasprices_batch_ua.json",
        offset=5,
    )
    return us.parse_us([batch], ctx), ctx


def test_step_two_prices_and_source():
    result, _ = happy_result()
    stations = by_id(result)
    assert result.source == "costco-us-gasprices"
    assert result.captured_at_utc == CAPTURE_START + timedelta(seconds=5)
    assert grades_of(stations["1364"]) == {"premium": "4.629", "regular": "3.999"}
    assert grades_of(stations["140"]) == {
        "diesel": "6.899",
        "premium": "5.699",
        "clear": "5.699",
        "regular": "4.899",
    }
    assert stations["1364"].id_origin == "ecom"
    assert stations["1364"].lat == pytest.approx(27.49462445)


def test_puerto_rico_carries_the_region_marker_for_the_unit_override():
    result, _ = happy_result()
    station = by_id(result)["335"]
    assert station.region == "PR"
    assert grades_of(station) == {"premium": "1.267", "regular": "1.097"}


def test_pre_opening_station_keeps_its_opening_date():
    result, _ = happy_result()
    station = by_id(result)["1838"]
    assert station.opening_date == date(2026, 10, 2)
    assert grades_of(station) == {"premium": "5.999", "regular": "3.999"}


def test_no_gas_seen_id_is_emitted_with_its_state():
    result, _ = happy_result()
    station = by_id(result)["120"]
    assert station.id_origin == "seen"
    assert station.ecom_state == "no_gas"
    assert grades_of(station) == {"premium": "3.829"}


def test_absent_priced_ids_warn_not_in_ecom_api():
    result, _ = happy_result()
    assert sorted(warning_details(result, "not_in_ecom_api")) == [
        "1680",
        "1765",
        "1772",
    ]


def test_timezone_falls_back_to_the_region_table_with_a_warning():
    result, _ = happy_result()
    stations = by_id(result)
    assert stations["1765"].timezone == "America/Phoenix"
    assert stations["1772"].timezone == "America/Los_Angeles"
    assert warning_details(result, "timezone_from_region") == ["1772"]


def test_unpolled_ids_in_a_batch_are_ignored():
    result, _ = happy_result()
    assert "1090" not in by_id(result)


def test_station_keys_are_unique():
    result, _ = happy_result()
    ids = [s.source_station_id for s in result.stations]
    assert len(ids) == len(set(ids))


# --------------------------------------------------------------------------- cached path


def test_step_one_failure_uses_cached_metadata():
    rows = [
        {
            "station_key": f"US-COSTCO-{i}",
            "country": "US",
            "source_station_id": str(i),
            "name": f"Store {i}",
            "region": "TX",
            "timezone": "America/Chicago",
            "last_seen_utc": "2026-09-14T18:17:00Z",
        }
        for i in range(1, 601)
    ]
    ctx = make_ctx(ecom=ecom_failed(), previous=stations_frame(rows))
    polled = us.polled_id_set(ctx)
    assert len(polled) == 603
    assert {p.ecom_state for p in polled} == {"unavailable"}
    assert polled[0].id_origin == "cache"

    body = b'{"1":{"premium":"5.899","regular":"5.399"}}'
    batch = response(
        "US-COSTCO/02-gasprices-001",
        "https://www.costco.com/AjaxGetGasPricesService?warehouseid=1",
        body=body,
    )
    result = us.parse_us([batch], ctx)
    assert len(result.stations) >= 585
    assert warning_details(result, "metadata_from_cache") == ["US"]
    assert by_id(result)["1"].timezone == "America/Chicago"


def test_lookup_only_rows_are_used_when_step_one_failed_with_no_cache():
    ctx = make_ctx(ecom=ecom_failed())
    lookup = response(
        "US-COSTCO/03-lookup-us", us.DEFAULT_LOOKUP_URL, fixture="us_lookup_us.json", offset=30
    )
    result = us.parse_us([lookup], ctx)
    stations = by_id(result)
    assert stations["1364"].id_origin == "lookup"
    assert stations["1364"].ecom_state == "unavailable"
    assert stations["1680"].id_origin == "lookup"
    assert stations["1"].region == "WA"
    assert stations["1"].timezone == "America/Los_Angeles"
    assert warning_details(result, "metadata_from_cache") == []


def test_merge_rule_keeps_step_two_prices_and_fills_only_unpriced_ids():
    ctx = make_ctx(ecom=ecom_ok())
    step_two = response(
        "US-COSTCO/02-gasprices-001",
        "https://www.costco.com/AjaxGetGasPricesService?warehouseid="
        "1772_335_1680_1765_1793_140_1838_120_1090_1364",
        fixture="us_gasprices_batch_ua.json",
        offset=5,
    )
    lookup = response(
        "US-COSTCO/03-lookup-us", us.DEFAULT_LOOKUP_URL, fixture="us_lookup_us.json", offset=30
    )
    result = us.parse_us([step_two, lookup], ctx)
    stations = by_id(result)
    assert stations["140"].id_origin == "ecom"
    assert grades_of(stations["140"])["regular"] == "4.899"
    assert stations["651"].id_origin == "lookup"
    assert grades_of(stations["651"]) == {"regular": "3.929", "premium": "4.649"}
    ids = [s.source_station_id for s in result.stations]
    assert len(ids) == len(set(ids))
    assert "1" not in stations


# --------------------------------------------------------------------------- fallback


def test_fallback_when_more_than_half_the_batches_fail():
    previous = stations_frame(
        [
            {
                "station_key": "US-COSTCO-120",
                "country": "US",
                "source_station_id": "120",
                "region": "HI",
                "last_seen_utc": "2026-09-14T18:17:00Z",
            }
        ]
    )
    ctx = make_ctx(ecom=ecom_ok(), previous=previous)

    def responder(key, url):
        if "lookup" in key:
            return response(key, url, fixture="us_lookup_us.json", offset=30)
        if key == "US-COSTCO/02-gasprices-001":
            return response(key, url, body=b"", status=403, error=None)
        return response(key, url, body=TOPUP_BODY, offset=40)

    client = FakeClient(responder)
    responses = us.fetch_us(client, ctx)
    keys = [r.key for r in responses]
    assert keys[0] == "US-COSTCO/02-gasprices-001"
    assert "US-COSTCO/03-lookup-us" in keys
    assert next(c[2] for c in client.calls if c[0] == "US-COSTCO/03-lookup-us") == "bulk"
    assert next(c[1] for c in client.calls if c[0] == "US-COSTCO/03-lookup-us") == us.lookup_url(
        ctx
    )


def test_fallback_when_costco_com_is_abandoned_partway_through_step_two():
    previous = stations_frame(
        [
            {
                "station_key": "US-COSTCO-120",
                "country": "US",
                "source_station_id": "120",
                "region": "HI",
                "last_seen_utc": "2026-09-14T18:17:00Z",
            },
            {
                "station_key": "US-COSTCO-1120",
                "country": "US",
                "source_station_id": "1120",
                "region": "DC",
                "last_seen_utc": "2026-09-14T18:17:00Z",
            },
        ]
    )
    ctx = make_ctx(ecom=ecom_ok(), previous=previous)
    polled = [p.source_station_id for p in us.polled_id_set(ctx)]
    assert len(us.batches(polled, us.batch_size(ctx))) == 2

    def responder(key, url):
        if "lookup" in key:
            return response(key, url, fixture="us_lookup_us.json", offset=30)
        return response(key, url, fixture="us_gasprices_batch_ua.json", offset=5)

    client = FakeClient(responder, abandon_after={"www.costco.com": 1})
    responses = us.fetch_us(client, ctx)
    keys = [r.key for r in responses]
    assert keys == ["US-COSTCO/02-gasprices-001", "US-COSTCO/03-lookup-us"]


def test_fallback_when_step_one_failed_and_there_is_no_cache():
    ctx = make_ctx(ecom=ecom_failed())
    client = FakeClient(fallback_responder)
    responses = us.fetch_us(client, ctx)
    assert [r.key for r in responses] == ["US-COSTCO/02-gasprices-001", "US-COSTCO/03-lookup-us"]


def test_force_fallback_skips_step_two_and_tops_up_the_rest():
    ctx = make_ctx(ecom=ecom_ok(), force_fallback={"US"})
    client = FakeClient(fallback_responder)
    responses = us.fetch_us(client, ctx)
    keys = [r.key for r in responses]
    assert keys == ["US-COSTCO/03-lookup-us", "US-COSTCO/04-gasprices-fb-001"]
    topup_url = next(c[1] for c in client.calls if c[0] == "US-COSTCO/04-gasprices-fb-001")
    assert us.ids_in_url(topup_url) == ["1793", "1765", "1772"]

    result = us.parse_us(responses, ctx)
    stations = by_id(result)
    assert result.source == "costco-ca-lookup-us"
    assert stations["1364"].id_origin == "lookup"
    assert stations["1793"].id_origin == "ecom"
    assert grades_of(stations["1793"]) == {"premium": "4.199", "regular": "3.499"}
    assert stations["1765"].id_origin == "extra"
    assert stations["1765"].name == "Chandler Business Center"
    assert stations["1772"].id_origin == "extra"
    assert "1" not in stations


def test_extras_skip_not_open_and_no_hours_in_the_lookup():
    ctx = make_ctx(ecom=ecom_ok(), force_fallback={"US"})
    client = FakeClient(fallback_responder)
    result = us.parse_us(us.fetch_us(client, ctx), ctx)
    camarillo = by_id(result)["1680"]
    assert camarillo.id_origin == "lookup"
    assert camarillo.opening_date is None
    assert camarillo.has_hours is None
    assert camarillo.lat == pytest.approx(34.218)
    assert grades_of(camarillo) == {"regular": "5.799", "premium": "6.099"}


# --------------------------------------------------------------------------- budgets


def test_budget_exhaustion_is_recorded_and_triggers_the_fallback():
    ctx = make_ctx(ecom=ecom_ok())

    def responder(key, url):
        if key == "US-COSTCO/02-gasprices-001":
            raise BudgetExceeded("country budget")
        if "lookup" in key:
            return response(key, url, fixture="us_lookup_us.json", offset=30)
        return response(key, url, body=TOPUP_BODY, offset=40)

    responses = us.fetch_us(FakeClient(responder), ctx)
    result = us.parse_us(responses, ctx)
    assert responses[0].error == "deadline_exceeded"
    assert "US-COSTCO/02-gasprices-001" in warning_details(result, "deadline_exceeded")
    assert "US-COSTCO/03-lookup-us" in [r.key for r in responses]
    assert result.errors == []


# --------------------------------------------------------------------------- protocol


def test_us_source_is_wired_to_the_module_functions():
    source = us.UsSource()
    assert source.country == "US"
    ctx = make_ctx(ecom=ecom_ok(), force_fallback={"US"})
    client = FakeClient(fallback_responder)
    responses = source.fetch(client, ctx)
    result = source.parse(responses, ctx)
    assert result.country == "US"
    assert result.requests == len(responses) == 2
    assert result.source == "costco-ca-lookup-us"
    assert len(result.stations) == 9


def test_lazy_registry_resolves_us_without_touching_base():
    assert SOURCES["US-COSTCO"].country == "US"
    ctx = make_ctx(ecom=ecom_ok(), force_fallback={"US"})
    client = FakeClient(fallback_responder)
    result = SOURCES["US-COSTCO"].parse(SOURCES["US-COSTCO"].fetch(client, ctx), ctx)
    assert result.country == "US"
    assert result.source == "costco-ca-lookup-us"
