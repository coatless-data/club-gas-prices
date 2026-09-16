from __future__ import annotations

from datetime import date
from pathlib import Path

import httpx
import polars as pl
import pytest

from costco_gas.config import load_config
from costco_gas.http import Client
from costco_gas.sources.base import CaptureContext
from costco_gas.sources.occ import (
    OccSource,
    au_city,
    au_region,
    au_region_from_postcode,
    clean,
    jp_city,
    jp_region,
    tw_region,
)

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).parent / "fixtures"


def make_ctx(cfg, *, capture_date=date(2026, 9, 15)) -> CaptureContext:
    return CaptureContext(
        capture_id="2026-09-15T1910Z",
        capture_date=capture_date,
        fetch_config=cfg.fetch_view(),
        interp_config=cfg.interp_view(),
        previous_stations=pl.DataFrame(),
        previous_fx=pl.DataFrame(),
        previous_status=None,
        shared={},
        force_fallback=set(),
    )


def fetch_responses(country: str, bodies: list[bytes]):
    """Fetch one country, serving ``bodies`` in request order."""
    cfg = load_config(ROOT)
    ctx = make_ctx(cfg)
    seen: list[httpx.Request] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        body = bodies[min(len(seen) - 1, len(bodies) - 1)]
        return httpx.Response(200, content=body, headers={"content-type": "application/json"})

    client = Client(cfg.http, transport=httpx.MockTransport(handler))
    source = OccSource(country)
    return source, ctx, source.fetch(client, ctx), seen


def run(country: str, bodies: list[bytes]):
    """Fetch and parse one country, serving ``bodies`` in request order."""
    source, ctx, responses, seen = fetch_responses(country, bodies)
    return source.parse(responses, ctx), seen


def station(result, source_station_id):
    matches = [s for s in result.stations if s.source_station_id == source_station_id]
    assert matches, f"{source_station_id} not in {[s.source_station_id for s in result.stations]}"
    return matches[0]


def test_clean_strips_ideographic_space_and_empty_becomes_none():
    assert clean("Sunbury ") == "Sunbury"
    assert clean("6167 ") == "6167"
    assert clean(" 埼玉県三郷市新三郷ららシティ3-1-2 ") == "埼玉県三郷市新三郷ららシティ3-1-2"
    assert clean("\u3000Tomiya\u3000") == "Tomiya"
    assert clean("") is None
    assert clean(None) is None


def test_au_postcode_fallback_covers_the_act_ranges():
    assert au_region_from_postcode("2609") == "ACT"
    assert au_region_from_postcode("2600") == "ACT"
    assert au_region_from_postcode("2618") == "ACT"
    assert au_region_from_postcode("2900") == "ACT"
    assert au_region_from_postcode("2920") == "ACT"
    assert au_region_from_postcode("2619") == "NSW"
    assert au_region_from_postcode("2765") == "NSW"
    assert au_region_from_postcode("3194") == "VIC"
    assert au_region_from_postcode("4509") == "QLD"
    assert au_region_from_postcode("5084") == "SA"
    assert au_region_from_postcode("6105") == "WA"
    assert au_region_from_postcode("7000") == "TAS"
    assert au_region_from_postcode("0800") == "NT"
    assert au_region_from_postcode("") is None
    assert au_region_from_postcode("not a postcode") is None
    # A token anywhere in town or formattedAddress wins over the postcode.
    assert (
        au_region("Casuarina WA", "137 Market Street Casuarina WA Australia 6167", "6167") == "WA"
    )
    assert au_region(None, "8 Chifley Dr Moorabbin Airport VIC Australia 3194", "3194") == "VIC"
    assert au_city("Moorabbin Airport VIC") == "Moorabbin Airport"


def test_jp_helpers_handle_the_special_prefecture_names():
    assert jp_region("北海道石狩市新港南2丁目733番地１") == "北海道"  # noqa: RUF001
    assert jp_region("京都府八幡市欽明台北5番地") == "京都府"
    assert jp_region("大阪府和泉市あゆみ野4-4-45") == "大阪府"
    assert jp_region("神奈川県川崎市…") == "神奈川県"
    assert jp_region(None) is None
    assert jp_city("熊本県上益城郡御船町大字小坂字宮田689-1") == "御船町"
    assert jp_city("福岡県小郡市上岩田818-3") == "小郡市"
    assert jp_city("愛知県名古屋市守山区中志段味…") == "名古屋市"


def test_tw_region_is_read_after_the_leading_postcode():
    assert tw_region("320 桃園市中壢區民族路六段508號 (中壢店)") == "桃園市"
    assert tw_region("242 新北市新莊區建國一路138號 (新莊店)") == "新北市"
    assert tw_region(None) is None


def test_gb_uses_display_name_as_city_and_keeps_county_towns_out():
    result, seen = run("GB", [(FIXTURES / "gb_stores.json").read_bytes()])

    assert result.country == "GB"
    assert result.source == "costco-occ"
    assert seen[0].headers["accept"] == "application/json"
    assert "fields=FULL" in str(seen[0].url)
    # Hayes has an empty gasTypes list, so it is not a station at all.
    assert sorted(s.source_station_id for s in result.stations) == ["Haydock", "Reading", "Sunbury"]

    sunbury = station(result, "Sunbury")
    assert sunbury.city == "Sunbury"  # displayName "Sunbury " with its trailing space gone
    assert sunbury.alt_id == "sunbury"
    assert sunbury.region is None
    assert sunbury.timezone == "Europe/London"
    assert sunbury.name_local is None
    assert sunbury.postcode == "TW16 5LN"
    assert sunbury.id_origin == "occ"
    assert [(p.grade_raw, p.price_raw) for p in sunbury.prices] == [
        ("5301", "162.9"),
        ("5302", "174.9"),
        ("5303", "184.9"),
    ]
    # address.town is the county "Merseyside" for Haydock; displayName is the town.
    assert station(result, "Haydock").city == "Haydock"


def test_mx_region_drops_the_iso_prefix_and_uses_the_timezone_table():
    result, _ = run("MX", [(FIXTURES / "mx_stores.json").read_bytes()])

    assert sorted(s.source_station_id for s in result.stations) == [
        "Arboledas",
        "Chihuahua",
        "Culiacán",
        "Mexicali",
    ]
    mexicali = station(result, "Mexicali")
    assert mexicali.alt_id == "costcoMexicoWharehouse750"
    assert mexicali.city == "Mexicali"
    assert mexicali.region == "BCN"
    assert mexicali.timezone == "America/Tijuana"
    assert [(p.grade_raw, p.price_raw) for p in mexicali.prices] == [
        ("Regular", "$20.89"),
        ("Premium", "$25.39"),
    ]
    assert station(result, "Culiacán").timezone == "America/Mazatlan"
    assert station(result, "Chihuahua").timezone == "America/Chihuahua"
    # MEX has no entry of its own, so the table's "*" default applies.
    arboledas = station(result, "Arboledas")
    assert (arboledas.city, arboledas.region) == ("Tlalnepantla", "MEX")
    assert arboledas.timezone == "America/Mexico_City"


def test_au_uses_the_numeric_warehouse_code_as_the_id():
    result, _ = run("AU", [(FIXTURES / "au_stores.json").read_bytes()])

    assert sorted(s.source_station_id for s in result.stations) == ["103", "109", "116", "118"]
    perth = station(result, "116")
    assert perth.alt_id == "Perth Airport"
    assert perth.name == "Perth Airport"
    assert perth.city == "Perth Airport"
    assert perth.region == "WA"  # no state token anywhere; postcode 6105 decides
    assert perth.timezone == "Australia/Perth"
    assert perth.lat == pytest.approx(-31.94378)

    casuarina = station(result, "118")
    assert casuarina.postcode == "6167"  # the source sends "6167 "
    assert (casuarina.city, casuarina.region) == ("Casuarina", "WA")

    canberra = station(result, "103")
    assert (canberra.city, canberra.region) == ("Canberra Airport", "ACT")
    assert canberra.timezone == "Australia/Sydney"
    assert station(result, "109").city == "Marsden Park"


def test_jp_takes_the_prefecture_and_municipality_from_line2():
    result, _ = run("JP", [(FIXTURES / "jp_stores.json").read_bytes()])

    assert sorted(s.source_station_id for s in result.stations) == [
        "Hisayama",
        "Maebashi",
        "Nonoichi",
        "Shinmisato",
        "Tomiya",
    ]
    tomiya = station(result, "Tomiya")
    assert tomiya.alt_id == "costcoJapanTomiyaWarehouse"
    assert tomiya.name_local == "富谷"
    assert (tomiya.city, tomiya.region) == ("富谷市", "宮城県")
    assert tomiya.postcode == "981-3313"  # line1, not postalCode, for Japan
    assert tomiya.address == "宮城県富谷市高屋敷26"
    assert tomiya.timezone == "Asia/Tokyo"
    # Kerosene is listed first; grades are mapped by label later, never by position.
    assert [(p.grade_raw, p.price_raw) for p in tomiya.prices] == [
        ("Kerosene", "¥124"),
        ("Diesel", "¥135"),
        ("Regular", "¥149"),
        ("Premium", "¥159"),
    ]

    shinmisato = station(result, "Shinmisato")
    assert shinmisato.postcode == "341-0009"  # line1 arrives as " 341-0009"
    assert (shinmisato.city, shinmisato.region) == ("三郷市", "埼玉県")
    assert shinmisato.address == "埼玉県三郷市新三郷ららシティ3-1-2"
    assert station(result, "Maebashi").region == "群馬県"
    assert station(result, "Hisayama").city == "久山町"  # 福岡県糟屋郡久山町…
    assert station(result, "Nonoichi").city == "野々市市"  # not 野々市


def test_tw_region_comes_from_the_formatted_address():
    result, _ = run("TW", [(FIXTURES / "tw_stores.json").read_bytes()])

    assert sorted(s.source_station_id for s in result.stations) == [
        "Chungli",
        "North_Taichung",
        "Xinzhuang",
    ]
    chungli = station(result, "Chungli")
    assert chungli.alt_id == "costcoTaiwanWarehouse010"
    assert chungli.name_local == "桃園中壢店"
    assert chungli.city is None
    assert chungli.region == "桃園市"
    assert chungli.timezone == "Asia/Taipei"
    # Diesel is listed first here, Kerosene first in Japan: position means nothing.
    assert [(p.grade_raw, p.price_raw) for p in chungli.prices] == [
        ("Diesel", "$28.6"),
        ("95", "$30.0"),
        ("98", "$31.5"),
    ]
    assert station(result, "Xinzhuang").region == "新北市"
    assert station(result, "North_Taichung").region == "台中市"
