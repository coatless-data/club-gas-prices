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
