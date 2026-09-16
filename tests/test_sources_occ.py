from __future__ import annotations

from costco_gas.sources.occ import (
    au_city,
    au_region,
    au_region_from_postcode,
    clean,
    jp_city,
    jp_region,
    tw_region,
)


def test_clean_strips_ideographic_space_and_empty_becomes_none():
    assert clean("Sunbury ") == "Sunbury"
    assert clean("6167 ") == "6167"
    assert clean(" 埼玉県三郷市新三郷ららシティ3-1-2 ") == "埼玉県三郷市新三郷ららシティ3-1-2"
    assert clean("　Tomiya　") == "Tomiya"
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
