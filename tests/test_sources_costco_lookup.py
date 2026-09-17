"""The one parser for costco.ca's lookup, and the divergences it closed.

The US and CA sources both read this endpoint and each had its own parser. Every
case below is one they actually disagreed on, measured before the extraction.
"""

from __future__ import annotations

from datetime import date

import pytest

from club_gas.sources import ca, costco_lookup, us


@pytest.mark.parametrize(
    "text",
    ["Aug 23, 1995", "AUG 23, 1995", "aug 23, 1995", "Aug. 23 1995", "August 23, 1995"],
)
def test_open_date_reads_any_casing(text):
    """The US table was title-cased and never normalised, so an uppercase month
    parsed as None -- and a not-yet-open warehouse then looked open and skipped
    the not_open filter."""
    assert costco_lookup.open_date(text) == date(1995, 8, 23)
    assert us.lookup_open_date(text) == date(1995, 8, 23)
    assert ca.parse_open_date(text) == date(1995, 8, 23)


@pytest.mark.parametrize("bad", ["", "Smarch 4, 2020", "Aug 32, 1995", "1995-08-23", None, 17])
def test_open_date_rejects_what_is_not_one(bad):
    assert costco_lookup.open_date(bad) is None


def test_row_identifiers_are_never_grades_whatever_their_casing():
    """CA did not lowercase before testing, so it emitted warehouseID and OID as
    grades. Those raise unknown_grade, which is in checks.DEGRADING_WARNINGS, so
    a casing change at Costco would have marked Canada degraded every capture."""
    row = {"warehouseID": "1364", "OID": "x", "WarehouseId": "1", "regular": "3.999"}

    assert costco_lookup.prices(row) == {"regular": "3.999"}
    assert us.lookup_prices({"gasPrices": row}) == {"regular": "3.999"}
    assert [p.grade_raw for p in ca._prices(row)] == ["regular"]


def test_a_numeric_price_is_kept_not_dropped():
    """CA accepted only str, so a JSON number silently lost a real price."""
    row = {"regular": "3.999", "premium": 4.299, "diesel": 4}

    assert costco_lookup.prices(row) == {"regular": "3.999", "premium": "4.299", "diesel": "4"}
    assert {p.grade_raw: p.price_raw for p in ca._prices(row)} == {
        "regular": "3.999",
        "premium": "4.299",
        "diesel": "4",
    }


def test_a_boolean_is_not_a_price():
    """bool is a subclass of int, so the numeric branch has to exclude it."""
    assert costco_lookup.prices({"regular": True, "premium": "3.1"}) == {"premium": "3.1"}


def test_the_two_sources_agree_on_every_case_above():
    """The point of the extraction: one implementation, so they cannot drift."""
    assert us.NON_GRADE_KEYS is costco_lookup.NON_GRADE_KEYS
    assert ca.NON_GRADE_KEYS is costco_lookup.NON_GRADE_KEYS


def test_warehouses_skips_the_leading_false_and_demands_an_array():
    body = b'\r\n[false, {"stlocID": "1364"}, {"nope": 1}, {"stlocID": "140"}]'
    assert [w["stlocID"] for w in costco_lookup.warehouses(body)] == ["1364", "140"]
    with pytest.raises(ValueError, match="not a JSON array"):
        costco_lookup.warehouses(b'{"stlocID": "1364"}')
