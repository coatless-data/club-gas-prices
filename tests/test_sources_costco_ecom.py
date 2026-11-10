"""The one reader for ecom-api's warehouse locator, in either response shape.

The capture reads ``salesLocations.json``; bundles captured before 2026-11 hold
``warehouses.json``, and ``rebuild`` and ``discover`` replay those, so both
shapes stay readable.
"""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest

from club_gas.sources import costco_ecom

FIXTURES = Path(__file__).parent / "fixtures"
# Trimmed from salesLocations.json as served on 2026-11-10.
SALES = json.loads((FIXTURES / "costco_sales_locations.json").read_text())
# Trimmed from warehouses.json as served on 2026-09-15.
WAREHOUSES = json.loads((FIXTURES / "us_ecom_warehouses.json").read_text())


def by_id(payload) -> dict[str, dict]:
    return dict(costco_ecom.locations(payload))


def test_locations_reads_the_sales_locations_shape():
    ids = [location_id for location_id, _ in costco_ecom.locations(SALES)]
    assert ids == ["1364", "140", "1808", "1772", "1799", "120", "729", "1790", "1324"]


def test_locations_still_reads_the_warehouses_shape():
    ids = {location_id for location_id, _ in costco_ecom.locations(WAREHOUSES)}
    assert {"1364", "140", "1838", "1324"} <= ids


@pytest.mark.parametrize(
    "payload",
    [None, [], "x", {}, {"warehouses": None}, {"salesLocations": {"1364": {}}}],
)
def test_locations_is_none_for_a_payload_with_no_list(payload):
    assert costco_ecom.locations(payload) is None


def test_locations_skips_entries_without_an_id():
    payload = {"salesLocations": [{"name": []}, "x", {"salesLocationId": " 17 "}]}
    assert costco_ecom.locations(payload) == [("17", {"salesLocationId": " 17 "})]


def test_fuel_opening_date_is_the_gas_service_date():
    """Franklin's gas opened on 2026-11-05, nineteen days before the warehouse.
    Gating on the warehouse date dropped the live station as not_open."""
    franklin = by_id(SALES)["1808"]
    assert franklin["openingDate"] == "2026-11-24"
    assert costco_ecom.fuel_opening_date(franklin) == date(2026, 11, 5)


def test_fuel_opening_date_falls_back_to_the_location_date():
    # The standalone gas station lists no services at all.
    assert costco_ecom.fuel_opening_date(by_id(SALES)["1772"]) == date(2026, 6, 24)
    # warehouses.json never carries a gas service date.
    lees_summit = by_id(WAREHOUSES)["1838"]
    assert costco_ecom.fuel_opening_date(lees_summit) == date(2026, 10, 2)


def test_fuel_opening_date_is_none_when_nothing_parses():
    assert costco_ecom.fuel_opening_date({"openingDate": "soon", "services": "x"}) is None


def test_a_standalone_gas_station_sells_gas_with_no_services_listed():
    mission_viejo = by_id(SALES)["1772"]
    assert mission_viejo["services"] == []
    assert costco_ecom.sells_gas(mission_viejo)


def test_sells_gas_reads_the_gas_service():
    sales = by_id(SALES)
    assert costco_ecom.sells_gas(sales["1364"])
    assert not costco_ecom.sells_gas(sales["120"])  # Hawaii Kai: no gas
    assert not costco_ecom.sells_gas(sales["729"])  # a Business Center without gas
    assert costco_ecom.sells_gas(by_id(WAREHOUSES)["1364"])
