"""One reader for ecom-api's warehouse locator, shared by US, CA and discover.

The capture reads ``salesLocations.json``. It lists everything
``warehouses.json`` does, plus the sites that file leaves out: Costco's
standalone gas stations (subType ``Standalone Gas Station``, #1772 Mission
Viejo) and warehouses months before they open. It also dates each service, and
the gas service usually opens weeks before its warehouse: on 2026-11-10,
622 of 708 gas dates differed from the warehouse's, and Franklin (#1808) had
been selling fuel for five days with nineteen to go before its warehouse
opened. Gating a station on the warehouse's date therefore dropped live prices
as ``not_open`` at every new site.

Bundles captured before the switch hold ``warehouses.json`` -- a
``warehouses`` list keyed by ``warehouseId``, with no service dates -- and
``rebuild`` and ``discover`` replay them, so both shapes are read.
"""

from __future__ import annotations

from datetime import date
from typing import Any

# (list key, id key) per response shape, newest first.
SHAPES = (("salesLocations", "salesLocationId"), ("warehouses", "warehouseId"))
STANDALONE_GAS = "Standalone Gas Station"


def _clean(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _iso_date(value: Any) -> date | None:
    text = _clean(value)
    if text is None:
        return None
    try:
        return date.fromisoformat(text[:10])
    except ValueError:
        return None


def locations(payload: Any) -> list[tuple[str, dict[str, Any]]] | None:
    """``(id, location)`` for every location in either response shape, or
    ``None`` when the payload holds no list of locations at all."""
    if not isinstance(payload, dict):
        return None
    for list_key, id_key in SHAPES:
        items = payload.get(list_key)
        if not isinstance(items, list):
            continue
        out = []
        for item in items:
            if isinstance(item, dict) and (location_id := _clean(item.get(id_key))):
                out.append((location_id, item))
        return out
    return None


def _services(location: dict[str, Any]) -> list[dict[str, Any]]:
    services = location.get("services")
    if not isinstance(services, list):
        return []
    return [service for service in services if isinstance(service, dict)]


def _gas_services(location: dict[str, Any]) -> list[dict[str, Any]]:
    return [s for s in _services(location) if _clean(s.get("code")) == "gas"]


def sells_gas(location: dict[str, Any]) -> bool:
    """A gas service, or a standalone gas station, which lists no services."""
    subtype = location.get("subType")
    if isinstance(subtype, dict) and _clean(subtype.get("code")) == STANDALONE_GAS:
        return True
    return bool(_gas_services(location))


def fuel_opening_date(location: dict[str, Any]) -> date | None:
    """When the location starts selling fuel: its gas service's own date when
    it has one, otherwise the location's."""
    for service in _gas_services(location):
        opened = _iso_date(service.get("openingDate"))
        if opened is not None:
            return opened
    return _iso_date(location.get("openingDate"))


__all__ = ["STANDALONE_GAS", "fuel_opening_date", "locations", "sells_gas"]
