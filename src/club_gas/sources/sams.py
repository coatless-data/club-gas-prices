"""Sam's Club (US): club roster from vivaldi, prices from the Tempo page query.

Two endpoints, because neither alone is enough.

`clubfinder/list` returns every US club in one request -- ids, coordinates,
address, and which grades each sells. It also returns a `gasPrices` array, and
**that array is stale**: measured on 2026-09-17 across six clubs in five states
it ran 31-43% below what the club's own page showed a member, against an AAA
national average of 4.37 USD. It is the roster, never the price.

Current prices come from `HyperLocalPagesTempo`, a persisted GraphQL query that
serves the fuel-centre page. It is one club per request -- `$nodeId` is typed
`String!`, so an array is rejected by the schema, and the only two operations
that touch fuel are both single-node. There is no bulk fuel endpoint; that was
established by probing rather than assumed.

Neither endpoint carries a usable price timestamp. `metadata.dateCreated` looks
like one and is inert: club 8248 moved 3.699 to 3.749 while its stamp stayed at
08:15:58.718Z. Capture time is the only honest time, exactly as with Costco.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from urllib.parse import quote

from club_gas.http import BudgetExceeded
from club_gas.sources.base import (
    CaptureContext,
    Error,
    FetchResult,
    RawPrice,
    RawResponse,
    RawStation,
    Warning,
)

COUNTRY = "US"
BRAND = "SAMS"
SOURCE = "sams-clubfinder"
FEED = f"{COUNTRY}-{BRAND}"

KEY_ROSTER = f"{FEED}/01-clubfinder"
KEY_PRICES = f"{FEED}/02-fuel-{{n:04d}}"

DEFAULT_ROSTER_URL = "https://www.samsclub.com/api/node/vivaldi/browse/v2/clubfinder/list"
DEFAULT_PRICE_URL = "https://www.samsclub.com/orchestra/home/graphql/HyperLocalPagesTempo"
# The persisted-query document hash. If Sam's re-registers the operation this
# changes and every price request 400s, which the floor check will catch.
PRICE_QUERY_SHA256 = "5d593ad32f0da732faa2c66ac5126cad60b1bf82901ecb59591b91ff62475237"

# Without this the gateway answers 400 with a CSRF message. Sending
# `content-type: application/json` instead produces a different 400 and does
# not work -- this specific header is the one that does.
PRICE_HEADERS = {
    "x-apollo-operation-name": "HyperLocalPagesTempo",
    "Accept": "application/json",
}

# `gasPrices[].name`, which is what the whole grade table is keyed on. The
# gradeId is deliberately not stored: it is a second lookup path for no gain.
GRADE_LABELS = ("UNLEAD", "MIDGRAD", "PREMIUM", "DIESEL", "MID CLR", "PREM CLR")

# The fuel centre inside `services`. REST calls it GAS; the GraphQL flavour of
# the same record calls it GAS_SAMS.
FUEL_SERVICE_NAMES = frozenset({"GAS", "GAS_SAMS"})

# `operationalHours` on a club is a near-useless default -- 596 of 601 clubs
# carry an identical 09:00-20:00 -- so only the fuel centre's own hours count.
_DAY_BUCKETS = ("monToFriHrs", "saturdayHrs", "sundayHrs")


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _payload(response: RawResponse) -> Any:
    if response.error is not None or response.status != 200 or not response.body:
        return None
    try:
        return json.loads(response.body)
    except (ValueError, UnicodeDecodeError):
        return None


def roster_url(url: str, origin_postcode: str) -> str:
    # Saturating both caps returns the whole country. They are real, enforced
    # caps, not decoration: nbrOfStores=50 returns exactly the nearest 50.
    cap = "2147483647"
    return f"{url}?singleLineAddr={quote(origin_postcode)}&nbrOfStores={cap}&distance={cap}"


def price_url(url: str, club_id: str, origin_postcode: str) -> str:
    variables = {
        "nodeId": str(club_id),
        "tenant": "SAMS_GLASS",
        "pageType": "HyperLocalPage",
        "pageId": "fuel-center",
        "layout": "",
        "locationInput": {
            "postalCode": origin_postcode,
            "storeCount": 5,
            "accessTypes": [],
            "nodeTypes": ["SAMS"],
        },
        "contentLayoutVersion": "v1",
    }
    encoded = quote(json.dumps(variables, separators=(",", ":")))
    return f"{url}/{PRICE_QUERY_SHA256}?variables={encoded}"


def fuel_club_ids(roster: Any) -> list[str]:
    """Clubs that sell fuel, by the only reliable signal.

    The `services` list carries a "gas" tag and it disagrees with `gasPrices` in
    both directions -- clubs 8132 and 4704 price fuel with no tag, club 7673
    tags it with no prices. Presence of a `gasPrices` array is the authority.
    """
    if not isinstance(roster, list):
        return []
    ids = []
    for club in roster:
        if isinstance(club, dict) and club.get("gasPrices") and club.get("id") is not None:
            ids.append(str(club["id"]))
    return ids


def _fuel_hours(club: dict) -> bool | None:
    """Whether the club's own record says the fuel centre has hours.

    `services` is a list of plain strings in the roster response and a list of
    objects in the per-club one, so both shapes are accepted.
    """
    services = club.get("services")
    if not isinstance(services, list) or not services:
        return None
    for entry in services:
        if isinstance(entry, str):
            if entry.lower() == "gas":
                return True
        elif isinstance(entry, dict) and entry.get("name") in FUEL_SERVICE_NAMES:
            return bool(entry.get("operationalHours"))
    return None


def _station(club: dict, prices: tuple[RawPrice, ...]) -> RawStation | None:
    club_id = club.get("id")
    if club_id is None:
        return None
    address = club.get("address") or {}
    point = club.get("geoPoint") or {}
    return RawStation(
        source_station_id=str(club_id),
        # Every club's fuel-centre page resolves from the id alone, with no
        # city slug, so the station page always gets a working link.
        alt_id=str(club_id),
        id_origin="occ",
        ecom_state=None,
        name=str(club.get("name") or "").strip() or None,
        name_local=None,
        address=(address.get("address1") or None),
        city=(address.get("city") or None),
        region=(address.get("state") or None),
        postcode=(address.get("postalCode") or None),
        lat=_as_float(point.get("latitude")),
        lon=_as_float(point.get("longitude")),
        # `timeZone` is a US abbreviation like "CST", not an IANA zone, so it is
        # resolved downstream from (timeZone, isDSTObserved). A state table gets
        # 17 fuel clubs wrong.
        timezone=None,
        opening_date=None,
        has_hours=_fuel_hours(club),
        prices=prices,
    )


def store_fuel_prices(payload: Any) -> dict | None:
    """The FuelPrices module inside a Tempo content layout, if it is there."""
    if not isinstance(payload, dict):
        return None
    modules = (((payload.get("data") or {}).get("contentLayout") or {}).get("modules")) or []
    for module in modules:
        if not isinstance(module, dict):
            continue
        configs = module.get("configs")
        if isinstance(configs, dict) and isinstance(configs.get("storeFuelPrices"), dict):
            return configs["storeFuelPrices"]
    return None


def _prices(block: dict | None) -> tuple[RawPrice, ...]:
    if not block:
        return ()
    out: list[RawPrice] = []
    for entry in block.get("prices") or []:
        if not isinstance(entry, dict) or entry.get("type") != "fuel":
            continue
        label = entry.get("name")
        value = _as_float(entry.get("price"))
        if not isinstance(label, str) or value is None or value <= 0:
            continue
        out.append(RawPrice(grade_raw=label, price_raw=f"{value:.3f}"))
    return tuple(out)


@dataclass
class SamsSource:
    country: str = COUNTRY
    brand: str = BRAND

    def fetch(self, client, ctx: CaptureContext) -> list[RawResponse]:
        cfg = _feed_cfg(ctx)
        responses: list[RawResponse] = []
        try:
            roster = client.request(
                KEY_ROSTER,
                roster_url(cfg["roster_url"], cfg["origin_postcode"]),
                headers={"Accept": "application/json"},
                expect_json=True,
            )
        except BudgetExceeded:
            return responses
        responses.append(roster)

        ids = fuel_club_ids(_payload(roster))
        for n, club_id in enumerate(ids, start=1):
            url = price_url(cfg["price_url"], club_id, cfg["origin_postcode"])
            if client.abandoned(url):
                break
            try:
                responses.append(
                    client.request(
                        KEY_PRICES.format(n=n),
                        url,
                        headers=PRICE_HEADERS,
                        expect_json=True,
                    )
                )
            except BudgetExceeded:
                # Whatever was collected still publishes; the floor check
                # decides whether a short sweep counts as degraded.
                break
        return responses

    def parse(self, responses: list[RawResponse], ctx: CaptureContext) -> FetchResult:
        captured_at = responses[0].received_at_utc if responses else _now(ctx)
        result = FetchResult(
            country=self.country,
            source=SOURCE,
            captured_at_utc=captured_at,
            brand=self.brand,
            requests=len(responses),
        )
        roster_response = next((r for r in responses if r.key == KEY_ROSTER), None)
        roster = _payload(roster_response) if roster_response is not None else None
        if not isinstance(roster, list):
            result.errors.append(
                Error(
                    code="roster_unavailable",
                    host="www.samsclub.com",
                    http_status=(roster_response.status if roster_response else None),
                )
            )
            return result

        by_id = {str(c["id"]): c for c in roster if isinstance(c, dict) and c.get("id") is not None}
        priced: dict[str, dict] = {}
        for response in responses:
            if not response.key.startswith(f"{FEED}/02-"):
                continue
            block = store_fuel_prices(_payload(response))
            if block is None:
                continue
            record_id = str(block.get("id") or "")
            # "CPF_FUELPRICE_PROD_6376" -> "6376". An unknown club is reported
            # as "INVALID STORE", which will not match and is dropped.
            club_id = record_id.rsplit("_", 1)[-1]
            if club_id in by_id:
                priced[club_id] = block

        for club_id in fuel_club_ids(roster):
            prices = _prices(priced.get(club_id))
            if not prices:
                result.warnings.append(Warning(code="no_current_price", detail=club_id))
                continue
            station = _station(by_id[club_id], prices)
            if station is not None:
                result.stations.append(station)
        return result


def _now(ctx: CaptureContext):
    return datetime.now(UTC)


def _feed_cfg(ctx: CaptureContext) -> dict:
    """Fetch settings for this feed, with the verified defaults as a fallback."""
    feeds = getattr(ctx.fetch_config, "feeds", None) or {}
    cfg = feeds.get(FEED) if isinstance(feeds, dict) else None
    get = (lambda k, d: getattr(cfg, k, None) or d) if cfg is not None else (lambda k, d: d)
    return {
        "roster_url": get("url", DEFAULT_ROSTER_URL),
        "price_url": get("price_url", DEFAULT_PRICE_URL),
        "origin_postcode": get("origin_postcode", "75001"),
    }


__all__ = [
    "BRAND",
    "COUNTRY",
    "FEED",
    "SOURCE",
    "SamsSource",
    "fuel_club_ids",
    "price_url",
    "roster_url",
    "store_fuel_prices",
]
