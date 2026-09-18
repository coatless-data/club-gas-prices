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

from club_gas.http import BudgetExceeded, perimeterx_challenge
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
HOST = "www.samsclub.com"

# A price response carrying one of these errors stands for a request that was
# never sent. `deadline_exceeded` is the marker fetch() leaves where the budget
# ran out; `host_abandoned` is the client declining a host it has given up on.
DEADLINE = "deadline_exceeded"
ABANDONED = "host_abandoned"
NOT_SENT = frozenset({DEADLINE, ABANDONED})

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

# `timeZone` is a US abbreviation, not an IANA zone, and the abbreviation alone
# is ambiguous: MST is Denver where DST is observed and Phoenix where it is not.
# The pair (timeZone, clubAttributes.isDSTObserved) resolves it. Measured across
# the 531 fuel clubs on 2026-09-17, exactly these six pairs occur.
#
# A state table cannot do this job: Florida, Indiana, Kentucky, Tennessee, South
# Dakota and Texas each have fuel clubs in two different zones.
IANA_BY_ZONE = {
    ("EST", True): "America/New_York",
    ("CST", True): "America/Chicago",
    ("MST", True): "America/Denver",
    ("PST", True): "America/Los_Angeles",
    ("MST", False): "America/Phoenix",
    ("HST", False): "Pacific/Honolulu",
    # Not seen with fuel today; Puerto Rico's clubs sell none. Here so a club
    # that gains a pump is not dropped for the want of one row.
    ("AST", False): "America/Puerto_Rico",
    ("AKST", True): "America/Anchorage",
}


def timezone_of(club: dict) -> str | None:
    """The IANA zone for a club, or None if the pair is one we have not seen."""
    abbreviation = club.get("timeZone")
    if not isinstance(abbreviation, str):
        return None
    observes_dst = bool((club.get("clubAttributes") or {}).get("isDSTObserved"))
    return IANA_BY_ZONE.get((abbreviation.strip().upper(), observes_dst))


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
        timezone=timezone_of(club),
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


def _below_regular(prices: tuple[RawPrice, ...]) -> tuple[tuple[RawPrice, ...], list[RawPrice]]:
    """Split off every grade priced below the same club's UNLEAD.

    Mid-grade cannot be cheaper than regular, yet club 6376 lists MIDGRAD at
    2.979 against an UNLEAD of 3.699, in the roster and in the Tempo record
    alike, and the figure has not moved while the other grades did. So any
    grade below the club's own UNLEAD is dropped. The caller warns about each
    one, because diesel can legitimately dip below regular, and if it ever does
    here the warning is where it shows.
    """
    regular = next((float(p.price_raw) for p in prices if p.grade_raw == "UNLEAD"), None)
    if regular is None:
        return prices, []
    kept = tuple(p for p in prices if float(p.price_raw) >= regular)
    return kept, [p for p in prices if float(p.price_raw) < regular]


def _position(key: str) -> int | None:
    """The 1-based place in the roster's fuel-club order a price key was sent for."""
    try:
        return int(key.rsplit("-", 1)[-1])
    except ValueError:
        return None


def _refusal(response: RawResponse) -> str | None:
    """What a failed response says about why, when it says anything."""
    if perimeterx_challenge(response.status, response.body):
        return "PerimeterX challenge"
    return response.error


def _price_errors(failed: list[RawResponse], sent: int) -> list[Error]:
    """One error per distinct failure, not one per club.

    A refused sweep fails every one of its ~531 requests in the same way, and
    the status keeps the errors of the last three failed captures, so an entry
    per request would bury the one fact that matters under hundreds of copies.
    """
    counts: dict[tuple[int | None, str | None], int] = {}
    for response in failed:
        reason = (response.status, _refusal(response))
        counts[reason] = counts.get(reason, 0) + 1
    return [
        Error(
            code="price_request_failed",
            host=HOST,
            http_status=status,
            detail=f"{count} of {sent} price requests" + (f": {why}" if why else ""),
        )
        for (status, why), count in counts.items()
    ]


def _stopped_response(key: str, url: str, at: datetime, error: str) -> RawResponse:
    return RawResponse(
        key=key,
        url=url,
        status=None,
        headers={},
        received_at_utc=at,
        elapsed_ms=0,
        body=b"",
        error=error,
    )


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
                # The client has given up on the host after repeated refusals.
                # Leave a marker like the deadline's, so parse() -- and a
                # rebuild -- can tell a sweep stopped by bot protection from
                # one that asked about every club.
                responses.append(
                    _stopped_response(
                        KEY_PRICES.format(n=n), url, responses[-1].received_at_utc, ABANDONED
                    )
                )
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
                # Whatever was collected still publishes. The marker goes into
                # the bundle with the rest, so parse() -- and a rebuild replaying
                # it -- can tell the clubs this sweep never asked about from the
                # ones that answered without a price.
                responses.append(
                    _stopped_response(
                        KEY_PRICES.format(n=n), url, responses[-1].received_at_utc, DEADLINE
                    )
                )
                break
        return responses

    def parse(self, responses: list[RawResponse], ctx: CaptureContext) -> FetchResult:
        captured_at = responses[0].received_at_utc if responses else _now(ctx)
        result = FetchResult(
            country=self.country,
            source=SOURCE,
            captured_at_utc=captured_at,
            brand=self.brand,
            requests=sum(1 for r in responses if r.error not in NOT_SENT),
        )
        roster_response = next((r for r in responses if r.key == KEY_ROSTER), None)
        roster = _payload(roster_response) if roster_response is not None else None
        if not isinstance(roster, list):
            result.errors.append(
                Error(
                    code="roster_unavailable",
                    host=HOST,
                    http_status=(roster_response.status if roster_response else None),
                    detail=(_refusal(roster_response) if roster_response else None),
                )
            )
            return result

        by_id = {str(c["id"]): c for c in roster if isinstance(c, dict) and c.get("id") is not None}
        ids = fuel_club_ids(roster)
        priced: dict[str, dict] = {}
        asked: set[str] = set()
        failed: list[RawResponse] = []
        sent = 0
        stopped_by: str | None = None
        for response in responses:
            if not response.key.startswith(f"{FEED}/02-"):
                continue
            if response.error in NOT_SENT:
                stopped_by = stopped_by or response.error
                continue
            sent += 1
            # fetch() numbers its requests by the club's place in this same
            # roster, so the key says which club was asked even when the answer
            # names none.
            position = _position(response.key)
            if position is not None and 0 < position <= len(ids):
                asked.add(ids[position - 1])
            payload = _payload(response)
            if payload is None:
                failed.append(response)
                continue
            block = store_fuel_prices(payload)
            if block is None:
                continue
            record_id = str(block.get("id") or "")
            # "CPF_FUELPRICE_PROD_6376" -> "6376". An unknown club is reported
            # as "INVALID STORE", which will not match and is dropped.
            club_id = record_id.rsplit("_", 1)[-1]
            if club_id in by_id:
                priced[club_id] = block
        result.errors.extend(_price_errors(failed, sent))

        unreached = 0
        for club_id in ids:
            if club_id not in asked and club_id not in priced:
                # Not the same as no_current_price: nothing was asked, so this
                # capture says nothing about the club, and publish leaves its
                # station's status as it was.
                result.warnings.append(Warning(code="not_reached", detail=club_id))
                unreached += 1
                continue
            prices, dropped = _below_regular(_prices(priced.get(club_id)))
            for price in dropped:
                result.warnings.append(
                    Warning(
                        code="below_regular",
                        detail=f"{club_id}:{price.grade_raw}={price.price_raw}",
                    )
                )
            if not prices:
                result.warnings.append(Warning(code="no_current_price", detail=club_id))
                continue
            station = _station(by_id[club_id], prices)
            if station is not None:
                result.stations.append(station)
        if stopped_by is not None:
            # Either way the rest of the roster went unasked, and either way
            # the feed is degraded (checks.DEGRADING_WARNINGS); the code says
            # which stopped it, because the remedies differ.
            result.warnings.append(
                Warning(
                    code="budget_exhausted" if stopped_by == DEADLINE else "sweep_abandoned",
                    detail=f"{unreached} of {len(ids)} fuel clubs not reached",
                )
            )
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
    "timezone_of",
]
