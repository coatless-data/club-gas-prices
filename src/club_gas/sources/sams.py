"""Sam's Club (US): club roster from the locator sitemap, prices from the
server-rendered fuel-centre pages.

Sam's Club serves each club's current fuel prices on the page a member would
open, `/club/<id>/fuel-center`, embedded in the page's Next.js `__NEXT_DATA__`.
The `gasPrices` array on the JSON club-finder is stale -- 31-43% below the page
for the six clubs checked on 2026-09-17 -- so it is never read as a price. This
source reads two public pages, on paths robots.txt permits:

`sitemap_locators.xml` lists every club as `/club/<id>-<city>-<st>`. Deduplicated
by id it is the roster -- about 604 clubs. It says nothing about fuel; that is
the fuel-centre page's job.

`/club/<id>/fuel-center` is one club per request. A club that sells fuel answers
200 with `initialTempoData` (the `storeFuelPrices` record and the club's IANA
time zone) and `initialNodeDetail` (name, address, coordinates). A club that
does **not** sell fuel answers 307 to `/club/<id>` -- so with redirects left
unfollowed, a non-fuel club costs one empty response, not a 250 KB page. There
is no bulk fuel page; the sitemap does not enumerate fuel centres separately.

Neither page carries a usable price timestamp. `metadata.dateCreated` is a
batch-write stamp that does not move when the price moves. Capture time is the
only honest time, exactly as with Costco.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

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
SOURCE = "sams-fuel-center"
FEED = f"{COUNTRY}-{BRAND}"

KEY_SITEMAP = f"{FEED}/01-sitemap"
# The club id goes on the end, so parse() can recover it from the key even when
# the answer is a redirect that names no club.
KEY_CLUB = f"{FEED}/02-{{club_id}}"
_KEY_CLUB_PREFIX = f"{FEED}/02-"
HOST = "www.samsclub.com"

# A response carrying one of these errors stands for a request that was never
# sent. `deadline_exceeded` is fetch()'s marker where the budget ran out;
# `host_abandoned` is the client declining a host it has given up on.
DEADLINE = "deadline_exceeded"
ABANDONED = "host_abandoned"
NOT_SENT = frozenset({DEADLINE, ABANDONED})

# A club with no fuel centre 307s from /club/<id>/fuel-center to /club/<id>.
REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
# The fuel record an unknown club returns, in place of a real one.
INVALID_STORE = "INVALID STORE"

DEFAULT_SITEMAP_URL = "https://www.samsclub.com/sitemap_locators.xml"
# `{club_id}` is filled per club. The bare id resolves without a city slug.
DEFAULT_CLUB_URL = "https://www.samsclub.com/club/{club_id}/fuel-center"

_CLUB_ID_RE = re.compile(r"/club/(\d+)")
_NEXTDATA_RE = re.compile(rb'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', re.DOTALL)


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def _text(response: RawResponse | None) -> str | None:
    if response is None or response.status != 200 or not response.body:
        return None
    return response.body.decode("utf-8", "replace")


def club_url(base: str, club_id: str) -> str:
    return base.format(club_id=club_id)


def club_ids(sitemap_text: str | None) -> list[str]:
    """Every club id in the locator sitemap, deduplicated, in numeric order.

    The sitemap lists each club as `/club/<id>-<city>-<st>` and repeats some, so
    the ids are collected into a set and sorted by value. Ordering is numeric so
    a sweep cut short by its budget stops on a stable prefix, not a random one.
    """
    if not sitemap_text:
        return []
    seen = {m.group(1) for m in _CLUB_ID_RE.finditer(sitemap_text)}
    return sorted(seen, key=int)


def _nextdata(body: bytes) -> dict | None:
    """The parsed `__NEXT_DATA__` JSON a Sam's Club page embeds, or None."""
    if not body:
        return None
    match = _NEXTDATA_RE.search(body)
    if match is None:
        return None
    try:
        data = json.loads(match.group(1))
    except (ValueError, UnicodeDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _page_props(data: dict) -> dict:
    props = data.get("props") if isinstance(data, dict) else None
    page_props = props.get("pageProps") if isinstance(props, dict) else None
    return page_props if isinstance(page_props, dict) else {}


def store_fuel_prices(payload: Any) -> dict | None:
    """The FuelPrices module inside a Tempo content layout, if it is there.

    The layout is the same whether it arrives as a GraphQL response body or
    embedded in a page's `__NEXT_DATA__`, so the page path wraps its
    `initialTempoData` as `{"data": initialTempoData}` and reuses this walker.
    """
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


def fuel_price_block(page_props: dict) -> dict | None:
    """The storeFuelPrices record from a fuel-centre page's props."""
    tempo = page_props.get("initialTempoData")
    if not isinstance(tempo, dict):
        return None
    return store_fuel_prices({"data": tempo})


def iana_timezone(page_props: dict) -> str | None:
    """The club's IANA zone, read straight from the page.

    The fuel-centre page carries a real IANA name (`America/Chicago`) under the
    store-details module's capabilities -- not the ambiguous US abbreviation the
    REST roster gave, so no (abbreviation, isDSTObserved) disambiguation is
    needed. The first non-empty value wins; the seven capability rows agree.
    """
    tempo = page_props.get("initialTempoData")
    modules = (((tempo or {}).get("contentLayout") or {}).get("modules")) or []
    for module in modules:
        configs = module.get("configs") if isinstance(module, dict) else None
        details = configs.get("storeDetails") if isinstance(configs, dict) else None
        capabilities = details.get("capabilities") if isinstance(details, dict) else None
        if not isinstance(capabilities, list):
            continue
        for capability in capabilities:
            zone = capability.get("timeZone") if isinstance(capability, dict) else None
            if isinstance(zone, str) and zone.strip():
                return zone.strip()
    return None


def node_detail(page_props: dict) -> dict:
    """The club's node-detail object: name, address, coordinates, hours."""
    detail = page_props.get("initialNodeDetail")
    data = detail.get("data") if isinstance(detail, dict) else None
    node = data.get("nodeDetail") if isinstance(data, dict) else None
    return node if isinstance(node, dict) else {}


def _has_hours(node: dict) -> bool | None:
    hours = node.get("operationalHours")
    if not isinstance(hours, list):
        return None
    return any(isinstance(day, dict) and not day.get("closed") for day in hours)


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
    2.979 against an UNLEAD of 3.799, in the page's own record, and the figure
    has not moved while the other grades did. So any grade below the club's own
    UNLEAD is dropped. The caller warns about each one, because diesel can
    legitimately dip below regular, and if it ever does here the warning shows.
    """
    regular = next((float(p.price_raw) for p in prices if p.grade_raw == "UNLEAD"), None)
    if regular is None:
        return prices, []
    kept = tuple(p for p in prices if float(p.price_raw) >= regular)
    return kept, [p for p in prices if float(p.price_raw) < regular]


def _station(
    club_id: str, node: dict, timezone: str | None, prices: tuple[RawPrice, ...]
) -> RawStation:
    address = node.get("address") or {}
    point = node.get("geoPoint") or {}
    name = str(node.get("name") or node.get("displayName") or "").strip() or None
    return RawStation(
        source_station_id=str(node.get("id") or club_id),
        # Every club's fuel-centre page resolves from the id alone, with no city
        # slug, so the station page always gets a working link.
        alt_id=str(club_id),
        id_origin="occ",
        ecom_state=None,
        name=name,
        name_local=None,
        address=(address.get("addressLineOne") or None),
        city=(address.get("city") or None),
        region=(address.get("state") or None),
        postcode=(address.get("postalCode") or None),
        lat=_as_float(point.get("latitude")),
        lon=_as_float(point.get("longitude")),
        timezone=timezone,
        opening_date=None,
        has_hours=_has_hours(node),
        prices=prices,
    )


def _club_id_of_key(key: str) -> str | None:
    """The club id a price key was sent for."""
    if not key.startswith(_KEY_CLUB_PREFIX):
        return None
    club_id = key[len(_KEY_CLUB_PREFIX) :]
    return club_id if club_id.isdigit() else None


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
            code="club_request_failed",
            host=HOST,
            http_status=status,
            detail=f"{count} of {sent} club requests" + (f": {why}" if why else ""),
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
            sitemap = client.request(KEY_SITEMAP, cfg["sitemap_url"], expect_json=False)
        except BudgetExceeded:
            return responses
        responses.append(sitemap)

        for club_id in club_ids(_text(sitemap)):
            url = club_url(cfg["club_url"], club_id)
            key = KEY_CLUB.format(club_id=club_id)
            if client.abandoned(url):
                # The client has given up on the host after repeated refusals.
                # Leave a marker like the deadline's, so parse() -- and a rebuild
                # -- can tell a sweep stopped by bot protection from one that
                # asked about every club.
                responses.append(
                    _stopped_response(key, url, responses[-1].received_at_utc, ABANDONED)
                )
                break
            try:
                responses.append(
                    # Redirects are NOT followed: a non-fuel club 307s to its
                    # 250 KB club home, and following it would download that page
                    # for nothing. The bare 307 is the signal that the club has
                    # no fuel centre.
                    client.request(key, url, expect_json=False, follow_redirects=False)
                )
            except BudgetExceeded:
                responses.append(
                    _stopped_response(key, url, responses[-1].received_at_utc, DEADLINE)
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
        sitemap_response = next((r for r in responses if r.key == KEY_SITEMAP), None)
        ids = club_ids(_text(sitemap_response))
        if not ids:
            result.errors.append(
                Error(
                    code="sitemap_unavailable",
                    host=HOST,
                    http_status=(sitemap_response.status if sitemap_response else None),
                    detail=(_refusal(sitemap_response) if sitemap_response else None),
                )
            )
            return result

        asked: set[str] = set()
        priced: set[str] = set()
        failed: list[RawResponse] = []
        sent = 0
        stopped_by: str | None = None
        for response in responses:
            if not response.key.startswith(_KEY_CLUB_PREFIX):
                continue
            if response.error in NOT_SENT:
                stopped_by = stopped_by or response.error
                continue
            sent += 1
            club_id = _club_id_of_key(response.key)
            if club_id is not None:
                asked.add(club_id)
            # A non-fuel club redirects; the empty 307 is expected, not a fault.
            if response.status in REDIRECT_STATUSES:
                continue
            data = _nextdata(response.body) if response.status == 200 else None
            if data is None:
                failed.append(response)
                continue
            props = _page_props(data)
            block = fuel_price_block(props)
            # A club that does not sell fuel can still answer 200 with an
            # INVALID STORE record (a fuel-centre URL that resolved to the club
            # home without a redirect); treat it as non-fuel, not a failure.
            if block is None or str(block.get("id") or "") == INVALID_STORE:
                continue
            if club_id is None:
                # The page named a club the key did not; recover it from the
                # record id, e.g. "CPF_FUELPRICE_PROD_6376" -> "6376".
                club_id = str(block.get("id") or "").rsplit("_", 1)[-1] or None
            prices, dropped = _below_regular(_prices(block))
            for price in dropped:
                result.warnings.append(
                    Warning(
                        code="below_regular",
                        detail=f"{club_id}:{price.grade_raw}={price.price_raw}",
                    )
                )
            if not prices:
                if club_id is not None:
                    result.warnings.append(Warning(code="no_current_price", detail=club_id))
                continue
            result.stations.append(
                _station(club_id or "", node_detail(props), iana_timezone(props), prices)
            )
            if club_id is not None:
                priced.add(club_id)
        result.errors.extend(_price_errors(failed, sent))

        unreached = [c for c in ids if c not in asked and c not in priced]
        if stopped_by is not None:
            # Either way the rest of the roster went unasked, and either way the
            # feed is degraded (checks.DEGRADING_WARNINGS); the code says which
            # stopped it, because the remedies differ.
            for club_id in unreached:
                result.warnings.append(Warning(code="not_reached", detail=club_id))
            result.warnings.append(
                Warning(
                    code="budget_exhausted" if stopped_by == DEADLINE else "sweep_abandoned",
                    detail=f"{len(unreached)} of {len(ids)} clubs not reached",
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
        "sitemap_url": get("url", DEFAULT_SITEMAP_URL),
        "club_url": get("price_url", DEFAULT_CLUB_URL),
    }


__all__ = [
    "BRAND",
    "COUNTRY",
    "FEED",
    "SOURCE",
    "SamsSource",
    "club_ids",
    "club_url",
    "fuel_price_block",
    "iana_timezone",
    "node_detail",
    "store_fuel_prices",
]
