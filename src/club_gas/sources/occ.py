"""Costco SAP Commerce (OCC) store endpoints: MX, GB, AU, JP and TW.

One fetcher serves all five countries; everything that differs between them is
either in ``config/countries.toml`` (URL, query parameters, region -> timezone
table) or in the small per-country helpers in this module.
"""

from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import Any
from urllib.parse import urlencode, urlsplit

from ..config import CountryConfig
from ..http import BudgetExceeded, Client
from .base import (
    CaptureContext,
    Error,
    FetchResult,
    RawPrice,
    RawResponse,
    RawStation,
    Warning,
)

SOURCE = "costco-occ"
MAX_PAGES = 20

AU_STATES = ("NSW", "VIC", "QLD", "WA", "SA", "TAS", "ACT", "NT")
AU_POSTCODE_STATES = {"2": "NSW", "3": "VIC", "4": "QLD", "5": "SA", "6": "WA", "7": "TAS"}
JP_PREFECTURE = re.compile(r"(北海道|東京都|(?:京都|大阪)府|\S{2,3}県)")
JP_MUNICIPALITY = re.compile(r"^(?:\S+?郡)?(\S+?[市区町村])(?![市区町村])")
TW_REGION = re.compile(r"^\d{3,5}\s*(\S{2}[市縣])")


def clean(value: Any) -> str | None:
    """Strip surrounding whitespace, including U+3000, and map "" to None.

    The storefront pads real values: the UK ``displayName`` "Sunbury ", the AU
    ``postalCode`` "6167 " and the JP ``line2`` " 埼玉県…". ``str.strip()``
    removes every character that ``str.isspace()`` accepts, and U+3000
    (IDEOGRAPHIC SPACE) is one of them, so no extra character class is needed.
    """
    if not isinstance(value, str):
        return None
    stripped = value.strip()
    return stripped or None


def page_url(cc: CountryConfig, page: int) -> str:
    """Build the URL for a zero-based page number.

    ``cc.params`` holds query parameters and nothing else, so it is urlencoded
    whole; the storefront's paging key ``currentPage`` is added here.
    """
    params = dict(cc.params)
    if page:
        params["currentPage"] = str(page)
    return f"{cc.url}?{urlencode(params)}"


def au_region_from_postcode(postcode: str | None) -> str | None:
    """Australian state from a 4-digit postcode (§4.4)."""
    if not postcode or len(postcode) != 4 or not postcode.isdigit():
        return None
    number = int(postcode)
    if 2600 <= number <= 2618 or 2900 <= number <= 2920:
        return "ACT"
    if postcode.startswith("08"):
        return "NT"
    return AU_POSTCODE_STATES.get(postcode[0])


def au_region(town: str | None, formatted: str | None, postcode: str | None) -> str | None:
    """First state token in ``town`` then ``formattedAddress``, else by postcode."""
    for text in (town, formatted):
        for token in re.findall(r"[A-Za-z]+", text or ""):
            if token.upper() in AU_STATES:
                return token.upper()
    return au_region_from_postcode(postcode)


def au_city(town: str | None) -> str | None:
    """The suburb: ``town`` with any trailing state token removed."""
    if not town:
        return None
    parts = town.split()
    while parts and parts[-1].upper() in AU_STATES:
        parts.pop()
    return " ".join(parts) or None


def jp_region(line2: str | None) -> str | None:
    """Prefecture at the start of the stripped ``line2``."""
    match = JP_PREFECTURE.match(line2 or "")
    return match.group(1) if match else None


def jp_city(line2: str | None) -> str | None:
    """Municipality that follows the prefecture.

    An optional ``…郡`` (district) is skipped first so that
    "福岡県糟屋郡久山町大字…" yields 久山町, and the regex still backtracks to
    the whole 小郡市 in "福岡県小郡市上岩田…". The negative lookahead keeps
    "石川県野々市市…" from stopping at 野々市.
    """
    prefecture = jp_region(line2)
    if prefecture is None:
        return None
    match = JP_MUNICIPALITY.match((line2 or "")[len(prefecture) :])
    return match.group(1) if match else None


def tw_region(formatted: str | None) -> str | None:
    """City or county after the leading postcode in ``formattedAddress``."""
    match = TW_REGION.match(formatted or "")
    return match.group(1) if match else None


class OccSource:
    """Source for one OCC country."""

    def __init__(self, country: str, brand: str = "COSTCO") -> None:
        self.country = country
        self.brand = brand
        # Every OCC country is Costco today, but the key prefix is the feed so
        # the bundle layout and the rebuild's grouping do not care.
        self.feed = f"{country}-{brand}"

    def fetch(self, client: Client, ctx: CaptureContext) -> list[RawResponse]:
        cc = ctx.fetch_config.countries[self.country]
        responses: list[RawResponse] = []
        page = 0
        for _ in range(MAX_PAGES):
            url = page_url(cc, page)
            if client.abandoned(url):
                break
            try:
                response = client.request(
                    f"{self.feed}/{page + 1:02d}-stores",
                    url,
                    headers={"Accept": "application/json"},
                    expect_json=True,
                )
            except BudgetExceeded:
                break
            responses.append(response)
            payload = _payload(response)
            if payload is None:
                break
            pagination = payload.get("pagination") or {}
            current = _as_int(pagination.get("currentPage"), page)
            total = _as_int(pagination.get("totalPages"), 1)
            # ``next_page <= page`` guards against a server that keeps echoing
            # the same currentPage, which would otherwise loop forever.
            next_page = current + 1
            if next_page >= total or next_page <= page:
                break
            page = next_page
        return responses

    def parse(self, responses: list[RawResponse], ctx: CaptureContext) -> FetchResult:
        cc = ctx.interp_config.countries[self.country]
        mine = sorted(
            (r for r in responses if r.key.startswith(f"{self.feed}/")),
            key=lambda r: r.key,
        )
        warnings: list[Warning] = []
        errors: list[Error] = []
        stations: list[RawStation] = []
        captured: datetime | None = None
        pagination: dict[str, Any] = {}

        for response in mine:
            payload = _payload(response)
            if payload is None:
                errors.append(
                    Error(
                        code="request_failed" if response.status is None else "http_error",
                        host=urlsplit(response.url).netloc,
                        http_status=response.status,
                        detail=response.error or f"{response.key}: unparseable body",
                    )
                )
                continue
            captured = (
                response.received_at_utc
                if captured is None
                else max(captured, response.received_at_utc)
            )
            pagination = payload.get("pagination") or pagination
            for store in payload.get("stores") or []:
                station = self._station(store, cc, warnings)
                if station is not None:
                    stations.append(station)

        current = _as_int(pagination.get("currentPage"), 0)
        total = _as_int(pagination.get("totalPages"), 1)
        if mine and current + 1 < total:
            warnings.append(
                Warning(
                    code="deadline_exceeded", detail=f"stopped after page {current + 1} of {total}"
                )
            )

        return FetchResult(
            country=self.country,
            source=SOURCE,
            captured_at_utc=captured or _capture_time(ctx),
            stations=stations,
            responses=list(responses),
            requests=len(mine),
            warnings=warnings,
            errors=errors,
        )

    def _station(
        self, store: dict[str, Any], cc: CountryConfig, warnings: list[Warning]
    ) -> RawStation | None:
        gas_types = store.get("gasTypes") or []
        if not gas_types:
            return None  # a store, not a gas station
        name = clean(store.get("name"))
        display_name = clean(store.get("displayName"))
        code = clean(store.get("warehouseCode"))
        address = store.get("address") or {}
        line1 = clean(address.get("line1"))
        line2 = clean(address.get("line2"))
        town = clean(address.get("town"))
        postal = clean(address.get("postalCode"))
        formatted = clean(address.get("formattedAddress"))
        geo = store.get("geoPoint") or {}

        number = _warehouse_number(code)
        if self.country in NUMERIC_ID_COUNTRIES:
            # Prefer the warehouse number: a store can be renamed, and the name
            # is the key every row of history is filed under.
            source_id = number or name
            alt_id = name
            if not number:
                warnings.append(
                    Warning(code="missing_warehouse_code", detail=display_name or name or "")
                )
        else:
            source_id = name
            alt_id = code
        if not source_id:
            warnings.append(Warning(code="missing_station_id", detail=display_name or code or ""))
            return None

        if self.country == "JP":
            city, region = jp_city(line2), jp_region(line2)
            postcode, street = line1, line2
        elif self.country == "TW":
            city, region = None, tw_region(formatted)
            postcode, street = postal, formatted or line1
        elif self.country == "AU":
            city, region = au_city(town), au_region(town, formatted, postal)
            postcode, street = postal, formatted or line1
        elif self.country == "GB":
            city, region = display_name, None
            postcode, street = postal, formatted or line1
        else:  # MX
            isocode = clean((address.get("region") or {}).get("isocode"))
            city = town
            region = isocode.removeprefix("MX-") if isocode else None
            postcode, street = postal, formatted or line1

        prices = tuple(
            RawPrice(grade_raw=label, price_raw=price)
            for label, price in ((clean(g.get("name")), clean(g.get("price"))) for g in gas_types)
            if label and price
        )
        return RawStation(
            source_station_id=source_id,
            alt_id=alt_id,
            id_origin="occ",
            ecom_state=None,
            name=name or display_name or source_id,
            name_local=display_name if self.country in ("JP", "TW") else None,
            address=street,
            city=city,
            region=region,
            postcode=postcode,
            lat=_as_float(geo.get("latitude")),
            lon=_as_float(geo.get("longitude")),
            # The country default is the "*" row of the table; resolving it is
            # CountryConfig's job, so this module never indexes cc.timezones.
            timezone=cc.timezone_for_region(region),
            opening_date=None,
            has_hours=None,
            prices=prices,
        )


# AU exposes the warehouse number bare ("109"); MX and TW bury it in a prefix
# ("costcoMexicoWharehouse750" -- Costco's own spelling). GB and JP are excluded
# on purpose: their codes are built from the store name ("coventry",
# "costcoJapanTomiyaWarehouse"), so a rename moves the code with it and keying
# on it would buy nothing.
NUMERIC_ID_COUNTRIES = ("AU", "MX", "TW")


def _warehouse_number(code: str | None) -> str | None:
    """The trailing warehouse number in an OCC `warehouseCode`, if it has one."""
    if not code:
        return None
    match = re.search(r"(\d+)$", code)
    return match.group(1) if match else None


def _as_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _capture_time(ctx: CaptureContext) -> datetime:
    return datetime.strptime(ctx.capture_id, "%Y-%m-%dT%H%MZ").replace(tzinfo=UTC)


def _payload(response: RawResponse) -> dict[str, Any] | None:
    """The decoded JSON object, or None when the response is unusable.

    Content-Type is deliberately not consulted: Costco's legacy endpoints serve
    valid JSON as ``text/html``, so only the status, the transport error and the
    body itself can say whether a response is usable.
    """
    if response.error or response.status != 200:
        return None
    try:
        payload = json.loads(response.body)
    except (ValueError, UnicodeDecodeError):
        return None
    return payload if isinstance(payload, dict) else None


def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default
