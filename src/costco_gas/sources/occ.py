"""Costco SAP Commerce (OCC) store endpoints: MX, GB, AU, JP and TW.

One fetcher serves all five countries; everything that differs between them is
either in ``config/countries.toml`` (URL, query parameters, region -> timezone
table) or in the small per-country helpers in this module.
"""

from __future__ import annotations

import re
from typing import Any
from urllib.parse import urlencode

from ..config import CountryConfig

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
