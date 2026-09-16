"""Turn a country's FetchResult into price rows and station rows (spec section 7)."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from datetime import date
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import polars as pl

from costco_gas.fx import FxRates
from costco_gas.schema import ROW_SCHEMA, STATION_SCHEMA
from costco_gas.sources.base import CaptureContext, FetchResult, RawStation, Warning

LITRES_PER_US_GALLON = 3.785411784

CURRENCY_BY_UNIT = {
    "USD/gal": "USD",
    "USD/L": "USD",
    "CAD/L": "CAD",
    "MXN/L": "MXN",
    "GBp/L": "GBP",
    "AUD/L": "AUD",
    "JPY/L": "JPY",
    "TWD/L": "TWD",
}

# Longest first, so "NT$" is removed before "$".
_CURRENCY_SYMBOLS = ("NT$", "$", "¥", "£", "€")


@dataclass(frozen=True)
class Drop:
    source_station_id: str
    reason: str
    detail: str = ""


@dataclass
class NormalizedCountry:
    rows: pl.DataFrame
    stations: pl.DataFrame
    drops: list[Drop] = field(default_factory=list)
    warnings: list[Warning] = field(default_factory=list)


def parse_price(price_raw: str) -> float | None:
    """Spec 7.2: strip whitespace, currency symbols and separators, then parse."""
    text = price_raw.strip()
    for symbol in _CURRENCY_SYMBOLS:
        text = text.replace(symbol, "")
    text = text.replace(",", "").replace(" ", "").replace("　", "").strip()
    if text.startswith("."):
        text = "0" + text
    try:
        value = float(text)
    except ValueError:
        return None
    if not math.isfinite(value):
        return None
    return value


def round_significant(value: float, digits: int) -> float:
    if value == 0.0 or not math.isfinite(value):
        return value
    return round(value, -math.floor(math.log10(abs(value))) + (digits - 1))


def local_per_litre(price: float, price_unit: str) -> float:
    """Spec 7.3."""
    if price_unit == "USD/gal":
        return price / LITRES_PER_US_GALLON
    if price_unit == "GBp/L":
        return price / 100.0
    return price


def row_source(country: str, id_origin: str, result_source: str) -> str:
    """Spec 6.1: the US costco.ca fallback and the CA price-service fallback
    label their rows differently from the country's primary source."""
    if country == "US" and id_origin == "lookup":
        return "costco-ca-lookup-us"
    if country == "CA" and id_origin == "cache":
        return "costco-ca-gasprices"
    return result_source
