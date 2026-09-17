"""One parser for costco.ca's AjaxWarehouseBrowseLookupView.

Both the US and CA sources read this endpoint -- the same host, the same JSON
shape, differing only in the `countryCode` parameter -- and each had grown its
own parser. The two had already diverged on real inputs:

- `"AUG 23, 1995"`: the US table was title-cased and never normalised case, so
  it returned None while CA returned the date. A not-yet-open warehouse then
  looked open and skipped the `not_open` filter.
- `{"warehouseID": ..., "OID": ...}`: the US lowercased the key before testing
  it against the non-grade set, CA did not. CA therefore emitted both as grades,
  raising `unknown_grade`, which is in `checks.DEGRADING_WARNINGS` -- so a
  casing change at Costco would have marked Canada degraded on every capture.

Each function below takes the more permissive of the two behaviours, which is
the one that keeps working when Costco changes a detail.
"""

from __future__ import annotations

import json
import re
from datetime import date
from typing import Any

# `warehouseid` and `oid` are row identifiers that sit in the same dict as the
# grades. Matched case-insensitively, because the feed has used both spellings.
NON_GRADE_KEYS = frozenset({"warehouseid", "oid"})

# strptime("%b") reads month names from the process locale, so a runner with a
# non-English LC_TIME would silently fail; this table does not.
MONTHS = {
    "jan": 1,
    "feb": 2,
    "mar": 3,
    "apr": 4,
    "may": 5,
    "jun": 6,
    "jul": 7,
    "aug": 8,
    "sep": 9,
    "oct": 10,
    "nov": 11,
    "dec": 12,
}
_OPEN_DATE = re.compile(r"^\s*([A-Za-z]{3,})\.?\s+(\d{1,2})\s*,?\s*(\d{4})\s*$")


def open_date(value: Any) -> date | None:
    """Parse the lookup's `"Aug 23, 1995"` opening dates, in any casing."""
    if not isinstance(value, str):
        return None
    match = _OPEN_DATE.match(value)
    if match is None:
        return None
    month = MONTHS.get(match.group(1)[:3].lower())
    if month is None:
        return None
    try:
        return date(int(match.group(3)), month, int(match.group(2)))
    except ValueError:
        return None


def is_grade_key(key: Any) -> bool:
    """Whether a key in the `gasPrices` dict names a fuel grade."""
    return str(key).lower() not in NON_GRADE_KEYS


def prices(grades: Any) -> dict[str, str]:
    """Grade label to price text, dropping the row identifiers.

    Numeric values are accepted and stringified: the feed has served both
    `"4.299"` and `4.299`, and dropping the numeric form loses a real price.
    """
    if not isinstance(grades, dict):
        return {}

    def usable(value: Any) -> bool:
        return isinstance(value, (str, int, float)) and not isinstance(value, bool)

    return {
        str(key): str(value) for key, value in grades.items() if is_grade_key(key) and usable(value)
    }


def warehouses(body: bytes) -> list[dict[str, Any]]:
    """Warehouse objects from a lookup body.

    The body starts with a CRLF and its element 0 is the boolean `false`; the
    warehouses are elements 1..n. Raises ValueError when the body is not the
    expected JSON array, which is one of the fallback triggers. The response's
    Content-Type is `text/html` even when the body is valid JSON, so it can
    never be used to tell a block from a good answer.
    """
    payload = json.loads(body.decode("utf-8", errors="strict"))
    if not isinstance(payload, list):
        raise ValueError("lookup body is not a JSON array")
    return [e for e in payload if isinstance(e, dict) and "stlocID" in e]


__all__ = ["MONTHS", "NON_GRADE_KEYS", "is_grade_key", "open_date", "prices", "warehouses"]
