"""Exchange rates for a capture.

Order: Frankfurter v2 (status "ok"), then the fawazahmed0 currency-api at an
explicit date (status "fallback"), then carry-forward from the previous capture's
fx.csv (status "carried-forward"). If nothing works the capture still succeeds
with no rows (status "failed") and null USD columns.

`units_per_usd` is the provider's raw rate (quote units per 1 USD) and is stored
unrounded; rows use `fx_usd_per_unit = 1 / units_per_usd`.

The bundle's fx.json is a JSON *array* of rate rows and carries no status, so
`to_json()` returns a list and `from_json()` takes the status as a keyword
argument (a rebuild reads it from the same bundle's status.json `fx.status`).
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import UTC, date, datetime, timedelta
from typing import TYPE_CHECKING, Any

from costco_gas.http import BudgetExceeded

if TYPE_CHECKING:
    from costco_gas.http import Client
    from costco_gas.sources.base import CaptureContext, RawResponse

CURRENCIES: tuple[str, ...] = ("CAD", "MXN", "GBP", "AUD", "JPY", "TWD")
FX_BUDGET_SECONDS = 90.0
FRANKFURTER_URL = "https://api.frankfurter.dev/v2/rates?base=USD&quotes=" + ",".join(CURRENCIES)
FAWAZ_URL = (
    "https://cdn.jsdelivr.net/npm/@fawazahmed0/currency-api@{day}"
    "/v1/currencies/usd.json"
)
SOURCE_FRANKFURTER = "frankfurter-v2"
SOURCE_FAWAZ = "fawazahmed0-currency-api"
SOURCE_IDENTITY = "identity"
CARRIED_PREFIX = "carried-forward:"
CARRY_FORWARD_MAX_AGE_DAYS = 7


@dataclass(frozen=True)
class FxRow:
    currency: str
    units_per_usd: float
    fx_usd_per_unit: float
    fx_rate_date: date | None
    fx_source: str
    fx_fetched_at_utc: datetime | None


@dataclass(frozen=True)
class FxRates:
    status: str
    rows: list[FxRow]
    capture_date: date | None = None

    def for_currency(self, code: str) -> FxRow | None:
        wanted = str(code).upper()
        if wanted == "USD":
            return FxRow(
                currency="USD",
                units_per_usd=1.0,
                fx_usd_per_unit=1.0,
                fx_rate_date=self.capture_date,
                fx_source=SOURCE_IDENTITY,
                fx_fetched_at_utc=None,
            )
        for row in self.rows:
            if row.currency == wanted:
                return row
        return None

    def to_json(self) -> list[dict[str, Any]]:
        """The contents of the bundle's fx.json: a JSON array, non-USD rows only."""
        return [
            {
                "currency": row.currency,
                "units_per_usd": row.units_per_usd,
                "fx_rate_date": (row.fx_rate_date.isoformat() if row.fx_rate_date else None),
                "fx_source": row.fx_source,
                "fx_fetched_at_utc": (
                    _iso_utc(row.fx_fetched_at_utc) if row.fx_fetched_at_utc else None
                ),
            }
            for row in self.rows
        ]

    @classmethod
    def from_json(
        cls,
        rows: list[dict[str, Any]],
        *,
        status: str,
        capture_date: date | None = None,
    ) -> FxRates:
        """Rebuild rates from a stored fx.json array, for rebuilds of old captures.

        `rows` is the array exactly as `to_json()` wrote it. `status` is not stored
        in fx.json, so the caller supplies it (rebuild.py takes it from the same
        bundle's status.json `fx.status`). `capture_date` is optional and only sets
        the date on the synthetic USD identity row.
        """
        out: list[FxRow] = []
        for item in rows or []:
            rate = _positive_float(item.get("units_per_usd"))
            if rate is None:
                continue
            out.append(
                FxRow(
                    currency=str(item.get("currency", "")).upper(),
                    units_per_usd=rate,
                    fx_usd_per_unit=1.0 / rate,
                    fx_rate_date=_parse_date(item.get("fx_rate_date")),
                    fx_source=str(item.get("fx_source", "")),
                    fx_fetched_at_utc=_parse_datetime(item.get("fx_fetched_at_utc")),
                )
            )
        return cls(status=status, rows=out, capture_date=capture_date)


def _iso_utc(value: datetime) -> str:
    aware = value if value.tzinfo else value.replace(tzinfo=UTC)
    return aware.astimezone(UTC).isoformat().replace("+00:00", "Z")


def _positive_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(number) or number <= 0:
        return None
    return number


def _parse_date(value: Any) -> date | None:
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value
    if isinstance(value, str):
        try:
            return date.fromisoformat(value[:10])
        except ValueError:
            return None
    return None


def _parse_datetime(value: Any) -> datetime | None:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError:
            return None
        return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
    return None


def _missing(found: dict[str, FxRow]) -> list[str]:
    return [code for code in CURRENCIES if code not in found]


def _fetch_json(client: Client, key: str, url: str) -> tuple[Any, RawResponse] | None:
    """One JSON GET. Returns None for anything that is not a usable 200."""
    if client.abandoned(url):
        return None
    try:
        response = client.request(key, url, profile="default", expect_json=True)
    except BudgetExceeded:
        raise
    except Exception:
        # Client already retried; any remaining failure just means "try the next
        # source". FX must never fail the capture.
        return None
    if response.error or response.status != 200:
        return None
    try:
        return json.loads(response.body.decode("utf-8")), response
    except (UnicodeDecodeError, json.JSONDecodeError):
        return None


def _collect_frankfurter(client: Client, found: dict[str, FxRow]) -> None:
    got = _fetch_json(client, "fx/frankfurter", FRANKFURTER_URL)
    if got is None:
        return
    payload, response = got
    if not isinstance(payload, list):
        return
    for item in payload:
        if not isinstance(item, dict):
            continue
        currency = str(item.get("quote", "")).upper()
        if currency not in CURRENCIES or currency in found:
            continue
        if str(item.get("base", "USD")).upper() != "USD":
            continue
        rate = _positive_float(item.get("rate"))
        rate_date = _parse_date(item.get("date"))
        if rate is None or rate_date is None:
            continue
        found[currency] = FxRow(
            currency=currency,
            units_per_usd=rate,
            fx_usd_per_unit=1.0 / rate,
            fx_rate_date=rate_date,
            fx_source=SOURCE_FRANKFURTER,
            fx_fetched_at_utc=response.received_at_utc,
        )


def _collect_fawaz(client: Client, capture_date: date, found: dict[str, FxRow]) -> None:
    """The capture's UTC date first, then the previous day. Never @latest."""
    for offset in (0, 1):
        day = capture_date - timedelta(days=offset)
        stamp = day.isoformat()
        got = _fetch_json(client, f"fx/fawazahmed0-{stamp}", FAWAZ_URL.format(day=stamp))
        if got is None:
            continue
        payload, response = got
        if not isinstance(payload, dict):
            continue
        quotes = payload.get("usd")
        if not isinstance(quotes, dict):
            continue
        rate_date = _parse_date(payload.get("date")) or day
        for currency in CURRENCIES:
            if currency in found:
                continue
            rate = _positive_float(quotes.get(currency.lower()))
            if rate is None:
                continue
            found[currency] = FxRow(
                currency=currency,
                units_per_usd=rate,
                fx_usd_per_unit=1.0 / rate,
                fx_rate_date=rate_date,
                fx_source=SOURCE_FAWAZ,
                fx_fetched_at_utc=response.received_at_utc,
            )
        if not _missing(found):
            return


def _status_for(rows: list[FxRow]) -> str:
    """The status names the least-fresh source that contributed a row."""
    if not rows:
        return "failed"
    if any(row.fx_source.startswith(CARRIED_PREFIX) for row in rows):
        return "carried-forward"
    if any(row.fx_source == SOURCE_FAWAZ for row in rows):
        return "fallback"
    return "ok"


def fetch_rates(client: Client, ctx: CaptureContext) -> FxRates:
    found: dict[str, FxRow] = {}
    try:
        with client.budget("fx", FX_BUDGET_SECONDS):
            _collect_frankfurter(client, found)
            if _missing(found):
                _collect_fawaz(client, ctx.capture_date, found)
    except BudgetExceeded:
        pass
    rows = [found[code] for code in CURRENCIES if code in found]
    return FxRates(status=_status_for(rows), rows=rows, capture_date=ctx.capture_date)
