"""Capture orchestration (spec 5.3)."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import tarfile
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.parse import urlencode

import polars as pl

from . import checks, schema
from .config import Config
from .fx import FxRates, fetch_rates
from .http import BudgetExceeded, Client
from .normalize import normalize
from .sources import us as us_source
from .sources.base import SOURCES, CaptureContext, RawResponse, Warning
from .store import ReleaseStore, sha256_file

# Paths are relative to the repository checkout the CLI runs in (capture.yml
# checks out the default branch, then runs from its root).
CONFIG_DIR = Path("config")
STATUS_PATH = Path("status/latest.json")

# The warehouse locator is shared by US and CA, so capture fetches it once
# (spec 4.2 step 1, 5.3 step 3). The client-identifier is a public id embedded
# in Costco's own front end; it lives in config/countries.toml under
# [countries.US] so it can be rotated without a code change, and these
# constants are only the built-in default.
ECOM_API_URL = "https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json"
ECOM_API_PARAMS = {"latitude": "0", "longitude": "0", "limit": "5000"}
ECOM_CLIENT_IDENTIFIER = "7c71124c-7bf1-44db-bc9d-498584cd66e5"

ECOM_BUDGET_S = 90.0
COUNTRY_BUDGET_S = 480.0
CAPTURE_BUDGET_S = 720.0

DROP_SCHEMA = {
    "source_station_id": pl.String,
    "reason": pl.String,
    "detail": pl.String,
}
US_ID_SET_SCHEMA = {
    "source_station_id": pl.String,
    "id_origin": pl.String,
    "ecom_state": pl.String,
}


@dataclass
class CaptureResult:
    status: dict
    all_failed: bool
    capture_id: str
    out: Path


def capture_id_for(now: datetime) -> str:
    """The capture id: UTC start time truncated to the minute (spec 6.1)."""
    return now.astimezone(timezone.utc).strftime("%Y-%m-%dT%H%MZ")


def github_output(key: str, value: str) -> None:
    """Append a step output when running under Actions; a no-op otherwise."""
    path = os.environ.get("GITHUB_OUTPUT")
    if not path:
        return
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(f"{key}={value}\n")


def _utc_text(value: datetime) -> str:
    return value.astimezone(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_response(directory: Path, resp: RawResponse, name: str) -> None:
    """Write one raw response as `<name>.body` plus `<name>.meta.json`."""
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{name}.body").write_bytes(resp.body)
    meta = {
        "key": resp.key,
        "url": resp.url,
        "status": resp.status,
        "headers": dict(resp.headers),
        "received_at_utc": _utc_text(resp.received_at_utc),
        "elapsed_ms": resp.elapsed_ms,
        "error": resp.error,
        "body_bytes": len(resp.body),
        "body_sha256": hashlib.sha256(resp.body).hexdigest(),
    }
    (directory / f"{name}.meta.json").write_text(
        json.dumps(meta, indent=1, ensure_ascii=False, sort_keys=True), encoding="utf-8"
    )
