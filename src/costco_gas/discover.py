"""Monthly US station discovery (spec 10.4).

Costco's metadata API does not list every station that sells fuel, and
relocations change warehouse ids. Once a month this command sweeps the legacy
price endpoint over the ids nobody polls and reports the ones that answer with
a plausible US price, so a maintainer can add the real ones to
``config/us_extra_ids.csv``.

Everything except the price sweep is read from a single capture bundle, so
discovery makes no metadata requests of its own.
"""

from __future__ import annotations

import io
import json
import re
import tarfile
import tempfile
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from .issues import Issues
from .store import AssetNotFound, ReleaseStore, StorageError

US_PRICE_URL = "https://www.costco.com/AjaxGetGasPricesService"
US_PRICE_UNIT = "USD/gal"
BATCH = 10  # the endpoint processes only the first 10 ids of a request
LOOKAHEAD = 200
# Used only when the config is missing a US entry; config/countries.toml holds
# the real values under [countries.US.bounds."USD/gal"].
USD_GAL_MIN = 2.00
USD_GAL_MAX = 11.00
PRE_OPEN_REGULAR = 3.999
PRE_OPEN_PREMIUM = 5.999
ID_COLUMNS = ("source_station_id", "warehouse_id", "station_id", "id")
STLOC_ID = re.compile(rb'"stlocID"\s*:\s*"?(\d+)')


@dataclass(frozen=True)
class DiscoverResult:
    candidates: list[dict]
    skipped_reason: str | None = None


@dataclass(frozen=True)
class BundleInputs:
    capture_id: str
    polled_ids: set[int]
    us_ecom_ids: set[int]
    ca_ecom_ids: set[int]
    ca_lookup_ids: set[int]
    station_ids: set[int]


def _month_tags(now: datetime) -> list[str]:
    first = now.date().replace(day=1)
    previous = (first - timedelta(days=1)).replace(day=1)
    return [f"data-{first:%Y-%m}", f"data-{previous:%Y-%m}"]


def _manifest_entries(manifest: dict) -> dict[str, dict]:
    """Accept either shape the month manifest may use: a mapping of capture id
    to entry, or a list of entries carrying their own ``capture_id``."""
    captures = manifest.get("captures", manifest)
    if isinstance(captures, list):
        return {
            entry["capture_id"]: entry
            for entry in captures
            if isinstance(entry, dict) and entry.get("capture_id")
        }
    if isinstance(captures, dict):
        return {key: value for key, value in captures.items() if isinstance(value, dict)}
    return {}


def _entry_status(entry: dict) -> dict:
    for key in ("status", "status_json"):
        if isinstance(entry.get(key), dict):
            return entry[key]
    return entry if "countries" in entry else {}


def _us_status(entry: dict) -> str | None:
    countries = _entry_status(entry).get("countries") or {}
    return (countries.get("US") or {}).get("status")


def select_bundle(store: ReleaseStore, *, now: datetime) -> tuple[str, str] | None:
    """Return ``(tag, asset name)`` of the newest uploaded capture bundle whose
    manifest entry shows US ``ok`` or ``degraded``, this month first."""
    for tag in _month_tags(now):
        if store.get_release(tag) is None:
            continue
        month = tag.removeprefix("data-")
        try:
            with tempfile.TemporaryDirectory() as workdir:
                name = f"manifest-{month}.json"
                path = store.read_resolved(tag, name, Path(workdir) / name)
                manifest = json.loads(Path(path).read_text(encoding="utf-8"))
            uploaded = {asset.name for asset in store.list_assets(tag) if asset.state == "uploaded"}
        except (AssetNotFound, StorageError, ValueError) as exc:
            print(f"::warning::could not read the manifest of {tag}: {exc}")
            continue
        best: tuple[str, str] | None = None
        for capture_id, entry in _manifest_entries(manifest).items():
            if _us_status(entry) not in ("ok", "degraded"):
                continue
            asset = f"capture-{capture_id}.tar.gz"
            if asset not in uploaded:
                continue
            if best is None or capture_id > best[0]:
                best = (capture_id, asset)
        if best is not None:
            return tag, best[1]
    return None


def read_bundle(store: ReleaseStore, tag: str, name: str) -> BundleInputs:
    with tempfile.TemporaryDirectory() as workdir:
        path = store.read_resolved(tag, name, Path(workdir) / name)
        return _read_bundle_file(Path(path))


def _read_bundle_file(path: Path) -> BundleInputs:
    capture_id = ""
    polled: set[int] = set()
    stations: set[int] = set()
    us_ecom: set[int] = set()
    ca_ecom: set[int] = set()
    ca_lookup: set[int] = set()
    with tarfile.open(path, "r:gz") as tar:
        for member in tar.getmembers():
            if not member.isfile():
                continue
            # Match by suffix so a leading directory inside the tar is harmless.
            name = "/" + member.name
            handle = tar.extractfile(member)
            if handle is None:
                continue
            if name.endswith("/capture.json"):
                try:
                    capture_id = str(json.loads(handle.read()).get("capture_id", ""))
                except ValueError:
                    capture_id = ""
            elif name.endswith("/inputs/us_id_set.csv"):
                polled |= _ids_from_csv(handle.read())
            elif name.endswith("/inputs/stations_used.csv"):
                stations |= _ids_from_csv(handle.read(), countries={"US", "CA"})
            elif "/responses/shared/" in name and name.endswith(".body"):
                found_us, found_ca = _ecom_ids(handle.read())
                us_ecom |= found_us
                ca_ecom |= found_ca
            elif "/responses/CA/" in name and name.endswith(".body"):
                ca_lookup |= {int(match) for match in STLOC_ID.findall(handle.read())}
    return BundleInputs(
        capture_id=capture_id,
        polled_ids=polled,
        us_ecom_ids=us_ecom,
        ca_ecom_ids=ca_ecom,
        ca_lookup_ids=ca_lookup,
        station_ids=stations,
    )


def _ids_from_csv(data: bytes, countries: set[str] | None = None) -> set[int]:
    try:
        frame = pl.read_csv(io.BytesIO(data), infer_schema_length=0)
    except Exception as exc:  # a malformed input must not stop discovery
        print(f"::warning::could not parse a bundle CSV: {exc}")
        return set()
    return _ids_from_frame(frame, countries=countries)


def _ids_from_frame(frame: pl.DataFrame | None, countries: set[str] | None = None) -> set[int]:
    if frame is None or frame.is_empty():
        return set()
    column = next((name for name in ID_COLUMNS if name in frame.columns), None)
    if column is None:
        return set()
    if countries and "country" in frame.columns:
        frame = frame.filter(pl.col("country").cast(pl.String).is_in(sorted(countries)))
    found: set[int] = set()
    for value in frame[column].to_list():
        text = str(value).strip()
        if text.isdigit():
            found.add(int(text))
    return found


def _ecom_ids(body: bytes) -> tuple[set[int], set[int]]:
    try:
        payload = json.loads(body)
    except ValueError:
        return set(), set()
    if not isinstance(payload, dict):
        return set(), set()
    us: set[int] = set()
    ca: set[int] = set()
    for warehouse in payload.get("warehouses") or []:
        if not isinstance(warehouse, dict):
            continue
        raw = str(warehouse.get("warehouseId", "")).strip()
        if not raw.isdigit():
            continue
        country = str((warehouse.get("address") or {}).get("countryName") or "").strip().upper()
        if country == "US":
            us.add(int(raw))
        elif country == "CA":
            ca.add(int(raw))
    return us, ca


def candidate_ids(bundle: BundleInputs, cfg) -> list[int]:
    extras = _ids_from_frame(getattr(cfg, "us_extra_ids", None))
    known = (
        bundle.us_ecom_ids
        | bundle.ca_ecom_ids
        | bundle.ca_lookup_ids
        | bundle.station_ids
        | extras
        | {0}
    )
    highest = max(known)
    excluded = bundle.polled_ids | bundle.ca_ecom_ids
    return [value for value in range(1, highest + LOOKAHEAD + 1) if value not in excluded]
