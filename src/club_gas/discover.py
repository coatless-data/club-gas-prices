"""Monthly US station discovery (spec 10.4).

Costco's metadata API does not list every station that sells fuel, and
relocations change warehouse ids. Once a month this command sweeps the legacy
price endpoint over the ids nobody polls and reports the ones that answer with
a plausible US price. Each new one is also appended to ``config/watch.csv`` on
hold, so captures record its prices from then on without publishing them, and
a maintainer who finds it real turns it on there (held-stations design).

Everything except the price sweep is read from a single capture bundle, so
discovery makes no metadata requests of its own.
"""

from __future__ import annotations

import csv
import io
import json
import re
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path

import polars as pl

from .issues import Issues, run_url
from .sources import costco_ecom
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
# The sweep asks Costco's endpoint, so the feed that matters is Costco's own.
US_FEED = "US-COSTCO"
# Bundles keep each feed's responses under its feed id. Those written before
# the United States gained a second chain used the bare country code.
CA_RESPONSES = ("/responses/CA-COSTCO/", "/responses/CA/")
# GitHub refuses an issue body longer than this, and Issues.ensure_open can
# only log the refusal, so an oversized report would never be seen at all.
ISSUE_BODY_LIMIT = 65_536
# Each watched id rides in a price batch of ten, so watching is cheap; the cap
# is for the month the endpoint answers for a whole range of ids at once.
MAX_NEW_WATCH = 25
WATCH_COLUMNS = ("feed", "source_station_id", "detected", "status", "publish_from", "note")


@dataclass(frozen=True)
class DiscoverResult:
    candidates: list[dict]
    skipped_reason: str | None = None
    # The candidate ids appended to config/watch.csv by this run.
    watched: list[str] = field(default_factory=list)


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
    """Costco's US feed status, or the country's for entries from before feeds.

    The country roll-up reads as badly as its worst feed, so a failing Sam's
    feed would otherwise hide every bundle whose Costco capture was fine.
    """
    status = _entry_status(entry)
    feed = (status.get("feeds") or {}).get(US_FEED)
    if feed is not None:
        return feed.get("status")
    return ((status.get("countries") or {}).get("US") or {}).get("status")


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
                # Costco rows only, for the reason candidate_ids gives.
                frame = _costco_rows(_read_csv(handle.read()))
                stations |= _ids_from_frame(frame, countries={"US", "CA"})
            elif "/responses/shared/" in name and name.endswith(".body"):
                found_us, found_ca = _ecom_ids(handle.read())
                us_ecom |= found_us
                ca_ecom |= found_ca
            elif any(part in name for part in CA_RESPONSES) and name.endswith(".body"):
                ca_lookup |= {int(match) for match in STLOC_ID.findall(handle.read())}
    return BundleInputs(
        capture_id=capture_id,
        polled_ids=polled,
        us_ecom_ids=us_ecom,
        ca_ecom_ids=ca_ecom,
        ca_lookup_ids=ca_lookup,
        station_ids=stations,
    )


def _read_csv(data: bytes) -> pl.DataFrame | None:
    try:
        return pl.read_csv(io.BytesIO(data), infer_schema_length=0)
    except Exception as exc:  # a malformed input must not stop discovery
        print(f"::warning::could not parse a bundle CSV: {exc}")
        return None


def _ids_from_csv(data: bytes, countries: set[str] | None = None) -> set[int]:
    return _ids_from_frame(_read_csv(data), countries=countries)


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
    us: set[int] = set()
    ca: set[int] = set()
    for raw, location in costco_ecom.locations(payload) or []:
        if not raw.isdigit():
            continue
        country = str((location.get("address") or {}).get("countryName") or "").strip().upper()
        if country == "US":
            us.add(int(raw))
        elif country == "CA":
            ca.add(int(raw))
    return us, ca


def _costco_rows(frame: pl.DataFrame | None) -> pl.DataFrame | None:
    if frame is None or frame.is_empty() or "brand" not in frame.columns:
        return frame
    return frame.filter(pl.col("brand").cast(pl.String) == "COSTCO")


def candidate_ids(bundle: BundleInputs, cfg) -> list[int]:
    # Costco rows only, here and in the bundle's stations. `highest` drives a
    # range() sweep against Costco's price endpoint, so one Sam's-shaped row --
    # four digits in the 4700-8300 band -- would lift it from ~2,100 ids to
    # ~6,800, roughly 470 extra batched requests a month against the endpoint
    # we are already trying to slim.
    extras = _ids_from_frame(_costco_rows(getattr(cfg, "us_extra_ids", None)))
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


def _price_url(cfg) -> str:
    country = (getattr(cfg, "countries", None) or {}).get("US")
    url = getattr(country, "url", None)
    if url and "AjaxGetGasPricesService" in url:
        return url
    return US_PRICE_URL


def _us_bounds(cfg) -> tuple[float, float]:
    """The regular-grade sanity bounds for the US price unit (spec 7.4).

    ``CountryConfig.bounds`` maps a price unit to a ``Bounds`` dataclass, so
    the lookup is always ``bounds[price_unit].for_grade(grade_raw)``.
    """
    country = (getattr(cfg, "countries", None) or {}).get("US")
    unit = getattr(country, "price_unit", None) or US_PRICE_UNIT
    bounds = (getattr(country, "bounds", None) or {}).get(unit)
    if bounds is None:
        return USD_GAL_MIN, USD_GAL_MAX
    low, high = bounds.for_grade("regular")
    return float(low), float(high)


def _as_float(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).strip())
    except ValueError:
        return None


def _is_reportable(prices: dict, low: float, high: float) -> bool:
    regular = _as_float(prices.get("regular"))
    if regular is None:
        # Premium-only rows are the unattended business-center placeholders.
        return False
    if not (low <= regular <= high):
        # Below the floor is a per-litre listing (Puerto Rico #335 at 1.097) or
        # a Canadian value leaking through (#1090 at 1.929); above the ceiling
        # is not a US pump price either.
        return False
    # The pre-opening placeholder a warehouse shows before it sells fuel.
    return not (
        regular == PRE_OPEN_REGULAR and _as_float(prices.get("premium")) == PRE_OPEN_PREMIUM
    )


def _payload(response) -> dict | None:
    if response.error or response.status != 200 or not response.body:
        print(
            f"::warning::discovery batch failed: {response.url} "
            f"status={response.status} error={response.error}"
        )
        return None
    try:
        payload = json.loads(response.body)
    except ValueError:
        print(f"::warning::discovery batch returned a non-JSON body: {response.url}")
        return None
    if not isinstance(payload, dict) or "errorMessage" in payload:
        return None
    return payload


def _candidate_row(candidate: dict, url: str) -> str:
    prices = candidate["prices"]
    other = (
        ", ".join(
            f"{key}={value}"
            for key, value in sorted(prices.items())
            if key not in ("regular", "premium", "diesel")
        )
        or "-"
    )
    link = f"{url}?warehouseid={candidate['warehouse_id']}"
    row = [
        candidate["warehouse_id"],
        str(prices.get("regular", "-")),
        str(prices.get("premium", "-")),
        str(prices.get("diesel", "-")),
        other,
        f"[prices]({link})",
    ]
    return "| " + " | ".join(row) + " |"


def _size(text: str) -> int:
    # UTF-8 bytes are never fewer than the characters GitHub counts, so a body
    # that fits by this measure fits by GitHub's.
    return len(text.encode("utf-8"))


def _omitted_note(count: int, run: str | None) -> str:
    where = (
        f"[the Discover run that wrote this]({run})" if run else "the Discover run that wrote this"
    )
    return (
        f"**{count} more warehouse id(s) did not fit:** GitHub caps an issue body at "
        f"{ISSUE_BODY_LIMIT:,} characters. Every candidate is listed in the log of {where}."
    )


def _watch_rows(cfg) -> list[dict]:
    """config/watch.csv's US-COSTCO rows, as loaded with the config."""
    frame = getattr(cfg, "watch", None)
    if not isinstance(frame, pl.DataFrame) or frame.is_empty():
        return []
    return [row for row in frame.iter_rows(named=True) if row.get("feed") == US_FEED]


def _watch_note(candidate: dict, now: datetime) -> str:
    prices = " ".join(f"{key}={value}" for key, value in sorted(candidate["prices"].items()))
    return f"found by the {now:%Y-%m-%d} sweep: {prices}"


def append_watch(path: Path, candidates: list[dict], now: datetime) -> None:
    """Append one `hold` row per candidate, leaving every existing line as written."""
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    if not existing.strip():
        writer.writerow(WATCH_COLUMNS)
    elif not existing.endswith("\n"):
        buffer.write("\n")
    for candidate in candidates:
        writer.writerow(
            [
                US_FEED,
                candidate["warehouse_id"],
                f"{now:%Y-%m-%d}",
                "hold",
                "",
                _watch_note(candidate, now),
            ]
        )
    with path.open("a" if existing.strip() else "w", encoding="utf-8", newline="") as handle:
        handle.write(buffer.getvalue())


def _watch_text(
    candidates: list[dict], watched: list[str], unwatched: list[str], on_hold: list[dict]
) -> list[str]:
    """The issue's closing lines: what was recorded, and how to publish it."""
    lines = [""]
    if watched:
        lines.append(
            f"{len(watched)} new id(s) above are now in `config/watch.csv` with status "
            "`hold`: every capture records their prices without publishing them."
        )
        if unwatched and len(watched) == MAX_NEW_WATCH:
            lines.append(
                f"Only the first {MAX_NEW_WATCH} of {len(candidates)} were added; add the "
                "rest by hand if they turn out to be real."
            )
    elif unwatched:
        lines.append(
            f"{len(unwatched)} id(s) above are not in `config/watch.csv`. Add the real ones "
            "there with status `hold` to record their prices until you decide."
        )
    held_before = [row for row in on_hold if row["source_station_id"] not in set(watched)]
    if held_before:
        listed = ", ".join(
            f"{row['source_station_id']} (detected {row['detected']})" for row in held_before
        )
        lines.append(f"Still on hold in `config/watch.csv`: {listed}.")
    lines += [
        "",
        "To publish a real station, set its `status` to `on` in `config/watch.csv` and its",
        "`publish_from` to the first `capture_date` (a UTC date) whose prices are real.",
        "If Costco's locator does not list it, also add it to `config/us_extra_ids.csv`",
        "with `brand` (COSTCO), `name`, `city`, `region`, `postcode`, `lat`, `lon`,",
        "`timezone` and `note`. Then run the Rebuild workflow (scope `month`) for each",
        "month since `publish_from`, so the captures it was held in are published too.",
        "Delete the rows that turn out to be placeholders, then close this issue.",
    ]
    return lines


def _issue_body(
    candidates: list[dict],
    bundle: BundleInputs,
    now: datetime,
    url: str,
    run: str | None = None,
    *,
    watched: list[str] | None = None,
    unwatched: list[str] | None = None,
    on_hold: list[dict] | None = None,
) -> tuple[str, int]:
    """The issue body, and how many of the candidates it lists.

    Rows are dropped from the end until the body fits GitHub's limit, and a
    note in their place says how many and where the whole list is.
    """
    head = [
        f"The monthly sweep of `AjaxGetGasPricesService` on {now:%Y-%m-%d} found "
        f"{len(candidates)} warehouse id(s) that price fuel but are not polled.",
        "",
        f"Excluded ids came from capture `{bundle.capture_id}`.",
        "",
        "| warehouse id | regular | premium | diesel | other grades | check |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    foot = _watch_text(candidates, watched or [], unwatched or [], on_hold or [])
    rows = [_candidate_row(candidate, url) for candidate in candidates]
    body = "\n".join(head + rows + foot)
    if _size(body) <= ISSUE_BODY_LIMIT:
        return body, len(rows)
    # Room for the note at its longest, since fewer omitted ids never make it
    # longer; each row then costs its own length plus one line break.
    budget = ISSUE_BODY_LIMIT - _size("\n".join([*head, "", _omitted_note(len(rows), run), *foot]))
    listed = 0
    for row in rows:
        budget -= _size(row) + 1
        if budget < 0:
            break
        listed += 1
    note = _omitted_note(len(rows) - listed, run)
    return "\n".join([*head, *rows[:listed], "", note, *foot]), listed


def _candidate_sort_key(candidate: dict) -> int:
    warehouse_id = candidate["warehouse_id"]
    return int(warehouse_id) if warehouse_id.isdigit() else 0


def discover(
    store: ReleaseStore, client, cfg, issues: Issues, *, now: datetime, watch: bool = False
) -> DiscoverResult:
    """Sweep, report, and with `watch` append the new candidates to config/watch.csv.

    `watch` is the Discover workflow's flag: a run on a maintainer's machine
    leaves their working tree alone.
    """
    selected = select_bundle(store, now=now)
    if selected is None:
        print(
            "::warning::no uploaded capture bundle with a successful US country was found in this "
            "month or the previous month; skipping discovery"
        )
        return DiscoverResult(candidates=[], skipped_reason="no_bundle")

    tag, name = selected
    bundle = read_bundle(store, tag, name)
    ids = candidate_ids(bundle, cfg)
    url = _price_url(cfg)
    low, high = _us_bounds(cfg)

    candidates: list[dict] = []
    for index in range(0, len(ids), BATCH):
        if client.abandoned(url):
            print("::warning::www.costco.com was abandoned; discovery stops with a partial sweep")
            break
        batch = ids[index : index + BATCH]
        query = f"{url}?warehouseid=" + "_".join(str(value) for value in batch)
        response = client.request(f"discover/{index // BATCH:04d}", query, expect_json=True)
        payload = _payload(response)
        if payload is None:
            continue
        for raw_id, prices in payload.items():
            if not isinstance(prices, dict) or not _is_reportable(prices, low, high):
                continue
            candidates.append(
                {
                    "warehouse_id": str(raw_id),
                    "prices": {str(key): str(value) for key, value in prices.items()},
                }
            )

    candidates.sort(key=_candidate_sort_key)
    rows = _watch_rows(cfg)
    on_hold = [row for row in rows if row.get("status") == "hold"]
    known = {str(row["source_station_id"]) for row in rows}
    new = [c for c in candidates if c["warehouse_id"] not in known]
    watched = _start_watching(cfg, new, now) if watch else []
    unwatched = [c["warehouse_id"] for c in new if c["warehouse_id"] not in set(watched)]
    if candidates:
        title = f"US station discovery {now:%Y-%m}"
        body, listed = _issue_body(
            candidates,
            bundle,
            now,
            url,
            run_url(),
            watched=watched,
            unwatched=unwatched,
            on_hold=on_hold,
        )
        if listed < len(candidates):
            # The issue points here for the rows it had no room for.
            print(
                f"::warning::the discovery issue lists {listed} of {len(candidates)} "
                "candidates; every candidate follows"
            )
            for candidate in candidates:
                print(_candidate_row(candidate, url))
        issues.ensure_open(title, body)
    print(
        f"discover: bundle {bundle.capture_id}, {len(ids)} candidate ids, "
        f"{len(candidates)} reported, {len(watched)} added to config/watch.csv"
    )
    return DiscoverResult(candidates=candidates, skipped_reason=None, watched=watched)


def _start_watching(cfg, candidates: list[dict], now: datetime) -> list[str]:
    """Append up to MAX_NEW_WATCH unwatched candidates to the checkout's watch.csv.

    Returns the ids added. Without a checkout (no `cfg.root`) nothing is written.
    """
    root = getattr(cfg, "root", None)
    new = candidates[:MAX_NEW_WATCH]
    if root is None or not new:
        return []
    path = Path(root) / "config" / "watch.csv"
    try:
        append_watch(path, new, now)
    except OSError as exc:
        print(f"::warning::could not add the candidates to {path}: {exc}")
        return []
    return [c["warehouse_id"] for c in new]
