"""Roll-ups: the daily grain, month and year closes, and the full `current` rebuild.

Release layout (spec 8.1-8.2):
  data-YYYY-MM  prerelease while open: club-gas-YYYY-MM-DD.csv.gz (capture grain),
                capture-<capture_id>.tar.gz, manifest-YYYY-MM.json; at close it also
                gets club-gas-YYYY-MM.parquet / .csv.gz (daily grain) and
                club-gas-YYYY-MM-captures.parquet (capture grain).
  data-YYYY     year files, written by the year close.
  current       the 7 all-time assets, rebuilt in full whenever a month closes.
"""

from __future__ import annotations

import gzip
import json
import re
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import polars as pl

from . import publish, schema
from .issues import Issues
from .store import AssetNotFound, recover_temporaries, recovery_tags, sha256_file

MONTH_TAG = re.compile(r"^data-(\d{4})-(\d{2})$")
YEAR_TAG = re.compile(r"^data-(\d{4})$")
DAILY_ASSET = re.compile(r"^club-gas-(\d{4}-\d{2}-\d{2})\.csv\.gz$")
BUNDLE_ASSET = re.compile(r"^capture-(.+)\.tar\.gz$")
TEMP_ASSET = re.compile(r"\.(next|old)-")

DAILY_SORT = ["capture_date", "country", "station_key", "grade_raw"]
CAPTURE_SORT = ["station_key", "grade_raw", "capture_id"]
CURRENT_DATA_ASSETS = (
    "club-gas-all.parquet",
    "club-gas-all.csv.gz",
    "club-gas-all-captures.parquet",
    "club-gas-latest.csv",
    "stations.csv",
    "fx.csv",
)
SCHEMA_URL = "https://github.com/coatless-data/club-gas-prices#getting-the-data"


@dataclass
class CloseResult:
    closed_months: list[str] = field(default_factory=list)
    closed_years: list[str] = field(default_factory=list)
    blocked_months: list[str] = field(default_factory=list)
    rebuilt_current: bool = False


DAILY_KEYS = ["capture_date", "station_key", "grade_raw"]
DAILY_SCHEMA: dict[str, pl.DataType] = {
    **schema.ROW_SCHEMA,
    "n_captures": pl.Int64,
    "price_min": pl.Float64,
    "price_max": pl.Float64,
}


def daily_grain(rows: pl.DataFrame) -> pl.DataFrame:
    """One row per (capture_date, station_key, grade_raw), spec 6.2.

    The kept row is the one with the greatest `captured_at_utc` that day;
    `n_captures`, `price_min` and `price_max` summarise the whole day.
    """
    if rows.height == 0:
        return pl.DataFrame(schema=DAILY_SCHEMA)
    summary = rows.group_by(DAILY_KEYS).agg(
        pl.col("capture_id").n_unique().cast(pl.Int64).alias("n_captures"),
        pl.col("price").min().alias("price_min"),
        pl.col("price").max().alias("price_max"),
    )
    newest = rows.sort(["captured_at_utc", "capture_id"]).group_by(DAILY_KEYS).last()
    return (
        newest.join(summary, on=DAILY_KEYS, how="left").select(list(DAILY_SCHEMA)).sort(DAILY_SORT)
    )


def month_tags(store) -> list[str]:
    return sorted(r.tag for r in store.list_releases() if MONTH_TAG.match(r.tag))


def read_month_manifest(store, tag: str) -> dict:
    month = tag.removeprefix("data-")
    name = f"manifest-{month}.json"
    with tempfile.TemporaryDirectory() as td:
        try:
            path = store.download(tag, name, Path(td) / name)
        except AssetNotFound:
            return {"schema_version": 1, "month": month, "captures": {}}
        return json.loads(path.read_text("utf-8"))


def write_month_manifest(store, tag: str, manifest: dict, *, token: str) -> None:
    month = tag.removeprefix("data-")
    name = f"manifest-{month}.json"
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / name
        path.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str), "utf-8")
        store.replace_atomic(tag, path, name, token)


def _recover_all(store) -> None:
    """§8.4 recovery on `current` and every data-* release."""
    for tag in recovery_tags(store):
        recover_temporaries(store, tag)


def _token(now: datetime) -> str:
    return "run-" + now.strftime("%Y-%m-%dT%H%MZ")


def _stamp(now: datetime) -> str:
    return now.strftime("%Y-%m-%dT%H%MZ")


def _write_csv_gz(df: pl.DataFrame, path: Path) -> None:
    with gzip.open(path, "wb") as fh:
        df.write_csv(fh)


def _parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _safe_issue(func, *args) -> None:
    """A failed issue API call is a ::warning, never a change of outcome (§8.6)."""
    try:
        func(*args)
    except Exception as exc:
        print(f"::warning::issue call failed: {exc}")


def _blocking_reasons(store, tag: str) -> list[str]:
    """§8.6 step 0. Recovery has already run, so anything left really is stuck.

    Merging an orphan bundle is `publish`'s job (§8.5 step 0), and capture.yml runs
    `publish` immediately before `close-periods`; close only verifies and refuses.
    """
    reasons: list[str] = []
    assets = store.list_assets(tag)
    for asset in assets:
        if TEMP_ASSET.search(asset.name):
            reasons.append(f"temporary asset `{asset.name}` is still present")
    manifest = read_month_manifest(store, tag)
    known = set(manifest.get("captures", {}))
    for asset in assets:
        match = BUNDLE_ASSET.match(asset.name)
        if match and asset.state == "uploaded" and match.group(1) not in known:
            reasons.append(f"bundle `{asset.name}` has no entry in the month manifest")
    # A day the manifest records rows for, but whose daily file is missing or not
    # `uploaded`, must block the month exactly like a temp asset or an orphan
    # bundle: `_close_month`'s own row-count checks compare `captures.height` and
    # `per_file` only over days that ARE present, so a missing day makes both
    # sides shrink together and would otherwise close the month short (spec
    # review round 4, Finding 1).
    expected = _expected_daily_rows(manifest)
    recorded_days = set(expected)
    uploaded = {
        match.group(1): asset.name
        for asset in assets
        if (match := DAILY_ASSET.match(asset.name)) and asset.state == "uploaded"
    }
    for day in sorted(recorded_days - set(uploaded)):
        reasons.append(
            f"day `{day}` has no uploaded daily file, though the month manifest records rows for it"
        )
    # A daily file whose row count disagrees with what the month manifest
    # records for it must block the month the same way, rather than let
    # `_close_month`'s own cross-check raise `ValueError` and abort the whole
    # `close_periods` call -- including every other month still to be checked
    # in the same invocation. The daily file and the manifest are each written
    # by their own `replace_atomic` call, so nothing guarantees they always
    # land together (a rebuild interrupted between the two, or manual
    # surgery, can leave them disagreeing); disagreeing about a fact this
    # basic needs a human, via the same blocked-month issue as every other
    # refusal here, not a crash (spec review, Task 16 round 2).
    mismatched = sorted(recorded_days & set(uploaded))
    if mismatched:
        with tempfile.TemporaryDirectory() as td:
            work = Path(td)
            for day in mismatched:
                name = uploaded[day]
                path = store.download(tag, name, work / name)
                actual = schema.read_rows_csv_gz(path).height
                if actual != expected[day]:
                    reasons.append(
                        f"day `{day}` has {actual} rows in `{name}` but the month "
                        f"manifest records {expected[day]}"
                    )
    return reasons


def _blocked_body(month: str, reasons: list[str], now: datetime) -> str:
    """Say what clears the refusal, because most reasons never clear on their own.

    Only an orphan bundle does: the next `publish` merges it (§8.5 step 0).
    Nothing that runs on a schedule reliably repairs a day whose daily file is
    missing or disagrees with the manifest, so telling the reader to wait for
    one would leave the month open until somebody noticed.
    """
    lines = "\n".join(f"- {reason}" for reason in reasons)
    return (
        f"`close-periods` refused to close `data-{month}` at {_stamp(now)}.\n\n"
        f"Reasons:\n{lines}\n\n"
        "A bundle with no manifest entry is merged by the next `publish` that "
        "succeeds (§8.5 step 0), and the month closes on the `close-periods` after "
        "it. No other reason clears on its own:\n\n"
        "1. Open **Actions -> Rebuild -> Run workflow**.\n"
        f"2. Set `scope` to `month` and `value` to `{month}`.\n"
        "3. The run rewrites each day's file and its manifest entries from the "
        "stored bundles, and the next `close-periods` closes the month.\n\n"
        "A reason still listed after the rebuild needs a look by hand. This issue "
        "closes automatically when the month closes."
    )


def _ensure_latest_is_current(store) -> None:
    release = store.get_release("current")
    if release is not None and not release.is_latest:
        store.update_release("current", make_latest="true")


def close_periods(
    store,
    cfg,
    *,
    now: datetime,
    rebuild_current: bool = False,
    issues: Issues | None = None,
) -> CloseResult:
    issues = issues if issues is not None else Issues(None, None)
    token = _token(now)
    _recover_all(store)
    run_month = now.strftime("%Y-%m")
    releases = {r.tag: r for r in store.list_releases()}
    closed_months: list[str] = []
    blocked_months: list[str] = []
    rebuilt = False

    for tag in month_tags(store):
        month = tag.removeprefix("data-")
        if month >= run_month or not releases[tag].prerelease:
            continue
        reasons = _blocking_reasons(store, tag)
        if reasons:
            blocked_months.append(month)
            _safe_issue(
                issues.ensure_open,
                f"Period close blocked: {month}",
                _blocked_body(month, reasons, now),
                ["period-close"],
            )
            continue
        _close_month(store, cfg, tag, month, token=token, now=now)
        # The data rebuild runs before `prerelease` is cleared on purpose (an
        # interrupted close is simply retried), so it still sees this month as
        # open and `current/manifest.json`'s closed_months cannot list it yet.
        # Refresh that list right after clearing `prerelease`: a small
        # manifest-only write, safe to lose to an interrupt since the next close
        # or an explicit rebuild repairs it (spec review round 4, Finding 2).
        _rebuild_current_impl(store, cfg, now=now)
        rebuilt = True
        store.update_release(tag, prerelease=False, make_latest="false")
        _refresh_current_periods(store, token=token)
        closed_months.append(month)
        _safe_issue(
            issues.close,
            f"Period close blocked: {month}",
            f"Closed by close-periods at {_stamp(now)}.",
        )

    if rebuild_current and not rebuilt:
        _rebuild_current_impl(store, cfg, now=now)
        rebuilt = True

    closed_years: list[str] = []
    if not blocked_months:
        closed_years = _close_years(store, cfg, now=now, token=token)
    if closed_years and not rebuilt:
        _refresh_current_periods(store, token=token)

    _ensure_latest_is_current(store)
    return CloseResult(closed_months, closed_years, blocked_months, rebuilt)


def _expected_daily_rows(manifest: dict) -> dict[str, int]:
    """Per date, the row count the newest capture that touched it recorded."""
    seen: dict[str, tuple[str, int]] = {}
    for capture_id, entry in (manifest.get("captures") or {}).items():
        for day, info in (entry.get("daily_files") or {}).items():
            rows = info.get("rows")
            if rows is None:
                continue
            previous = seen.get(day)
            if previous is None or capture_id > previous[0]:
                seen[day] = (capture_id, int(rows))
    return {day: value[1] for day, value in seen.items()}


def _read_daily_files(store, tag: str, work: Path) -> tuple[pl.DataFrame, dict[str, int]]:
    daily: dict[str, str] = {}
    for asset in store.list_assets(tag):
        match = DAILY_ASSET.match(asset.name)
        if match and asset.state == "uploaded":
            daily[match.group(1)] = asset.name
    frames: list[pl.DataFrame] = []
    per_file: dict[str, int] = {}
    for day in sorted(daily):
        path = store.download(tag, daily[day], work / f"{tag}-{daily[day]}")
        frame = schema.read_rows_csv_gz(path)
        per_file[day] = frame.height
        frames.append(frame)
    if not frames:
        return pl.DataFrame(schema=schema.ROW_SCHEMA), per_file
    return pl.concat(frames, how="vertical"), per_file


def _month_body(month: str, captures: pl.DataFrame, cfg, *, now: datetime) -> str:
    """The closed month's release notes, with its coverage counted per feed.

    Per feed rather than per country: two chains share a country, and a
    per-country count would hide one chain's missing captures behind the
    other's.
    """
    lines = [
        f"# Warehouse-club gas prices {month}",
        "",
        "## Coverage (captures per day, by feed)",
        "",
    ]
    if captures.is_empty():
        lines.append("_No rows._")
    else:
        coverage = (
            captures.select(
                "capture_date",
                pl.concat_str(["country", "brand"], separator="-").alias("feed"),
                "capture_id",
            )
            .unique()
            .group_by(["capture_date", "feed"])
            .agg(pl.len().alias("captures"))
        )
        wide = (
            coverage.pivot(on="feed", index="capture_date", values="captures")
            .fill_null(0)
            .sort("capture_date")
        )
        feeds = sorted(c for c in wide.columns if c != "capture_date")
        lines.append("| date | " + " | ".join(feeds) + " |")
        lines.append("|---" * (len(feeds) + 1) + "|")
        for row in wide.iter_rows(named=True):
            cells = " | ".join(str(row[c]) for c in feeds)
            lines.append(f"| {row['capture_date']} | {cells} |")
    lines += [
        "",
        "## Files",
        "",
        f"- `club-gas-{month}.parquet`, `club-gas-{month}.csv.gz` — daily grain",
        f"- `club-gas-{month}-captures.parquet` — capture grain",
        f"- `club-gas-{month}-DD.csv.gz` — one file per UTC day, capture grain",
        "- `capture-<capture_id>.tar.gz` — raw capture bundles",
        f"- `manifest-{month}.json` — per-capture manifest",
        "",
        f"Schema: {SCHEMA_URL}",
        "",
        f"Closed {_stamp(now)}.",
        "",
        cfg.site.notice_text(),
    ]
    return "\n".join(lines)


def _close_month(store, cfg, tag: str, month: str, *, token: str, now: datetime) -> None:
    manifest = read_month_manifest(store, tag)
    expected = _expected_daily_rows(manifest)
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        captures, per_file = _read_daily_files(store, tag, work)
        if not per_file:
            raise ValueError(f"{tag}: no daily files to close")
        total = sum(per_file.values())
        if captures.height != total:
            raise ValueError(
                f"{tag}: capture grain has {captures.height} rows, the daily files hold {total}"
            )
        for day, rows in expected.items():
            if day in per_file and per_file[day] != rows:
                raise ValueError(f"{tag}: {day} has {per_file[day]} rows, manifest records {rows}")
        grain = daily_grain(captures)
        parquet = work / f"club-gas-{month}.parquet"
        schema.write_parquet(grain, parquet, sort_by=DAILY_SORT)
        store.replace_atomic(tag, parquet, parquet.name, token)
        csv_gz = work / f"club-gas-{month}.csv.gz"
        _write_csv_gz(grain.sort(DAILY_SORT), csv_gz)
        store.replace_atomic(tag, csv_gz, csv_gz.name, token)
        caps = work / f"club-gas-{month}-captures.parquet"
        schema.write_parquet(captures, caps, sort_by=CAPTURE_SORT)
        store.replace_atomic(tag, caps, caps.name, token)
        store.update_release(
            tag, title=f"Data {month}", body=_month_body(month, captures, cfg, now=now)
        )


def _read_current_stations(store, work: Path) -> pl.DataFrame:
    try:
        path = store.download("current", "stations.csv", work / "stations-existing.csv")
    except AssetNotFound:
        return pl.DataFrame(schema=schema.STATION_SCHEMA)
    # Polars' own reader, not the strict `schema.read_csv` (matching publish.py's
    # `_update_current`, which reads its own `current` CSVs back the same way).
    # Both writers now emit `schema`'s CSV_DATETIME_FORMAT, but an asset written
    # before that carries Polars' default datetime text, and this reader accepts
    # either.
    return pl.read_csv(path, schema=schema.STATION_SCHEMA)


def _latest_rows(captures: pl.DataFrame) -> pl.DataFrame:
    """§8.2: for each station_key, every row of its newest capture."""
    if captures.is_empty():
        return captures
    newest = captures.group_by("station_key").agg(pl.col("captured_at_utc").max())
    joined = captures.join(newest, on=["station_key", "captured_at_utc"], how="inner")
    return (
        joined.sort("capture_id")
        .unique(subset=["station_key", "grade_raw"], keep="last", maintain_order=True)
        .sort(["country", "station_key", "grade_raw"])
    )


def _fx_frame(rows: list[dict]) -> pl.DataFrame:
    """§6.4: fx.csv rows come from each capture's fx.json, which is an array of
    non-USD rows `{currency, units_per_usd, fx_rate_date, fx_source,
    fx_fetched_at_utc}`. `fx_usd_per_unit` is derived here, as it is at capture time.
    """
    if not rows:
        return pl.DataFrame(schema=schema.FX_SCHEMA)
    built = []
    for row in rows:
        units = float(row["units_per_usd"])
        rate_date = row.get("fx_rate_date")
        fetched = row.get("fx_fetched_at_utc")
        built.append(
            {
                "capture_id": row["capture_id"],
                "currency": row["currency"],
                "units_per_usd": units,
                "fx_usd_per_unit": 1.0 / units,
                "fx_rate_date": date.fromisoformat(rate_date) if rate_date else None,
                "fx_source": row.get("fx_source"),
                "fx_fetched_at_utc": _parse_utc(fetched) if fetched else None,
            }
        )
    frame = (
        pl.DataFrame(built, schema=schema.FX_SCHEMA)
        .unique(subset=["capture_id", "currency"], keep="last", maintain_order=True)
        .sort(["capture_id", "currency"])
    )
    return schema.round_fx_columns(frame)


def _apply_links(stations: pl.DataFrame, links: pl.DataFrame) -> pl.DataFrame:
    mapping: dict[str, str] = {}
    if links is not None and not links.is_empty():
        mapping = dict(
            zip(
                links["old_station_key"].to_list(),
                links["new_station_key"].to_list(),
                strict=True,
            )
        )
    if mapping:
        column = pl.col("station_key").replace_strict(mapping, default=None, return_dtype=pl.String)
    else:
        column = pl.lit(None, dtype=pl.String)
    return stations.with_columns(column.alias("superseded_by"))


def _rebuild_stations(
    captures: pl.DataFrame,
    existing: pl.DataFrame,
    links: pl.DataFrame,
    closed_after_days: dict[str, int] | None = None,
    *,
    newest_status: dict | None = None,
    disabled_feeds: frozenset[str] = frozenset(),
) -> pl.DataFrame:
    """§6.3 full rebuild: recompute the three seen columns, keep everything else.

    Additive by construction: it starts from every station already recorded and
    only ever joins in ones it has not seen. A station that has left its feed
    keeps its row and its history, and is relabelled by the same rule publish
    applies incrementally.
    """
    if captures.is_empty():
        return (
            _apply_links(existing, links)
            .select(list(schema.STATION_SCHEMA))
            .cast(schema.STATION_SCHEMA)
        )
    agg = (
        captures.group_by("station_key")
        .agg(
            pl.col("captured_at_utc").min().alias("first_seen_utc"),
            pl.col("captured_at_utc").max().alias("last_seen_utc"),
            pl.col("grade_raw").unique().sort().alias("_grades"),
        )
        .with_columns(pl.col("_grades").list.join("|").alias("grades_seen"))
        .drop("_grades")
    )
    kept = (
        existing.join(agg, on="station_key", how="left", suffix="_new")
        .with_columns(
            pl.coalesce(["first_seen_utc_new", "first_seen_utc"]).alias("first_seen_utc"),
            pl.coalesce(["last_seen_utc_new", "last_seen_utc"]).alias("last_seen_utc"),
            pl.coalesce(["grades_seen_new", "grades_seen"]).alias("grades_seen"),
        )
        .select(list(schema.STATION_SCHEMA))
    )
    newest = captures.sort(["captured_at_utc", "capture_id"]).group_by("station_key").last()
    country_newest = captures.group_by("country").agg(
        pl.col("capture_id").max().alias("_country_newest")
    )
    fresh = (
        newest.join(existing.select("station_key"), on="station_key", how="anti")
        .join(agg, on="station_key", how="inner")
        .join(country_newest, on="country", how="left")
        .with_columns(
            pl.lit(None, dtype=pl.String).alias("alt_id"),
            pl.when(pl.col("capture_id") == pl.col("_country_newest"))
            .then(pl.lit("active"))
            .otherwise(pl.lit("missing"))
            .alias("status"),
            pl.lit(None, dtype=pl.String).alias("superseded_by"),
        )
        .select(list(schema.STATION_SCHEMA))
    )
    out = pl.concat([kept, fresh], how="vertical")
    out = _restate_absent(
        out,
        captures,
        closed_after_days or {},
        newest_status=newest_status,
        disabled_feeds=disabled_feeds,
    )
    return _apply_links(out, links).cast(schema.STATION_SCHEMA).sort("station_key")


def _restate_absent(
    stations: pl.DataFrame,
    captures: pl.DataFrame,
    closed_after_days: dict[str, int],
    *,
    newest_status: dict | None = None,
    disabled_feeds: frozenset[str] = frozenset(),
) -> pl.DataFrame:
    """Relabel stations the newest capture no longer lists, the same way publish
    does, so a rebuilt `current` and an incrementally-built one agree.

    That means publish's three cases (publish.update_stations), judged on the
    newest capture's status: a feed that succeeded relabels its absent stations
    except those its sweep was cut short before asking about; a switched-off
    feed relabels them all, since nobody is collecting it; a feed that failed or
    was skipped says nothing about its stations, so they keep their status.
    Without a status to judge by -- a month manifest from before statuses were
    recorded -- every absent station is relabelled, as before.
    """
    if stations.is_empty() or captures.is_empty():
        return stations
    newest_at = captures["captured_at_utc"].max()
    listed = set(captures.filter(pl.col("captured_at_utc") == newest_at)["station_key"].to_list())
    feeds = (newest_status or {}).get("feeds") or {}
    succeeded = {fid for fid, block in feeds.items() if block.get("status") in ("ok", "degraded")}
    unreached = {
        (fid, warning.get("detail"))
        for fid, block in feeds.items()
        for warning in block.get("warnings") or []
        if warning.get("code") == "not_reached"
    }
    rows = []
    for record in stations.to_dicts():
        if record["station_key"] not in listed and record["status"] != "active":
            fid = f"{record['country']}-{record['brand']}"
            if not feeds:
                judged = True
            elif fid in succeeded:
                judged = (fid, record["source_station_id"]) not in unreached
            else:
                judged = fid in disabled_feeds
            if judged:
                record["status"] = publish.station_status(
                    record["last_seen_utc"], newest_at, closed_after_days.get(record["country"])
                )
        rows.append(record)
    return pl.DataFrame(rows, schema=schema.STATION_SCHEMA)


def _newest_capture_by_feed(statuses: dict[str, dict]) -> dict[str, str]:
    """The same rule `publish` applies incrementally (§8.5): per FEED, the
    greatest capture_id whose status block for that feed was ok or degraded.

    Per feed rather than per country, because a capture that ran only one
    chain must not decide the fate of the other chain's stations."""
    newest: dict[str, str] = {}
    for capture_id, status in statuses.items():
        for fid, block in ((status or {}).get("feeds") or {}).items():
            if block.get("status") in ("ok", "degraded") and capture_id > newest.get(fid, ""):
                newest[fid] = capture_id
    return newest


def _current_body(cfg) -> str:
    return (
        "# Current warehouse-club gas prices\n\n"
        "All-time files at both grains, the latest snapshot, the station and FX "
        f"tables, and `manifest.json`.\n\nSchema: {SCHEMA_URL}\n\n{cfg.site.notice_text()}\n"
    )


def _rebuild_current_impl(store, cfg, *, now: datetime) -> None:
    """§8.6 full `current` rebuild: closed month files + every month manifest +
    the daily files of every prerelease data-* release.

    `close_periods` already recovers every release before it ever reaches here,
    but `rebuild_current` (this function, under its public name) is also called
    directly -- Task 16 does so -- so recovery has to run here too, matching
    store.py's own contract that it runs at the start of publish, close-periods
    AND rebuild. Without it, a release left mid-`replace_atomic` would rebuild
    `current` while silently dropping that asset: `DAILY_ASSET` never matches a
    `.next-`/`.old-` name (spec review round 4, Finding 3). Recovery is
    idempotent, so calling it again here when `close_periods` already did costs
    only an extra listing, never a behavior change.
    """
    _recover_all(store)
    token = _token(now)
    releases = {r.tag: r for r in store.list_releases()}
    frames: list[pl.DataFrame] = []
    fx_rows: list[dict] = []
    statuses: dict[str, dict] = {}
    closed_months: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for tag in month_tags(store):
            month = tag.removeprefix("data-")
            manifest = read_month_manifest(store, tag)
            for capture_id, entry in (manifest.get("captures") or {}).items():
                for row in entry.get("fx") or []:
                    fx_rows.append({**row, "capture_id": capture_id})
                if entry.get("status"):
                    statuses[capture_id] = entry["status"]
            if releases[tag].prerelease:
                for asset in store.list_assets(tag):
                    match = DAILY_ASSET.match(asset.name)
                    if match and asset.state == "uploaded":
                        path = store.download(tag, asset.name, work / f"{tag}-{asset.name}")
                        frames.append(schema.read_rows_csv_gz(path))
            else:
                closed_months.append(month)
                name = f"club-gas-{month}-captures.parquet"
                path = store.download(tag, name, work / name)
                frames.append(
                    pl.read_parquet(path).select(list(schema.ROW_SCHEMA)).cast(schema.ROW_SCHEMA)
                )
        captures = (
            pl.concat(frames, how="vertical") if frames else pl.DataFrame(schema=schema.ROW_SCHEMA)
        )
        grain = daily_grain(captures)
        latest = _latest_rows(captures)
        fx = _fx_frame(fx_rows)
        # `current` must exist before anything reads it back: on the very first
        # rebuild ever, `list_assets("current")` raises StorageError for a release
        # that does not exist at all, which is a different failure than the
        # AssetNotFound `_read_current_stations` handles for a merely-missing
        # asset. publish.py's `ensure_current` runs first for the same reason.
        store.ensure_release("current", "Current", _current_body(cfg), False, "true")
        stations = _rebuild_stations(
            captures,
            _read_current_stations(store, work),
            cfg.station_links,
            {code: c.closed_after_days for code, c in cfg.countries.items()},
            newest_status=statuses[max(statuses)] if statuses else None,
            disabled_feeds=frozenset(fid for fid, feed in cfg.feeds.items() if not feed.enabled),
        )
        assets: dict[str, dict] = {}
        outputs: dict[str, Path] = {}

        def put(name: str, writer) -> None:
            path = work / name
            writer(path)
            store.replace_atomic("current", path, name, token)
            outputs[name] = path
            assets[name] = {"sha256": sha256_file(path), "size": path.stat().st_size}

        put(
            "club-gas-all.parquet",
            lambda p: schema.write_parquet(grain, p, sort_by=DAILY_SORT),
        )
        put("club-gas-all.csv.gz", lambda p: _write_csv_gz(grain.sort(DAILY_SORT), p))
        put(
            "club-gas-all-captures.parquet",
            lambda p: schema.write_parquet(captures, p, sort_by=CAPTURE_SORT),
        )
        put("club-gas-latest.csv", latest.write_csv)
        # Through `schema.write_csv`, exactly as publish._update_current writes
        # them: same validation, same key check, same rounding, same canonical
        # date formats, so the incremental and full writers cannot drift apart.
        put(
            "stations.csv",
            lambda p: schema.write_csv(
                stations,
                p,
                schema=schema.STATION_SCHEMA,
                sort_by=schema.STATION_SORT,
                validate=schema.validate_stations,
            ),
        )
        put(
            "fx.csv",
            lambda p: schema.write_csv(
                fx,
                p,
                schema=schema.FX_SCHEMA,
                sort_by=schema.FX_SORT,
                validate=schema.validate_fx,
            ),
        )
        if set(assets) != set(CURRENT_DATA_ASSETS):
            raise ValueError(
                f"current rebuild wrote {sorted(assets)}, expected {sorted(CURRENT_DATA_ASSETS)}"
            )
        # `merged_captures` (Task 14, spec review round 3) is the commit marker
        # `read_or_rebuild_manifest` trusts to decide a bundle is already merged
        # into `current`. A full rebuild starts from nothing and reads every row it
        # writes to `club-gas-all-captures.parquet` right here as `captures`, so
        # the correct value is exactly the capture ids present in that frame -- not
        # whatever a previous, possibly stale or incomplete, manifest.json claimed.
        # Carrying the old list over, or omitting the key, would either resurrect
        # the exact silent-data-loss hole that marker was added to close (a
        # capture claimed done whose rows are not actually in `current`) or make
        # the next incremental publish re-merge bundles it does not need to.
        merged_captures = (
            sorted(captures["capture_id"].unique().to_list()) if not captures.is_empty() else []
        )
        manifest = {
            "schema_version": 1,
            "status": statuses[max(statuses)] if statuses else None,
            "closed_months": sorted(closed_months),
            "closed_years": sorted(m.group(1) for tag in releases if (m := YEAR_TAG.match(tag))),
            "newest_capture_by_feed": _newest_capture_by_feed(statuses),
            "assets": assets,
            "merged_captures": merged_captures,
        }
        # The dashboard's five files are part of `current` too, and a rebuild
        # that wrote only the six above left them describing the dataset as it
        # was BEFORE the rebuild -- and absent from the manifest that the site
        # verifies its download against, so the site could not render at all.
        # Built here exactly as publish._write_current builds them: same inputs,
        # same moment in the manifest's life, so the two writers cannot drift.
        for name, site_path in publish._build_site_assets(
            cfg, work, outputs, manifest, now=now
        ).items():
            store.replace_atomic("current", site_path, name, token)
            assets[name] = {
                "sha256": sha256_file(site_path),
                "size": site_path.stat().st_size,
            }

        path = work / "manifest.json"
        path.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str), "utf-8")
        store.replace_atomic("current", path, "manifest.json", token)


rebuild_current = _rebuild_current_impl


def _refresh_current_periods(store, *, token: str) -> None:
    """Update only the closed-period lists when nothing else about `current` changed."""
    releases = [r.tag for r in store.list_releases()]
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / "manifest.json"
        try:
            store.download("current", "manifest.json", path)
        except AssetNotFound:
            return
        manifest = json.loads(path.read_text("utf-8"))
        manifest["closed_months"] = sorted(
            tag.removeprefix("data-")
            for tag in releases
            if MONTH_TAG.match(tag) and not store.get_release(tag).prerelease
        )
        manifest["closed_years"] = sorted(
            m.group(1) for tag in releases if (m := YEAR_TAG.match(tag))
        )
        path.write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str), "utf-8")
        store.replace_atomic("current", path, "manifest.json", token)


def _year_body(year: str, months: list[str], cfg) -> str:
    listed = "\n".join(f"- `{tag}`" for tag in months)
    return (
        f"# Warehouse-club gas prices {year}\n\n"
        f"Concatenated from the closed month releases:\n\n{listed}\n\n"
        f"- `club-gas-{year}.parquet`, `club-gas-{year}.csv.gz` — daily grain\n"
        f"- `club-gas-{year}-captures.parquet` — capture grain\n"
        f"- `manifest-{year}.json` — the SHA-256 of each input month file\n\n"
        f"Schema: {SCHEMA_URL}\n\n{cfg.site.notice_text()}\n"
    )


def _close_years(store, cfg, *, now: datetime, token: str) -> list[str]:
    releases = {r.tag: r for r in store.list_releases()}
    by_year: dict[str, list[str]] = {}
    for tag in month_tags(store):
        by_year.setdefault(MONTH_TAG.match(tag).group(1), []).append(tag)
    closed: list[str] = []
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for year in sorted(by_year):
            if int(year) >= now.year:
                continue
            months = sorted(by_year[year])
            if any(releases[tag].prerelease for tag in months):
                continue
            shas: dict[str, dict[str, str]] = {}
            paths: dict[str, Path] = {}
            for tag in months:
                month = tag.removeprefix("data-")
                entry = {}
                for name in (
                    f"club-gas-{month}.parquet",
                    f"club-gas-{month}-captures.parquet",
                ):
                    path = store.download(tag, name, work / name)
                    entry[name] = sha256_file(path)
                    paths[name] = path
                shas[month] = entry
            year_tag = f"data-{year}"
            recorded = None
            if year_tag in releases:
                try:
                    path = store.download(
                        year_tag,
                        f"manifest-{year}.json",
                        work / f"manifest-{year}.json",
                    )
                    recorded = json.loads(path.read_text("utf-8")).get("months")
                except AssetNotFound:
                    recorded = None
            if recorded == shas:
                continue
            store.ensure_release(
                year_tag, f"Data {year}", _year_body(year, months, cfg), False, "false"
            )
            grain = pl.concat(
                [
                    pl.read_parquet(paths[f"club-gas-{tag.removeprefix('data-')}.parquet"])
                    for tag in months
                ],
                how="vertical",
            )
            caps = pl.concat(
                [
                    pl.read_parquet(paths[f"club-gas-{tag.removeprefix('data-')}-captures.parquet"])
                    for tag in months
                ],
                how="vertical",
            )
            parquet = work / f"club-gas-{year}.parquet"
            schema.write_parquet(grain, parquet, sort_by=DAILY_SORT)
            store.replace_atomic(year_tag, parquet, parquet.name, token)
            csv_gz = work / f"club-gas-{year}.csv.gz"
            _write_csv_gz(grain.sort(DAILY_SORT), csv_gz)
            store.replace_atomic(year_tag, csv_gz, csv_gz.name, token)
            caps_path = work / f"club-gas-{year}-captures.parquet"
            schema.write_parquet(caps, caps_path, sort_by=CAPTURE_SORT)
            store.replace_atomic(year_tag, caps_path, caps_path.name, token)
            manifest = work / f"manifest-{year}.json"
            manifest.write_text(
                json.dumps(
                    {"schema_version": 1, "year": year, "months": shas},
                    indent=1,
                    sort_keys=True,
                ),
                "utf-8",
            )
            store.replace_atomic(year_tag, manifest, manifest.name, token)
            closed.append(year)
    return closed
