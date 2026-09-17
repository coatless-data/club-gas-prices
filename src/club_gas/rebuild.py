"""Rebuild history from stored capture bundles (§8.7).

Fetch-time inputs (URLs, parameters, `us_extra_ids.csv`, `http.toml`, and the
`inputs/` frames) come from the bundle, so a rebuild can never change which
stations were polled. Interpretation config (`grades.csv`, `station_links.csv`,
and the units, bounds, region and timezone tables, floors and `stale_after_days`
in `countries.toml`) comes from the live checkout, which is what makes a config
fix reach history. Only `parse` and `normalize` re-run; no request is replayed.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import tarfile
import tempfile
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path

import polars as pl

from .fx import FxRates
from .normalize import normalize
from .rollup import month_tags, read_month_manifest, write_month_manifest
from .schema import (
    FX_SCHEMA,
    ROW_SCHEMA,
    ROW_SORT,
    STATION_SCHEMA,
    validate_rows,
    write_rows_csv_gz,
)
from .sources.base import SOURCES, CaptureContext, RawResponse
from .store import AssetNotFound, recover_temporaries, recovery_tags, sha256_file

STATE_ASSET = "rebuild-state.json"
INTERP_CONFIG_FILES = ("grades.csv", "station_links.csv", "countries.toml")


@dataclass
class RebuildResult:
    months: list[str] = field(default_factory=list)
    captures: int = 0
    resumed: bool = False


def git_sha() -> str:
    sha = os.environ.get("GITHUB_SHA")
    if sha:
        return sha
    try:
        done = subprocess.run(
            ["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True
        )
        return done.stdout.strip()
    except Exception:
        return "unknown"


def interp_config_sha256(cfg) -> dict[str, str]:
    """Hash the live checkout's interpretation config; `cfg.root` is that checkout."""
    out: dict[str, str] = {}
    for name in INTERP_CONFIG_FILES:
        path = cfg.root / "config" / name
        out[f"config/{name}"] = hashlib.sha256(path.read_bytes()).hexdigest()
    return out


def _token(now: datetime) -> str:
    return "run-" + now.strftime("%Y-%m-%dT%H%MZ")


def _parse_utc(text: str) -> datetime:
    return datetime.fromisoformat(text.replace("Z", "+00:00"))


def _read_input_csv(path: Path, schema: dict) -> pl.DataFrame:
    """Read one `inputs/` frame back from a capture bundle.

    The producer is `capture._write_bundle`, which writes these with a bare
    `DataFrame.write_csv`: its datetimes carry Polars' own text
    (`2026-09-15T18:17:40.000000+0000`), not `schema.CSV_DATETIME_FORMAT`.
    `pl.read_csv(schema=...)` parses both forms; the strict `schema.read_csv`
    parses only the canonical one and raises `InvalidOperationError` on the
    other, which would abort every rebuild of a real bundle.

    A capture with no previous state writes a frame with no data rows: a header
    line on its own, or a bare newline when the frame has no columns either
    (`pl.DataFrame().write_csv()`, the >0-byte case the old `st_size` guard
    missed). Both mean "empty", and only the second would reach `pl.read_csv`,
    which raises `NoDataError` on it.

    `schema=` assigns columns by position and ignores the header, so the header
    is checked first. These bundles are read years after they were written: if
    a future column order ever disagrees with the one a bundle was written
    with, that has to fail loudly here rather than quietly load a column's
    values into its neighbour.
    """
    if not path.exists():
        return pl.DataFrame(schema=schema)
    raw = path.read_bytes()
    if not raw.strip():
        return pl.DataFrame(schema=schema)
    header = raw.split(b"\n", 1)[0].decode("utf-8").strip().split(",")
    if header != list(schema):
        raise ValueError(f"{path}: column order is {header}, expected {list(schema)}")
    return pl.read_csv(raw, schema=schema)


def _load_responses(
    root: Path,
) -> tuple[dict[str, RawResponse], dict[str, list[RawResponse]]]:
    shared: dict[str, RawResponse] = {}
    by_country: dict[str, list[RawResponse]] = {}
    if not root.exists():
        return shared, by_country
    for meta_path in sorted(root.rglob("*.meta.json")):
        meta = json.loads(meta_path.read_text("utf-8"))
        stem = meta_path.name[: -len(".meta.json")]
        body_path = meta_path.with_name(f"{stem}.body")
        response = RawResponse(
            key=meta["key"],
            url=meta["url"],
            status=meta.get("status"),
            headers=meta.get("headers") or {},
            received_at_utc=_parse_utc(meta["received_at_utc"]),
            elapsed_ms=int(meta.get("elapsed_ms") or 0),
            body=body_path.read_bytes() if body_path.exists() else b"",
            error=meta.get("error"),
        )
        group = meta_path.relative_to(root).parts[0]
        if group == "shared":
            shared[stem] = response
        else:
            by_country.setdefault(group, []).append(response)
    for responses in by_country.values():
        responses.sort(key=lambda r: r.key)
    return shared, by_country


def load_bundle_config(bundle_root: Path):
    from .config import load_config

    return load_config(bundle_root).fetch_view()


def _rebuild_capture(store, cfg, tag: str, capture_id: str, work: Path) -> pl.DataFrame:
    name = f"capture-{capture_id}.tar.gz"
    archive = work / name
    try:
        store.download(tag, name, archive)
    except AssetNotFound as exc:
        raise ValueError(f"{tag}: bundle {name} is missing, cannot rebuild") from exc
    root = work / f"bundle-{capture_id}"
    if root.exists():
        shutil.rmtree(root)
    root.mkdir(parents=True)
    with tarfile.open(archive) as tar:
        tar.extractall(root, filter="data")
    meta = json.loads((root / "capture.json").read_text("utf-8"))
    capture_date = date.fromisoformat(meta["capture_date"])
    status_doc = json.loads((root / "status.json").read_text("utf-8"))
    fx = FxRates.from_json(
        json.loads((root / "fx.json").read_text("utf-8")),
        status=(status_doc.get("fx") or {}).get("status") or "failed",
        capture_date=capture_date,
    )
    previous_status = None
    status_path = root / "inputs" / "status_previous.json"
    if status_path.exists() and status_path.stat().st_size:
        previous_status = json.loads(status_path.read_text("utf-8"))
    shared, by_country = _load_responses(root / "responses")
    ctx = CaptureContext(
        capture_id=meta["capture_id"],
        capture_date=capture_date,
        fetch_config=load_bundle_config(root),
        interp_config=cfg.interp_view(),
        previous_stations=_read_input_csv(root / "inputs" / "stations_used.csv", STATION_SCHEMA),
        previous_fx=_read_input_csv(root / "inputs" / "fx_used.csv", FX_SCHEMA),
        previous_status=previous_status,
        shared=shared,
        force_fallback=set(),
    )
    frames: list[pl.DataFrame] = []
    for country in sorted(by_country):
        source = SOURCES.get(country)
        if source is None:
            continue
        try:
            result = source.parse(by_country[country], ctx)
            out = normalize(result, fx, ctx)
        except Exception as exc:
            raise RuntimeError(f"{capture_id} {country}: {exc}") from exc
        if not out.rows.is_empty():
            frames.append(out.rows.select(list(ROW_SCHEMA)).cast(ROW_SCHEMA))
    if not frames:
        return pl.DataFrame(schema=ROW_SCHEMA)
    return pl.concat(frames, how="vertical")


def _read_state(store, tag: str) -> dict | None:
    with tempfile.TemporaryDirectory() as td:
        try:
            path = store.download(tag, STATE_ASSET, Path(td) / STATE_ASSET)
        except AssetNotFound:
            return None
        return json.loads(path.read_text("utf-8"))


def _write_state(store, tag: str, state: dict, *, token: str) -> None:
    with tempfile.TemporaryDirectory() as td:
        path = Path(td) / STATE_ASSET
        path.write_text(json.dumps(state, indent=1, sort_keys=True), "utf-8")
        store.replace_atomic(tag, path, STATE_ASSET, token)


def _in_scope(tag: str, scope: str, value: str | None) -> bool:
    if scope == "month":
        return tag == f"data-{value}"
    if scope == "year":
        return tag.startswith(f"data-{value}-")
    return True


def _rebuild_month(
    store, cfg, tag: str, *, scope, value, sha, config_sha, token
) -> tuple[int, bool]:
    manifest = read_month_manifest(store, tag)
    entries = manifest.get("captures") or {}
    days = sorted(
        {day for entry in entries.values() for day in (entry.get("rows_by_capture_date") or {})}
    )
    state = _read_state(store, tag)
    resumed = False
    if (
        state
        and state.get("complete") is False
        and state.get("scope") == scope
        and state.get("value") == value
        and state.get("git_sha") == sha
        and state.get("config_sha256") == config_sha
        and state.get("last_completed_day")
    ):
        resumed = True
        days = [day for day in days if day > state["last_completed_day"]]
    count = 0
    with tempfile.TemporaryDirectory() as td:
        work = Path(td)
        for day in days:
            capture_ids = sorted(
                cid
                for cid, entry in entries.items()
                if day in (entry.get("rows_by_capture_date") or {})
            )
            frames = []
            for capture_id in capture_ids:
                rows = _rebuild_capture(store, cfg, tag, capture_id, work)
                frames.append(rows)
                entries[capture_id]["rows_by_capture_date"][day] = rows.height
                entries[capture_id]["rebuild_git_sha"] = sha
                entries[capture_id]["config_sha256"] = dict(config_sha)
                count += 1
            merged = (
                pl.concat(frames, how="vertical") if frames else pl.DataFrame(schema=ROW_SCHEMA)
            )
            merged = merged.sort(ROW_SORT)
            validate_rows(merged)
            name = f"club-gas-{day}.csv.gz"
            path = work / name
            write_rows_csv_gz(merged, path)
            store.replace_atomic(tag, path, name, token)
            info = {"sha256": sha256_file(path), "rows": merged.height}
            for capture_id in capture_ids:
                entries[capture_id].setdefault("daily_files", {})[day] = info
            # The manifest is persisted for THIS day before the checkpoint is
            # allowed to move past it, not once at the end of the whole month.
            # A crash between these two writes just makes the next run redo
            # this one day (idempotent, safe); persisting the manifest only
            # once after the entire loop, as before, could leave a completed
            # checkpoint pointing past a day whose manifest entry still held
            # its pre-rebuild row count and hash -- disagreeing with the daily
            # file `replace_atomic` had already made durable, which is exactly
            # what makes `_close_month`'s cross-check raise (spec review,
            # Task 16 round 2).
            write_month_manifest(store, tag, manifest, token=token)
            _write_state(
                store,
                tag,
                {
                    "schema_version": 1,
                    "scope": scope,
                    "value": value,
                    "git_sha": sha,
                    "config_sha256": config_sha,
                    "last_completed_day": day,
                    "complete": False,
                },
                token=token,
            )
        _write_state(
            store,
            tag,
            {
                "schema_version": 1,
                "scope": scope,
                "value": value,
                "git_sha": sha,
                "config_sha256": config_sha,
                "last_completed_day": days[-1] if days else None,
                "complete": True,
            },
            token=token,
        )
    return count, resumed


def rebuild(store, cfg, *, scope: str, value: str | None, now: datetime) -> RebuildResult:
    if scope not in ("month", "year", "all"):
        raise ValueError(f"unknown scope: {scope}")
    if scope in ("month", "year") and not value:
        raise ValueError(f"scope {scope} needs a value")
    token = _token(now)
    for recovery_tag in recovery_tags(store):
        recover_temporaries(store, recovery_tag)
    tags = [tag for tag in month_tags(store) if _in_scope(tag, scope, value)]
    if not tags:
        return RebuildResult()
    releases = {r.tag: r for r in store.list_releases()}
    for tag in tags:  # step 1: reopen every closed month before anything is rewritten
        if not releases[tag].prerelease:
            store.update_release(tag, prerelease=True, make_latest="false")
    sha = git_sha()
    config_sha = interp_config_sha256(cfg)
    result = RebuildResult()
    for tag in tags:
        count, resumed = _rebuild_month(
            store,
            cfg,
            tag,
            scope=scope,
            value=value,
            sha=sha,
            config_sha=config_sha,
            token=token,
        )
        result.months.append(tag.removeprefix("data-"))
        result.captures += count
        result.resumed = result.resumed or resumed
    return result
