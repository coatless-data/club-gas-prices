"""Rebuild tests (§8.7). No network: every response comes from a stored bundle."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from club_gas.capture import run_capture
from club_gas.config import load_config
from club_gas.http import Client
from club_gas.rebuild import rebuild
from club_gas.rollup import read_month_manifest
from club_gas.schema import (
    FX_SCHEMA,
    FX_SORT,
    STATION_SCHEMA,
    STATION_SORT,
    read_rows_csv_gz,
    validate_fx,
    validate_stations,
    write_csv,
)
from club_gas.store import open_store
from helpers_rebuild import (
    REPO_ROOT,
    au_body,
    checkout_with_config,
    drop_au_e10,
    make_bundle,
    previous_fx,
    previous_stations,
    restore_config,
    seed_month,
)

NOW = datetime(2026, 10, 2, 4, 41, tzinfo=UTC)


def _daily(store, tag, day, tmp_path, name="d.csv.gz"):
    return read_rows_csv_gz(store.download(tag, f"club-gas-{day}.csv.gz", tmp_path / name))


def test_rebuild_applies_the_checkouts_grades_to_stored_responses(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha-one")
    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    bundle = make_bundle(
        tmp_path / "bundles",
        capture_id="2026-09-01T1817Z",
        config_dir=checkout / "config",
    )
    seed_month(
        store,
        "2026-09",
        bundles={"2026-09-01T1817Z": bundle},
        work=tmp_path / "seed",
    )
    drop_au_e10(checkout)

    result = rebuild(store, load_config(checkout), scope="month", value="2026-09", now=NOW)

    assert result.months == ["2026-09"]
    assert result.captures == 1
    assert result.resumed is False
    rows = _daily(store, "data-2026-09", "2026-09-01", tmp_path, "before.csv.gz")
    assert rows.height == 3
    e10 = rows.filter(pl.col("grade_raw") == "E10").row(0, named=True)
    assert e10["station_key"] == "AU-COSTCO-109"
    assert e10["price"] == 2.127
    assert e10["grade"] == "other"

    restore_config(checkout)
    rebuild(store, load_config(checkout), scope="month", value="2026-09", now=NOW)

    rows = _daily(store, "data-2026-09", "2026-09-01", tmp_path, "after.csv.gz")
    e10 = rows.filter(pl.col("grade_raw") == "E10").row(0, named=True)
    assert e10["grade"] == "regular"
    assert e10["price_local_per_litre"] == 2.127
    assert e10["currency"] == "AUD"


class RecordingStore:
    def __init__(self, inner):
        self.inner = inner
        self.log: list[str] = []

    def __getattr__(self, name):
        attr = getattr(self.inner, name)
        if not callable(attr):
            return attr

        def call(*args, **kwargs):
            if name in ("replace_atomic", "upload_new"):
                self.log.append(f"{name}:{args[0]}:{args[2]}")
            elif name == "update_release":
                self.log.append(f"update_release:{args[0]}:prerelease={kwargs.get('prerelease')}")
            return attr(*args, **kwargs)

        return call


def test_rebuild_reopens_a_closed_month_before_rewriting_anything(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha-two")
    store = RecordingStore(open_store(f"local:{tmp_path / 'releases'}"))
    checkout = checkout_with_config(tmp_path)
    bundle = make_bundle(
        tmp_path / "bundles",
        capture_id="2026-09-01T1817Z",
        config_dir=checkout / "config",
    )
    seed_month(
        store,
        "2026-09",
        bundles={"2026-09-01T1817Z": bundle},
        work=tmp_path / "seed",
        prerelease=False,
    )
    store.log.clear()

    rebuild(store, load_config(checkout), scope="month", value="2026-09", now=NOW)

    assert store.get_release("data-2026-09").prerelease is True
    reopened = store.log.index("update_release:data-2026-09:prerelease=True")
    first_write = min(i for i, entry in enumerate(store.log) if entry.startswith("replace_atomic:"))
    assert reopened < first_write


def test_rebuild_stamps_the_manifest_with_the_git_sha_and_config_shas(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha-three")
    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    bundle = make_bundle(
        tmp_path / "bundles",
        capture_id="2026-09-01T1817Z",
        config_dir=checkout / "config",
    )
    seed_month(
        store,
        "2026-09",
        bundles={"2026-09-01T1817Z": bundle},
        work=tmp_path / "seed",
    )

    rebuild(store, load_config(checkout), scope="month", value="2026-09", now=NOW)

    manifest = read_month_manifest(store, "data-2026-09")
    entry = manifest["captures"]["2026-09-01T1817Z"]
    assert entry["rebuild_git_sha"] == "sha-three"
    assert set(entry["config_sha256"]) == {
        "config/grades.csv",
        "config/station_links.csv",
        "config/countries.toml",
    }
    assert entry["rows_by_capture_date"] == {"2026-09-01": 3}
    assert entry["daily_files"]["2026-09-01"]["rows"] == 3


def _two_day_month(store, tmp_path, checkout):
    bundles = {
        "2026-09-01T1817Z": make_bundle(
            tmp_path / "bundles",
            capture_id="2026-09-01T1817Z",
            config_dir=checkout / "config",
            price_e10="$2.127",
        ),
        "2026-09-02T1817Z": make_bundle(
            tmp_path / "bundles",
            capture_id="2026-09-02T1817Z",
            config_dir=checkout / "config",
            price_e10="$2.187",
        ),
    }
    return seed_month(store, "2026-09", bundles=bundles, work=tmp_path / "seed")


def _put_state(store, tag, state, tmp_path):
    path = tmp_path / "rebuild-state.json"
    path.write_text(json.dumps(state, indent=1, sort_keys=True), "utf-8")
    store.replace_atomic(tag, path, "rebuild-state.json", "run-test")


def test_rebuild_resumes_after_the_last_completed_day_when_everything_matches(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("GITHUB_SHA", "sha-four")
    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    _two_day_month(store, tmp_path, checkout)
    cfg = load_config(checkout)
    from club_gas.rebuild import interp_config_sha256

    _put_state(
        store,
        "data-2026-09",
        {
            "schema_version": 1,
            "scope": "month",
            "value": "2026-09",
            "git_sha": "sha-four",
            "config_sha256": interp_config_sha256(cfg),
            "last_completed_day": "2026-09-01",
            "complete": False,
        },
        tmp_path,
    )

    result = rebuild(store, cfg, scope="month", value="2026-09", now=NOW)

    assert result.resumed is True
    assert result.captures == 1
    day_one = _daily(store, "data-2026-09", "2026-09-01", tmp_path, "one.csv.gz")
    assert day_one.height == 0
    day_two = _daily(store, "data-2026-09", "2026-09-02", tmp_path, "two.csv.gz")
    assert day_two.height == 3
    state = json.loads(
        store.download("data-2026-09", "rebuild-state.json", tmp_path / "state.json").read_text()
    )
    assert state["complete"] is True
    assert state["last_completed_day"] == "2026-09-02"


def test_rebuild_restarts_when_the_git_sha_differs(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha-new")
    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    _two_day_month(store, tmp_path, checkout)
    cfg = load_config(checkout)
    from club_gas.rebuild import interp_config_sha256

    _put_state(
        store,
        "data-2026-09",
        {
            "schema_version": 1,
            "scope": "month",
            "value": "2026-09",
            "git_sha": "sha-old",
            "config_sha256": interp_config_sha256(cfg),
            "last_completed_day": "2026-09-01",
            "complete": False,
        },
        tmp_path,
    )

    result = rebuild(store, cfg, scope="month", value="2026-09", now=NOW)

    assert result.resumed is False
    assert result.captures == 2
    assert _daily(store, "data-2026-09", "2026-09-01", tmp_path, "r1.csv.gz").height == 3


def test_a_second_rebuild_after_a_grades_change_reprocesses_every_day(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha-five")
    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    _two_day_month(store, tmp_path, checkout)
    drop_au_e10(checkout)
    first = rebuild(store, load_config(checkout), scope="month", value="2026-09", now=NOW)
    assert first.captures == 2

    restore_config(checkout)
    second = rebuild(store, load_config(checkout), scope="month", value="2026-09", now=NOW)

    assert second.resumed is False
    assert second.captures == 2
    for day in ("2026-09-01", "2026-09-02"):
        rows = _daily(store, "data-2026-09", day, tmp_path, f"{day}.csv.gz")
        assert rows.filter(pl.col("grade_raw") == "E10")["grade"].to_list() == ["regular"]


def test_rebuild_of_an_open_month_then_refreshes_current(tmp_path, monkeypatch):
    monkeypatch.setenv("GITHUB_SHA", "sha-six")
    from club_gas.rollup import rebuild_current

    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    _two_day_month(store, tmp_path, checkout)
    cfg = load_config(checkout)
    rebuild(store, cfg, scope="month", value="2026-09", now=NOW)

    rebuild_current(store, cfg, now=NOW)

    stations = pl.read_csv(store.download("current", "stations.csv", tmp_path / "stations.csv"))
    assert "AU-COSTCO-109" in stations["station_key"].to_list()
    captures = pl.read_parquet(
        store.download("current", "club-gas-all-captures.parquet", tmp_path / "all.parquet")
    )
    assert captures.height == 6
    fx = pl.read_csv(store.download("current", "fx.csv", tmp_path / "fx.csv"))
    assert fx["currency"].to_list() == ["AUD", "AUD"]
    manifest = json.loads(
        store.download("current", "manifest.json", tmp_path / "m.json").read_text()
    )
    assert manifest["newest_capture_by_feed"] == {"AU-COSTCO": "2026-09-02T1817Z"}


def test_rebuild_interrupted_mid_month_lets_close_periods_close_cleanly_after_resume(
    tmp_path, monkeypatch
):
    """A crash between two days must never leave the manifest describing a day
    differently from the daily file `rebuild` already published for it: that
    disagreement used to make `close_periods`'s row-count cross-check raise and
    abort the whole close, including every other month in the same invocation
    (spec review, Task 16 round 2)."""
    monkeypatch.setenv("GITHUB_SHA", "sha-seven")
    store = open_store(f"local:{tmp_path / 'releases'}")
    checkout = checkout_with_config(tmp_path)
    _two_day_month(store, tmp_path, checkout)
    cfg = load_config(checkout)

    import club_gas.rebuild as rebuild_module

    real_rebuild_capture = rebuild_module._rebuild_capture
    calls = {"n": 0}

    def flaky(store_, cfg_, tag, capture_id, work):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("simulated crash mid-month")
        return real_rebuild_capture(store_, cfg_, tag, capture_id, work)

    monkeypatch.setattr(rebuild_module, "_rebuild_capture", flaky)

    with pytest.raises(RuntimeError, match="simulated crash mid-month"):
        rebuild(store, cfg, scope="month", value="2026-09", now=NOW)

    # Day one is already durable: its daily file and its manifest entry agree,
    # even though the "crash" happened while day two was only just starting.
    manifest = read_month_manifest(store, "data-2026-09")
    entry_one = manifest["captures"]["2026-09-01T1817Z"]
    assert entry_one["rows_by_capture_date"] == {"2026-09-01": 3}
    day_one = _daily(store, "data-2026-09", "2026-09-01", tmp_path, "mid.csv.gz")
    assert day_one.height == entry_one["daily_files"]["2026-09-01"]["rows"] == 3

    monkeypatch.setattr(rebuild_module, "_rebuild_capture", real_rebuild_capture)
    result = rebuild(store, cfg, scope="month", value="2026-09", now=NOW)
    assert result.resumed is True
    assert result.captures == 1

    from club_gas.rollup import close_periods

    close_result = close_periods(store, cfg, now=datetime(2026, 11, 2, tzinfo=UTC))

    assert close_result.closed_months == ["2026-09"]
    assert close_result.blocked_months == []


# --------------------------------------------------------------- producer/consumer

CAPTURE_NOW = datetime(2026, 9, 1, 18, 17, 40, tzinfo=UTC)
CAPTURE_ID = "2026-09-01T1817Z"
FX_BODY = (REPO_ROOT / "tests" / "fixtures" / "fx" / "frankfurter_v2.json").read_bytes()


def _capture_transport():
    """Serve the AU stores fixture and the Frankfurter rates; 404 everything else."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "api.frankfurter.dev":
            return httpx.Response(
                200, content=FX_BODY, headers={"Content-Type": "application/json"}
            )
        if host == "www.costco.com.au":
            return httpx.Response(
                200, content=au_body(), headers={"Content-Type": "application/json"}
            )
        return httpx.Response(404, content=b"{}")

    return httpx.MockTransport(handler)


def _seed_current_previous_state(store, work: Path, capture_id: str) -> None:
    """`current/stations.csv` and `current/fx.csv`, written the way publish writes them."""
    store.ensure_release("current", "current", "", False, "true")
    work.mkdir(parents=True, exist_ok=True)
    stations = work / "stations.csv"
    write_csv(
        previous_stations(capture_id),
        stations,
        schema=STATION_SCHEMA,
        sort_by=STATION_SORT,
        validate=validate_stations,
    )
    store.upload_new("current", stations, "stations.csv")
    fx = work / "fx.csv"
    write_csv(previous_fx(capture_id), fx, schema=FX_SCHEMA, sort_by=FX_SORT, validate=validate_fx)
    store.upload_new("current", fx, "fx.csv")


@pytest.mark.parametrize("with_previous_state", [True, False])
def test_rebuild_reads_a_bundle_written_by_the_real_capture_path(
    tmp_path, monkeypatch, with_previous_state
):
    """§8.7 end to end: `capture._write_bundle` is the producer, `rebuild` the consumer.

    Every other test here hands `rebuild` a bundle the helpers built. This one
    runs `run_capture` for real, takes the `bundle.tar.gz` it wrote, and rebuilds
    from that -- the only shape of test in which a format break between the two
    shows up at all.

    Both previous-state cases have to round-trip. Without a readable `current`
    -- the very first capture -- `ctx.previous_stations` is a frame with no
    columns at all, and `write_csv` turns that into a bare newline: one byte, so
    a `st_size == 0` guard does not see it as empty and the reader gets handed a
    file with no header to match its schema against.
    """
    monkeypatch.setenv("GITHUB_SHA", "sha-round-trip")
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    checkout = checkout_with_config(tmp_path)
    (checkout / "status").mkdir(exist_ok=True)
    monkeypatch.chdir(checkout)
    store = open_store(f"local:{tmp_path / 'releases'}")
    if with_previous_state:
        _seed_current_previous_state(store, tmp_path / "seed-current", CAPTURE_ID)
    cfg = load_config(checkout)

    result = run_capture(
        cfg,
        store,
        tmp_path / "out",
        countries=["AU"],
        force_fallback=set(),
        now=CAPTURE_NOW,
        client=Client(cfg.http, transport=_capture_transport()),
    )

    assert result.capture_id == CAPTURE_ID
    assert result.status["feeds"]["AU-COSTCO"]["rows"] == 3
    unavailable = "previous_state_unavailable" in {w["code"] for w in result.status["warnings"]}
    assert unavailable is not with_previous_state
    stored = result.out.parent / "bundle" / "inputs" / "stations_used.csv"
    assert stored.read_bytes() != b"" and (stored.stat().st_size == 1) is not with_previous_state

    seed_month(
        store,
        "2026-09",
        bundles={CAPTURE_ID: result.out / "bundle.tar.gz"},
        work=tmp_path / "seed",
    )

    rebuilt = rebuild(store, cfg, scope="month", value="2026-09", now=NOW)

    assert rebuilt.months == ["2026-09"]
    assert rebuilt.captures == 1
    rows = _daily(store, "data-2026-09", "2026-09-01", tmp_path, "round-trip.csv.gz")
    assert rows.height == 3
    assert rows["capture_id"].unique().to_list() == [CAPTURE_ID]
    e10 = rows.filter(pl.col("grade_raw") == "E10").row(0, named=True)
    assert e10["station_key"] == "AU-COSTCO-109"
    assert e10["price"] == 2.127
    assert e10["currency"] == "AUD"


def test_a_bundle_whose_columns_moved_is_refused_not_misread(tmp_path: Path):
    """`schema=` assigns by position, so the header is the only thing that can object.

    These frames are read back years after they were written. A column order
    that no longer matches would load each value into its neighbour among the
    adjacent String columns — silently, and into published data.
    """
    from club_gas.rebuild import _read_input_csv
    from club_gas.schema import FX_SCHEMA

    names = list(FX_SCHEMA)
    swapped = [names[1], names[0], *names[2:]]
    path = tmp_path / "fx_used.csv"
    path.write_text(",".join(swapped) + "\n")

    with pytest.raises(ValueError) as exc:
        _read_input_csv(path, FX_SCHEMA)
    assert "column order" in str(exc.value)

    path.write_text(",".join(names) + "\n")
    assert _read_input_csv(path, FX_SCHEMA).height == 0


def test_a_response_group_no_source_claims_refuses_the_rebuild(tmp_path, monkeypatch):
    """A group the current SOURCES does not know is what a renamed dispatch key
    looks like from inside a rebuild.

    The old behaviour was `continue`, which would have dropped every row for
    that group out of a closed month while reporting success -- and the bundle
    is the only copy the data has. It refuses instead.
    """
    monkeypatch.setenv("GITHUB_SHA", "sha-unknown-group")
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    checkout = checkout_with_config(tmp_path)
    (checkout / "status").mkdir(exist_ok=True)
    monkeypatch.chdir(checkout)
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = load_config(checkout)

    result = run_capture(
        cfg,
        store,
        tmp_path / "out",
        countries=["AU"],
        force_fallback=set(),
        now=CAPTURE_NOW,
        client=Client(cfg.http, transport=_capture_transport()),
    )
    seed_month(
        store,
        "2026-09",
        bundles={CAPTURE_ID: result.out / "bundle.tar.gz"},
        work=tmp_path / "seed",
    )

    # The bundle holds AU-COSTCO responses; a SOURCES that knows only some other
    # feed is what renaming a dispatch key looks like to a later rebuild.
    import club_gas.rebuild as rebuild_module

    monkeypatch.setattr(rebuild_module, "SOURCES", {"AU-SAMS": object()})

    with pytest.raises(RuntimeError, match="which no source claims"):
        rebuild(store, cfg, scope="month", value="2026-09", now=NOW)
