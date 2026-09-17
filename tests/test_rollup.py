"""Roll-up tests. LocalReleaseStore only, no network."""

from __future__ import annotations

import json
from datetime import UTC, date, datetime

import polars as pl
import pytest

from club_gas import rollup
from club_gas.rollup import close_periods, daily_grain, rebuild_current
from club_gas.store import next_name, open_store, sha256_file, sha256_label
from helpers_rollup import (
    RecordingIssues,
    RecordingStore,
    price_row,
    rows_frame,
    seed_month,
    stations_frame,
    stub_config,
    write_csv_gz,
)

CURRENT_DATA_ASSET_NAMES = {
    "club-gas-all.parquet",
    "club-gas-all.csv.gz",
    "club-gas-all-captures.parquet",
    "club-gas-latest.csv",
    "stations.csv",
    "fx.csv",
}
# The dashboard downloads these five from `current` and verifies each against
# manifest.json, so a rebuild that leaves them out of the manifest -- or leaves
# them describing the pre-rebuild dataset -- takes the site down.
CURRENT_ASSET_NAMES = CURRENT_DATA_ASSET_NAMES | {
    "site-meta.json",
    "site-latest.json",
    "site-stations.json",
    "site-summary-daily.parquet",
    "site-history.parquet",
}
CURRENT_MANIFEST_KEYS = {
    "schema_version",
    "status",
    "closed_months",
    "closed_years",
    "newest_capture_by_feed",
    "assets",
}


def test_daily_grain_keeps_the_last_capture_of_the_day():
    rows = rows_frame(
        [
            price_row(
                capture_id="2026-09-15T0017Z",
                station_key="US-COSTCO-1364",
                grade_raw="regular",
                grade="regular",
                price=3.899,
            ),
            price_row(
                capture_id="2026-09-15T1817Z",
                station_key="US-COSTCO-1364",
                grade_raw="regular",
                grade="regular",
                price=3.999,
            ),
            price_row(
                capture_id="2026-09-15T1817Z",
                station_key="US-COSTCO-1364",
                grade_raw="premium",
                grade="premium",
                price=4.629,
            ),
            price_row(
                capture_id="2026-09-16T0017Z",
                station_key="US-COSTCO-1364",
                grade_raw="regular",
                grade="regular",
                price=4.099,
            ),
        ]
    )

    out = daily_grain(rows)

    assert out.height == 3
    day15 = out.filter(
        (pl.col("capture_date") == date(2026, 9, 15)) & (pl.col("grade_raw") == "regular")
    ).row(0, named=True)
    assert day15["capture_id"] == "2026-09-15T1817Z"
    assert day15["price"] == 3.999
    assert day15["n_captures"] == 2
    assert day15["price_min"] == 3.899
    assert day15["price_max"] == 3.999
    premium = out.filter(pl.col("grade_raw") == "premium").row(0, named=True)
    assert premium["n_captures"] == 1
    assert premium["price_min"] == 4.629
    assert premium["price_max"] == 4.629
    day16 = out.filter(pl.col("capture_date") == date(2026, 9, 16)).row(0, named=True)
    assert day16["price"] == 4.099
    assert day16["n_captures"] == 1


def _us_rows(capture_id: str, price: float = 3.999) -> pl.DataFrame:
    return rows_frame(
        [
            price_row(
                capture_id=capture_id,
                station_key="US-COSTCO-1364",
                grade_raw="regular",
                grade="regular",
                price=price,
            ),
            price_row(
                capture_id=capture_id,
                station_key="US-COSTCO-1364",
                grade_raw="premium",
                grade="premium",
                price=price + 0.63,
            ),
        ]
    )


def _status(capture_id: str) -> dict:
    return {
        "schema_version": 1,
        "capture_id": capture_id,
        "feeds": {"US-COSTCO": {"status": "ok", "rows": 2}},
    }


def test_close_periods_blocks_a_month_that_has_an_orphan_bundle(tmp_path):
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    seed_month(
        store,
        "2026-08",
        {"2026-08-31": _us_rows("2026-08-31T1817Z")},
        captures={
            "2026-08-31T1817Z": {
                "status": _status("2026-08-31T1817Z"),
                "rows_by_capture_date": {"2026-08-31": 2},
            }
        },
        orphan_bundles=["capture-2026-08-31T2317Z.tar.gz"],
        work=tmp_path / "seed-08",
    )
    issues = RecordingIssues()

    result = close_periods(store, cfg, now=datetime(2026, 9, 15, 18, 17, tzinfo=UTC), issues=issues)

    assert result.blocked_months == ["2026-08"]
    assert result.closed_months == []
    assert store.get_release("data-2026-08").prerelease is True
    assert issues.opened[0][0] == "Period close blocked: 2026-08"
    assert "capture-2026-08-31T2317Z.tar.gz" in issues.opened[0][1]


def _seed_august(store, tmp_path, *, fx=None):
    return seed_month(
        store,
        "2026-08",
        {
            "2026-08-30": _us_rows("2026-08-30T1817Z", 3.899),
            "2026-08-31": _us_rows("2026-08-31T1817Z", 3.999),
        },
        captures={
            "2026-08-30T1817Z": {
                "status": _status("2026-08-30T1817Z"),
                "rows_by_capture_date": {"2026-08-30": 2},
                "fx": fx or [],
            },
            "2026-08-31T1817Z": {
                "status": _status("2026-08-31T1817Z"),
                "rows_by_capture_date": {"2026-08-31": 2},
                "fx": fx or [],
            },
        },
        work=tmp_path / "seed-08",
    )


def test_close_periods_blocks_a_month_whose_recorded_day_has_no_uploaded_file(tmp_path):
    """Finding 1 (spec review round 4): a day the manifest records rows for, but
    whose daily file is missing, must block the month -- `_close_month`'s own
    row-count checks compare `captures.height` and `per_file` only over days
    that ARE present, so a missing day would otherwise shrink both sides
    together and close the month short."""
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    _seed_august(store, tmp_path)
    missing = next(
        a for a in store.list_assets("data-2026-08") if a.name == "club-gas-2026-08-31.csv.gz"
    )
    store.delete("data-2026-08", missing.id)
    issues = RecordingIssues()

    result = close_periods(store, cfg, now=datetime(2026, 9, 1, 3, 17, tzinfo=UTC), issues=issues)

    assert result.blocked_months == ["2026-08"]
    assert result.closed_months == []
    assert store.get_release("data-2026-08").prerelease is True
    assert issues.opened[0][0] == "Period close blocked: 2026-08"
    assert "2026-08-31" in issues.opened[0][1]
    names = {a.name for a in store.list_assets("data-2026-08")}
    assert "club-gas-2026-08.parquet" not in names


def test_month_close_writes_both_grains_and_clears_prerelease_last(tmp_path):
    store = RecordingStore(open_store(f"local:{tmp_path / 'releases'}"))
    cfg = stub_config(tmp_path)
    _seed_august(store, tmp_path)

    result = close_periods(
        store,
        cfg,
        now=datetime(2026, 9, 1, 3, 17, tzinfo=UTC),
        issues=RecordingIssues(),
    )

    assert result.closed_months == ["2026-08"]
    names = {a.name for a in store.list_assets("data-2026-08")}
    assert {
        "club-gas-2026-08.parquet",
        "club-gas-2026-08.csv.gz",
        "club-gas-2026-08-captures.parquet",
    } <= names
    captures = pl.read_parquet(
        store.download(
            "data-2026-08",
            "club-gas-2026-08-captures.parquet",
            tmp_path / "caps.parquet",
        )
    )
    assert captures.height == 4
    grain = pl.read_parquet(
        store.download("data-2026-08", "club-gas-2026-08.parquet", tmp_path / "grain.parquet")
    )
    assert grain.height == 4
    assert set(grain["n_captures"].to_list()) == {1}
    assert store.get_release("data-2026-08").prerelease is False
    cleared = store.log.index("update_release:data-2026-08:prerelease=False")
    assert cleared > store.log.index(
        "replace_atomic:data-2026-08:club-gas-2026-08-captures.parquet"
    )
    assert cleared > store.log.index("replace_atomic:current:manifest.json")


def test_month_close_refreshes_current_manifest_closed_months_in_the_same_run(tmp_path):
    """Finding 2 (spec review round 4): the data rebuild runs while the month is
    still prerelease (so an interrupted close simply retries), which means
    current/manifest.json's closed_months cannot list this month yet by the time
    that rebuild runs. It must be refreshed again right after `prerelease` is
    cleared, in the same close_periods call -- not left stale until the next
    close or an explicit --rebuild-current."""
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    _seed_august(store, tmp_path)

    result = close_periods(
        store, cfg, now=datetime(2026, 9, 1, 3, 17, tzinfo=UTC), issues=RecordingIssues()
    )

    assert result.closed_months == ["2026-08"]
    manifest = json.loads(
        store.download("current", "manifest.json", tmp_path / "m.json").read_text()
    )
    assert manifest["closed_months"] == ["2026-08"]


def test_month_close_is_safe_to_repeat(tmp_path):
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    _seed_august(store, tmp_path)
    now = datetime(2026, 9, 1, 3, 17, tzinfo=UTC)
    close_periods(store, cfg, now=now, issues=RecordingIssues())
    before = {a.name: a.size for a in store.list_assets("data-2026-08")}

    again = close_periods(store, cfg, now=now, issues=RecordingIssues())

    assert again.closed_months == []
    assert {a.name: a.size for a in store.list_assets("data-2026-08")} == before


def test_close_periods_blocks_a_month_whose_daily_file_disagrees_with_the_manifest(tmp_path):
    """A row-count mismatch between a daily file and the month manifest must
    block the month via the same `_blocking_reasons` path as every other
    refusal, not raise out of `close_periods` and abort the rest of the
    invocation (spec review, Task 16 round 2)."""
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    manifest = _seed_august(store, tmp_path)
    manifest["captures"]["2026-08-31T1817Z"]["daily_files"]["2026-08-31"]["rows"] = 99
    path = tmp_path / "manifest-2026-08.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), "utf-8")
    store.replace_atomic("data-2026-08", path, "manifest-2026-08.json", "run-test")
    issues = RecordingIssues()

    result = close_periods(
        store,
        cfg,
        now=datetime(2026, 9, 1, 3, 17, tzinfo=UTC),
        issues=issues,
    )

    assert result.blocked_months == ["2026-08"]
    assert result.closed_months == []
    assert store.get_release("data-2026-08").prerelease is True
    assert issues.opened[0][0] == "Period close blocked: 2026-08"
    assert "manifest records 99" in issues.opened[0][1]


def test_a_close_refuses_a_capture_grain_that_lost_rows_the_daily_files_hold(tmp_path, monkeypatch):
    """Spec 8.6 step 2, an explicit acceptance criterion (§14.6): the
    capture-grain row count must equal the sum of the daily files' rows.

    Today `_close_month` builds that frame by concatenating the same daily
    files it counts, so the two sides agree by construction and no seeded
    month can make them differ -- which is exactly why the check reads as
    redundant and why deleting it costs nothing until the day the frame stops
    coming straight from those files. The month files are the permanent copy of
    that history and `prerelease` is cleared right after them, so a close that
    wrote a short capture-grain file would freeze the loss. Here the read is
    made to drop one row, the way a partial read or a filtered concat would,
    with `per_file` still matching the manifest so that no other check can
    fire.
    """
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    _seed_august(store, tmp_path)
    real = rollup._read_daily_files

    def short(store_arg, tag, work):
        frame, per_file = real(store_arg, tag, work)
        return frame.head(frame.height - 1), per_file

    monkeypatch.setattr(rollup, "_read_daily_files", short)

    with pytest.raises(ValueError, match=r"capture grain has 3 rows, the daily files hold 4"):
        close_periods(store, cfg, now=datetime(2026, 9, 1, 3, 17, tzinfo=UTC))

    # Nothing was written and the month is still open, so a fixed run can retry.
    names = {a.name for a in store.list_assets("data-2026-08")}
    assert "club-gas-2026-08.parquet" not in names
    assert "club-gas-2026-08-captures.parquet" not in names
    assert store.get_release("data-2026-08").prerelease is True


def test_a_mismatched_month_does_not_block_a_healthy_month_in_the_same_run(tmp_path):
    """The mismatch in `data-2026-08` must not stop `data-2026-09`, seeded
    clean, from closing in the same `close_periods` invocation -- and it must
    sort before it, so the old uncaught `ValueError` would have aborted the
    whole call before `2026-09` was ever reached (spec review, Task 16
    round 2)."""
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    manifest = _seed_august(store, tmp_path)
    manifest["captures"]["2026-08-31T1817Z"]["daily_files"]["2026-08-31"]["rows"] = 99
    path = tmp_path / "manifest-2026-08.json"
    path.write_text(json.dumps(manifest, indent=1, sort_keys=True), "utf-8")
    store.replace_atomic("data-2026-08", path, "manifest-2026-08.json", "run-test")
    seed_month(
        store,
        "2026-09",
        {"2026-09-15": _us_rows("2026-09-15T1817Z", 4.099)},
        captures={
            "2026-09-15T1817Z": {
                "status": _status("2026-09-15T1817Z"),
                "rows_by_capture_date": {"2026-09-15": 2},
            }
        },
        work=tmp_path / "seed-09",
    )
    issues = RecordingIssues()

    result = close_periods(store, cfg, now=datetime(2026, 10, 1, 3, 17, tzinfo=UTC), issues=issues)

    assert result.blocked_months == ["2026-08"]
    assert result.closed_months == ["2026-09"]
    assert store.get_release("data-2026-08").prerelease is True
    assert store.get_release("data-2026-09").prerelease is False


AUD_FX = [
    {
        "currency": "AUD",
        "units_per_usd": 1.399,
        "fx_rate_date": "2026-09-14",
        "fx_source": "frankfurter-v2",
        "fx_fetched_at_utc": "2026-08-31T18:17:40Z",
    }
]


def test_full_rebuild_reads_closed_month_files_and_closed_month_manifest_fx(tmp_path):
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    _seed_august(store, tmp_path, fx=AUD_FX)
    close_periods(
        store,
        cfg,
        now=datetime(2026, 9, 1, 3, 17, tzinfo=UTC),
        issues=RecordingIssues(),
    )
    seed_month(
        store,
        "2026-09",
        {"2026-09-15": _us_rows("2026-09-15T1817Z", 4.099)},
        captures={
            "2026-09-15T1817Z": {
                "status": _status("2026-09-15T1817Z"),
                "rows_by_capture_date": {"2026-09-15": 2},
                "fx": [],
            }
        },
        work=tmp_path / "seed-09",
    )

    rebuild_current(store, cfg, now=datetime(2026, 9, 15, 18, 30, tzinfo=UTC))

    caps = pl.read_parquet(
        store.download("current", "club-gas-all-captures.parquet", tmp_path / "all.parquet")
    )
    assert caps.height == 6
    assert set(caps["capture_id"].unique().to_list()) == {
        "2026-08-30T1817Z",
        "2026-08-31T1817Z",
        "2026-09-15T1817Z",
    }
    fx = pl.read_csv(store.download("current", "fx.csv", tmp_path / "fx.csv"))
    assert fx["currency"].to_list() == ["AUD", "AUD"]
    assert fx["fx_usd_per_unit"].to_list() == [0.7147962831, 0.7147962831]
    latest = pl.read_csv(store.download("current", "club-gas-latest.csv", tmp_path / "latest.csv"))
    assert latest.height == 2
    assert set(latest["capture_id"].to_list()) == {"2026-09-15T1817Z"}
    manifest = json.loads(
        store.download("current", "manifest.json", tmp_path / "m.json").read_text()
    )
    assert manifest["status"]["capture_id"] == "2026-09-15T1817Z"
    assert manifest["closed_months"] == ["2026-08"]
    assert manifest["newest_capture_by_feed"] == {"US-COSTCO": "2026-09-15T1817Z"}
    assert set(manifest["assets"]) == CURRENT_ASSET_NAMES


def test_full_rebuild_manifest_equals_the_incremental_one(tmp_path):
    """The rebuilt current/manifest.json is the same document publish writes: the
    same keys, and newest_capture_by_feed recomputed rather than carried over."""
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    store.ensure_release("current", "Current", "", False, "true")
    stale = tmp_path / "stale-manifest.json"
    stale.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": {
                    "schema_version": 1,
                    "capture_id": "2026-07-01T1817Z",
                    "feeds": {"JP-COSTCO": {"status": "ok"}},
                },
                "closed_months": [],
                "closed_years": [],
                "newest_capture_by_feed": {
                    "US-COSTCO": "2026-07-01T1817Z",
                    "JP-COSTCO": "2026-07-01T1817Z",
                },
                "assets": {},
            },
            indent=1,
            sort_keys=True,
        ),
        "utf-8",
    )
    store.upload_new("current", stale, "manifest.json")
    _seed_august(store, tmp_path)

    rebuild_current(store, cfg, now=datetime(2026, 9, 1, 3, 17, tzinfo=UTC))

    manifest = json.loads(
        store.download("current", "manifest.json", tmp_path / "out.json").read_text()
    )
    assert set(manifest) >= CURRENT_MANIFEST_KEYS
    assert manifest["newest_capture_by_feed"] == {"US-COSTCO": "2026-08-31T1817Z"}
    assert manifest["status"]["capture_id"] == "2026-08-31T1817Z"
    assert manifest["closed_months"] == []
    assert manifest["closed_years"] == []
    assert set(manifest["assets"]) == CURRENT_ASSET_NAMES


def test_full_rebuild_refreshes_the_site_files_it_lists(tmp_path):
    """A rebuild that only relists stale site files would be worse than useless.

    The manifest is what the dashboard verifies its download against, so a
    site file carried over from before the rebuild passes the checksum and
    renders the OLD dataset -- one capture's map over another's history, the
    exact thing publishing them inside the transaction exists to prevent.
    """
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    store.ensure_release("current", "Current", "", False, "true")
    stale = tmp_path / "stale-site.json"
    stale.write_text('{"stale": true}', "utf-8")
    store.upload_new("current", stale, "site-latest.json")
    _seed_august(store, tmp_path)

    rebuild_current(store, cfg, now=datetime(2026, 9, 1, 3, 17, tzinfo=UTC))

    manifest = json.loads(
        store.download("current", "manifest.json", tmp_path / "out.json").read_text()
    )
    got = store.download("current", "site-latest.json", tmp_path / "site-latest.json")
    assert json.loads(got.read_text()) != {"stale": True}
    # download() verifies bytes against the listing; the manifest is the
    # separate claim the dashboard checks, so assert that one explicitly.
    assert manifest["assets"]["site-latest.json"]["size"] == got.stat().st_size
    assert manifest["assets"]["site-latest.json"]["sha256"] == sha256_file(got)


def test_full_rebuild_keeps_alt_id_and_status_and_applies_station_links(tmp_path):
    store = open_store(f"local:{tmp_path / 'releases'}")
    links = pl.DataFrame(
        {
            "old_station_key": ["US-COSTCO-1364"],
            "new_station_key": ["US-COSTCO-9999"],
            "effective_date": ["2026-09-01"],
            "note": ["relocated"],
        }
    )
    cfg = stub_config(tmp_path, links=links)
    store.ensure_release("current", "Current", "", False, "true")
    existing = stations_frame(
        [
            {
                "station_key": "US-COSTCO-1364",
                "country": "US",
                "brand": "COSTCO",
                "source_station_id": "1364",
                "alt_id": "Mansfield",
                "name": "Mansfield",
                "name_local": None,
                "address": None,
                "city": "Mansfield",
                "region": "TX",
                "postcode": None,
                "lat": 32.7,
                "lon": -97.1,
                "timezone": "America/Chicago",
                "grades_seen": "regular",
                "first_seen_utc": datetime(2026, 7, 1, 0, 17, tzinfo=UTC),
                "last_seen_utc": datetime(2026, 7, 1, 0, 17, tzinfo=UTC),
                "status": "missing",
                "superseded_by": None,
            }
        ]
    )
    path = tmp_path / "stations.csv"
    existing.write_csv(path)
    store.upload_new("current", path, "stations.csv")
    _seed_august(store, tmp_path)

    rebuild_current(store, cfg, now=datetime(2026, 9, 1, 3, 17, tzinfo=UTC))

    out = pl.read_csv(store.download("current", "stations.csv", tmp_path / "out-stations.csv"))
    row = out.filter(pl.col("station_key") == "US-COSTCO-1364").row(0, named=True)
    assert row["alt_id"] == "Mansfield"
    assert row["status"] == "missing"
    assert row["grades_seen"] == "premium|regular"
    assert row["first_seen_utc"].startswith("2026-08-30")
    assert row["last_seen_utc"].startswith("2026-08-31")
    assert row["superseded_by"] == "US-COSTCO-9999"


def test_full_rebuild_recomputes_merged_captures_to_match_the_rebuilt_captures_file(
    tmp_path,
):
    """Carried from Task 14: `merged_captures` in current/manifest.json is the commit
    marker `read_or_rebuild_manifest` trusts to decide a bundle is already merged into
    `current`. A full rebuild must recompute it to exactly the capture ids whose rows
    it actually wrote into the rebuilt all-captures file -- never carry a stale or
    over-claiming value over, since that would silently resurrect the exact
    data-loss hole `merged_captures` was added to close.
    """
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    store.ensure_release("current", "Current", "", False, "true")
    stale = tmp_path / "stale-manifest.json"
    stale.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "status": None,
                "closed_months": [],
                "closed_years": [],
                "newest_capture_by_feed": {},
                "assets": {},
                # Neither id is a real capture in this test's data: one predates
                # everything seeded below, the other never existed at all. Both
                # must be gone from the rebuilt manifest.
                "merged_captures": ["2020-01-01T0000Z", "2099-01-01T0000Z"],
            },
            indent=1,
            sort_keys=True,
        ),
        "utf-8",
    )
    store.upload_new("current", stale, "manifest.json")
    _seed_august(store, tmp_path)
    seed_month(
        store,
        "2026-09",
        {"2026-09-15": _us_rows("2026-09-15T1817Z", 4.099)},
        captures={
            "2026-09-15T1817Z": {
                "status": _status("2026-09-15T1817Z"),
                "rows_by_capture_date": {"2026-09-15": 2},
            }
        },
        work=tmp_path / "seed-09",
    )

    rebuild_current(store, cfg, now=datetime(2026, 9, 15, 19, 0, tzinfo=UTC))

    caps = pl.read_parquet(
        store.download("current", "club-gas-all-captures.parquet", tmp_path / "caps.parquet")
    )
    manifest = json.loads(
        store.download("current", "manifest.json", tmp_path / "m.json").read_text()
    )
    assert sorted(manifest["merged_captures"]) == sorted(caps["capture_id"].unique().to_list())
    assert manifest["merged_captures"] == [
        "2026-08-30T1817Z",
        "2026-08-31T1817Z",
        "2026-09-15T1817Z",
    ]


def test_rebuild_current_recovers_an_interrupted_replace_before_reading(tmp_path):
    """Finding 3 (spec review round 4): `rebuild_current` is a public entry point
    Task 16 calls directly, not only through close_periods, so it must run
    recovery itself -- store.py's own contract says recovery runs at the start
    of publish, close-periods AND rebuild. `DAILY_ASSET` never matches a
    `.next-`/`.old-` name, so without recovery here an interrupted day's file
    would be silently dropped from the rebuilt `current`."""
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    store.ensure_release("data-2026-09", "Data 2026-09", "", True, "false")
    frame = _us_rows("2026-09-15T1817Z", 4.099)
    path = tmp_path / "club-gas-2026-09-15.csv.gz"
    write_csv_gz(frame, path)
    name = "club-gas-2026-09-15.csv.gz"
    # Simulates a replace_atomic that uploaded and verified the new bytes but
    # crashed before promoting them to the live name: exactly what an
    # interrupted first-ever write of this day's file leaves behind.
    store.upload_new("data-2026-09", path, next_name(name, "run-test", 1), label=sha256_label(path))

    rebuild_current(store, cfg, now=datetime(2026, 9, 15, 19, 0, tzinfo=UTC))

    names = {a.name for a in store.list_assets("data-2026-09")}
    assert name in names
    caps = pl.read_parquet(
        store.download("current", "club-gas-all-captures.parquet", tmp_path / "caps.parquet")
    )
    assert caps.height == 2
    assert set(caps["capture_id"].to_list()) == {"2026-09-15T1817Z"}


def _close_both_months_of_2026(store, cfg, tmp_path):
    _seed_august(store, tmp_path)
    seed_month(
        store,
        "2026-09",
        {"2026-09-15": _us_rows("2026-09-15T1817Z", 4.099)},
        captures={
            "2026-09-15T1817Z": {
                "status": _status("2026-09-15T1817Z"),
                "rows_by_capture_date": {"2026-09-15": 2},
            }
        },
        work=tmp_path / "seed-09",
    )
    close_periods(
        store,
        cfg,
        now=datetime(2026, 10, 1, 3, 17, tzinfo=UTC),
        issues=RecordingIssues(),
    )


def test_year_close_creates_the_release_first_and_the_manifest_last(tmp_path):
    store = RecordingStore(open_store(f"local:{tmp_path / 'releases'}"))
    cfg = stub_config(tmp_path)
    _close_both_months_of_2026(store, cfg, tmp_path)
    store.log.clear()

    result = close_periods(
        store,
        cfg,
        now=datetime(2027, 1, 5, 3, 17, tzinfo=UTC),
        issues=RecordingIssues(),
    )

    assert result.closed_years == ["2026"]
    assert store.get_release("data-2026").prerelease is False
    names = {a.name for a in store.list_assets("data-2026")}
    assert {
        "club-gas-2026.parquet",
        "club-gas-2026.csv.gz",
        "club-gas-2026-captures.parquet",
        "manifest-2026.json",
    } <= names
    caps = pl.read_parquet(
        store.download("data-2026", "club-gas-2026-captures.parquet", tmp_path / "y.parquet")
    )
    assert caps.height == 6
    assert store.log.index("ensure_release:data-2026") < store.log.index(
        "replace_atomic:data-2026:club-gas-2026.parquet"
    )
    assert store.log.index("replace_atomic:data-2026:manifest-2026.json") == max(
        i for i, entry in enumerate(store.log) if entry.startswith("replace_atomic:data-2026:")
    )


def test_year_close_does_nothing_when_the_month_shas_already_match(tmp_path):
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    _close_both_months_of_2026(store, cfg, tmp_path)
    now = datetime(2027, 1, 5, 3, 17, tzinfo=UTC)
    close_periods(store, cfg, now=now, issues=RecordingIssues())

    again = close_periods(store, cfg, now=now, issues=RecordingIssues())

    assert again.closed_years == []


def test_year_close_is_skipped_while_a_month_is_blocked(tmp_path):
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    _seed_august(store, tmp_path)
    close_periods(
        store,
        cfg,
        now=datetime(2026, 9, 1, 3, 17, tzinfo=UTC),
        issues=RecordingIssues(),
    )
    seed_month(
        store,
        "2026-09",
        {"2026-09-15": _us_rows("2026-09-15T1817Z", 4.099)},
        captures={
            "2026-09-15T1817Z": {
                "status": _status("2026-09-15T1817Z"),
                "rows_by_capture_date": {"2026-09-15": 2},
            }
        },
        orphan_bundles=["capture-2026-09-16T1817Z.tar.gz"],
        work=tmp_path / "seed-09",
    )

    result = close_periods(
        store,
        cfg,
        now=datetime(2027, 1, 5, 3, 17, tzinfo=UTC),
        issues=RecordingIssues(),
    )

    assert result.blocked_months == ["2026-09"]
    assert result.closed_years == []
    assert store.get_release("data-2026") is None


def test_rebuild_current_flag_runs_without_a_close_and_restores_latest(tmp_path):
    store = open_store(f"local:{tmp_path / 'releases'}")
    cfg = stub_config(tmp_path)
    store.ensure_release("current", "Current", "", False, "false")
    seed_month(
        store,
        "2026-09",
        {"2026-09-15": _us_rows("2026-09-15T1817Z", 4.099)},
        captures={
            "2026-09-15T1817Z": {
                "status": _status("2026-09-15T1817Z"),
                "rows_by_capture_date": {"2026-09-15": 2},
            }
        },
        work=tmp_path / "seed-09",
    )

    result = close_periods(
        store,
        cfg,
        now=datetime(2026, 9, 15, 19, 0, tzinfo=UTC),
        rebuild_current=True,
        issues=RecordingIssues(),
    )

    assert result.closed_months == []
    assert result.rebuilt_current is True
    assert store.get_release("current").is_latest is True
    names = {a.name for a in store.list_assets("current")}
    assert "club-gas-all.parquet" in names and "manifest.json" in names
