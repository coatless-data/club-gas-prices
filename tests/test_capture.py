"""Tests for costco_gas.capture (spec 5.3)."""

from __future__ import annotations

import json
import shutil
import tarfile
from datetime import UTC, datetime, timedelta, timezone
from pathlib import Path

import httpx
import polars as pl
import pytest

from costco_gas import capture as capture_module
from costco_gas.capture import capture_id_for, github_output, run_capture
from costco_gas.config import load_config
from costco_gas.http import Client
from costco_gas.store import LocalReleaseStore

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Fixtures committed by earlier tasks; this module only reads them.
TW_BODY = (FIXTURES / "tw_stores.json").read_bytes()
FX_BODY = (FIXTURES / "fx" / "frankfurter_v2.json").read_bytes()
ECOM_BODY = (FIXTURES / "us_ecom_warehouses.json").read_bytes()

NOW = datetime(2026, 9, 15, 18, 17, 40, tzinfo=UTC)
CAPTURE_ID = "2026-09-15T1817Z"


def test_capture_id_is_utc_truncated_to_the_minute():
    now = datetime(2026, 9, 15, 18, 17, 40, 512000, tzinfo=UTC)
    assert capture_id_for(now) == "2026-09-15T1817Z"


def test_capture_id_converts_a_non_utc_clock():
    now = datetime(2026, 9, 15, 20, 17, 40, tzinfo=timezone(timedelta(hours=2)))
    assert capture_id_for(now) == "2026-09-15T1817Z"


def test_github_output_appends_when_set(tmp_path: Path, monkeypatch):
    target = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    github_output("capture_id", "2026-09-15T1817Z")
    github_output("all_failed", "false")
    assert target.read_text(encoding="utf-8") == ("capture_id=2026-09-15T1817Z\nall_failed=false\n")


def test_github_output_is_a_noop_when_unset(monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    github_output("capture_id", "2026-09-15T1817Z")  # must not raise


def make_transport(*, tw_status: int = 200, ca_status: int = 200) -> httpx.MockTransport:
    """Serve the captured 2026-09-15 bodies; every other host 404s."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "api.frankfurter.dev":
            return httpx.Response(
                200, content=FX_BODY, headers={"Content-Type": "application/json"}
            )
        if host == "www.costco.com.tw":
            if tw_status != 200:
                return httpx.Response(tw_status, content=b"Access Denied")
            return httpx.Response(
                200, content=TW_BODY, headers={"Content-Type": "application/json"}
            )
        if host == "ecom-api.costco.com":
            return httpx.Response(
                200, content=ECOM_BODY, headers={"Content-Type": "application/json"}
            )
        if host == "www.costco.ca":
            if ca_status != 200:
                return httpx.Response(ca_status, content=b"Access Denied")
            return httpx.Response(200, content=b"[false]", headers={"Content-Type": "text/html"})
        return httpx.Response(404, content=b"{}")

    return httpx.MockTransport(handler)


@pytest.fixture()
def workspace(tmp_path: Path, monkeypatch):
    """A checkout-shaped working directory: config/, status/ and an empty store."""
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    (tmp_path / "status").mkdir()
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GITHUB_OUTPUT", str(tmp_path / "gh_output"))
    monkeypatch.delenv("GITHUB_SHA", raising=False)
    monkeypatch.delenv("GITHUB_RUN_ID", raising=False)
    return tmp_path


def test_minimal_run_writes_status_and_exit_outputs(workspace: Path):
    cfg = load_config(workspace)
    store = LocalReleaseStore(workspace / "releases")
    client = Client(cfg.http, transport=make_transport())

    result = run_capture(
        cfg,
        store,
        workspace / "out",
        countries=[],
        force_fallback=set(),
        now=NOW,
        client=client,
    )

    status_path = workspace / "out" / "capture" / "status.json"
    status = json.loads(status_path.read_text(encoding="utf-8"))

    assert result.capture_id == CAPTURE_ID
    assert result.out == workspace / "out" / "capture"
    assert status["capture_id"] == CAPTURE_ID
    assert set(status["countries"]) == {"US", "CA", "MX", "GB", "AU", "JP", "TW"}
    assert {block["status"] for block in status["countries"].values()} == {"skipped"}
    # No country succeeded, so the workflow gate must fail the run.
    assert result.all_failed is True
    # The current release is empty, so both previous frames are missing.
    # status["warnings"] holds build_status's dict-shaped {code, detail}
    # entries (checks.py, Task 12), not plain strings.
    assert "previous_state_unavailable" in {w["code"] for w in status["warnings"]}
    assert status["fx"]["status"] == "ok"

    output = (workspace / "gh_output").read_text(encoding="utf-8").splitlines()
    assert output[0] == f"capture_id={CAPTURE_ID}"
    assert "all_failed=true" in output


def test_previous_state_is_read_from_current_with_read_resolved(workspace: Path):
    cfg = load_config(workspace)
    store = LocalReleaseStore(workspace / "releases")
    store.ensure_release("current", "current", "", False, "true")
    stations = workspace / "stations.csv"
    pl.DataFrame({"station_key": ["TW-Chungli"], "country": ["TW"]}).write_csv(stations)
    fx = workspace / "fx.csv"
    pl.DataFrame({"capture_id": ["2026-09-14T1817Z"], "currency": ["TWD"]}).write_csv(fx)
    store.upload_new("current", stations, "stations.csv")
    store.upload_new("current", fx, "fx.csv")

    client = Client(cfg.http, transport=make_transport())
    result = run_capture(
        cfg,
        store,
        workspace / "out",
        countries=[],
        force_fallback=set(),
        now=NOW,
        client=client,
    )

    assert "previous_state_unavailable" not in {w["code"] for w in result.status["warnings"]}
    used = workspace / "out" / "state" / "stations.csv"
    assert pl.read_csv(used)["station_key"].to_list() == ["TW-Chungli"]


def test_taiwan_capture_writes_per_country_and_capture_outputs(workspace: Path):
    cfg = load_config(workspace)
    store = LocalReleaseStore(workspace / "releases")
    client = Client(cfg.http, transport=make_transport())

    result = run_capture(
        cfg,
        store,
        workspace / "out",
        countries=["TW"],
        force_fallback=set(),
        now=NOW,
        client=client,
    )

    status = result.status
    assert status["countries"]["TW"]["status"] == "ok"
    assert status["countries"]["TW"]["stations"] == 3
    assert status["countries"]["TW"]["rows"] == 7
    assert status["countries"]["US"]["status"] == "skipped"
    assert result.all_failed is False

    country_dir = workspace / "out" / "countries" / "TW"
    assert (country_dir / "rows.csv.gz").exists()
    assert (country_dir / "stations.csv").exists()
    assert (country_dir / "drops.csv").exists()
    assert json.loads((country_dir / "country.json").read_text())["status"] == "ok"
    bodies = sorted(p.name for p in (country_dir / "responses").glob("*.body"))
    assert bodies, "the raw TW response must be stored for rebuild"

    rows = pl.read_csv(workspace / "out" / "capture" / "rows.csv.gz")
    assert rows.height == 7
    assert sorted(rows["station_key"].unique().to_list()) == [
        "TW-Chungli",
        "TW-North_Taichung",
        "TW-Xinzhuang",
    ]
    assert rows["capture_id"].unique().to_list() == [CAPTURE_ID]

    fx_rows = json.loads((workspace / "out" / "capture" / "fx.json").read_text())
    # fx.json is an array, never an object (spec 4.5).
    assert isinstance(fx_rows, list)
    twd = [r for r in fx_rows if r["currency"] == "TWD"]
    assert twd == [
        {
            "currency": "TWD",
            "units_per_usd": 31.709,
            "fx_rate_date": "2026-09-14",
            "fx_source": "frankfurter-v2",
            "fx_fetched_at_utc": twd[0]["fx_fetched_at_utc"],
        }
    ]


def test_bundle_holds_the_spec_8_2_contents(workspace: Path):
    cfg = load_config(workspace)
    store = LocalReleaseStore(workspace / "releases")
    client = Client(cfg.http, transport=make_transport())

    run_capture(
        cfg,
        store,
        workspace / "out",
        countries=["TW"],
        force_fallback=set(),
        now=NOW,
        client=client,
    )

    with tarfile.open(workspace / "out" / "capture" / "bundle.tar.gz", "r:gz") as tar:
        names = set(tar.getnames())
        capture_json = json.loads(tar.extractfile("capture.json").read())

    for required in (
        "capture.json",
        "fx.json",
        "status.json",
        "rows.csv.gz",
        "stations.csv",
        "inputs/stations_used.csv",
        "inputs/fx_used.csv",
        "inputs/status_previous.json",
        "inputs/us_id_set.csv",
        "config/countries.toml",
        "config/grades.csv",
        "config/http.toml",
    ):
        assert required in names, required
    assert any(n.startswith("responses/TW/") for n in names)
    assert capture_json["capture_id"] == CAPTURE_ID
    assert capture_json["capture_date"] == "2026-09-15"
    assert set(capture_json["config_sha256"]) >= {"countries.toml", "grades.csv"}


def test_a_blocked_country_fails_alone_and_sets_all_failed(workspace: Path):
    cfg = load_config(workspace)
    store = LocalReleaseStore(workspace / "releases")
    client = Client(cfg.http, transport=make_transport(tw_status=403))

    result = run_capture(
        cfg,
        store,
        workspace / "out",
        countries=["TW"],
        force_fallback=set(),
        now=NOW,
        client=client,
    )

    assert result.status["countries"]["TW"]["status"] == "failed"
    assert result.all_failed is True
    # status.json is still written, so the command still exits 0 (spec 5.3 step 6).
    assert (workspace / "out" / "capture" / "status.json").exists()
    assert "all_failed=true" in (workspace / "gh_output").read_text()


def test_ecom_api_is_fetched_once_for_ca_and_recorded(workspace: Path):
    cfg = load_config(workspace)
    store = LocalReleaseStore(workspace / "releases")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "api.frankfurter.dev":
            return httpx.Response(
                200, content=FX_BODY, headers={"Content-Type": "application/json"}
            )
        if request.url.host == "ecom-api.costco.com":
            assert request.headers["client-identifier"]
            return httpx.Response(
                200, content=ECOM_BODY, headers={"Content-Type": "application/json"}
            )
        return httpx.Response(403, content=b"Access Denied")

    client = Client(cfg.http, transport=httpx.MockTransport(handler))
    result = run_capture(
        cfg,
        store,
        workspace / "out",
        countries=["CA"],
        force_fallback=set(),
        now=NOW,
        client=client,
    )

    assert result.status["ecom_api"] == {"attempted": True, "http_status": 200}
    assert seen.count("ecom-api.costco.com") == 1
    # The shared response is stored once, not once per country (spec 8.2).
    assert (workspace / "out" / "shared" / "ecom-api.body").exists()
    with tarfile.open(workspace / "out" / "capture" / "bundle.tar.gz", "r:gz") as tar:
        assert "responses/shared/ecom-api.body" in tar.getnames()


def test_ecom_api_is_not_fetched_when_neither_us_nor_ca_is_selected(workspace: Path):
    cfg = load_config(workspace)
    store = LocalReleaseStore(workspace / "releases")
    seen: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(request.url.host)
        if request.url.host == "api.frankfurter.dev":
            return httpx.Response(
                200, content=FX_BODY, headers={"Content-Type": "application/json"}
            )
        return httpx.Response(200, content=TW_BODY, headers={"Content-Type": "application/json"})

    client = Client(cfg.http, transport=httpx.MockTransport(handler))
    result = run_capture(
        cfg,
        store,
        workspace / "out",
        countries=["TW"],
        force_fallback=set(),
        now=NOW,
        client=client,
    )

    assert "ecom-api.costco.com" not in seen
    assert result.status["ecom_api"] == {"attempted": False, "http_status": None}


# --- review findings (Task 13 re-review) -----------------------------------
#
# Finding 1 (Critical): `_write_bundle` used to recompute the US polled-id
# frame with an unguarded second call to `polled_id_frame(ctx)`, after
# status.json was already durable, so a metadata glitch that `fetch_us`'s own
# (guarded) call to the same function would otherwise degrade gracefully
# instead crashed the whole capture uncaught. The fix computes the frame once,
# inside `run_country`'s guarded US path, and carries it forward; `_us_id_set`
# only ever reads it back. The two tests below cover, respectively, that the
# frame is carried rather than recomputed, and that a bundle-stage failure
# (of any kind) is recorded as a warning instead of propagating.


def test_us_id_set_is_read_from_collected_and_never_recomputed(monkeypatch):
    def must_not_be_called(ctx):
        raise AssertionError("_us_id_set must never call polled_id_frame itself")

    monkeypatch.setattr(capture_module.us_source, "polled_id_frame", must_not_be_called)

    # US absent (skipped/never run): empty frame with the right columns.
    empty = capture_module._us_id_set({})
    assert empty.height == 0
    assert empty.columns == list(capture_module.US_ID_SET_SCHEMA)

    # US present but its guarded path recorded no frame (e.g. it failed):
    # still an empty frame with the right columns, not a crash.
    us_failed = capture_module._us_id_set({"US": {"us_id_set": None}})
    assert us_failed.height == 0
    assert us_failed.columns == list(capture_module.US_ID_SET_SCHEMA)

    # US present with a frame already computed by run_country: read back
    # exactly, with no second call to polled_id_frame (the monkeypatch above
    # would have raised had one happened).
    frame = pl.DataFrame(
        {
            "source_station_id": ["109"],
            "id_origin": ["ecom"],
            "ecom_state": ["gas"],
        }
    )
    carried = capture_module._us_id_set({"US": {"us_id_set": frame}})
    assert carried.equals(frame)


def test_a_bundle_failure_is_recorded_as_a_warning_and_never_fatal(workspace: Path, monkeypatch):
    def boom(out, capture_out, ctx, run, collected):
        raise RuntimeError("disk exploded")

    monkeypatch.setattr(capture_module, "_write_bundle", boom)

    cfg = load_config(workspace)
    store = LocalReleaseStore(workspace / "releases")
    client = Client(cfg.http, transport=make_transport())

    result = run_capture(
        cfg,
        store,
        workspace / "out",
        countries=["TW"],
        force_fallback=set(),
        now=NOW,
        client=client,
    )

    # The country itself is unaffected; only the bundle step blew up.
    assert result.status["countries"]["TW"]["status"] == "ok"
    assert "bundle_failed" in {w["code"] for w in result.status["warnings"]}
    detail = next(w["detail"] for w in result.status["warnings"] if w["code"] == "bundle_failed")
    assert "disk exploded" in detail

    # status.json on disk was rewritten with the same warning after the
    # failure, and no bundle.tar.gz was produced -- but nothing crashed.
    on_disk = json.loads((workspace / "out" / "capture" / "status.json").read_text())
    assert "bundle_failed" in {w["code"] for w in on_disk["warnings"]}
    assert not (workspace / "out" / "capture" / "bundle.tar.gz").exists()


# Finding 2 (Important): every prior `run_capture` call used at most one
# country, so `ThreadPoolExecutor(max_workers=len(countries))` never actually
# ran more than one worker and the shared-`Client` orchestration was asserted,
# not proven. This test runs three OCC countries together and checks
# per-country isolation under real concurrency: each country's response
# directory holds exactly its own fixture body (a race that cross-wired two
# threads' writes would corrupt this), and the shared client saw exactly one
# request per host.


def test_three_countries_run_concurrently_without_cross_talk(workspace: Path):
    jp_body = (FIXTURES / "jp_stores.json").read_bytes()
    mx_body = (FIXTURES / "mx_stores.json").read_bytes()
    seen_hosts: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen_hosts.append(request.url.host)
        if request.url.host == "api.frankfurter.dev":
            return httpx.Response(
                200, content=FX_BODY, headers={"Content-Type": "application/json"}
            )
        if request.url.host == "www.costco.com.tw":
            return httpx.Response(
                200, content=TW_BODY, headers={"Content-Type": "application/json"}
            )
        if request.url.host == "www.costco.co.jp":
            return httpx.Response(
                200, content=jp_body, headers={"Content-Type": "application/json"}
            )
        if request.url.host == "www.costco.com.mx":
            return httpx.Response(
                200, content=mx_body, headers={"Content-Type": "application/json"}
            )
        return httpx.Response(404, content=b"{}")

    cfg = load_config(workspace)
    store = LocalReleaseStore(workspace / "releases")
    client = Client(cfg.http, transport=httpx.MockTransport(handler))

    result = run_capture(
        cfg,
        store,
        workspace / "out",
        countries=["TW", "JP", "MX"],
        force_fallback=set(),
        now=NOW,
        client=client,
    )

    # Real production floors (config/countries.toml) are far above what the
    # trimmed JP/MX fixtures provide, so those two are legitimately
    # "degraded", not "ok" -- only TW's fixture matches its floor.
    expected = {
        "TW": (3, 7, TW_BODY),
        "JP": (5, 20, jp_body),
        "MX": (4, 8, mx_body),
    }
    for cc, (n_stations, n_rows, body) in expected.items():
        block = result.status["countries"][cc]
        assert block["status"] in ("ok", "degraded"), (cc, block["status"])
        assert block["stations"] == n_stations
        assert block["rows"] == n_rows

        country_dir = workspace / "out" / "countries" / cc
        response_bodies = list((country_dir / "responses").glob("*.body"))
        assert len(response_bodies) == 1, (cc, response_bodies)
        # Each directory holds exactly its own country's raw response: a race
        # between the three worker threads would show up here as a mismatch.
        assert response_bodies[0].read_bytes() == body

    rows = pl.read_csv(workspace / "out" / "capture" / "rows.csv.gz")
    assert set(rows["country"].unique().to_list()) == {"TW", "JP", "MX"}
    assert rows.height == 7 + 20 + 8

    # The shared Client's request log is global across threads; a race in the
    # pool would show up as a missing or duplicated host hit here.
    assert seen_hosts.count("www.costco.com.tw") == 1
    assert seen_hosts.count("www.costco.co.jp") == 1
    assert seen_hosts.count("www.costco.com.mx") == 1
    assert result.all_failed is False


# Finding 3 (Important): the exception-isolation branch in `run_country` (the
# `except Exception` that writes traceback.txt) was never exercised by a test
# -- the existing TW-403 test returns a graceful FetchResult, it never raises.
# This test makes one country's `fetch` genuinely raise and checks that only
# that country is marked failed, with its traceback recorded, while the other
# country's outputs are unaffected and the capture still returns normally
# (the CLI would still exit 0).


def test_a_raising_source_fails_only_that_country_and_records_its_traceback(
    workspace: Path, monkeypatch
):
    def boom(client, ctx):
        raise RuntimeError("boom: JP fetch exploded")

    monkeypatch.setattr(capture_module.SOURCES["JP"], "fetch", boom)

    cfg = load_config(workspace)
    store = LocalReleaseStore(workspace / "releases")
    client = Client(cfg.http, transport=make_transport())

    result = run_capture(
        cfg,
        store,
        workspace / "out",
        countries=["TW", "JP"],
        force_fallback=set(),
        now=NOW,
        client=client,
    )

    jp_block = result.status["countries"]["JP"]
    assert jp_block["status"] == "failed"
    assert any(e["code"] == "exception" for e in jp_block["errors"])
    assert "boom: JP fetch exploded" in jp_block["errors"][-1]["detail"]

    jp_dir = workspace / "out" / "countries" / "JP"
    traceback_text = (jp_dir / "traceback.txt").read_text(encoding="utf-8")
    assert "RuntimeError" in traceback_text
    assert "boom: JP fetch exploded" in traceback_text
    # The isolation path still writes JP's (empty) per-country outputs.
    assert (jp_dir / "rows.csv.gz").exists()
    assert (jp_dir / "stations.csv").exists()
    assert (jp_dir / "drops.csv").exists()

    # TW is unaffected by JP's crash.
    assert result.status["countries"]["TW"]["status"] == "ok"
    tw_dir = workspace / "out" / "countries" / "TW"
    assert (tw_dir / "rows.csv.gz").exists()
    rows = pl.read_csv(workspace / "out" / "capture" / "rows.csv.gz")
    assert set(rows["country"].unique().to_list()) == {"TW"}

    # The capture still produced a durable status.json and returned normally
    # (no exception escaped run_capture; the CLI would exit 0).
    assert (workspace / "out" / "capture" / "status.json").exists()
    assert result.all_failed is False


# ------------------------------------------------- previous_state_unavailable (6.5)

CA_LOOKUP_BODY = (FIXTURES / "ca_lookup.body").read_bytes()
CA_ECOM_BODY = (FIXTURES / "ca_ecom_warehouses.json").read_bytes()


def ca_transport() -> httpx.MockTransport:
    """Serve the CA lookup and its metadata, so CA reaches `ok` on its primary path."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "api.frankfurter.dev":
            return httpx.Response(
                200, content=FX_BODY, headers={"Content-Type": "application/json"}
            )
        if host == "ecom-api.costco.com":
            return httpx.Response(
                200, content=CA_ECOM_BODY, headers={"Content-Type": "application/json"}
            )
        if host == "www.costco.ca":
            return httpx.Response(
                200, content=CA_LOOKUP_BODY, headers={"Content-Type": "text/html;charset=UTF-8"}
            )
        return httpx.Response(404, content=b"{}")

    return httpx.MockTransport(handler)


def lower_ca_floor(workspace: Path) -> None:
    """CA's real floor is 78 stations and `ca_lookup.body` holds a handful.

    Lowering it is what lets this test show the *only* reason CA is degraded,
    rather than having the floor degrade it whatever the warning does.
    """
    path = workspace / "config" / "countries.toml"
    text = path.read_text(encoding="utf-8")
    head, marker, tail = text.partition("[countries.CA]\n")
    assert marker, "no [countries.CA] section"
    path.write_text(head + marker + tail.replace("floor = 78", "floor = 1", 1), encoding="utf-8")


def seed_current_state(store: LocalReleaseStore, workspace: Path) -> None:
    """Enough of `current` for `_read_previous_state` to read both frames."""
    store.ensure_release("current", "current", "", False, "true")
    stations = workspace / "seed-stations.csv"
    pl.DataFrame({"station_key": ["CA-1324"], "country": ["CA"]}).write_csv(stations)
    fx = workspace / "seed-fx.csv"
    pl.DataFrame({"capture_id": ["2026-09-14T1817Z"], "currency": ["CAD"]}).write_csv(fx)
    store.upload_new("current", stations, "stations.csv")
    store.upload_new("current", fx, "fx.csv")


def run_ca_and_us(workspace: Path, store: LocalReleaseStore, out: str) -> dict:
    cfg = load_config(workspace)
    return run_capture(
        cfg,
        store,
        workspace / out,
        countries=["US", "CA"],
        force_fallback=set(),
        now=NOW,
        client=Client(cfg.http, transport=ca_transport()),
    ).status


def test_an_unreadable_current_degrades_ca_through_its_own_country_block(workspace: Path):
    """Spec 5.3 step 1 and 6.5: `previous_state_unavailable` makes US and CA degraded.

    `checks.evaluate_country` only ever inspects a country's own `warning_codes`,
    so recording this on the capture-level list alone left US and CA reporting
    `ok` while running without their `seen` ids and, for CA, without the cached
    metadata its fallback needs.
    """
    lower_ca_floor(workspace)
    missing = run_ca_and_us(workspace, LocalReleaseStore(workspace / "releases-empty"), "out-a")

    assert "previous_state_unavailable" in {w["code"] for w in missing["warnings"]}
    for code in ("US", "CA"):
        assert "previous_state_unavailable" in {
            w["code"] for w in missing["countries"][code]["warnings"]
        }
    assert missing["countries"]["CA"]["status"] == "degraded"
    assert missing["countries"]["CA"]["rows"] > 0

    store = LocalReleaseStore(workspace / "releases-seeded")
    seed_current_state(store, workspace)
    present = run_ca_and_us(workspace, store, "out-b")

    assert "previous_state_unavailable" not in {w["code"] for w in present["warnings"]}
    for code in ("US", "CA"):
        assert "previous_state_unavailable" not in {
            w["code"] for w in present["countries"][code]["warnings"]
        }
    # Same responses, same floor: the only thing that changed is the previous state.
    assert present["countries"]["CA"]["status"] == "ok"
    assert present["countries"]["CA"]["rows"] == missing["countries"]["CA"]["rows"]


def set_budgets(workspace: Path, **values: float) -> None:
    """Rewrite config/http.toml's [budgets] table, exactly as an operator would.

    [budgets] is the last table in the file, so replacing everything after its
    header replaces the whole table; names left out fall back to the built-in
    defaults.
    """
    path = workspace / "config" / "http.toml"
    head, marker, _rest = path.read_text(encoding="utf-8").partition("[budgets]\n")
    assert marker, "no [budgets] table"
    body = "".join(f'"{name}" = {value}\n' for name, value in values.items())
    path.write_text(head + marker + body, encoding="utf-8")


def test_capture_takes_its_deadlines_from_the_budgets_table(workspace: Path):
    """Editing `[budgets]` has to change what the capture actually does.

    The four numbers were hardcoded in capture.py and fx.py, so the parsed table
    was dead config: this test sets `fx` and `ecom-api` to zero and nothing else,
    which exhausts both before their first request.
    """
    set_budgets(workspace, fx=0.0, **{"ecom-api": 0.0})
    cfg = load_config(workspace)
    assert cfg.http.budgets["country"] == 480.0  # left out above, so still the default
    store = LocalReleaseStore(workspace / "releases")
    client = Client(cfg.http, transport=make_transport())

    result = run_capture(
        cfg,
        store,
        workspace / "out",
        countries=["US"],
        force_fallback=set(),
        now=NOW,
        client=client,
    )

    assert result.status["fx"] == {"status": "failed", "source": None, "rate_date": None}
    assert result.status["ecom_api"] == {"attempted": True, "http_status": None}
    # Neither request ever left: both budgets were spent before the first one.
    keys = [entry["key"] for entry in client.log]
    assert not [key for key in keys if key.startswith("fx/") or key == "shared/ecom-api"]
    # US still ran on the default country budget, which the table left alone.
    assert [key for key in keys if key.startswith("US/")]
