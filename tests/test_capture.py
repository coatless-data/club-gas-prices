"""Tests for costco_gas.capture (spec 5.3)."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import pytest

from costco_gas.capture import capture_id_for, github_output


def test_capture_id_is_utc_truncated_to_the_minute():
    now = datetime(2026, 9, 15, 18, 17, 40, 512000, tzinfo=timezone.utc)
    assert capture_id_for(now) == "2026-09-15T1817Z"


def test_capture_id_converts_a_non_utc_clock():
    from datetime import timedelta

    now = datetime(2026, 9, 15, 20, 17, 40, tzinfo=timezone(timedelta(hours=2)))
    assert capture_id_for(now) == "2026-09-15T1817Z"


def test_github_output_appends_when_set(tmp_path: Path, monkeypatch):
    target = tmp_path / "gh_output"
    monkeypatch.setenv("GITHUB_OUTPUT", str(target))
    github_output("capture_id", "2026-09-15T1817Z")
    github_output("all_failed", "false")
    assert target.read_text(encoding="utf-8") == (
        "capture_id=2026-09-15T1817Z\nall_failed=false\n"
    )


def test_github_output_is_a_noop_when_unset(monkeypatch):
    monkeypatch.delenv("GITHUB_OUTPUT", raising=False)
    github_output("capture_id", "2026-09-15T1817Z")  # must not raise


import shutil

import httpx
import polars as pl

from costco_gas.capture import run_capture
from costco_gas.config import load_config
from costco_gas.http import Client
from costco_gas.store import LocalReleaseStore

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

# Fixtures committed by earlier tasks; this module only reads them.
TW_BODY = (FIXTURES / "tw_stores.json").read_bytes()
FX_BODY = (FIXTURES / "fx" / "frankfurter_v2.json").read_bytes()
ECOM_BODY = (FIXTURES / "us_ecom_warehouses.json").read_bytes()

NOW = datetime(2026, 9, 15, 18, 17, 40, tzinfo=timezone.utc)
CAPTURE_ID = "2026-09-15T1817Z"


def make_transport(*, tw_status: int = 200, ca_status: int = 200) -> httpx.MockTransport:
    """Serve the captured 2026-09-15 bodies; every other host 404s."""

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "api.frankfurter.dev":
            return httpx.Response(200, content=FX_BODY,
                                  headers={"Content-Type": "application/json"})
        if host == "www.costco.com.tw":
            if tw_status != 200:
                return httpx.Response(tw_status, content=b"Access Denied")
            return httpx.Response(200, content=TW_BODY,
                                  headers={"Content-Type": "application/json"})
        if host == "ecom-api.costco.com":
            return httpx.Response(200, content=ECOM_BODY,
                                  headers={"Content-Type": "application/json"})
        if host == "www.costco.ca":
            if ca_status != 200:
                return httpx.Response(ca_status, content=b"Access Denied")
            return httpx.Response(200, content=b"[false]",
                                  headers={"Content-Type": "text/html"})
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

    assert "previous_state_unavailable" not in {
        w["code"] for w in result.status["warnings"]
    }
    used = workspace / "out" / "state" / "stations.csv"
    assert pl.read_csv(used)["station_key"].to_list() == ["TW-Chungli"]
