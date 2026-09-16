"""Tests for the costco-gas command line."""

from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import ClassVar

import pytest

from costco_gas import cli

REPO_ROOT = Path(__file__).resolve().parents[1]


class _Cfg:
    countries: ClassVar[dict[str, None]] = {
        "US": None,
        "CA": None,
        "MX": None,
        "GB": None,
        "AU": None,
        "JP": None,
        "TW": None,
    }


def test_split_csv_normalises_and_drops_blanks():
    assert cli.split_csv(" us , ca ,") == ["US", "CA"]
    assert cli.split_csv("") == []
    assert cli.split_csv(None) == []


def test_select_countries_expands_all_and_keeps_config_order():
    assert cli.select_countries("all", _Cfg()) == list(_Cfg.countries)
    assert cli.select_countries("", _Cfg()) == list(_Cfg.countries)
    assert cli.select_countries("tw,us", _Cfg()) == ["US", "TW"]


def test_select_countries_rejects_an_unknown_code():
    with pytest.raises(SystemExit) as excinfo:
        cli.select_countries("ZZ", _Cfg())
    assert "unknown countries: ZZ" in str(excinfo.value)


def test_open_configured_store_defaults_to_the_project_repo(monkeypatch):
    captured = {}

    def fake_open_store(spec):
        captured["spec"] = spec

    monkeypatch.setattr(cli, "open_store", fake_open_store)
    monkeypatch.delenv("COSTCO_GAS_STORE", raising=False)
    cli.open_configured_store()
    assert captured["spec"] == "github:coatless-dashboard/costco-gas-prices"
    monkeypatch.setenv("COSTCO_GAS_STORE", "local:./releases")
    cli.open_configured_store()
    assert captured["spec"] == "local:./releases"


def test_capture_subcommand_calls_run_capture_and_prints_json(tmp_path, monkeypatch, capsys):
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COSTCO_GAS_STORE", f"local:{tmp_path / 'releases'}")

    seen = {}

    def fake_run_capture(cfg, store, out, *, countries, force_fallback, now, client=None):
        seen["countries"] = countries
        seen["force_fallback"] = force_fallback
        seen["out"] = out
        return cli.run_capture_result_for_test(out)

    monkeypatch.setattr(cli, "run_capture", fake_run_capture)
    code = cli.main(["capture", "--out", "out", "--countries", "tw", "--force-fallback", "US"])

    assert code == 0
    assert seen["countries"] == ["TW"]
    assert seen["force_fallback"] == {"US"}
    assert seen["out"] == Path("out")
    printed = json.loads(capsys.readouterr().out)
    assert printed["capture_id"] == "2026-09-15T1817Z"
    assert printed["all_failed"] is False


def test_publish_subcommand_calls_publish(tmp_path, monkeypatch, capsys):
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("COSTCO_GAS_STORE", f"local:{tmp_path / 'releases'}")

    seen = {}

    class _Result:
        capture_id = "2026-09-15T1817Z"
        warnings: ClassVar[list[str]] = ["reconciled:2026-09-15T1217Z"]
        assets_written: ClassVar[list[str]] = ["current/manifest.json"]

    def fake_publish(store, capture_dir, cfg, *, now):
        seen["capture_dir"] = capture_dir
        return _Result()

    monkeypatch.setattr(cli, "publish", fake_publish)
    code = cli.main(["publish", "out/capture"])

    assert code == 0
    assert seen["capture_dir"] == Path("out/capture")
    printed = json.loads(capsys.readouterr().out)
    assert printed["capture_id"] == "2026-09-15T1817Z"
    assert printed["warnings"] == ["reconciled:2026-09-15T1217Z"]
