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
