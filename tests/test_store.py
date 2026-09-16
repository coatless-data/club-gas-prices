"""Tests for costco_gas.store: release storage, atomic replace and recovery."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from costco_gas import store


def test_temporary_name_helpers_round_trip():
    assert store.next_name("stations.csv", "2026-09-15T1817Z", 1) == (
        "stations.csv.next-2026-09-15T1817Z-1"
    )
    assert store.old_name("stations.csv", "2026-09-15T1817Z") == (
        "stations.csv.old-2026-09-15T1817Z"
    )
    assert store.split_temp_name("stations.csv.next-2026-09-15T1817Z-1") == (
        "stations.csv",
        "next",
    )
    assert store.split_temp_name("costco-gas-all.parquet.old-run-2026-10-01T0300Z") == (
        "costco-gas-all.parquet",
        "old",
    )
    assert store.split_temp_name("costco-gas-2026-09-15.csv.gz") is None


def test_sha256_label_is_the_github_digest_format(tmp_path: Path):
    path = tmp_path / "rows.csv"
    path.write_bytes(b"capture_id,price\n2026-09-15T1817Z,3.999\n")
    expected = hashlib.sha256(path.read_bytes()).hexdigest()
    assert store.sha256_file(path) == expected
    assert store.sha256_label(path) == f"sha256:{expected}"


def test_ensure_release_creates_once_and_then_returns_it_untouched(tmp_path: Path):
    s = store.LocalReleaseStore(tmp_path / "releases")
    assert s.get_release("current") is None

    created = s.ensure_release("current", "Current data", "all-time files", False, "true")
    assert created.tag == "current"
    assert created.title == "Current data"
    assert created.body == "all-time files"
    assert created.prerelease is False
    assert created.make_latest == "true"
    assert created.is_latest is True
    assert created.immutable is False

    again = s.ensure_release("current", "other title", "other body", True, "false")
    assert again.title == "Current data"
    assert again.prerelease is False
    assert s.get_latest().tag == "current"


def test_update_release_and_list_releases(tmp_path: Path):
    s = store.LocalReleaseStore(tmp_path / "releases")
    s.ensure_release("current", "Current data", "b", False, "true")
    s.ensure_release("data-2026-09", "September 2026", "b", True, "false")

    assert [r.tag for r in s.list_releases()] == ["current", "data-2026-09"]

    closed = s.update_release("data-2026-09", prerelease=False, body="closed month")
    assert closed.prerelease is False
    assert closed.body == "closed month"
    assert s.get_release("data-2026-09").prerelease is False
    assert s.get_release("data-2026-09").title == "September 2026"

    with pytest.raises(store.StorageError, match="release not found"):
        s.update_release("data-2030-01", body="nope")


def test_is_latest_tracks_the_latest_marker(tmp_path: Path):
    # `make_latest` is the value a write asked for; `is_latest` is what the
    # store answers now for "which release is Latest?". Task 15 reads
    # `is_latest` to decide whether the Latest pointer has drifted off
    # `current`, so both stores have to populate it.
    s = store.LocalReleaseStore(tmp_path / "releases")
    s.ensure_release("current", "Current data", "b", False, "true")
    s.ensure_release("data-2026-09", "September 2026", "b", True, "false")

    assert s.get_release("current").is_latest is True
    assert s.get_release("data-2026-09").is_latest is False
    assert {r.tag: r.is_latest for r in s.list_releases()} == {
        "current": True,
        "data-2026-09": False,
    }

    moved = s.update_release("data-2026-09", make_latest="true")

    assert moved.is_latest is True
    assert s.get_release("current").is_latest is False
    assert s.get_latest().tag == "data-2026-09"


def test_ensure_release_refuses_an_immutable_release(tmp_path: Path):
    # Immutable releases must be disabled for the repository: a release that
    # cannot be edited can never receive a new daily file, so refuse loudly
    # instead of failing halfway through a publish.
    root = tmp_path / "releases"
    s = store.LocalReleaseStore(root)
    s.ensure_release("current", "Current data", "b", False, "true")

    sidecar = root / "current" / "_release.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["immutable"] = True
    sidecar.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(store.StorageError, match="immutable release: current"):
        s.ensure_release("current", "Current data", "b", False, "true")
