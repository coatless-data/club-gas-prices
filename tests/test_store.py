"""Tests for costco_gas.store: release storage, atomic replace and recovery."""

from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
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


def _seed(tmp_path: Path) -> tuple[store.LocalReleaseStore, Path]:
    root = tmp_path / "releases"
    s = store.LocalReleaseStore(root)
    s.ensure_release("data-2026-09", "September 2026", "b", True, "false")
    return s, root


def test_upload_new_records_digest_size_and_state(tmp_path: Path):
    s, root = _seed(tmp_path)
    src = tmp_path / "rows.csv"
    src.write_bytes(b"capture_id,price\n2026-09-15T1817Z,3.999\n")

    asset = s.upload_new("data-2026-09", src, "costco-gas-2026-09-15.csv.gz")
    assert asset.name == "costco-gas-2026-09-15.csv.gz"
    assert asset.state == "uploaded"
    assert asset.size == src.stat().st_size
    assert asset.digest == store.sha256_label(src)
    assert asset.label is None

    listed = s.list_assets("data-2026-09")
    assert [a.name for a in listed] == ["costco-gas-2026-09-15.csv.gz"]
    assert listed[0].id == asset.id
    assert (root / "data-2026-09" / "costco-gas-2026-09-15.csv.gz").read_bytes() == (
        src.read_bytes()
    )


def test_upload_new_keeps_an_explicit_label(tmp_path: Path):
    s, _ = _seed(tmp_path)
    src = tmp_path / "rows.csv"
    src.write_bytes(b"rows-v1\n")
    asset = s.upload_new("data-2026-09", src, "rows.csv.next-tok-1", label="sha256:abc")
    assert asset.label == "sha256:abc"


def test_upload_new_on_a_duplicate_name_reports_422(tmp_path: Path):
    s, _ = _seed(tmp_path)
    src = tmp_path / "rows.csv"
    src.write_bytes(b"rows-v1\n")
    s.upload_new("data-2026-09", src, "rows.csv")
    with pytest.raises(store.StorageError) as excinfo:
        s.upload_new("data-2026-09", src, "rows.csv")
    assert excinfo.value.status == 422


def test_list_assets_raises_when_the_release_is_missing(tmp_path: Path):
    # A missing release must never look like "no assets": a wrong store spec
    # or a stale path would otherwise present exactly like "never published"
    # to `download`, which reads its listing through `list_assets`.
    s, _ = _seed(tmp_path)
    with pytest.raises(store.StorageError, match="release not found: data-2030-01"):
        s.list_assets("data-2030-01")


def test_created_at_is_strictly_increasing(tmp_path: Path):
    s, _ = _seed(tmp_path)
    src = tmp_path / "rows.csv"
    src.write_bytes(b"rows-v1\n")
    first = s.upload_new("data-2026-09", src, "a.csv")
    second = s.upload_new("data-2026-09", src, "b.csv")
    assert second.created_at > first.created_at


def test_download_writes_the_bytes_and_creates_parents(tmp_path: Path):
    s, _ = _seed(tmp_path)
    src = tmp_path / "rows.csv"
    src.write_bytes(b"rows-v1\n")
    s.upload_new("data-2026-09", src, "rows.csv")

    dest = s.download("data-2026-09", "rows.csv", tmp_path / "out" / "nested" / "d.csv")
    assert dest.read_bytes() == b"rows-v1\n"


def test_download_raises_asset_not_found_when_the_asset_is_missing(tmp_path: Path):
    s, _ = _seed(tmp_path)
    with pytest.raises(store.AssetNotFound):
        s.download("data-2026-09", "rows.csv", tmp_path / "d.csv")


def test_download_propagates_release_not_found_instead_of_asset_not_found(
    tmp_path: Path,
):
    # A missing release must never present as "this file was never
    # published": callers treat AssetNotFound as "first publish" and would
    # otherwise start from empty and destroy history for what is really a
    # wrong store spec or a stale path.
    s, _ = _seed(tmp_path)
    with pytest.raises(store.StorageError, match="release not found: data-2030-01"):
        s.download("data-2030-01", "rows.csv", tmp_path / "d.csv")


def test_download_raises_storage_error_when_only_temporaries_exist(tmp_path: Path):
    # A missing <name> with a .next-* present is an interrupted replace, not a
    # file that was never published. Callers treat AssetNotFound as "first
    # publish" and would start from an empty frame, which would lose history.
    s, _ = _seed(tmp_path)
    src = tmp_path / "rows.csv"
    src.write_bytes(b"rows-v1\n")
    s.upload_new("data-2026-09", src, "rows.csv.next-tok-1", label=store.sha256_label(src))
    with pytest.raises(store.StorageError, match="recovery"):
        s.download("data-2026-09", "rows.csv", tmp_path / "d.csv")


def test_download_detects_a_corrupted_asset(tmp_path: Path):
    s, root = _seed(tmp_path)
    src = tmp_path / "rows.csv"
    src.write_bytes(b"rows-v1\n")
    s.upload_new("data-2026-09", src, "rows.csv")
    # Same length, different bytes: the size still matches, the digest does not.
    (root / "data-2026-09" / "rows.csv").write_bytes(b"rows-XX\n")

    with pytest.raises(store.StorageError, match="digest mismatch"):
        s.download("data-2026-09", "rows.csv", tmp_path / "d.csv")


def test_rename_can_clear_the_label_and_delete_removes_the_asset(tmp_path: Path):
    s, root = _seed(tmp_path)
    src = tmp_path / "rows.csv"
    src.write_bytes(b"rows-v1\n")
    uploaded = s.upload_new("data-2026-09", src, "rows.csv.next-tok-1", label="sha256:abc")

    renamed = s.rename("data-2026-09", uploaded.id, "rows.csv", label="")
    assert renamed.name == "rows.csv"
    assert renamed.label is None
    assert renamed.created_at == uploaded.created_at
    assert (root / "data-2026-09" / "rows.csv").exists()
    assert not (root / "data-2026-09" / "rows.csv.next-tok-1").exists()

    s.delete("data-2026-09", renamed.id)
    assert s.list_assets("data-2026-09") == []
    with pytest.raises(store.StorageError):
        s.delete("data-2026-09", renamed.id)


def test_rename_onto_an_existing_name_reports_422(tmp_path: Path):
    s, _ = _seed(tmp_path)
    src = tmp_path / "rows.csv"
    src.write_bytes(b"rows-v1\n")
    s.upload_new("data-2026-09", src, "rows.csv")
    other = s.upload_new("data-2026-09", src, "rows.csv.next-tok-1")
    with pytest.raises(store.StorageError) as excinfo:
        s.rename("data-2026-09", other.id, "rows.csv")
    assert excinfo.value.status == 422


def test_read_resolved_prefers_the_plain_name(tmp_path: Path):
    s, _ = _seed(tmp_path)
    live = tmp_path / "live.csv"
    live.write_bytes(b"live\n")
    stale = tmp_path / "stale.csv"
    stale.write_bytes(b"stale\n")
    s.upload_new("data-2026-09", live, "stations.csv")
    s.upload_new(
        "data-2026-09",
        stale,
        "stations.csv.next-tok-1",
        label=store.sha256_label(stale),
    )

    dest = s.read_resolved("data-2026-09", "stations.csv", tmp_path / "r.csv")
    assert dest.read_bytes() == b"live\n"


def test_read_resolved_takes_the_newest_verifying_next(tmp_path: Path):
    s, _ = _seed(tmp_path)
    good = tmp_path / "good.csv"
    good.write_bytes(b"good\n")
    torn = tmp_path / "torn.csv"
    torn.write_bytes(b"torn\n")

    s.upload_new("data-2026-09", good, "stations.csv.next-tok-1", label=store.sha256_label(good))
    # Uploaded later, but its label does not describe its bytes: a torn upload.
    s.upload_new("data-2026-09", torn, "stations.csv.next-tok-2", label="sha256:" + "0" * 64)

    dest = s.read_resolved("data-2026-09", "stations.csv", tmp_path / "r.csv")
    assert dest.read_bytes() == b"good\n"


def test_read_resolved_falls_back_to_the_newest_old(tmp_path: Path):
    s, _ = _seed(tmp_path)
    older = tmp_path / "older.csv"
    older.write_bytes(b"older\n")
    newer = tmp_path / "newer.csv"
    newer.write_bytes(b"newer\n")
    torn = tmp_path / "torn.csv"
    torn.write_bytes(b"torn\n")

    s.upload_new("data-2026-09", older, "stations.csv.old-tok-a")
    s.upload_new("data-2026-09", newer, "stations.csv.old-tok-b")
    s.upload_new("data-2026-09", torn, "stations.csv.next-tok-1", label="sha256:" + "0" * 64)

    dest = s.read_resolved("data-2026-09", "stations.csv", tmp_path / "r.csv")
    assert dest.read_bytes() == b"newer\n"


def test_read_resolved_raises_when_nothing_resolves(tmp_path: Path):
    s, _ = _seed(tmp_path)
    with pytest.raises(store.AssetNotFound):
        s.read_resolved("data-2026-09", "stations.csv", tmp_path / "r.csv")
    with pytest.raises(store.AssetNotFound):
        s.read_resolved("data-2030-01", "stations.csv", tmp_path / "r.csv")


def test_read_resolved_never_modifies_the_release(tmp_path: Path):
    s, _ = _seed(tmp_path)
    good = tmp_path / "good.csv"
    good.write_bytes(b"good\n")
    s.upload_new("data-2026-09", good, "stations.csv.next-tok-1", label=store.sha256_label(good))
    before = [(a.name, a.id, a.label) for a in s.list_assets("data-2026-09")]

    s.read_resolved("data-2026-09", "stations.csv", tmp_path / "r.csv")

    assert [(a.name, a.id, a.label) for a in s.list_assets("data-2026-09")] == before


class FakeClock:
    """A monotonic clock that only moves when something sleeps."""

    def __init__(self) -> None:
        self.t = 0.0
        self.slept: list[float] = []

    def now(self) -> float:
        return self.t

    def sleep(self, seconds: float) -> None:
        self.slept.append(seconds)
        self.t += seconds


def test_replace_atomic_creates_the_asset_when_it_is_absent(tmp_path: Path):
    s, _ = _seed(tmp_path)
    src = tmp_path / "stations.csv"
    src.write_bytes(b"stations-v1\n")

    asset = s.replace_atomic("data-2026-09", src, "stations.csv", "2026-09-15T1817Z")

    assert asset.name == "stations.csv"
    assert asset.label is None  # the sha256: label is cleared on promotion
    assert [a.name for a in s.list_assets("data-2026-09")] == ["stations.csv"]
    got = s.download("data-2026-09", "stations.csv", tmp_path / "got.csv")
    assert got.read_bytes() == b"stations-v1\n"


def test_replace_atomic_swaps_the_content_and_removes_the_old_copy(tmp_path: Path):
    s, _ = _seed(tmp_path)
    v1 = tmp_path / "v1.csv"
    v1.write_bytes(b"stations-v1\n")
    v2 = tmp_path / "v2.csv"
    v2.write_bytes(b"stations-v2\n")
    s.upload_new("data-2026-09", v1, "stations.csv")

    s.replace_atomic("data-2026-09", v2, "stations.csv", "2026-09-15T1817Z")

    assert [a.name for a in s.list_assets("data-2026-09")] == ["stations.csv"]
    got = s.download("data-2026-09", "stations.csv", tmp_path / "got.csv")
    assert got.read_bytes() == b"stations-v2\n"


def test_replace_atomic_never_reuses_a_temporary_name(tmp_path: Path):
    s, _ = _seed(tmp_path)
    v1 = tmp_path / "v1.csv"
    v1.write_bytes(b"stations-v1\n")
    v2 = tmp_path / "v2.csv"
    v2.write_bytes(b"stations-v2\n")
    leftover = tmp_path / "leftover.csv"
    leftover.write_bytes(b"leftover\n")

    s.upload_new("data-2026-09", v1, "stations.csv")
    s.upload_new(
        "data-2026-09",
        leftover,
        "stations.csv.next-tok-1",
        label=store.sha256_label(leftover),
    )

    s.replace_atomic("data-2026-09", v2, "stations.csv", "tok")

    names = sorted(a.name for a in s.list_assets("data-2026-09"))
    assert names == ["stations.csv", "stations.csv.next-tok-1"]
    got = s.download("data-2026-09", "stations.csv", tmp_path / "got.csv")
    assert got.read_bytes() == b"stations-v2\n"


def test_replace_atomic_deletes_an_upload_that_never_verifies(tmp_path: Path):
    class CorruptingStore(store.LocalReleaseStore):
        """Records a wrong digest for every upload, like a torn transfer."""

        def upload_new(self, tag, path, name, label=None):
            asset = super().upload_new(tag, path, name, label=label)
            data = self._read(tag)
            data["assets"][name]["digest"] = "sha256:" + "0" * 64
            self._write(tag, data)
            return asset

    root = tmp_path / "releases"
    seeder = store.LocalReleaseStore(root)
    seeder.ensure_release("data-2026-09", "September 2026", "b", True, "false")
    v1 = tmp_path / "v1.csv"
    v1.write_bytes(b"stations-v1\n")
    seeder.upload_new("data-2026-09", v1, "stations.csv")

    clock = FakeClock()
    s = CorruptingStore(root, sleep=clock.sleep, monotonic=clock.now)
    v2 = tmp_path / "v2.csv"
    v2.write_bytes(b"stations-v2\n")

    with pytest.raises(store.StorageError, match="did not verify"):
        s.replace_atomic("data-2026-09", v2, "stations.csv", "tok")

    assert len(clock.slept) == 60  # polled once a second for 60 seconds
    assert [a.name for a in s.list_assets("data-2026-09")] == ["stations.csv"]
    got = seeder.download("data-2026-09", "stations.csv", tmp_path / "got.csv")
    assert got.read_bytes() == b"stations-v1\n"


def _set_state(root: Path, tag: str, name: str, state: str) -> None:
    """Force an asset's state in the sidecar, as a half-finished upload looks."""
    sidecar = root / tag / "_release.json"
    data = json.loads(sidecar.read_text(encoding="utf-8"))
    data["assets"][name]["state"] = state
    sidecar.write_text(json.dumps(data), encoding="utf-8")


def _write(tmp_path: Path, name: str, payload: bytes) -> Path:
    path = tmp_path / name
    path.write_bytes(payload)
    return path


def test_recover_deletes_leftovers_when_the_live_name_is_present(tmp_path: Path):
    # Crash after step 1 of replace_atomic: <name> is still live and a verified
    # .next-* is sitting next to it.
    s, _ = _seed(tmp_path)
    v1 = _write(tmp_path, "v1.csv", b"stations-v1\n")
    v2 = _write(tmp_path, "v2.csv", b"stations-v2\n")
    s.upload_new("data-2026-09", v1, "stations.csv")
    s.upload_new("data-2026-09", v2, "stations.csv.next-tok-1", label=store.sha256_label(v2))

    actions = store.recover_temporaries(s, "data-2026-09")

    assert [a.name for a in s.list_assets("data-2026-09")] == ["stations.csv"]
    assert actions == ["deleted-leftover:stations.csv.next-tok-1"]
    got = s.download("data-2026-09", "stations.csv", tmp_path / "got.csv")
    assert got.read_bytes() == b"stations-v1\n"


def test_recover_promotes_the_next_when_the_live_name_is_gone(tmp_path: Path):
    # Crash between step 3 and step 4: <name> was renamed away, the new upload
    # verifies, so it becomes <name> and the old copy is dropped.
    s, _ = _seed(tmp_path)
    v1 = _write(tmp_path, "v1.csv", b"stations-v1\n")
    v2 = _write(tmp_path, "v2.csv", b"stations-v2\n")
    s.upload_new("data-2026-09", v1, "stations.csv.old-tok")
    s.upload_new("data-2026-09", v2, "stations.csv.next-tok-1", label=store.sha256_label(v2))

    actions = store.recover_temporaries(s, "data-2026-09")

    assert [a.name for a in s.list_assets("data-2026-09")] == ["stations.csv"]
    assert actions == [
        "promoted:stations.csv.next-tok-1",
        "deleted-leftover:stations.csv.old-tok",
    ]
    assert s.list_assets("data-2026-09")[0].label is None
    got = s.download("data-2026-09", "stations.csv", tmp_path / "got.csv")
    assert got.read_bytes() == b"stations-v2\n"


def test_recover_deletes_the_old_copy_when_the_promotion_already_happened(
    tmp_path: Path,
):
    # Crash between step 4 and step 5.
    s, _ = _seed(tmp_path)
    v1 = _write(tmp_path, "v1.csv", b"stations-v1\n")
    v2 = _write(tmp_path, "v2.csv", b"stations-v2\n")
    s.upload_new("data-2026-09", v2, "stations.csv")
    s.upload_new("data-2026-09", v1, "stations.csv.old-tok")

    actions = store.recover_temporaries(s, "data-2026-09")

    assert [a.name for a in s.list_assets("data-2026-09")] == ["stations.csv"]
    assert actions == ["deleted-leftover:stations.csv.old-tok"]
    got = s.download("data-2026-09", "stations.csv", tmp_path / "got.csv")
    assert got.read_bytes() == b"stations-v2\n"


def test_recover_restores_the_old_copy_when_no_next_verifies(tmp_path: Path):
    s, _ = _seed(tmp_path)
    v1 = _write(tmp_path, "v1.csv", b"stations-v1\n")
    torn = _write(tmp_path, "torn.csv", b"stations-XX\n")
    s.upload_new("data-2026-09", v1, "stations.csv.old-tok")
    s.upload_new("data-2026-09", torn, "stations.csv.next-tok-1", label="sha256:" + "0" * 64)

    actions = store.recover_temporaries(s, "data-2026-09")

    assert [a.name for a in s.list_assets("data-2026-09")] == ["stations.csv"]
    assert actions == [
        "restored:stations.csv.old-tok",
        "deleted-leftover:stations.csv.next-tok-1",
    ]
    got = s.download("data-2026-09", "stations.csv", tmp_path / "got.csv")
    assert got.read_bytes() == b"stations-v1\n"


def test_recover_deletes_temporaries_that_never_finished_uploading(tmp_path: Path):
    s, root = _seed(tmp_path)
    v1 = _write(tmp_path, "v1.csv", b"stations-v1\n")
    half = _write(tmp_path, "half.csv", b"stations-v2\n")
    s.upload_new("data-2026-09", v1, "stations.csv.old-tok")
    s.upload_new("data-2026-09", half, "stations.csv.next-tok-1", label=store.sha256_label(half))
    _set_state(root, "data-2026-09", "stations.csv.next-tok-1", "starter")

    actions = store.recover_temporaries(s, "data-2026-09")

    assert actions == [
        "deleted-incomplete:stations.csv.next-tok-1",
        "restored:stations.csv.old-tok",
    ]
    assert [a.name for a in s.list_assets("data-2026-09")] == ["stations.csv"]
    got = s.download("data-2026-09", "stations.csv", tmp_path / "got.csv")
    assert got.read_bytes() == b"stations-v1\n"


def test_recover_takes_the_newest_verifying_next_and_clears_the_rest(tmp_path: Path):
    s, _ = _seed(tmp_path)
    old_try = _write(tmp_path, "a.csv", b"stations-v2\n")
    new_try = _write(tmp_path, "b.csv", b"stations-v3\n")
    previous = _write(tmp_path, "c.csv", b"stations-v1\n")
    s.upload_new("data-2026-09", previous, "stations.csv.old-tok")
    s.upload_new(
        "data-2026-09",
        old_try,
        "stations.csv.next-tok-1",
        label=store.sha256_label(old_try),
    )
    s.upload_new(
        "data-2026-09",
        new_try,
        "stations.csv.next-tok-2",
        label=store.sha256_label(new_try),
    )

    store.recover_temporaries(s, "data-2026-09")

    assert [a.name for a in s.list_assets("data-2026-09")] == ["stations.csv"]
    got = s.download("data-2026-09", "stations.csv", tmp_path / "got.csv")
    assert got.read_bytes() == b"stations-v3\n"


def test_recover_prefers_a_newer_old_over_a_stale_verifying_next(tmp_path: Path):
    # replace_atomic deliberately never deletes a foreign-token leftover, so a
    # self-consistent .next from an earlier, abandoned run can still verify
    # even though a later run already produced a newer .old. The newer one
    # must win by created_at -- not "any verifying .next beats any .old" --
    # or recovery would resurrect stale content over the run that came after.
    s, _ = _seed(tmp_path)
    stale = _write(tmp_path, "stale.csv", b"stations-stale\n")
    newer = _write(tmp_path, "newer.csv", b"stations-newer\n")
    s.upload_new(
        "data-2026-09",
        stale,
        "stations.csv.next-stale-1",
        label=store.sha256_label(stale),
    )
    s.upload_new("data-2026-09", newer, "stations.csv.old-tok")

    actions = store.recover_temporaries(s, "data-2026-09")

    assert actions == [
        "restored:stations.csv.old-tok",
        "deleted-leftover:stations.csv.next-stale-1",
    ]
    assert [a.name for a in s.list_assets("data-2026-09")] == ["stations.csv"]
    got = s.download("data-2026-09", "stations.csv", tmp_path / "got.csv")
    assert got.read_bytes() == b"stations-newer\n"


def test_recover_handles_several_names_and_a_closed_release(tmp_path: Path):
    s, _ = _seed(tmp_path)
    s.update_release("data-2026-09", prerelease=False)  # a closed month
    rows = _write(tmp_path, "rows.csv", b"rows-v1\n")
    manifest = _write(tmp_path, "m.json", b"{}\n")
    s.upload_new("data-2026-09", rows, "costco-gas-2026-09-15.csv.gz")
    s.upload_new(
        "data-2026-09",
        rows,
        "costco-gas-2026-09-15.csv.gz.next-tok-1",
        label=store.sha256_label(rows),
    )
    s.upload_new(
        "data-2026-09",
        manifest,
        "manifest-2026-09.json.next-tok-1",
        label=store.sha256_label(manifest),
    )

    store.recover_temporaries(s, "data-2026-09")

    assert sorted(a.name for a in s.list_assets("data-2026-09")) == [
        "costco-gas-2026-09-15.csv.gz",
        "manifest-2026-09.json",
    ]
    assert s.get_release("data-2026-09").prerelease is False


def test_recover_is_a_no_op_without_temporaries(tmp_path: Path):
    s, _ = _seed(tmp_path)
    rows = _write(tmp_path, "rows.csv", b"rows-v1\n")
    s.upload_new("data-2026-09", rows, "rows.csv")
    assert store.recover_temporaries(s, "data-2026-09") == []


def _seed_four_releases(tmp_path: Path) -> store.LocalReleaseStore:
    s = store.LocalReleaseStore(tmp_path / "releases")
    s.ensure_release("current", "Current data", "b", False, "true")
    s.ensure_release("data-2026-09", "September 2026", "b", True, "false")
    s.ensure_release("data-2026", "2026", "b", False, "false")
    s.ensure_release("notes", "not a data release", "b", False, "false")
    return s


def test_recovery_tags_lists_current_and_every_data_release(tmp_path: Path):
    s = _seed_four_releases(tmp_path)
    assert store.recovery_tags(s) == ["current", "data-2026", "data-2026-09"]


def test_recover_temporaries_over_recovery_tags_leaves_other_releases_alone(
    tmp_path: Path,
):
    s = _seed_four_releases(tmp_path)
    payload = _write(tmp_path, "p.csv", b"p\n")
    label = store.sha256_label(payload)
    s.upload_new("current", payload, "stations.csv.old-tok")
    s.upload_new("data-2026", payload, "costco-gas-2026.parquet.next-tok-1", label=label)
    s.upload_new("notes", payload, "readme.txt.old-tok")

    actions = {tag: store.recover_temporaries(s, tag) for tag in store.recovery_tags(s)}

    assert sorted(tag for tag, done in actions.items() if done) == [
        "current",
        "data-2026",
    ]
    assert actions["data-2026-09"] == []
    assert [a.name for a in s.list_assets("current")] == ["stations.csv"]
    assert [a.name for a in s.list_assets("data-2026")] == ["costco-gas-2026.parquet"]
    # `notes` is not a data release, so recovery never looked at it.
    assert [a.name for a in s.list_assets("notes")] == ["readme.txt.old-tok"]


class FakeGitHub:
    """An in-memory stand-in for the GitHub Releases REST API.

    Served through httpx.MockTransport, so no test ever touches the network.
    Asset downloads are answered on a separate host, the way GitHub answers
    them from `browser_download_url`.
    """

    def __init__(self, owner: str = "acme", repo: str = "gas") -> None:
        self.owner = owner
        self.repo = repo
        self.base = f"/repos/{owner}/{repo}"
        self.releases: dict[int, dict] = {}
        self.assets: dict[int, dict] = {}
        self.latest_tag: str | None = None
        self.calls: list[dict] = []
        self.fail_promote: dict[str, int] = {}
        self.rate_limit_queue: list[dict] = []
        self.null_digest = False
        self._next_release = 1
        self._next_asset = 1000
        self._clock = datetime(2026, 9, 15, 18, 0, 0, tzinfo=UTC)

    # -- seeding ---------------------------------------------------------
    def add_release(
        self,
        tag: str,
        *,
        prerelease: bool = False,
        immutable: bool = False,
        latest: bool = False,
    ) -> dict:
        release_id = self._next_release
        self._next_release += 1
        self.releases[release_id] = {
            "id": release_id,
            "tag_name": tag,
            "name": tag,
            "body": "",
            "prerelease": prerelease,
            "immutable": immutable,
            "html_url": f"https://example.invalid/{tag}",
        }
        if latest:
            self.latest_tag = tag
        return self.releases[release_id]

    def add_asset(
        self,
        tag: str,
        name: str,
        data: bytes,
        *,
        label: str | None = None,
        state: str = "uploaded",
    ) -> dict:
        release = self._release_by_tag(tag)
        assert release is not None
        return self._store_asset(release["id"], name, data, label=label, state=state)

    def _store_asset(
        self,
        release_id: int,
        name: str,
        data: bytes,
        *,
        label: str | None = None,
        state: str = "uploaded",
    ) -> dict:
        asset_id = self._next_asset
        self._next_asset += 1
        self._clock += timedelta(seconds=1)
        self.assets[asset_id] = {
            "id": asset_id,
            "release_id": release_id,
            "name": name,
            "label": label or "",
            "state": state,
            "body": data,
            "created_at": self._clock.strftime("%Y-%m-%dT%H:%M:%SZ"),
        }
        return self.assets[asset_id]

    def _release_by_tag(self, tag: str | None) -> dict | None:
        for release in self.releases.values():
            if release["tag_name"] == tag:
                return release
        return None

    def asset_names(self, tag: str) -> list[str]:
        release = self._release_by_tag(tag)
        assert release is not None
        return sorted(a["name"] for a in self.assets.values() if a["release_id"] == release["id"])

    def _asset(self, tag: str, name: str) -> dict:
        release = self._release_by_tag(tag)
        assert release is not None
        for asset in self.assets.values():
            if asset["release_id"] == release["id"] and asset["name"] == name:
                return asset
        raise KeyError(name)

    def body_of(self, tag: str, name: str) -> bytes:
        return self._asset(tag, name)["body"]

    def label_of(self, tag: str, name: str) -> str:
        return self._asset(tag, name)["label"]

    def corrupt_body(self, tag: str, name: str, data: bytes) -> None:
        """Overwrite a stored asset's bytes in place, as a torn transfer would.

        The caller is responsible for keeping `len(data)` equal to the
        original body so a size check alone cannot catch the corruption --
        only a digest or a hash of the actual bytes can.
        """
        self._asset(tag, name)["body"] = data

    # -- serialisation ---------------------------------------------------
    def _asset_json(self, asset: dict) -> dict:
        digest = None if self.null_digest else "sha256:" + hashlib.sha256(asset["body"]).hexdigest()
        return {
            "id": asset["id"],
            "name": asset["name"],
            "label": asset["label"],
            "state": asset["state"],
            "size": len(asset["body"]),
            "digest": digest,
            "created_at": asset["created_at"],
            "browser_download_url": f"https://downloads.invalid/assets/{asset['id']}",
        }

    # -- transport -------------------------------------------------------
    def transport(self) -> httpx.MockTransport:
        return httpx.MockTransport(self.handle)

    def handle(self, request: httpx.Request) -> httpx.Response:
        self.calls.append(
            {
                "method": request.method,
                "url": str(request.url),
                "auth": request.headers.get("authorization"),
            }
        )
        path = request.url.path
        if request.url.host == "downloads.invalid":
            asset_id = int(path.rsplit("/", 1)[-1])
            return httpx.Response(200, content=self.assets[asset_id]["body"])
        if request.method in ("POST", "PATCH", "DELETE") and self.rate_limit_queue:
            headers = self.rate_limit_queue.pop(0)
            return httpx.Response(429, headers=headers, json={"message": "rate limited"})

        if request.method == "GET" and path == f"{self.base}/releases/latest":
            release = self._release_by_tag(self.latest_tag)
            if release is None:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json=release)
        if request.method == "GET" and path.startswith(f"{self.base}/releases/tags/"):
            release = self._release_by_tag(path.rsplit("/", 1)[-1])
            if release is None:
                return httpx.Response(404, json={"message": "Not Found"})
            return httpx.Response(200, json=release)
        if request.method == "GET" and path == f"{self.base}/releases":
            # Newest created first and paginated, the way the real endpoint
            # answers: the repository's oldest release -- `current` -- is the
            # last item, so it is the first thing an unpaginated read loses.
            items = list(reversed(self.releases.values()))
            page = int(request.url.params.get("page", "1"))
            per_page = int(request.url.params.get("per_page", "100"))
            return httpx.Response(200, json=items[(page - 1) * per_page : page * per_page])
        if request.method == "POST" and path == f"{self.base}/releases":
            payload = json.loads(request.content)
            release = self.add_release(
                payload["tag_name"],
                prerelease=bool(payload.get("prerelease")),
                latest=payload.get("make_latest") == "true",
            )
            release["name"] = payload.get("name", "")
            release["body"] = payload.get("body", "")
            return httpx.Response(201, json=release)

        match = re.fullmatch(rf"{re.escape(self.base)}/releases/(\d+)", path)
        if match and request.method == "PATCH":
            release = self.releases[int(match.group(1))]
            payload = json.loads(request.content)
            for key in ("name", "body", "prerelease"):
                if key in payload:
                    release[key] = payload[key]
            if payload.get("make_latest") == "true":
                self.latest_tag = release["tag_name"]
            return httpx.Response(200, json=release)

        match = re.fullmatch(rf"{re.escape(self.base)}/releases/(\d+)/assets", path)
        if match and request.method == "GET":
            release_id = int(match.group(1))
            items = [
                self._asset_json(a) for a in self.assets.values() if a["release_id"] == release_id
            ]
            page = int(request.url.params.get("page", "1"))
            per_page = int(request.url.params.get("per_page", "100"))
            return httpx.Response(200, json=items[(page - 1) * per_page : page * per_page])
        if match and request.method == "POST":
            release_id = int(match.group(1))
            name = request.url.params["name"]
            clash = any(
                a["release_id"] == release_id and a["name"] == name for a in self.assets.values()
            )
            if clash:
                return httpx.Response(422, json={"message": "Validation Failed"})
            asset = self._store_asset(
                release_id, name, request.content, label=request.url.params.get("label")
            )
            return httpx.Response(201, json=self._asset_json(asset))

        match = re.fullmatch(rf"{re.escape(self.base)}/releases/assets/(\d+)", path)
        if match and request.method == "PATCH":
            asset = self.assets[int(match.group(1))]
            payload = json.loads(request.content)
            new_name = payload.get("name", asset["name"])
            remaining = self.fail_promote.get(new_name, 0)
            if remaining and asset["name"].startswith(f"{new_name}.next-"):
                self.fail_promote[new_name] = remaining - 1
                return httpx.Response(422, json={"message": "Validation Failed"})
            asset["name"] = new_name
            if "label" in payload:
                asset["label"] = payload["label"] or ""
            return httpx.Response(200, json=self._asset_json(asset))
        if match and request.method == "DELETE":
            self.assets.pop(int(match.group(1)), None)
            return httpx.Response(204)

        return httpx.Response(404, json={"message": f"no route for {request.method} {path}"})


@pytest.fixture
def writer_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GITHUB_TOKEN", "ghs-test-token")
    monkeypatch.setenv("COSTCO_GAS_WRITER", "1")


def test_github_reads_need_no_token_and_no_auth_on_downloads(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    fake = FakeGitHub()
    fake.add_release("current", latest=True)
    fake.add_asset("current", "stations.csv", b"station_key,country\nUS-1364,US\n")
    s = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())

    release = s.get_release("current")
    assert release is not None
    assert release.tag == "current"
    assert release.is_latest is True
    assert s.get_release("data-2030-01") is None
    assert s.get_latest().tag == "current"
    assert [a.name for a in s.list_assets("current")] == ["stations.csv"]

    dest = s.download("current", "stations.csv", tmp_path / "stations.csv")
    assert dest.read_bytes() == b"station_key,country\nUS-1364,US\n"

    downloads = [c for c in fake.calls if "downloads.invalid" in c["url"]]
    assert len(downloads) == 1
    # browser_download_url is public: it costs no API quota, and the signed
    # redirect target rejects an Authorization header.
    assert downloads[0]["auth"] is None


def test_github_get_release_returns_none_when_the_release_is_missing(writer_env):
    fake = FakeGitHub()
    s = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())
    assert s.get_release("data-2030-01") is None


def test_github_list_assets_raises_when_the_release_is_missing(writer_env):
    fake = FakeGitHub()
    s = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())
    with pytest.raises(store.StorageError, match="release not found: data-2030-01"):
        s.list_assets("data-2030-01")


def test_github_download_propagates_release_not_found_instead_of_asset_not_found(
    tmp_path: Path, writer_env
):
    # A 404 on the release itself -- a wrong repo, a renamed release, a token
    # that cannot see it -- must never look like "this file was never
    # published", or publish logic would happily start writing fresh history.
    fake = FakeGitHub()
    s = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())
    with pytest.raises(store.StorageError, match="release not found: data-2030-01"):
        s.download("data-2030-01", "stations.csv", tmp_path / "d.csv")


def test_github_read_resolved_raises_asset_not_found_when_the_release_is_missing(
    tmp_path: Path, writer_env
):
    # read_resolved is the pre-recovery, pre-publish read path: the very
    # first capture legitimately finds no `current` release yet.
    fake = FakeGitHub()
    s = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())
    with pytest.raises(store.AssetNotFound):
        s.read_resolved("current", "stations.csv", tmp_path / "d.csv")


def test_github_is_latest_comes_from_the_releases_latest_endpoint(writer_env):
    # The Latest pointer is repository-wide, so it cannot be read off the
    # release payload: the store asks GET /releases/latest and compares tags.
    fake = FakeGitHub()
    fake.add_release("data-2026-09", prerelease=True)
    fake.add_release("current", latest=True)
    s = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())

    assert s.get_release("current").is_latest is True
    assert s.get_release("data-2026-09").is_latest is False
    assert s.get_latest().is_latest is True
    assert {r.tag: r.is_latest for r in s.list_releases()} == {
        "current": True,
        "data-2026-09": False,
    }

    moved = s.update_release("data-2026-09", make_latest="true")

    assert moved.make_latest == "true"
    assert moved.is_latest is True
    assert s.get_release("current").is_latest is False


def test_github_list_releases_reads_every_page(writer_env):
    """`GET /releases` is paginated, newest created first.

    `current` is created once, before any `data-*` release, and is never
    recreated, so it is the last item the endpoint returns. An unpaginated read
    drops it as soon as the repository holds more than one page of releases:
    `recovery_tags` would stop returning `current` -- §8.4 recovery would never
    run again on the release that holds the published data -- and `month_tags`
    would silently truncate, hiding older months from every roll-up.
    """
    fake = FakeGitHub()
    fake.add_release("current", latest=True)
    months = [f"data-{year}-{month:02d}" for year in range(2026, 2036) for month in range(1, 13)]
    for tag in months:
        fake.add_release(tag, prerelease=True)
    s = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())

    tags = [r.tag for r in s.list_releases()]

    assert len(tags) == 121
    assert sorted(tags) == sorted(["current", *months])
    # `current` is only reachable on the second page, which is the whole point.
    assert "current" not in tags[:100]
    listings = [c["url"] for c in fake.calls if "/releases?" in c["url"]]
    assert listings == [
        "https://api.github.com/repos/acme/gas/releases?per_page=100&page=1",
        "https://api.github.com/repos/acme/gas/releases?per_page=100&page=2",
    ]
    # And the consequence the pages exist for: recovery still sees `current`.
    assert store.recovery_tags(s)[0] == "current"


def test_github_writes_require_the_token_and_the_writer_flag(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    fake = FakeGitHub()
    fake.add_release("data-2026-09", prerelease=True)
    payload = _write(tmp_path, "rows.csv", b"rows-v1\n")

    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setenv("COSTCO_GAS_WRITER", "1")
    no_token = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())
    with pytest.raises(store.StorageError, match="GITHUB_TOKEN"):
        no_token.upload_new("data-2026-09", payload, "rows.csv")

    monkeypatch.setenv("GITHUB_TOKEN", "ghs-test-token")
    monkeypatch.delenv("COSTCO_GAS_WRITER", raising=False)
    no_flag = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())
    with pytest.raises(store.StorageError, match="COSTCO_GAS_WRITER"):
        no_flag.ensure_release("current", "Current data", "b", False, "true")

    assert not [c for c in fake.calls if c["method"] in ("POST", "PATCH", "DELETE")]


def test_github_ensure_release_creates_updates_and_refuses_immutable(writer_env):
    fake = FakeGitHub()
    s = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())

    created = s.ensure_release("data-2026-09", "September 2026", "body", True, "false")
    assert created.tag == "data-2026-09"
    assert created.prerelease is True
    assert created.make_latest == "false"
    assert created.is_latest is False
    assert s.ensure_release("data-2026-09", "x", "y", False, "false").title == ("September 2026")

    closed = s.update_release("data-2026-09", prerelease=False, body="closed")
    assert closed.prerelease is False
    assert closed.body == "closed"
    assert closed.is_latest is False
    assert [r.tag for r in s.list_releases()] == ["data-2026-09"]

    fake.add_release("current", immutable=True)
    with pytest.raises(store.StorageError, match="immutable release: current"):
        s.ensure_release("current", "Current data", "b", False, "true")


def test_github_upload_new_reports_a_422_for_a_duplicate_name(tmp_path: Path, writer_env):
    fake = FakeGitHub()
    fake.add_release("data-2026-09", prerelease=True)
    fake.add_asset("data-2026-09", "capture-2026-09-15T1817Z.tar.gz", b"bundle")
    s = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())
    payload = _write(tmp_path, "bundle.tar.gz", b"bundle")

    with pytest.raises(store.StorageError) as excinfo:
        s.upload_new("data-2026-09", payload, "capture-2026-09-15T1817Z.tar.gz")
    assert excinfo.value.status == 422

    uploaded = s.upload_new("data-2026-09", payload, "capture-2026-09-15T1818Z.tar.gz")
    assert uploaded.state == "uploaded"
    assert fake.body_of("data-2026-09", "capture-2026-09-15T1818Z.tar.gz") == b"bundle"


def test_github_read_resolved_verifies_by_hashing_when_digest_is_null(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    fake = FakeGitHub()
    fake.null_digest = True  # older assets carry no digest
    fake.add_release("current")
    good = b"stations-v2\n"
    fake.add_asset(
        "current",
        "stations.csv.next-tok-1",
        good,
        label="sha256:" + hashlib.sha256(good).hexdigest(),
    )
    fake.add_asset("current", "stations.csv.next-tok-2", b"torn\n", label="sha256:" + "0" * 64)
    s = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())

    dest = s.read_resolved("current", "stations.csv", tmp_path / "s.csv")
    assert dest.read_bytes() == good


def test_github_replace_atomic_accepts_a_null_digest_upload_that_hashes_correctly(
    tmp_path: Path, writer_env
):
    # `_verify_asset`'s download-and-hash fallback (for the null-digest case)
    # is only exercised through a poll loop like `replace_atomic`'s, not
    # through `read_resolved`'s own inline hashing. This proves it accepts a
    # genuine upload, and does so by hashing the bytes at most once.
    fake = FakeGitHub()
    fake.null_digest = True
    fake.add_release("current")
    fake.add_asset("current", "stations.csv", b"stations-v1\n")
    clock = FakeClock()
    s = store.GitHubReleaseStore(
        "acme", "gas", transport=fake.transport(), sleep=clock.sleep, monotonic=clock.now
    )
    payload = _write(tmp_path, "v2.csv", b"stations-v2\n")

    asset = s.replace_atomic("current", payload, "stations.csv", "tok")

    assert asset.name == "stations.csv"
    assert fake.body_of("current", "stations.csv") == b"stations-v2\n"
    downloads = [c for c in fake.calls if "downloads.invalid" in c["url"]]
    assert len(downloads) == 1  # hashed once, not re-downloaded on every poll
    assert clock.slept == []


def test_github_replace_atomic_rejects_a_null_digest_upload_that_hashes_wrong(
    tmp_path: Path, writer_env
):
    # A torn transfer that GitHub reports as `state: "uploaded"` with a
    # matching size but no digest must still be rejected -- and rejected
    # after hashing it exactly once, not after 60 repeated downloads.
    fake = FakeGitHub()
    fake.null_digest = True
    fake.add_release("current")
    fake.add_asset("current", "stations.csv", b"stations-v1\n")

    class CorruptingGitHubStore(store.GitHubReleaseStore):
        """The transport silently stores torn bytes for every upload."""

        def upload_new(self, tag, path, name, label=None):
            asset = super().upload_new(tag, path, name, label=label)
            fake.corrupt_body(tag, name, b"stations-XX\n")  # same length, wrong bytes
            return asset

    clock = FakeClock()
    s = CorruptingGitHubStore(
        "acme", "gas", transport=fake.transport(), sleep=clock.sleep, monotonic=clock.now
    )
    payload = _write(tmp_path, "v2.csv", b"stations-v2\n")

    with pytest.raises(store.StorageError, match="did not verify"):
        s.replace_atomic("current", payload, "stations.csv", "tok")

    downloads = [c for c in fake.calls if "downloads.invalid" in c["url"]]
    assert len(downloads) == 1  # rejected after hashing once, not retried
    assert clock.slept == []
    assert fake.body_of("current", "stations.csv") == b"stations-v1\n"


def test_github_replace_atomic_retries_a_422_promotion(tmp_path: Path, writer_env):
    fake = FakeGitHub()
    fake.add_release("current")
    fake.add_asset("current", "stations.csv", b"stations-v1\n")
    fake.fail_promote["stations.csv"] = 2
    clock = FakeClock()
    s = store.GitHubReleaseStore(
        "acme", "gas", transport=fake.transport(), sleep=clock.sleep, monotonic=clock.now
    )
    payload = _write(tmp_path, "v2.csv", b"stations-v2\n")

    asset = s.replace_atomic("current", payload, "stations.csv", "2026-09-15T1817Z")

    assert clock.slept == [5.0, 10.0]
    assert asset.name == "stations.csv"
    assert asset.label is None
    assert fake.asset_names("current") == ["stations.csv"]
    assert fake.body_of("current", "stations.csv") == b"stations-v2\n"
    assert fake.label_of("current", "stations.csv") == ""


def test_github_replace_atomic_rolls_back_when_the_promotion_keeps_failing(
    tmp_path: Path, writer_env
):
    fake = FakeGitHub()
    fake.add_release("current")
    fake.add_asset("current", "stations.csv", b"stations-v1\n")
    fake.fail_promote["stations.csv"] = 99
    clock = FakeClock()
    s = store.GitHubReleaseStore(
        "acme", "gas", transport=fake.transport(), sleep=clock.sleep, monotonic=clock.now
    )
    payload = _write(tmp_path, "v2.csv", b"stations-v2\n")

    with pytest.raises(store.StorageError, match="could not promote"):
        s.replace_atomic("current", payload, "stations.csv", "tok")

    assert clock.slept == [5.0, 10.0, 20.0, 40.0, 80.0]
    # The previous copy is back under its real name, so readers are never empty.
    assert fake.body_of("current", "stations.csv") == b"stations-v1\n"
    assert "stations.csv.next-tok-1" in fake.asset_names("current")
    # And the next recovery pass clears the unusable upload.
    store.recover_temporaries(s, "current")
    assert fake.asset_names("current") == ["stations.csv"]


def test_recover_promotes_the_next_when_it_ties_the_old_to_the_second(writer_env):
    """On an exact `created_at` tie the verified `.next` wins, not the `.old`.

    GitHub stamps an asset's `created_at` to the second, and step 3 and step 4
    of one `replace_atomic` -- rename the live copy to `.old`, rename the new
    copy in -- are two requests apart, so a tie is ordinary rather than
    hypothetical. A crash between them leaves exactly this state. The `.next`
    is the copy that run verified and was installing, and the `.old` is the
    copy it was replacing, so preferring the `.old` on a tie would undo a
    completed publish. Only a *newer* `.old`, from a later run, outranks a
    stale `.next`, which the test above covers.
    """
    fake = FakeGitHub()
    fake.add_release("current", latest=True)
    payload = b"stations-v2\n"
    label = "sha256:" + hashlib.sha256(payload).hexdigest()
    incoming = fake.add_asset("current", "stations.csv.next-tok-1", payload, label=label)
    previous = fake.add_asset("current", "stations.csv.old-tok", b"stations-v1\n")
    previous["created_at"] = incoming["created_at"]  # the same second
    s = store.GitHubReleaseStore("acme", "gas", transport=fake.transport())

    actions = store.recover_temporaries(s, "current")

    assert actions == [
        "promoted:stations.csv.next-tok-1",
        "deleted-leftover:stations.csv.old-tok",
    ]
    assert fake.asset_names("current") == ["stations.csv"]
    assert fake.body_of("current", "stations.csv") == payload


def test_open_store_builds_both_kinds(tmp_path: Path):
    local = store.open_store(f"local:{tmp_path / 'releases'}")
    assert isinstance(local, store.LocalReleaseStore)
    local.ensure_release("current", "Current data", "b", False, "true")
    assert (tmp_path / "releases" / "current" / "_release.json").exists()

    remote = store.open_store("github:coatless-dashboard/costco-gas-prices")
    assert isinstance(remote, store.GitHubReleaseStore)
    assert store.DEFAULT_STORE == "github:coatless-dashboard/costco-gas-prices"

    with pytest.raises(store.StorageError, match="unsupported store spec"):
        store.open_store("s3://bucket/prefix")
    with pytest.raises(store.StorageError, match="unsupported store spec"):
        store.open_store("github:no-slash")


def _paced_store(fake: FakeGitHub, clock: FakeClock, **kwargs) -> object:
    return store.GitHubReleaseStore(
        "acme",
        "gas",
        transport=fake.transport(),
        sleep=clock.sleep,
        monotonic=clock.now,
        **kwargs,
    )


def test_writes_are_paced_to_sixty_per_minute(writer_env):
    fake = FakeGitHub()
    fake.add_release("current")
    for i in range(61):
        fake.add_asset("current", f"asset-{i:03d}.bin", b"x")
    clock = FakeClock()
    s = _paced_store(fake, clock)

    for asset in s.list_assets("current"):
        s.delete("current", asset.id)

    # The 61st write waits for the first one to leave the 60-second window.
    assert clock.slept == [60.0]
    assert fake.asset_names("current") == []


def test_writes_are_paced_to_450_per_hour(writer_env):
    fake = FakeGitHub()
    fake.add_release("current")
    for i in range(451):
        fake.add_asset("current", f"asset-{i:03d}.bin", b"x")
    clock = FakeClock()
    s = _paced_store(fake, clock)

    for asset in s.list_assets("current"):
        s.delete("current", asset.id)

    assert clock.slept.count(60.0) == 7  # one per full minute of writes
    assert clock.slept[-1] == 3180.0  # the 451st waits out the rest of the hour
    assert fake.asset_names("current") == []


def test_retry_after_is_honoured(writer_env):
    fake = FakeGitHub()
    fake.add_release("current")
    fake.add_asset("current", "stations.csv", b"stations-v1\n")
    fake.rate_limit_queue.append({"retry-after": "7"})
    clock = FakeClock()
    s = _paced_store(fake, clock)

    asset_id = s.list_assets("current")[0].id
    s.delete("current", asset_id)

    assert clock.slept == [7.0]
    assert fake.asset_names("current") == []


def test_x_ratelimit_reset_is_honoured(writer_env):
    fake = FakeGitHub()
    fake.add_release("current")
    fake.add_asset("current", "stations.csv", b"stations-v1\n")
    epoch = 1_789_000_000.0
    fake.rate_limit_queue.append(
        {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(epoch) + 45)}
    )
    clock = FakeClock()
    s = _paced_store(fake, clock, now_epoch=lambda: epoch)

    asset_id = s.list_assets("current")[0].id
    s.delete("current", asset_id)

    assert clock.slept == [45.0]
    assert fake.asset_names("current") == []
