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


def test_list_assets_of_a_missing_release_is_empty(tmp_path: Path):
    s, _ = _seed(tmp_path)
    assert s.list_assets("data-2030-01") == []


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


def test_download_raises_asset_not_found_when_nothing_is_there(tmp_path: Path):
    s, _ = _seed(tmp_path)
    with pytest.raises(store.AssetNotFound):
        s.download("data-2026-09", "rows.csv", tmp_path / "d.csv")
    with pytest.raises(store.AssetNotFound):
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
