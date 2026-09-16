"""Release storage.

Every data file this project publishes lives as an asset on a GitHub Release.
An asset is never overwritten in place: `replace_atomic` uploads the new bytes
under a reserved temporary name, verifies them, renames the live copy out of the
way, renames the new copy in, and only then deletes the old one.
`recover_temporaries` finishes or unwinds a replace that was interrupted, and it
is the only implementation of recovery in the package.

Reserved temporary names are `<name>.next-<token>-<n>` and `<name>.old-<token>`.
Only `read_resolved` and `recover_temporaries` may look at them.

`sha256_file`, `sha256_label` and `DEFAULT_STORE` also live here: capture, the
CLI, publish, rollup and rebuild import them from this module instead of
keeping copies of their own.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import tempfile
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Protocol

DEFAULT_STORE = "github:coatless-dashboard/costco-gas-prices"
SIDECAR_NAME = "_release.json"
LATEST_NAME = "_latest.json"


class AssetNotFound(Exception):
    """The asset does not exist, and neither does any temporary form of it."""


class StorageError(Exception):
    """Any other storage failure.

    `status` carries the HTTP status when the failure came from a response.
    A status of 422 means "an asset with that name already exists", which
    callers handle by listing the assets and comparing digests.
    """

    def __init__(self, message: str, *, status: int | None = None) -> None:
        super().__init__(message)
        self.status = status


def sha256_file(path: Path) -> str:
    """Return the hex SHA-256 of a file."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_label(path: Path) -> str:
    """Return the SHA-256 of a file in GitHub's `sha256:<hex>` digest format."""
    return f"sha256:{sha256_file(path)}"


def next_name(name: str, token: str, n: int) -> str:
    return f"{name}.next-{token}-{n}"


def old_name(name: str, token: str) -> str:
    return f"{name}.old-{token}"


def split_temp_name(asset_name: str) -> tuple[str, str] | None:
    """Split a reserved temporary name into (base name, "next" | "old").

    Returns None for an ordinary asset name.
    """
    for marker, kind in ((".next-", "next"), (".old-", "old")):
        index = asset_name.rfind(marker)
        if index > 0:
            return asset_name[:index], kind
    return None


@dataclass(frozen=True)
class Asset:
    name: str
    id: int
    size: int
    state: str  # "uploaded" once the bytes are there
    digest: str | None  # "sha256:<hex>", or None for assets GitHub never hashed
    label: str | None
    created_at: datetime
    download_url: str | None = None


@dataclass(frozen=True)
class Release:
    tag: str
    id: int
    title: str
    body: str
    prerelease: bool
    make_latest: str | None  # the write-side value: "true" | "false", else None
    is_latest: bool  # what the store answers now: the Latest release?
    immutable: bool
    url: str


class ReleaseStore(Protocol):
    def get_release(self, tag: str) -> Release | None: ...
    def get_latest(self) -> Release | None: ...
    def ensure_release(
        self,
        tag: str,
        title: str,
        body: str,
        prerelease: bool,
        make_latest: Literal["true", "false"],
    ) -> Release: ...
    def update_release(
        self,
        tag: str,
        *,
        title: str | None = None,
        body: str | None = None,
        prerelease: bool | None = None,
        make_latest: Literal["true", "false"] | None = None,
    ) -> Release: ...
    def list_releases(self) -> list[Release]: ...
    def list_assets(self, tag: str) -> list[Asset]: ...
    def download(self, tag: str, name: str, dest: Path) -> Path: ...
    def read_resolved(self, tag: str, name: str, dest: Path) -> Path: ...
    def upload_new(
        self, tag: str, path: Path, name: str, label: str | None = None
    ) -> Asset: ...
    def replace_atomic(self, tag: str, path: Path, name: str, token: str) -> Asset: ...
    def rename(
        self, tag: str, asset_id: int, new_name: str, label: str | None = None
    ) -> Asset: ...
    def delete(self, tag: str, asset_id: int) -> None: ...


class _BaseStore:
    """Behaviour shared by every store, written against the primitives below."""

    _sleep: Callable[[float], None]
    _monotonic: Callable[[], float]

    def list_assets(self, tag: str) -> list[Asset]:  # pragma: no cover - interface
        raise NotImplementedError

    def upload_new(
        self, tag: str, path: Path, name: str, label: str | None = None
    ) -> Asset:  # pragma: no cover - interface
        raise NotImplementedError

    def rename(
        self, tag: str, asset_id: int, new_name: str, label: str | None = None
    ) -> Asset:  # pragma: no cover - interface
        raise NotImplementedError

    def delete(self, tag: str, asset_id: int) -> None:  # pragma: no cover - interface
        raise NotImplementedError

    def _fetch_asset(self, tag: str, asset: Asset, dest: Path) -> Path:
        # pragma: no cover - interface
        raise NotImplementedError

    def _asset_by_name(self, tag: str, name: str) -> Asset | None:
        for asset in self.list_assets(tag):
            if asset.name == name:
                return asset
        return None

    def _temps_for(
        self, assets: list[Asset], base: str, kind: str | None = None
    ) -> list[Asset]:
        """Temporary assets of `base`, newest first."""
        out = []
        for asset in assets:
            parsed = split_temp_name(asset.name)
            if parsed is None or parsed[0] != base:
                continue
            if kind is not None and parsed[1] != kind:
                continue
            out.append(asset)
        return sorted(out, key=lambda a: a.created_at, reverse=True)

    def download(self, tag: str, name: str, dest: Path) -> Path:
        assets = self.list_assets(tag)
        match = next((a for a in assets if a.name == name), None)
        if match is None:
            temps = self._temps_for(assets, name)
            if temps:
                raise StorageError(
                    f"{tag}:{name} is missing while {len(temps)} temporary assets "
                    f"exist; run recovery before reading it"
                )
            raise AssetNotFound(f"{tag}:{name}")
        path = self._fetch_asset(tag, match, Path(dest))
        size = path.stat().st_size
        if size != match.size:
            raise StorageError(f"size mismatch for {tag}:{name}: {size} != {match.size}")
        if match.digest is not None:
            got = sha256_label(path)
            if got != match.digest:
                raise StorageError(
                    f"digest mismatch for {tag}:{name}: {got} != {match.digest}"
                )
        return path

    def _verify_asset(
        self, tag: str, asset: Asset, expected_label: str, expected_size: int | None
    ) -> bool:
        """True when the stored asset really holds the bytes `expected_label` names."""
        if asset.state != "uploaded":
            return False
        if expected_size is not None and asset.size != expected_size:
            return False
        if asset.digest is not None:
            return asset.digest == expected_label
        # GitHub reports no digest for older assets: download and hash instead.
        scratch = Path(tempfile.mkdtemp(prefix="costco-gas-verify-"))
        try:
            path = self._fetch_asset(tag, asset, scratch / "asset")
            return sha256_label(path) == expected_label
        finally:
            shutil.rmtree(scratch, ignore_errors=True)

    def read_resolved(self, tag: str, name: str, dest: Path) -> Path:
        """Read `<name>`, or the best temporary stand-in, without changing anything.

        This is for readers that run before recovery has had a chance to run:
        the capture command and discovery. Everything else reads `download`.
        Order: `<name>`; then the newest `.next-*` whose bytes hash to its own
        label; then the newest `.old-*`.
        """
        assets = self.list_assets(tag)
        match = next((a for a in assets if a.name == name), None)
        if match is not None:
            return self.download(tag, name, dest)

        for candidate in self._temps_for(assets, name, "next"):
            if candidate.state != "uploaded" or not candidate.label:
                continue
            if candidate.digest is not None and candidate.digest != candidate.label:
                continue
            path = self._fetch_asset(tag, candidate, Path(dest))
            if sha256_label(path) == candidate.label:
                return path

        olds = self._temps_for(assets, name, "old")
        for candidate in olds:
            if candidate.state == "uploaded":
                return self._fetch_asset(tag, candidate, Path(dest))

        raise AssetNotFound(f"{tag}:{name}")


class LocalReleaseStore(_BaseStore):
    """Releases in a local directory: one directory per tag, plus a JSON sidecar.

    The sidecar `_release.json` holds the release attributes and, for every
    asset, its id, label, digest, state and created_at. Asset bytes are plain
    files next to it, so a test can inspect or corrupt them directly. The
    repository-wide Latest pointer is a single `_latest.json` beside the tag
    directories.
    """

    def __init__(
        self,
        root: Path | str,
        *,
        sleep: Callable[[float], None] = time.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._root = Path(root)
        self._root.mkdir(parents=True, exist_ok=True)
        self._sleep = sleep
        self._monotonic = monotonic
        self._lock = threading.RLock()

    # -- sidecar ---------------------------------------------------------
    def _dir(self, tag: str) -> Path:
        return self._root / tag

    def _sidecar(self, tag: str) -> Path:
        return self._dir(tag) / SIDECAR_NAME

    def _read(self, tag: str) -> dict | None:
        path = self._sidecar(tag)
        if not path.exists():
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    def _write(self, tag: str, data: dict) -> None:
        path = self._sidecar(tag)
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.parent / (SIDECAR_NAME + ".tmp")
        tmp.write_text(json.dumps(data, indent=2, sort_keys=True), encoding="utf-8")
        tmp.replace(path)

    def _require(self, tag: str) -> dict:
        data = self._read(tag)
        if data is None:
            raise StorageError(f"release not found: {tag}")
        return data

    # -- releases --------------------------------------------------------
    def _latest_tag(self) -> str | None:
        """The tag the Latest marker points at, or None when there is none."""
        path = self._root / LATEST_NAME
        if not path.exists():
            return None
        return str(json.loads(path.read_text(encoding="utf-8"))["tag"])

    def get_release(self, tag: str) -> Release | None:
        data = self._read(tag)
        if data is None:
            return None
        return Release(
            tag=tag,
            id=int(data["id"]),
            title=data.get("title", ""),
            body=data.get("body", ""),
            prerelease=bool(data.get("prerelease", False)),
            make_latest=data.get("make_latest"),
            is_latest=self._latest_tag() == tag,
            immutable=bool(data.get("immutable", False)),
            url=str(self._dir(tag)),
        )

    def get_latest(self) -> Release | None:
        tag = self._latest_tag()
        return None if tag is None else self.get_release(tag)

    def _set_latest(self, tag: str) -> None:
        (self._root / LATEST_NAME).write_text(json.dumps({"tag": tag}), encoding="utf-8")

    def ensure_release(
        self,
        tag: str,
        title: str,
        body: str,
        prerelease: bool,
        make_latest: Literal["true", "false"],
    ) -> Release:
        with self._lock:
            existing = self.get_release(tag)
            if existing is not None:
                if existing.immutable:
                    raise StorageError(f"immutable release: {tag}")
                return existing
            self._dir(tag).mkdir(parents=True, exist_ok=True)
            self._write(
                tag,
                {
                    "id": len(self.list_releases()) + 1,
                    "title": title,
                    "body": body,
                    "prerelease": prerelease,
                    "make_latest": make_latest,
                    "immutable": False,
                    "next_asset_id": 1,
                    "assets": {},
                },
            )
            if make_latest == "true":
                self._set_latest(tag)
            release = self.get_release(tag)
            assert release is not None
            return release

    def update_release(
        self,
        tag: str,
        *,
        title: str | None = None,
        body: str | None = None,
        prerelease: bool | None = None,
        make_latest: Literal["true", "false"] | None = None,
    ) -> Release:
        with self._lock:
            data = self._require(tag)
            if title is not None:
                data["title"] = title
            if body is not None:
                data["body"] = body
            if prerelease is not None:
                data["prerelease"] = prerelease
            if make_latest is not None:
                data["make_latest"] = make_latest
            self._write(tag, data)
            if make_latest == "true":
                self._set_latest(tag)
            release = self.get_release(tag)
            assert release is not None
            return release

    def list_releases(self) -> list[Release]:
        out: list[Release] = []
        for child in sorted(self._root.iterdir()):
            if child.is_dir() and (child / SIDECAR_NAME).exists():
                release = self.get_release(child.name)
                if release is not None:
                    out.append(release)
        return out

    # -- assets ----------------------------------------------------------
    def _asset_from_entry(self, tag: str, name: str, entry: dict) -> Asset:
        path = self._dir(tag) / name
        return Asset(
            name=name,
            id=int(entry["id"]),
            size=path.stat().st_size if path.exists() else 0,
            state=entry.get("state", "uploaded"),
            digest=entry.get("digest"),
            label=entry.get("label") or None,
            created_at=datetime.fromisoformat(entry["created_at"]),
        )

    def list_assets(self, tag: str) -> list[Asset]:
        data = self._read(tag)
        if data is None:
            return []
        return [
            self._asset_from_entry(tag, name, entry)
            for name, entry in sorted(data["assets"].items())
        ]

    def _new_created_at(self, data: dict) -> datetime:
        stamp = datetime.now(UTC)
        existing = [
            datetime.fromisoformat(entry["created_at"]) for entry in data["assets"].values()
        ]
        if existing:
            newest = max(existing)
            if stamp <= newest:
                stamp = newest + timedelta(microseconds=1)
        return stamp

    def upload_new(
        self, tag: str, path: Path, name: str, label: str | None = None
    ) -> Asset:
        path = Path(path)
        with self._lock:
            data = self._require(tag)
            if "/" in name or name in (SIDECAR_NAME, SIDECAR_NAME + ".tmp"):
                raise StorageError(f"reserved asset name: {name}")
            if name in data["assets"]:
                raise StorageError(f"asset exists: {tag}:{name}", status=422)
            shutil.copyfile(path, self._dir(tag) / name)
            entry = {
                "id": int(data["next_asset_id"]),
                "label": label,
                "state": "uploaded",
                "digest": sha256_label(path),
                "created_at": self._new_created_at(data).isoformat(),
            }
            data["next_asset_id"] = int(data["next_asset_id"]) + 1
            data["assets"][name] = entry
            self._write(tag, data)
            return self._asset_from_entry(tag, name, entry)

    def _find_by_id(self, data: dict, asset_id: int) -> str:
        for name, entry in data["assets"].items():
            if int(entry["id"]) == asset_id:
                return name
        raise StorageError(f"asset id not found: {asset_id}")

    def rename(
        self, tag: str, asset_id: int, new_name: str, label: str | None = None
    ) -> Asset:
        with self._lock:
            data = self._require(tag)
            name = self._find_by_id(data, asset_id)
            if new_name != name and new_name in data["assets"]:
                raise StorageError(f"asset exists: {tag}:{new_name}", status=422)
            entry = data["assets"].pop(name)
            if label is not None:
                entry["label"] = label or None
            (self._dir(tag) / name).replace(self._dir(tag) / new_name)
            data["assets"][new_name] = entry
            self._write(tag, data)
            return self._asset_from_entry(tag, new_name, entry)

    def delete(self, tag: str, asset_id: int) -> None:
        with self._lock:
            data = self._require(tag)
            name = self._find_by_id(data, asset_id)
            del data["assets"][name]
            (self._dir(tag) / name).unlink(missing_ok=True)
            self._write(tag, data)

    def _fetch_asset(self, tag: str, asset: Asset, dest: Path) -> Path:
        source = self._dir(tag) / asset.name
        if not source.exists():
            raise StorageError(f"asset bytes are missing: {tag}:{asset.name}")
        dest = Path(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, dest)
        return dest
