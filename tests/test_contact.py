"""The contact address is published only in a form address harvesters skip.

The README spells it out with [at] and [dot]. Written whole in any file here, it
is one scrape of a public repository away from a spam list, so this test builds
it from its parts and never holds it as one string.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

LOCAL = "support"
DOMAIN = ".".join(("caffeinatedmath", "com"))


def repository_files(root: Path) -> list[Path]:
    """Tracked files, and untracked ones git would add, so a new file is caught
    before its first commit rather than after."""
    listed = subprocess.run(
        ["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
        cwd=root,
        capture_output=True,
        check=True,
    ).stdout.decode("utf-8")
    return [root / name for name in listed.split("\0") if name]


def test_no_file_in_the_repository_holds_the_whole_address():
    # The domain after an @ covers every local part at it, not just this one.
    needles = [f"{LOCAL}@{DOMAIN}".encode(), f"@{DOMAIN}".encode()]
    files = [path for path in repository_files(ROOT) if path.is_file()]
    assert files, "git listed no files"
    # Bytes rather than text, so a file that does not decode is searched too
    # rather than skipped.
    leaks = [
        path.relative_to(ROOT).as_posix()
        for path in files
        if any(needle in path.read_bytes().lower() for needle in needles)
    ]
    assert leaks == []
