"""Capture-time alerts (spec 10.2).

``alerts`` runs after ``publish`` and ``close-periods`` in capture.yml. It is
the only step that knows whether those two succeeded, so besides opening and
closing issues it finishes the bookkeeping in ``status.json``: the publish
outcome, the publish failure counter, and the list of captures whose bundle
never reached its month release.

The issue titles are the identity of each alert; ``Issues`` (created in Task
14) refreshes an open issue with the same title rather than opening a second.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .issues import Issues
from .store import AssetNotFound, ReleaseStore, StorageError

PUBLISH_FAILURE_THRESHOLD = 2
CAPTURE_FAILURE_THRESHOLD = 3


def run_alerts(
    status_path: Path,
    *,
    publish_outcome: str,
    close_outcome: str,
    store: ReleaseStore,
    issues: Issues,
    now: datetime,
) -> dict:
    """Update ``status_path`` in place and raise or clear the (A) issues."""
    status_path = Path(status_path)
    status = json.loads(status_path.read_text(encoding="utf-8"))

    _apply_publish_bookkeeping(status, publish_outcome, store)
    status["close"] = {"outcome": close_outcome}

    _publish_issue(status, issues, now, publish_outcome)

    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    return status


def _apply_publish_bookkeeping(status: dict, publish_outcome: str, store: ReleaseStore) -> None:
    publish = status.setdefault("publish", {})
    publish["outcome"] = publish_outcome

    failures = int(publish.get("consecutive_failures") or 0)
    if publish_outcome == "failure":
        failures += 1
    elif publish_outcome == "success":
        failures = 0
    publish["consecutive_failures"] = failures

    unpublished = [dict(entry) for entry in (publish.get("unpublished") or [])]
    capture_id = status.get("capture_id")
    if (
        publish_outcome == "failure"
        and capture_id
        and not any(entry.get("capture_id") == capture_id for entry in unpublished)
    ):
        unpublished.append(
            {
                "capture_id": capture_id,
                "run_id": status.get("run_id"),
                "run_url": status.get("run_url"),
            }
        )
    # On every run, drop the entries whose bundle is now uploaded: publish's
    # reconciliation step merges those bundles on its own.
    publish["unpublished"] = [
        entry for entry in unpublished if not _bundle_uploaded(store, entry.get("capture_id"))
    ]


def _bundle_uploaded(store: ReleaseStore, capture_id: str | None) -> bool:
    if not capture_id:
        return False
    tag = "data-" + capture_id[:7]  # "2026-09-15T1817Z" -> "data-2026-09"
    name = f"capture-{capture_id}.tar.gz"
    try:
        if store.get_release(tag) is None:
            return False
        return any(asset.name == name and asset.state == "uploaded" for asset in store.list_assets(tag))
    except (AssetNotFound, StorageError) as exc:
        print(f"::warning::could not check for {name} in {tag}: {exc}")
        return False


def _publish_issue(status: dict, issues: Issues, now: datetime, publish_outcome: str) -> None:
    publish = status.get("publish", {})
    if publish["consecutive_failures"] >= PUBLISH_FAILURE_THRESHOLD or publish["unpublished"]:
        issues.ensure_open("Publish failing", _publish_failing_body(status, now))
    elif publish_outcome == "success":
        issues.close(
            "Publish failing",
            f"Publish succeeded in capture `{status.get('capture_id')}` and nothing is unpublished.",
        )


def _publish_failing_body(status: dict, now: datetime) -> str:
    publish = status.get("publish", {})
    lines = [
        f"`publish` has failed {publish.get('consecutive_failures', 0)} time(s) in a row.",
        "",
        f"- Latest capture: `{status.get('capture_id')}`",
        f"- Checked at: {now:%Y-%m-%dT%H:%MZ}",
        "",
        "### Captures whose bundle never reached its month release",
        "",
    ]
    unpublished = publish.get("unpublished") or []
    if unpublished:
        lines += ["| capture_id | run |", "| --- | --- |"]
        for entry in unpublished:
            run_url = entry.get("run_url") or ""
            run_id = entry.get("run_id")
            run_cell = f"[{run_id}]({run_url})" if run_url else str(run_id)
            lines.append(f"| `{entry.get('capture_id')}` | {run_cell} |")
        lines += [
            "",
            "Recover each one while its workflow artifact still exists (30 days):",
            "",
            "1. Open **Actions -> Rebuild -> Run workflow**.",
            "2. Set `scope` to `artifact` and `value` to the run id above.",
            "3. The run downloads that run's `capture-*` artifacts and publishes each one.",
        ]
    else:
        lines.append("None: every capture's bundle is uploaded.")
    lines += ["", "This issue closes automatically after a successful publish with nothing unpublished."]
    return "\n".join(lines)
