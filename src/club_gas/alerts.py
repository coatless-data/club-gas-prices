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
    _feed_issues(status, issues, now)
    _ecom_api_issue(status, issues)
    _grade_issues(status, issues)
    _close_period_issue(status, issues, close_outcome)

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
        return any(
            asset.name == name and asset.state == "uploaded" for asset in store.list_assets(tag)
        )
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
            f"Publish succeeded in capture `{status.get('capture_id')}` and nothing is "
            "unpublished.",
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
    lines += [
        "",
        "This issue closes automatically after a successful publish with nothing unpublished.",
    ]
    return "\n".join(lines)


def _feed_issues(status: dict, issues: Issues, now: datetime) -> None:
    """One issue per FEED.

    The issue's identity is its title, so two chains in one country sharing a
    "Capture failing: US" title would have one closing the other's issue. The
    title names the feed, and so does the body -- a reader has to know which
    chain to go and look at.
    """
    for fid, entry in (status.get("feeds") or {}).items():
        state = entry.get("status")
        if state == "skipped":
            # A skipped feed carries every field over unchanged, so it must
            # not open, refresh or close anything.
            continue
        title = f"Capture failing: {fid}"
        if state == "failed":
            if int(entry.get("consecutive_failures") or 0) >= CAPTURE_FAILURE_THRESHOLD:
                issues.ensure_open(
                    title, _capture_failing_body(fid, entry, now), ["capture-failure"]
                )
        elif state in ("ok", "degraded"):
            issues.close(title, f"`{fid}` was `{state}` in capture `{status.get('capture_id')}`.")


def _capture_failing_body(fid: str, entry: dict, now: datetime) -> str:
    lines = [
        f"`{fid}` has failed {entry.get('consecutive_failures', 0)} captures in a row.",
        "",
        f"- Last successful capture: `{entry.get('last_success_capture_id') or 'none recorded'}`",
        f"- Checked at: {now:%Y-%m-%dT%H:%MZ}",
        "",
        "### Recent failed captures",
        "",
        "| capture_id | run | errors |",
        "| --- | --- | --- |",
    ]
    for item in entry.get("recent_errors") or []:
        errors = (
            "; ".join(_error_text(error) for error in item.get("errors") or []) or "(none recorded)"
        )
        run_url = item.get("run_url") or ""
        run_cell = f"[run]({run_url})" if run_url else "(no run url)"
        lines.append(f"| `{item.get('capture_id', '?')}` | {run_cell} | {errors} |")
    lines += [
        "",
        "This issue closes automatically after the next `ok` or `degraded` capture for this feed.",
    ]
    return "\n".join(lines)


def _error_text(error: object) -> str:
    if not isinstance(error, dict):
        return str(error).replace("|", "\\|")
    parts = [str(error.get("code", "error"))]
    for key in ("host", "http_status", "detail"):
        value = error.get(key)
        if value not in (None, ""):
            parts.append(f"{key}={value}")
    return " ".join(parts).replace("|", "\\|")


def _ecom_api_issue(status: dict, issues: Issues) -> None:
    ecom = status.get("ecom_api") or {}
    title = "ecom-api client-identifier rejected"
    if ecom.get("http_status") == 401:
        issues.ensure_open(
            title,
            "\n".join(
                [
                    "`ecom-api.costco.com` answered HTTP 401 for the warehouse locator.",
                    "",
                    "The public `client-identifier` has most likely been rotated. It lives in",
                    "`config/countries.toml`, under `[countries.US]`, as the "
                    "`ecom_client_identifier`",
                    "key (a sibling of the `[countries.US.params]` query table).",
                    "Until it is replaced, US and Canada run on cached metadata and are reported "
                    "as `degraded`.",
                    "",
                    f"- Capture: `{status.get('capture_id')}`",
                    f"- Run: {status.get('run_url') or '(none)'}",
                    "",
                    "This issue closes automatically after a capture whose ecom-api request "
                    "returns 200.",
                ]
            ),
        )
    elif ecom.get("attempted") and ecom.get("http_status") == 200:
        issues.close(title, f"ecom-api returned 200 in capture `{status.get('capture_id')}`.")


def _grade_issues(status: dict, issues: Issues) -> None:
    seen: set[tuple[str, str]] = set()
    for country, entry in (status.get("feeds") or {}).items():
        if entry.get("status") == "skipped":
            continue
        for warning in entry.get("warnings") or []:
            if warning.get("code") != "unknown_grade":
                continue
            label = str(warning.get("detail", "")).strip()
            if not label or (country, label) in seen:
                continue
            seen.add((country, label))
            issues.ensure_open(
                f"Unknown grade label: {country} {label}",
                "\n".join(
                    [
                        f"Capture `{status.get('capture_id')}` saw the grade label `{label}` for "
                        f"`{country}`,",
                        "which is not in `config/grades.csv`. Its rows were stored as `other`, "
                        "so they are",
                        "excluded from every comparison until the label is mapped.",
                        "",
                        "1. Add the row to `config/grades.csv` with its `grade`, `priority`, "
                        "`label`, `spec`,",
                        "   `spec_source` and `spec_source_url`.",
                        "2. Run **Actions -> Rebuild** for the affected month so the stored "
                        "rows are remapped.",
                        "3. Close this issue.",
                    ]
                ),
            )


def _close_period_issue(status: dict, issues: Issues, close_outcome: str) -> None:
    title = "Period close failing"
    if close_outcome == "failure":
        # close-periods opens this itself when it catches its own error; alerts
        # repeats it here because a killed or timed-out step cannot.
        issues.ensure_open(
            title,
            "\n".join(
                [
                    f"`close-periods` did not succeed in capture `{status.get('capture_id')}`.",
                    "",
                    f"- Run: {status.get('run_url') or '(none)'}",
                    "",
                    "Month and year releases stay open until it succeeds. `current` is still "
                    "updated by",
                    "`publish`, so the dashboard keeps working.",
                    "",
                    "This issue closes automatically after a `close-periods` run that exits 0.",
                ]
            ),
        )
    elif close_outcome == "success":
        issues.close(title, f"`close-periods` succeeded in capture `{status.get('capture_id')}`.")
