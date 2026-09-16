"""Contract tests for the GitHub Actions workflows.

These assert on the literal text of the workflow files rather than on a parsed
YAML tree, because the rules being enforced are about the literal text: GitHub
rejects a bare `!` at the start of an `if:` value, an action pin is only a pin
if the ref written in the file is a commit SHA, and a CLI call is only correct
if the subcommand and flags are spelled exactly as their subparser defines
them. A YAML parser normalizes quoting and would hide all three.
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import textwrap
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
SERIAL_CONCURRENCY = (
    "concurrency:\n  group: costco-gas-data\n  cancel-in-progress: false\n"
)


def read(name: str) -> str:
    return (WORKFLOWS / name).read_text(encoding="utf-8")


def unwrapped_if_expressions(name: str) -> list[str]:
    """`if:` values that GitHub would not parse, i.e. those starting with `!`."""
    bad = []
    for line in read(name).splitlines():
        match = re.match(r"\s*if:\s*(.*)$", line)
        if match and match.group(1).startswith("!"):
            bad.append(line.strip())
    return bad


def action_pins(name: str) -> list[tuple[str, str, str | None]]:
    """(action, ref, trailing version comment) for every `uses:` line."""
    pins = []
    for line in read(name).splitlines():
        match = re.search(r"uses:\s*(\S+)@(\S+)(\s+#\s*(\S+))?", line)
        if match:
            pins.append((match.group(1), match.group(2), match.group(4)))
    return pins


def assert_conventions(name: str) -> None:
    assert unwrapped_if_expressions(name) == []
    pins = action_pins(name)
    assert pins, f"{name} has no `uses:` lines"
    for action, ref, comment in pins:
        if action == "astral-sh/setup-uv":
            # Pinned by commit SHA: a tag can be moved onto different code, and
            # this action runs with a write-capable token in capture and rebuild.
            assert re.fullmatch(r"[0-9a-f]{40}", ref), (name, action, ref)
            assert comment is not None and comment.startswith("v"), (name, comment)
        else:
            assert re.fullmatch(r"v\d+", ref), (name, action, ref)


def test_capture_conventions():
    assert_conventions("capture.yml")
    text = read("capture.yml")
    # capture, discover and rebuild all write releases, so they share one
    # concurrency group and never cancel: cancelling a writer mid-run would
    # leave `.next-*` assets behind for the next run to recover.
    assert SERIAL_CONCURRENCY in text
    assert "permissions:\n  contents: write\n  issues: write\n" in text


def test_capture_triggers_and_job_settings():
    text = read("capture.yml")
    assert text.startswith("name: Capture\n")
    assert '- cron: "17 */6 * * *"' in text
    assert "run-name: ${{ inputs.dry_run && 'Capture (dry run)' || 'Capture' }}" in text
    assert "    timeout-minutes: 45\n" in text
    assert '      COSTCO_GAS_WRITER: "1"\n' in text
    assert "      COUNTRIES: ${{ inputs.countries || 'all' }}\n" in text
    assert "      FORCE_FALLBACK: ${{ inputs.force_fallback || '' }}\n" in text
    assert (
        "ref: ${{ github.event.repository.default_branch || github.ref_name }}" in text
    )
    assert "fetch-depth: 0" in text


def test_capture_step_commands_conditions_and_timeouts():
    text = read("capture.yml")
    # Every command is spelled exactly as its subparser defines it: `capture`
    # takes --out, --countries and --force-fallback; `publish` takes the capture
    # directory positionally; `close-periods` takes no flag here; `alerts` takes
    # status.json positionally plus the two outcome flags.
    commands = [
        'run: uv run costco-gas capture --out out --countries "$COUNTRIES"'
        ' --force-fallback "$FORCE_FALLBACK"',
        "run: uv run costco-gas publish out/capture",
        "run: uv run costco-gas close-periods",
        "uv run costco-gas alerts out/capture/status.json",
        '--publish-outcome "${{ steps.publish.outcome }}"',
        '--close-outcome "${{ steps.close.outcome }}"',
    ]
    for fragment in commands:
        assert fragment in text, fragment
    expected = [
        "if: ${{ always() }}",
        "if: ${{ inputs.dry_run != true && steps.capture.outputs.all_failed != 'true' }}",
        "if: ${{ steps.publish.outcome == 'success' }}",
        (
            "if: ${{ !cancelled() && inputs.dry_run != true"
            " && steps.capture.outcome == 'success' }}"
        ),
        (
            "if: ${{ inputs.dry_run != true"
            " && (steps.capture.outputs.all_failed == 'true'"
            " || steps.publish.outcome == 'failure') }}"
        ),
    ]
    for fragment in expected:
        assert fragment in text, fragment
    assert text.count("timeout-minutes: 15") == 1
    assert text.count("timeout-minutes: 10") == 2


def test_capture_uploads_the_artifact_on_every_run():
    text = read("capture.yml")
    assert (
        "name: capture-${{ steps.capture.outputs.capture_id"
        " || format('run-{0}-{1}', github.run_id, github.run_attempt) }}" in text
    )
    assert "retention-days: 30" in text
