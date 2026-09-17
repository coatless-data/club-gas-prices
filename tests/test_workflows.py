"""Contract tests for the GitHub Actions workflows.

These assert on the literal text of the workflow files rather than on a parsed
YAML tree, because the rules being enforced are about the literal text: GitHub
rejects a bare `!` at the start of an `if:` value, an action pin is only a pin
if the ref written in the file is a commit SHA, and a CLI call is only correct
if the subcommand and flags are spelled exactly as their subparser defines
them. A YAML parser normalizes quoting and would hide all three.
"""

from __future__ import annotations

import re
from pathlib import Path

WORKFLOWS = Path(__file__).resolve().parents[1] / ".github" / "workflows"
SERIAL_CONCURRENCY = "concurrency:\n  group: club-gas-data\n  cancel-in-progress: false\n"


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
        if action.startswith("actions/"):
            # GitHub's own actions, on a floating major tag: the account that
            # would have to be compromised to move one is GitHub's own, and the
            # tag is how security fixes arrive without a commit here.
            assert re.fullmatch(r"v\d+", ref), (name, action, ref)
        else:
            # Everything third-party is pinned by commit SHA with the version in
            # a trailing comment. A tag can be moved onto different code, and
            # these run beside a write-capable token.
            assert re.fullmatch(r"[0-9a-f]{40}", ref), (name, action, ref)
            assert comment is not None and comment.startswith("v"), (name, comment)


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
    assert '      CLUB_GAS_WRITER: "1"\n' in text
    assert "      COUNTRIES: ${{ inputs.countries || 'all' }}\n" in text
    assert "      FORCE_FALLBACK: ${{ inputs.force_fallback || '' }}\n" in text
    assert "ref: ${{ github.event.repository.default_branch || github.ref_name }}" in text
    assert "fetch-depth: 0" in text


def test_capture_step_commands_conditions_and_timeouts():
    text = read("capture.yml")
    # Every command is spelled exactly as its subparser defines it: `capture`
    # takes --out, --countries and --force-fallback; `publish` takes the capture
    # directory positionally; `close-periods` takes no flag here; `alerts` takes
    # status.json positionally plus the two outcome flags.
    commands = [
        'run: uv run club-gas capture --out out --countries "$COUNTRIES"'
        ' --force-fallback "$FORCE_FALLBACK"',
        "run: uv run club-gas publish out/capture",
        "run: uv run club-gas close-periods",
        "uv run club-gas alerts out/capture/status.json",
        '--publish-outcome "${{ steps.publish.outcome }}"',
        '--close-outcome "${{ steps.close.outcome }}"',
    ]
    for fragment in commands:
        assert fragment in text, fragment
    expected = [
        "if: ${{ always() }}",
        "if: ${{ inputs.dry_run != true && steps.capture.outputs.all_failed != 'true' }}",
        "if: ${{ steps.publish.outcome == 'success' }}",
        ("if: ${{ !cancelled() && inputs.dry_run != true && steps.capture.outcome == 'success' }}"),
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


def test_discover_conventions_and_settings():
    assert_conventions("discover.yml")
    text = read("discover.yml")
    assert text.startswith("name: Discover\n")
    assert SERIAL_CONCURRENCY in text
    assert "permissions:\n  contents: read\n  issues: write\n" in text
    assert '- cron: "41 3 2 * *"' in text
    assert "    timeout-minutes: 20\n" in text
    # `discover` takes only an optional --root, which defaults to the working
    # directory, so the workflow passes no flag at all.
    assert "run: uv run club-gas discover\n" in text
    # discover only reads releases, so it must never claim the writer flag.
    assert "CLUB_GAS_WRITER" not in text


def test_rebuild_conventions_and_settings():
    assert_conventions("rebuild.yml")
    text = read("rebuild.yml")
    assert text.startswith("name: Rebuild\n")
    assert SERIAL_CONCURRENCY in text
    assert "permissions:\n  contents: write\n  issues: write\n  actions: read\n" in text
    assert "    timeout-minutes: 350\n" in text
    assert '      CLUB_GAS_WRITER: "1"\n' in text
    assert "      SCOPE: ${{ inputs.scope }}\n" in text
    assert "      VALUE: ${{ inputs.value }}\n" in text
    assert "ref: ${{ github.event.repository.default_branch || github.ref_name }}" in text
    # store.py refuses a GitHub release write without CLUB_GAS_WRITER=1, so it
    # is declared by exactly the two workflows that write releases.
    writers = {
        name
        for name in ("capture.yml", "discover.yml", "rebuild.yml")
        if 'CLUB_GAS_WRITER: "1"' in read(name)
    }
    assert writers == {"capture.yml", "rebuild.yml"}


def test_rebuild_dispatches_every_scope():
    text = read("rebuild.yml")
    for option in (
        "          - month\n",
        "          - year\n",
        "          - all\n",
        "          - artifact\n",
    ):
        assert option in text, option
    assert "if: ${{ inputs.scope == 'artifact' }}" in text
    assert "pattern: capture-*" in text
    assert "run-id: ${{ inputs.value }}" in text
    assert 'case "$SCOPE" in' in text
    assert 'uv run club-gas rebuild --month "$VALUE"' in text
    assert 'uv run club-gas rebuild --year "$VALUE"' in text
    assert "uv run club-gas rebuild --all" in text
    assert 'uv run club-gas publish "$d"' in text
    assert "if: ${{ steps.rebuild.outcome == 'success' }}" in text
    assert "run: uv run club-gas close-periods --rebuild-current" in text


def test_test_workflow_conventions():
    """test.yml pins like every other workflow.

    It was the one file no test read, which is how its third-party action came
    to be pinned by a rule `assert_conventions` did not actually state.
    """
    assert_conventions("test.yml")
    text = read("test.yml")
    assert text.startswith("name: Test\n")
    # The lint gate reads every workflow, so it has to keep running on all of them.
    assert "raven-actions/actionlint@" in text


def test_rebuild_accepts_both_artifact_layouts():
    """`download-artifact` changed where a single-match `pattern` lands.

    Through v4 every artifact got its own subdirectory. From v5 a `pattern`
    matching exactly one artifact extracts straight into `path` instead -- and a
    capture run holds exactly one artifact, so that is the ordinary case here,
    not an edge one. The glob has to accept both or the artifact scope breaks on
    every run it is pointed at.
    """
    text = read("rebuild.yml")
    assert "dirs=(artifact/*/capture)" in text
    assert '[ "${#dirs[@]}" -eq 0 ] && [ -d artifact/capture ]' in text
    assert "dirs=(artifact/capture)" in text
    # `artifact/capture` carries no wildcard, so nullglob would not drop it if it
    # did not exist: it has to stay behind the -d test, never in the glob list.
    assert "dirs=(artifact/*/capture artifact/capture)" not in text


def test_every_workflow_runs_on_the_same_image():
    """One runner label across the fleet, whatever it is.

    The workflows hand work to each other -- Capture uploads an artifact Rebuild
    reads -- so a split fleet means "passes in test, fails in production" with no
    reason to suspect the OS. This asserts they agree, not which one they agree
    on.
    """
    labels = {}
    for name in ("test.yml", "capture.yml", "discover.yml", "rebuild.yml"):
        found = re.findall(r"^\s*runs-on:\s*(\S+)", read(name), re.M)
        assert len(found) == 1, (name, found)
        labels[name] = found[0]
    assert len(set(labels.values())) == 1, labels
