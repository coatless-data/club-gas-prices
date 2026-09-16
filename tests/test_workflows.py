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


def test_render_conventions():
    assert_conventions("render.yml")
    text = read("render.yml")
    assert "concurrency:\n  group: render\n  cancel-in-progress: true\n" in text
    assert "permissions:\n  contents: read\n  pages: write\n  id-token: write\n" in text
    # Render only reads releases, so it must never claim the writer flag.
    assert "COSTCO_GAS_WRITER" not in text


def test_render_triggers_and_job_condition():
    text = read("render.yml")
    assert text.startswith("name: Render\n")
    assert "workflows: [Capture, Rebuild]" in text
    assert (
        "if: ${{ github.event_name != 'workflow_run'"
        " || (github.event.workflow_run.conclusion == 'success'"
        " && !contains(github.event.workflow_run.display_title, 'dry run')) }}" in text
    )
    for path in (
        "      - site/**",
        "      - src/costco_gas/sitedata.py",
        "      - config/grades.csv",
        "      - config/site.toml",
        "      - pyproject.toml",
        "      - uv.lock",
        "      - .github/workflows/render.yml",
    ):
        assert path in text, path
    assert "status/**" not in text
    assert "    timeout-minutes: 30\n" in text
    assert "CARTO_BASEMAP_KEY: ${{ vars.CARTO_BASEMAP_KEY }}" in text
    assert "version: 1.10.18" in text


def test_render_never_reads_temporary_asset_names():
    # `.next-*` and `.old-*` are mid-write names that only read_resolved may
    # read, so Render asks for the five assets by their exact names.
    text = read("render.yml")
    for name in (
        "manifest.json",
        "costco-gas-all.parquet",
        "costco-gas-latest.csv",
        "stations.csv",
        "fx.csv",
    ):
        assert f"--pattern {name}" in text, name
    assert ".next-" not in text
    assert ".old-" not in text


def test_render_stages_no_source_files_and_smoke_tests():
    text = read("render.yml")
    assert "uv run costco-gas site-data --current state/current --out site/data" in text
    assert "quarto render site/index.qmd" in text
    assert "find _site \\( -name '*.qmd' -o -name '*.scss' \\) -print" in text
    assert "uv sync --locked --group smoke" in text
    assert "uv run python tests/smoke/smoke_site.py _site" in text
    assert "uses: actions/upload-pages-artifact@v3" in text
    assert "uses: actions/deploy-pages@v4" in text


def extract_verify_script(text: str) -> str:
    """The python program render.yml writes into $RUNNER_TEMP with a heredoc."""
    match = re.search(r"<<'PY'\n(.*?)\n\s*PY\n", text, re.DOTALL)
    assert match is not None, "render.yml has no embedded PY heredoc"
    return textwrap.dedent(match.group(1)) + "\n"


def write_current(directory: Path) -> None:
    payload = {
        "costco-gas-all.parquet": b"PAR1-stand-in-for-a-parquet-file",
        "costco-gas-latest.csv": b"station_key,price\nUS-1364,3.999\n",
        "stations.csv": b"station_key\nUS-1364\n",
        "fx.csv": b"capture_id,currency\n2026-09-15T1817Z,CAD\n",
    }
    assets = {}
    for name, body in payload.items():
        (directory / name).write_bytes(body)
        assets[name] = {"sha256": hashlib.sha256(body).hexdigest(), "size": len(body)}
    (directory / "manifest.json").write_text(
        json.dumps({"assets": assets}), encoding="utf-8"
    )


def run_verify(tmp_path: Path) -> subprocess.CompletedProcess[str]:
    script = tmp_path / "verify_current.py"
    script.write_text(extract_verify_script(read("render.yml")), encoding="utf-8")
    return subprocess.run(
        [sys.executable, str(script), str(tmp_path / "current")],
        capture_output=True,
        text=True,
        check=False,
    )


def test_render_verify_script_accepts_a_consistent_release(tmp_path):
    (tmp_path / "current").mkdir()
    write_current(tmp_path / "current")
    done = run_verify(tmp_path)
    assert done.returncode == 0, done.stderr
    assert "verified 4 assets" in done.stdout


def test_render_verify_script_rejects_a_torn_release(tmp_path):
    # A release read while replace_atomic is renaming assets can mix versions,
    # which is what the 5 retries in the workflow are there to ride out.
    (tmp_path / "current").mkdir()
    write_current(tmp_path / "current")
    (tmp_path / "current" / "fx.csv").write_bytes(
        b"capture_id,currency\n2026-09-15T1817Z,MXN\n"
    )
    done = run_verify(tmp_path)
    assert done.returncode != 0
    assert "fx.csv" in done.stderr


def test_render_verify_script_rejects_a_missing_asset(tmp_path):
    (tmp_path / "current").mkdir()
    write_current(tmp_path / "current")
    (tmp_path / "current" / "stations.csv").unlink()
    done = run_verify(tmp_path)
    assert done.returncode != 0
    assert "stations.csv is missing" in done.stderr


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
    assert "run: uv run costco-gas discover\n" in text
    # discover only reads releases, so it must never claim the writer flag.
    assert "COSTCO_GAS_WRITER" not in text
