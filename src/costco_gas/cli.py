"""costco-gas command line (spec 5.4, 12.2)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from .alerts import run_alerts
from .capture import CaptureResult, run_capture
from .config import load_config
from .discover import discover
from .http import Client
from .issues import Issues
from .publish import publish
from .rebuild import RebuildResult, rebuild
from .rollup import CloseResult, close_periods
from .sitedata import build_site_data
from .store import DEFAULT_STORE, open_store


def open_configured_store():
    """Pick the release store from COSTCO_GAS_STORE (spec 12.1)."""
    return open_store(os.environ.get("COSTCO_GAS_STORE") or DEFAULT_STORE)


def utc_now() -> datetime:
    return datetime.now(UTC).replace(microsecond=0)


def split_csv(value: str | None) -> list[str]:
    if not value:
        return []
    return [part.strip().upper() for part in value.split(",") if part.strip()]


def select_countries(value: str | None, cfg) -> list[str]:
    known = list(cfg.countries)
    wanted = split_csv(value)
    if not wanted or wanted == ["ALL"]:
        return known
    unknown = [code for code in wanted if code not in known]
    if unknown:
        raise SystemExit("unknown countries: " + ",".join(unknown))
    return [code for code in known if code in wanted]


def run_capture_result_for_test(out: Path) -> CaptureResult:
    """A stand-in CaptureResult; tests monkeypatch run_capture and call this."""
    return CaptureResult(
        status={}, all_failed=False, capture_id="2026-09-15T1817Z", out=Path(out) / "capture"
    )


def cmd_capture(args) -> int:
    cfg = load_config(Path("."))
    result = run_capture(
        cfg,
        open_configured_store(),
        Path(args.out),
        countries=select_countries(args.countries, cfg),
        force_fallback=set(split_csv(args.force_fallback)),
        now=utc_now(),
    )
    print(
        json.dumps(
            {
                "capture_id": result.capture_id,
                "all_failed": result.all_failed,
                "out": str(result.out),
            }
        )
    )
    # Exit 0 whenever status.json was written; the workflow gate uses all_failed.
    return 0


def cmd_publish(args) -> int:
    cfg = load_config(Path("."))
    result = publish(open_configured_store(), Path(args.dir), cfg, now=utc_now())
    print(
        json.dumps(
            {
                "capture_id": result.capture_id,
                "warnings": result.warnings,
                "assets_written": result.assets_written,
            }
        )
    )
    return 0


def cmd_close_periods(args) -> int:
    cfg = load_config(Path("."))
    store = open_configured_store()
    issues = Issues(os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN"))
    now = utc_now()
    try:
        result: CloseResult = close_periods(
            store, cfg, now=now, rebuild_current=args.rebuild_current, issues=issues
        )
    except Exception as exc:
        print(f"close-periods failed: {exc}", file=sys.stderr)
        try:
            issues.ensure_open(
                "Period close failing",
                f"`close-periods` raised at {now:%Y-%m-%dT%H%MZ}:\n\n```\n{exc}\n```",
                ["period-close"],
            )
        except Exception as issue_exc:
            print(f"::warning::issue call failed: {issue_exc}")
        return 1
    try:
        issues.close("Period close failing", f"close-periods exited 0 at {now:%Y-%m-%dT%H%MZ}.")
    except Exception as issue_exc:
        print(f"::warning::issue call failed: {issue_exc}")
    print(
        json.dumps(
            {
                "closed_months": result.closed_months,
                "closed_years": result.closed_years,
                "blocked_months": result.blocked_months,
                "rebuilt_current": result.rebuilt_current,
            }
        )
    )
    return 0


def cmd_rebuild(args) -> int:
    cfg = load_config(Path("."))
    store = open_configured_store()
    if args.month:
        scope, value = "month", args.month
    elif args.year:
        scope, value = "year", args.year
    else:
        scope, value = "all", None
    result: RebuildResult = rebuild(store, cfg, scope=scope, value=value, now=utc_now())
    print(
        json.dumps(
            {
                "months": result.months,
                "captures": result.captures,
                "resumed": result.resumed,
            }
        )
    )
    return 0


def cmd_alerts(args) -> int:
    run_alerts(
        Path(args.status),
        publish_outcome=args.publish_outcome,
        close_outcome=args.close_outcome,
        store=open_configured_store(),
        issues=Issues(os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")),
        now=utc_now(),
    )
    return 0


def cmd_discover(args) -> int:
    cfg = load_config(Path(args.root))
    result = discover(
        open_configured_store(),
        Client(cfg.http),
        cfg,
        Issues(os.environ.get("GITHUB_REPOSITORY"), os.environ.get("GITHUB_TOKEN")),
        now=utc_now(),
    )
    print(
        json.dumps({"candidates": len(result.candidates), "skipped_reason": result.skipped_reason})
    )
    return 0


def cmd_site_data(args) -> int:
    build_site_data(
        Path(args.current),
        Path(args.out),
        load_config(Path(args.root)),
        now=utc_now(),
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="costco-gas")
    sub = parser.add_subparsers(dest="command", required=True)

    capture = sub.add_parser("capture", help="collect prices into DIR")
    capture.add_argument("--out", required=True)
    capture.add_argument("--countries", default="all")
    capture.add_argument("--force-fallback", default="")
    capture.set_defaults(func=cmd_capture)

    publish_cmd = sub.add_parser("publish", help="publish a capture directory")
    publish_cmd.add_argument("dir", help="the capture directory, e.g. out/capture")
    publish_cmd.set_defaults(func=cmd_publish)

    close = sub.add_parser("close-periods", help="close finished months and years")
    close.add_argument(
        "--rebuild-current",
        action="store_true",
        help="rebuild the current release in full even if no month closed",
    )
    close.set_defaults(func=cmd_close_periods)

    alerts_cmd = sub.add_parser("alerts", help="open and close the capture issues (spec 10.2)")
    alerts_cmd.add_argument("status", help="path to the capture's status.json")
    alerts_cmd.add_argument(
        "--publish-outcome",
        default="skipped",
        help="the publish step's outcome: success, failure, skipped or cancelled",
    )
    alerts_cmd.add_argument(
        "--close-outcome",
        default="skipped",
        help="the close-periods step's outcome: success, failure, skipped or cancelled",
    )
    alerts_cmd.set_defaults(func=cmd_alerts)

    rebuild_parser = sub.add_parser(
        "rebuild", help="re-parse stored capture bundles and rewrite history"
    )
    scope_group = rebuild_parser.add_mutually_exclusive_group(required=True)
    scope_group.add_argument("--month", help="a single month, e.g. 2026-09")
    scope_group.add_argument("--year", help="every month of a year, e.g. 2026")
    scope_group.add_argument("--all", action="store_true", help="every month")
    rebuild_parser.set_defaults(func=cmd_rebuild)

    discover_cmd = sub.add_parser(
        "discover", help="sweep unpolled US warehouse ids for fuel prices (spec 10.4)"
    )
    discover_cmd.add_argument("--root", default=".", help="repository root holding config/")
    discover_cmd.set_defaults(func=cmd_discover)

    site_data_cmd = sub.add_parser(
        "site-data", help="build the dashboard's data files from a current release (spec 9.1)"
    )
    site_data_cmd.add_argument(
        "--current", required=True, help="directory holding the downloaded current assets"
    )
    site_data_cmd.add_argument("--out", required=True, help="output directory, normally site/data")
    site_data_cmd.add_argument("--root", default=".", help="repository root holding config/")
    site_data_cmd.set_defaults(func=cmd_site_data)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
