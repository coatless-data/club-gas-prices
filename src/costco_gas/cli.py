"""costco-gas command line (spec 5.4, 12.2)."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import UTC, datetime
from pathlib import Path

from .capture import CaptureResult, run_capture
from .config import load_config
from .publish import publish
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

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
