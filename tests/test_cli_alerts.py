import json
from pathlib import Path

from club_gas.cli import main


def _status(tmp_path: Path) -> Path:
    document = {
        "schema_version": 1,
        "capture_id": "2026-09-15T1817Z",
        "run_id": 123,
        "run_url": "https://github.com/coatless-data/club-gas-prices/actions/runs/123",
        "ecom_api": {"attempted": True, "http_status": 200},
        "publish": {"outcome": None, "consecutive_failures": 0, "unpublished": []},
        "close": {},
        "countries": {
            "US": {"status": "ok", "warnings": [], "errors": [], "consecutive_failures": 0}
        },
    }
    path = tmp_path / "status.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def test_alerts_subcommand_updates_the_status_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLUB_GAS_STORE", f"local:{tmp_path / 'releases'}")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    path = _status(tmp_path)

    code = main(
        [
            "alerts",
            str(path),
            "--publish-outcome",
            "failure",
            "--close-outcome",
            "failure",
        ]
    )

    assert code == 0
    updated = json.loads(path.read_text(encoding="utf-8"))
    assert updated["publish"]["outcome"] == "failure"
    assert updated["publish"]["consecutive_failures"] == 1
    assert updated["close"] == {"outcome": "failure"}
    # No token, so every issue action is printed rather than sent.
    printed = capsys.readouterr().out
    assert "[issues] ensure_open: Publish failing" in printed
    assert "[issues] ensure_open: Period close failing" in printed
