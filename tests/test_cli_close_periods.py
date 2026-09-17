"""The close-periods subcommand: issue and exit-code rules (§8.6)."""

from __future__ import annotations

from typing import ClassVar

import pytest

from club_gas import cli


class FakeIssues:
    instances: ClassVar[list[FakeIssues]] = []

    def __init__(self, repo=None, token=None):
        self.opened: list[str] = []
        self.closed: list[str] = []
        FakeIssues.instances.append(self)

    def ensure_open(self, title, body, labels=None):
        self.opened.append(title)

    def close(self, title, comment):
        self.closed.append(title)


@pytest.fixture(autouse=True)
def _wired(monkeypatch, tmp_path):
    FakeIssues.instances = []
    monkeypatch.setenv("CLUB_GAS_STORE", f"local:{tmp_path / 'releases'}")
    monkeypatch.setattr(cli, "Issues", FakeIssues)
    monkeypatch.setattr(cli, "load_config", lambda root: object())


def test_close_periods_exits_zero_and_closes_the_failing_issue(monkeypatch, capsys):
    seen = {}

    def fake_close(store, cfg, *, now, rebuild_current=False, issues=None):
        seen["rebuild_current"] = rebuild_current
        return cli.CloseResult(["2026-08"], [], [], True)

    monkeypatch.setattr(cli, "close_periods", fake_close)

    code = cli.main(["close-periods", "--rebuild-current"])

    assert code == 0
    assert seen["rebuild_current"] is True
    assert FakeIssues.instances[0].closed == ["Period close failing"]
    assert "2026-08" in capsys.readouterr().out


def test_close_periods_opens_the_failing_issue_and_exits_one(monkeypatch):
    def boom(store, cfg, *, now, rebuild_current=False, issues=None):
        raise RuntimeError("release write refused")

    monkeypatch.setattr(cli, "close_periods", boom)

    code = cli.main(["close-periods"])

    assert code == 1
    assert FakeIssues.instances[0].opened == ["Period close failing"]
