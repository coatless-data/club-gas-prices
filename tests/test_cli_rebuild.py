"""The rebuild subcommand's argument mapping (§8.7)."""

from __future__ import annotations

import pytest

from club_gas import cli


@pytest.fixture(autouse=True)
def _wired(monkeypatch, tmp_path):
    monkeypatch.setenv("CLUB_GAS_STORE", f"local:{tmp_path / 'releases'}")
    monkeypatch.setattr(cli, "load_config", lambda root: object())


@pytest.mark.parametrize(
    ("argv", "scope", "value"),
    [
        (["rebuild", "--month", "2026-09"], "month", "2026-09"),
        (["rebuild", "--year", "2026"], "year", "2026"),
        (["rebuild", "--all"], "all", None),
    ],
)
def test_rebuild_maps_each_flag_to_a_scope(monkeypatch, capsys, argv, scope, value):
    seen = {}

    def fake_rebuild(store, cfg, *, scope, value, now):
        seen["scope"] = scope
        seen["value"] = value
        return cli.RebuildResult(["2026-09"], 4, False)

    monkeypatch.setattr(cli, "rebuild", fake_rebuild)

    code = cli.main(argv)

    assert code == 0
    assert seen == {"scope": scope, "value": value}
    assert "2026-09" in capsys.readouterr().out


def test_rebuild_requires_exactly_one_scope_flag():
    with pytest.raises(SystemExit):
        cli.main(["rebuild"])
