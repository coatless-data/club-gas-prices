from types import SimpleNamespace

from club_gas import cli
from club_gas.discover import DiscoverResult


def test_discover_subcommand_wires_the_pieces(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("CLUB_GAS_STORE", f"local:{tmp_path / 'releases'}")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    seen = {}

    # load_config returns the real Config; only .http is read here, to build the
    # HTTP client.
    config = SimpleNamespace(http="HTTP-CONFIG")
    monkeypatch.setattr(cli, "load_config", lambda root: config)
    monkeypatch.setattr(cli, "Client", lambda http_config: f"CLIENT({http_config})")

    def fake_discover(store, client, cfg, issues, *, now, watch=False):
        seen.update(store=store, client=client, cfg=cfg, issues=issues, now=now, watch=watch)
        return DiscoverResult(candidates=[{"warehouse_id": "1364", "prices": {"regular": "3.999"}}])

    monkeypatch.setattr(cli, "discover", fake_discover)

    assert cli.main(["discover"]) == 0
    # A run on a maintainer's machine leaves their config/watch.csv alone.
    assert seen["watch"] is False
    assert seen["cfg"] is config
    assert seen["client"] == "CLIENT(HTTP-CONFIG)"
    assert seen["issues"].dry is True
    assert seen["now"].tzinfo is not None
    assert '"candidates": 1' in capsys.readouterr().out


def test_discover_appends_to_watch_csv_only_when_asked(tmp_path, monkeypatch):
    monkeypatch.setenv("CLUB_GAS_STORE", f"local:{tmp_path / 'releases'}")
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    monkeypatch.setattr(cli, "load_config", lambda root: SimpleNamespace(http="HTTP-CONFIG"))
    monkeypatch.setattr(cli, "Client", lambda http_config: "CLIENT")
    seen = {}

    def fake_discover(store, client, cfg, issues, *, now, watch=False):
        seen["watch"] = watch
        return DiscoverResult(candidates=[])

    monkeypatch.setattr(cli, "discover", fake_discover)
    assert cli.main(["discover", "--watch"]) == 0
    assert seen["watch"] is True
