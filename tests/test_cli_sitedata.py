from pathlib import Path

from costco_gas import cli


def test_site_data_subcommand_passes_the_directories(tmp_path, monkeypatch):
    seen = {}
    monkeypatch.setattr(cli, "load_config", lambda root: "CONFIG")

    def fake_build(current_dir, out_dir, cfg, *, now):
        seen.update(current=Path(current_dir), out=Path(out_dir), cfg=cfg, now=now)

    monkeypatch.setattr(cli, "build_site_data", fake_build)

    code = cli.main(
        ["site-data", "--current", str(tmp_path / "current"), "--out", str(tmp_path / "data")]
    )

    assert code == 0
    assert seen["current"] == tmp_path / "current"
    assert seen["out"] == tmp_path / "data"
    assert seen["cfg"] == "CONFIG"
    assert seen["now"].tzinfo is not None
