"""The package imports and exports the two names every later module relies on."""

import tomllib
from pathlib import Path

import club_gas
from club_gas import COUNTRIES

PYPROJECT = Path(__file__).resolve().parents[1] / "pyproject.toml"


def test_the_package_lists_the_seven_countries_in_capture_order():
    assert COUNTRIES == ("US", "CA", "MX", "GB", "AU", "JP", "TW")
    assert club_gas.COUNTRIES is COUNTRIES


def test_the_package_reports_the_installed_version():
    assert club_gas.__version__ == "0.1.0"


def test_the_package_describes_every_chain_it_collects():
    """It said "Collect Costco fuel prices" for a while after Sam's Club became a
    feed. The docstring and pyproject's description are the same sentence, and
    neither names one chain."""
    project = tomllib.loads(PYPROJECT.read_text(encoding="utf-8"))["project"]
    assert club_gas.__doc__.strip() == project["description"]
    assert "warehouse-club" in project["description"]
    assert "Costco" not in project["description"]
