"""The package imports and exports the two names every later module relies on."""

import costco_gas
from costco_gas import COUNTRIES


def test_the_package_lists_the_seven_countries_in_capture_order():
    assert COUNTRIES == ("US", "CA", "MX", "GB", "AU", "JP", "TW")
    assert costco_gas.COUNTRIES is COUNTRIES


def test_the_package_reports_the_installed_version():
    assert costco_gas.__version__ == "0.1.0"
