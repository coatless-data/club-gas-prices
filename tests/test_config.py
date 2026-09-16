"""Loading and validating the files in config/."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from costco_gas.config import ConfigError, load_config

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def cfg():
    return load_config(REPO_ROOT)


@pytest.fixture
def config_copy(tmp_path: Path) -> Path:
    """A writable copy of config/ under tmp_path, for the validation tests."""
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    return tmp_path


def test_config_records_its_root(cfg):
    # rebuild re-reads and hashes the live interpretation config through this
    # path, so the root passed to load_config has to survive on the object.
    assert isinstance(cfg.root, Path)
    assert cfg.root == REPO_ROOT


def test_http_policy(cfg):
    http = cfg.http
    # A plain desktop Chrome UA. Appending a project token to it made
    # www.costco.com reset the connection on 2026-09-15, so nothing may be
    # appended (research responses/id_price_ua_token.meta.json).
    assert http.costco_user_agent.startswith("Mozilla/5.0 ")
    assert http.costco_user_agent.endswith("Safari/537.36")
    assert "costco-gas-prices" not in http.costco_user_agent
    assert http.project_user_agent.startswith("costco-gas-prices/")
    assert http.x_project == "https://github.com/coatless-dashboard/costco-gas-prices"
    assert "costco.com" in http.costco_host_suffixes
    assert "costco.co.jp" in http.costco_host_suffixes

    assert http.profiles["default"].connect == 10.0
    assert http.profiles["default"].read == 20.0
    assert http.profiles["default"].total == 60.0
    assert http.profiles["bulk"].read == 150.0
    assert http.profiles["bulk"].total == 150.0

    assert http.max_attempts == 3
    assert http.backoff_seconds == (1.0, 2.0)
    assert http.backoff_jitter == 0.5
    assert http.min_interval_seconds == 1.0
    assert http.block_signals_before_abandon == 2
    assert http.budgets["fx"] == 90.0
    assert http.budgets["ecom-api"] == 90.0
    assert http.budgets["country"] == 480.0
    assert http.budgets["capture"] == 720.0


def test_countries_are_the_seven_v1_countries(cfg):
    assert sorted(cfg.countries) == ["AU", "CA", "GB", "JP", "MX", "TW", "US"]
    for code, country in cfg.countries.items():
        assert country.code == code


def test_us_fetch_parameters(cfg):
    us = cfg.countries["US"]
    assert us.url == "https://www.costco.com/AjaxGetGasPricesService"
    # The price service takes ?warehouseid=<id>_<id>... built per batch, so it
    # has no static query table. params never holds anything but query keys.
    assert us.params == {}
    # Only the first 10 ids of an AjaxGetGasPricesService request are processed.
    assert us.batch_size == 10
    assert us.seen_within_days == 30
    assert us.ecom_url == (
        "https://ecom-api.costco.com/core/warehouse-locator/v1/warehouses.json"
    )
    assert us.ecom_client_identifier == "7c71124c-7bf1-44db-bc9d-498584cd66e5"
    assert us.ecom_params["latitude"] == "0"
    assert us.ecom_params["limit"] == "5000"
    # The costco.ca US lookup took 30.6 s on 2026-09-15, so sources/us.py asks
    # the client for the "bulk" timeout profile when it requests this URL.
    assert us.fallback_url == "https://www.costco.ca/AjaxWarehouseBrowseLookupView"
    assert us.fallback_params["countryCode"] == "US"
    assert us.fallback_params["hasGas"] == "true"


def test_ca_fetch_parameters(cfg):
    ca = cfg.countries["CA"]
    assert ca.url == "https://www.costco.ca/AjaxWarehouseBrowseLookupView"
    assert ca.params == {
        "hasGas": "true",
        "populateWarehouseDetails": "true",
        "countryCode": "CA",
    }
    assert ca.batch_size == 10
    assert ca.seen_within_days == 30
    assert ca.fallback_url == "https://www.costco.ca/AjaxGetGasPricesService"
    assert ca.fallback_params == {}
    # ecom-api is fetched once per capture from the US entry.
    assert ca.ecom_url is None
    assert ca.ecom_client_identifier is None


def test_occ_fetch_parameters(cfg):
    gb = cfg.countries["GB"]
    assert gb.url == "https://www.costco.co.uk/rest/v2/uk/stores"
    # params is urlencoded whole, so it is flat and every value is a string.
    assert gb.params == {
        "fields": "FULL",
        "pageSize": "100",
        "lang": "en_GB",
        "curr": "GBP",
    }
    mx = cfg.countries["MX"]
    assert mx.url == "https://www.costco.com.mx/rest/v2/mexico/stores"
    assert mx.params["returnAllStores"] == "true"
    assert mx.params["radius"] == "3000000"
    assert mx.batch_size is None
    assert mx.fallback_url is None
    assert mx.fallback_params == {}


def test_units_bounds_floors_and_staleness(cfg):
    us = cfg.countries["US"]
    assert us.price_unit == "USD/gal"
    # Puerto Rico prices per litre: #335 regular was 1.097 on 2026-09-15.
    assert us.unit_overrides == {"PR": "USD/L"}
    assert us.unit_for_region(None) == "USD/gal"
    assert us.unit_for_region("CA") == "USD/gal"
    assert us.unit_for_region("PR") == "USD/L"
    assert (us.bounds["USD/gal"].min, us.bounds["USD/gal"].max) == (2.0, 11.0)
    assert (us.bounds["USD/L"].min, us.bounds["USD/L"].max) == (0.5, 2.5)
    assert us.bounds["USD/gal"].for_grade("regular") == (2.0, 11.0)
    assert us.bounds["USD/gal"].for_grade("clear") == (2.0, 11.0)
    assert us.floor == 585
    assert us.stale_after_days == 3

    ca = cfg.countries["CA"]
    assert ca.price_unit == "CAD/L"
    # A warehouse with no pump publishes .959 CAD/L; the minimum rejects it.
    assert (ca.bounds["CAD/L"].min, ca.bounds["CAD/L"].max) == (1.0, 3.5)
    assert ca.floor == 78

    jp = cfg.countries["JP"]
    assert jp.price_unit == "JPY/L"
    assert (jp.bounds["JPY/L"].min, jp.bounds["JPY/L"].max) == (90.0, 250.0)
    # Kerosene is heating fuel and is much cheaper than road fuel.
    assert jp.bounds["JPY/L"].for_grade("Kerosene") == (60.0, 250.0)
    assert jp.stale_after_days == 10

    assert cfg.countries["GB"].price_unit == "GBp/L"
    assert (cfg.countries["GB"].bounds["GBp/L"].min,
            cfg.countries["GB"].bounds["GBp/L"].max) == (100.0, 250.0)
    assert cfg.countries["TW"].stale_after_days == 14
    assert [cfg.countries[c].floor for c in ("MX", "GB", "AU", "JP", "TW")] == [
        18, 21, 14, 26, 3
    ]


def test_timezone_tables(cfg):
    us = cfg.countries["US"]
    assert us.timezones["WA"] == "America/Los_Angeles"
    assert us.timezones["TX"] == "America/Chicago"
    assert us.timezones["PR"] == "America/Puerto_Rico"
    assert us.timezone_for_region("HI") == "Pacific/Honolulu"
    # US has no "*" default: a station whose region is unknown has no timezone
    # and is dropped rather than guessed.
    assert us.timezone_for_region(None) is None
    assert us.timezone_for_region("ZZ") is None

    assert cfg.countries["CA"].timezones["BC"] == "America/Vancouver"
    mx = cfg.countries["MX"]
    assert mx.timezones["*"] == "America/Mexico_City"
    assert mx.timezone_for_region("BCN") == "America/Tijuana"
    assert mx.timezone_for_region("SON") == "America/Hermosillo"
    assert mx.timezone_for_region("ROO") == "America/Cancun"
    assert mx.timezone_for_region("JAL") == "America/Mexico_City"  # the default
    au = cfg.countries["AU"]
    assert au.timezone_for_region("ACT") == "Australia/Sydney"
    assert au.timezone_for_region("WA") == "Australia/Perth"
    # GB, JP and TW are single-zone: the "*" entry is the whole table, and it
    # is only ever reached through timezone_for_region.
    assert cfg.countries["GB"].timezone_for_region(None) == "Europe/London"
    assert cfg.countries["GB"].timezone_for_region("Berkshire") == "Europe/London"
    assert cfg.countries["JP"].timezone_for_region(None) == "Asia/Tokyo"
    assert cfg.countries["TW"].timezone_for_region(None) == "Asia/Taipei"


def test_grade_table(cfg):
    grades = cfg.grades
    assert grades.map("US", "regular").grade == "regular"
    assert grades.map("US", "clear").grade == "other"
    assert grades.map("GB", "5301").grade == "regular"
    assert grades.map("GB", "5301").label == "Unleaded Petrol"
    assert grades.map("GB", "5303").grade == "diesel"
    assert grades.map("TW", "95").grade == "regular"
    assert grades.map("JP", "Kerosene").grade == "other"
    # Australia lists E10 in NSW and QLD and Unleaded 91 elsewhere; both are the
    # regular grade, and the higher priority wins if a station lists both.
    assert grades.map("AU", "Unleaded 91").grade == "regular"
    assert grades.map("AU", "Unleaded 91").priority == 2
    assert grades.map("AU", "E10").grade == "regular"
    assert grades.map("AU", "E10").priority == 1
    # An unknown label is not an error at load time: normalization maps it to
    # "other" and warns, so a new label never breaks a capture.
    assert grades.map("AU", "Unleaded 95") is None
    assert grades.map("XX", "regular") is None
    mx_regular = grades.map("MX", "Regular")
    assert mx_regular.spec_source == "reported"
    assert mx_regular.spec_source_url.startswith("https://")


def test_us_extra_ids_table(cfg):
    df = cfg.us_extra_ids
    assert df.columns == [
        "source_station_id", "name", "city", "region",
        "postcode", "lat", "lon", "timezone", "note",
    ]
    ids = df["source_station_id"].to_list()
    assert ids == ["1680", "1765", "1772"]
    # ecom-api does not list these ids at all, so the CSV is their only source
    # of coordinates and timezone: a null here means normalize drops them.
    assert df["name"].null_count() == 0
    assert df["lat"].null_count() == 0
    assert df["lon"].null_count() == 0
    assert df["timezone"].null_count() == 0
    camarillo = df.filter(df["source_station_id"] == "1680").row(0, named=True)
    assert camarillo["region"] == "CA"
    assert camarillo["lat"] == 34.2164
    assert camarillo["lon"] == -119.0376
    assert camarillo["timezone"] == "America/Los_Angeles"
    chandler = df.filter(df["source_station_id"] == "1765").row(0, named=True)
    assert chandler["region"] == "AZ"
    assert chandler["lat"] == 33.3062
    assert chandler["lon"] == -111.8413
    assert chandler["timezone"] == "America/Phoenix"
    mission_viejo = df.filter(df["source_station_id"] == "1772").row(0, named=True)
    assert mission_viejo["region"] == "CA"
    assert mission_viejo["lat"] == 33.6103
    assert mission_viejo["lon"] == -117.689
    assert mission_viejo["timezone"] == "America/Los_Angeles"
    # #1772's identity is reported, not confirmed, and the note says so.
    assert "unverified" in mission_viejo["note"]


def test_station_links_table_starts_empty(cfg):
    assert cfg.station_links.columns == [
        "old_station_key", "new_station_key", "effective_date", "note",
    ]
    assert cfg.station_links.height == 0


def test_site_config(cfg):
    site = cfg.site
    assert site.notice.startswith("Unofficial.")
    assert "Costco Wholesale Corporation" in site.notice
    assert site.basemap_key_env == "CARTO_BASEMAP_KEY"
    # sitedata copies one provider table into meta.json and adds "provider",
    # so these key names are exactly what the dashboard's Leaflet layer reads.
    assert set(site.basemaps) == {"carto", "osm"}
    for provider in site.basemaps.values():
        assert set(provider) == {
            "light_url", "dark_url", "subdomains",
            "max_zoom", "dark_filter", "attribution",
        }
    osm = site.basemaps["osm"]
    assert osm["light_url"] == "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    assert osm["dark_url"] == osm["light_url"]
    assert osm["subdomains"] == ""
    assert osm["max_zoom"] == 19
    # OSM has a single style, so dark mode inverts the tile pane instead.
    assert osm["dark_filter"] is True
    carto = site.basemaps["carto"]
    assert carto["light_url"].endswith("/light_all/{z}/{x}/{y}{r}.png?key={key}")
    assert carto["dark_url"].endswith("/dark_all/{z}/{x}/{y}{r}.png?key={key}")
    assert carto["subdomains"] == "abcd"
    assert carto["max_zoom"] == 20
    assert carto["dark_filter"] is False


def test_missing_config_directory(tmp_path: Path):
    with pytest.raises(ConfigError, match="no config directory"):
        load_config(tmp_path)


def test_query_parameters_must_be_strings(config_copy: Path):
    # A number here would be urlencoded as "100" anyway, but allowing it opens
    # the door to non-query keys drifting back into params.
    path = config_copy / "config" / "countries.toml"
    text = path.read_text(encoding="utf-8")
    broken = text.replace(
        'pageSize = "100"\nlang = "en_GB"', 'pageSize = 100\nlang = "en_GB"'
    )
    assert broken != text
    path.write_text(broken, encoding="utf-8")
    with pytest.raises(ConfigError, match="query parameter 'pageSize'"):
        load_config(config_copy)
