"""The README is the only documentation shipped with the repository, so the
sections a reader needs are asserted here rather than left to review.

Task 1 wrote a short scaffolding README; these tests describe the document that
replaces it, and they fail with AssertionError on content, not FileNotFoundError.
"""

from __future__ import annotations

from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"

NOTICE = (
    "Unofficial. Not affiliated with, endorsed by, or connected to Costco Wholesale"
    " Corporation. Prices are collected from Costco's public websites and may differ"
    " from the price at the pump."
)


def readme() -> str:
    return README.read_text(encoding="utf-8")


def flowed() -> str:
    """The README as one line, without blockquote markers, so that wrapped prose
    can be matched as a single sentence."""
    lines = [line.lstrip("> ") for line in readme().splitlines()]
    return " ".join(" ".join(lines).split())


def test_notice_is_an_important_callout_near_the_top():
    text = readme()
    assert "> [!IMPORTANT]" in text
    assert NOTICE in flowed()
    assert "## Using the dashboard" in text
    assert text.index("> [!IMPORTANT]") < text.index("## Using the dashboard")


def test_required_sections_are_present_in_order():
    text = readme()
    headings = [
        "# Costco Gas Prices",
        "## Using the dashboard",
        "## Getting the data",
        "## Data conventions",
        "## Repository layout",
        "## Running it",
        "## Sources",
        "## License",
    ]
    missing = [heading for heading in headings if heading not in text]
    assert missing == [], missing
    positions = [text.index(h) for h in headings]
    assert positions == sorted(positions)


def test_page_table_lists_every_dashboard_page():
    text = flowed()
    for page in ("| Map |", "| Compare |", "| Trends |", "| Station |", "| About |"):
        assert page in text, page
    assert "?station=<station_key>#station" in text


def test_release_layout_names_every_asset_at_both_grains():
    text = readme()
    for asset in (
        "costco-gas-all.parquet",
        "costco-gas-all.csv.gz",
        "costco-gas-all-captures.parquet",
        "costco-gas-latest.csv",
        "stations.csv",
        "fx.csv",
        "manifest.json",
        "costco-gas-YYYY-MM-DD.csv.gz",
        "capture-<capture_id>.tar.gz",
        "manifest-YYYY-MM.json",
        "costco-gas-YYYY-MM.parquet",
        "costco-gas-YYYY-MM-captures.parquet",
        "costco-gas-YYYY.parquet",
        "costco-gas-YYYY-captures.parquet",
        "manifest-YYYY.json",
    ):
        assert asset in text, asset
    assert "Capture grain" in text
    assert "Daily grain" in text
    assert "`n_captures`, `price_min` and `price_max`" in flowed()


def test_schema_table_lists_every_row_column():
    text = readme()
    for column in (
        "`capture_id`",
        "`capture_date`",
        "`captured_at_utc`",
        "`local_date`",
        "`country`",
        "`station_key`",
        "`source_station_id`",
        "`source`",
        "`name`",
        "`name_local`",
        "`address`",
        "`city`",
        "`region`",
        "`postcode`",
        "`lat`, `lon`",
        "`timezone`",
        "`grade_raw`",
        "`grade`",
        "`price_raw`",
        "`price`",
        "`price_unit`",
        "`currency`",
        "`price_local_per_litre`",
        "`fx_usd_per_unit`",
        "`fx_rate_date`",
        "`fx_source`",
        "`fx_fetched_at_utc`",
        "`price_usd_per_litre`",
        "`price_usd_per_gallon`",
    ):
        assert column in text, column


def test_data_conventions_cover_the_five_required_topics():
    text = flowed()
    assert "**Units.**" in text
    assert "3.785411784" in text
    assert "**Grades.**" in text
    assert "`spec_source` is `source` when Costco itself states the specification" in text
    assert "**Exchange rates.**" in text
    assert "carried-forward:<original source>" in text
    assert "**Placeholders and pre-opening prices.**" in text
    assert "**UTC days and local dates.**" in text


def test_running_it_matches_the_supported_commands():
    text = readme()
    assert "uv run costco-gas capture --out out" in text
    assert "COSTCO_GAS_STORE=local:./releases uv run costco-gas publish out/capture" in text
    assert "uv run costco-gas site-data --current state/current --out site/data" in text
    assert "quarto preview site/index.qmd" in text
    assert "Publishing to the GitHub store from a local machine is unsupported" in flowed()


def test_sources_and_basemap_attribution():
    text = flowed()
    for host in (
        "ecom-api.costco.com",
        "www.costco.com/AjaxGetGasPricesService",
        "www.costco.ca/AjaxWarehouseBrowseLookupView",
        "www.costco.com.mx/rest/v2/mexico/stores",
        "www.costco.co.uk/rest/v2/uk/stores",
        "www.costco.com.au/rest/v2/australia/stores",
        "www.costco.co.jp/rest/v2/japan/stores",
        "www.costco.com.tw/rest/v2/taiwan/stores",
        "api.frankfurter.dev/v2/rates",
    ):
        assert host in text, host
    assert "© OpenStreetMap contributors, © CARTO" in text
    assert "https://carto.com/attributions" in text
    assert "The code is MIT licensed." in text
