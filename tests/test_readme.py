"""The README is the only documentation shipped with the repository, so the
sections a reader needs are asserted here rather than left to review.

Task 1 wrote a short scaffolding README; these tests describe the document that
replaces it, and they fail with AssertionError on content, not FileNotFoundError.
"""

from __future__ import annotations

from pathlib import Path

README = Path(__file__).resolve().parents[1] / "README.md"


def notice_clauses() -> list[str]:
    """Built from config/site.toml, so the README cannot become a fourth copy.

    The README flows the clauses into one sentence rather than repeating the
    exact joined string, so each clause is checked on its own.
    """
    import tomllib

    site = tomllib.loads((README.parent / "config" / "site.toml").read_text(encoding="utf-8"))[
        "site"
    ]
    return [site["notice_prefix"], *site["notices"].values()]


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
    text_flowed = flowed()
    for clause in notice_clauses():
        # Trailing punctuation differs where clauses are joined with "nor".
        assert clause.rstrip(".") in text_flowed, clause
    assert "## The dashboard" in text
    assert text.index("> [!IMPORTANT]") < text.index("## The dashboard")


def test_required_sections_are_present_in_order():
    text = readme()
    headings = [
        "# Club Gas Prices",
        "## The dashboard",
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


def test_the_dashboard_section_names_the_five_files_it_reads():
    text = flowed()
    for asset in (
        "site-meta.json",
        "site-latest.json",
        "site-stations.json",
        "site-summary-daily.parquet",
        "site-history.parquet",
    ):
        assert asset in text, asset


def test_release_layout_names_every_asset_at_both_grains():
    text = readme()
    for asset in (
        "club-gas-all.parquet",
        "club-gas-all.csv.gz",
        "club-gas-all-captures.parquet",
        "club-gas-latest.csv",
        "stations.csv",
        "fx.csv",
        "manifest.json",
        "club-gas-YYYY-MM-DD.csv.gz",
        "capture-<capture_id>.tar.gz",
        "manifest-YYYY-MM.json",
        "club-gas-YYYY-MM.parquet",
        "club-gas-YYYY-MM-captures.parquet",
        "club-gas-YYYY.parquet",
        "club-gas-YYYY-captures.parquet",
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
        "`brand`",
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


def test_station_key_documents_the_chain_segment():
    """It read `{country}-{source_station_id}` for a while after it stopped being
    that, which is the kind of drift only an assertion catches."""
    assert "{country}-{brand}-{source_station_id}" in readme()


def test_sources_table_is_keyed_by_feed():
    text = readme()
    for feed in (
        "US-COSTCO",
        "CA-COSTCO",
        "MX-COSTCO",
        "GB-COSTCO",
        "AU-COSTCO",
        "JP-COSTCO",
        "TW-COSTCO",
        "US-SAMS",
    ):
        assert f"| {feed} |" in text, feed


def test_data_conventions_cover_the_five_required_topics():
    text = flowed()
    assert "**Units.**" in text
    assert "3.785411784" in text
    assert "**Grades.**" in text
    assert "`spec_source` is `source` when the retailer itself states the specification" in text
    assert "**Exchange rates.**" in text
    assert "carried-forward:<original source>" in text
    assert "**Placeholders and pre-opening prices.**" in text
    assert "**UTC days and local dates.**" in text


def test_running_it_matches_the_supported_commands():
    text = readme()
    assert "uv run club-gas capture --out out" in text
    assert "CLUB_GAS_STORE=local:./releases uv run club-gas publish out/capture" in text
    assert "uv run club-gas site-data --current state/current --out site/data" in text
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


def test_station_status_names_every_value_publish_writes():
    """It read `active` or `missing` for a while after publish began writing
    `closed` for a station gone longer than `closed_after_days`."""
    from datetime import UTC, datetime, timedelta

    from club_gas.publish import station_status

    now = datetime(2026, 9, 18, tzinfo=UTC)
    written = {
        "active",
        station_status(now - timedelta(days=1), now, 45),
        station_status(now - timedelta(days=46), now, 45),
    }
    paragraph = next(p for p in readme().split("\n\n") if p.startswith("`stations.csv` holds"))
    for value in sorted(written):
        assert f"`{value}`" in paragraph, value


def test_no_link_points_into_the_gitignored_docs_directory():
    """docs/ is local working notes and never published, so a path into it is a
    dead link for every reader on GitHub."""
    import re

    assert re.findall(r"(?<![\w./-])docs/\S*", readme()) == []


def test_the_contact_address_is_spelled_out_for_people_not_harvesters():
    """The whole address is never written into the repository (test_contact.py
    holds that line), so a reader gets it with the @ and the dot in words."""
    assert "support [at] caffeinatedmath [dot] com" in flowed()
