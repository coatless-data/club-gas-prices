"""Sam's Club from capture to the dashboard's files, with no network.

Every other Sam's test stops at one stage: the source against a recorded body,
the budget inside `run_capture`, the station key inside `normalize`. None of
them follows a Sam's row through publish and into the files the site reads,
which is the path the first real Sam's capture would take. This one does, in
the same capture as Costco's US feed, because two chains in one country is the
case each stage has to keep apart.

The roster is the recorded `clubfinder/list` answer. Every price answer is the
recorded Tempo record for club 6376, relabelled for the club asked about.
"""

from __future__ import annotations

import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import httpx
import polars as pl
import pytest

from club_gas.capture import run_capture
from club_gas.config import load_config
from club_gas.http import Client
from club_gas.publish import publish
from club_gas.sitedata import build_site_data
from club_gas.store import LocalReleaseStore

REPO_ROOT = Path(__file__).resolve().parents[1]
FIXTURES = Path(__file__).resolve().parent / "fixtures"

ROSTER = (FIXTURES / "sams_clubfinder.json").read_bytes()
TEMPO = (FIXTURES / "sams_fuel_6376.json").read_bytes()
FX_BODY = (FIXTURES / "fx" / "frankfurter_v2.json").read_bytes()
ECOM_BODY = (FIXTURES / "us_ecom_warehouses.json").read_bytes()
COSTCO_PRICES = json.loads((FIXTURES / "us" / "gasprices_batch.json").read_bytes())

NOW = datetime(2026, 9, 15, 18, 17, 40, tzinfo=UTC)
CAPTURE_ID = "2026-09-15T1817Z"
LITRES_PER_GALLON = 3.785411784

# What each fuel club in the roster answers. 6376 is the recorded record as it
# came, whose MIDGRAD of 2.979 sits below its own UNLEAD; 8248 carries the move
# measured on 2026-09-17; the other two are priced in the same range.
CLUB_PRICES = {
    "6376": None,
    "8299": {"UNLEAD": 3.549, "PREMIUM": 4.149},
    "8248": {"UNLEAD": 3.749, "PREMIUM": 4.449, "DIESEL": 5.699},
    "4857": {"UNLEAD": 3.799, "MID CLR": 4.379, "PREMIUM": 4.299, "PREM CLR": 4.659},
}


def tempo_for(club_id: str) -> bytes:
    payload = json.loads(TEMPO)
    block = payload["data"]["contentLayout"]["modules"][0]["configs"]["storeFuelPrices"]
    block["id"] = f"CPF_FUELPRICE_PROD_{club_id}"
    prices = CLUB_PRICES[club_id]
    if prices is not None:
        block["prices"] = [{"name": k, "price": v, "type": "fuel"} for k, v in prices.items()]
    return json.dumps(payload).encode()


def transport() -> httpx.MockTransport:
    """FX, the warehouse locator, Costco's price batches, and Sam's two endpoints."""
    as_json = {"Content-Type": "application/json"}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "www.samsclub.com":
            if "clubfinder" in request.url.path:
                return httpx.Response(200, content=ROSTER, headers=as_json)
            club_id = json.loads(request.url.params["variables"])["nodeId"]
            return httpx.Response(200, content=tempo_for(club_id), headers=as_json)
        if host == "www.costco.com":
            # The price service answers only for the ids it was asked about.
            asked = request.url.params["warehouseid"].split("_")
            batch = {k: v for k, v in COSTCO_PRICES.items() if k in asked}
            return httpx.Response(200, content=json.dumps(batch).encode(), headers=as_json)
        if host == "api.frankfurter.dev":
            return httpx.Response(200, content=FX_BODY, headers=as_json)
        if host == "ecom-api.costco.com":
            return httpx.Response(200, content=ECOM_BODY, headers=as_json)
        return httpx.Response(404, content=b"{}")

    return httpx.MockTransport(handler)


@pytest.fixture()
def site(tmp_path: Path, monkeypatch) -> Path:
    """Capture the United States with Sam's switched on, publish, build the site."""
    shutil.copytree(REPO_ROOT / "config", tmp_path / "config")
    feeds = tmp_path / "config" / "feeds.toml"
    text = feeds.read_text(encoding="utf-8")
    assert "enabled = false" in text
    # On here and nowhere else: the committed config keeps the feed off.
    feeds.write_text(text.replace("enabled = false", "enabled = true"), encoding="utf-8")
    (tmp_path / "status").mkdir()
    monkeypatch.chdir(tmp_path)
    for name in ("GITHUB_OUTPUT", "GITHUB_REPOSITORY", "GITHUB_TOKEN", "GITHUB_RUN_ID"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("CARTO_BASEMAP_KEY", raising=False)

    cfg = load_config(tmp_path)
    assert cfg.feeds["US-SAMS"].enabled
    store = LocalReleaseStore(tmp_path / "releases")
    captured = run_capture(
        cfg,
        store,
        tmp_path / "out",
        countries=["US"],
        force_fallback=set(),
        now=NOW,
        client=Client(cfg.http, transport=transport()),
    )
    assert {fid: block["status"] for fid, block in captured.status["feeds"].items()} == {
        "US-COSTCO": "degraded",
        "US-SAMS": "degraded",
        "CA-COSTCO": "skipped",
        "MX-COSTCO": "skipped",
        "GB-COSTCO": "skipped",
        "AU-COSTCO": "skipped",
        "JP-COSTCO": "skipped",
        "TW-COSTCO": "skipped",
    }

    publish(store, captured.out, cfg, now=NOW)

    current = tmp_path / "current"
    current.mkdir()
    for asset in store.list_assets("current"):
        store.download("current", asset.name, current / asset.name)
    out = tmp_path / "site-data"
    build_site_data(current, out, cfg, now=NOW)
    return out


def test_sams_rows_are_keyed_by_chain_in_every_site_file(site: Path):
    sams_keys = {"US-SAMS-6376", "US-SAMS-8299", "US-SAMS-8248", "US-SAMS-4857"}

    history = pl.read_parquet(site / "history.parquet")
    sams = history.filter(pl.col("brand") == "SAMS")
    assert set(sams["station_key"].to_list()) == sams_keys
    costco = history.filter(pl.col("brand") == "COSTCO")["station_key"].to_list()
    assert costco and all(key.startswith("US-COSTCO-") for key in costco)

    latest = {r["station_key"]: r for r in json.loads((site / "latest.json").read_text())}
    assert {k for k, r in latest.items() if r["brand"] == "SAMS"} == sams_keys
    addison = latest["US-SAMS-6376"]
    assert (addison["country"], addison["brand"], addison["region"]) == ("US", "SAMS", "TX")
    assert addison["grades"]["regular"]["grade_raw"] == "UNLEAD"
    assert addison["grades"]["regular"]["price"] == 3.699
    # The MIDGRAD below the club's own UNLEAD never reached a station.
    assert addison["other"] == {}
    assert set(latest["US-SAMS-4857"]["other"]) == {"MID CLR", "PREM CLR"}

    stations = {r["station_key"]: r for r in json.loads((site / "stations.json").read_text())}
    assert {k for k, r in stations.items() if r["brand"] == "SAMS"} == sams_keys
    assert stations["US-SAMS-6376"]["source_station_id"] == "6376"
    assert stations["US-SAMS-6376"]["status"] == "active"


def test_summary_daily_keeps_the_two_us_chains_apart(site: Path):
    summary = pl.read_parquet(site / "summary_daily.parquet")
    regular = summary.filter(
        pl.col("level") == "country", pl.col("country") == "US", pl.col("grade") == "regular"
    ).sort("brand")

    assert regular["brand"].to_list() == ["COSTCO", "SAMS"]
    sams = regular.row(1, named=True)
    assert sams["n_stations"] == 4
    # Sam's four clubs and nothing else: 3.549, 3.699, 3.749 and 3.799 a gallon.
    assert sams["median_local_per_litre"] == pytest.approx(
        (3.699 + 3.749) / 2 / LITRES_PER_GALLON, abs=1e-4
    )
    texas = summary.filter(
        pl.col("level") == "region", pl.col("region") == "TX", pl.col("grade") == "regular"
    )
    # Texas has stations of both chains, and still one row for each.
    assert dict(texas.select("brand", "n_stations").iter_rows()) == {"COSTCO": 1, "SAMS": 3}


def test_meta_carries_the_sams_feed_and_names_the_chain(site: Path):
    meta = json.loads((site / "meta.json").read_text(encoding="utf-8"))

    # Four clubs against a floor of 460 is a short sweep, which is degraded.
    assert meta["feeds"]["US-SAMS"] == {
        "country": "US",
        "brand": "SAMS",
        "status": "degraded",
        "last_success_capture_id": CAPTURE_ID,
    }
    assert set(meta["feeds"]) == {"US-COSTCO", "US-SAMS"}
    assert meta["countries"]["US"]["last_success_capture_id"] == CAPTURE_ID
    # Sam's is in the data now, so the About page lists its grades and the
    # notice names it.
    assert {row["grade_raw"] for row in meta["grades"] if row["brand"] == "SAMS"} == {
        "UNLEAD",
        "PREMIUM",
        "DIESEL",
        "MIDGRAD",
        "MID CLR",
        "PREM CLR",
    }
    assert any("Sam's" in line for line in meta["notice"])
