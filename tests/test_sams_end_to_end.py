"""Sam's Club from capture to the dashboard's files, with no network.

Every other Sam's test stops at one stage: the source against a recorded body,
the budget inside `run_capture`, the station key inside `normalize`. None of
them follows a Sam's row through publish and into the files the site reads,
which is the path the first real Sam's capture would take. This one does, in
the same capture as Costco's US feed, because two chains in one country is the
case each stage has to keep apart.

The roster is a locator sitemap listing the four fuel clubs and one non-fuel
club; each fuel club answers a fuel-centre page whose `__NEXT_DATA__` carries its
prices, and the non-fuel club 307s and is skipped.
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

FX_BODY = (FIXTURES / "fx" / "frankfurter_v2.json").read_bytes()
ECOM_BODY = (FIXTURES / "us_ecom_warehouses.json").read_bytes()
COSTCO_PRICES = json.loads((FIXTURES / "us" / "gasprices_batch.json").read_bytes())

NOW = datetime(2026, 9, 15, 18, 17, 40, tzinfo=UTC)
CAPTURE_ID = "2026-09-15T1817Z"
LITRES_PER_GALLON = 3.785411784

# Each fuel club's (state, city, prices). 6376's MIDGRAD of 2.979 sits below its
# own UNLEAD and is dropped; three clubs are in Texas, one in Florida.
SAMS_CLUBS = {
    "6376": (
        "TX",
        "Dallas",
        {"UNLEAD": 3.699, "PREMIUM": 4.399, "DIESEL": 5.699, "MIDGRAD": 2.979},
    ),
    "8299": ("TX", "Houston", {"UNLEAD": 3.549, "PREMIUM": 4.149}),
    "8248": ("TX", "Dallas", {"UNLEAD": 3.749, "PREMIUM": 4.449, "DIESEL": 5.699}),
    "4857": (
        "FL",
        "Orlando",
        {"UNLEAD": 3.799, "MID CLR": 4.379, "PREMIUM": 4.299, "PREM CLR": 4.659},
    ),
}
# In the sitemap but with no fuel centre: it 307s to /club/6225 and is skipped.
SAMS_NON_FUEL = ["6225"]
_TZ_BY_STATE = {"TX": "America/Chicago", "FL": "America/New_York"}


def sams_sitemap() -> bytes:
    ids = [*SAMS_CLUBS, *SAMS_NON_FUEL]
    locs = "".join(f"<loc>https://www.samsclub.com/club/{i}-city-xx</loc>" for i in ids)
    return f'<?xml version="1.0"?><urlset>{locs}</urlset>'.encode()


def sams_fuel_page(club_id: str) -> bytes:
    state, city, prices = SAMS_CLUBS[club_id]
    block = {
        "countryCode": "US",
        "id": f"CPF_FUELPRICE_PROD_{club_id}",
        "metadata": {"dateCreated": "2026-09-15T08:15:34.337Z", "createdBy": "CPF_STORAGE"},
        "prices": [{"name": k, "price": v, "type": "fuel"} for k, v in prices.items()],
    }
    next_data = {
        "props": {
            "pageProps": {
                "initialTempoData": {
                    "contentLayout": {
                        "modules": [
                            {
                                "configs": {
                                    "storeDetails": {
                                        "capabilities": [{"timeZone": _TZ_BY_STATE[state]}]
                                    }
                                }
                            },
                            {"configs": {"storeFuelPrices": block}},
                        ]
                    }
                },
                "initialNodeDetail": {
                    "data": {
                        "nodeDetail": {
                            "id": club_id,
                            "name": f"{city} Sam's Club",
                            "displayName": f"{city} Sam's Club",
                            "address": {
                                "addressLineOne": "1 Main St",
                                "city": city,
                                "state": state,
                                "postalCode": "75244",
                                "country": "US",
                            },
                            "geoPoint": {"latitude": 32.9, "longitude": -96.8},
                            "operationalHours": [
                                {"day": "Monday", "start": "06:00", "end": "22:00", "closed": False}
                            ],
                        }
                    }
                },
            }
        }
    }
    script = json.dumps(next_data)
    return (
        f"<!doctype html><html><body>"
        f'<script id="__NEXT_DATA__" type="application/json">{script}</script>'
        f"</body></html>"
    ).encode()


def transport() -> httpx.MockTransport:
    """FX, the warehouse locator, Costco's price batches, and Sam's two endpoints."""
    as_json = {"Content-Type": "application/json"}

    def handler(request: httpx.Request) -> httpx.Response:
        host = request.url.host
        if host == "www.samsclub.com":
            path = request.url.path
            if "sitemap" in path:
                return httpx.Response(200, content=sams_sitemap())
            # /club/<id>/fuel-center -- a fuel club answers 200, a non-fuel club
            # 307s to /club/<id>, which the source leaves unfollowed and skips.
            club_id = path.split("/club/")[1].split("/")[0]
            if club_id in SAMS_CLUBS:
                return httpx.Response(
                    200, content=sams_fuel_page(club_id), headers={"Content-Type": "text/html"}
                )
            return httpx.Response(307, headers={"Location": f"/club/{club_id}"})
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
    # The committed config now captures Sam's Club from its public pages; the
    # end-to-end run exercises it straight from that config.
    assert "enabled = true" in text
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
        # A no-op sleep so the ~6 paced Sam's requests do not really wait.
        client=Client(cfg.http, transport=transport(), sleep=lambda _: None),
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
