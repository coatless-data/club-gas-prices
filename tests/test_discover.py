import io
import json
import tarfile
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

import polars as pl

from costco_gas.discover import candidate_ids, read_bundle, select_bundle
from costco_gas.store import open_store

NOW = datetime(2026, 10, 2, 3, 41, tzinfo=UTC)

# Trimmed from docs/superpowers/research/2026-09-15/responses/us_ecom_warehouses.body.
# 1838 is the highest US id and 1775 the highest Canadian id in that response.
ECOM_BODY = {
    "warehouses": [
        {
            "warehouseId": "1838",
            "name": [{"value": "Lees Summit", "localeCode": "en-US"}],
            "address": {
                "line1": "10 SE OLDHAM PARKWAY",
                "city": "LEES SUMMIT",
                "territory": "MO",
                "postalCode": "64081-2986",
                "countryName": "US",
                "latitude": 38.90107,
                "longitude": -94.37105,
            },
            "timeZone": "America/Chicago",
            "openingDate": "2026-10-02",
            "services": [{"code": "food"}, {"code": "gas"}],
        },
        {
            "warehouseId": "1775",
            "name": [{"value": "Quebec City Bus Ctr", "localeCode": "en-US"}],
            "address": {
                "line1": "999 RUE DU MARAIS",
                "city": "QUEBEC CITY",
                "territory": "QC",
                "postalCode": "G1M 3T9",
                "countryName": "CA",
                "latitude": 46.82432536217208,
                "longitude": -71.28868094385392,
            },
            "timeZone": "America/Toronto",
            "openingDate": "2025-12-05",
            "services": [{"code": "specialty_item"}],
        },
        {
            "warehouseId": "5231",
            "name": [{"value": "Southampton", "localeCode": "en-US"}],
            "address": {
                "line1": "REGENTS PARK ROAD",
                "city": "SOUTHAMPTON",
                "postalCode": "SO15 8TA",
                "countryName": "GB",
                "latitude": 50.91565,
                "longitude": -1.439563,
            },
            "timeZone": "Europe/London",
            "openingDate": "2013-05-04",
        },
    ]
}

# Trimmed from responses/ca_lookup_default_ua.body: element 0 is always false.
CA_LOOKUP_BODY = [
    False,
    {"stlocID": 1324, "displayName": "1324", "city": "ST JOHNS", "state": "NL", "country": "CA"},
    {
        "stlocID": 1790,
        "displayName": "1790",
        "city": "LLOYDMINSTER",
        "state": "AB",
        "country": "CA",
    },
]

POLLED_IDS = ["140", "1680", "1765", "1772", "1793"]


def _bundle_bytes(capture_id: str) -> bytes:
    """Build a capture bundle with the members discovery reads (spec 8.2)."""
    capture_json = json.dumps({"capture_id": capture_id, "capture_date": capture_id[:10]})
    members = {
        "capture.json": capture_json.encode(),
        "inputs/us_id_set.csv": (
            "source_station_id,id_origin,ecom_state\n"
            + "".join(f"{i},ecom,gas\n" for i in POLLED_IDS)
        ).encode(),
        "inputs/stations_used.csv": (
            b"station_key,country,source_station_id,name\n"
            b"US-1364,US,1364,Bradenton\n"
            b"CA-1324,CA,1324,St Johns\n"
            b"JP-Tomiya,JP,Tomiya,Tomiya\n"
        ),
        "responses/shared/ecom-api.body": json.dumps(ECOM_BODY).encode(),
        "responses/CA/01-lookup.body": json.dumps(CA_LOOKUP_BODY).encode(),
        "responses/US/03-gasprices.body": b'{"1364":{"regular":"3.999"}}',
    }
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name, payload in members.items():
            info = tarfile.TarInfo(f"capture-{capture_id}/{name}")
            info.size = len(payload)
            tar.addfile(info, io.BytesIO(payload))
    return buffer.getvalue()


def _entry(capture_id: str, us_status: str) -> dict:
    return {
        "capture_id": capture_id,
        "status": {
            "capture_id": capture_id,
            "countries": {"US": {"status": us_status}, "CA": {"status": "ok"}},
        },
        "rows_by_capture_date": {capture_id[:10]: 1700},
    }


def _seed(tmp_path: Path, *, tag="data-2026-10", entries=None, bundles=None):
    store = open_store(f"local:{tmp_path / 'releases'}")
    store.ensure_release(tag, tag, "", True, "false")
    entries = entries if entries is not None else {}
    manifest = tmp_path / f"manifest-{tag.removeprefix('data-')}.json"
    manifest.write_text(json.dumps({"captures": entries}), encoding="utf-8")
    store.upload_new(tag, manifest, manifest.name)
    for capture_id in bundles or []:
        path = tmp_path / f"capture-{capture_id}.tar.gz"
        path.write_bytes(_bundle_bytes(capture_id))
        store.upload_new(tag, path, f"capture-{capture_id}.tar.gz")
    return store


def test_selects_the_newest_uploaded_bundle_with_a_successful_us(tmp_path):
    entries = {
        "2026-10-01T0017Z": _entry("2026-10-01T0017Z", "ok"),
        "2026-10-01T0617Z": _entry("2026-10-01T0617Z", "degraded"),
        "2026-10-01T1217Z": _entry("2026-10-01T1217Z", "ok"),  # bundle never uploaded
        "2026-10-01T1817Z": _entry("2026-10-01T1817Z", "failed"),
    }
    store = _seed(
        tmp_path,
        entries=entries,
        bundles=["2026-10-01T0017Z", "2026-10-01T0617Z", "2026-10-01T1817Z"],
    )
    assert select_bundle(store, now=NOW) == ("data-2026-10", "capture-2026-10-01T0617Z.tar.gz")


def test_falls_back_to_the_previous_month(tmp_path):
    store = _seed(
        tmp_path,
        tag="data-2026-09",
        entries={"2026-09-30T1817Z": _entry("2026-09-30T1817Z", "ok")},
        bundles=["2026-09-30T1817Z"],
    )
    store.ensure_release("data-2026-10", "data-2026-10", "", True, "false")
    assert select_bundle(store, now=NOW) == ("data-2026-09", "capture-2026-09-30T1817Z.tar.gz")


def test_no_usable_bundle_returns_none(tmp_path):
    store = _seed(
        tmp_path,
        entries={"2026-10-01T1817Z": _entry("2026-10-01T1817Z", "failed")},
        bundles=["2026-10-01T1817Z"],
    )
    assert select_bundle(store, now=NOW) is None


def test_read_bundle_pulls_every_id_source(tmp_path):
    store = _seed(
        tmp_path,
        entries={"2026-10-01T0617Z": _entry("2026-10-01T0617Z", "ok")},
        bundles=["2026-10-01T0617Z"],
    )
    bundle = read_bundle(store, "data-2026-10", "capture-2026-10-01T0617Z.tar.gz")
    assert bundle.capture_id == "2026-10-01T0617Z"
    assert bundle.polled_ids == {140, 1680, 1765, 1772, 1793}
    assert bundle.us_ecom_ids == {1838}
    assert bundle.ca_ecom_ids == {1775}
    assert bundle.ca_lookup_ids == {1324, 1790}
    # The Japanese station key is not numeric and the GB warehouse is not counted.
    assert bundle.station_ids == {1364, 1324}


def test_candidate_range_runs_to_the_highest_id_plus_200(tmp_path):
    store = _seed(
        tmp_path,
        entries={"2026-10-01T0617Z": _entry("2026-10-01T0617Z", "ok")},
        bundles=["2026-10-01T0617Z"],
    )
    bundle = read_bundle(store, "data-2026-10", "capture-2026-10-01T0617Z.tar.gz")
    cfg = SimpleNamespace(
        us_extra_ids=pl.DataFrame(
            {"source_station_id": ["1680", "1765", "1772"], "city": ["a", "b", "c"]}
        )
    )
    ids = candidate_ids(bundle, cfg)
    assert ids[0] == 1
    assert ids[-1] == 2038  # the highest US ecom id, 1838, plus 200
    assert 1364 in ids  # known only from stations.csv, so still a candidate
    assert 1775 not in ids  # a Canadian ecom-api id is excluded
    for polled in (140, 1680, 1765, 1772, 1793):
        assert polled not in ids
    assert len(ids) == 2038 - 6  # 5 polled ids plus the Canadian 1775


from costco_gas.config import Bounds  # noqa: E402
from costco_gas.discover import US_PRICE_URL, discover  # noqa: E402
from costco_gas.issues import Issues  # noqa: E402
from costco_gas.sources.base import RawResponse  # noqa: E402

FIXTURE = Path(__file__).resolve().parent / "fixtures" / "us_gasprices_discover.json"


class FakeClient:
    """Stands in for http.Client: no network, no pacing, records every batch."""

    def __init__(self, prices: dict[str, dict], abandon_after: int | None = None):
        self.prices = prices
        self.abandon_after = abandon_after
        self.requested: list[list[int]] = []

    def abandoned(self, url: str) -> bool:
        return self.abandon_after is not None and len(self.requested) >= self.abandon_after

    def request(
        self, key, url, *, profile="default", headers=None, expect_json=True
    ) -> RawResponse:
        ids = url.split("warehouseid=", 1)[1].split("_")
        self.requested.append([int(value) for value in ids])
        body = {value: self.prices.get(value, {}) for value in ids}
        return RawResponse(
            key=key,
            url=url,
            status=200,
            # The endpoint really does serve JSON as text/html; never use this
            # header to decide whether the body is JSON.
            headers={"Content-Type": "text/html;charset=UTF-8"},
            received_at_utc=NOW,
            elapsed_ms=120,
            body=json.dumps(body).encode(),
            error=None,
        )


def _cfg() -> SimpleNamespace:
    """The shape config.load_config produces: bounds keyed by price unit, each
    a Bounds dataclass with .for_grade()."""
    return SimpleNamespace(
        countries={
            "US": SimpleNamespace(
                url=US_PRICE_URL,
                price_unit="USD/gal",
                bounds={
                    "USD/gal": Bounds(min=2.00, max=11.00, grade_overrides={}),
                    "USD/L": Bounds(min=0.50, max=2.50, grade_overrides={}),
                },
            )
        },
        us_extra_ids=pl.DataFrame(
            {"source_station_id": ["1680", "1765", "1772"], "city": ["a", "b", "c"]}
        ),
    )


def test_sweep_reports_only_plausible_new_stations(tmp_path):
    store = _seed(
        tmp_path,
        entries={"2026-10-01T0617Z": _entry("2026-10-01T0617Z", "ok")},
        bundles=["2026-10-01T0617Z"],
    )
    client = FakeClient(json.loads(FIXTURE.read_text(encoding="utf-8")))
    issues = Issues(None, None)

    result = discover(store, client, _cfg(), issues, now=NOW)

    assert result.skipped_reason is None
    assert [c["warehouse_id"] for c in result.candidates] == ["1364"]
    assert result.candidates[0]["prices"] == {"premium": "4.629", "regular": "3.999"}
    assert "ensure_open: US station discovery 2026-10" in issues.actions


def test_sweep_batches_by_ten_and_skips_excluded_ids(tmp_path):
    store = _seed(
        tmp_path,
        entries={"2026-10-01T0617Z": _entry("2026-10-01T0617Z", "ok")},
        bundles=["2026-10-01T0617Z"],
    )
    client = FakeClient(json.loads(FIXTURE.read_text(encoding="utf-8")))
    discover(store, client, _cfg(), Issues(None, None), now=NOW)

    assert all(len(batch) <= 10 for batch in client.requested)
    swept = [value for batch in client.requested for value in batch]
    assert swept[0] == 1
    assert max(swept) == 2038
    assert 1775 not in swept
    for polled in (140, 1680, 1765, 1772, 1793):
        assert polled not in swept


def test_no_usable_bundle_warns_and_reports_nothing(tmp_path, capsys):
    store = _seed(
        tmp_path,
        entries={"2026-10-01T1817Z": _entry("2026-10-01T1817Z", "failed")},
        bundles=["2026-10-01T1817Z"],
    )
    client = FakeClient({})
    issues = Issues(None, None)

    result = discover(store, client, _cfg(), issues, now=NOW)

    assert result.candidates == []
    assert result.skipped_reason == "no_bundle"
    assert client.requested == []
    assert issues.actions == []
    assert "::warning::" in capsys.readouterr().out


def test_no_candidates_opens_no_issue(tmp_path):
    store = _seed(
        tmp_path,
        entries={"2026-10-01T0617Z": _entry("2026-10-01T0617Z", "ok")},
        bundles=["2026-10-01T0617Z"],
    )
    # Only the placeholders and the out-of-range prices answer.
    prices = {
        key: value
        for key, value in json.loads(FIXTURE.read_text(encoding="utf-8")).items()
        if key in ("120", "335", "1090", "1838")
    }
    issues = Issues(None, None)
    result = discover(store, client := FakeClient(prices), _cfg(), issues, now=NOW)
    assert result.candidates == []
    assert issues.actions == []
    assert client.requested  # the sweep still ran


def test_sweep_stops_when_the_host_is_abandoned(tmp_path, capsys):
    store = _seed(
        tmp_path,
        entries={"2026-10-01T0617Z": _entry("2026-10-01T0617Z", "ok")},
        bundles=["2026-10-01T0617Z"],
    )
    client = FakeClient(json.loads(FIXTURE.read_text(encoding="utf-8")), abandon_after=3)
    result = discover(store, client, _cfg(), Issues(None, None), now=NOW)
    assert len(client.requested) == 3
    assert result.candidates == []
    assert "abandoned" in capsys.readouterr().out
