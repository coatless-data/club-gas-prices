import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from costco_gas.alerts import run_alerts
from costco_gas.issues import Issues
from costco_gas.store import open_store

NOW = datetime(2026, 9, 15, 18, 19, 2, tzinfo=UTC)
COUNTRIES = ("US", "CA", "MX", "GB", "AU", "JP", "TW")


def _country(status: str = "ok", **overrides) -> dict:
    entry = {
        "status": status,
        "source": "costco-us-gasprices",
        "stations": 596,
        "rows": 1262,
        "warnings": [],
        "errors": [],
        "consecutive_failures": 0,
        "last_success_capture_id": "2026-09-15T1817Z",
        "recent_errors": [],
    }
    entry.update(overrides)
    return entry


def _status(tmp_path: Path, **overrides) -> Path:
    document = {
        "schema_version": 1,
        "capture_id": "2026-09-15T1817Z",
        "run_id": 123,
        "run_url": "https://github.com/coatless-dashboard/costco-gas-prices/actions/runs/123",
        "fx": {"status": "ok", "source": "frankfurter-v2", "rate_date": "2026-09-15"},
        "ecom_api": {"attempted": True, "http_status": 200},
        "publish": {"outcome": None, "consecutive_failures": 0, "unpublished": []},
        "close": {},
        "warnings": [],
        "countries": {code: _country() for code in COUNTRIES},
    }
    document.update(overrides)
    path = tmp_path / "status.json"
    path.write_text(json.dumps(document, indent=2), encoding="utf-8")
    return path


def _store(tmp_path: Path):
    return open_store(f"local:{tmp_path / 'releases'}")


def _upload_bundle(store, tmp_path: Path, capture_id: str) -> None:
    tag = "data-" + capture_id[:7]
    store.ensure_release(tag, tag, "", True, "false")
    path = tmp_path / f"capture-{capture_id}.tar.gz"
    path.write_bytes(b"not a real tarball, only its presence matters here")
    store.upload_new(tag, path, f"capture-{capture_id}.tar.gz")


@pytest.fixture
def issues() -> Issues:
    return Issues(None, None)


def test_failure_increments_the_counter_and_records_the_capture(tmp_path, issues):
    path = _status(tmp_path)
    result = run_alerts(
        path,
        publish_outcome="failure",
        close_outcome="skipped",
        store=_store(tmp_path),
        issues=issues,
        now=NOW,
    )
    assert result["publish"]["outcome"] == "failure"
    assert result["publish"]["consecutive_failures"] == 1
    assert result["publish"]["unpublished"] == [
        {
            "capture_id": "2026-09-15T1817Z",
            "run_id": 123,
            "run_url": "https://github.com/coatless-dashboard/costco-gas-prices/actions/runs/123",
        }
    ]
    # One failure is below the threshold of 2, but a recorded capture opens it anyway.
    assert "ensure_open: Publish failing" in issues.actions
    assert json.loads(path.read_text(encoding="utf-8")) == result


def test_failure_after_the_bundle_uploaded_records_nothing(tmp_path, issues):
    store = _store(tmp_path)
    _upload_bundle(store, tmp_path, "2026-09-15T1817Z")
    result = run_alerts(
        _status(tmp_path),
        publish_outcome="failure",
        close_outcome="skipped",
        store=store,
        issues=issues,
        now=NOW,
    )
    assert result["publish"]["consecutive_failures"] == 1
    assert result["publish"]["unpublished"] == []
    assert "ensure_open: Publish failing" not in issues.actions


def test_second_failure_opens_the_issue_on_the_counter(tmp_path, issues):
    store = _store(tmp_path)
    _upload_bundle(store, tmp_path, "2026-09-15T1817Z")
    path = _status(tmp_path, publish={"outcome": "failure", "consecutive_failures": 1, "unpublished": []})
    result = run_alerts(
        path, publish_outcome="failure", close_outcome="skipped", store=store, issues=issues, now=NOW
    )
    assert result["publish"]["consecutive_failures"] == 2
    assert "ensure_open: Publish failing" in issues.actions


def test_skipped_leaves_the_counter_and_records_nothing(tmp_path, issues):
    path = _status(tmp_path, publish={"outcome": "failure", "consecutive_failures": 1, "unpublished": []})
    result = run_alerts(
        path, publish_outcome="skipped", close_outcome="skipped", store=_store(tmp_path), issues=issues, now=NOW
    )
    assert result["publish"]["outcome"] == "skipped"
    assert result["publish"]["consecutive_failures"] == 1
    assert result["publish"]["unpublished"] == []
    # Scoped to the publish issue: the default fixture's `ok` countries and
    # 200 ecom_api close their own issues regardless of the publish outcome.
    assert not [a for a in issues.actions if "Publish failing" in a]


def test_success_keeps_the_issue_open_until_the_entry_resolves(tmp_path):
    store = _store(tmp_path)
    path = _status(tmp_path)

    first = Issues(None, None)
    run_alerts(path, publish_outcome="failure", close_outcome="skipped", store=store, issues=first, now=NOW)
    assert "ensure_open: Publish failing" in first.actions

    second = Issues(None, None)
    result = run_alerts(path, publish_outcome="success", close_outcome="success", store=store, issues=second, now=NOW)
    assert result["publish"]["consecutive_failures"] == 0
    assert len(result["publish"]["unpublished"]) == 1
    assert "ensure_open: Publish failing" in second.actions
    assert "close: Publish failing" not in second.actions

    _upload_bundle(store, tmp_path, "2026-09-15T1817Z")
    third = Issues(None, None)
    result = run_alerts(path, publish_outcome="success", close_outcome="success", store=store, issues=third, now=NOW)
    assert result["publish"]["unpublished"] == []
    assert "close: Publish failing" in third.actions


def test_close_outcome_is_recorded(tmp_path, issues):
    result = run_alerts(
        _status(tmp_path),
        publish_outcome="success",
        close_outcome="failure",
        store=_store(tmp_path),
        issues=issues,
        now=NOW,
    )
    assert result["close"] == {"outcome": "failure"}


def test_capture_failing_opens_at_three_consecutive_failures(tmp_path, issues):
    countries = {code: _country() for code in COUNTRIES}
    countries["JP"] = _country(
        "failed",
        consecutive_failures=3,
        last_success_capture_id="2026-09-14T0617Z",
        errors=[{"code": "http_error", "host": "www.costco.co.jp", "http_status": 403, "detail": "blocked"}],
        recent_errors=[
            {
                "capture_id": "2026-09-15T1817Z",
                "run_url": "https://github.com/coatless-dashboard/costco-gas-prices/actions/runs/123",
                "errors": [{"code": "http_error", "host": "www.costco.co.jp", "http_status": 403}],
            }
        ],
    )
    result = run_alerts(
        _status(tmp_path, countries=countries),
        publish_outcome="success",
        close_outcome="success",
        store=_store(tmp_path),
        issues=issues,
        now=NOW,
    )
    assert result["countries"]["JP"]["status"] == "failed"
    assert "ensure_open: Capture failing: JP" in issues.actions
    assert not [a for a in issues.actions if a.startswith("ensure_open: Capture failing: US")]


def test_capture_failing_stays_shut_below_the_threshold(tmp_path, issues):
    countries = {code: _country() for code in COUNTRIES}
    countries["TW"] = _country("failed", consecutive_failures=2)
    run_alerts(
        _status(tmp_path, countries=countries),
        publish_outcome="success",
        close_outcome="success",
        store=_store(tmp_path),
        issues=issues,
        now=NOW,
    )
    assert "ensure_open: Capture failing: TW" not in issues.actions


@pytest.mark.parametrize("country_status", ["ok", "degraded"])
def test_capture_failing_closes_on_a_success(tmp_path, country_status):
    issues = Issues(None, None)
    countries = {code: _country() for code in COUNTRIES}
    countries["MX"] = _country(country_status)
    run_alerts(
        _status(tmp_path, countries=countries),
        publish_outcome="success",
        close_outcome="success",
        store=_store(tmp_path),
        issues=issues,
        now=NOW,
    )
    assert "close: Capture failing: MX" in issues.actions


def test_a_skipped_country_changes_no_alert(tmp_path, issues):
    countries = {code: _country() for code in COUNTRIES}
    countries["GB"] = _country(
        "skipped",
        consecutive_failures=5,
        warnings=[{"code": "unknown_grade", "detail": "5304"}],
    )
    run_alerts(
        _status(tmp_path, countries=countries),
        publish_outcome="success",
        close_outcome="success",
        store=_store(tmp_path),
        issues=issues,
        now=NOW,
    )
    assert not [a for a in issues.actions if "GB" in a]


def test_ecom_api_401_opens_and_200_closes(tmp_path):
    rejected = Issues(None, None)
    run_alerts(
        _status(tmp_path, ecom_api={"attempted": True, "http_status": 401}),
        publish_outcome="success",
        close_outcome="success",
        store=_store(tmp_path),
        issues=rejected,
        now=NOW,
    )
    assert "ensure_open: ecom-api client-identifier rejected" in rejected.actions

    recovered = Issues(None, None)
    run_alerts(
        _status(tmp_path),
        publish_outcome="success",
        close_outcome="success",
        store=_store(tmp_path),
        issues=recovered,
        now=NOW,
    )
    assert "close: ecom-api client-identifier rejected" in recovered.actions


def test_ecom_api_not_attempted_touches_nothing(tmp_path, issues):
    run_alerts(
        _status(tmp_path, ecom_api={"attempted": False, "http_status": None}),
        publish_outcome="success",
        close_outcome="success",
        store=_store(tmp_path),
        issues=issues,
        now=NOW,
    )
    assert not [a for a in issues.actions if "ecom-api" in a]


def test_unknown_grade_opens_one_issue_per_label(tmp_path, issues):
    countries = {code: _country() for code in COUNTRIES}
    countries["AU"] = _country(
        warnings=[
            {"code": "unknown_grade", "detail": "Premium 95"},
            {"code": "unknown_grade", "detail": "Premium 95"},
            {"code": "timezone_from_region", "detail": "109"},
        ]
    )
    countries["JP"] = _country(warnings=[{"code": "unknown_grade", "detail": "Kerosene Winter"}])
    run_alerts(
        _status(tmp_path, countries=countries),
        publish_outcome="success",
        close_outcome="success",
        store=_store(tmp_path),
        issues=issues,
        now=NOW,
    )
    opened = [a for a in issues.actions if a.startswith("ensure_open: Unknown grade label")]
    assert opened == [
        "ensure_open: Unknown grade label: AU Premium 95",
        "ensure_open: Unknown grade label: JP Kerosene Winter",
    ]


def test_period_close_failure_opens_and_success_closes(tmp_path):
    failing = Issues(None, None)
    run_alerts(
        _status(tmp_path),
        publish_outcome="success",
        close_outcome="failure",
        store=_store(tmp_path),
        issues=failing,
        now=NOW,
    )
    assert "ensure_open: Period close failing" in failing.actions

    recovered = Issues(None, None)
    run_alerts(
        _status(tmp_path),
        publish_outcome="success",
        close_outcome="success",
        store=_store(tmp_path),
        issues=recovered,
        now=NOW,
    )
    assert "close: Period close failing" in recovered.actions


def test_period_close_skipped_touches_nothing(tmp_path, issues):
    run_alerts(
        _status(tmp_path),
        publish_outcome="success",
        close_outcome="skipped",
        store=_store(tmp_path),
        issues=issues,
        now=NOW,
    )
    assert not [a for a in issues.actions if "Period close" in a]
