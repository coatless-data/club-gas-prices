import json

import httpx
import pytest

from club_gas.issues import Issues

OPEN_ISSUES = [
    {"number": 8, "title": "Publish failing", "pull_request": {"url": "https://example/pr/8"}},
    {"number": 7, "title": "Publish failing", "body": "stale body"},
    {"number": 6, "title": "Capture failing: JP", "body": "x"},
]


def _issues_with_transport(open_issues):
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content) if request.content else None
        calls.append((request.method, request.url.path, body))
        if request.method == "GET" and request.url.path == "/repos/o/r/issues":
            page = int(request.url.params.get("page", "1"))
            return httpx.Response(200, json=open_issues if page == 1 else [])
        if request.method == "POST" and request.url.path == "/repos/o/r/issues":
            return httpx.Response(201, json={"number": 99})
        if request.method == "PATCH" and request.url.path.startswith("/repos/o/r/issues/"):
            return httpx.Response(200, json={"number": 7})
        if request.method == "POST" and request.url.path.endswith("/comments"):
            return httpx.Response(201, json={"id": 1})
        return httpx.Response(404, json={"message": "unexpected"})

    issues = Issues("o/r", "token-123")
    issues.transport = httpx.MockTransport(handler)
    return issues, calls


def test_dry_mode_records_and_prints(capsys):
    issues = Issues(None, None)
    issues.ensure_open("Publish failing", "body text")
    issues.close("Capture failing: JP", "recovered in 2026-09-15T1817Z")
    printed = capsys.readouterr().out
    assert issues.actions == ["ensure_open: Publish failing", "close: Capture failing: JP"]
    assert "Publish failing" in printed
    assert "body text" in printed


def test_dry_mode_when_only_the_token_is_missing():
    issues = Issues("o/r", None)
    issues.ensure_open("Publish failing", "body")
    assert issues.actions == ["ensure_open: Publish failing"]


def test_ensure_open_refreshes_an_existing_open_issue():
    issues, calls = _issues_with_transport(OPEN_ISSUES)
    issues.ensure_open("Publish failing", "fresh body", ["publish-failure"])
    patches = [c for c in calls if c[0] == "PATCH"]
    assert len(patches) == 1
    # Issue 8 has the same title but is a pull request, so it must be ignored.
    assert patches[0][1] == "/repos/o/r/issues/7"
    assert patches[0][2] == {"body": "fresh body"}
    assert not [c for c in calls if c[0] == "POST"]


def test_ensure_open_creates_the_issue_with_its_label():
    issues, calls = _issues_with_transport([])
    issues.ensure_open("Capture failing: US", "three failures", ["capture-failure"])
    posts = [c for c in calls if c[0] == "POST"]
    assert posts[0][1] == "/repos/o/r/issues"
    assert posts[0][2] == {
        "title": "Capture failing: US",
        "body": "three failures",
        "labels": ["capture-failure"],
    }


def test_close_comments_then_closes():
    issues, calls = _issues_with_transport(OPEN_ISSUES)
    issues.close("Publish failing", "published in 2026-09-15T1817Z")
    assert (
        "POST",
        "/repos/o/r/issues/7/comments",
        {"body": "published in 2026-09-15T1817Z"},
    ) in calls
    assert (
        "PATCH",
        "/repos/o/r/issues/7",
        {"state": "closed", "state_reason": "completed"},
    ) in calls


def test_close_is_a_no_op_when_nothing_is_open():
    issues, calls = _issues_with_transport([])
    issues.close("Publish failing", "nothing to do")
    assert [c[0] for c in calls] == ["GET"]


def test_api_failure_is_a_warning_not_an_exception(capsys):
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"message": "boom"})

    issues = Issues("o/r", "token-123")
    issues.transport = httpx.MockTransport(handler)
    issues.ensure_open("Publish failing", "body")
    assert "::warning::" in capsys.readouterr().out


def test_pagination_follows_a_full_page():
    first_page = [{"number": i, "title": f"noise {i}"} for i in range(100)]
    issues, calls = _issues_with_transport(first_page)
    issues.ensure_open("Publish failing", "body")
    pages = [c for c in calls if c[0] == "GET"]
    assert len(pages) == 2
    assert [c[0] for c in calls if c[0] == "POST"] == ["POST"]


@pytest.mark.parametrize("repo,token,dry", [(None, None, True), ("o/r", "t", False)])
def test_dry_flag(repo, token, dry):
    assert Issues(repo, token).dry is dry
