"""Shared GitHub issue helper.

Every alert in this project is a GitHub issue whose *title* is its identity: an
open issue with the same title is refreshed rather than duplicated. Without a
repository or a token the helper prints what it would do, so the same code path
runs locally, in tests and in CI.

Open issues are found with ``GET /repos/{repo}/issues`` rather than the search
API, because search results are eventually consistent and would let a retry
open a second copy of the same issue. That endpoint also returns pull requests,
which carry a ``pull_request`` key and are skipped.
"""

from __future__ import annotations

import os
import textwrap

import httpx

GITHUB_API = "https://api.github.com"
PER_PAGE = 100
TIMEOUT_S = 30.0


def run_url() -> str | None:
    """This workflow run's page, or None outside GitHub Actions."""
    server = os.environ.get("GITHUB_SERVER_URL") or "https://github.com"
    repo = os.environ.get("GITHUB_REPOSITORY")
    run_id = os.environ.get("GITHUB_RUN_ID")
    if not repo or not run_id:
        return None
    return f"{server}/{repo}/actions/runs/{run_id}"


class Issues:
    """Open, refresh and close GitHub issues by title."""

    def __init__(self, repo: str | None, token: str | None) -> None:
        self.repo = repo or None
        self.token = token or None
        self.dry = self.repo is None or self.token is None
        self.actions: list[str] = []
        # Tests assign an httpx.MockTransport here before the first call.
        self.transport: httpx.BaseTransport | None = None
        self._client: httpx.Client | None = None

    def ensure_open(self, title: str, body: str, labels: list[str] | None = None) -> None:
        if self.dry:
            self._print(f"ensure_open: {title}", body)
            return
        try:
            existing = self._find_open(title)
            if existing is None:
                payload: dict[str, object] = {"title": title, "body": body}
                if labels:
                    payload["labels"] = list(labels)
                self._request("POST", f"/repos/{self.repo}/issues", payload)
            else:
                self._request(
                    "PATCH", f"/repos/{self.repo}/issues/{existing['number']}", {"body": body}
                )
        except httpx.HTTPError as exc:
            print(f"::warning::issue API call failed for {title!r}: {exc}")

    def close(self, title: str, comment: str) -> None:
        if self.dry:
            self._print(f"close: {title}", comment)
            return
        try:
            existing = self._find_open(title)
            if existing is None:
                return
            number = existing["number"]
            self._request("POST", f"/repos/{self.repo}/issues/{number}/comments", {"body": comment})
            self._request(
                "PATCH",
                f"/repos/{self.repo}/issues/{number}",
                {"state": "closed", "state_reason": "completed"},
            )
        except httpx.HTTPError as exc:
            print(f"::warning::issue API call failed for {title!r}: {exc}")

    def _print(self, action: str, detail: str) -> None:
        self.actions.append(action)
        print(f"[issues] {action}")
        if detail:
            print(textwrap.indent(detail.strip(), "    "))

    def _http(self) -> httpx.Client:
        if self._client is None:
            self._client = httpx.Client(
                base_url=GITHUB_API,
                timeout=TIMEOUT_S,
                transport=self.transport,
                headers={
                    "Authorization": f"Bearer {self.token}",
                    "Accept": "application/vnd.github+json",
                    "X-GitHub-Api-Version": "2022-11-28",
                },
            )
        return self._client

    def _request(
        self,
        method: str,
        path: str,
        json_body: dict[str, object] | None = None,
        params: dict[str, object] | None = None,
    ) -> httpx.Response:
        response = self._http().request(method, path, json=json_body, params=params)
        response.raise_for_status()
        self.actions.append(f"{method} {path}")
        return response

    def _find_open(self, title: str) -> dict | None:
        page = 1
        while True:
            response = self._request(
                "GET",
                f"/repos/{self.repo}/issues",
                None,
                {"state": "open", "per_page": PER_PAGE, "page": page},
            )
            items = response.json()
            for item in items:
                if "pull_request" in item:
                    continue
                if item.get("title") == title:
                    return item
            if len(items) < PER_PAGE:
                return None
            page += 1
