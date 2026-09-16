"""The shared HTTP client: identification, deadlines, retries, pacing, blocks.

One Client is built per capture and shared by every country thread, so pacing,
block-signal counts and host abandonment are global to the capture.

This module is the only place that reads the wall clock. It records when a
response was observed; it never computes anything from "now".
"""

from __future__ import annotations

import os
import random
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from urllib.parse import urlsplit

import httpx

from costco_gas.config import HttpConfig, TimeoutProfile
from costco_gas.sources.base import RawResponse

# Kept in the capture bundle. Everything else, cookies above all, is dropped.
RECORDED_HEADERS = frozenset(
    {
        "date",
        "etag",
        "cache-control",
        "content-type",
        "server",
        "x-timer",
        "server-timing",
    }
)


class BudgetExceeded(Exception):
    """A named budget ran out. The caller returns what it already has."""


class _RequestDeadline(Exception):
    """Internal: this request's total deadline expired while streaming."""


def host_of(url: str) -> str:
    return urlsplit(url).hostname or ""


class Client:
    def __init__(self, cfg: HttpConfig, *, transport=None) -> None:
        self.cfg = cfg
        self.signals: dict[str, int] = {}
        self.log: list[dict[str, object]] = []
        self._http = httpx.Client(transport=transport, follow_redirects=True)
        self._lock = threading.Lock()
        self._next_allowed: dict[str, float] = {}
        # A budget opened on the thread that built the Client (the capture, FX
        # and metadata budgets) applies to every thread. A budget opened on a
        # country thread applies only to that thread, so one country's budget
        # never aborts another country's request.
        self._owner_thread = threading.get_ident()
        self._global_budgets: list[tuple[str, float]] = []
        self._local = threading.local()
        self._contact_email = os.environ.get("CONTACT_EMAIL", "").strip()

    # -- lifecycle ---------------------------------------------------------

    def close(self) -> None:
        self._http.close()

    def __enter__(self) -> Client:
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.close()

    # -- budgets -----------------------------------------------------------

    @contextmanager
    def budget(self, name: str, seconds: float) -> Iterator[None]:
        entry = (name, time.monotonic() + seconds)
        stack = self._budget_stack()
        stack.append(entry)
        try:
            yield
        finally:
            stack.remove(entry)

    def _budget_stack(self) -> list[tuple[str, float]]:
        if threading.get_ident() == self._owner_thread:
            return self._global_budgets
        stack = getattr(self._local, "budgets", None)
        if stack is None:
            stack = []
            self._local.budgets = stack
        return stack

    def _active_budgets(self) -> list[tuple[str, float]]:
        active = list(self._global_budgets)
        if threading.get_ident() != self._owner_thread:
            active.extend(getattr(self._local, "budgets", []))
        return active

    def _tightest_budget(self) -> tuple[str | None, float]:
        name: str | None = None
        remaining = float("inf")
        now = time.monotonic()
        for budget_name, deadline in self._active_budgets():
            left = deadline - now
            if left < remaining:
                name, remaining = budget_name, left
        return name, remaining

    def _check_budget(self) -> None:
        name, remaining = self._tightest_budget()
        if remaining <= 0:
            raise BudgetExceeded(name or "budget")

    # -- pacing ------------------------------------------------------------

    def _pace(self, host: str) -> None:
        """At most one request per second per host, across all threads."""
        with self._lock:
            now = time.monotonic()
            next_allowed = self._next_allowed.get(host, now)
            wait = max(0.0, next_allowed - now)
            self._next_allowed[host] = max(now, next_allowed) + self.cfg.min_interval_seconds
        if wait > 0:
            time.sleep(wait)

    # -- headers -----------------------------------------------------------

    def _is_costco(self, host: str) -> bool:
        return any(
            host == suffix or host.endswith("." + suffix)
            for suffix in self.cfg.costco_host_suffixes
        )

    def _headers(self, host: str, extra: dict[str, str] | None) -> dict[str, str]:
        # Accept: application/json is required (one storefront returns XML
        # without it). Accept-Encoding: gzip keeps the big metadata response
        # under 200 KB.
        headers = {"Accept": "application/json", "Accept-Encoding": "gzip"}
        if self._is_costco(host):
            # Nothing may be appended to this UA: a project token in it made
            # www.costco.com reset the connection. Identification rides on
            # X-Project instead, which that host accepted.
            headers["User-Agent"] = self.cfg.costco_user_agent
            headers["X-Project"] = self.cfg.x_project
        else:
            headers["User-Agent"] = self.cfg.project_user_agent
        if self._contact_email:
            headers["From"] = self._contact_email
        if extra:
            headers.update(extra)
        return headers

    # -- blocks ------------------------------------------------------------

    def abandoned(self, url: str) -> bool:
        host = host_of(url)
        with self._lock:
            return self.signals.get(host, 0) >= self.cfg.block_signals_before_abandon

    def _block_signal(
        self, status: int | None, body: bytes, error: str | None, expect_json: bool
    ) -> bool:
        """One signal at most per logical request, after its retries finish.

        Content-Type is deliberately never consulted: Costco's legacy
        endpoints serve valid JSON as text/html, so a Content-Type test would
        flag every good US price response as a block.
        """
        if status in (403, 429):
            return True
        if error is not None and "timeout" in error:
            return True
        if b"cpr_chlge" in body:
            return True
        if expect_json and body:
            head = body.lstrip(b"\xef\xbb\xbf").lstrip()
            if head.startswith(b"<"):
                return True
        return False

    # -- requests ----------------------------------------------------------

    def request(
        self,
        key: str,
        url: str,
        *,
        profile: str = "default",
        headers: dict[str, str] | None = None,
        expect_json: bool = True,
    ) -> RawResponse:
        host = host_of(url)
        if self.abandoned(url):
            return RawResponse(
                key=key,
                url=url,
                status=None,
                headers={},
                received_at_utc=datetime.now(UTC),
                elapsed_ms=0,
                body=b"",
                error="host_abandoned",
            )

        timeouts = self.cfg.profiles.get(profile) or self.cfg.profiles["default"]
        request_headers = self._headers(host, headers)
        started = time.monotonic()
        attempts = 0
        status: int | None = None
        response_headers: dict[str, str] = {}
        body = b""
        error: str | None = None

        while True:
            attempts += 1
            self._check_budget()
            self._pace(host)
            self._check_budget()
            status, response_headers, body, error = self._attempt(url, request_headers, timeouts)
            retryable = error is not None or status == 429 or (status is not None and status >= 500)
            if not retryable or attempts >= self.cfg.max_attempts:
                break
            self._sleep_backoff(attempts)

        elapsed_ms = int((time.monotonic() - started) * 1000)
        signal = self._block_signal(status, body, error, expect_json)
        if signal:
            with self._lock:
                self.signals[host] = self.signals.get(host, 0) + 1
        self.log.append(
            {
                "key": key,
                "url": url,
                "host": host,
                "attempts": attempts,
                "status": status,
                "elapsed_ms": elapsed_ms,
                "error": error,
                "block_signal": signal,
            }
        )
        return RawResponse(
            key=key,
            url=url,
            status=status,
            headers=response_headers,
            received_at_utc=datetime.now(UTC),
            elapsed_ms=elapsed_ms,
            body=body,
            error=error,
        )

    def _attempt(
        self, url: str, headers: dict[str, str], timeouts: TimeoutProfile
    ) -> tuple[int | None, dict[str, str], bytes, str | None]:
        # httpx applies read to every read, including the wait for response
        # headers. The total deadline has no httpx equivalent, so the body is
        # streamed and checked chunk by chunk.
        timeout = httpx.Timeout(
            connect=timeouts.connect,
            read=timeouts.read,
            write=timeouts.read,
            pool=timeouts.connect,
        )
        deadline = time.monotonic() + timeouts.total
        try:
            with self._http.stream("GET", url, headers=headers, timeout=timeout) as response:
                chunks: list[bytes] = []
                for chunk in response.iter_bytes():
                    chunks.append(chunk)
                    self._check_deadline(deadline)
                self._check_deadline(deadline)
                return (
                    response.status_code,
                    _recorded_headers(response.headers),
                    b"".join(chunks),
                    None,
                )
        except _RequestDeadline:
            return None, {}, b"", "total_timeout"
        except httpx.TimeoutException as exc:
            return None, {}, b"", f"timeout: {type(exc).__name__}"
        except httpx.HTTPError as exc:
            return None, {}, b"", f"transport: {type(exc).__name__}: {exc}"

    def _check_deadline(self, deadline: float) -> None:
        # The budget is checked first, so an exhausted budget aborts the
        # in-flight request with BudgetExceeded rather than a timeout.
        self._check_budget()
        if time.monotonic() >= deadline:
            raise _RequestDeadline

    def _sleep_backoff(self, attempt: int) -> None:
        schedule = self.cfg.backoff_seconds
        if not schedule:
            return
        index = min(attempt - 1, len(schedule) - 1)
        base = schedule[index]
        if base <= 0:
            return
        jitter = self.cfg.backoff_jitter
        time.sleep(max(0.0, base * (1.0 + random.uniform(-jitter, jitter))))


def _recorded_headers(headers: httpx.Headers) -> dict[str, str]:
    return {
        name.lower(): value for name, value in headers.items() if name.lower() in RECORDED_HEADERS
    }
