"""Shared test fixtures.

The one thing here is a clock: `Client`'s retry schedule and per-host pacing are
real waits, so a test of an error path used to cost about a second of wall
clock. That is the wrong incentive, because error paths are where the coverage
gaps are. Every `Client` built in a test records its waits instead of serving
them, unless the test asks for the real thing.
"""

from __future__ import annotations

import pytest

from club_gas import http as http_module


@pytest.fixture(autouse=True)
def _fast_client_sleep(request, monkeypatch):
    """Make Client's waits free, and expose what it would have waited.

    A test that genuinely measures the schedule marks itself with
    `@pytest.mark.real_sleep` and gets the real `time.sleep`.
    """
    if request.node.get_closest_marker("real_sleep"):
        return None

    waits: list[float] = []
    original = http_module.Client.__init__

    def patched(self, cfg, *, transport=None, sleep=None):
        original(self, cfg, transport=transport, sleep=sleep or waits.append)

    monkeypatch.setattr(http_module.Client, "__init__", patched)
    return waits


def pytest_configure(config):
    config.addinivalue_line(
        "markers", "real_sleep: this test measures the retry schedule, so let it actually sleep"
    )
