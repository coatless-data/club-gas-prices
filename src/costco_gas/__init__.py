"""Collect Costco fuel prices worldwide and publish them as GitHub Releases."""

from importlib.metadata import PackageNotFoundError, version

COUNTRIES: tuple[str, ...] = ("US", "CA", "MX", "GB", "AU", "JP", "TW")

try:
    __version__ = version("costco-gas-prices")
except PackageNotFoundError:  # pragma: no cover - source checkout without install
    __version__ = "0.0.0"

__all__ = ["COUNTRIES", "__version__"]
