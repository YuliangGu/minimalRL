"""MinimalRL: small, composable reinforcement learning reference implementations."""

from importlib.metadata import PackageNotFoundError, version

__all__ = ["__version__"]

try:
    __version__ = version("minimalrl")
except PackageNotFoundError:  # pragma: no cover - during development
    __version__ = "0.0.0"
