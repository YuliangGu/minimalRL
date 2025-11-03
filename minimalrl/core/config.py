"""Configuration helpers for MinimalRL."""

from __future__ import annotations

from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Type, TypeVar

try:
    import yaml
except ImportError:  # pragma: no cover - optional dependency
    yaml = None  # type: ignore[assignment]

T = TypeVar("T", bound="BaseConfig")


class ConfigError(RuntimeError):
    """Raised when configuration loading fails."""


@dataclass
class BaseConfig:
    """Base class that offers convenience hooks for configs."""

    seed: int = 0
    device: str = "cuda"

    @classmethod
    def from_dict(cls: Type[T], data: Dict[str, Any]) -> T:
        """Instantiate the config, ignoring unknown keys."""

        valid_keys = {f.name for f in fields(cls)}
        filtered = {key: value for key, value in data.items() if key in valid_keys}
        return cls(**filtered)  # type: ignore[arg-type]


@dataclass
class ExperimentConfig(BaseConfig):
    """High-level run settings shared by most algorithms."""

    env_id: str = "CartPole-v1"
    total_steps: int = 200_000
    eval_interval: int = 10_000
    log_dir: Path = field(default_factory=lambda: Path("runs"))


def load_config(path: str | Path, config_cls: Type[T] = ExperimentConfig) -> T:
    """Load a YAML configuration file into the requested dataclass."""

    path = Path(path)
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    with path.open("r", encoding="utf-8") as fh:
        if yaml is None:
            raise ConfigError("PyYAML is not installed; cannot parse configuration files.")
        raw = yaml.safe_load(fh) or {}

    if not isinstance(raw, dict):
        raise ConfigError("Config file must deserialize into a mapping.")

    return config_cls.from_dict(raw)
