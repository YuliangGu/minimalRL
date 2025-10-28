"""Entry point for Dreamer-style agents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from minimalrl.core.config import ExperimentConfig, load_config


@dataclass
class DreamerConfig(ExperimentConfig):
    latent_dim: int = 64
    hidden_dim: int = 200
    imagination_horizon: int = 15


def train(config: DreamerConfig) -> None:
    """Train a Dreamer agent. Implementation forthcoming."""

    raise NotImplementedError("Dreamer training loop not yet implemented.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a Dreamer agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config, DreamerConfig) if args.config else DreamerConfig()
    train(config)


if __name__ == "__main__":  # pragma: no cover
    main()
