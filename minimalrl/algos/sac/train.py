"""Entry point for Soft Actor-Critic training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from minimalrl.core.config import ExperimentConfig, load_config


@dataclass
class SACConfig(ExperimentConfig):
    learning_rate: float = 3e-4
    tau: float = 0.005
    gamma: float = 0.99
    batch_size: int = 256
    init_temperature: float = 0.1


def train(config: SACConfig) -> None:
    """Train a SAC agent. Implementation intentionally deferred."""

    raise NotImplementedError("SAC training loop not yet implemented.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SAC agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config, SACConfig) if args.config else SACConfig()
    train(config)


if __name__ == "__main__":  # pragma: no cover
    main()
