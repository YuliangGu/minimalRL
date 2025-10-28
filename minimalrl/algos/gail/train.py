"""Entry point for training GAIL agents."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

from minimalrl.core.config import ExperimentConfig, load_config


@dataclass
class GAILConfig(ExperimentConfig):
    discriminator_updates: int = 5
    policy_updates: int = 1
    batch_size: int = 1024


def train(config: GAILConfig) -> None:
    """Train a GAIL agent. Implementation to be filled in."""

    raise NotImplementedError("GAIL training loop not yet implemented.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a GAIL agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config, GAILConfig) if args.config else GAILConfig()
    train(config)


if __name__ == "__main__":  # pragma: no cover
    main()
