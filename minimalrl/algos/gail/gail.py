"""GAIL discriminator and helper utilities."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from minimalrl.core.networks import MLP


@dataclass
class DiscriminatorConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = (256, 256)


class Discriminator(nn.Module):
    """Binary classifier distinguishing expert from policy trajectories."""

    def __init__(self, config: DiscriminatorConfig):
        super().__init__()
        self.model = MLP(config.obs_dim + config.action_dim, config.hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = torch.cat([obs, actions], dim=-1)
        return torch.sigmoid(self.model(x)).squeeze(-1)
