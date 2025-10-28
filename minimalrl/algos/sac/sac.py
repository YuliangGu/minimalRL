"""Core SAC modules: actors, critics, and temperature updates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn

from minimalrl.core.distributions import TanhNormal
from minimalrl.core.networks import DiagonalGaussianHead, MLP


@dataclass
class GaussianPolicyConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: Tuple[int, ...] = (256, 256)


class GaussianPolicy(nn.Module):
    """Diagonal Gaussian actor with Tanh squashing."""

    def __init__(self, config: GaussianPolicyConfig):
        super().__init__()
        self.backbone = MLP(config.obs_dim, config.hidden_sizes, config.hidden_sizes[-1])
        self.head = DiagonalGaussianHead(config.hidden_sizes[-1], config.action_dim)

    def forward(self, obs: torch.Tensor) -> TanhNormal:  # type: ignore[override]
        features = torch.relu(self.backbone(obs))
        return TanhNormal(self.head(features))


@dataclass
class CriticConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: Tuple[int, ...] = (256, 256)


class Critic(nn.Module):
    """Q-network approximator."""

    def __init__(self, config: CriticConfig):
        super().__init__()
        input_dim = config.obs_dim + config.action_dim
        self.net = MLP(input_dim, config.hidden_sizes, 1)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = torch.cat([obs, action], dim=-1)
        return self.net(x).squeeze(-1)


class Temperature(nn.Module):
    """Learnable entropy temperature."""

    def __init__(self, init_log_alpha: float = 0.0):
        super().__init__()
        self.log_alpha = nn.Parameter(torch.tensor(init_log_alpha))

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()
