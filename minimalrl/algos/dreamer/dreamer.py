"""Dreamer-lite RSSM components."""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class RSSMConfig:
    latent_dim: int = 64
    hidden_dim: int = 200


class RSSM(nn.Module):
    """Deterministic-stochastic recurrent state-space model skeleton."""

    def __init__(self, config: RSSMConfig):
        super().__init__()
        self.config = config
        self.rnn = nn.GRUCell(config.latent_dim, config.hidden_dim)
        self.prior = nn.Linear(config.hidden_dim, config.latent_dim * 2)
        self.posterior = nn.Linear(config.hidden_dim + config.latent_dim, config.latent_dim * 2)

    def imagine_step(self, latent: torch.Tensor) -> torch.Tensor:
        hidden = self.rnn(latent, torch.zeros_like(latent))
        mean, log_std = torch.chunk(self.prior(hidden), 2, dim=-1)
        std = torch.exp(log_std)
        return mean + std * torch.randn_like(std)
