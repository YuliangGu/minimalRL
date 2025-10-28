"""PPO for continuous action spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn as nn


def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class ActorCritic(nn.Module):
    """Joint policy and value network used by PPO for continuous actions."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (64, 64)):
        super().__init__()
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0]), std=np.sqrt(2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1]), std=np.sqrt(2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[1], action_dim), std=0.01),
        )
        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0]), std=np.sqrt(2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1]), std=np.sqrt(2)),
            nn.Tanh(),
            layer_init(nn.Linear(hidden_sizes[1], 1), std=1.0),
        )
        self.actor_log_std = nn.Parameter(torch.zeros(1, action_dim))  

        print("[ActorCritic] observation dim:", obs_dim, "action dim:", action_dim)
        print("[ActorCritic] parameter group:")
        for name, param in self.named_parameters():
            print("  ", name, param.shape)

    def value(self, obs: torch.Tensor) -> torch.Tensor:
        return self.critic(obs)

    def act(self, obs: torch.Tensor, action: Optional[torch.Tensor] = None) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        action_mean = self.actor_mean(obs)
        action_log_std = self.actor_log_std.expand_as(action_mean)
        action_std = torch.exp(action_log_std)

        probs = torch.distributions.Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        # let action be a = [a_1, a_2, ..., a_n] and assume independence between dimensions.
        # then, prob(a) = prob(a_1) * prob(a_2) * ... * prob(a_n)
        # and H(a_1, a_2, ..., a_n) = H(a_1) + H(a_2) + ... + H(a_n)
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(obs)
