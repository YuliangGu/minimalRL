from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
from torch import nn
from torch.distributions import Normal

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class Actor(nn.Module):
    """Diagonal Gaussian actor with Tanh squashing."""
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256), action_scale: float = 1.0):
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0]), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1]), std=np.sqrt(2)),
            nn.ReLU(),
        )
        self.action_mean = layer_init(nn.Linear(hidden_sizes[1], action_dim), std=0.01)
        self.action_log_std = layer_init(nn.Linear(hidden_sizes[1], action_dim), std=0.01)
        self.action_scale = action_scale

    def forward(self, obs, deterministic: bool = False) -> tuple[torch.Tensor, torch.Tensor]:
        net_out = self.net(obs)
        action_mean = self.action_mean(net_out)
        action_log_std = self.action_log_std(net_out)
        action_log_std = torch.clamp(action_log_std, LOG_STD_MIN, LOG_STD_MAX)
        action_std = torch.exp(action_log_std)

        probs = Normal(action_mean, action_std)
        if deterministic:
            action = action_mean
        else:
            action = probs.rsample()

        log_prob = probs.log_prob(action).sum(axis=-1) 
        log_prob -= (2 * (np.log(2) - action - nn.functional.softplus(-2 * action))).sum(axis=1) # see SAC paper appendix C

        action = torch.tanh(action) * self.action_scale

        entropy = probs.entropy().sum(axis=-1)
        
        return action, log_prob

class Critic(nn.Module):
    """Single Q-function critic."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256)):
        super().__init__()
        self.q_net = nn.Sequential(
            layer_init(nn.Linear(obs_dim + action_dim, hidden_sizes[0]), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1]), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[1], 1), std=1.0),
        )

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        q_input = torch.cat([obs, action], dim=-1)
        q_value = self.q_net(q_input).squeeze(-1)
        return q_value

class ActorCriticSAC(nn.Module):
    """Combined actor-critic module for SAC."""

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256), action_scale: float = 1.0):
        super().__init__()
        self.actor = Actor(obs_dim, action_dim, hidden_sizes, action_scale)
        self.critic1 = Critic(obs_dim, action_dim, hidden_sizes)
        self.critic2 = Critic(obs_dim, action_dim, hidden_sizes)

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action, _ = self.actor(obs, deterministic)
        return action
