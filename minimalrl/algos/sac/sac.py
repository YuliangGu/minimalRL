from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np
import copy
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
        scale = float(self.action_scale)

        action_mean = self.action_mean(net_out)
        action_log_std = self.action_log_std(net_out)
        action_log_std = torch.clamp(action_log_std, LOG_STD_MIN, LOG_STD_MAX)
        
        action_std = torch.exp(action_log_std)
        probs = Normal(action_mean, action_std)

        if deterministic:
            pre_tanh = action_mean
        else:
            pre_tanh = probs.rsample()
        
        base_log_prob = probs.log_prob(pre_tanh).sum(dim=-1)
        log_det = torch.log1p(-torch.tanh(pre_tanh).pow(2) + 1e-6).sum(dim=-1)
        log_prob = base_log_prob - log_det

        action = torch.tanh(pre_tanh)
        if scale != 1.0:
            action = action * scale

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

    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: Tuple[int, ...] = (256, 256), action_scale: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.actor = Actor(obs_dim, action_dim, hidden_sizes, action_scale)
        self.critic1 = Critic(obs_dim, action_dim, hidden_sizes)
        self.critic2 = Critic(obs_dim, action_dim, hidden_sizes)
        
        # target networks
        self.critic1_target = copy.deepcopy(self.critic1)
        self.critic2_target = copy.deepcopy(self.critic2)

        for param in self.critic1_target.parameters():
            param.requires_grad = False
        for param in self.critic2_target.parameters():
            param.requires_grad = False

        if temperature <= 0.0:
            raise ValueError("temperature must be positive for log_alpha computation")

        self.log_alpha = nn.Parameter(torch.tensor(np.log(temperature), dtype=torch.float32))

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        action, _ = self.actor(obs, deterministic)
        return action

    def soft_update(self, tau: float) -> None:
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(tau * param.data + (1.0 - tau) * target_param.data)
