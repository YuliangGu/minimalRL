"""Network building blocks shared across algorithms."""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import torch
from torch import nn


# Orthogonal initialization for MLPs
def init_weights(layer, std: float = 5/3, bias_const: float = 0.0):     # 5/3 for Tanh, sqrt(2) for ReLU
    if isinstance(layer, nn.Linear):
        nn.init.orthogonal_(layer.weight, gain=std)
        if layer.bias is not None:
            nn.init.constant_(layer.bias, bias_const)

class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden_sizes: Sequence[int], out_dim: int, activation=nn.ReLU):
        super().__init__()
        layers = []
        prev_size = in_dim
        for size in hidden_sizes:
            layers += [nn.Linear(prev_size, size), activation()]
            prev_size = size
        layers.append(nn.Linear(prev_size, out_dim))
        self.net = nn.Sequential(*layers)

        std = 5/3 if activation == nn.Tanh else (2 ** 0.5)
        self.net.apply(lambda layer: init_weights(layer, std=std))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor: 
        return self.net(x)


class NatureCNN(nn.Module):
    """Small CNN for pixel-based environments."""

    def __init__(self, in_channels: int, features_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),
            nn.ReLU(),
        )
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(7 * 7 * 64, features_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  
        x = x / 255.0
        x = self.conv(x)
        x = self.flatten(x)
        return torch.relu(self.fc(x))


class DiagonalGaussianHead(nn.Module):
    """Produces a diagonal Gaussian distribution from feature vectors."""

    def __init__(self, in_dim: int, action_dim: int, log_std_bounds: Iterable[float] = (-5.0, 2.0)):
        super().__init__()
        self.mu = nn.Linear(in_dim, action_dim)
        self.log_std = nn.Linear(in_dim, action_dim)
        self.log_std_bounds = tuple(log_std_bounds)

    def forward(self, x: torch.Tensor) -> torch.distributions.Normal: 
        mu = self.mu(x)
        log_std = self.log_std(x)
        min_log_std, max_log_std = self.log_std_bounds
        log_std = torch.tanh(log_std)   # squash to [-1, 1]
        log_std = min_log_std + 0.5 * (max_log_std - min_log_std) * (log_std + 1)
        std = log_std.exp()
        return torch.distributions.Normal(mu, std)
