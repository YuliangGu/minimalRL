"""Neural policy used for behavior cloning / DAgger."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical, Distribution, Independent, Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

from minimalrl.algos.sac.sac import Actor

LOG_STD_MAX = 2
LOG_STD_MIN = -20

def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

@dataclass
class SACExpertPolicy:
    """Wrapper that turns a saved SAC actor into a callable expert policy."""

    actor: Actor
    device: torch.device
    obs_dim: int

    @classmethod
    def from_checkpoint(
        cls,
        checkpoint: str | Path,
        *,
        device: torch.device | str | None = None,
    ) -> "SACExpertPolicy":
        path = Path(checkpoint)
        if device is None:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        device = torch.device(device)
        payload = torch.load(path, map_location=device)
        actor_state = payload["actor"]
        obs_dim = int(payload["obs_dim"])
        action_dim = int(payload["action_dim"])
        hidden_sizes = tuple(int(v) for v in payload["hidden_sizes"])
        action_scale = float(payload.get("action_scale", 1.0))
        actor = Actor(obs_dim, action_dim, hidden_sizes, action_scale).to(device)
        actor.load_state_dict(actor_state, strict=True)
        actor.eval()
        return cls(actor=actor, device=device, obs_dim=obs_dim)

    def _distribution_and_mean(
        self, obs_tensor: torch.Tensor
    ) -> Tuple[TransformedDistribution, torch.Tensor]:
        with torch.no_grad():
            net_out = self.actor.net(obs_tensor)
            mean = self.actor.action_mean(net_out)
            log_std = torch.clamp(self.actor.action_log_std(net_out), LOG_STD_MIN, LOG_STD_MAX)
            std = torch.exp(log_std)
        base = Independent(Normal(mean, std), 1)
        transforms = [TanhTransform(cache_size=1)]
        scale = float(self.actor.action_scale)
        if scale != 1.0:
            transforms.append(AffineTransform(loc=0.0, scale=scale))
        dist = TransformedDistribution(base, transforms)
        return dist, mean

    def __call__(self, obs: np.ndarray | torch.Tensor) -> TransformedDistribution:
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs.to(self.device)
        obs_tensor = obs_tensor.view(-1, self.obs_dim)
        dist, _ = self._distribution_and_mean(obs_tensor)
        return dist

    def act(
        self,
        obs: np.ndarray | torch.Tensor,
        *,
        deterministic: bool = True,
    ) -> np.ndarray:
        if isinstance(obs, np.ndarray):
            obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        else:
            obs_tensor = obs.to(self.device)
        obs_tensor = obs_tensor.view(-1, self.obs_dim)
        dist, mean = self._distribution_and_mean(obs_tensor)
        if deterministic:
            action = torch.tanh(mean)
            scale = float(self.actor.action_scale)
            if scale != 1.0:
                action = action * scale
        else:
            action = dist.sample()
        return action.detach().cpu().numpy()


def materialize_sac_expert(checkpoint: str | Path, *, device: torch.device | str | None = None) -> SACExpertPolicy:
    return SACExpertPolicy.from_checkpoint(checkpoint, device=device)

class BehaviorCloningPolicy(nn.Module):
    """A simple MLP policy for behavior cloning."""

    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        hidden_sizes: Tuple[int, ...] = (128, 128),
        action_std: float = 0.1,
        action_scale: float = 1.0,
        discrete_actions: bool = False,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            layer_init(nn.Linear(obs_dim, hidden_sizes[0]), std=np.sqrt(2)),
            nn.ReLU(),
            layer_init(nn.Linear(hidden_sizes[0], hidden_sizes[1]), std=np.sqrt(2)),
            nn.ReLU(),
        )
        self.discrete_actions = discrete_actions
        if discrete_actions:
            self.action_layer = nn.Linear(hidden_sizes[-1], action_dim)
        else:
            self.action_mean = layer_init(nn.Linear(hidden_sizes[-1], action_dim), std=0.01)
            self.action_logstd = nn.Parameter(torch.ones(1, action_dim) * np.log(action_std))
            self.action_scale = action_scale

    def _dist_and_stats(
        self, obs: torch.Tensor
    ) -> Tuple[Distribution, torch.Tensor]:
        obs = obs.view(obs.shape[0], -1)
        net_out = self.net(obs)
        if self.discrete_actions:
            logits = self.action_layer(net_out)
            dist = Categorical(logits=logits)
            return dist, logits

        action_mean = self.action_mean(net_out)
        action_logstd = torch.clamp(self.action_logstd, LOG_STD_MIN, LOG_STD_MAX)
        action_logstd = action_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        base = Independent(Normal(action_mean, action_std), 1)
        transforms = [TanhTransform(cache_size=1)]
        scale = float(self.action_scale)
        if scale != 1.0:
            transforms.append(AffineTransform(loc=0.0, scale=scale))
        dist = TransformedDistribution(base, transforms)
        return dist, action_mean

    def distribution(self, obs: torch.Tensor) -> Distribution:
        dist, _ = self._dist_and_stats(obs)
        return dist

    def act(self, obs: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        dist, stats = self._dist_and_stats(obs)
        if deterministic:
            if self.discrete_actions:
                return torch.argmax(stats, dim=-1)
            action = torch.tanh(stats)
            scale = float(self.action_scale)
            if scale != 1.0:
                action = action * scale
            return action
        if self.discrete_actions:
            return dist.sample()
        return dist.rsample()

    def forward(self, obs: torch.Tensor) -> Distribution:
        return self.distribution(obs)
