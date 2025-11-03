"""Policy network for REINFORCE."""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn

from minimalrl.core.networks import MLP


@dataclass
class ReinforcePolicyConfig:
    obs_dim: int
    action_dim: int
    hidden_sizes: tuple[int, ...] = (64, 64)

class ReinforcePolicy(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden_sizes: tuple[int, ...] = (64, 64)):
        super().__init__()
        self.policy_net = MLP(
            in_dim=obs_dim,
            hidden_sizes=hidden_sizes,
            out_dim=action_dim,
            activation=nn.Tanh,
        )

    """ This implementation is for educational purposes.
    Equivalent to using torch.distributions.Categorical"""
    def act(self, obs):
        # obs shape: [B/N, obs_dim]
        batch = obs.view(obs.shape[0], -1)              
        z = self.policy_net(batch)
        log_probs = z - torch.logsumexp(z, dim=-1, keepdim=True)                    # logsumexp trick
        probs = log_probs.exp()                                                     # probabilities
        actions = torch.multinomial(probs, num_samples=1).squeeze(-1)               # multinomial distribution
        return actions
    
    def get_log_probs(self, obs, actions):
        batch = obs.view(obs.shape[0], -1)
        z = self.policy_net(batch)
        log_probs = z - torch.logsumexp(z, dim=-1, keepdim=True)                    # logsumexp trick
        selected_log_probs = log_probs.gather(-1, actions.long().unsqueeze(-1)).squeeze(-1)
        entropy = -(log_probs.exp() * log_probs).sum(dim=-1)
        return selected_log_probs, entropy
