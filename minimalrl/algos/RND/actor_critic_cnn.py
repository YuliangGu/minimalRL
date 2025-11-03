"""PPO for continuous action spaces."""

from __future__ import annotations

from typing import Tuple

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RandomNetworkDistillation(nn.Module):
    """Random Network Distillation module."""

    def __init__(self, in_channels: int = 1, image_shape: Tuple[int, int] = (84, 84)):
        super().__init__()
        h, w = image_shape

        def conv_out(size, k, s, p=0):
            return (size + 2 * p - k) // s + 1
        h2 = conv_out(conv_out(conv_out(h, 8, 4), 4, 2), 3, 1)
        w2 = conv_out(conv_out(conv_out(w, 8, 4), 4, 2), 3, 1)
        feature_output = h2 * w2 * 64

        # Target network (frozen)
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
        )
        for param in self.target.parameters():
            param.requires_grad = False    # freeze target network

        # Predictor network (trainable) with a MLP head to have architectural asymmetry
        self.predictor = nn.Sequential(
            layer_init(nn.Conv2d(in_channels, 32, 8, stride=4)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.LeakyReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.LeakyReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_output, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 512)),
        )
    
    def forward(self, obs):
        # obs is expected in NCHW and float32
        target_features = self.target(obs)
        predictor_features = self.predictor(obs)
        return predictor_features, target_features

class ActorCriticCNN(nn.Module):
    """Nature CNN-based actor-critic for discrete action spaces."""
    def __init__(self, obs_shape: Tuple[int, int, int], action_dim: int, FiLM: bool = False):
        super().__init__()
        c, h, w = obs_shape  # expect NCHW 

        def conv_out(size, k, s, p=0):
            return (size + 2 * p - k) // s + 1

        h2 = conv_out(conv_out(conv_out(h, 8, 4), 4, 2), 3, 1)
        w2 = conv_out(conv_out(conv_out(w, 8, 4), 4, 2), 3, 1)
        feature_dim = h2 * w2 * 64

        # Shared backbone for actor and critic
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(feature_dim, 512)),
            nn.ReLU(),
            layer_init(nn.Linear(512, 448)),           
            nn.ReLU(inplace=True),
        )
        self.residual = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(inplace=True),
        )

        # Actor head
        self.actor = nn.Sequential(
            layer_init(nn.Linear(448, 256), std=0.01),
            nn.ReLU(),
            layer_init(nn.Linear(256, action_dim), std=0.01),
        )

        # Critic head(s)
        self.critic_int = layer_init(nn.Linear(448, 1), std=1.0)
        self.critic_ext = layer_init(nn.Linear(448, 1), std=1.0)
        
        # FiLM (optional)
        self.FiLM = FiLM
        if FiLM:
            self.film_scale = layer_init(nn.Linear(448, 448), std=0.01)
            self.film_shift = layer_init(nn.Linear(448, 448), std=0.01)
    
    def _truck(self, obs):
        x = obs / 255.0
        x = self.encoder(x)
        z = self.residual(x)
        return x, z

    def act(self, obs, action=None):
        x, z = self._truck(obs)

        # actor
        logits = self.actor(z)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        logp = dist.log_prob(action)
        ent = dist.entropy()
        
        # critic 
        if self.FiLM:
            scale = self.film_scale(z)
            shift = self.film_shift(z)
            v_in = x * (scale + 1) + shift
        else:
            v_in = x + z                      
        v_int = self.critic_int(v_in).squeeze(-1)
        v_ext = self.critic_ext(v_in).squeeze(-1)
        return action, logp, ent, v_int, v_ext

    def value(self, obs):
        x, z = self._truck(obs)
        if self.FiLM:
            scale = self.film_scale(z)
            shift = self.film_shift(z)
            v_in = x * (scale + 1) + shift 
        else:
            v_in = x + z
        v_int = self.critic_int(v_in).squeeze(-1)
        v_ext = self.critic_ext(v_in).squeeze(-1)

        return v_int, v_ext

        
