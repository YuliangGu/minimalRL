"""PPO for continuous action spaces."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

import torch
import torch.nn as nn
from torch.distributions import Categorical

def layer_init(layer, std = np.sqrt(2), bias_const = 0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer

class RandomNetworkDistillation(nn.Module):
    """Random Network Distillation module."""

    def __init__(self, input_size, output_size):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        feature_output = 7 * 7 * 64   # for 84x84 input after conv layers

        # Target network (frozen)
        self.target = nn.Sequential(
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
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
            layer_init(nn.Conv2d(1, 32, 8, stride=4)),
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
        target_features = self.target(obs)
        predictor_features = self.predictor(obs)

        return predictor_features, target_features

class ActorCriticCNN(nn.Module):
    """Actor-Critic network with CNN backbone for image observations."""

    feature_dim = 7 * 7 * 64   # for 84x84 input after conv layers

    def __init__(self, obs_shape, action_dim, FiLM = False):
        super().__init__()
        c, h, w = obs_shape     # for wrappered atari envs, c=4, h=84, w=84

        def conv_out_dim(size, kernel, stride, padding=0):
            return (size + 2 * padding - kernel) // stride + 1

        # Shared backbone for actor and critic
        self.encoder = nn.Sequential(
            layer_init(nn.Conv2d(c, 32, 8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, 4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 64, 3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
            layer_init(nn.Linear(self.feature_dim, 512)),       
            nn.ReLU(),
            layer_init(nn.Linear(512, 448)),           
            nn.ReLU(),
        )
        self.residual = nn.Sequential(
            layer_init(nn.Linear(448, 448), std=0.01),
            nn.ReLU(),
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
        
        # FiLM or Concat
        if FiLM:
            self.FiLM = True
            self.film_scale = layer_init(nn.Linear(448, 448), std=0.01)
            self.film_shift = layer_init(nn.Linear(448, 448), std=0.01)
        else:
            self.FiLM = False

    def act(self, obs, action=None):
        # encode observation
        x = obs / 255.0                         # normalize pixel values
        x = self.encoder(x)

        # residual connection
        z = self.residual(x)

        # actor
        logits = self.actor(z)
        probs = Categorical(logits=logits)
        if action is None:
            action = probs.sample()
        
        # critic (input: x + z or film(x, z))
        if self.FiLM:
            scale = self.film_scale(z)
            shift = self.film_shift(z)
            x = x * (scale + 1) + shift      # FiLM
        else:
            x = x + z                        
        value_int = self.critic_int(x).squeeze(-1)
        value_ext = self.critic_ext(x).squeeze(-1)

        return action, probs.log_prob(action), probs.entropy(), value_int, value_ext
    
    def value(self, obs):
        x = obs / 255.0                        
        x = self.encoder(x)
        z = self.residual(x)
        if self.FiLM:
            scale = self.film_scale(z)
            shift = self.film_shift(z)
            x = x * (scale + 1) + shift      
        else:
            x = x + z                        
        value_int = self.critic_int(x).squeeze(-1)
        value_ext = self.critic_ext(x).squeeze(-1)

        return value_int, value_ext


        
