"""Rollout actor utilities for distributed settings."""

from __future__ import annotations

from dataclasses import dataclass
from multiprocessing import Process, Queue
from typing import Callable

import numpy as np


@dataclass
class ActorConfig:
    env_id: str
    seed: int
    steps_per_batch: int = 1_000


def start_actor(config: ActorConfig, policy_fn: Callable[[np.ndarray], np.ndarray], queue: Queue) -> Process:
    """Start an actor process that pushes trajectories to a queue."""

    def worker():
        from minimalrl.core.envs import make_env

        env = make_env(config.env_id, config.seed)
        obs, _ = env.reset()
        obs = obs.astype(np.float32)
        while True:
            trajectories = []
            for _ in range(config.steps_per_batch):
                action = policy_fn(obs)
                next_obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                trajectories.append((obs, action, reward, next_obs, done))
                obs = next_obs.astype(np.float32)
                if done:
                    obs, _ = env.reset()
                    obs = obs.astype(np.float32)
            queue.put(trajectories)

    process = Process(target=worker, daemon=True)
    process.start()
    return process
