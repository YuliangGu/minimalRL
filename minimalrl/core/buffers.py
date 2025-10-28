"""Experience buffer implementations used across algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Tuple

import numpy as np

Batch = Dict[str, np.ndarray]


@dataclass
class ReplayBuffer:
    capacity: int
    obs_shape: Tuple[int, ...]
    action_shape: Tuple[int, ...]

    def __post_init__(self) -> None:
        self.ptr = 0
        self.full = False
        self.observations = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, *self.action_shape), dtype=np.float32)
        self.rewards = np.zeros((self.capacity,), dtype=np.float32)
        self.discounts = np.ones((self.capacity,), dtype=np.float32)
        self.next_observations = np.zeros((self.capacity, *self.obs_shape), dtype=np.float32)

    def add(self, obs, action, reward, discount, next_obs) -> None:
        self.observations[self.ptr] = obs
        self.actions[self.ptr] = action
        self.rewards[self.ptr] = reward
        self.discounts[self.ptr] = discount
        self.next_observations[self.ptr] = next_obs
        self.ptr = (self.ptr + 1) % self.capacity
        self.full = self.full or self.ptr == 0

    def sample(self, batch_size: int) -> Batch:
        limit = self.capacity if self.full else self.ptr
        indices = np.random.randint(0, limit, size=batch_size)
        return {
            "observations": self.observations[indices],
            "actions": self.actions[indices],
            "rewards": self.rewards[indices],
            "discounts": self.discounts[indices],
            "next_observations": self.next_observations[indices],
        }


@dataclass
class TrajectoryBuffer:
    """Stores full trajectories for policy-gradient style algorithms."""

    max_length: int

    def __post_init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self._storage: List[Batch] = []

    def append(self, batch: Batch) -> None:
        if len(self._storage) >= self.max_length:
            self._storage.pop(0)
        self._storage.append(batch)

    def flush(self) -> Batch:
        if not self._storage:
            raise RuntimeError("No trajectories stored.")
        stacked = {
            key: np.concatenate([step[key] for step in self._storage], axis=0)
            for key in self._storage[0]
        }
        self.reset()
        return stacked


class PrioritizedReplayBuffer(ReplayBuffer):
    """Optional prioritized replay buffer with proportional prioritization."""

    alpha: float = 0.6
    epsilon: float = 1e-6

    def __post_init__(self) -> None:
        super().__post_init__()
        self.priorities = np.ones((self.capacity,), dtype=np.float32)

    def add(self, obs, action, reward, discount, next_obs) -> None:  # type: ignore[override]
        super().add(obs, action, reward, discount, next_obs)
        index = (self.ptr - 1) % self.capacity
        self.priorities[index] = self.priorities.max() if self.full else 1.0

    def sample(self, batch_size: int, beta: float = 0.4) -> Tuple[Batch, np.ndarray, np.ndarray]:
        limit = self.capacity if self.full else self.ptr
        probs = self.priorities[:limit] ** self.alpha
        probs /= probs.sum()
        indices = np.random.choice(limit, size=batch_size, p=probs)
        batch = super().sample(batch_size)
        weights = (limit * probs[indices]) ** (-beta)
        weights /= weights.max()
        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        self.priorities[indices] = np.maximum(priorities, self.epsilon)
