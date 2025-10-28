"""Environment helpers for gymnasium and optional EnvPool backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional

import gymnasium as gym

try:
    import envpool
except ImportError:  # pragma: no cover - optional dependency
    envpool = None


@dataclass
class EnvFactory:
    """Factory that builds environments with consistent seeding."""

    env_id: str
    seed: int
    render_mode: Optional[str] = None

    def __call__(self) -> gym.Env:
        env = gym.make(self.env_id, render_mode=self.render_mode)
        env.reset(seed=self.seed)
        return env


def make_env(env_id: str, seed: int, render_mode: Optional[str] = None) -> gym.Env:
    """Return a single gymnasium environment instance."""

    return EnvFactory(env_id, seed, render_mode)()


def make_vector_env(
    env_id: str,
    seed: int,
    num_envs: int,
    asynchronous: bool = False,
    backend: str = "gymnasium",
):
    if backend == "envpool":
        if envpool is None:
            raise RuntimeError("EnvPool is not installed. Use backend='gymnasium' instead.")
        return envpool.make(env_id, env_type="gym", num_envs=num_envs, seed=seed)

    maker: Callable[[], gym.Env] = EnvFactory(env_id, seed)
    if asynchronous:
        return gym.vector.AsyncVectorEnv([maker for _ in range(num_envs)])
    return gym.vector.SyncVectorEnv([maker for _ in range(num_envs)])
