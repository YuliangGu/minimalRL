"""Lightweight smoke tests for common Atari environments."""

from __future__ import annotations

import pytest

gymnasium = pytest.importorskip("gymnasium")
pytest.importorskip("ale_py")

import numpy as np
import envpool

from minimalrl.core.envs import make_env, make_vector_env

ATARI_ENV_IDS = ("ALE/Pong-v5", "ALE/Breakout-v5")


@pytest.mark.parametrize("env_id", ATARI_ENV_IDS)
def test_single_atari_env_reset(env_id: str) -> None:
    env = make_env(env_id, seed=123)
    try:
        obs, info = env.reset()
        assert isinstance(info, dict)
        assert obs.shape == env.observation_space.shape
        assert obs.dtype == env.observation_space.dtype
    finally:
        env.close()


@pytest.mark.parametrize("env_id", ATARI_ENV_IDS)
def test_vector_atari_env_step(env_id: str) -> None:
    num_envs = 10
    envs = envpool.make(
        env_id,
        env_type="gym",
        num_envs=num_envs,
        episodic_life=True,
    reward_clip=True,
  )
    try:
        obs, info = env.reset()
        assert obs.shape == (num_envs,) + env.single_observation_space.shape
        assert obs.dtype == env.single_observation_space.dtype
        actions = np.array([env.action_space.sample() for _ in range(num_envs)])
        next_obs, rewards, dones, truncs, infos = env.step(actions)
        assert next_obs.shape == (num_envs,) + env.single_observation_space.shape
        assert next_obs.dtype == env.single_observation_space.dtype
        assert rewards.shape == (num_envs,)
        assert dones.shape == (num_envs,)
        assert truncs.shape == (num_envs,)
        for info in infos:
            assert isinstance(info, dict)
    finally:
        env.close()
    
