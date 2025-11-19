"""Unit tests for the DAgger behavior cloning module."""

from __future__ import annotations

import numpy as np
import torch

from minimalrl.algos.bc.dataset import DaggerDataset
from minimalrl.algos.bc.train import DAggerConfig, train


class CartPoleSignExpert:
    """Simple heuristic: push towards the pole's leaning direction."""

    def __call__(self, obs: np.ndarray) -> int:
        return 0 if obs[2] < 0 else 1


def test_dagger_dataset_roundtrip() -> None:
    dataset = DaggerDataset(obs_shape=(4,), action_shape=(), discrete_actions=True, capacity=5)
    dataset.add(np.zeros(4, dtype=np.float32), 0)
    dataset.extend([np.ones(4, dtype=np.float32)], [1])
    assert len(dataset) == 2
    obs, acts = next(iter(dataset.dataloader(batch_size=2)))
    assert obs.shape == (2, 4)
    assert acts.dtype == torch.long


def test_dagger_training_smoke(tmp_path) -> None:
    config = DAggerConfig(
        env_id="CartPole-v1",
        dagger_iterations=2,
        rollout_steps=32,
        batch_size=16,
        epochs_per_iteration=1,
        initial_expert_steps=32,
        eval_episodes=0,
        log_dir=tmp_path,
        seed=0,
    )
    policy = train(config, expert_callable=CartPoleSignExpert())
    obs = torch.zeros(4)
    logits = policy(obs.unsqueeze(0))
    assert logits.shape[-1] == 2
