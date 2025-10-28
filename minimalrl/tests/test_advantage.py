import torch

from minimalrl.algos.ppo.ppo import compute_gae


def test_compute_gae_zero_dones():
    rewards = torch.tensor([1.0, 1.0, 1.0])
    values = torch.tensor([0.5, 0.5, 0.5, 0.0])
    dones = torch.zeros_like(rewards)
    advantages = compute_gae(rewards, values, dones, gamma=0.99, gae_lambda=0.95)
    assert advantages.shape == rewards.shape
    assert torch.all(torch.isfinite(advantages))
