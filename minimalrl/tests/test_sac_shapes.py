import torch

from minimalrl.algos.sac.sac import Critic, CriticConfig, GaussianPolicy, GaussianPolicyConfig


def test_policy_outputs_distribution():
    config = GaussianPolicyConfig(obs_dim=3, action_dim=2)
    policy = GaussianPolicy(config)
    dist = policy(torch.zeros(4, config.obs_dim))
    sample = dist.sample()
    assert sample.shape == (4, config.action_dim)


def test_critic_output_shape():
    config = CriticConfig(obs_dim=3, action_dim=2)
    critic = Critic(config)
    q = critic(torch.zeros(4, config.obs_dim), torch.zeros(4, config.action_dim))
    assert q.shape == (4,)
