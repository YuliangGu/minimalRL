import torch

from minimalrl.algos.cql.cql import CQLHyperParams, conservative_loss


class DummyCritic(torch.nn.Module):
    def forward(self, obs: torch.Tensor, act: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        return torch.sum(obs * 0.1 + act * 0.2, dim=-1)


class DummyPolicy(torch.nn.Module):
    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # pragma: no cover
        return torch.zeros_like(obs)


def test_conservative_loss_runs():
    critic = DummyCritic()
    policy = DummyPolicy()
    batch = 32
    obs = torch.zeros(batch, 3)
    actions = torch.zeros(batch, 2)
    rewards = torch.zeros(batch)
    next_obs = torch.zeros(batch, 3)
    discounts = torch.ones(batch)
    params = CQLHyperParams()

    loss = conservative_loss(critic, policy, obs, actions, rewards, next_obs, discounts, params)
    assert torch.isfinite(loss)
