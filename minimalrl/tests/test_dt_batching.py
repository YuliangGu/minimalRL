import torch

from minimalrl.algos.decision_transformer.dt import DecisionTransformer, DecisionTransformerConfig


def test_decision_transformer_shape():
    config = DecisionTransformerConfig(state_dim=5, action_dim=2, seq_length=4)
    model = DecisionTransformer(config)
    batch = 3
    states = torch.zeros(batch, config.seq_length, config.state_dim)
    returns = torch.zeros(batch, config.seq_length)
    timesteps = torch.arange(config.seq_length).repeat(batch, 1).float()
    logits = model(states, returns, timesteps)
    assert logits.shape == (batch, config.seq_length, config.action_dim)
