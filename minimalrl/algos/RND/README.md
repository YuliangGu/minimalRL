# Random Network Distillation

- `train.py` integrates RND with PPO to add intrinsic rewards for exploration-heavy tasks.
- `actor_critic_cnn.py` hosts the convolutional actor-critic used for Atari experiments.
- `theory.md` covers the predictor/target setup and intrinsic reward derivation.

Head to `docs/algorithms/rnd.md` for a project-level summary.

## Quickstart
```bash
python -m minimalrl.algos.RND.train --help
```
By default this spins up PPO+RND on `Pong-v5`; adjust environments and schedules with CLI flags. EnvPool is required for the default loop, so install with the `envpool` extra.
