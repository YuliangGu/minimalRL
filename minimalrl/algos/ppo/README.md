# Proximal Policy Optimization

- `train.py` runs PPO with generalized advantage estimation and minibatch updates.
- `actor_critic.py` defines the shared actor-critic model used by PPO.
- `theory.md` explains the surrogate objective, clipping heuristic, and auxiliary losses.

Additional notes live in `docs/algorithms/ppo.md`.

## Quickstart
```bash
python -m minimalrl.algos.ppo.train --help
```
The default configuration launches PPO on Gymnasium's `HalfCheetah-v4`; override hyperparameters via YAML configs passed with `--config`.
