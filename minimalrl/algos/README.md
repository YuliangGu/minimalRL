# minimalRL Algorithms

| Algorithm | Module | Entrypoint | Notes |
|-----------|--------|-----------|-------|
| REINFORCE | `minimalrl.algos.reinforce` | `python -m minimalrl.algos.reinforce.train` | Baseline Monte Carlo policy gradients with baselines in `policy.py`. |
| PPO | `minimalrl.algos.ppo` | `python -m minimalrl.algos.ppo.train` | Clipped surrogate objective with GAE and minibatch updates; see `actor_critic.py`. |
| SAC | `minimalrl.algos.sac` | `python -m minimalrl.algos.sac.train` | Maximum-entropy actor-critic with twin critics and temperature tuning. |
| DAgger BC | `minimalrl.algos.bc` | `python -m minimalrl.algos.bc.train --expert module:Policy` | DAgger-style behavior cloning with optional offline seeding. |
| RND | `minimalrl.algos.RND` | `python -m minimalrl.algos.RND.train` | PPO augmented with intrinsic rewards from random network distillation. |

Algorithm-specific READMEs live alongside the training scripts for quick implementation context, while deeper theoretical notes sit in `docs/algorithms/`.
