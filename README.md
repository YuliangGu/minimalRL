# minimalRL

minimalRL is a lightweight PyTorch playbook for reinforcement learning experiments.

## Highlights
- Clean reference implementations of classic and curiosity-driven algorithms.
- Small core of reusable utilities and experiment configs.
- Ready scripts for training, evaluation, and logging.
- Lightweight theory notes for each algorithm, linked directly from the docs.
- Static theory microsite in `docs/theory/index.html`; keep it in sync with `make theory-site`.

## Quick Start
```bash
git clone https://github.com/your-org/minimalRL.git
cd minimalRL
python -m venv .venv && source .venv/bin/activate
python -m minimalrl.algos.ppo.train
```

Once installed, the `Makefile` mirrors the common workflows:

## Repository Layout
```
minimalrl/
  core/         # experiment config, logging, and torch helpers
  algos/        # self-contained algorithm implementations
  dist/         # optional distributed training utilities
  examples/     # runnable templates and CLI helpers
  tests/        # targeted unit and smoke tests
configs/        # YAML configs 
docs/           # Theory notes and longer-form guides
```

## Algorithms at a Glance
- PPO (`python -m minimalrl.algos.ppo.train`) — clipped policy optimisation with GAE. Doc: [`docs/algorithms/ppo.md`](docs/algorithms/ppo.md).
- REINFORCE (`python -m minimalrl.algos.reinforce.train`) — vanilla policy gradient baseline. Doc: [`docs/algorithms/reinforce.md`](docs/algorithms/reinforce.md).
- DAgger BC (`python -m minimalrl.algos.bc.train --expert module:Policy`) — dataset aggregation behavior cloning. Doc: [`docs/source/algorithms/bc/index.rst`](docs/source/algorithms/bc/index.rst).
- SAC (`python -m minimalrl.algos.sac.train`) — entropy-regularised actor-critic for continuous control. Doc: [`docs/algorithms/sac.md`](docs/algorithms/sac.md).
- RND (`python -m minimalrl.algos.RND.train`) — intrinsic reward exploration head atop PPO. Doc: [`docs/algorithms/rnd.md`](docs/algorithms/rnd.md).

Each algorithm directory carries a short README with entry points and implementation notes: see [`minimalrl/algos/`](minimalrl/algos/README.md).

## Contributing
Open pull requests with focused changes, document new configs, and add targeted tests. Include links to any new theory notes under `docs/`.

## License
Released under the [MIT License](LICENSE).
