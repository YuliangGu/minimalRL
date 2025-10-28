# minimalrl

MinimalRL is a reinforcement learning playbook.

## Project Goals
- **Minimalism.** Only extract utilities when multiple algorithms use them.
- **Wide coverage.** Includes on-policy, off-policy, imitation, offline, world-model, transformer, and diffusion-based methods.
- **Scalability.** Lightweight DDP, actor/learner, and replay server utilities demonstrate distributed training patterns without heavy dependencies.

## Quick Start
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .[dev]
```

Run a quick PPO smoke test:
```bash
make train-ppo-cartpole
```

## Repository Layout
```
minimalrl/
  core/                 # Common utilities that stay small and composable
  algos/                # Each algorithm is self-contained
  dist/                 # Optional distributed helpers
  examples/             # Reproducible entry points and shell scripts
  tests/                # Focused unit + smoke tests
configs/                # YAML configuration files for experiments
```

## Contributing
Pull requests are welcome. Keep changes tightly scoped, document new configurations, and add tests for new utilities.

## License
MinimalRL is released under the [MIT License](LICENSE).
