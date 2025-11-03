# Soft Actor-Critic

- `sac.py` implements the maximum-entropy actor-critic update.
- `train.py` wires SAC into the training pipeline.
- Theory notes are tracked in `docs/algorithms/sac.md` until a dedicated `theory.md` is written.

## Quickstart
```bash
python -m minimalrl.algos.sac.train --help
```
The CLI is wired up for future work; the actual training loop is stubbed and currently raises `NotImplementedError`.
