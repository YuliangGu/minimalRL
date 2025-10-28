# Student-Teacher Distillation (Skeleton)

This module provides a scaffold for experiments where a **teacher** policy supervises a **student** policy. It intentionally omits the concrete implementation so that you can adapt it to imitation learning, policy distillation, or representation transfer projects.

## High-Level Flow
1. Initialize the teacher (expert policy, ensemble, or offline checkpoint).
2. Instantiate the student model architecture and optimizer.
3. Generate teacher-labelled data via rollouts or dataset replay.
4. Minimize a supervised objective (e.g., KL divergence, MSE on Q-values).
5. Periodically evaluate the student in the target environment.

The `StudentTeacherTrainer` class in `train.py` mirrors these stages with placeholder methods. Fill them in with project-specific logic:
- `warmup_teacher`: preload buffers or adapt the teacher to the environment.
- `collect_batch`: gather observations and teacher annotations.
- `distill_step`: compute the distillation loss and apply a gradient update.
- `maybe_evaluate`: log student performance, optionally against the teacher.

## Suggested Extensions
- **Multi-task transfer:** condition the student on task identifiers and reuse a single teacher across tasks.
- **Confidence-aware distillation:** weight the loss by the teacher's action entropy or Q-value variance.
- **Progressive resizing:** start with compact observations, then progressively increase resolution or history length.

## References
- Hinton et al., *Distilling the Knowledge in a Neural Network* (2015).
- Rusu et al., *Policy Distillation* (ICLR 2016).
- Czarnecki et al., *Distilling Policy Distillation* (AISTATS 2019).

Add diagrams or experiment-specific notes here to document your setup once the implementation is complete.
