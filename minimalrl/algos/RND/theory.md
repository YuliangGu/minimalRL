# Random Network Distillation (RND)

**High-level idea.** RND augments reinforcement learning with an intrinsic reward by training a predictor network to match the output of a randomly initialized, fixed target network. Novel states yield large prediction errors, producing a continual curiosity signal.

**Architecture.** Two networks share the same architecture:

- **Target network** $\hat{f}_\theta(x)$ is randomly initialized and frozen; it maps observations to a feature embedding.
- **Predictor network** $f_\phi(x)$ is trained to regress onto the target’s features.

**Intrinsic reward.** During rollouts, the agent receives an intrinsic reward proportional to the prediction error, often scaled and normalized:
$$
r^{\text{int}}_t \propto \|f_\phi(x_t) - \hat{f}_\theta(x_t)\|_2^2.
$$
Combined with the environment’s extrinsic reward, this drives exploration toward novel states.

## Training details
- Architectural asymmetry between predictor and policy networks helps avoid collapse.
- Observation and reward normalization stabilizes both intrinsic and extrinsic signals.
- Dropout on the predictor acts as additional regularization.
