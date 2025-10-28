# Random Network Distillation (RND)

**High-level Idea** Random Network Distillation (RND) augments reinforcement learning with an intrinsic reward. The main idea is to train a predictor network to match the output of a randomly initialized, fixed target network. Whenever the agent encounters novel states, the prediction error are high, thereby providing a continual curiosity signal.

**Architecture** RND uses two networks that share the same architecture:

- **Target Network** $\hat{f}_\theta(x)$: randomly initialized and *frozen*. It maps observations to a feature embedding. 
- **Predictor network** $f_\phi(x)$: initialized separately and trained to regress onto the target’s features.

**Intrinsic Reward** During rollouts, the agent processes each observation with RND and receives an intrinsic reward proportional to the prediction error, often scaled and normalized:
$
\begin{equation}
 r^{\text{int}}_t \propto \|f_\phi(x_t) - \hat{f}_\theta(x_t)\|_2^2.
\end{equation}
$
Combining this with the environment’s extrinsic reward encourages exploration on *novel* states.

## Some Training Details

- Architectural asymmetry
- obs/reward normalization
- *dropout* when training RND
