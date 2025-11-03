## Surrogate objective
Given a reference policy $\pi_{\theta_{\mathrm{old}}}$, define the likelihood ratio $r_t(\theta) := \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\theta_{\mathrm{old}}}(a_t|s_t)}$. The vanilla policy-gradient surrogate is
$$
\mathcal{L}^{\mathrm{PG}}(\theta) = \mathbb{E}\!\left[r_t(\theta) A_t\right],
$$
which pushes the new policy in the direction of positive advantages and away from negative ones.

## PPO objective
PPO keeps the gradient signal but clips the ratio before it can leave a band $[1-\epsilon,1+\epsilon]$:
$$
\mathcal{L}^{\mathrm{CLIP}}(\theta) = \mathbb{E}\!\left[\min \big(r_t(\theta) A_t,\; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t\big)\right].
$$
This preserves the lower-variance surrogate while limiting abrupt, destructive updates.
> 

## Value and entropy losses
PPO typically augments the objective with a squared-error value regression and an entropy bonus:
$$
\mathcal{L}(\theta) = \mathcal{L}^{\mathrm{CLIP}}(\theta) - c_1 \mathbb{E}\big[(V_{\theta}(s_t)-\hat{V}_t)^2\big] + c_2 \mathbb{E}\big[\mathcal{H}(\pi_{\theta}(\cdot|s_t))\big],
$$
where $c_1$ and $c_2$ weight the critic fit and exploration bonus. Updates alternate between collecting trajectories with $\pi_{\theta_{\mathrm{old}}}$ and optimizing $\theta$ under the clipped objective so that $\theta_{\mathrm{old}} \leftarrow \theta$ after several gradient steps.
