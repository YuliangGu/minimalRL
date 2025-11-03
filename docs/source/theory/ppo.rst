Surrogate Objective
-------------------

Given a reference policy :math:`\pi_{\theta_{\mathrm{old}}}`, define the likelihood ratio
:math:`r_t(\theta) := \pi_{\theta}(a_t \mid s_t) / \pi_{\theta_{\mathrm{old}}}(a_t \mid s_t)`. The vanilla policy-gradient surrogate is

.. math::

   \mathcal{L}^{\mathrm{PG}}(\theta) = \mathbb{E}\big[r_t(\theta) A_t\big],

which pushes the new policy toward positive advantages and away from negative ones.

Why Clipping
------------

Large updates can make :math:`r_t(\theta)` drift far from :math:`1`, over-emphasizing a few trajectories and breaking the trust-region intuition of small policy changes. Empirically this leads to collapsed policies or unstable improvement.

PPO Objective
-------------

PPO keeps the gradient signal but clips the ratio before it can leave the band :math:`[1-\epsilon, 1+\epsilon]`:

.. math::

   \mathcal{L}^{\mathrm{CLIP}}(\theta) = \mathbb{E}\Big[\min \big( r_t(\theta) A_t,\; \mathrm{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \big)\Big].

This preserves the low-variance surrogate while limiting destructive updates.

Value and Entropy Losses
------------------------

PPO typically augments the objective with a squared-error value regression and an entropy bonus:

.. math::

   \mathcal{L}(\theta) = \mathcal{L}^{\mathrm{CLIP}}(\theta) - c_1 \mathbb{E}\big[(V_{\theta}(s_t) - \hat{V}_t)^2\big] + c_2 \mathbb{E}\big[\mathcal{H}(\pi_{\theta}(\cdot \mid s_t))\big],

where :math:`c_1` and :math:`c_2` weight the critic fit and exploration bonus. Updates alternate between collecting trajectories with :math:`\pi_{\theta_{\mathrm{old}}}` and optimizing :math:`\theta` under the clipped objective so that :math:`\theta_{\mathrm{old}} \leftarrow \theta` after several gradient steps.
