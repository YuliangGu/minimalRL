Policy Gradient Theorem
=======================

Consider an MDP and a parametrized policy :math:`\pi_{\theta} := \pi(a_t \mid s_t; \theta)`. Define the episodic return objective

.. math::

   J(\theta) = \mathbb{E}_{p_{\theta}} \big[R(\tau)\big],

with *controlled* trajectory distribution :math:`p_{\theta} := p(\tau; \theta)`. Compute the gradient of :math:`J(\theta)` with respect to :math:`\theta`:

.. math::

   \begin{aligned}
   \nabla_{\theta} J(\theta)
      &= \nabla_{\theta} \mathbb{E}_{p_{\theta}}\big[ R(\tau) \big] \\
      &= \mathbb{E}_{p_{\theta}}\big[ \nabla_{\theta} \log p_{\theta}(\tau)\, R(\tau) \big] \qquad \text{(log-derivative trick)} \\
      &= \mathbb{E}_{p_{\theta}}\big[ \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)\, R(\tau)\big].
   \end{aligned}

Split the total return at time :math:`t`:

.. math::

   R(\tau) = R_{<t} + R_{\ge t}.

Let :math:`g_t := \nabla_{\theta} \log \pi_{\theta}(a_t \mid s_t)` and zero out the past:

.. math::

   \begin{aligned}
   \mathbb{E}_{p_{\theta}}\big[ g_t\, R_{<t}\big]
      &= \mathbb{E}_{p_{\theta}} \big[ \mathbb{E}[ g_t\, R_{<t} \mid S_{0:t}, A_{0:t-1}] \big] \\
      &= \mathbb{E}_{p_{\theta}} \big[ R_{<t}\, \mathbb{E}[g_t \mid S_{0:t}, A_{0:t-1}] \big] \\
      &= \mathbb{E}_{p_{\theta}} \big[ R_{<t}\, \underbrace{\mathbb{E}[g_t \mid S_t]}_{=0} \big] = 0.
   \end{aligned}

Define the **reward-to-go**

.. math::

   G_t := \sum_{k=t}^{T-1} \gamma^{k-t} r_k \quad \Longrightarrow \quad R_{\ge t} = \gamma^t G_t.

Summing over :math:`t` gives the policy gradient in the reward-to-go form:

.. math::

   \nabla_{\theta} J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{p_{\theta}} \big[ \gamma^t g_t G_t \big].

The action-value function is

.. math::

   Q^{\pi}(s, a) = \mathbb{E}\big[ G_t \mid S_t = s, A_t = a \big],

which yields the familiar expression found in many texts:

.. math::

   \nabla_{\theta} J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{p_{\theta}} \big[ \gamma^t g_t Q^{\pi}(s_t, a_t) \big].

The discounting :math:`\gamma^t` can be absorbed into the (unnormalized) discounted occupancy measure.

REINFORCE with Baselines
------------------------

Introduce any baseline :math:`b(s_t)` that depends only on the state. Since :math:`\mathbb{E}_{p_{\theta}}[g_t\, b(s_t)] = 0`, subtracting it from the return leaves the gradient unbiased:

.. math::

   \nabla_{\theta} J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{p_{\theta}} \big[ \gamma^t g_t \big(G_t - b(s_t)\big) \big].

Choosing :math:`b(s_t) = V^{\pi}(s_t)` turns the estimator into the advantage form :math:`A^{\pi}(s_t, a_t)`.
