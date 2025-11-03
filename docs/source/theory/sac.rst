Soft Actor-Critic
=================

We derive the soft actor-critic objective in the *Control as Inference (CaI)* framework, which ties reinforcement learning/control to probabilistic modeling.

Control as Inference
--------------------

Introduce per-step optimality variables :math:`O_t` with likelihood

.. math::

   p(O_t = 1 \mid s_t, a_t) \propto \exp\big(r(s_t, a_t)\big),

where higher reward increases the probability of being optimal. The joint of a trajectory :math:`\tau = (s_0, a_0, \dots)` and :math:`\mathbf{O} := O_{1:T}` is

.. math::

   \begin{aligned}
   p(\tau, \mathbf{O}=1)
      &= p(s_1) \prod_t p(s_{t+1} \mid s_t, a_t)\, p(O_t = 1 \mid s_t, a_t) \\
      &= \Big( p(s_1) \prod_t p(s_{t+1} \mid s_t, a_t) \Big) \exp\Big( \sum_t r(s_t, a_t) \Big).
   \end{aligned}

Maximizing a variational ELBO over policies :math:`q(a \mid s)` (keeping true dynamics) yields the maximum-entropy objective with discount :math:`\gamma`:

.. math::

   J(\pi) = \mathbb{E}_{\tau \sim \pi} \Big[ \sum_{t=0}^{\infty} \gamma^t \big( r(s_t, a_t) - \alpha \log \pi(a_t \mid s_t) \big) \Big].

Soft Values and Optimal Policy
------------------------------

Define soft values under :math:`\pi`:

.. math::

   \begin{aligned}
   Q^{\pi}(s, a) &= r(s, a) + \gamma\, \mathbb{E}_{s' \sim p}\big[ V^{\pi}(s') \big], \\
   V^{\pi}(s) &= \mathbb{E}_{a \sim \pi}\big[ Q^{\pi}(s, a) - \alpha \log \pi(a \mid s) \big].
   \end{aligned}

Optimality gives the log-sum-exp value and Boltzmann policy:

.. math::

   \begin{aligned}
   V^{*}(s) &= \alpha \log \int \exp\big(Q^{*}(s, a)/\alpha\big)\, da, \\
   \pi^{*}(a \mid s) &= \exp\Big( \tfrac{Q^{*}(s, a) - V^{*}(s)}{\alpha} \Big).
   \end{aligned}

Soft Policy Iteration
---------------------

**Evaluation:** solve the soft Bellman equations for :math:`Q^{\pi}`.

**Improvement:** reverse-KL projection at each state,

.. math::

   \pi_{\text{new}}(\cdot \mid s) = \arg\min_{\pi} \mathrm{KL}\Big( \pi(\cdot \mid s) \Big\| \tfrac{\exp(Q^{\pi}(s, \cdot)/\alpha)}{Z(s)} \Big).

SAC (Off-Policy Realization)
----------------------------

Replay buffer :math:`\mathcal{D}`, twin critics :math:`Q_{\phi_1}, Q_{\phi_2}`, target critics :math:`\bar{\phi}_i`, actor :math:`\pi_{\theta}`, temperature :math:`\alpha`.

**Critic target (clipped double-Q):**

.. math::

   y = r + \gamma\, \mathbb{E}_{a' \sim \pi_{\theta}(\cdot \mid s')} \Big[ \min_{i \in \{1,2\}} Q_{\bar{\phi}_i}(s', a') - \alpha \log \pi_{\theta}(a' \mid s') \Big].

**Critic loss (per head):**

.. math::

   \mathcal{L}_Q(\phi_i) = \mathbb{E}_{(s, a, r, s') \sim \mathcal{D}} \big[ ( Q_{\phi_i}(s, a) - y )^2 \big].

**Actor loss (reverse-KL in expectation):**

.. math::

   \mathcal{J}_{\pi}(\theta) = \mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi_{\theta}} \Big[ \alpha \log \pi_{\theta}(a \mid s) - \min_i Q_{\phi_i}(s, a) \Big].

**Temperature (dual) with target entropy :math:`\bar{\mathcal{H}}`:**

.. math::

   \mathcal{J}(\alpha) = \mathbb{E}_{s \sim \mathcal{D},\, a \sim \pi_{\theta}} \Big[ \alpha\big( - \log \pi_{\theta}(a \mid s) - \bar{\mathcal{H}} \big) \Big],

.. math::

   \nabla_{\alpha} \mathcal{J} = \mathbb{E}\big[ - \log \pi_{\theta}(a \mid s) - \bar{\mathcal{H}} \big].

**Reparameterization and :math:`\tanh` squash (continuous actions):**

.. math::

   u = \mu_{\theta}(s) + \sigma_{\theta}(s) \odot \varepsilon,\quad \varepsilon \sim \mathcal{N}(0, I), \qquad a = \tanh(u),

.. math::

   \log \pi_{\theta}(a \mid s) = \log \mathcal{N}\big(u; \mu_{\theta}, \sigma_{\theta}^2\big) - \sum_i \log\big(1 - a_i^2\big) \quad \text{with } a = \tanh(u).

**Target update (Polyak):**

.. math::

   \bar{\phi}_i \leftarrow \tau \phi_i + (1 - \tau) \bar{\phi}_i.

Minimal Training Loop (Pseudocode)
----------------------------------

.. code-block:: text

   for each environment step:
       a ~ πθ(·|s)      # sample action
       s', r, done = env.step(a)
       D.add(s, a, r, s', done)
       s ← s'

   for each gradient step:
       B = sample_minibatch(D)

       # Critic update (both heads)
       with no_grad:
           a' ~ πθ(·|s')
           y = r + γ * ( min_i Q̄_i(s', a') - α * log πθ(a'|s') )

       minimize Σ_i (Q_φi(s, a) - y)^2  over φ1, φ2

       # Actor update (reparameterized)
       ã = fθ(ε; s)     # tanh-Gaussian sample
       minimize [ α * log πθ(ã|s) - min_i Q_φi(s, ã) ] over θ

       # Temperature update
       minimize α * ( - log πθ(ã|s) - H_target ) over α

       # Target networks
       φ̄_i ← τ φ_i + (1 - τ) φ̄_i
