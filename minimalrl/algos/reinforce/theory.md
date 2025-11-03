# Policy Gradient Theorem
Consider an MDP and a parametrized policy $\pi_{\theta} := \pi(a_t|s_t;\theta)$. Define the objective (episodic return) 
$$
\begin{equation}
J(\theta) = \mathbb{E}_{p_{\theta}} [R(\tau)],
\end{equation}
$$
with *controlled* trajectory distribution $p_{\theta}:= p(\tau;\theta)$. Compute the gradient of (1) with respect to the parameter $\theta$,
$$
\begin{align}
    \nabla_{\theta} J(\theta) &=  \nabla_{\theta} \mathbb{E}_{p_{\theta}}\; \big[ R(\tau) \big]\\
    &= \mathbb{E}_{p_{\theta}}\; \big[ \nabla_{\theta} \log p_{\theta}(\tau)\;R(\tau) \big]\;\; \text{(log-derivative trick)} \\
    &= \mathbb{E}_{p_{\theta}}\; \big[ \sum_{t=0}^{T-1} \nabla_{\theta} \log \pi_{\theta}\; R(\tau)\big].
\end{align}
$$
Split the total return at time $t$, 
$$
\begin{equation}
    R(\tau) = R_{<t} + R_{\ge t}.
\end{equation}
$$
Next, we abbreviate score as $g_t := \nabla_{\theta} \log \pi_{\theta}(a_t|s_t)$ and *zero out* the past
$$
\begin{align}
    \mathbb{E}_{p_{\theta}}\; \big[ g_t\; R_{<t}\big] &= \mathbb{E}_{p_{\theta}} \big[ \mathbb{E}[g_t\; R_{<t}|S_{0:t},A_{0:t-1}]\big] \;\; \text{(tower property)} \\
    &= \mathbb{E}_{p_{\theta}} \big[R_{<t}\; \mathbb{E}[g_t|S_{0:t},A_{0:t-1}]\big] \\
     &= \mathbb{E}_{p_{\theta}} \big[R_{<t}\; \underbrace{\mathbb{E}[g_t|S_t]}_{=0}\big] = 0\;\; \text{(score identity)}.
\end{align}
$$
Define the **reward-to-go**,
$$
\begin{equation}
    G_t := \sum_{k=t}^{T-1} \gamma^{k-t} r_k \Longrightarrow\; R_{\ge t} = \gamma^t G_t.
\end{equation}
$$
Summing over $t$ gives the policy gradient in the form of reward-to-go:
$$
\begin{equation}
    \nabla_{\theta} J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{p_{\theta}} [\gamma^t g_t G_t].
\end{equation}
$$
The action value function is
$$
\begin{equation}
     Q^{\pi}(s, a) = \mathbb{E}[G_t |S_t = s, A_t = a].
\end{equation}
$$
which gives the familiar form of policy gradient in many texts
$$
\begin{equation}
    \nabla_{\theta} J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{p_{\theta}} [\gamma^t g_t Q^{\pi}(s_t, a_t)].
\end{equation}
$$
The discounting $\gamma^t$ can be further absorbed into the *discounted* (unnormalized) occupancy measure.

## REINFORCE with baselines
Introduce any baseline $b(s_t)$ that depends only on the state. Since $\mathbb{E}_{p_{\theta}}[g_t\,b(s_t)] = 0$, subtracting it from the return leaves the gradient unbiased:
$$ 
\nabla_{\theta} J(\theta) = \sum_{t=0}^{T-1} \mathbb{E}_{p_{\theta}} \big[\gamma^t g_t \big(G_t - b(s_t)\big)\big].
$$
Choosing $b(s_t) = V^{\pi}(s_t)$ turns the estimator into the advantage $A^{\pi}(s_t,a_t)$ form.
