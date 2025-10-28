"""Minimal REINFORCE training loop (fixed)."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import List

import numpy as np
import torch

from minimalrl.core.config import ExperimentConfig, load_config
from minimalrl.core.envs import make_env, make_vector_env
from minimalrl.core.logger import Logger, LoggerConfig
from minimalrl.core.torch_utils import seed_all

from minimalrl.algos.reinforce.policy import ReinforcePolicy, ReinforcePolicyConfig


@dataclass
class ReinforceConfig(ExperimentConfig):
    hidden_sizes: tuple[int, ...] = (128, 128)
    learning_rate: float = 3e-4
    gamma: float = 0.99
    num_steps: int = 200
    num_envs: int = 4
    num_iterations: int = 200
    ent_coef: float = 0.0005


def train(config: ReinforceConfig) -> None:
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Vectorized env is fine for REINFORCE; we’ll treat each env as its own episode stream.
    env = make_vector_env(config.env_id, config.seed, num_envs=config.num_envs, backend="gymnasium")
    obs_shape = env.single_observation_space.shape
    act_shape = env.single_action_space.shape
    T, N = config.num_steps, config.num_envs

    try:
        from gymnasium.spaces import Discrete
        is_discrete = isinstance(env.single_action_space, Discrete)
    except Exception:
        is_discrete = hasattr(env.single_action_space, "n")

    action_dtype = torch.long if is_discrete else torch.float32

    obs_dim = int(np.prod(obs_shape))
    if is_discrete:
        action_dim = env.single_action_space.n      
    else:
        action_dim = int(np.prod(act_shape))   

    agent = ReinforcePolicy(obs_dim=obs_dim, action_dim=action_dim, hidden_sizes=config.hidden_sizes)
    agent.to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    logger = Logger(LoggerConfig(log_dir=config.log_dir))

    # Rollout buffers
    obs_buff  = torch.empty((T, N) + obs_shape, dtype=torch.float32, device=device)
    if len(act_shape) == 0:
        act_buff = torch.empty((T, N), dtype=action_dtype, device=device)
    else:
        act_buff = torch.empty((T, N) + act_shape, dtype=action_dtype, device=device)
    rew_buff  = torch.empty((T, N), dtype=torch.float32, device=device)
    done_buff = torch.empty((T, N), dtype=torch.bool,    device=device)

    # Reset
    out = env.reset(seed=config.seed)
    obs_np = out[0] if isinstance(out, tuple) else out
    obs = torch.as_tensor(obs_np, device=device, dtype=torch.float32)

    ep_ret = torch.zeros(N, dtype=torch.float32, device=device)
    completed_returns: List[float] = []

    global_step = 0
    for it in range(config.num_iterations):
        for t in range(T):
            obs_buff[t] = obs       # shape: [N, obs_dim]

            with torch.no_grad():
                actions = agent.act(obs)
            act_buff[t] = actions

            next_obs_np, reward_np, term_np, trunc_np, infos = env.step(actions.detach().cpu().numpy())

            rew  = torch.as_tensor(reward_np, device=device, dtype=torch.float32).view(-1)
            term = torch.as_tensor(term_np,   device=device, dtype=torch.bool).view(-1)
            trunc= torch.as_tensor(trunc_np,  device=device, dtype=torch.bool).view(-1)
            done = torch.logical_or(term, trunc)

            rew_buff[t]  = rew
            done_buff[t] = done

            ep_ret += rew
            if done.any():
                for i in done.nonzero(as_tuple=False).flatten():
                    completed_returns.append(float(ep_ret[i]))
                    ep_ret[i] = 0.0

            obs = torch.as_tensor(next_obs_np, device=device, dtype=torch.float32)
            global_step += N

        # Compute returns-to-go
        returns_to_go = torch.zeros_like(rew_buff)
        running = torch.zeros(N, dtype=torch.float32, device=device)
        for t in reversed(range(T)):
            not_done = (~done_buff[t]).float()  # 1.0 if not done, 0.0 if done
            running = rew_buff[t] + config.gamma * running * not_done
            returns_to_go[t] = running

        # Flatten
        b_obs = obs_buff.view(T * N, *obs_shape)
        b_ret = returns_to_go.view(T * N)
        if len(act_shape) == 0:
            b_act = act_buff.view(T * N)
        else:
            b_act = act_buff.view(T * N, *act_shape)

        # Get log probs and entropy
        logp, entropy = agent.get_log_probs(b_obs, b_act)

        # Update policy (no baseline)
        loss = - (logp * b_ret).mean()
        
        if config.ent_coef:
            loss = loss - config.ent_coef * entropy.mean()

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
        optimizer.step()

        # Log
        recent = completed_returns[-10:]  # recent 10 returns
        avg_return = float(np.mean(recent)) if recent else float(ep_ret.mean().item())
        logger.log({"loss": float(loss.item()), "avg_ep_return": avg_return}, step=global_step)

    logger.close()



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a REINFORCE agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()

    if args.config:
        train(load_config(args.config, ReinforceConfig))
    else:
        train(ReinforceConfig())


if __name__ == "__main__":  # pragma: no cover
    main()
