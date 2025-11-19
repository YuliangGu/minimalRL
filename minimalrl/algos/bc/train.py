"""DAgger-style behavior cloning trainer."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import torch

from minimalrl.core.config import ExperimentConfig, load_config
from minimalrl.core.logger import Logger, LoggerConfig
from minimalrl.core.torch_utils import seed_all

from minimalrl.algos.bc.policy import SACExpertPolicy, BehaviorCloningPolicy, materialize_sac_expert
from minimalrl.algos.bc.dataset import DAggerDataset

@dataclass
class DAggerConfig(ExperimentConfig):
    # Overrides
    env_id: str = "InvertedPendulum-v4"
    num_iterations: int = 1000
    hidden_sizes: tuple[int, ...] = (256, 256)
    learning_rate: float = 3e-4
    batch_size: int = 512
    path_to_expert: Optional[str] = None

    warmup_steps: int = 1000
    num_envs: int = 4

def train(config: DAggerConfig) -> None:
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.batch_size

    envs = gym.vector.SyncVectorEnv(
        [lambda: gym.make(config.env_id) for _ in range(config.num_envs)]
    )
    obs_dim = int(np.prod(envs.single_observation_space.shape))
    action_space = envs.single_action_space
    discrete_actions = isinstance(action_space, gym.spaces.Discrete)
    if discrete_actions:
        action_dim = int(action_space.n)
        action_scale = 1.0
    else:
        action_dim = int(np.prod(action_space.shape))
        action_scale = float(action_space.high[0]) if hasattr(action_space, "high") else 1.0

    logger = Logger(config=LoggerConfig(), experiment_name=f"dagger_{config.env_id}")
    dataset = DAggerDataset(device=device, discrete_actions=discrete_actions)

    expert = materialize_sac_expert(config.path_to_expert, device=device)
    bc_policy = BehaviorCloningPolicy(
        obs_dim,
        action_dim,
        config.hidden_sizes,
        action_scale=action_scale,
        discrete_actions=discrete_actions,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(bc_policy.parameters(), lr=config.learning_rate)

    obs, _ = envs.reset(seed=config.seed)
    obs = obs.astype(np.float32)
    total_env_steps = 0
    last_loss: Optional[float] = None
    log_interval = max(1, config.num_iterations // 20)
    eps = 1e-6

    def _ensure_action_dtype(actions: np.ndarray) -> np.ndarray:
        if discrete_actions:
            return actions.astype(np.int64, copy=False)
        return actions.astype(np.float32, copy=False)

    def _step_env(actions: np.ndarray) -> np.ndarray:
        next_obs, _, terminated, truncated, _ = envs.step(actions)
        done = np.logical_or(terminated, truncated)
        if np.any(done):
            done_idx = np.where(done)[0]
            reset_obs, _ = envs.reset_done(done_idx)
            next_obs[done_idx] = reset_obs
        return next_obs.astype(np.float32)

    def _bc_loss(batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        bs = batch["obs"].shape[0]
        obs_batch = batch["obs"].view(bs, -1)
        actions = batch["action"]
        dist = bc_policy(obs_batch)
        if discrete_actions:
            targets = actions.view(-1).long()
            return -dist.log_prob(targets).mean()

        targets = actions.view(bs, action_dim)
        safe_targets = torch.clamp(targets, -action_scale + eps, action_scale - eps)
        log_prob = dist.log_prob(safe_targets)
        return -log_prob.mean()

    # Warmup with expert rollouts
    while total_env_steps < config.warmup_steps:
        expert_actions = _ensure_action_dtype(expert.act(obs, deterministic=True))
        dataset.extend(obs, expert_actions)
        obs = _step_env(expert_actions)
        total_env_steps += config.num_envs

    # Main DAgger loop
    for iteration in range(config.num_iterations):
        expert_actions = _ensure_action_dtype(expert.act(obs, deterministic=True))
        dataset.extend(obs, expert_actions)

        obs_tensor = torch.as_tensor(obs, dtype=torch.float32, device=device).view(config.num_envs, -1)
        with torch.no_grad():
            policy_actions = bc_policy.act(obs_tensor).cpu().numpy()
        policy_actions = _ensure_action_dtype(policy_actions)

        obs = _step_env(policy_actions)
        total_env_steps += config.num_envs

        if len(dataset) >= batch_size:
            batch = dataset.sample_batch(batch_size, device=device)
            loss = _bc_loss(batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            last_loss = float(loss.detach().cpu().item())

        if last_loss is not None and (iteration + 1) % log_interval == 0:
            logger.log(
                {
                    "bc/loss": last_loss,
                    "dataset/size": len(dataset),
                    "env/steps": total_env_steps,
                },
                step=total_env_steps,
            )



def main() -> None:
    parser = argparse.ArgumentParser(description="Train a DAgger-style behavior cloning policy.")
    parser.add_argument("--config", type=str, default=None, help="Optional YAML config path.")
    parser.add_argument(
        "--expert",
        type=str,
        default=None,
        help="Override the expert policy dotted path (module:callable).",
    )
    args = parser.parse_args()

    cfg = load_config(args.config, DAggerConfig) if args.config else DAggerConfig()
    if args.expert:
        cfg.expert_policy = args.expert

    train(cfg)


if __name__ == "__main__":  # pragma: no cover
    main()
