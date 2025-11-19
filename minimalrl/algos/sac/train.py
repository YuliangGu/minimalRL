"""Entry point for Soft Actor-Critic training."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import time

import gymnasium as gym
import numpy as np
import torch

from minimalrl.core.config import ExperimentConfig, load_config
from minimalrl.core.torch_utils import seed_all
from minimalrl.core.logger import Logger, LoggerConfig

from minimalrl.algos.sac.sac import ActorCriticSAC
from minimalrl.core.buffers import ReplayBuffer

@dataclass
class SACConfig(ExperimentConfig):
    # Overrides
    env_id: str = "InvertedPendulum-v4"
    learning_rate: float = 3e-4
    batch_size: int = 256
    warmup_steps: int = 5000
    total_steps: int = 1_000_000
    replay_buffer_size: int = 1_000_000
    update_frequency: int = 1

    save_frequency: int = 10000  # save every n steps; if negative, disables saving
    save_data: bool = True
    save_dir: Path = Path("SACagents")
    
    hidden_sizes: tuple[int, ...] = (256, 256)
    gamma: float = 0.99
    tau: float = 0.005
    alpha: float = -1.0                 # target entropy; if negative, defaults to -action_dim
    init_temperature: float = 0.1      

def train(config: SACConfig) -> None:
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    batch_size = config.batch_size
    replay_buffer_size = config.replay_buffer_size
    gamma = config.gamma
    tau = config.tau

    env = gym.make(config.env_id)
    obs_shape = env.observation_space.shape
    action_shape = env.action_space.shape
    assert env.action_space.__class__.__name__ == "Box", "This implementation only supports continuous action spaces."

    target_entropy = config.alpha if config.alpha >= 0.0 else -float(action_shape[0])
    init_temperature = max(config.init_temperature, 1e-6)

    replay_buffer = ReplayBuffer(replay_buffer_size, obs_shape, action_shape)
    action_scale = float(env.action_space.high[0]) if hasattr(env.action_space, "high") else 1.0
    print(f"Action scale: {action_scale}")
    agent = ActorCriticSAC(
        obs_shape[0],
        action_shape[0],
        config.hidden_sizes,
        action_scale=action_scale,
        temperature=init_temperature,
    ).to(device)
    optimizers = {
        "act_opt": torch.optim.Adam(agent.actor.parameters(), lr=config.learning_rate),
        "q_opt": torch.optim.Adam(
            list(agent.critic1.parameters()) + list(agent.critic2.parameters()), lr=config.learning_rate
        ),
        "alpha_opt": torch.optim.Adam([agent.log_alpha], lr=config.learning_rate),
    }

    run_name = f"SAC-{config.env_id}_{config.seed}_{int(time.time())}"
    logger = Logger(LoggerConfig(log_dir=config.log_dir, run_name=run_name))

    save_dir = Path(config.save_dir)
    if config.save_frequency > 0:
        save_dir.mkdir(parents=True, exist_ok=True)

    steps = 0
    obs, _ = env.reset()
    episode_return = 0.0
    episode_length = 0

    for _ in range(config.total_steps):
        obs_t = torch.tensor(obs, dtype=torch.float32, device=device).unsqueeze(0)

        if steps < config.warmup_steps:
            action = env.action_space.sample()
        else:
            with torch.no_grad():
                action_t, _ = agent.actor.forward(obs_t)
            action = action_t.cpu().numpy()[0]

        next_obs, reward, done, truncated, info = env.step(action)
        episode_return += reward
        episode_length += 1

        replay_buffer.add(obs, action, reward, float(done), next_obs)

        obs = next_obs
        steps += 1

        if done or truncated:
            logger.log({
                "training/episode_return": episode_return,
                "training/episode_length": episode_length,
            }, step=steps)
            obs, _ = env.reset()
            episode_return = 0.0
            episode_length = 0

        if steps >= config.warmup_steps and steps % config.update_frequency == 0:
            for _ in range(config.update_frequency):
                obs_b, action_b, reward_b, done_b, next_obs_b = replay_buffer.sample_t(batch_size, device).values()

                # critic loss
                q1, q2 = agent.critic1(obs_b, action_b), agent.critic2(obs_b, action_b)
                with torch.no_grad():
                    next_action_b, next_log_prob_b = agent.actor(next_obs_b)
                    q1_target, q2_target = agent.critic1_target(next_obs_b, next_action_b), agent.critic2_target(next_obs_b, next_action_b)
                    q_target = torch.min(q1_target, q2_target)
                    backup = reward_b + (1 - done_b) * gamma * (q_target - torch.exp(agent.log_alpha) * next_log_prob_b)
                
                loss_q1 = ((q1 - backup) ** 2).mean()
                loss_q2 = ((q2 - backup) ** 2).mean()
                loss_q = loss_q1 + loss_q2

                optimizers["q_opt"].zero_grad()
                loss_q.backward()
                optimizers["q_opt"].step()

                # actor loss
                action_b_new, log_prob_b = agent.actor(obs_b)
                q1_new, q2_new = agent.critic1(obs_b, action_b_new), agent.critic2(obs_b, action_b_new)
                q_new = torch.min(q1_new, q2_new)
                loss_actor = (torch.exp(agent.log_alpha) * log_prob_b - q_new).mean()

                optimizers["act_opt"].zero_grad()
                loss_actor.backward()
                optimizers["act_opt"].step()

                # temperature loss
                alpha_loss = -(agent.log_alpha * (log_prob_b + target_entropy).detach()).mean()

                optimizers["alpha_opt"].zero_grad()
                alpha_loss.backward()
                optimizers["alpha_opt"].step()

                # Polyak averaging for target networks
                with torch.no_grad():
                    agent.soft_update(tau)

                # Logging
                logger.log({
                    "training/loss_q1": loss_q1.item(),
                    "training/loss_q2": loss_q2.item(),
                    "training/loss_actor": loss_actor.item(),
                    "training/alpha_loss": alpha_loss.item(),
                    "training/alpha": torch.exp(agent.log_alpha).item(),
                }, step=steps)

                # Save model periodically
                if config.save_frequency > 0 and steps % config.save_frequency == 0:
                    checkpoint_path = save_dir / f"sac_actor_{config.env_id}_step{steps}_seed{config.seed}.pt"
                    torch.save(
                        {
                            "actor": agent.actor.state_dict(),
                            "obs_dim": obs_shape[0],
                            "action_dim": action_shape[0],
                            "hidden_sizes": config.hidden_sizes,
                            "action_scale": action_scale,
                        },
                        checkpoint_path,
                    )

                    if config.save_data:
                        limit = replay_buffer.capacity if replay_buffer.full else replay_buffer.ptr
                        if limit > 0:
                            data_path = save_dir / f"sac_data_{config.env_id}_seed{config.seed}.pt"
                            observations = torch.as_tensor(
                                replay_buffer.observations[:limit], dtype=torch.float32
                            )
                            actions = torch.as_tensor(replay_buffer.actions[:limit], dtype=torch.float32)
                            torch.save({"observations": observations, "actions": actions}, data_path)
                        

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a SAC agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config, SACConfig) if args.config else SACConfig()
    train(config)


if __name__ == "__main__":  # pragma: no cover
    main()
