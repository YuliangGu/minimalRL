"""PPO(continous) training loop."""

from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import gymnasium as gym
import gymnasium.wrappers as gym_wrappers
import numpy as np
import torch

from minimalrl.algos.ppo.actor_critic import ActorCritic
from minimalrl.core.config import ExperimentConfig, load_config
from minimalrl.core.envs import make_env
from minimalrl.core.logger import Logger, LoggerConfig
from minimalrl.core.torch_utils import seed_all


def wrap_env(env):
    """Continuous action specific wrappers. 
    See CleanRL implementations and https://iclr-blog-track.github.io/2022/03/25/ppo-implementation-details/
    """
    env = gym_wrappers.RecordEpisodeStatistics(env)
    env = gym_wrappers.ClipAction(env)                 
    env = gym_wrappers.NormalizeObservation(env)        # IMPORTANT for continuous control
    env = gym_wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10.0, 10.0),)
    env = gym_wrappers.NormalizeReward(env, gamma=0.99)
    env = gym_wrappers.TransformReward(env, lambda reward: np.clip(reward, -10.0, 10.0),)
    return env

@dataclass
class PPOConfig(ExperimentConfig):
    # Overrides
    env_id: str = "HalfCheetah-v4"
    learning_rate: float = 3e-4
    batch_size: int = 64      
    epochs: int = 10
    num_steps: int = 2048
    num_envs: int = 1
    num_iterations: int = 2000
    lr_min: float = 3e-5
    lr_max: float = 3e-3

    hidden_sizes: tuple[int, ...] = (64, 64, 64)
    epsilon: float = 0.2
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.0
    vf_coef: float = 0.5
    target_kl: float = 0.015


def train(config: PPOConfig) -> None:
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsilon = config.epsilon
    batch_size = config.batch_size
    epochs = config.epochs

    envs = gym.vector.SyncVectorEnv(
        [lambda idx=i: wrap_env(make_env(config.env_id, config.seed + idx)) for i in range(config.num_envs)]
    )
    agent = ActorCritic(
        obs_dim=np.prod(envs.single_observation_space.shape),
        action_dim=np.prod(envs.single_action_space.shape),
        hidden_sizes=config.hidden_sizes,
    )
    agent.to(device)
    optimizer = torch.optim.Adam(agent.parameters(), lr=config.learning_rate, eps=1e-5)
    base_lr, lr_min, lr_max = config.learning_rate, config.lr_min, config.lr_max
    current_lr = base_lr

    run_name = f"PPO_{config.env_id}_{config.seed}_{int(time.time())}"
    logger = Logger(LoggerConfig(log_dir=config.log_dir, run_name=run_name))

    # Rollout buffers
    T, N = config.num_steps, config.num_envs
    obs_buff  = torch.empty((T, N) + envs.single_observation_space.shape, dtype=torch.float32, device=device)
    act_buff  = torch.empty((T, N) + envs.single_action_space.shape, dtype=torch.float32, device=device)
    logp_buff = torch.empty((T, N), dtype=torch.float32, device=device)
    rew_buff  = torch.empty((T, N), dtype=torch.float32, device=device)
    val_buff  = torch.empty((T, N), dtype=torch.float32, device=device)
    done_buff = torch.empty((T, N), dtype=torch.bool,    device=device)

    # Training loop
    steps = 0
    obs, _ = envs.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
    done_t = torch.zeros(N, dtype=torch.bool, device=device)

    for itr in range(config.num_iterations):
        for t in range(T):
            obs_buff[t] = obs_t
            done_buff[t] = done_t
            with torch.no_grad():
                action_t, logp_t, _, value_t = agent.act(obs_t)
            act_buff[t] = action_t
            logp_buff[t] = logp_t
            val_buff[t] = value_t.squeeze(-1)

            obs, reward, terminated, truncated, info = envs.step(action_t.cpu().numpy())
            done_np = np.logical_or(terminated, truncated)
            rew_buff[t] = torch.tensor(reward, dtype=torch.float32, device=device)
            
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            done_t = torch.tensor(done_np, dtype=torch.bool, device=device)

            steps += N
            if "final_info" in info:
                for item in info["final_info"]:
                    if item and "episode" in item:
                        ep_ret = item["episode"]["r"]
                        ep_len = item["episode"]["l"]
                        logger.log({"episode/return": float(ep_ret[0]), "episode/length": float(ep_len[0])}, step=steps)
        
        # GAE: estimate advantage function by exponentially weighting TD-errors
        with torch.no_grad():
            last_value = agent.value(obs_t).squeeze(-1)
            adv_buff = torch.zeros((T, N), dtype=torch.float32, device=device)
            last_gae_lam = 0
            for t in reversed(range(T)):
                if t == T - 1:
                    next_non_terminal = 1.0 - done_t.float()
                    next_values = last_value
                else:
                    next_non_terminal = 1.0 - done_buff[t + 1].float()
                    next_values = val_buff[t + 1]
                delta = rew_buff[t] + config.gamma * next_values * next_non_terminal - val_buff[t]
                adv_buff[t] = last_gae_lam = delta + config.gamma * config.gae_lambda * next_non_terminal * last_gae_lam
            ret_buff = adv_buff + val_buff 
        
        # Flatten the batch
        b_obs = obs_buff.view(T * N, *envs.single_observation_space.shape)
        b_act = act_buff.view(T * N, *envs.single_action_space.shape)
        b_logp = logp_buff.view(T * N)
        b_ret = ret_buff.view(T * N)
        b_adv = adv_buff.view(T * N)
        b_val = val_buff.view(T * N)

        # PPO update
        b_inds = np.arange(T * N)
        last_epoch_kl = 0.0
        for epoch in range(epochs):
            np.random.shuffle(b_inds)
            epoch_kls = []
            for start in range(0, T * N, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]

                _, logp, entropy, value = agent.act(b_obs[mb_inds], b_act[mb_inds])
                ratio = torch.exp(logp - b_logp[mb_inds]) # density ratio

                # KL divergence (KL is estimated by the surrogate f = r - 1 - log r)
                approx_kl = (ratio - 1 - (logp - b_logp[mb_inds])).mean().item()
                epoch_kls.append(float(approx_kl))

                # normalize advantage (minibatch-level)
                mb_adv = b_adv[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # policy loss
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()
                
                # value loss 
                mb_ret = b_ret[mb_inds]
                value_loss_raw = ((value.squeeze(-1) - mb_ret) ** 2).mean()
                value_clipped = b_val[mb_inds] + torch.clamp(value.squeeze(-1) - b_val[mb_inds], -epsilon, epsilon)
                value_loss_clipped = ((value_clipped - mb_ret) ** 2).mean()
                value_loss = 0.5 * torch.max(value_loss_raw, value_loss_clipped)

                # entropy loss
                entropy_loss = -entropy.mean()

                loss = policy_loss + config.vf_coef * value_loss + config.ent_coef * entropy_loss
            
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer.step()

            # Adjust learning rate based on KL divergence
            if epoch_kls:
                mean_kl = float(np.mean(epoch_kls))
                last_epoch_kl = mean_kl
                if mean_kl > 1.5 * config.target_kl and current_lr > lr_min:
                    current_lr = max(current_lr / 1.5, lr_min)
                elif mean_kl < 0.5 * config.target_kl and current_lr < lr_max:
                    current_lr = min(current_lr * 1.5, lr_max)
                for param_group in optimizer.param_groups:
                    param_group["lr"] = current_lr

        logger.log({
            "training/loss": float(loss.item()),
            "training/policy_loss": float(policy_loss.item()),
            "training/value_loss": float(value_loss.item()),
            "training/entropy_loss": float(entropy_loss.item()),
            "training/approx_kl": float(last_epoch_kl),
        }, step=steps)

        # DEBUG
        logger.log({
            "debug/lr": current_lr,
            "debug/kl": last_epoch_kl,
        }, step=steps)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config, PPOConfig) if args.config else PPOConfig()
    train(config)


if __name__ == "__main__": 
    main()
