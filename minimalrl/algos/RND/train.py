from __future__ import annotations

import argparse
import time
from collections import deque
from dataclasses import dataclass
from typing import Tuple

import envpool
import gym
import numpy as np
import torch
from actor_critic_cnn import ActorCriticCNN, RandomNetworkDistillation
from gym.wrappers.normalize import RunningMeanStd

from minimalrl.core.config import ExperimentConfig, load_config
from minimalrl.core.logger import Logger, LoggerConfig
from minimalrl.core.torch_utils import seed_all


# Some helpers for handling different image shape conventions
def space_chw(shape: Tuple[int, int, int]) -> Tuple[int, int, int]:
    """Return (C,H,W) regardless of whether env returns CHW or HWC."""
    assert len(shape) == 3
    if shape[0] in (1, 3, 4):
        return shape[0], shape[1], shape[2]           # CHW
    else:
        return shape[2], shape[0], shape[1]           # HWC -> CHW

def ensure_nchw(x: torch.Tensor) -> torch.Tensor:
    """Convert [N,H,W,C] to [N,C,H,W] if needed."""
    assert x.ndim == 4, f"obs must be 4D [N,C,H,W] or [N,H,W,C], got {x.shape}"
    if x.shape[1] in (1, 3, 4):  # already NCHW
        return x
    return x.permute(0, 3, 1, 2).contiguous()

def last_frame_nchw(x: torch.Tensor) -> torch.Tensor:
    """Return the last stacked frame in NCHW -> shape [N,1,H,W]."""
    x = ensure_nchw(x)
    return x[:, -1:, ...]

# wrappers (CleanRL/OpenAI Baselines style)
class RewardForwardFilter: 
    def __init__(self, gamma):
        self.rewems = None
        self.gamma = gamma

    def update(self, rews):
        if self.rewems is None:
            self.rewems = rews
        else:
            self.rewems = self.rewems * self.gamma + rews
        return self.rewems

class RecordEpisodeStatistics(gym.Wrapper): # CleanRL's record episode return and length
    def __init__(self, env, deque_size=100):
        super().__init__(env)
        self.num_envs = getattr(env, "num_envs", 1)
        self.episode_returns = None
        self.episode_lengths = None

    def reset(self, **kwargs):
        observations = super().reset(**kwargs)
        self.episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        self.lives = np.zeros(self.num_envs, dtype=np.int32)
        self.returned_episode_returns = np.zeros(self.num_envs, dtype=np.float32)
        self.returned_episode_lengths = np.zeros(self.num_envs, dtype=np.int32)
        return observations

    def step(self, action):
        observations, rewards, dones, infos = super().step(action)
        self.episode_returns += infos["reward"]
        self.episode_lengths += 1
        self.returned_episode_returns[:] = self.episode_returns
        self.returned_episode_lengths[:] = self.episode_lengths
        self.episode_returns *= 1 - infos["terminated"]
        self.episode_lengths *= 1 - infos["terminated"]
        infos["r"] = self.returned_episode_returns
        infos["l"] = self.returned_episode_lengths
        return (
            observations,
            rewards,
            dones,
            infos,
        )

@dataclass
class RNDConfig(ExperimentConfig):
    # Overrides
    env_id: str = "Pong-v5"
    learning_rate: float = 1e-4
    batch_size: int = 64
    epochs: int = 5
    num_steps: int = 128
    num_envs: int = 16
    num_iterations: int = 20000

    epsilon: float = 0.2
    gamma: float = 0.999
    gae_lambda: float = 0.95
    ent_coef: float = 0.001
    vf_coef: float = 0.5
    vf_clip_coef: float = 0.2
    rnd_update_proportion: float = 0.25


def train(config: RNDConfig) -> None:
    seed_all(config.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    epsilon = config.epsilon
    batch_size = config.batch_size
    epochs = config.epochs

    envs = envpool.make(config.env_id, 
                        num_envs=config.num_envs, 
                        env_type="gym",
                        episodic_life=True, 
                        reward_clip=True, 
                        seed=config.seed, 
                        repeat_action_probability=0.25,)
    envs.num_envs = config.num_envs
    envs.single_action_space = envs.action_space
    envs.single_observation_space = envs.observation_space
    envs = RecordEpisodeStatistics(envs)

    C, H, W = space_chw(envs.observation_space.shape)

    rnd = RandomNetworkDistillation(in_channels=1, image_shape=(84, 84)).to(device)
    agent = ActorCriticCNN((C, H, W), envs.action_space.n, FiLM=False).to(device)
    
    params = list(agent.parameters()) + list(rnd.predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate, eps=1e-5)
    run_name = f"RND_PPO_{config.env_id}_{config.seed}_{int(time.time())}"
    logger = Logger(LoggerConfig(log_dir=config.log_dir, run_name=run_name))

    # Obs/reward normalization (following CLeanRL and orginal RND implementation)
    obs_rms = RunningMeanStd(shape=(1, 1, H, W))
    rew_rms = RunningMeanStd()
    rew_filter = RewardForwardFilter(config.gamma)

    # Bootstrap obs_rms with random actions
    obs = envs.reset()
    obs_batch = []
    for _ in range(config.num_steps * 50):
        act = np.random.randint(0, envs.action_space.n, size=(config.num_envs,))
        obs, rew, done, _ = envs.step(act)
        obs_t = torch.as_tensor(obs)
        last = last_frame_nchw(obs_t).cpu().numpy() # [N,1,H,W]
        obs_batch.append(last)
        if len(obs_batch) > config.num_envs * config.num_steps:
            obs_rms.update(np.concatenate(obs_batch, axis=0))
            obs_batch = []

    # Rollout buffers
    T, N = config.num_steps, config.num_envs
    obs_buff = torch.empty((T, N, C, H, W), dtype=torch.float32, device=device)
    act_buff = torch.empty((T, N), dtype=torch.int64, device=device)
    logp_buff = torch.empty((T, N), dtype=torch.float32, device=device)
    ext_val_buff = torch.empty((T, N), dtype=torch.float32, device=device)
    int_val_buff = torch.empty((T, N), dtype=torch.float32, device=device)
    rew_buff = torch.empty((T, N), dtype=torch.float32, device=device)
    rew_curiosity_buff = torch.empty((T, N), dtype=torch.float32, device=device)
    done_buff = torch.empty((T, N), dtype=torch.bool, device=device)
    avg_returns = deque(maxlen=20)

    # Training loop
    steps = 0
    obs = envs.reset()
    obs_t = torch.as_tensor(obs, device=device, dtype=torch.float32)
    obs_t = ensure_nchw(obs_t)
    done_t = torch.zeros(N, dtype=torch.bool, device=device)

    for itr in range(config.num_iterations):
        frac = 1.0 - (itr - 1.0) / config.num_iterations
        lrnow = frac * config.learning_rate
        optimizer.param_groups[0]["lr"] = lrnow

        for t in range(T):
            obs_buff[t] = obs_t
            done_buff[t] = done_t
            with torch.no_grad():
                action_t, logp_t, _, int_val_t, ext_val_t = agent.act(obs_t)
                ext_val_buff[t] = ext_val_t
                int_val_buff[t] = int_val_t

            act_buff[t] = action_t
            logp_buff[t] = logp_t

            obs, reward, done, info = envs.step(action_t.cpu().numpy())
            rew_buff[t] = torch.as_tensor(reward, dtype=torch.float32, device=device)
            obs_t = torch.as_tensor(obs, dtype=torch.float32, device=device)
            obs_t = ensure_nchw(obs_t)
            done_t = torch.as_tensor(done, dtype=torch.bool, device=device)

            # RND uses normalized last frame
            last = last_frame_nchw(obs_t)  # [N,1,H,W]
            obs_mean = torch.as_tensor(obs_rms.mean, dtype=last.dtype, device=device)
            obs_var = torch.as_tensor(obs_rms.var, dtype=last.dtype, device=device)
            norm_last = torch.clamp((last - obs_mean) / torch.sqrt(obs_var + 1e-8), -5.0, 5.0)

            # Compute curiosity(intrinsic) reward
            pred_feat_t, target_feat_t = rnd(norm_last)
            rnd_loss_t = (pred_feat_t - target_feat_t).pow(2).mean(1)
            rew_curiosity_raw = rnd_loss_t.detach()

            # Normalize intrinsic reward (no normalization on the extrinsic reward)
            rew_discounted = rew_filter.update(rew_curiosity_raw.cpu().numpy())
            rew_rms.update_from_moments(
                np.mean(rew_discounted),
                np.var(rew_discounted),
                rew_discounted.size,
            )
            rew_curiosity_norm = torch.as_tensor(
                rew_discounted / np.sqrt(rew_rms.var + 1e-8),
                dtype=torch.float32,
                device=device,
            )
            rew_curiosity_buff[t] = rew_curiosity_norm

            for idx, d in enumerate(done):
                if d and info["lives"][idx] == 0:
                    avg_returns.append(info["r"][idx])
                    epi_ret = np.average(avg_returns)
                    print(f"Iteration {itr}, Env {idx}, Episode Return: {info['r'][idx]}, Average Return: {epi_ret}")
                    logger.log({
                        "train/episode_return": info["r"][idx],
                        "train/episode_length": info["l"][idx],
                        "train/average_return": epi_ret,
                        "train/curiosity_reward": rew_curiosity_raw[idx].item(),
                    }, step=steps)

            steps += N


        # GAE
        with torch.no_grad():
            last_int_val, last_ext_val = agent.value(obs_t)
            int_adv_buff = torch.zeros_like(rew_buff)
            ext_adv_buff = torch.zeros_like(rew_curiosity_buff)
            int_last_gae_lam = 0.0
            ext_last_gae_lam = 0.0

            for t in reversed(range(T)):
                if t == T - 1:
                    int_next_nonterminal = 1.0 - done_t.float()
                    int_next_value = last_int_val
                    ext_next_nonterminal = 1.0 - done_t.float()
                    ext_next_value = last_ext_val
                else:
                    int_next_nonterminal = 1.0 - done_buff[t + 1].float()
                    int_next_value = int_val_buff[t + 1]
                    ext_next_nonterminal = 1.0 - done_buff[t + 1].float()
                    ext_next_value = ext_val_buff[t + 1]

                int_delta = rew_curiosity_buff[t] + config.gamma * int_next_value * int_next_nonterminal - int_val_buff[t]
                int_adv_buff[t] = int_last_gae_lam = int_delta + config.gamma * config.gae_lambda * int_next_nonterminal * int_last_gae_lam

                ext_delta = rew_buff[t] + config.gamma * ext_next_value * ext_next_nonterminal - ext_val_buff[t]
                ext_adv_buff[t] = ext_last_gae_lam = ext_delta + config.gamma * config.gae_lambda * ext_next_nonterminal * ext_last_gae_lam
            
            int_ret_buff = int_adv_buff + int_val_buff
            ext_ret_buff = ext_adv_buff + ext_val_buff
        
        b_obs = obs_buff.reshape(T * N, C, H, W)
        b_act = act_buff.reshape(T * N)
        b_logp = logp_buff.reshape(T * N)
        b_int_val = int_val_buff.reshape(T * N)
        b_ext_val = ext_val_buff.reshape(T * N)
        b_int_ret = int_ret_buff.reshape(T * N)
        b_ext_ret = ext_ret_buff.reshape(T * N)
        b_int_adv = int_adv_buff.reshape(T * N)
        b_ext_adv = ext_adv_buff.reshape(T * N)
        b_adv = 1.0 * b_int_adv + 2.0 * b_ext_adv # default weights for intrinsic/extrinsic advantage

        # Update observation normalization statistics
        last_batch = last_frame_nchw(b_obs).cpu().numpy()
        obs_rms.update(last_batch)

        rnd_obs_batch = torch.as_tensor(last_batch, dtype=torch.float32, device=device)
        obs_mean = torch.as_tensor(obs_rms.mean, dtype=rnd_obs_batch.dtype, device=device)
        obs_var = torch.as_tensor(obs_rms.var, dtype=rnd_obs_batch.dtype, device=device)
        rnd_obs_batch = torch.clamp((rnd_obs_batch - obs_mean) / torch.sqrt(obs_var + 1e-8), -5.0, 5.0)

        # Optimize policy and value network
        b_inds = np.arange(T * N)
        for epoch in range(epochs):
            np.random.shuffle(b_inds)
            for start in range(0, T * N, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]

                rnd_pred_feat, rnd_target_feat = rnd(rnd_obs_batch[mb_inds])
                rnd_loss = torch.nn.functional.mse_loss(rnd_pred_feat, rnd_target_feat.detach(), reduction="none").mean(-1)

                # Randomly mask a proportion of RND loss to stabilize training (as in the original RND paper)
                mask = (torch.rand_like(rnd_loss) < config.rnd_update_proportion).float()
                rnd_loss = (rnd_loss * mask).sum() / torch.clamp(mask.sum(), min=1.0)

                # Standard PPO update
                _, logp, entropy, int_value, ext_value = agent.act(b_obs[mb_inds], b_act[mb_inds])
                ratio = torch.exp(logp - b_logp[mb_inds])

                with torch.no_grad():
                    kl_div = ((ratio - 1) - torch.log(ratio)).mean()

                mb_adv = b_adv[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # value losses (intrinsic value are not clipped)
                ext_v_loss_raw = (ext_value - b_ext_ret[mb_inds]) ** 2
                ext_v_clipped = b_ext_val[mb_inds] + torch.clamp(ext_value - b_ext_val[mb_inds], -config.vf_clip_coef, config.vf_clip_coef)
                ext_v_loss_clipped = (ext_v_clipped - b_ext_ret[mb_inds]) ** 2
                ext_value_loss = 0.5 * torch.max(ext_v_loss_raw, ext_v_loss_clipped).mean()

                int_v_loss = 0.5 * ((int_value - b_int_ret[mb_inds]) ** 2).mean()
                value_loss = int_v_loss + ext_value_loss

                entropy_loss = -entropy.mean()

                loss = policy_loss + config.vf_coef * value_loss + config.ent_coef * entropy_loss + rnd_loss

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(params, max_norm=0.5)
                optimizer.step()

        logger.log(
            {
                "losses/policy_loss": policy_loss.item(),
                "losses/value_loss": value_loss.item(),
                "losses/int_value_loss": int_v_loss.item(),
                "losses/ext_value_loss": ext_value_loss.item(),
                "losses/entropy_loss": entropy_loss.item(),
                "losses/rnd_loss": rnd_loss.item(),
                "losses/kl_div": kl_div.item(),
                "debug/lr": optimizer.param_groups[0]["lr"],
            },
            step=steps,
        )

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config, RNDConfig) if args.config else RNDConfig()
    train(config)


if __name__ == "__main__": 
    main()
