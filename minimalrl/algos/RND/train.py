"""PPO(continous) training loop."""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import numpy as np
from collections import deque
import torch
import time
import envpool
import gym
from gym.wrappers.normalize import RunningMeanStd

from minimalrl.core.config import ExperimentConfig, load_config
from minimalrl.core.logger import Logger, LoggerConfig
from minimalrl.core.torch_utils import seed_all

from minimalrl.algos.RND.actor_critic_cnn import ActorCriticCNN, RandomNetworkDistillation

# Helper class to compute discounted reward sums (CleanRL/OpenAI Baselines style)
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
    num_iterations: int = 2000

    epsilon: float = 0.2
    gamma: float = 0.999
    gae_lambda: float = 0.95
    ent_coef: float = 0.001
    vf_coef: float = 0.5
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

    rnd = RandomNetworkDistillation(1, envs.action_space.n).to(device)
    agent = ActorCriticCNN(envs.observation_space.shape, envs.action_space.n, FiLM=False).to(device)
    params = list(agent.parameters()) + list(rnd.predictor.parameters())
    optimizer = torch.optim.Adam(params, lr=config.learning_rate, eps=1e-5)
    run_name = f"RND_PPO_{config.env_id}_{config.seed}_{int(time.time())}"
    logger = Logger(LoggerConfig(log_dir=config.log_dir, run_name=run_name))

    # Obs/reward normalization (following CLeanRL and orginal RND implementation)
    obs_rms = RunningMeanStd(shape=(1, 1, 84, 84)) 
    rew_rms = RunningMeanStd()
    rew_filter = RewardForwardFilter(config.gamma)
    obs = envs.reset()
    obs_batch = []
    for _ in range(config.num_steps * 50):
        act = np.random.randint(0, envs.action_space.n, size=(config.num_envs,))
        obs, rew, done, _ = envs.step(act)
        obs_last = obs[:, -1:, :, :].reshape(-1, 1, 84, 84)
        obs_batch += list(obs_last)

        if len(obs_batch) == config.num_envs * config.num_steps:
            obs_batch_np = np.stack(obs_batch)
            obs_rms.update(obs_batch_np)
            obs_batch = []

    # Rollout buffers
    T, N = config.num_steps, config.num_envs
    obs_buff  = torch.empty((T, N) + envs.observation_space.shape, dtype=torch.float32, device=device)
    act_buff  = torch.empty((T, N) + envs.action_space.shape, dtype=torch.float32, device=device)
    logp_buff = torch.empty((T, N), dtype=torch.float32, device=device)
    ext_val_buff = torch.empty((T, N), dtype=torch.float32, device=device)
    int_val_buff = torch.empty((T, N), dtype=torch.float32, device=device)
    rew_buff  = torch.empty((T, N), dtype=torch.float32, device=device)
    rew_curiosity_buff  = torch.empty((T, N), dtype=torch.float32, device=device)
    done_buff = torch.empty((T, N), dtype=torch.bool, device=device)
    avg_returns = deque(maxlen=20)

    # Training loop
    steps = 0
    obs = envs.reset()
    obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
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
                ext_val_buff[t] = ext_val_t.squeeze(-1)
                int_val_buff[t] = int_val_t.squeeze(-1)

            act_buff[t] = action_t
            logp_buff[t] = logp_t

            obs, reward, done, info = envs.step(action_t.cpu().numpy())
            rew_buff[t] = torch.tensor(reward, dtype=torch.float32, device=device).squeeze(-1)
            obs_t = torch.tensor(obs, dtype=torch.float32, device=device)
            done_t = torch.tensor(done, dtype=torch.bool, device=device).squeeze(-1)

            obs_slice_t = obs_t[:, -1:, :, :].reshape(N, 1, 84, 84)
            obs_slice_t = (obs_slice_t - torch.tensor(obs_rms.mean, device=device)) / torch.sqrt(torch.tensor(obs_rms.var, device=device) + 1e-8)
            obs_slice_t = torch.clamp(obs_slice_t, -5.0, 5.0).float()

            pred_feat_t, target_feat_t = rnd(obs_slice_t)
            rnd_loss_t = (pred_feat_t - target_feat_t).pow(2).mean(1)
            rew_curiosity_t = rnd_loss_t.detach()
            rew_curiosity_buff[t] = rew_curiosity_t

            for idx, d in enumerate(done):
                if d and info["lives"][idx] == 0:
                    avg_returns.append(info["r"][idx])
                    epi_ret = np.average(avg_returns)
                    print(f"Iteration {itr}, Env {idx}, Episode Return: {info['r'][idx]}, Average Return: {epi_ret}")
                    logger.log({
                        "train/episode_return": info["r"][idx],
                        "train/episode_length": info["l"][idx],
                        "train/average_return": epi_ret,
                        "train/curiosity_reward": rew_curiosity_t[idx].item(),
                    }, step=steps)

            steps += N

        # discounted return for curiosity rewards
        rew_curiosity_np = np.array([rew_filter.update(rew_curiosity_env) for rew_curiosity_env in rew_curiosity_buff.cpu().numpy().T])
        rew_mean, rew_std, rew_count = np.mean(rew_curiosity_np), np.std(rew_curiosity_np), len(rew_curiosity_np)
        rew_rms.update_from_moments(rew_mean, rew_std**2, rew_count)
        
        # normalize curiosity rewards
        rew_curiosity_buff = rew_curiosity_buff / torch.sqrt(torch.tensor(rew_rms.var, device=device) + 1e-8)


        # GAE: estimate advantage function by exponentially weighting TD-errors
        with torch.no_grad():
            last_int_val, last_ext_val = agent.value(obs_t)
            last_ext_val = last_ext_val.squeeze(-1)
            last_int_val = last_int_val.squeeze(-1)

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
        
        # Flatten the batch
        b_obs = obs_buff.reshape(T * N, *envs.observation_space.shape)
        b_act = act_buff.reshape(T * N, *envs.action_space.shape)
        b_logp = logp_buff.reshape(T * N)
        b_int_val = int_val_buff.reshape(T * N)
        b_ext_val = ext_val_buff.reshape(T * N)
        b_int_ret = int_ret_buff.reshape(T * N)
        b_ext_ret = ext_ret_buff.reshape(T * N)
        b_int_adv = int_adv_buff.reshape(T * N)
        b_ext_adv = ext_adv_buff.reshape(T * N)
        b_adv = 0.5 * b_int_adv + 2.0 * b_ext_adv

        obs_rms.update(b_obs[:, -1:, :, :].reshape(-1, 1, 84, 84).cpu().numpy())

        # fetch minibatches and update policy
        rnd_obs_batch = b_obs[:, -1:, :, :].reshape(-1, 1, 84, 84)
        rnd_obs_batch = (rnd_obs_batch - torch.tensor(obs_rms.mean, device=device)) / torch.sqrt(torch.tensor(obs_rms.var, device=device) + 1e-8)
        rnd_obs_batch = torch.clamp(rnd_obs_batch, -5.0, 5.0).float()

        b_inds = np.arange(T * N)
        for epoch in range(epochs):
            np.random.shuffle(b_inds)
            for start in range(0, T * N, batch_size):
                end = start + batch_size
                mb_inds = b_inds[start:end]

                rnd_pred_feat, rnd_target_feat = rnd(rnd_obs_batch[mb_inds])
                rnd_loss = torch.nn.functional.mse_loss(rnd_pred_feat, rnd_target_feat.detach(), reduction="none").mean(-1) # per-sample loss

                # "dropout" for RND loss (update_proportion in the original paper)
                dropout_mask = (torch.rand_like(rnd_loss) < config.rnd_update_proportion).float()
                rnd_loss = (rnd_loss * dropout_mask).sum() / torch.max(dropout_mask.sum(), torch.tensor(1.0, device=device)) # avoid div-by-zero

                # PPO
                _, logp, entropy, int_value, ext_value = agent.act(b_obs[mb_inds], b_act.long()[mb_inds]) # use long because of Categorical
                ratio = torch.exp(logp - b_logp[mb_inds]) 

                # kl approximation
                with torch.no_grad():
                    kl_div = ((ratio - 1) - torch.log(ratio)).mean()

                mb_adv = b_adv[mb_inds]
                mb_adv = (mb_adv - mb_adv.mean()) / (mb_adv.std() + 1e-8)

                # pg loss
                surr1 = ratio * mb_adv
                surr2 = torch.clamp(ratio, 1.0 - epsilon, 1.0 + epsilon) * mb_adv
                policy_loss = -torch.min(surr1, surr2).mean()

                # value loss
                int_value, ext_value = int_value.squeeze(-1), ext_value.squeeze(-1)

                # extrinsic value loss
                ext_v_loss_raw = (ext_value - b_ext_ret[mb_inds]) ** 2
                ext_v_clipped = b_ext_val[mb_inds] + torch.clamp(ext_value - b_ext_val[mb_inds], -epsilon, epsilon)
                ext_v_loss_clipped = (ext_v_clipped - b_ext_ret[mb_inds]) ** 2
                ext_value_loss = 0.5 * torch.max(ext_v_loss_raw, ext_v_loss_clipped).mean()

                # intrinsic value loss
                int_v_loss = 0.5 * ((int_value - b_int_ret[mb_inds]) ** 2).mean() # no clipping for intrinsic value
                value_loss = int_v_loss + ext_value_loss

                # entropy loss
                entropy_loss = -entropy.mean()

                loss = policy_loss + config.vf_coef * value_loss + config.ent_coef * entropy_loss + 0.5 * rnd_loss
            
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(agent.parameters(), max_norm=0.5)
                optimizer.step()

        logger.log({
            "losses/policy_loss": policy_loss.item(),
            "losses/value_loss": value_loss.item(),
            "losses/int_value_loss": int_v_loss.item(),
            "losses/ext_value_loss": ext_value_loss.item(),
            "losses/entropy_loss": entropy_loss.item(),
            "losses/rnd_loss": rnd_loss.item(),
            "losses/kl_div": kl_div.item(),
        }, step=steps)

        # DEBUG
        logger.log({
            "debug/lr": optimizer.param_groups[0]["lr"],
        }, step=steps)

def main() -> None:
    parser = argparse.ArgumentParser(description="Train a PPO agent.")
    parser.add_argument("--config", type=str, default=None, help="Path to YAML config file.")
    args = parser.parse_args()

    config = load_config(args.config, RNDConfig) if args.config else RNDConfig()
    train(config)


if __name__ == "__main__": 
    main()
