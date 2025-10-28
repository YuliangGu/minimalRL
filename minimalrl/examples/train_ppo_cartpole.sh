#!/usr/bin/env bash
set -euo pipefail

if [[ "${1:-}" == "--ci" ]]; then
  python - <<'PY'
import gymnasium as gym

env = gym.make("CartPole-v1")
obs, _ = env.reset()
action = env.action_space.sample()
next_obs, reward, terminated, truncated, _ = env.step(action)
print({"obs": obs.shape, "next_obs": next_obs.shape, "reward": reward, "done": terminated or truncated})
PY
  exit 0
fi

echo "PPO training loop not implemented yet. Run: python -m minimalrl.algos.ppo.train --config configs/ppo_cartpole.yaml"
