"""Evaluation helpers for deterministic rollouts and video dumps."""

from __future__ import annotations

from pathlib import Path
from typing import Callable, Optional

import gymnasium as gym
import numpy as np


def evaluate_policy(
    policy_fn: Callable[[np.ndarray], np.ndarray],
    env: gym.Env,
    episodes: int = 5,
    render: bool = False,
    video_path: Optional[Path] = None,
) -> float:
    """Run deterministic evaluation episodes and return mean reward."""

    rewards = []
    video_frames = []
    for _ in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        while not done:
            action = policy_fn(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += float(reward)
            if render:
                frame = env.render()
                if frame is not None:
                    video_frames.append(frame)
        rewards.append(total_reward)

    if video_path is not None and video_frames:
        save_video(np.array(video_frames), video_path)

    return float(np.mean(rewards))


def save_video(frames: np.ndarray, path: Path, fps: int = 30) -> None:
    """Persist RGB frames to an MP4 file using imageio if available."""

    try:
        import imageio.v2 as imageio
    except ImportError:  # pragma: no cover - optional dependency
        raise RuntimeError("imageio is required to save evaluation videos.")

    path.parent.mkdir(parents=True, exist_ok=True)
    imageio.mimwrite(path.as_posix(), frames.astype(np.uint8), fps=fps)
