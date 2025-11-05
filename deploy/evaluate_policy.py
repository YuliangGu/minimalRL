#!/usr/bin/env python
"""Utility script for running exported policies in gymnasium environments.

Supports TorchScript (`.pt` / `.pth`) and ONNX artifacts out of the box.
Fill in additional adapters as you expand to other deployment targets.
"""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Protocol

import gymnasium as gym
import numpy as np
import torch

try:
    import onnxruntime as ort
except ImportError:  # optional dependency
    ort = None


class PolicyRunner(Protocol):
    def __call__(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        ...


class TorchScriptPolicy:
    """Thin wrapper that feeds numpy observations into a TorchScript module."""

    def __init__(self, artifact_path: pathlib.Path, device: torch.device) -> None:
        self.device = device
        self.module = torch.jit.load(str(artifact_path), map_location=device)
        self.module.eval()

    def __call__(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device).unsqueeze(0)
        with torch.no_grad():
            action = self.module(obs_t)
        return action.squeeze(0).cpu().numpy()


class ONNXPolicy:
    """Simple ONNX Runtime inference wrapper."""

    def __init__(self, artifact_path: pathlib.Path, providers: list[str] | None = None) -> None:
        if ort is None:
            raise ImportError("onnxruntime is required to load ONNX artifacts")
        self.session = ort.InferenceSession(str(artifact_path), providers=providers or ort.get_available_providers())
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

    def __call__(self, obs: np.ndarray, deterministic: bool = True) -> np.ndarray:
        obs_np = np.asarray(obs, dtype=np.float32)[None, :]
        outputs = self.session.run([self.output_name], {self.input_name: obs_np})
        return np.asarray(outputs[0])[0]


@dataclass
class EvalConfig:
    artifact_path: pathlib.Path
    env_id: str
    num_episodes: int
    device: torch.device
    render: bool
    seed: int


def build_policy(cfg: EvalConfig) -> PolicyRunner:
    suffix = cfg.artifact_path.suffix.lower()
    if suffix in {".pt", ".pth"}:
        return TorchScriptPolicy(cfg.artifact_path, cfg.device)
    if suffix == ".onnx":
        return ONNXPolicy(cfg.artifact_path)
    raise ValueError(f"Unsupported artifact extension: {cfg.artifact_path}")


def evaluate(cfg: EvalConfig) -> dict[str, float]:
    env = gym.make(cfg.env_id)
    metrics: list[float] = []
    policy = build_policy(cfg)

    for episode in range(cfg.num_episodes):
        obs, _ = env.reset(seed=cfg.seed + episode)
        done = False
        ret = 0.0
        steps = 0
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, _ = env.step(action)
            ret += float(reward)
            steps += 1
            done = bool(terminated or truncated)
            if cfg.render:
                env.render()
        metrics.append(ret)
        print(f"Episode {episode:03d} | return={ret:.2f} | steps={steps}")

    env.close()
    returns = np.asarray(metrics, dtype=np.float32)
    summary = {"mean_return": float(returns.mean()), "std_return": float(returns.std(ddof=1) if len(returns) > 1 else 0.0)}
    print("\nSummary:", summary)
    return summary


def parse_args() -> EvalConfig:
    parser = argparse.ArgumentParser(description="Evaluate exported policies")
    parser.add_argument("artifact", type=pathlib.Path, help="Path to TorchScript (.pt) or ONNX (.onnx) file")
    parser.add_argument("--env-id", default="HalfCheetah-v4", help="Gymnasium environment id")
    parser.add_argument("--episodes", type=int, default=10, help="Number of evaluation episodes")
    parser.add_argument("--seed", type=int, default=0, help="Base random seed")
    parser.add_argument("--render", action="store_true", help="Render the environment while evaluating")
    parser.add_argument("--device", default="cpu", help="Torch device for TorchScript artifacts")
    args = parser.parse_args()

    return EvalConfig(
        artifact_path=args.artifact,
        env_id=args.env_id,
        num_episodes=args.episodes,
        device=torch.device(args.device),
        render=bool(args.render),
        seed=args.seed,
    )


def main() -> None:
    cfg = parse_args()
    evaluate(cfg)


if __name__ == "__main__":
    main()
