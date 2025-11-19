#!/usr/bin/env python
"""Starter script for exporting minimalRL actors to TorchScript and/or ONNX."""

from __future__ import annotations

import argparse
import pathlib
from dataclasses import dataclass
from typing import Sequence, Tuple

import torch

from minimalrl.algos.ppo.actor_critic import ActorCritic as PPOActorCritic
from minimalrl.algos.sac.sac import ActorCriticSAC


class DeterministicPolicyWrapper(torch.nn.Module):
    """Wraps a stochastic actor to provide a deterministic forward pass for export."""

    def __init__(self, module: torch.nn.Module, algo: str) -> None:
        super().__init__()
        self.module = module
        self.algo = algo

    def forward(self, obs: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        if self.algo == "sac":
            action, _ = self.module(obs, deterministic=True)
            return action
        if self.algo == "ppo":
            return self.module.actor_mean(obs)            # PPO actor is Gaussian; we expose the mean action for deployment.
        raise ValueError(f"Unsupported algo wrapper: {self.algo}")


@dataclass
class ExportConfig:
    algo: str
    state_dict: pathlib.Path
    obs_dim: int
    action_dim: int
    hidden_sizes: Tuple[int, ...]
    action_scale: float
    init_temperature: float
    torchscript_out: pathlib.Path | None
    onnx_out: pathlib.Path | None
    verify: bool
    device: torch.device


def build_actor(cfg: ExportConfig) -> torch.nn.Module:
    if cfg.algo == "sac":
        agent = ActorCriticSAC(
            cfg.obs_dim,
            cfg.action_dim,
            cfg.hidden_sizes,
            action_scale=cfg.action_scale,
            temperature=cfg.init_temperature,
        )
        actor = agent.actor
    elif cfg.algo == "ppo":
        actor = PPOActorCritic(cfg.obs_dim, cfg.action_dim, cfg.hidden_sizes)
    else:
        raise ValueError("algo must be 'sac' or 'ppo'")

    state = torch.load(cfg.state_dict, map_location="cpu")
    actor.load_state_dict(state, strict=False)
    return DeterministicPolicyWrapper(actor, cfg.algo)


def export_torchscript(module: torch.nn.Module, cfg: ExportConfig, example: torch.Tensor) -> None:
    traced = torch.jit.trace(module, example)
    traced.save(str(cfg.torchscript_out))
    print(f"[torchscript] saved to {cfg.torchscript_out}")


def export_onnx(module: torch.nn.Module, cfg: ExportConfig, example: torch.Tensor) -> None:
    torch.onnx.export(
        module,
        example,
        str(cfg.onnx_out),
        input_names=["obs"],
        output_names=["action"],
        dynamic_axes={"obs": {0: "batch"}, "action": {0: "batch"}},
        opset_version=17,
    )
    print(f"[onnx] saved to {cfg.onnx_out}")

    if cfg.verify:
        import onnxruntime as ort

        session = ort.InferenceSession(str(cfg.onnx_out))
        ref = module(example).detach().cpu().numpy()
        ort_out = session.run([session.get_outputs()[0].name], {session.get_inputs()[0].name: example.cpu().numpy()})[0]
        max_err = float(abs(ref - ort_out).max())
        print(f"[onnx] verification max error: {max_err:.3e}")


def parse_hidden_sizes(values: Sequence[str]) -> Tuple[int, ...]:
    if not values:
        raise ValueError("--hidden-sizes requires at least one value")
    return tuple(int(v) for v in values)


def parse_args() -> ExportConfig:
    parser = argparse.ArgumentParser(description="Export policies for deployment")
    parser.add_argument("--algo", choices=["sac", "ppo"], required=True)
    parser.add_argument("--state-dict", type=pathlib.Path, required=True, help="Path to actor state_dict (.pt)")
    parser.add_argument("--obs-dim", type=int, required=True)
    parser.add_argument("--action-dim", type=int, required=True)
    parser.add_argument("--hidden-sizes", nargs="+", default=[256, 256], help="Hidden layer sizes")
    parser.add_argument("--action-scale", type=float, default=1.0, help="Only used for SAC actors")
    parser.add_argument("--init-temperature", type=float, default=0.2, help="Initial alpha for SAC actor construction")
    parser.add_argument("--torchscript-out", type=pathlib.Path, help="Output path for TorchScript artifact")
    parser.add_argument("--onnx-out", type=pathlib.Path, help="Output path for ONNX artifact")
    parser.add_argument("--verify", action="store_true", help="Run ONNX Runtime parity check")
    parser.add_argument("--device", default="cpu", help="Device used during tracing/export")
    args = parser.parse_args()

    if not args.torchscript_out and not args.onnx_out:
        parser.error("Specify at least one of --torchscript-out or --onnx-out")

    return ExportConfig(
        algo=args.algo,
        state_dict=args.state_dict,
        obs_dim=args.obs_dim,
        action_dim=args.action_dim,
        hidden_sizes=parse_hidden_sizes(args.hidden_sizes),
        action_scale=args.action_scale,
        init_temperature=max(args.init_temperature, 1e-6),
        torchscript_out=args.torchscript_out,
        onnx_out=args.onnx_out,
        verify=bool(args.verify),
        device=torch.device(args.device),
    )


def main() -> None:
    cfg = parse_args()
    module = build_actor(cfg).to(cfg.device)
    module.eval()
    example = torch.zeros(1, cfg.obs_dim, dtype=torch.float32, device=cfg.device)

    if cfg.torchscript_out:
        export_torchscript(module, cfg, example)
    if cfg.onnx_out:
        export_onnx(module.cpu(), cfg, example.cpu())


if __name__ == "__main__":
    main()
