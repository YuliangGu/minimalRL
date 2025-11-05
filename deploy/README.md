# Delocy Deployment Practice

This space collects quick-start recipes for turning trained minimalRL policies into artifacts that can be evaluated and shipped. Use it as a checklist when you need to validate a new algorithm outside the training loop.

## What you can do here
- **Offline evaluation**: run deterministic rollouts from saved checkpoints and log stats in a reproducible way.
- **Export artifacts**: trace/script actors to TorchScript or export to ONNX for consumption in non-Python services.
- **Integration hooks**: sketch adapters for downstream simulators or robots without touching the core algos package.

## Files in this folder
- `evaluate_policy.py`: CLI entry point that loads a checkpoint and runs evaluation episodes.
- `export_policy.py`: helpers for producing TorchScript/ONNX payloads plus a quick verification pass.
- `notes/` (create as needed): keep environment-specific adapters, ROS launch snippets, etc.

## Suggested workflow
1. Train a policy with PPO/SAC and save checkpoints via `torch.save` (actor + config).
2. Use `export_policy.py` to serialize the actor once you are happy with training curves. Always validate exported models with the built-in parity checks.
3. Run `evaluate_policy.py` to produce deterministic metrics, rollout videos, or cached trajectories that downstream consumers can trust.
4. Package the resulting weights + config hash in your deployment repo (Docker image, ROS workspace, etc.).

## Tips
- Save normalization stats from env wrappers (observation/reward running means) alongside the model; deployment quality hinges on matching preprocessing.
- Keep target entropy, action scaling, and observation clipping logic together with the exported model if the runtime will not use the minimalRL wrappers.
- Prefer TorchScript for PyTorch-only stacks; reach for ONNX when another runtime (ORT, TensorRT) needs the policy.
- Automate smoke tests: one rollout in the gym env plus an ONNX Runtime inference comparison already catches most export issues.

Feel free to extend this folder with additional scripts (e.g., ROS bridge, Unity binding, Mujoco playback). Just keep them small and well-documented so they remain a reference playbook.
