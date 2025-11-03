"""Lightweight logging utilities supporting stdout, JSONL, and optional W&B."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

try:
    import wandb  # type: ignore[import]
    _wandb_import_error = None
except Exception as exc:  # pragma: no cover - optional dependency
    wandb = None
    _wandb_import_error = exc

try:
    from torch.utils.tensorboard import SummaryWriter
except ImportError:  # pragma: no cover - optional dependency
    SummaryWriter = None  # type: ignore[misc]


@dataclass
class LoggerConfig:
    run_name: Optional[str] = None
    log_dir: Path = Path("runs")
    project: Optional[str] = None
    group: Optional[str] = None
    enable_tensorboard: bool = True
    enable_wandb: bool = False


class Logger:
    def __init__(self, config: LoggerConfig):
        self.config = config
        if self.config.run_name is not None:
            self.config.log_dir = self.config.log_dir / self.config.run_name
        self.config.log_dir.mkdir(parents=True, exist_ok=True)
        self._tb = None
        self._wandb_run = None

        if self.config.enable_tensorboard:
            if SummaryWriter is None:
                raise RuntimeError(
                    "TensorBoard logging requested but unavailable. Install with `pip install tensorboard`."
                )
            self._tb = SummaryWriter(self.config.log_dir.as_posix())

        if self.config.enable_wandb:
            if wandb is None:
                message = "wandb extra not installed. Install with `pip install .[wandb]`."
                if _wandb_import_error is not None:
                    message = f"{message} (original error: {_wandb_import_error})"
                raise RuntimeError(message)
            self._wandb_run = wandb.init( 
                project=self.config.project,
                group=self.config.group,
                dir=self.config.log_dir,
                reinit=True,
            )

    def log(self, metrics: Dict[str, Any], step: Optional[int] = None, silence: bool = False) -> None:
        if step is None:
            step = 0

        processed = {key: _to_scalar(value, key) for key, value in metrics.items()}

        if self._tb is not None:
            for key, value in processed.items():
                self._tb.add_scalar(key, value, global_step=step)
            self._tb.flush()

        if self._wandb_run is not None:
            self._wandb_run.log(processed, step=step)

    def close(self) -> None:
        if self._tb is not None:
            self._tb.close()
        if self._wandb_run is not None:
            self._wandb_run.finish()

    def __del__(self) -> None:  # pragma: no cover - best effort cleanup
        try:
            self.close()
        except Exception:
            pass


def _to_scalar(value: Any, key: str) -> float:
    """Convert supported value types into float for logging backends."""

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, np.generic):
        return float(value.item())

    if isinstance(value, np.ndarray):
        if value.size != 1:
            raise ValueError(f"Metric '{key}' expects a scalar, received array with shape {value.shape}.")
        return float(value.item())

    if isinstance(value, torch.Tensor):
        if value.numel() != 1:
            raise ValueError(f"Metric '{key}' expects a scalar, received tensor with shape {tuple(value.shape)}.")
        return float(value.detach().cpu().item())

    raise TypeError(f"Unsupported metric type for '{key}': {type(value).__name__}.")
