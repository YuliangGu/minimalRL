"""Utility helpers for reproducible PyTorch training loops."""

from __future__ import annotations

import random
from collections.abc import Iterable, Iterator
from contextlib import contextmanager
from typing import Optional

import numpy as np
import torch


def seed_all(seed: int) -> None:
    """Seed Python, NumPy, and PyTorch RNGs."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def polyak_update(source: torch.nn.Module, target: torch.nn.Module, tau: float) -> None:
    """Perform Polyak averaging of target parameters toward source."""

    with torch.no_grad():
        for src, tgt in zip(source.parameters(), target.parameters()):
            tgt.data.mul_(1 - tau).add_(tau * src.data)


def clip_gradients(parameters: Iterable[torch.Tensor], max_norm: float) -> float:
    """Gradient clipping helper that returns the total norm."""

    return torch.nn.utils.clip_grad_norm_(parameters, max_norm)


@contextmanager
def autocast(enabled: bool = True, dtype: Optional[torch.dtype] = None) -> Iterator[None]:
    """Context manager that toggles AMP autocast."""

    if not enabled:
        yield
        return
    with torch.autocast(device_type="cuda", dtype=dtype):
        yield
