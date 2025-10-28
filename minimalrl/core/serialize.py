"""Serialization helpers for checkpoints."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import torch


@dataclass
class Checkpoint:
    policy: Dict[str, Any]
    optimizer: Dict[str, Any]
    replay: Optional[Dict[str, Any]] = None
    config: Optional[Dict[str, Any]] = None


def save_checkpoint(path: Path, checkpoint: Checkpoint) -> None:
    """Persist a checkpoint dictionary to disk."""

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint.__dict__, path)


def load_checkpoint(path: Path) -> Checkpoint:
    """Load a checkpoint from disk."""

    data = torch.load(path, map_location="cpu")
    return Checkpoint(**data)
