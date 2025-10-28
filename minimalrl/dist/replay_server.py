"""Simple shared replay buffer server for multiprocessing setups."""

from __future__ import annotations

from multiprocessing import Manager
from typing import Any, List, Tuple


class ReplayServer:
    """Minimal Manager-backed replay server."""

    def __init__(self, capacity: int):
        self.manager = Manager()
        self.buffer: List[Tuple[Any, ...]] = self.manager.list()  # type: ignore[assignment]
        self.capacity = capacity

    def push(self, transition: Tuple[Any, ...]) -> None:
        if len(self.buffer) >= self.capacity:
            self.buffer.pop(0)
        self.buffer.append(transition)

    def sample(self, batch_size: int) -> List[Tuple[Any, ...]]:
        import random

        return random.sample(list(self.buffer), batch_size)
