"""Dataset and expert helpers for DAgger-style behavior cloning."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

class DAggerDataset:
    """Dataset for DAgger-style behavior cloning."""

    def __init__(
        self,
        *,
        discrete_actions: bool = False,
        device: str | torch.device | None = None,
    ) -> None:
        self.discrete_actions = discrete_actions
        self.device = torch.device(device or "cpu")
        self._obs_storage: torch.Tensor | None = None
        self._action_storage: torch.Tensor | None = None
        self._obs_shape: tuple[int, ...] | None = None
        self._action_shape: tuple[int, ...] | None = None
        self._size = 0
        self._capacity = 0

    def __len__(self) -> int:
        return self._size

    def add(self, obs: np.ndarray | torch.Tensor, action: np.ndarray | torch.Tensor) -> None:
        obs_tensor = self._coerce_obs(obs, batched=False)
        action_tensor = self._coerce_action(action, batched=False)
        self._store_batch(obs_tensor, action_tensor)

    def extend(
        self,
        obs_batch: np.ndarray | torch.Tensor | list[Any],
        action_batch: np.ndarray | torch.Tensor | list[Any],
    ) -> None:
        obs_tensor = self._coerce_obs(obs_batch, batched=True)
        action_tensor = self._coerce_action(action_batch, batched=True)
        if obs_tensor.shape[0] == 0:
            return
        self._store_batch(obs_tensor, action_tensor)

    def sample_batch(
        self,
        batch_size: int,
        *,
        device: str | torch.device | None = None,
    ) -> Dict[str, torch.Tensor]:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive.")
        if len(self) < batch_size:
            raise ValueError("Not enough samples in DAggerDataset to draw a batch.")
        target_device = torch.device(device or self.device)
        idxs = torch.randint(0, self._size, size=(batch_size,), device="cpu")
        obs = self._obs_storage[idxs].to(target_device)
        actions = self._action_storage[idxs].to(target_device)
        return {"obs": obs, "action": actions}

    def _store_batch(self, obs_tensor: torch.Tensor, action_tensor: torch.Tensor) -> None:
        if obs_tensor.shape[0] != action_tensor.shape[0]:
            raise ValueError("Observation and action batches must be the same length.")
        if self._obs_storage is None:
            self._initialize_storage(obs_tensor, action_tensor)
        else:
            self._validate_shapes(obs_tensor, action_tensor)
        needed = self._size + obs_tensor.shape[0]
        self._ensure_capacity(needed)
        end = self._size + obs_tensor.shape[0]
        self._obs_storage[self._size:end].copy_(obs_tensor)
        reshaped_actions = self._reshape_actions(action_tensor)
        self._action_storage[self._size:end].copy_(reshaped_actions)
        self._size = end
    
    def _coerce_obs(self, obs: Any, *, batched: bool) -> torch.Tensor:
        """ Coerce observation input to a torch.Tensor. """
        obs_tensor = torch.as_tensor(obs, dtype=torch.float32)
        if not batched:
            obs_tensor = obs_tensor.unsqueeze(0)
        return obs_tensor.contiguous()

    def _coerce_action(self, action: Any, *, batched: bool) -> torch.Tensor:
        action_tensor = torch.as_tensor(action, dtype=self._action_dtype)
        if action_tensor.ndim == 0:
            action_tensor = action_tensor.unsqueeze(0)
        if not batched:
            action_tensor = action_tensor.unsqueeze(0)
        if not self.discrete_actions and action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(-1)
        return action_tensor.contiguous()

    @property
    def _action_dtype(self) -> torch.dtype:
        return torch.long if self.discrete_actions else torch.float32

    def _initialize_storage(self, obs_tensor: torch.Tensor, action_tensor: torch.Tensor) -> None:
        self._obs_shape = tuple(obs_tensor.shape[1:])
        self._action_shape = tuple(action_tensor.shape[1:])
        initial_capacity = max(obs_tensor.shape[0], 1024)
        obs_dims = (initial_capacity,) + self._obs_shape
        action_dims = (initial_capacity,) + self._action_shape
        self._obs_storage = torch.empty(obs_dims, dtype=torch.float32)
        self._action_storage = torch.empty(action_dims, dtype=self._action_dtype)
        self._capacity = initial_capacity

    def _ensure_capacity(self, required: int) -> None:
        if required <= self._capacity:
            return
        new_capacity = max(required, max(1, self._capacity * 2))
        obs_shape = (new_capacity,) + self._obs_shape
        action_shape = (new_capacity,) + self._action_shape
        new_obs = torch.empty(obs_shape, dtype=self._obs_storage.dtype)
        new_actions = torch.empty(action_shape, dtype=self._action_storage.dtype)
        if self._size:
            new_obs[: self._size] = self._obs_storage[: self._size]
            new_actions[: self._size] = self._action_storage[: self._size]
        self._obs_storage = new_obs
        self._action_storage = new_actions
        self._capacity = new_capacity

    def _validate_shapes(self, obs_tensor: torch.Tensor, action_tensor: torch.Tensor) -> None:
        if obs_tensor.shape[1:] != self._obs_shape:
            raise ValueError("Observation shape mismatch for dataset append.")
        if action_tensor.shape[1:] != self._action_shape:
            raise ValueError("Action shape mismatch for dataset append.")

    def _reshape_actions(self, action_tensor: torch.Tensor) -> torch.Tensor:
        if self._action_shape:
            return action_tensor.view(-1, *self._action_shape)
        return action_tensor.view(-1)

    def _reset_storage(self) -> None:
        self._obs_storage = None
        self._action_storage = None
        self._obs_shape = None
        self._action_shape = None
        self._size = 0
        self._capacity = 0

    @staticmethod
    def _extract_first(data: Dict[str, Any], keys: tuple[str, ...]) -> Any | None:
        for key in keys:
            if key in data:
                return data[key]
        return None

