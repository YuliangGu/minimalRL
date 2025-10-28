"""Skeleton scaffolding for student-teacher style algorithms."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

import torch


class TeacherPolicy(Protocol):
    """Interface definition for teacher policies."""

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        ...

    def distillation_payload(self, obs: torch.Tensor) -> dict[str, torch.Tensor]:
        ...


class StudentPolicy(Protocol):
    """Interface definition for student policies."""

    def act(self, obs: torch.Tensor) -> torch.Tensor:
        ...

    def update_from_teacher(self, payload: dict[str, torch.Tensor]) -> torch.Tensor:
        ...


@dataclass
class StudentTeacherBundle:
    """Container for paired policies and shared training state."""

    teacher: TeacherPolicy
    student: StudentPolicy
    metadata: dict[str, Any] | None = None


def build_student_teacher_bundle(config: dict[str, Any]) -> StudentTeacherBundle:
    """Factory placeholder for creating the paired policies."""

    raise NotImplementedError("Implement student-teacher bundle construction.")


__all__ = [
    "TeacherPolicy",
    "StudentPolicy",
    "StudentTeacherBundle",
    "build_student_teacher_bundle",
]
