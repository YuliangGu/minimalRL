"""Training loop skeleton for student-teacher style distillation."""

from __future__ import annotations

from dataclasses import dataclass

from minimalrl.core.config import ExperimentConfig
from minimalrl.core.logger import Logger, LoggerConfig
from minimalrl.core.torch_utils import seed_all

from minimalrl.algos.student_teacher import StudentTeacherBundle, build_student_teacher_bundle


@dataclass
class StudentTeacherConfig(ExperimentConfig):
    """Configuration stub for student-teacher experiments."""

    teacher_path: str | None = None
    student_arch: str = "mlp"
    distillation_objective: str = "kl"
    warmup_steps: int = 1_000
    total_updates: int = 50_000
    batch_size: int = 256
    log_interval: int = 1_000


class StudentTeacherTrainer:
    """High-level orchestration skeleton for distillation experiments."""

    def __init__(self, config: StudentTeacherConfig):
        self.config = config
        self.logger = Logger(LoggerConfig(log_dir=config.log_dir, enable_tensorboard=True))
        self.bundle: StudentTeacherBundle | None = None

    def setup(self) -> None:
        """Instantiate teacher and student models."""

        self.bundle = build_student_teacher_bundle({"config": self.config})

    def warmup_teacher(self) -> None:
        """Optional phase for running the teacher before distillation starts."""

        raise NotImplementedError("Teacher warmup phase is pending implementation.")

    def collect_batch(self) -> dict[str, None]:
        """Placeholder for collecting a batch of targets from the teacher."""

        raise NotImplementedError("Batch collection logic is pending implementation.")

    def distill_step(self, batch: dict[str, None]) -> None:
        """Placeholder for updating the student with teacher supervision."""

        raise NotImplementedError("Student update logic is pending implementation.")

    def maybe_evaluate(self, step: int) -> None:
        """Optional evaluation hook for tracking student progress."""

        raise NotImplementedError("Evaluation routine is pending implementation.")

    def train(self) -> None:
        """Main training driver."""

        raise NotImplementedError("Full student-teacher training loop is pending implementation.")


def train(config: StudentTeacherConfig) -> None:
    """Entry-point for launching student-teacher distillation."""

    seed_all(config.seed)
    trainer = StudentTeacherTrainer(config)
    trainer.setup()
    trainer.train()
