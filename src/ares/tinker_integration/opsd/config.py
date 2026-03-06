"""OPSD-specific training configuration.

Extends ``TrainingConfig`` with parameters for the phasic training loop:
evaluation, self-reflection, teacher re-attempts, and reverse-KL distillation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ares.tinker_integration import config as config_mod


@dataclass
class OPSDConfig(config_mod.TrainingConfig):
    """Configuration for On-Policy Self-Distillation training.

    Inherits all fields from ``TrainingConfig`` and adds OPSD-specific controls.

    The main loop runs ``num_batches`` RL batches total.  Every ``opsd_every``
    batches, the OPSD phases fire:

    1. Identify hard tasks from the RL batch (0% success — no separate eval).
    2. Self-reflect on failed traces to generate privileged information.
    3. Teacher (same model + privileged context) re-attempts hard tasks.
    4. Filter tasks teacher solved but student couldn't.
    5. Run ``num_distillation_steps`` gradient steps on the distillable tasks.
    """

    # OPSD scheduling
    opsd_every: int = 1  # Run OPSD phases every N RL batches.

    # Evaluation phase
    eval_group_size: int = 6

    # Teacher re-attempt phase (defaults to group_size if not explicitly set).
    teacher_group_size: int = 0

    # Self-reflection phase
    max_reflection_tokens: int = 4096
    max_condensed_trace_tokens: int = 4096
    num_traces_for_reflection: int = 4
    reflection_cache_cycles: int = 3  # Reuse cached reflections for N consecutive cycles.

    # Distillation phase
    num_distillation_steps: int = 1
    distill_kl_penalty_coef: float = 1.0
    distill_kl_discount_factor: float = 0.0
    distill_min_batch_size: int = 0  # Min datums to run distillation (0=no minimum, always train).
    distill_learning_rate: float = 0.0  # LR for distillation (0=use main learning_rate).

    @property
    def effective_distill_learning_rate(self) -> float:
        """Return the learning rate to use for distillation steps."""
        return self.distill_learning_rate if self.distill_learning_rate > 0 else self.learning_rate

    def validate(self) -> None:
        """Validate the configuration. Raises ValueError on invalid state."""
        super().validate()
        if self.opsd_every <= 0:
            raise ValueError("opsd_every must be positive")
        if self.eval_group_size <= 0:
            raise ValueError("eval_group_size must be positive")
        # Default teacher_group_size to group_size so the teacher gets the
        # same number of attempts as the student during RL.
        if self.teacher_group_size <= 0:
            self.teacher_group_size = self.group_size
        if self.num_distillation_steps <= 0:
            raise ValueError("num_distillation_steps must be positive")
        if self.num_traces_for_reflection <= 0:
            raise ValueError("num_traces_for_reflection must be positive")
        if self.distill_min_batch_size < 0:
            raise ValueError("distill_min_batch_size must be non-negative")
        if self.distill_learning_rate < 0:
            raise ValueError("distill_learning_rate must be non-negative")
        if self.reflection_cache_cycles < 0:
            raise ValueError("reflection_cache_cycles must be non-negative")
