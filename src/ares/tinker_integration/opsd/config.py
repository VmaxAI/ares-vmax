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

    1. Evaluate student on all tasks to find hard tasks (0% success).
    2. Self-reflect on failed traces to generate privileged information.
    3. Teacher (same model + privileged context) re-attempts hard tasks.
    4. Filter tasks teacher solved but student couldn't.
    5. Run ``num_distillation_steps`` gradient steps on the distillable tasks.
    """

    # OPSD scheduling
    opsd_every: int = 1  # Run OPSD phases every N RL batches.

    # Evaluation phase
    eval_group_size: int = 6

    # Teacher re-attempt phase
    teacher_group_size: int = 6

    # Self-reflection phase
    max_reflection_tokens: int = 1024
    max_condensed_trace_tokens: int = 2048
    num_traces_for_reflection: int = 2

    # Distillation phase
    num_distillation_steps: int = 5
    distill_kl_penalty_coef: float = 1.0
    distill_kl_discount_factor: float = 0.0

    def validate(self) -> None:
        """Validate the configuration. Raises ValueError on invalid state."""
        super().validate()
        if self.opsd_every <= 0:
            raise ValueError("opsd_every must be positive")
        if self.eval_group_size <= 0:
            raise ValueError("eval_group_size must be positive")
        if self.teacher_group_size <= 0:
            raise ValueError("teacher_group_size must be positive")
        if self.num_distillation_steps <= 0:
            raise ValueError("num_distillation_steps must be positive")
        if self.num_traces_for_reflection <= 0:
            raise ValueError("num_traces_for_reflection must be positive")
