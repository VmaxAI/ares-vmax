"""OPSD-specific training configuration.

Extends ``TrainingConfig`` with parameters for iterative phasic training:
evaluation, self-reflection, teacher re-attempts, and reverse-KL distillation.
"""

from __future__ import annotations

from dataclasses import dataclass

from ares.tinker_integration import config as config_mod


@dataclass
class OPSDConfig(config_mod.TrainingConfig):
    """Configuration for On-Policy Self-Distillation training.

    Inherits all fields from ``TrainingConfig`` and adds OPSD-specific controls
    for the iterative phasic procedure:

    1. Student RL (``rl_batches_per_iteration`` batches).
    2. Evaluate student on all tasks to find hard tasks (0% success).
    3. Self-reflect on failed traces to generate privileged information.
    4. Teacher (same model + privileged context) re-attempts hard tasks.
    5. Filter tasks teacher solved but student couldn't.
    6. On-policy distillation with reverse-KL objective.
    7. Repeat for ``num_iterations``.
    """

    # OPSD iteration control
    num_iterations: int = 3
    rl_batches_per_iteration: int = 10

    # Evaluation phase
    eval_group_size: int = 6

    # Teacher re-attempt phase
    teacher_group_size: int = 6

    # Self-reflection phase
    max_reflection_tokens: int = 1024
    max_condensed_trace_tokens: int = 2048
    num_traces_for_reflection: int = 2

    # Distillation phase
    distill_batches: int = 5
    distill_groups_per_batch: int = 10
    distill_group_size: int = 4
    distill_kl_penalty_coef: float = 1.0
    distill_kl_discount_factor: float = 0.0

    def validate(self) -> None:
        """Validate the configuration. Raises ValueError on invalid state."""
        super().validate()
        if self.num_iterations <= 0:
            raise ValueError("num_iterations must be positive")
        if self.rl_batches_per_iteration <= 0:
            raise ValueError("rl_batches_per_iteration must be positive")
        if self.eval_group_size <= 0:
            raise ValueError("eval_group_size must be positive")
        if self.teacher_group_size <= 0:
            raise ValueError("teacher_group_size must be positive")
        if self.distill_batches <= 0:
            raise ValueError("distill_batches must be positive")
        if self.distill_groups_per_batch <= 0:
            raise ValueError("distill_groups_per_batch must be positive")
        if self.distill_group_size <= 0:
            raise ValueError("distill_group_size must be positive")
        if self.num_traces_for_reflection <= 0:
            raise ValueError("num_traces_for_reflection must be positive")

    def to_rl_config(self) -> config_mod.TrainingConfig:
        """Create a ``TrainingConfig`` for the RL phase of an OPSD iteration."""
        return config_mod.TrainingConfig(
            harness=self.harness,
            model_name=self.model_name,
            renderer_name=self.renderer_name,
            env_type=self.env_type,
            task_dir=self.task_dir,
            preset_name=self.preset_name,
            num_tasks=self.num_tasks,
            learning_rate=self.learning_rate,
            lora_rank=self.lora_rank,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            group_size=self.group_size,
            groups_per_batch=self.groups_per_batch,
            num_batches=self.rl_batches_per_iteration,
            max_trajectory_tokens=self.max_trajectory_tokens,
            loss_fn=self.loss_fn,
            remove_constant_reward_groups=self.remove_constant_reward_groups,
            grad_clip_norm=self.grad_clip_norm,
            kl_penalty_coef=self.kl_penalty_coef,
            auto_stop_minutes=self.auto_stop_minutes,
            max_concurrent_sandboxes=self.max_concurrent_sandboxes,
            sandbox_cpus=self.sandbox_cpus,
            sandbox_memory_gb=self.sandbox_memory_gb,
            sandbox_disk_gb=self.sandbox_disk_gb,
            snapshot_template_name=self.snapshot_template_name,
            max_steps_off_policy=self.max_steps_off_policy,
            async_rollout_retries=self.async_rollout_retries,
            async_builder_buffer=self.async_builder_buffer,
            log_path=self.log_path,
            wandb_project=self.wandb_project,
            wandb_name=self.wandb_name,
            save_every=self.save_every,
            eval_every=self.eval_every,
            base_url=self.base_url,
            load_checkpoint_path=self.load_checkpoint_path,
        )
