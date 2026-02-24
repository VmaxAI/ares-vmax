"""Training configuration for ARES + Tinker RL.

Provides a simple dataclass with proven defaults from the working reference
(WORKING_TINKER/terminal_rl/train.py) merged with useful options from the
existing ARES integration (examples/05_roger_tinker_train.py).

Supports two harness modes:
- ``terminal``: Direct tmux terminal control via HarborTerminalTinkerEnv (default).
- ``code-agent``: ARES CodeEnvironment with agent harness via AresCodeTinkerEnv.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrainingConfig:
    """Configuration for ARES + Tinker RL training.

    Supports two harness modes:
    - ``terminal``: Direct tmux terminal control (default). Supports both task_dir and preset_name.
    - ``code-agent``: ARES CodeEnvironment with agent harness. Requires preset_name.

    Supports two task sources:
    - ``task_dir``: Single task directory path (terminal harness only).
    - ``preset_name``: ARES preset (e.g., "sbv-terminus2") for multi-task training.

    Exactly one of ``task_dir`` or ``preset_name`` must be specified.
    """

    # Harness
    harness: str = "terminal"

    # Model
    model_name: str = ""
    renderer_name: str | None = None
    env_type: str = "daytona"

    # Task source (one of these must be specified)
    task_dir: str | None = None
    preset_name: str | None = None
    num_tasks: int | None = None

    # Proven defaults from working reference
    learning_rate: float = 4e-5
    lora_rank: int = 32
    temperature: float = 1.0
    max_tokens: int = 4096
    group_size: int = 8
    groups_per_batch: int = 1
    num_batches: int = 25
    max_trajectory_tokens: int = 32768

    # Training options (from roger_tinker_train.py)
    loss_fn: str = "importance_sampling"
    remove_constant_reward_groups: bool = False
    grad_clip_norm: float = 0.5
    kl_penalty_coef: float = 0.0

    # Sandbox safety
    auto_stop_minutes: int = 30
    max_concurrent_sandboxes: int | None = 20

    # Sandbox resources (None = use harbor/task defaults)
    sandbox_cpus: int | None = None
    sandbox_memory_gb: int | None = None
    sandbox_disk_gb: int | None = None

    # Daytona snapshot support (None = use declarative image builds)
    snapshot_template_name: str | None = None

    # Async rollout (None = sync mode, set to e.g. 5 for async)
    max_steps_off_policy: int | None = None
    async_rollout_retries: int = 5
    async_builder_buffer: int = 2

    # Logging
    log_path: str = ""
    wandb_project: str | None = None
    wandb_name: str | None = None
    save_every: int = 10
    eval_every: int = 0
    base_url: str | None = None
    load_checkpoint_path: str | None = None

    def validate(self) -> None:
        """Validate the configuration. Raises ValueError on invalid state."""
        if self.harness not in ("terminal", "code-agent"):
            raise ValueError(f"harness must be 'terminal' or 'code-agent', got {self.harness!r}")
        if not self.model_name:
            raise ValueError("model_name is required")
        if not self.log_path:
            raise ValueError("log_path is required")
        if not self.task_dir and not self.preset_name:
            raise ValueError("Either task_dir or preset_name must be specified")
        if self.task_dir and self.preset_name:
            raise ValueError("Only one of task_dir or preset_name can be specified")
        if self.harness == "code-agent" and not self.preset_name:
            raise ValueError("code-agent harness requires preset_name (task_dir is not supported)")
        if self.group_size <= 0:
            raise ValueError("group_size must be positive")
        if self.groups_per_batch <= 0:
            raise ValueError("groups_per_batch must be positive")
        if self.num_batches <= 0:
            raise ValueError("num_batches must be positive")
        if self.async_rollout_retries < 1:
            raise ValueError("async_rollout_retries must be at least 1")
        if self.async_builder_buffer < 0:
            raise ValueError("async_builder_buffer must be non-negative")
        if self.snapshot_template_name is not None and "{name}" not in self.snapshot_template_name:
            raise ValueError("snapshot_template_name must contain '{name}' placeholder")
