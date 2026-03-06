"""Training entry point for ARES + Tinker RL.

Merges the working reference's training config (WORKING_TINKER/terminal_rl/train.py)
with ARES-specific monkey-patches for grad clipping and error-resilient rollouts
(from examples/05_roger_tinker_train.py).

Supports two harness modes via ``config.harness``:
- ``terminal``: Direct tmux terminal control (default).
- ``code-agent``: ARES CodeEnvironment with agent harness.
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from ares.tinker_integration import ares_env
from ares.tinker_integration import config as config_mod
from ares.tinker_integration import dataset
from ares.tinker_integration import monkey_patches
from ares.tinker_integration import terminal_env

_LOGGER = logging.getLogger(__name__)


def _make_harbor_env_config(env_type: str, *, snapshot_template_name: str | None = None) -> Any:
    """Create a Harbor EnvironmentConfig for the given type ("daytona" or "docker")."""
    try:
        cfg_mod = importlib.import_module("harbor.models.trial.config")
        env_cfg_cls = cfg_mod.EnvironmentConfig
        kwargs: dict[str, Any] = {}
        if snapshot_template_name is not None:
            kwargs["snapshot_template_name"] = snapshot_template_name
        return env_cfg_cls(type=env_type, kwargs=kwargs)
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Failed to import 'harbor.models.trial.config'. "
            "tinker_integration requires the optional 'harbor' dependency. "
            "Install it in your environment (and ensure it is importable) to run training."
        ) from e


async def run_training(config: config_mod.TrainingConfig) -> None:
    """Run terminal-based RL training with the given configuration.

    This is the main entry point. It:
    1. Validates env vars (TINKER_API_KEY, optionally DAYTONA_API_KEY).
    2. Loads tasks from task_dir or ARES preset.
    3. Creates the TerminalRLDatasetBuilder.
    4. Configures tinker_cookbook.rl.train.Config with proven defaults.
    5. Applies shared monkey-patches (grad clipping, error resilience, etc.).
    6. Calls tinker_cookbook.rl.train.main(cfg).
    """
    config.validate()

    # Fail fast if env vars aren't set.
    if "TINKER_API_KEY" not in os.environ:
        raise ValueError("TINKER_API_KEY environment variable is not set")
    if config.env_type == "daytona" and "DAYTONA_API_KEY" not in os.environ:
        raise ValueError("DAYTONA_API_KEY environment variable is not set (required for daytona env_type)")

    # Override ARES's global Daytona auto-stop interval so that DaytonaContainer.start()
    # uses the training config value instead of the default 30 minutes.
    if config.env_type == "daytona":
        os.environ["DAYTONA_AUTO_STOP_INTERVAL"] = str(config.auto_stop_minutes)
        from ares import config as ares_config_mod

        ares_config_mod.reload()
        _LOGGER.info("Set Daytona auto-stop interval to %d minutes", config.auto_stop_minutes)

    # In async mode, add a builder buffer so each batch produces a few extra builders.
    is_async = config.max_steps_off_policy is not None
    builder_buffer = config.async_builder_buffer if is_async else 0

    dataset_builder: Any
    if config.harness == "code-agent":
        assert config.preset_name is not None  # guaranteed by validate()
        container_factory = ares_env._get_container_factory(config.env_type)
        dataset_builder = ares_env.AresRLDatasetBuilder(
            preset_name=config.preset_name,
            num_tasks=config.num_tasks,
            group_size=config.group_size,
            container_factory=container_factory,
            groups_per_batch=config.groups_per_batch,
            num_batches=config.num_batches,
            renderer_name=config.renderer_name,
            model_name_for_tokenizer=config.model_name,
            max_trajectory_tokens=config.max_trajectory_tokens,
            max_tokens=config.max_tokens,
            builder_buffer=builder_buffer,
            snapshot_template_name=config.snapshot_template_name,
        )
    else:
        if config.task_dir:
            tasks = dataset.load_tasks_from_task_dir(config.task_dir)
            _LOGGER.info("Loaded 1 task from task_dir: %s", config.task_dir)
        else:
            assert config.preset_name is not None  # guaranteed by validate()
            tasks = dataset.load_tasks_from_preset(config.preset_name, num_tasks=config.num_tasks)

        env_cfg = _make_harbor_env_config(config.env_type, snapshot_template_name=config.snapshot_template_name)

        dataset_builder = dataset.TerminalRLDatasetBuilder(
            tasks=tasks,
            group_size=config.group_size,
            environment=env_cfg,
            groups_per_batch=config.groups_per_batch,
            num_batches=config.num_batches,
            renderer_name=config.renderer_name,
            model_name_for_tokenizer=config.model_name,
            max_trajectory_tokens=config.max_trajectory_tokens,
            max_tokens=config.max_tokens,
            gym_env_kwargs={
                "auto_stop_minutes": config.auto_stop_minutes,
                "sandbox_cpus": config.sandbox_cpus,
                "sandbox_memory_gb": config.sandbox_memory_gb,
                "sandbox_disk_gb": config.sandbox_disk_gb,
            },
            builder_buffer=builder_buffer,
        )

    # Limit concurrent sandbox creation requests to prevent Daytona 429 bursts.
    terminal_env.set_max_concurrent_sandboxes(config.max_concurrent_sandboxes)

    # Import tinker training module.
    tinker_train = importlib.import_module("tinker_cookbook.rl.train")
    train_config_cls = tinker_train.Config

    # Build training config (mirrors working reference's train.py lines 94-106).
    train_cfg_kwargs: dict[str, Any] = {
        "model_name": config.model_name,
        "log_path": config.log_path,
        "dataset_builder": dataset_builder,
        "learning_rate": config.learning_rate,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
        "eval_every": config.eval_every,
        "save_every": config.save_every,
        "lora_rank": config.lora_rank,
        "base_url": config.base_url,
        "remove_constant_reward_groups": config.remove_constant_reward_groups,
        "loss_fn": config.loss_fn,
        "kl_penalty_coef": config.kl_penalty_coef,
    }

    if config.load_checkpoint_path:
        train_cfg_kwargs["load_checkpoint_path"] = config.load_checkpoint_path

    if config.wandb_project:
        train_cfg_kwargs["wandb_project"] = config.wandb_project
    if config.wandb_name:
        train_cfg_kwargs["wandb_name"] = config.wandb_name

    # Async rollout config.
    if config.max_steps_off_policy is not None:
        async_config_cls = tinker_train.AsyncConfig
        train_cfg_kwargs["async_config"] = async_config_cls(
            max_steps_off_policy=config.max_steps_off_policy,
            groups_per_batch=config.groups_per_batch,
        )

    cfg = train_config_cls(**train_cfg_kwargs)

    task_source = config.preset_name or config.task_dir or "unknown"
    _LOGGER.info(
        "Starting training: harness=%s, model=%s, tasks=%s, batches=%d, group_size=%d",
        config.harness,
        config.model_name,
        task_source,
        config.num_batches,
        config.group_size,
    )
    _LOGGER.info(
        "Hyperparameters: lr=%s, rank=%d, loss_fn=%s, grad_clip=%.2f",
        config.learning_rate,
        config.lora_rank,
        config.loss_fn,
        config.grad_clip_norm,
    )
    if is_async:
        _LOGGER.info(
            "Async mode: max_steps_off_policy=%d, rollout_retries=%d, builder_buffer=%d",
            config.max_steps_off_policy,
            config.async_rollout_retries,
            builder_buffer,
        )
    else:
        _LOGGER.info("Sync mode (on-policy)")
    _LOGGER.info("Log path: %s", config.log_path)

    # Run training with shared monkey-patches.
    with monkey_patches.MonkeyPatchContext(
        grad_clip_norm=config.grad_clip_norm,
        rollout_max_retries=config.async_rollout_retries,
    ):
        await tinker_train.main(cfg)
