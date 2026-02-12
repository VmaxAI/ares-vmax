"""Training entry point for ARES + Tinker terminal-based RL.

Merges the working reference's training config (WORKING_TINKER/terminal_rl/train.py)
with ARES-specific monkey-patches for grad clipping and error-resilient rollouts
(from examples/05_roger_tinker_train.py).
"""

from __future__ import annotations

import importlib
import logging
import os
from typing import Any

from ares.tinker_integration import config as config_mod
from ares.tinker_integration import dataset

_LOGGER = logging.getLogger(__name__)


def _make_harbor_env_config(env_type: str) -> Any:
    """Create a Harbor EnvironmentConfig for the given type ("daytona" or "docker")."""
    try:
        cfg_mod = importlib.import_module("harbor.models.trial.config")
        env_cfg_cls = cfg_mod.EnvironmentConfig
        return env_cfg_cls(type=env_type)
    except Exception as e:  # pragma: no cover
        raise ImportError(
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
    5. Monkey-patches optim_step for grad clipping.
    6. Monkey-patches do_group_rollout_and_filter_constant_reward for error resilience.
    7. Calls tinker_cookbook.rl.train.main(cfg).
    """
    config.validate()

    # Fail fast if env vars aren't set.
    if "TINKER_API_KEY" not in os.environ:
        raise ValueError("TINKER_API_KEY environment variable is not set")
    if config.env_type == "daytona" and "DAYTONA_API_KEY" not in os.environ:
        raise ValueError("DAYTONA_API_KEY environment variable is not set (required for daytona env_type)")

    # Load tasks.
    if config.task_dir:
        tasks = dataset.load_tasks_from_task_dir(config.task_dir)
        _LOGGER.info("Loaded 1 task from task_dir: %s", config.task_dir)
    else:
        assert config.preset_name is not None  # guaranteed by validate()
        tasks = dataset.load_tasks_from_preset(config.preset_name, num_tasks=config.num_tasks)

    # Create Harbor environment config.
    env_cfg = _make_harbor_env_config(config.env_type)

    # Build dataset.
    dataset_builder = dataset.TerminalRLDatasetBuilder(
        tasks=tasks,
        group_size=config.group_size,
        environment=env_cfg,
        groups_per_batch=config.groups_per_batch,
        num_batches=config.num_batches,
        renderer_name=config.renderer_name,
        model_name_for_tokenizer=config.model_name,
        max_trajectory_tokens=config.max_trajectory_tokens,
        gym_env_kwargs={"auto_stop_minutes": config.auto_stop_minutes},
    )

    # Import tinker training module.
    tinker_mod = importlib.import_module("tinker")
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

    # Monkey-patch optim_step for grad clipping (from roger_tinker_train.py lines 585-601).
    _original_optim_step = tinker_train.optim_step
    grad_clip_norm = config.grad_clip_norm

    async def _optim_step_with_grad_clip(
        training_client: Any,
        learning_rate: float,
    ) -> None:
        adam_params_cls = tinker_mod.AdamParams
        adam_params = adam_params_cls(
            learning_rate=learning_rate,
            beta1=0.9,
            beta2=0.95,
            eps=1e-8,
            grad_clip_norm=grad_clip_norm,
        )
        optim_step_future = await training_client.optim_step_async(adam_params)
        await optim_step_future.result_async()

    tinker_train.optim_step = _optim_step_with_grad_clip  # type: ignore[assignment]

    # Monkey-patch do_group_rollout_and_filter_constant_reward for error resilience
    # (from roger_tinker_train.py lines 604-615).
    _original_do_group_rollout_and_filter = tinker_train.do_group_rollout_and_filter_constant_reward

    async def _safe_do_group_rollout_and_filter(*args: Any, **kwargs: Any) -> Any:
        try:
            return await _original_do_group_rollout_and_filter(*args, **kwargs)
        except Exception as e:
            _LOGGER.warning("Group rollout skipped due to error (training continues): %s: %s", type(e).__name__, e)
            return None

    tinker_train.do_group_rollout_and_filter_constant_reward = _safe_do_group_rollout_and_filter  # type: ignore[assignment]

    _LOGGER.info(
        "Starting training: model=%s, tasks=%d, batches=%d, group_size=%d",
        config.model_name,
        len(tasks),
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
    _LOGGER.info("Log path: %s", config.log_path)

    # Run training.
    try:
        await tinker_train.main(cfg)
    finally:
        tinker_train.optim_step = _original_optim_step  # type: ignore[assignment]
        tinker_train.do_group_rollout_and_filter_constant_reward = _original_do_group_rollout_and_filter  # type: ignore[assignment]
