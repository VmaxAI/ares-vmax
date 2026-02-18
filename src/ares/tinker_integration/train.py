"""Training entry point for ARES + Tinker RL.

Merges the working reference's training config (WORKING_TINKER/terminal_rl/train.py)
with ARES-specific monkey-patches for grad clipping and error-resilient rollouts
(from examples/05_roger_tinker_train.py).

Supports two harness modes via ``config.harness``:
- ``terminal``: Direct tmux terminal control (default).
- ``code-agent``: ARES CodeEnvironment with agent harness.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
import os
import random
import time
from typing import Any

from ares.tinker_integration import ares_env
from ares.tinker_integration import config as config_mod
from ares.tinker_integration import dataset
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
    except Exception as e:  # pragma: no cover
        raise ImportError(
            "tinker_integration requires the optional 'harbor' dependency. "
            "Install it in your environment (and ensure it is importable) to run training."
        ) from e


def _log_rollout_complete(task_label: str, result: Any, num_envs: int, elapsed: float) -> None:
    """Log a one-line summary after a successful group rollout."""
    rewards = result.get_total_rewards()
    turns = [len(t.transitions) for t in result.trajectories_G]

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    mean_turns = sum(turns) / len(turns) if turns else 0.0

    reward_parts = " ".join(f"{r:.2f}" for r in rewards)
    turns_parts = " ".join(str(t) for t in turns)

    _LOGGER.info(
        "Rollout done | task=%s | %.1fs | envs=%d | reward=%.3f [%s] | turns=%.1f [%s] | sandboxes: closed",
        task_label,
        elapsed,
        num_envs,
        mean_reward,
        reward_parts,
        mean_turns,
        turns_parts,
    )


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

    # Override ARES's global Daytona auto-stop interval so that DaytonaContainer.start()
    # uses the training config value instead of the default 30 minutes.  This is needed
    # for the code-agent harness (which creates containers via ares.make() -> DaytonaContainer)
    # and also acts as a belt-and-suspenders for the terminal harness.
    if config.env_type == "daytona":
        os.environ["DAYTONA_AUTO_STOP_INTERVAL"] = str(config.auto_stop_minutes)
        # Force re-creation of the frozen config with the new env var.
        from ares import config as ares_config_mod

        ares_config_mod.CONFIG = ares_config_mod._Config()  # type: ignore[misc]
        _LOGGER.info("Set Daytona auto-stop interval to %d minutes", config.auto_stop_minutes)

    # In async mode, add a builder buffer so each batch produces a few extra builders.
    # This compensates for any rollouts that permanently fail and lose their builder
    # (the training loop needs exactly groups_per_batch non-None groups per step).
    is_async = config.max_steps_off_policy is not None
    builder_buffer = config.async_builder_buffer if is_async else 0

    dataset_builder: Any
    if config.harness == "code-agent":
        # ARES CodeEnvironment harness — wraps ares.make() environments.
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
        # Terminal harness (default) — tmux + JSON commands.
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
            gym_env_kwargs={
                "auto_stop_minutes": config.auto_stop_minutes,
                "sandbox_cpus": config.sandbox_cpus,
                "sandbox_memory_gb": config.sandbox_memory_gb,
                "sandbox_disk_gb": config.sandbox_disk_gb,
            },
            builder_buffer=builder_buffer,
        )

    # Limit concurrent sandbox creation requests to prevent Daytona 429 bursts.
    # Pass None to disable (retry handles throttling), or an int to cap concurrency.
    terminal_env.set_max_concurrent_sandboxes(config.max_concurrent_sandboxes)

    # Import tinker training module.
    tinker_mod = importlib.import_module("tinker")
    tinker_train = importlib.import_module("tinker_cookbook.rl.train")
    train_config_cls = tinker_train.Config

    # Monkey-patch wandb config.update to allow value changes.
    # tinker_cookbook logs the full Config (including the non-serializable
    # dataset_builder) twice: once via wandb.init(config=...) and again in
    # log_hparams -> wandb.config.update(). The second call sees a different
    # object id for dataset_builder and raises ConfigError.  The env var
    # WANDB_ALLOW_VAL_CHANGE is not checked by wandb's _sanitize(), so we
    # patch the method directly.
    import wandb.sdk.wandb_config as _wbcfg

    _orig_config_update = _wbcfg.Config.update

    def _config_update_allow(self: Any, d: Any, allow_val_change: Any = True) -> None:
        return _orig_config_update(self, d, allow_val_change=allow_val_change)

    _wbcfg.Config.update = _config_update_allow  # type: ignore[assignment]

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

    # Monkey-patch do_group_rollout_and_filter_constant_reward for error resilience.
    #
    # CRITICAL for async mode: tinker_cookbook's async training loop requires exactly
    # groups_per_batch non-None groups per train step.  If a rollout fails and returns
    # None, that builder is permanently lost (unlike stale rejections, which requeue).
    # With only num_batches * groups_per_batch total builders, even a few lost builders
    # can cause the training loop to deadlock waiting for groups that will never arrive.
    #
    # Retries with exponential backoff ensure transient Daytona errors (429, connection
    # issues) don't permanently lose builders.  This is the primary defense against the
    # async deadlock.
    _original_do_group_rollout_and_filter = tinker_train.do_group_rollout_and_filter_constant_reward
    rollout_max_retries = config.async_rollout_retries

    async def _safe_do_group_rollout_and_filter(*args: Any, **kwargs: Any) -> Any:
        for attempt in range(1, rollout_max_retries + 1):
            try:
                return await _original_do_group_rollout_and_filter(*args, **kwargs)
            except Exception as e:
                if attempt < rollout_max_retries:
                    delay = min(10.0 * (2 ** (attempt - 1)), 120.0)
                    delay *= 0.5 + random.random()  # jitter
                    _LOGGER.warning(
                        "Group rollout failed (attempt %d/%d), retrying in %.0fs: %s: %s",
                        attempt,
                        rollout_max_retries,
                        delay,
                        type(e).__name__,
                        e,
                    )
                    await asyncio.sleep(delay)
                else:
                    _LOGGER.error(
                        "Group rollout failed after %d attempts (training continues): %s: %s",
                        rollout_max_retries,
                        type(e).__name__,
                        e,
                    )
                    return None

    tinker_train.do_group_rollout_and_filter_constant_reward = _safe_do_group_rollout_and_filter  # type: ignore[assignment]

    # Monkey-patch do_train_step_and_get_sampling_client to skip batches where
    # all rollouts failed.  When every group returns None (caught above), the
    # filtered trajectory_groups_P list is empty and compute_trajectory_metrics
    # crashes with ZeroDivisionError.  We detect this and skip the train step,
    # returning a fresh sampling client from the unchanged model weights.
    _original_do_train_step = tinker_train.do_train_step_and_get_sampling_client

    async def _safe_do_train_step(*args: Any, **kwargs: Any) -> Any:
        # The 7th positional arg (index 6) is trajectory_groups_P.
        trajectory_groups = args[6] if len(args) > 6 else kwargs.get("trajectory_groups_P", [])
        if not trajectory_groups:
            _LOGGER.warning("All rollouts in batch failed — skipping train step (no weight update)")
            # Return a sampling client from unchanged weights and empty metrics.
            training_client_arg = args[2] if len(args) > 2 else kwargs["training_client"]
            sampling_client = await training_client_arg.save_weights_and_get_sampling_client_async()
            return sampling_client, {}
        return await _original_do_train_step(*args, **kwargs)

    tinker_train.do_train_step_and_get_sampling_client = _safe_do_train_step  # type: ignore[assignment]

    # Monkey-patch do_group_rollout to close sandboxes on failure.  tinker_cookbook's
    # do_single_rollout/do_group_rollout have no cleanup logic, so failed rollouts
    # leak Daytona sandboxes.  This is safe because AsyncTerminalGymEnv.close() is
    # idempotent — calling it on an already-closed env is a no-op.
    #
    # IMPORTANT: We save and restore `make_envs` to prevent wrapper chaining.
    # In async mode, stale sample rejection requeues the env_group_builder.  If we
    # leave our tracking wrapper on `make_envs`, the next rollout wraps it again,
    # creating a growing chain.  The `finally` block restores the original.
    _original_do_group_rollout = tinker_train.do_group_rollout
    _original_make_envs_attr = "_ares_original_make_envs"

    async def _do_group_rollout_with_cleanup(env_group_builder: Any, policy: Any) -> Any:
        # Get the TRUE original make_envs (not a previously wrapped version).
        original_make_envs = getattr(env_group_builder, _original_make_envs_attr, env_group_builder.make_envs)
        # Store it so future calls (after requeue) always use the original.
        setattr(env_group_builder, _original_make_envs_attr, original_make_envs)

        created_envs: list[Any] = []

        async def _tracking_make_envs() -> Any:
            envs = await original_make_envs()
            created_envs.extend(envs)
            return envs

        env_group_builder.make_envs = _tracking_make_envs
        tags = env_group_builder.logging_tags()
        task_label = tags[-1] if tags else "unknown"
        t_start = time.monotonic()

        try:
            result = await _original_do_group_rollout(env_group_builder, policy)
            elapsed = time.monotonic() - t_start
            _log_rollout_complete(task_label, result, len(created_envs), elapsed)
            return result
        except BaseException:
            elapsed = time.monotonic() - t_start
            num_envs = len(created_envs)
            _LOGGER.warning(
                "Rollout FAILED | task=%s | %.1fs | envs=%d | sandboxes: closing...",
                task_label,
                elapsed,
                num_envs,
            )
            t_close = time.monotonic()
            for env in created_envs:
                try:
                    await env.close()
                except Exception as close_err:
                    _LOGGER.debug("Failed to close sandbox during cleanup: %s", close_err)
            close_elapsed = time.monotonic() - t_close
            _LOGGER.warning(
                "Rollout FAILED | task=%s | envs=%d | sandboxes: closed! (%.1fs cleanup)",
                task_label,
                num_envs,
                close_elapsed,
            )
            raise
        finally:
            # Always restore original make_envs to prevent wrapper chaining on requeue.
            env_group_builder.make_envs = original_make_envs

    tinker_train.do_group_rollout = _do_group_rollout_with_cleanup  # type: ignore[assignment]

    # Suppress noisy Harbor/Daytona debug logging (shell commands, API calls).
    for noisy_logger in ("harbor", "daytona_sdk", "daytona", "httpx"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

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

    # Run training.
    try:
        await tinker_train.main(cfg)
    finally:
        tinker_train.optim_step = _original_optim_step  # type: ignore[assignment]
        tinker_train.do_group_rollout_and_filter_constant_reward = _original_do_group_rollout_and_filter  # type: ignore[assignment]
        tinker_train.do_train_step_and_get_sampling_client = _original_do_train_step  # type: ignore[assignment]
        tinker_train.do_group_rollout = _original_do_group_rollout  # type: ignore[assignment]
        _wbcfg.Config.update = _orig_config_update  # type: ignore[assignment]
