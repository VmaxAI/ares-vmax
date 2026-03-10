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
import functools
import importlib
import inspect
import logging
import os
import pathlib
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
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "Failed to import 'harbor.models.trial.config'. "
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

    # Extract meta-task aggregate metrics from trajectory step metrics.
    meta_stats = _extract_meta_stats(result)
    meta_suffix = ""
    if meta_stats:
        n = meta_stats["total"]
        hard = meta_stats["valid"] - meta_stats["frontier_solved"]
        meta_suffix = (
            f" | patches={meta_stats['patches']}/{n}"
            f" | valid={meta_stats['valid']}/{n}"
            f" | hard={hard}/{n}"
            f" | frontier_solved={meta_stats['frontier_solved']}/{n}"
        )

    _LOGGER.info(
        "Rollout done | task=%s | %.1fs | envs=%d | reward=%.3f [%s] | turns=%.1f [%s]%s | sandboxes: closed",
        task_label,
        elapsed,
        num_envs,
        mean_reward,
        reward_parts,
        mean_turns,
        turns_parts,
        meta_suffix,
    )


def _extract_meta_stats(result: Any) -> dict[str, float]:
    """Extract aggregate meta-task stats from trajectory metrics.

    Looks for ``bug_valid``, ``frontier_solved``, ``produces_patch`` in the
    final step metrics of each trajectory. Also collects rubric axis pass counts
    and holistic quality score sums when available. Returns empty dict if no meta
    metrics are found (non-meta-task rollout).

    All values are raw counts or sums — rates/averages are computed only at
    logging time in ``_log_meta_stats_to_wandb`` to avoid incorrect aggregation
    when summing across groups.
    """
    total = 0
    patches = 0
    no_test_mods = 0
    valid = 0
    inner_created = 0
    frontier_solved = 0
    # Rubric metrics: raw counts for correct aggregation across groups.
    rubric_axis_passes: dict[str, int] = {}
    rubric_axis_totals: dict[str, int] = {}
    rubric_quality_sum = 0.0
    rubric_quality_count = 0
    rubric_reward_sum = 0.0
    rubric_reward_count = 0

    for trajectory in result.trajectories_G:
        if not trajectory.transitions:
            continue
        # The last transition holds the final StepResult.metrics.
        last_metrics = getattr(trajectory.transitions[-1], "metrics", None)
        if last_metrics is None or "bug_valid" not in last_metrics:
            continue
        total += 1
        if last_metrics.get("produces_patch", 0.0) > 0:
            patches += 1
        if last_metrics.get("no_test_mods", 0.0) > 0:
            no_test_mods += 1
        if last_metrics.get("bug_valid", 0.0) > 0:
            valid += 1
        if last_metrics.get("inner_task_created", 0.0) > 0:
            inner_created += 1
        if last_metrics.get("frontier_solved", 0.0) > 0:
            frontier_solved += 1
        # Rubric axis pass/fail metrics (keyed as rubric_axis_<name>)
        for key, val in last_metrics.items():
            if key.startswith("rubric_axis_"):
                rubric_axis_totals[key] = rubric_axis_totals.get(key, 0) + 1
                if val > 0:
                    rubric_axis_passes[key] = rubric_axis_passes.get(key, 0) + 1
        hqs = last_metrics.get("rubric_holistic_quality_score")
        if hqs is not None:
            rubric_quality_sum += hqs
            rubric_quality_count += 1
        rs = last_metrics.get("rubric_score")
        if rs is not None:
            rubric_reward_sum += rs
            rubric_reward_count += 1

    if total == 0:
        return {}
    stats: dict[str, float] = {
        "total": total,
        "patches": patches,
        "no_test_mods": no_test_mods,
        "valid": valid,
        "inner_created": inner_created,
        "frontier_solved": frontier_solved,
    }
    # Store raw counts for rubric axes (rates computed at log time)
    for key, axis_total in rubric_axis_totals.items():
        stats[f"{key}_passes"] = rubric_axis_passes.get(key, 0)
        stats[f"{key}_total"] = axis_total
    if rubric_quality_count > 0:
        stats["rubric_quality_sum"] = rubric_quality_sum
        stats["rubric_quality_count"] = rubric_quality_count
    if rubric_reward_count > 0:
        stats["rubric_reward_sum"] = rubric_reward_sum
        stats["rubric_reward_count"] = rubric_reward_count
    return stats


def _aggregate_batch_meta_stats(trajectory_groups: list[Any]) -> dict[str, float]:
    """Aggregate meta-task stats across all trajectory groups in a training batch.

    All values are raw counts/sums so simple addition is correct.
    """
    totals: dict[str, float] = {}
    for group in trajectory_groups:
        if group is None:
            continue
        stats = _extract_meta_stats(group)
        if not stats:
            continue
        for k, v in stats.items():
            totals[k] = totals.get(k, 0) + v
    return totals


def _log_meta_stats_to_wandb(meta_stats: dict[str, float]) -> None:
    """Log aggregated meta-task pipeline stats to W&B (if active)."""
    if not meta_stats:
        return
    try:
        import wandb

        if wandb.run is None:
            return
    except ImportError:
        return

    n = meta_stats["total"]
    valid = meta_stats["valid"]
    hard = valid - meta_stats["frontier_solved"]

    log_data: dict[str, float] = {
        "meta/n_rollouts": n,
        "meta/produces_patch_rate": meta_stats["patches"] / n,
        "meta/empty_patch_rate": 1.0 - meta_stats["patches"] / n,
        "meta/no_test_mods_rate": meta_stats["no_test_mods"] / n,
        "meta/valid_bug_rate": valid / n,
        "meta/frontier_solved_rate": meta_stats["frontier_solved"] / max(1, valid),
        "meta/hard_bug_rate": hard / n,
        "meta/hard_bug_count": hard,
    }

    # Rubric metrics — compute rates from raw counts
    for key, val in meta_stats.items():
        if key.endswith("_total") and key.startswith("rubric_axis_"):
            # rubric_axis_realism_total → realism
            axis_name = key.removeprefix("rubric_axis_").removesuffix("_total")
            passes_key = f"rubric_axis_{axis_name}_passes"
            axis_total = val
            axis_passes = meta_stats.get(passes_key, 0)
            if axis_total > 0:
                log_data[f"meta/rubric/{axis_name}_pass_rate"] = axis_passes / axis_total
    rubric_quality_count = meta_stats.get("rubric_quality_count", 0)
    if rubric_quality_count > 0:
        log_data["meta/rubric/holistic_quality_avg"] = meta_stats["rubric_quality_sum"] / rubric_quality_count
    rubric_reward_count = meta_stats.get("rubric_reward_count", 0)
    if rubric_reward_count > 0:
        log_data["meta/rubric/reward_avg"] = meta_stats["rubric_reward_sum"] / rubric_reward_count

    wandb.log(log_data, commit=False)  # committed with next tinker_cookbook log call


async def run_training(config: config_mod.TrainingConfig, tasks: list | None = None) -> None:
    """Run terminal-based RL training with the given configuration.

    This is the main entry point. It:
    1. Validates env vars (TINKER_API_KEY, optionally DAYTONA_API_KEY).
    2. Loads tasks from task_dir or ARES preset (unless ``tasks`` is provided).
    3. Creates the TerminalRLDatasetBuilder.
    4. Configures tinker_cookbook.rl.train.Config with proven defaults.
    5. Monkey-patches optim_step for grad clipping.
    6. Monkey-patches do_group_rollout_and_filter_constant_reward for error resilience.
    7. Calls tinker_cookbook.rl.train.main(cfg).

    Args:
        config: Training configuration.
        tasks: Optional pre-loaded list of harbor.Task objects. When provided,
            skips task loading from task_dir/preset_name. This allows callers
            (e.g., demiurge-swe) to inject tasks directly.
    """
    config.validate(allow_no_task_source=tasks is not None)

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
        from ares import config as ares_config_mod

        ares_config_mod.reload()
        _LOGGER.info("Set Daytona auto-stop interval to %d minutes", config.auto_stop_minutes)

    # In async mode, add a builder buffer so each batch produces a few extra builders.
    # This compensates for any rollouts that permanently fail and lose their builder
    # (the training loop needs exactly groups_per_batch non-None groups per step).
    is_async = config.max_steps_off_policy is not None
    builder_buffer = config.async_builder_buffer if is_async else 0

    dataset_builder: Any
    if config.harness == "code-agent":
        # ARES CodeEnvironment harness — wraps ares.make() or injected tasks.
        container_factory = ares_env._get_container_factory(config.env_type)

        # Build a custom code agent factory if a config override is provided.
        code_agent_factory: Any | None = None
        if config.code_agent_config_path:
            from ares.code_agents import mini_swe_agent

            code_agent_factory = functools.partial(
                mini_swe_agent.MiniSWECodeAgent, config_path=config.code_agent_config_path
            )
            _LOGGER.info("Using custom code-agent config: %s", config.code_agent_config_path)

        # Collect API keys from host environment to pass into sandboxes.
        # These are needed by test.sh scripts (create_inner_task.py for issue
        # generation, run_inner_agent.py for frontier model trials).
        sandbox_env_keys = [
            "ANTHROPIC_API_KEY",
            "OPENAI_API_KEY",
            "DAYTONA_API_KEY",
            "DAYTONA_API_URL",
            "CHAT_COMPLETION_API_KEY",
            "CHAT_COMPLETION_BASE_URL",
            "HF_TOKEN",
        ]
        sandbox_env: dict[str, str] = {k: v for k in sandbox_env_keys if (v := os.environ.get(k))}
        if sandbox_env:
            _LOGGER.info("Passing %d env vars to sandboxes: %s", len(sandbox_env), list(sandbox_env.keys()))
        else:
            _LOGGER.warning("No API keys found in host environment — sandbox scripts may fail")

        trial_log_path = pathlib.Path(config.log_path) / "trials" if config.log_path else None

        if tasks is not None:
            # Task-based builder: demiurge-swe injects Harbor Task objects directly.
            _LOGGER.info("Using %d injected tasks with code-agent harness", len(tasks))
            dataset_builder = ares_env.AresRLDatasetBuilder(
                tasks=tasks,
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
                code_agent_factory=code_agent_factory,
                sandbox_env=sandbox_env or None,
                trial_log_path=trial_log_path,
            )
        else:
            # Preset-based builder: tasks from ARES registry.
            assert config.preset_name is not None  # guaranteed by validate()
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
                code_agent_factory=code_agent_factory,
                sandbox_env=sandbox_env or None,
                trial_log_path=trial_log_path,
            )
    else:
        # Terminal harness (default) — tmux + JSON commands.
        if tasks is not None:
            _LOGGER.info("Using %d injected tasks (skipping task_dir/preset loading)", len(tasks))
        elif config.task_dir:
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
                    cause = e.__cause__ or e.__context__
                    cause_msg = f" | caused by {type(cause).__name__}: {cause}" if cause else ""
                    _LOGGER.warning(
                        "Group rollout failed (attempt %d/%d), retrying in %.0fs: %s: %s%s",
                        attempt,
                        rollout_max_retries,
                        delay,
                        type(e).__name__,
                        e,
                        cause_msg,
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

    # Monkey-patch remove_constant_reward_groups to handle None values.
    # _safe_do_group_rollout_and_filter returns None on permanent failure; the
    # sync path collects these into trajectory_groups_P via gather_with_progress
    # and passes them to remove_constant_reward_groups, which crashes calling
    # group.get_total_rewards() on None.  Filter them out first.
    _original_remove_constant_reward_groups = tinker_train.remove_constant_reward_groups

    def _safe_remove_constant_reward_groups(trajectory_groups: list[Any]) -> list[Any]:
        filtered = [g for g in trajectory_groups if g is not None]
        if not filtered:
            _LOGGER.warning("All rollouts in batch returned None — nothing to filter")
            return filtered
        return _original_remove_constant_reward_groups(filtered)

    tinker_train.remove_constant_reward_groups = _safe_remove_constant_reward_groups  # type: ignore[assignment]

    # Monkey-patch do_train_step_and_get_sampling_client to handle None values
    # and skip batches where all rollouts failed.  When every group returns
    # None (caught above), compute_trajectory_metrics crashes with
    # ZeroDivisionError.  We filter Nones and skip the train step if empty.
    _original_do_train_step = tinker_train.do_train_step_and_get_sampling_client
    _do_train_step_sig = inspect.signature(_original_do_train_step)

    # Validate that the parameters we depend on exist in the current Tinker version.
    _required_train_step_params = {"training_client", "env_group_builders_P", "trajectory_groups_P"}
    _actual_params = set(_do_train_step_sig.parameters)
    if not _required_train_step_params.issubset(_actual_params):
        missing = _required_train_step_params - _actual_params
        raise RuntimeError(
            f"do_train_step_and_get_sampling_client signature changed: missing {missing}. "
            f"Actual parameters: {list(_do_train_step_sig.parameters)}. "
            f"The monkey-patch in train.py needs to be updated."
        )

    async def _safe_do_train_step(*args: Any, **kwargs: Any) -> Any:
        # Bind positional + keyword args to named parameters so we can
        # robustly access them by name regardless of call-site conventions.
        bound = _do_train_step_sig.bind(*args, **kwargs)
        bound.apply_defaults()

        trajectory_groups: list[Any] = bound.arguments["trajectory_groups_P"]

        # Filter out None trajectory groups (from crashed rollouts) and keep
        # env_group_builders_P in sync so the two lists stay aligned.
        if any(g is None for g in trajectory_groups):
            original_count = len(trajectory_groups)
            env_builders: list[Any] = bound.arguments["env_group_builders_P"]
            pairs = [(b, g) for b, g in zip(env_builders, trajectory_groups, strict=False) if g is not None]
            bound.arguments["env_group_builders_P"] = [p[0] for p in pairs]
            bound.arguments["trajectory_groups_P"] = [p[1] for p in pairs]
            trajectory_groups = bound.arguments["trajectory_groups_P"]
            _LOGGER.warning(
                "TRAIN STEP | filtered %d None groups (%d -> %d valid)",
                original_count - len(trajectory_groups),
                original_count,
                len(trajectory_groups),
            )

        if not trajectory_groups:
            _LOGGER.warning("TRAIN STEP | all rollouts failed — skipping (no weight update)")
            training_client = bound.arguments["training_client"]
            sampling_client = await training_client.save_weights_and_get_sampling_client_async()
            return sampling_client, {}

        # Log aggregated meta-task pipeline stats to W&B before the train step.
        # commit=False so they're committed alongside tinker_cookbook's training metrics.
        batch_meta = _aggregate_batch_meta_stats(trajectory_groups)
        if batch_meta:
            n = batch_meta["total"]
            _LOGGER.info(
                "TRAIN STEP | training on %d groups | meta: patches=%d/%d valid=%d/%d hard=%d/%d",
                len(trajectory_groups),
                batch_meta["patches"],
                n,
                batch_meta["valid"],
                n,
                batch_meta["valid"] - batch_meta["frontier_solved"],
                n,
            )
        else:
            _LOGGER.info("TRAIN STEP | training on %d groups", len(trajectory_groups))
        _log_meta_stats_to_wandb(batch_meta)

        return await _original_do_train_step(*bound.args, **bound.kwargs)

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
        tinker_train.remove_constant_reward_groups = _original_remove_constant_reward_groups  # type: ignore[assignment]
        tinker_train.do_train_step_and_get_sampling_client = _original_do_train_step  # type: ignore[assignment]
        tinker_train.do_group_rollout = _original_do_group_rollout  # type: ignore[assignment]
        _wbcfg.Config.update = _orig_config_update  # type: ignore[assignment]
