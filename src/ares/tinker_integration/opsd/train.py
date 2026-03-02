"""OPSD training orchestrator.

Runs ``num_batches`` RL batches.  Every ``opsd_every`` batches, triggers:

1. **Identify hard tasks** from the RL batch (tasks with 0% success —
   no separate student evaluation needed).
2. **Self-Reflect** — generate compact hints from failed traces.
3. **Teacher Re-attempt** — same model + privileged context re-attempts hard tasks.
4. **Filter** — keep tasks where teacher succeeded but student failed.
5. **Distill** — ``num_distillation_steps`` gradient steps with reverse-KL
   on ALL distillable tasks (same ``group_size`` as RL).
"""

from __future__ import annotations

import asyncio
import importlib
import json
import logging
import os
from pathlib import Path
import random
import time
from typing import Any

from ares.tinker_integration import ares_env
from ares.tinker_integration import dataset
from ares.tinker_integration import monkey_patches
from ares.tinker_integration import terminal_env
from ares.tinker_integration.opsd import config as opsd_config_mod
from ares.tinker_integration.opsd import distillation
from ares.tinker_integration.opsd import evaluation as eval_mod
from ares.tinker_integration.opsd import privileged_env
from ares.tinker_integration.opsd import reflection as reflection_mod
from ares.tinker_integration.rl import train as rl_train_mod

_LOGGER = logging.getLogger(__name__)


def _make_harbor_env_config(env_type: str, *, snapshot_template_name: str | None = None) -> Any:
    """Create a Harbor EnvironmentConfig for the given type."""
    return rl_train_mod._make_harbor_env_config(env_type, snapshot_template_name=snapshot_template_name)


def _make_builder_factory(
    config: opsd_config_mod.OPSDConfig,
    renderer: Any,
    *,
    privileged: bool = False,
    reflections: dict[str, str] | None = None,
) -> Any:
    """Create a builder factory function for evaluation/distillation.

    Returns a callable ``(task, group_size) -> EnvGroupBuilder`` that creates
    the appropriate builder for the harness mode.
    """
    if config.harness == "code-agent":
        container_factory = ares_env._get_container_factory(config.env_type)

        def _code_agent_factory(task: Any, group_size: int) -> Any:
            task_idx = int(task) if isinstance(task, int) else task
            task_name = eval_mod._resolve_task_name(task, config.preset_name)

            if privileged and reflections and task_name in reflections:
                return privileged_env.PrivilegedAresEnvGroupBuilder(
                    preset_name=config.preset_name,  # type: ignore[arg-type]
                    task_idx=int(task_idx),
                    group_size=group_size,
                    renderer=renderer,
                    container_factory=container_factory,
                    reflection=reflections[task_name],
                    max_trajectory_tokens=config.max_trajectory_tokens,
                    max_tokens=config.max_tokens,
                    snapshot_template_name=config.snapshot_template_name,
                )
            return ares_env.AresEnvGroupBuilder(
                preset_name=config.preset_name,  # type: ignore[arg-type]
                task_idx=int(task_idx),
                group_size=group_size,
                renderer=renderer,
                container_factory=container_factory,
                max_trajectory_tokens=config.max_trajectory_tokens,
                max_tokens=config.max_tokens,
                snapshot_template_name=config.snapshot_template_name,
            )

        return _code_agent_factory

    # Terminal harness.
    env_cfg = _make_harbor_env_config(config.env_type, snapshot_template_name=config.snapshot_template_name)
    gym_env_kwargs = {
        "auto_stop_minutes": config.auto_stop_minutes,
        "sandbox_cpus": config.sandbox_cpus,
        "sandbox_memory_gb": config.sandbox_memory_gb,
        "sandbox_disk_gb": config.sandbox_disk_gb,
    }

    def _terminal_factory(task: Any, group_size: int) -> Any:
        task_name = eval_mod._resolve_task_name(task, config.preset_name)

        if privileged and reflections and task_name in reflections:
            return privileged_env.PrivilegedTerminalEnvGroupBuilder(
                task=task,
                group_size=group_size,
                environment=env_cfg,
                renderer=renderer,
                reflection=reflections[task_name],
                max_trajectory_tokens=config.max_trajectory_tokens,
                max_tokens=config.max_tokens,
                gym_env_kwargs=gym_env_kwargs,
            )
        return dataset.TerminalEnvGroupBuilder(
            task=task,
            group_size=group_size,
            environment=env_cfg,
            renderer=renderer,
            max_trajectory_tokens=config.max_trajectory_tokens,
            max_tokens=config.max_tokens,
            gym_env_kwargs=gym_env_kwargs,
        )

    return _terminal_factory


def _build_task_results_from_rl_batch(
    batch_tasks: list[Any],
    trajectory_groups: list[Any],
    preset_name: str | None,
) -> list[eval_mod.TaskEvalResult]:
    """Build per-task eval results from RL batch rollout data.

    Groups rollouts by unique task (since random.choices may pick duplicates),
    aggregates rewards and trajectories, and returns TaskEvalResult objects.
    """
    # Group by task identity (int index or task object).
    task_data: dict[Any, dict[str, Any]] = {}
    for task, tg in zip(batch_tasks, trajectory_groups, strict=True):
        if tg is None:
            continue
        key = task
        if key not in task_data:
            task_data[key] = {"task": task, "rewards": [], "trajectories": []}
        task_data[key]["rewards"].extend(tg.get_total_rewards())
        task_data[key]["trajectories"].extend(tg.trajectories_G)

    results: list[eval_mod.TaskEvalResult] = []
    for info in task_data.values():
        rewards = info["rewards"]
        mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
        all_failed = all(r == 0.0 for r in rewards) if rewards else True
        task_name = eval_mod._resolve_task_name(info["task"], preset_name)
        results.append(
            eval_mod.TaskEvalResult(
                task=info["task"],
                task_name=task_name,
                rewards=rewards,
                trajectories=info["trajectories"],
                mean_reward=mean_reward,
                all_failed=all_failed,
            )
        )

    return results


async def _do_rl_batch(
    config: opsd_config_mod.OPSDConfig,
    training_client: Any,
    sampling_client: Any,
    tasks: list[Any],
    tokenizer: Any,
    ml_logger: Any,
    global_batch: int,
    builder_factory: Any,
) -> tuple[Any, list[eval_mod.TaskEvalResult]]:
    """Run a single RL training batch.

    Returns:
        (sampling_client, task_results) where task_results are per-task eval
        results built from the RL rollout data (used by OPSD phases to skip
        a separate student evaluation).
    """
    tinker_train = importlib.import_module("tinker_cookbook.rl.train")

    t_start = time.time()
    metrics: dict[str, Any] = {
        "progress/batch": global_batch,
        "opsd/phase": "rl",
    }

    # Sample tasks for this batch.
    if len(tasks) == 1:
        batch_tasks = [tasks[0]] * config.groups_per_batch
    else:
        batch_tasks = random.choices(tasks, k=config.groups_per_batch)

    builders = [builder_factory(task, config.group_size) for task in batch_tasks]

    # Run rollouts concurrently.
    do_rollout = tinker_train.do_group_rollout_and_filter_constant_reward
    trajectory_groups = await asyncio.gather(
        *[
            asyncio.create_task(
                do_rollout(
                    sampling_client,
                    builder,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    do_remove_constant_reward_groups=False,
                ),
                name=f"rl_rollout_{i}",
            )
            for i, builder in enumerate(builders)
        ],
    )

    # Build per-task results from RL data (for OPSD phases).
    task_results = _build_task_results_from_rl_batch(batch_tasks, trajectory_groups, config.preset_name)

    # Filter None trajectory groups.
    valid_pairs = [(b, tg) for b, tg in zip(builders, trajectory_groups, strict=True) if tg is not None]
    if not valid_pairs:
        _LOGGER.warning("RL | batch %d | all rollouts failed, skipping", global_batch)
        ml_logger.log_metrics(metrics, step=global_batch)
        return sampling_client, task_results

    valid_builders = [p[0] for p in valid_pairs]
    valid_trajectory_groups = [p[1] for p in valid_pairs]

    # Train step.
    sampling_client, train_metrics = await tinker_train.do_train_step_and_get_sampling_client(
        cfg=tinker_train.Config(
            model_name=config.model_name,
            log_path=config.log_path,
            dataset_builder=None,
            learning_rate=config.learning_rate,
            max_tokens=config.max_tokens,
            temperature=config.temperature,
            loss_fn=config.loss_fn,
            lora_rank=config.lora_rank,
            kl_penalty_coef=config.kl_penalty_coef,
        ),
        i_batch=global_batch,
        training_client=training_client,
        service_client=None,
        tokenizer=tokenizer,
        env_group_builders_P=valid_builders,
        trajectory_groups_P=valid_trajectory_groups,
    )

    metrics.update(train_metrics)
    metrics["time/total"] = time.time() - t_start
    ml_logger.log_metrics(metrics, step=global_batch)

    return sampling_client, task_results


async def _run_distillation_steps(
    config: opsd_config_mod.OPSDConfig,
    training_client: Any,
    sampling_client: Any,
    distillable: list[tuple[eval_mod.TaskEvalResult, eval_mod.TaskEvalResult]],
    reflections: dict[str, str],
    renderer: Any,
    ml_logger: Any,
    global_batch: int,
    distill_buffer: list[dict[str, Any]] | None = None,
) -> tuple[Any, int]:
    """Run distillation gradient steps on ALL distillable tasks.

    Each step: student rollouts (standard) on all distillable tasks with the
    same ``group_size`` as RL, compute teacher KL, and train.

    If ``config.distill_min_batch_size > 0``, data is accumulated in
    ``distill_buffer`` and training only proceeds when the buffer contains
    enough datums.  This prevents catastrophically small gradient updates.
    """
    tinker_train = importlib.import_module("tinker_cookbook.rl.train")
    rl_data = importlib.import_module("tinker_cookbook.rl.data_processing")
    checkpoint_utils = importlib.import_module("tinker_cookbook.checkpoint_utils")

    # Standard (non-privileged) builder factory for student rollouts.
    builder_factory = _make_builder_factory(config, renderer)

    # All distillable tasks and their names.
    distill_tasks = [student_r.task for student_r, _ in distillable]
    distill_task_names = [student_r.task_name for student_r, _ in distillable]

    _LOGGER.info(
        "=== DISTILL | tasks=%d | steps=%d | group_size=%d | min_batch=%d ===",
        len(distill_tasks),
        config.num_distillation_steps,
        config.group_size,
        config.distill_min_batch_size,
    )

    # --- Single rollout: student generates on-policy once ---
    t_rollout = time.time()
    builders = [builder_factory(task, config.group_size) for task in distill_tasks]

    do_rollout = tinker_train.do_group_rollout_and_filter_constant_reward
    trajectory_groups = await asyncio.gather(
        *[
            asyncio.create_task(
                do_rollout(
                    sampling_client,
                    builder,
                    temperature=config.temperature,
                    max_tokens=config.max_tokens,
                    do_remove_constant_reward_groups=False,
                ),
                name=f"distill_rollout_{i}",
            )
            for i, builder in enumerate(builders)
        ],
    )

    # Filter None groups.
    valid_triples = [
        (b, tg, name)
        for b, tg, name in zip(builders, trajectory_groups, distill_task_names, strict=True)
        if tg is not None
    ]
    if not valid_triples:
        _LOGGER.warning("DISTILL | all rollouts failed, skipping distillation")
        return sampling_client, global_batch

    valid_builders = [t[0] for t in valid_triples]
    valid_trajectory_groups = [t[1] for t in valid_triples]
    valid_task_names = [t[2] for t in valid_triples]

    # Compute advantages and assemble training data (once).
    taglist = [b.logging_tags() for b in valid_builders]
    traj_metrics = importlib.import_module("tinker_cookbook.rl.metric_util").compute_trajectory_metrics(
        valid_trajectory_groups, taglist
    )

    advantages = rl_data.compute_advantages(valid_trajectory_groups)
    data_d, metadata_d = rl_data.assemble_training_data(valid_trajectory_groups, advantages)

    if not data_d:
        _LOGGER.warning("DISTILL | no training data after assembly, skipping")
        return sampling_client, global_batch

    # Map each datum to its task name.
    task_names_for_data = [valid_task_names[m["group_idx"]] for m in metadata_d]

    # Compute teacher KL penalty (once — same on-policy data, same reflections).
    kl_metrics = await distillation.incorporate_teacher_kl(
        data_d,
        reflections,
        task_names_for_data,
        sampling_client,
        renderer,
        config,
    )

    rollout_time = time.time() - t_rollout
    _LOGGER.info(
        "DISTILL | rollout done | valid_groups=%d | datums=%d | rollout_time=%.1fs",
        len(valid_triples),
        len(data_d),
        rollout_time,
    )

    # --- Check minimum batch size threshold ---
    if config.distill_min_batch_size > 0 and distill_buffer is not None:
        distill_buffer.extend(data_d)
        _LOGGER.info(
            "DISTILL | accumulated %d datums in buffer (total=%d, min=%d)",
            len(data_d),
            len(distill_buffer),
            config.distill_min_batch_size,
        )
        if len(distill_buffer) < config.distill_min_batch_size:
            _LOGGER.info(
                "DISTILL | buffer below threshold (%d < %d), deferring training",
                len(distill_buffer),
                config.distill_min_batch_size,
            )
            # Log that we skipped but accumulated.
            skip_metrics: dict[str, Any] = {
                "progress/batch": global_batch,
                "opsd/phase": "distill_deferred",
                "opsd/distill/buffer_size": len(distill_buffer),
                "opsd/distill/num_datums_this_cycle": len(data_d),
                "opsd/distill/num_valid_groups": len(valid_triples),
                "opsd/distill_num_tasks": len(distill_tasks),
            }
            skip_metrics.update(kl_metrics)
            ml_logger.log_metrics(skip_metrics, step=global_batch)
            return sampling_client, global_batch

        # Buffer is large enough — train on the full buffer.
        _LOGGER.info(
            "DISTILL | buffer ready (%d >= %d), training on full buffer",
            len(distill_buffer),
            config.distill_min_batch_size,
        )
        data_d = list(distill_buffer)
        distill_buffer.clear()

    distill_lr = config.effective_distill_learning_rate

    # --- Gradient steps on the on-policy data ---
    for step_idx in range(config.num_distillation_steps):
        t_start = time.time()
        metrics: dict[str, Any] = {
            "progress/batch": global_batch,
            "opsd/phase": "distill",
            "opsd/distill_step": step_idx,
            "opsd/distill_num_tasks": len(distill_tasks),
            "opsd/distill/num_valid_groups": len(valid_triples),
            "opsd/distill/num_datums": len(data_d),
            "opsd/distill/learning_rate": distill_lr,
        }
        metrics.update(traj_metrics)
        metrics.update(kl_metrics)

        # Train step on the data.
        await tinker_train.train_step(
            data_D=data_d,
            training_client=training_client,
            learning_rate=distill_lr,
            num_substeps=1,
            loss_fn=config.loss_fn,
        )
        sampling_client = await training_client.save_weights_and_get_sampling_client_async()

        metrics["time/total"] = time.time() - t_start
        if step_idx == 0:
            metrics["time/rollout"] = rollout_time
        ml_logger.log_metrics(metrics, step=global_batch)

        # Save checkpoint periodically.
        if config.save_every > 0 and (global_batch + 1) % config.save_every == 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"opsd_distill_batch{global_batch}",
                log_path=config.log_path,
                kind="both",
                loop_state={"batch": global_batch, "phase": "distill"},
            )

        global_batch += 1

    _LOGGER.info("=== DISTILL DONE | global_batch=%d ===", global_batch)
    return sampling_client, global_batch


async def _run_opsd_phases(
    config: opsd_config_mod.OPSDConfig,
    training_client: Any,
    sampling_client: Any,
    rl_batch_results: list[eval_mod.TaskEvalResult],
    renderer: Any,
    tokenizer: Any,
    ml_logger: Any,
    global_batch: int,
    distill_buffer: list[dict[str, Any]] | None = None,
    reflection_cache: dict[str, tuple[str, int]] | None = None,
) -> tuple[Any, int]:
    """Run OPSD phases: (RL results as eval) → reflect → teacher → filter → distill.

    Instead of a separate student evaluation phase, uses per-task results
    from the RL batch to identify hard tasks (0% success).  This avoids
    redundant rollouts on tasks we already have data for.

    Args:
        distill_buffer: Shared buffer that accumulates distillation datums
            across OPSD cycles when ``config.distill_min_batch_size > 0``.
        reflection_cache: Mapping of task_name → (reflection_text, age_in_cycles).
            Reused for up to ``config.reflection_cache_cycles`` consecutive cycles
            to save LLM calls and speed up OPSD phases.
    """
    checkpoint_utils = importlib.import_module("tinker_cookbook.checkpoint_utils")

    # Use RL batch results as student eval (no separate evaluation phase).
    hard_tasks = [r for r in rl_batch_results if r.all_failed]
    solved_tasks = [r for r in rl_batch_results if not r.all_failed]
    all_rewards = [r.mean_reward for r in rl_batch_results]
    mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    # Log evaluation metrics (from RL batch data).
    eval_metrics: dict[str, Any] = {
        "opsd/eval/total_tasks": len(rl_batch_results),
        "opsd/eval/num_hard": len(hard_tasks),
        "opsd/eval/num_solved": len(solved_tasks),
        "opsd/eval/solve_rate": len(solved_tasks) / max(1, len(rl_batch_results)),
        "opsd/eval/mean_reward": mean_reward,
        "opsd/eval/source": "rl_batch",
        "opsd/phase": "eval",
    }
    for tr in rl_batch_results:
        eval_metrics[f"opsd/task/{tr.task_name}/student_reward"] = tr.mean_reward
        eval_metrics[f"opsd/task/{tr.task_name}/is_hard"] = float(tr.all_failed)
    ml_logger.log_metrics(eval_metrics, step=global_batch)

    _LOGGER.info(
        "OPSD EVAL (from RL batch) | tasks=%d | solved=%d | hard=%d | mean_reward=%.3f",
        len(rl_batch_results),
        len(solved_tasks),
        len(hard_tasks),
        mean_reward,
    )

    _save_opsd_state(config.log_path, "eval_done", global_batch, hard_task_names=[r.task_name for r in hard_tasks])

    # Early exit: no hard tasks.
    if not hard_tasks:
        _LOGGER.info("No hard tasks — all solved! Skipping OPSD phases.")
        return sampling_client, global_batch

    # Phase 2: Self-reflection (with caching).
    # Check which tasks already have a fresh-enough cached reflection.
    tasks_needing_reflection: list[eval_mod.TaskEvalResult] = []
    cached_reflections: dict[str, str] = {}
    cache_max_age = config.reflection_cache_cycles

    if reflection_cache is not None and cache_max_age > 0:
        for task_r in hard_tasks:
            cached = reflection_cache.get(task_r.task_name)
            if cached is not None:
                text, age = cached
                if age < cache_max_age:
                    cached_reflections[task_r.task_name] = text
                    # Increment age.
                    reflection_cache[task_r.task_name] = (text, age + 1)
                    continue
            tasks_needing_reflection.append(task_r)
    else:
        tasks_needing_reflection = list(hard_tasks)

    # Generate new reflections only for tasks without a cache hit.
    if tasks_needing_reflection:
        new_reflections = await reflection_mod.generate_reflections(
            sampling_client,
            tasks_needing_reflection,
            config,
            renderer,
            tokenizer,
        )
        # Store in cache.
        if reflection_cache is not None:
            for task_name, text in new_reflections.items():
                reflection_cache[task_name] = (text, 0)
        cached_reflections.update(new_reflections)

    reflections = cached_reflections

    _LOGGER.info(
        "REFLECTION | total=%d | cached=%d | new=%d",
        len(reflections),
        len(hard_tasks) - len(tasks_needing_reflection),
        len(tasks_needing_reflection),
    )

    reflection_metrics: dict[str, Any] = {
        "opsd/reflection/num_tasks": len(hard_tasks),
        "opsd/reflection/num_generated": len(tasks_needing_reflection),
        "opsd/reflection/num_cached": len(hard_tasks) - len(tasks_needing_reflection),
        "opsd/phase": "reflection",
    }
    ml_logger.log_metrics(reflection_metrics, step=global_batch)

    _save_opsd_state(
        config.log_path,
        "reflection_done",
        global_batch,
        hard_task_names=[r.task_name for r in hard_tasks],
        reflections=reflections,
    )

    # Phase 3: Teacher re-attempts (same model + privileged context).
    teacher_builder_factory = _make_builder_factory(
        config,
        renderer,
        privileged=True,
        reflections=reflections,
    )
    teacher_tasks = [r.task for r in hard_tasks]
    teacher_result = await eval_mod.evaluate_tasks(
        sampling_client,
        teacher_tasks,
        config,
        group_size=config.teacher_group_size,
        builder_factory=teacher_builder_factory,
        phase_label="teacher_eval",
    )

    # Log teacher metrics.
    teacher_metrics: dict[str, Any] = {
        "opsd/teacher/num_attempted": len(hard_tasks),
        "opsd/teacher/num_solved": teacher_result.num_solved,
        "opsd/teacher/solve_rate_on_hard": teacher_result.num_solved / max(1, len(hard_tasks)),
        "opsd/teacher/mean_reward": teacher_result.mean_reward,
        "opsd/phase": "teacher",
    }
    for tr in teacher_result.task_results:
        teacher_metrics[f"opsd/task/{tr.task_name}/teacher_reward"] = tr.mean_reward
        teacher_metrics[f"opsd/task/{tr.task_name}/teacher_solved"] = float(not tr.all_failed)
    ml_logger.log_metrics(teacher_metrics, step=global_batch)

    # Phase 4: Filter teacher-solved.
    distillable = eval_mod.filter_teacher_solved(hard_tasks, teacher_result)
    if not distillable:
        _LOGGER.info("Teacher couldn't solve any hard tasks. Skipping distillation.")
        return sampling_client, global_batch

    # Phase 5: Distillation gradient steps.
    sampling_client, global_batch = await _run_distillation_steps(
        config,
        training_client,
        sampling_client,
        distillable,
        reflections,
        renderer,
        ml_logger,
        global_batch,
        distill_buffer=distill_buffer,
    )

    _save_opsd_state(config.log_path, "distill_done", global_batch)

    # Save post-OPSD checkpoint.
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name=f"opsd_batch{global_batch}",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": global_batch, "phase": "opsd_complete"},
    )

    return sampling_client, global_batch


def _save_opsd_state(
    log_path: str,
    phase: str,
    global_batch: int,
    hard_task_names: list[str] | None = None,
    reflections: dict[str, str] | None = None,
) -> None:
    """Save OPSD state for resume."""
    state = {
        "phase": phase,
        "global_batch": global_batch,
        "hard_task_names": hard_task_names or [],
        "reflections": reflections or {},
    }
    state_path = Path(log_path) / "opsd_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))
    _LOGGER.info("OPSD STATE | saved to %s | phase=%s | batch=%d", state_path, phase, global_batch)


async def run_opsd_training(config: opsd_config_mod.OPSDConfig) -> None:
    """Run the full OPSD training loop.

    Runs ``num_batches`` RL batches.  Every ``opsd_every`` batches, triggers
    the OPSD phases (eval → reflect → teacher → filter → distill).
    """
    config.validate()

    # Fail fast if env vars aren't set.
    if "TINKER_API_KEY" not in os.environ:
        raise ValueError("TINKER_API_KEY environment variable is not set")
    if config.env_type == "daytona" and "DAYTONA_API_KEY" not in os.environ:
        raise ValueError("DAYTONA_API_KEY environment variable is not set (required for daytona env_type)")

    # Override Daytona auto-stop interval.
    if config.env_type == "daytona":
        os.environ["DAYTONA_AUTO_STOP_INTERVAL"] = str(config.auto_stop_minutes)
        from ares import config as ares_config_mod

        ares_config_mod.reload()

    # Limit concurrent sandbox creation.
    terminal_env.set_max_concurrent_sandboxes(config.max_concurrent_sandboxes)

    # Suppress noisy logging.
    for noisy_logger in ("harbor", "daytona_sdk", "daytona", "httpx"):
        logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    # Import tinker modules.
    tinker = importlib.import_module("tinker")
    checkpoint_utils = importlib.import_module("tinker_cookbook.checkpoint_utils")
    ml_log = importlib.import_module("tinker_cookbook.utils.ml_log")
    tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
    renderers_mod = importlib.import_module("tinker_cookbook.renderers")

    # Load tasks.
    if config.harness == "code-agent":
        assert config.preset_name is not None
        import ares as ares_mod

        preset_info = ares_mod.info(config.preset_name)
        total_tasks = preset_info.num_tasks
        if config.num_tasks is not None:
            total_tasks = min(total_tasks, config.num_tasks)
        tasks: list[Any] = list(range(total_tasks))
        _LOGGER.info("Loaded %d task indices from preset '%s'", len(tasks), config.preset_name)
    else:
        if config.task_dir:
            tasks = dataset.load_tasks_from_task_dir(config.task_dir)
        else:
            assert config.preset_name is not None
            tasks = dataset.load_tasks_from_preset(config.preset_name, num_tasks=config.num_tasks)
        _LOGGER.info("Loaded %d tasks", len(tasks))

    # Setup tinker service.
    service_client = tinker.ServiceClient(base_url=config.base_url)
    training_client = await service_client.create_lora_training_client_async(config.model_name, rank=config.lora_rank)

    # Load checkpoint if resuming.
    if config.load_checkpoint_path:
        future = await training_client.load_state_with_optimizer_async(config.load_checkpoint_path)
        _ = await future.result_async()
        _LOGGER.info("Loaded checkpoint from %s", config.load_checkpoint_path)

    # Setup tokenizer and renderer.
    tokenizer = tokenizer_utils.get_tokenizer(config.model_name)
    renderer_name = config.renderer_name or config.model_name
    renderer = renderers_mod.get_renderer(renderer_name, tokenizer=tokenizer)

    # Setup wandb logging.
    wandb_name = config.wandb_name or f"opsd-{config.preset_name or 'custom'}-{config.model_name.split('/')[-1]}"
    ml_logger = ml_log.setup_logging(
        log_dir=config.log_path,
        wandb_project=config.wandb_project,
        config=config,
        wandb_name=wandb_name,
    )

    # Get initial sampling client.
    sampling_client = await training_client.save_weights_and_get_sampling_client_async()
    global_batch = 0

    builder_factory = _make_builder_factory(config, renderer)

    # Distillation replay buffer: accumulates datums across OPSD cycles
    # until we have enough for a stable gradient update.
    distill_buffer: list[dict[str, Any]] = []

    # Reflection cache: avoids re-generating reflections for the same hard
    # tasks across consecutive OPSD cycles.  Maps task_name → (text, age).
    reflection_cache: dict[str, tuple[str, int]] = {}

    _LOGGER.info(
        "Starting OPSD training: harness=%s, model=%s, tasks=%d, num_batches=%d, opsd_every=%d",
        config.harness,
        config.model_name,
        len(tasks),
        config.num_batches,
        config.opsd_every,
    )
    if config.distill_min_batch_size > 0:
        _LOGGER.info("Distillation replay buffer enabled: min_batch_size=%d", config.distill_min_batch_size)
    if config.effective_distill_learning_rate != config.learning_rate:
        _LOGGER.info(
            "Distillation learning rate: %s (RL: %s)",
            config.effective_distill_learning_rate,
            config.learning_rate,
        )

    # Apply monkey-patches for the entire OPSD training.
    with monkey_patches.MonkeyPatchContext(
        grad_clip_norm=config.grad_clip_norm,
        rollout_max_retries=config.async_rollout_retries,
    ):
        for batch_idx in range(config.num_batches):
            _LOGGER.info(
                "========== RL BATCH %d/%d | global_batch=%d ==========",
                batch_idx + 1,
                config.num_batches,
                global_batch,
            )

            # RL batch.
            sampling_client, rl_batch_results = await _do_rl_batch(
                config,
                training_client,
                sampling_client,
                tasks,
                tokenizer,
                ml_logger,
                global_batch,
                builder_factory,
            )

            # Save checkpoint periodically.
            if config.save_every > 0 and (global_batch + 1) % config.save_every == 0:
                await checkpoint_utils.save_checkpoint_async(
                    training_client=training_client,
                    name=f"opsd_rl_batch{global_batch}",
                    log_path=config.log_path,
                    kind="both",
                    loop_state={"batch": global_batch, "phase": "rl"},
                )

            global_batch += 1

            # OPSD phases every N batches — uses RL batch results as student
            # eval (no separate evaluation on all tasks).
            if (batch_idx + 1) % config.opsd_every == 0:
                _LOGGER.info("========== OPSD PHASES after batch %d ==========", batch_idx + 1)
                sampling_client, global_batch = await _run_opsd_phases(
                    config,
                    training_client,
                    sampling_client,
                    rl_batch_results,
                    renderer,
                    tokenizer,
                    ml_logger,
                    global_batch,
                    distill_buffer=distill_buffer,
                    reflection_cache=reflection_cache,
                )

    # Save final checkpoint.
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="opsd_final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": global_batch, "phase": "final"},
    )

    ml_logger.close()
    _LOGGER.info("OPSD training completed. Total batches: %d", global_batch)
