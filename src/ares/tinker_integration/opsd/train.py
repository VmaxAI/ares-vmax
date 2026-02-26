"""OPSD training orchestrator.

Runs ``num_batches`` RL batches.  Every ``opsd_every`` batches, triggers:

1. **Evaluate** student on ALL tasks, identify hard tasks (0% success).
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


async def _do_rl_batch(
    config: opsd_config_mod.OPSDConfig,
    training_client: Any,
    sampling_client: Any,
    tasks: list[Any],
    tokenizer: Any,
    ml_logger: Any,
    global_batch: int,
    builder_factory: Any,
) -> Any:
    """Run a single RL training batch. Returns updated sampling_client."""
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

    # Filter None trajectory groups.
    valid_pairs = [(b, tg) for b, tg in zip(builders, trajectory_groups, strict=True) if tg is not None]
    if not valid_pairs:
        _LOGGER.warning("RL | batch %d | all rollouts failed, skipping", global_batch)
        ml_logger.log_metrics(metrics, step=global_batch)
        return sampling_client

    valid_builders = [p[0] for p in valid_pairs]
    valid_trajectory_groups = [p[1] for p in valid_pairs]

    # Train step.
    sampling_client, train_metrics = await tinker_train.do_train_step_and_get_sampling_client(
        config=tinker_train.Config(
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

    return sampling_client


async def _run_distillation_steps(
    config: opsd_config_mod.OPSDConfig,
    training_client: Any,
    sampling_client: Any,
    distillable: list[tuple[eval_mod.TaskEvalResult, eval_mod.TaskEvalResult]],
    reflections: dict[str, str],
    renderer: Any,
    ml_logger: Any,
    global_batch: int,
) -> tuple[Any, int]:
    """Run distillation gradient steps on ALL distillable tasks.

    Each step: student rollouts (standard) on all distillable tasks with the
    same ``group_size`` as RL, compute teacher KL, and train.
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
        "=== DISTILL | tasks=%d | steps=%d | group_size=%d ===",
        len(distill_tasks),
        config.num_distillation_steps,
        config.group_size,
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

    # --- Multiple gradient steps on the same on-policy data ---
    for step_idx in range(config.num_distillation_steps):
        t_start = time.time()
        metrics: dict[str, Any] = {
            "progress/batch": global_batch,
            "opsd/phase": "distill",
            "opsd/distill_step": step_idx,
            "opsd/distill_num_tasks": len(distill_tasks),
            "opsd/distill/num_valid_groups": len(valid_triples),
            "opsd/distill/num_datums": len(data_d),
        }
        metrics.update(traj_metrics)
        metrics.update(kl_metrics)

        # Train step on the same data.
        sampling_client = await tinker_train.train_step(
            training_client=training_client,
            data_D=data_d,
            learning_rate=config.learning_rate,
            loss_fn=config.loss_fn,
        )

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
    tasks: list[Any],
    renderer: Any,
    tokenizer: Any,
    ml_logger: Any,
    global_batch: int,
) -> tuple[Any, int]:
    """Run all OPSD phases: eval → reflect → teacher → filter → distill."""
    checkpoint_utils = importlib.import_module("tinker_cookbook.checkpoint_utils")

    # Phase 1: Evaluate student on ALL tasks.
    builder_factory = _make_builder_factory(config, renderer)
    eval_result = await eval_mod.evaluate_tasks(
        sampling_client,
        tasks,
        config,
        group_size=config.eval_group_size,
        builder_factory=builder_factory,
        phase_label="student_eval",
    )

    # Log evaluation metrics.
    eval_metrics: dict[str, Any] = {
        "opsd/eval/total_tasks": len(eval_result.task_results),
        "opsd/eval/num_hard": eval_result.num_hard,
        "opsd/eval/num_solved": eval_result.num_solved,
        "opsd/eval/solve_rate": eval_result.num_solved / max(1, len(eval_result.task_results)),
        "opsd/eval/mean_reward": eval_result.mean_reward,
        "opsd/phase": "eval",
    }
    for tr in eval_result.task_results:
        eval_metrics[f"opsd/task/{tr.task_name}/student_reward"] = tr.mean_reward
        eval_metrics[f"opsd/task/{tr.task_name}/is_hard"] = float(tr.all_failed)
    ml_logger.log_metrics(eval_metrics, step=global_batch)

    _save_opsd_state(
        config.log_path, "eval_done", global_batch, hard_task_names=[r.task_name for r in eval_result.hard_tasks]
    )

    # Early exit: no hard tasks.
    if not eval_result.hard_tasks:
        _LOGGER.info("No hard tasks — all solved! Skipping OPSD phases.")
        return sampling_client, global_batch

    # Phase 2: Self-reflection.
    reflections = await reflection_mod.generate_reflections(
        sampling_client,
        eval_result.hard_tasks,
        config,
        renderer,
        tokenizer,
    )

    reflection_metrics: dict[str, Any] = {
        "opsd/reflection/num_tasks": len(eval_result.hard_tasks),
        "opsd/reflection/num_generated": len(reflections),
        "opsd/phase": "reflection",
    }
    ml_logger.log_metrics(reflection_metrics, step=global_batch)

    _save_opsd_state(
        config.log_path,
        "reflection_done",
        global_batch,
        hard_task_names=[r.task_name for r in eval_result.hard_tasks],
        reflections=reflections,
    )

    # Phase 3: Teacher re-attempts (same model + privileged context).
    teacher_builder_factory = _make_builder_factory(
        config,
        renderer,
        privileged=True,
        reflections=reflections,
    )
    teacher_tasks = [r.task for r in eval_result.hard_tasks]
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
        "opsd/teacher/num_attempted": len(eval_result.hard_tasks),
        "opsd/teacher/num_solved": teacher_result.num_solved,
        "opsd/teacher/solve_rate_on_hard": teacher_result.num_solved / max(1, len(eval_result.hard_tasks)),
        "opsd/teacher/mean_reward": teacher_result.mean_reward,
        "opsd/phase": "teacher",
    }
    for tr in teacher_result.task_results:
        teacher_metrics[f"opsd/task/{tr.task_name}/teacher_reward"] = tr.mean_reward
        teacher_metrics[f"opsd/task/{tr.task_name}/teacher_solved"] = float(not tr.all_failed)
    ml_logger.log_metrics(teacher_metrics, step=global_batch)

    # Phase 4: Filter teacher-solved.
    distillable = eval_mod.filter_teacher_solved(eval_result.hard_tasks, teacher_result)
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

    _LOGGER.info(
        "Starting OPSD training: harness=%s, model=%s, tasks=%d, num_batches=%d, opsd_every=%d",
        config.harness,
        config.model_name,
        len(tasks),
        config.num_batches,
        config.opsd_every,
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
            sampling_client = await _do_rl_batch(
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

            # OPSD phases every N batches.
            if (batch_idx + 1) % config.opsd_every == 0:
                _LOGGER.info("========== OPSD PHASES after batch %d ==========", batch_idx + 1)
                sampling_client, global_batch = await _run_opsd_phases(
                    config,
                    training_client,
                    sampling_client,
                    tasks,
                    renderer,
                    tokenizer,
                    ml_logger,
                    global_batch,
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
