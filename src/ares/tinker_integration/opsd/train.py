"""OPSD training orchestrator.

Implements the iterative phasic training loop:

1. **Student RL** — Standard RL training for ``rl_batches_per_iteration`` batches.
2. **Student Evaluation** — Run student on ALL tasks, identify hard tasks (0% success).
3. **Self-Reflection** — Generate compact hints from failed interaction traces.
4. **Teacher Re-attempt** — Same model + privileged context re-attempts hard tasks.
5. **Filter** — Keep tasks where teacher succeeded but student failed.
6. **On-Policy Distillation** — Reverse KL from teacher to student on distillable tasks.
7. **Repeat** for ``num_iterations``.
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


async def _run_rl_phase(
    config: opsd_config_mod.OPSDConfig,
    training_client: Any,
    sampling_client: Any,
    tasks: list[Any],
    renderer: Any,
    tokenizer: Any,
    ml_logger: Any,
    global_batch: int,
    iteration: int,
) -> tuple[Any, int]:
    """Run RL phase for one OPSD iteration.

    Instead of calling the full tinker_cookbook main loop, we replicate the
    sync training loop for ``rl_batches_per_iteration`` batches to maintain
    shared training_client and unified logging.
    """
    tinker_train = importlib.import_module("tinker_cookbook.rl.train")
    checkpoint_utils = importlib.import_module("tinker_cookbook.checkpoint_utils")

    builder_factory = _make_builder_factory(config, renderer)

    _LOGGER.info(
        "=== RL PHASE | iteration=%d | batches=%d | groups_per_batch=%d ===",
        iteration + 1,
        config.rl_batches_per_iteration,
        config.groups_per_batch,
    )

    for batch_idx in range(config.rl_batches_per_iteration):
        t_start = time.time()
        metrics: dict[str, Any] = {
            "progress/batch": global_batch,
            "opsd/phase": "rl",
            "opsd/iteration": iteration,
            "opsd/rl_batch": batch_idx,
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
            _LOGGER.warning("RL PHASE | batch %d | all rollouts failed, skipping", global_batch)
            global_batch += 1
            continue

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

        # Save checkpoint periodically.
        if config.save_every > 0 and (global_batch + 1) % config.save_every == 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"opsd_rl_iter{iteration}_batch{global_batch}",
                log_path=config.log_path,
                kind="both",
                loop_state={"batch": global_batch, "iteration": iteration, "phase": "rl"},
            )

        global_batch += 1

    _LOGGER.info("=== RL PHASE DONE | iteration=%d | global_batch=%d ===", iteration + 1, global_batch)
    return sampling_client, global_batch


async def _run_distillation_phase(
    config: opsd_config_mod.OPSDConfig,
    training_client: Any,
    sampling_client: Any,
    distillable: list[tuple[eval_mod.TaskEvalResult, eval_mod.TaskEvalResult]],
    reflections: dict[str, str],
    renderer: Any,
    tokenizer: Any,  # noqa: ARG001
    ml_logger: Any,
    global_batch: int,
    iteration: int,
) -> tuple[Any, int]:
    """Run distillation phase for one OPSD iteration.

    Student generates on-policy, teacher (same weights + privileged context)
    provides dense per-token supervision via reverse KL.
    """
    tinker_train = importlib.import_module("tinker_cookbook.rl.train")
    rl_data = importlib.import_module("tinker_cookbook.rl.data_processing")
    checkpoint_utils = importlib.import_module("tinker_cookbook.checkpoint_utils")

    # Standard (non-privileged) builder factory for student rollouts.
    builder_factory = _make_builder_factory(config, renderer)

    # Extract distillable tasks.
    distill_tasks = [student_r.task for student_r, _ in distillable]
    distill_task_names = [student_r.task_name for student_r, _ in distillable]

    _LOGGER.info(
        "=== DISTILL PHASE | iteration=%d | tasks=%d | batches=%d ===",
        iteration + 1,
        len(distill_tasks),
        config.distill_batches,
    )

    for batch_idx in range(config.distill_batches):
        t_start = time.time()
        metrics: dict[str, Any] = {
            "progress/batch": global_batch,
            "opsd/phase": "distill",
            "opsd/iteration": iteration,
            "opsd/distill_batch": batch_idx,
        }

        # Sample tasks for this distillation batch (with replacement if needed).
        if len(distill_tasks) >= config.distill_groups_per_batch:
            indices = random.sample(range(len(distill_tasks)), config.distill_groups_per_batch)
        else:
            indices = random.choices(range(len(distill_tasks)), k=config.distill_groups_per_batch)

        batch_tasks = [distill_tasks[i] for i in indices]
        batch_task_names = [distill_task_names[i] for i in indices]
        builders = [builder_factory(task, config.distill_group_size) for task in batch_tasks]

        # Run student rollouts (standard, non-privileged).
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
            for b, tg, name in zip(builders, trajectory_groups, batch_task_names, strict=True)
            if tg is not None
        ]
        if not valid_triples:
            _LOGGER.warning("DISTILL | batch %d | all rollouts failed, skipping", global_batch)
            global_batch += 1
            continue

        valid_builders = [t[0] for t in valid_triples]
        valid_trajectory_groups = [t[1] for t in valid_triples]
        valid_task_names = [t[2] for t in valid_triples]

        # Compute advantages and assemble training data.
        taglist = [b.logging_tags() for b in valid_builders]
        traj_metrics = importlib.import_module("tinker_cookbook.rl.metric_util").compute_trajectory_metrics(
            valid_trajectory_groups, taglist
        )
        metrics.update(traj_metrics)

        advantages = rl_data.compute_advantages(valid_trajectory_groups)
        data_d, metadata_d = rl_data.assemble_training_data(valid_trajectory_groups, advantages)

        if not data_d:
            _LOGGER.warning("DISTILL | batch %d | no training data, skipping", global_batch)
            global_batch += 1
            continue

        # Map each datum to its task name using metadata.
        task_names_for_data = [valid_task_names[m["group_idx"]] for m in metadata_d]

        # Incorporate teacher KL penalty.
        kl_metrics = await distillation.incorporate_teacher_kl(
            data_d,
            reflections,
            task_names_for_data,
            sampling_client,
            renderer,
            config,
        )
        metrics.update(kl_metrics)

        # Train step.
        tinker_train_step = tinker_train.train_step
        sampling_client = await tinker_train_step(
            training_client=training_client,
            data_D=data_d,
            learning_rate=config.learning_rate,
            loss_fn=config.loss_fn,
        )

        metrics["time/total"] = time.time() - t_start
        ml_logger.log_metrics(metrics, step=global_batch)

        # Save checkpoint periodically.
        if config.save_every > 0 and (global_batch + 1) % config.save_every == 0:
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"opsd_distill_iter{iteration}_batch{global_batch}",
                log_path=config.log_path,
                kind="both",
                loop_state={"batch": global_batch, "iteration": iteration, "phase": "distill"},
            )

        global_batch += 1

    _LOGGER.info("=== DISTILL PHASE DONE | iteration=%d | global_batch=%d ===", iteration + 1, global_batch)
    return sampling_client, global_batch


def _save_opsd_state(
    log_path: str,
    iteration: int,
    phase: str,
    global_batch: int,
    hard_task_names: list[str] | None = None,
    reflections: dict[str, str] | None = None,
) -> None:
    """Save OPSD iteration state for resume."""
    state = {
        "iteration": iteration,
        "phase": phase,
        "global_batch": global_batch,
        "hard_task_names": hard_task_names or [],
        "reflections": reflections or {},
    }
    state_path = Path(log_path) / "opsd_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))
    _LOGGER.info("OPSD STATE | saved to %s | iteration=%d | phase=%s", state_path, iteration, phase)


async def run_opsd_training(config: opsd_config_mod.OPSDConfig) -> None:
    """Run the full OPSD training loop.

    This is the main entry point for OPSD training. It:
    1. Sets up tinker clients, renderer, tokenizer, and wandb.
    2. Loads tasks from preset or task_dir.
    3. Runs the iterative phasic loop.
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

    _LOGGER.info(
        "Starting OPSD training: harness=%s, model=%s, tasks=%d, iterations=%d",
        config.harness,
        config.model_name,
        len(tasks),
        config.num_iterations,
    )

    # Apply monkey-patches for the entire OPSD training.
    with monkey_patches.MonkeyPatchContext(
        grad_clip_norm=config.grad_clip_norm,
        rollout_max_retries=config.async_rollout_retries,
    ):
        for iteration in range(config.num_iterations):
            _LOGGER.info(
                "========== OPSD ITERATION %d/%d | global_batch=%d ==========",
                iteration + 1,
                config.num_iterations,
                global_batch,
            )

            # Phase 1: Student RL
            sampling_client, global_batch = await _run_rl_phase(
                config,
                training_client,
                sampling_client,
                tasks,
                renderer,
                tokenizer,
                ml_logger,
                global_batch,
                iteration,
            )
            _save_opsd_state(config.log_path, iteration, "rl_done", global_batch)

            # Phase 2: Evaluate student on ALL tasks
            builder_factory = _make_builder_factory(config, renderer)
            eval_result = await eval_mod.evaluate_tasks(
                sampling_client,
                tasks,
                config,
                group_size=config.eval_group_size,
                builder_factory=builder_factory,
                phase_label=f"student_eval_iter{iteration}",
            )

            # Log evaluation metrics.
            eval_metrics = {
                "opsd/eval/total_tasks": len(eval_result.task_results),
                "opsd/eval/num_hard": eval_result.num_hard,
                "opsd/eval/num_solved": eval_result.num_solved,
                "opsd/eval/solve_rate": eval_result.num_solved / max(1, len(eval_result.task_results)),
                "opsd/eval/mean_reward": eval_result.mean_reward,
                "opsd/iteration": iteration,
                "opsd/phase": "eval",
            }
            # Per-task metrics.
            for tr in eval_result.task_results:
                eval_metrics[f"opsd/task/{tr.task_name}/student_reward"] = tr.mean_reward
                eval_metrics[f"opsd/task/{tr.task_name}/is_hard"] = float(tr.all_failed)
            ml_logger.log_metrics(eval_metrics, step=global_batch)

            _save_opsd_state(
                config.log_path,
                iteration,
                "eval_done",
                global_batch,
                hard_task_names=[r.task_name for r in eval_result.hard_tasks],
            )

            # Phase 3: Filter hard tasks
            if not eval_result.hard_tasks:
                _LOGGER.info("No hard tasks — all solved! Skipping OPSD phases for iteration %d.", iteration + 1)
                continue

            # Phase 4: Self-reflection
            reflections = await reflection_mod.generate_reflections(
                sampling_client,
                eval_result.hard_tasks,
                config,
                renderer,
                tokenizer,
            )

            reflection_metrics = {
                "opsd/reflection/num_tasks": len(eval_result.hard_tasks),
                "opsd/reflection/num_generated": len(reflections),
                "opsd/iteration": iteration,
                "opsd/phase": "reflection",
            }
            ml_logger.log_metrics(reflection_metrics, step=global_batch)

            _save_opsd_state(
                config.log_path,
                iteration,
                "reflection_done",
                global_batch,
                hard_task_names=[r.task_name for r in eval_result.hard_tasks],
                reflections=reflections,
            )

            # Phase 5: Teacher re-attempts (same model + privileged context)
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
                phase_label=f"teacher_eval_iter{iteration}",
            )

            # Log teacher metrics.
            teacher_metrics = {
                "opsd/teacher/num_attempted": len(eval_result.hard_tasks),
                "opsd/teacher/num_solved": teacher_result.num_solved,
                "opsd/teacher/solve_rate_on_hard": teacher_result.num_solved / max(1, len(eval_result.hard_tasks)),
                "opsd/teacher/mean_reward": teacher_result.mean_reward,
                "opsd/iteration": iteration,
                "opsd/phase": "teacher",
            }
            for tr in teacher_result.task_results:
                teacher_metrics[f"opsd/task/{tr.task_name}/teacher_reward"] = tr.mean_reward
                teacher_metrics[f"opsd/task/{tr.task_name}/teacher_solved"] = float(not tr.all_failed)
            ml_logger.log_metrics(teacher_metrics, step=global_batch)

            # Phase 6: Filter teacher-solved
            distillable = eval_mod.filter_teacher_solved(eval_result.hard_tasks, teacher_result)
            if not distillable:
                _LOGGER.info(
                    "Teacher couldn't solve any hard tasks. Skipping distillation for iteration %d.",
                    iteration + 1,
                )
                continue

            # Phase 7: On-policy distillation
            sampling_client, global_batch = await _run_distillation_phase(
                config,
                training_client,
                sampling_client,
                distillable,
                reflections,
                renderer,
                tokenizer,
                ml_logger,
                global_batch,
                iteration,
            )

            _save_opsd_state(config.log_path, iteration, "distill_done", global_batch)

            # Save iteration checkpoint.
            await checkpoint_utils.save_checkpoint_async(
                training_client=training_client,
                name=f"opsd_iter{iteration}_final",
                log_path=config.log_path,
                kind="both",
                loop_state={"batch": global_batch, "iteration": iteration, "phase": "complete"},
            )

    # Save final checkpoint.
    await checkpoint_utils.save_checkpoint_async(
        training_client=training_client,
        name="opsd_final",
        log_path=config.log_path,
        kind="both",
        loop_state={"batch": global_batch, "iteration": config.num_iterations, "phase": "final"},
    )

    ml_logger.close()
    _LOGGER.info("OPSD training completed. Total batches: %d", global_batch)
