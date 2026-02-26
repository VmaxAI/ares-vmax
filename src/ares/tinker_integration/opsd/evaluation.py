"""Student and teacher evaluation phases for OPSD.

Runs the model on all tasks, collects per-task rewards and trajectories,
and identifies hard tasks (0% success across all group rollouts).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
import importlib
import logging
from typing import Any

from ares.tinker_integration.opsd import config as opsd_config_mod

_LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TaskEvalResult:
    """Per-task evaluation result."""

    task: Any
    task_name: str
    rewards: list[float]
    trajectories: list[Any]
    mean_reward: float
    all_failed: bool


@dataclass(frozen=True)
class EvalPhaseResult:
    """Aggregate evaluation result across all tasks."""

    task_results: list[TaskEvalResult]
    hard_tasks: list[TaskEvalResult]
    num_hard: int
    num_solved: int
    mean_reward: float


def _resolve_task_name(task: Any, preset_name: str | None) -> str:
    """Resolve a task to a canonical name string."""
    if isinstance(task, int):
        # Code-agent harness uses integer task indices.
        if preset_name:
            try:
                from ares import registry

                spec = registry._REGISTRY[preset_name]  # type: ignore[attr-defined]
                harbor_task = spec.ds[task]
                return str(harbor_task.name)
            except Exception:
                pass
        return f"{preset_name}:{task}" if preset_name else f"task:{task}"
    # Terminal harness uses Harbor Task objects with .name.
    return str(getattr(task, "name", str(task)))


async def _evaluate_single_task(
    sampling_client: Any,
    env_group_builder: Any,
    task: Any,
    task_name: str,
    *,
    temperature: float,
    max_tokens: int,
) -> TaskEvalResult | None:
    """Run one task's group rollout and collect results."""
    tinker_train = importlib.import_module("tinker_cookbook.rl.train")
    do_rollout = tinker_train.do_group_rollout_and_filter_constant_reward

    try:
        result = await do_rollout(
            sampling_client,
            env_group_builder,
            temperature=temperature,
            max_tokens=max_tokens,
            do_remove_constant_reward_groups=False,
        )
    except Exception as exc:
        _LOGGER.warning("EVAL | task=%s | rollout failed: %s: %s", task_name, type(exc).__name__, exc)
        return None

    if result is None:
        _LOGGER.warning("EVAL | task=%s | rollout returned None", task_name)
        return None

    rewards = result.get_total_rewards()
    trajectories = list(result.trajectories_G)
    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    all_failed = all(r == 0.0 for r in rewards) if rewards else True

    _LOGGER.info(
        "EVAL | task=%s | reward=%.3f [%s] | all_failed=%s",
        task_name,
        mean_reward,
        " ".join(f"{r:.2f}" for r in rewards),
        all_failed,
    )

    return TaskEvalResult(
        task=task,
        task_name=task_name,
        rewards=rewards,
        trajectories=trajectories,
        mean_reward=mean_reward,
        all_failed=all_failed,
    )


async def evaluate_tasks(
    sampling_client: Any,
    tasks: list[Any],
    config: opsd_config_mod.OPSDConfig,
    *,
    group_size: int,
    builder_factory: Any,
    phase_label: str = "eval",
) -> EvalPhaseResult:
    """Evaluate model on all tasks concurrently.

    Args:
        sampling_client: Tinker sampling client (current model weights).
        tasks: List of tasks (Harbor Task objects or integer indices).
        config: OPSD config for temperature, max_tokens, etc.
        group_size: Number of rollouts per task.
        builder_factory: Callable(task, group_size) -> EnvGroupBuilder.
        phase_label: Label for logging (e.g., "eval", "teacher").

    Returns:
        EvalPhaseResult with per-task results and hard-task identification.
    """
    _LOGGER.info(
        "=== %s PHASE | evaluating %d tasks with group_size=%d ===", phase_label.upper(), len(tasks), group_size
    )

    builders = [builder_factory(task, group_size) for task in tasks]
    task_names = [_resolve_task_name(task, config.preset_name) for task in tasks]

    coros = [
        _evaluate_single_task(
            sampling_client,
            builder,
            task,
            task_name,
            temperature=config.temperature,
            max_tokens=config.max_tokens,
        )
        for builder, task, task_name in zip(builders, tasks, task_names, strict=True)
    ]

    results = await asyncio.gather(*coros)

    task_results: list[TaskEvalResult] = [r for r in results if r is not None]
    hard_tasks = [r for r in task_results if r.all_failed]
    solved_tasks = [r for r in task_results if not r.all_failed]
    all_rewards = [r.mean_reward for r in task_results]
    mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0

    eval_result = EvalPhaseResult(
        task_results=task_results,
        hard_tasks=hard_tasks,
        num_hard=len(hard_tasks),
        num_solved=len(solved_tasks),
        mean_reward=mean_reward,
    )

    _LOGGER.info(
        "=== %s PHASE DONE | tasks=%d | solved=%d | hard=%d | mean_reward=%.3f ===",
        phase_label.upper(),
        len(task_results),
        eval_result.num_solved,
        eval_result.num_hard,
        eval_result.mean_reward,
    )

    return eval_result


def filter_teacher_solved(
    hard_tasks: list[TaskEvalResult],
    teacher_result: EvalPhaseResult,
) -> list[tuple[TaskEvalResult, TaskEvalResult]]:
    """Filter tasks where teacher solved but student failed.

    Returns:
        List of (student_result, teacher_result) pairs for distillable tasks.
    """
    teacher_by_name = {r.task_name: r for r in teacher_result.task_results}

    distillable: list[tuple[TaskEvalResult, TaskEvalResult]] = []
    for student_r in hard_tasks:
        teacher_r = teacher_by_name.get(student_r.task_name)
        if teacher_r is None:
            continue
        if not teacher_r.all_failed:
            distillable.append((student_r, teacher_r))

    _LOGGER.info(
        "FILTER | hard=%d | teacher_solved=%d | distillable=%d",
        len(hard_tasks),
        len(distillable),
        len(distillable),
    )
    return distillable
