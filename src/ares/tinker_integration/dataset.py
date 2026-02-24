"""Multi-task dataset layer for Tinker RL training with ARES presets.

Adapted from WORKING_TINKER/terminal_rl/wrapped_env.py (lines 276-427). The working
reference only supports a single task (``HarborSingleTaskRLDataset``). This module
extends that to support ARES's multi-task presets while keeping single-task as a
special case (when ``len(tasks) == 1``, all groups use the same task).
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
import importlib
import logging
from pathlib import Path
import random
from typing import Any

from ares.tinker_integration import terminal_env
from ares.tinker_integration import tinker_env

_LOGGER = logging.getLogger(__name__)


class TerminalEnvGroupBuilder:
    """Build a group of Harbor terminal envs for a single task.

    For each rollout group, instantiate ``group_size`` environments for the *same*
    task (different sandbox instances), collect trajectories, and let the RL algorithm
    center rewards within the group (GRPO-style).
    """

    def __init__(
        self,
        *,
        task: Any,
        group_size: int,
        environment: Any,
        renderer: tinker_env.RendererProtocol,
        max_trajectory_tokens: int = 32 * 1024,
        max_tokens: int = 4096,
        gym_env_kwargs: dict[str, Any] | None = None,
    ):
        self._task = task
        self._group_size = int(group_size)
        if self._group_size <= 0:
            raise ValueError("group_size must be positive")

        self._environment = environment
        self._renderer = renderer
        self._max_trajectory_tokens = int(max_trajectory_tokens)
        self._max_tokens = int(max_tokens)
        self._gym_env_kwargs = gym_env_kwargs or {}

    async def make_envs(self) -> Sequence[tinker_env.HarborTerminalTinkerEnv]:
        # Import check (mirrors HarborTerminalTinkerEnv behavior)
        importlib.import_module("tinker")
        importlib.import_module("tinker_cookbook")

        envs: list[tinker_env.HarborTerminalTinkerEnv] = []
        for _ in range(self._group_size):
            gym = terminal_env.AsyncTerminalGymEnv(
                task=self._task,
                environment=self._environment,
                **self._gym_env_kwargs,
            )
            envs.append(
                tinker_env.HarborTerminalTinkerEnv(
                    gym_env=gym,
                    renderer=self._renderer,
                    max_trajectory_tokens=self._max_trajectory_tokens,
                    reserved_generation_tokens=self._max_tokens,
                )
            )
        return envs

    async def compute_group_rewards(
        self,
        trajectory_group: list[Any],
        env_group: Sequence[tinker_env.HarborTerminalTinkerEnv],  # noqa: ARG002
    ) -> list[tuple[float, dict[str, Any]]]:
        # Default: no additional group reward.
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        return ["harbor-terminal", self._task.name]


class TerminalRLDataset:
    """RL dataset that yields EnvGroupBuilders sampling from multiple tasks.

    When ``len(tasks) == 1``, all groups use the same task (identical to the
    single-task working reference). With multiple tasks, each batch randomly
    samples ``groups_per_batch`` tasks.

    ``builder_buffer`` adds extra builders per batch (for async mode only).
    In async mode, the training loop needs exactly ``groups_per_batch`` non-None
    groups per step.  If any rollout permanently fails (returns None), that builder
    is lost forever.  The buffer provides spare builders so the training loop can
    still accumulate enough groups even after some losses.
    """

    def __init__(
        self,
        *,
        tasks: list[Any],
        num_batches: int,
        groups_per_batch: int,
        group_builder_thunk: Callable[[Any], TerminalEnvGroupBuilder],
        builder_buffer: int = 0,
    ):
        if not tasks:
            raise ValueError("tasks list must not be empty")
        self._tasks = tasks
        self._num_batches = int(num_batches)
        self._groups_per_batch = int(groups_per_batch)
        self._builder_buffer = max(0, int(builder_buffer))
        self._group_builder_thunk = group_builder_thunk

    def __len__(self) -> int:
        return self._num_batches

    def get_batch(self, index: int) -> list[TerminalEnvGroupBuilder]:
        if index < 0 or index >= self._num_batches:
            raise IndexError("batch index out of range")

        effective_groups = self._groups_per_batch + self._builder_buffer

        # Sample tasks for this batch. With a single task, all groups use it.
        if len(self._tasks) == 1:
            sampled_tasks = [self._tasks[0]] * effective_groups
        else:
            sampled_tasks = random.choices(self._tasks, k=effective_groups)

        return [self._group_builder_thunk(task) for task in sampled_tasks]


class TerminalRLDatasetBuilder:
    """tinker-cookbook-compatible RLDatasetBuilder for terminal tasks.

    Supports both single-task (via ``task_dir``) and multi-task (via ``tasks``
    list from ARES's ``load_harbor_dataset()``).
    """

    def __init__(
        self,
        *,
        tasks: list[Any],
        group_size: int,
        environment: Any,
        renderer: tinker_env.RendererProtocol | None = None,
        renderer_name: str | None = None,
        model_name_for_tokenizer: str | None = None,
        groups_per_batch: int = 1,
        num_batches: int = 1,
        max_trajectory_tokens: int = 32 * 1024,
        max_tokens: int = 4096,
        gym_env_kwargs: Mapping[str, Any] | None = None,
        builder_buffer: int = 0,
    ):
        if not tasks:
            raise ValueError("tasks list must not be empty")
        self._tasks = tasks
        self._group_size = int(group_size)
        self._environment = environment
        self._renderer = renderer
        self._renderer_name = renderer_name
        self._model_name_for_tokenizer = model_name_for_tokenizer
        self._groups_per_batch = int(groups_per_batch)
        self._num_batches = int(num_batches)
        self._max_trajectory_tokens = int(max_trajectory_tokens)
        self._max_tokens = int(max_tokens)
        self._gym_env_kwargs: dict[str, Any] = dict(gym_env_kwargs or {})
        self._builder_buffer = max(0, int(builder_buffer))

    async def __call__(self) -> tuple[TerminalRLDataset, None]:
        # Return (train_dataset, test_dataset|None)
        renderer = self._renderer
        if renderer is None:
            if not self._renderer_name or not self._model_name_for_tokenizer:
                raise ValueError("Provide either renderer=... or (renderer_name=..., model_name_for_tokenizer=...)")
            tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
            get_tokenizer = tokenizer_utils.get_tokenizer
            tokenizer = get_tokenizer(self._model_name_for_tokenizer)

            renderers_mod = importlib.import_module("tinker_cookbook.renderers")
            get_renderer = renderers_mod.get_renderer
            renderer = get_renderer(self._renderer_name, tokenizer=tokenizer)

        gym_env_kwargs = self._gym_env_kwargs
        group_size = self._group_size
        max_trajectory_tokens = self._max_trajectory_tokens
        max_tokens = self._max_tokens
        environment = self._environment

        def thunk(task: Any) -> TerminalEnvGroupBuilder:
            return TerminalEnvGroupBuilder(
                task=task,
                group_size=group_size,
                environment=environment,
                renderer=renderer,
                max_trajectory_tokens=max_trajectory_tokens,
                max_tokens=max_tokens,
                gym_env_kwargs=gym_env_kwargs,
            )

        return (
            TerminalRLDataset(
                tasks=self._tasks,
                groups_per_batch=self._groups_per_batch,
                num_batches=self._num_batches,
                group_builder_thunk=thunk,
                builder_buffer=self._builder_buffer,
            ),
            None,
        )


def load_tasks_from_preset(preset_name: str, num_tasks: int | None = None) -> list[Any]:
    """Load Harbor tasks from an ARES preset name.

    Uses ARES's ``load_harbor_dataset()`` under the hood, looking up the
    dataset name and version from the registered preset.

    Args:
        preset_name: ARES preset name (e.g., "sbv-terminus2", "tbench-terminus2").
        num_tasks: Optional limit on number of tasks to return.

    Returns:
        List of ``harbor.models.task.task.Task`` objects.
    """
    from ares import registry

    preset_info = registry.info(preset_name)
    spec = registry._REGISTRY[preset_name]  # type: ignore[attr-defined]

    # HarborSpec caches the dataset in .ds
    tasks = spec.ds

    if num_tasks is not None:
        tasks = tasks[:num_tasks]

    _LOGGER.info(
        "Loaded %d tasks from preset '%s' (total available: %d)",
        len(tasks),
        preset_name,
        preset_info.num_tasks,
    )
    return tasks


def load_tasks_from_task_dir(task_dir: str | Path) -> list[Any]:
    """Load a single Harbor task from a task directory path.

    Args:
        task_dir: Path to the Harbor task directory.

    Returns:
        List containing a single ``harbor.models.task.task.Task`` object.
    """
    harbor = terminal_env._import_harbor()
    task_cls = harbor["Task"]
    return [task_cls(task_dir=task_dir)]
