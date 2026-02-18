"""Tinker Env adapter wrapping ARES CodeEnvironment for RL training.

Ported from examples/05_tinker_train.py (TinkerCompatibleEnv pattern). Provides the
``AresCodeTinkerEnv`` class that wraps an ARES ``CodeEnvironment`` (created via
``ares.make()``) as a Tinker-compatible RL environment, enabling training with any
ARES agent harness (Mini-SWE-Agent, Terminus2, etc.) on any preset.
"""

from __future__ import annotations

from collections.abc import Sequence
import contextlib
import importlib
import logging
from typing import Any

import ares
from ares.containers import containers
from ares.containers import daytona
from ares.containers import docker
from ares.llms import response
from ares.tinker_integration import dataset
from ares.tinker_integration import tinker_env

_LOGGER = logging.getLogger(__name__)

CONTEXT_LEN_BUFFER = 10


def _get_text_content(message: dict[str, Any]) -> str:
    """Extract text content from a renderer Message, stripping thinking parts."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")  # type: ignore[index]


def _middle_truncate(model_input: Any, max_context_len: int) -> Any:
    """Truncate model input from the middle when exceeding max context length.

    Preserves both the beginning (task context) and end (recent history)
    of the conversation while removing middle content.
    """
    tinker = importlib.import_module("tinker")

    num_tokens_to_truncate = model_input.length - max_context_len + CONTEXT_LEN_BUFFER
    if num_tokens_to_truncate <= 0:
        return model_input

    center_idx = model_input.length // 2
    truncate_start_idx = center_idx - num_tokens_to_truncate // 2
    truncate_end_idx = center_idx + num_tokens_to_truncate // 2

    curr_ints = model_input.to_ints()
    new_ints = curr_ints[:truncate_start_idx] + curr_ints[truncate_end_idx:]
    return tinker.ModelInput.from_ints(new_ints)


def _get_container_factory(env_type: str) -> containers.ContainerFactory:
    """Map env_type string to the appropriate container factory class."""
    if env_type == "daytona":
        return daytona.DaytonaContainer
    if env_type == "docker":
        return docker.DockerContainer
    raise ValueError(f"Unknown env_type: {env_type!r}. Expected 'daytona' or 'docker'.")


class AresCodeTinkerEnv:
    """Adapt ARES CodeEnvironment to tinker-cookbook's RL Env interface.

    This wraps an ARES environment (created via ``ares.make()``) so that:
    - Observation: a tinker.ModelInput created by a renderer over the LLM request messages.
    - Action: model completion tokens (list[int]) that decode into assistant text.
    - The assistant text is converted to an ARES LLMResponse and fed back to the env.
    - Reward comes from the ARES environment's reward computation.

    This enables training any ARES agent harness (Mini-SWE-Agent, Terminus2, etc.)
    with Tinker's RL infrastructure.
    """

    def __init__(
        self,
        *,
        env: ares.Environment,  # type: ignore[type-arg]
        renderer: tinker_env.RendererProtocol,
        max_trajectory_tokens: int = 32 * 1024,
        max_tokens: int = 4096,
    ):
        try:  # pragma: no cover
            importlib.import_module("tinker")
            importlib.import_module("tinker_cookbook")
        except Exception as e:  # pragma: no cover
            raise ImportError("AresCodeTinkerEnv requires 'tinker' + 'tinker-cookbook' to be installed.") from e

        self._env = env
        self._renderer = renderer
        self._max_trajectory_tokens = int(max_trajectory_tokens)
        self._max_tokens = max(0, int(max_tokens))
        self._closed = False

    def _ts_to_model_input(self, ts: ares.TimeStep) -> Any:  # type: ignore[type-arg]
        """Convert a TimeStep's observation (LLMRequest) to a tinker.ModelInput."""
        tinker = importlib.import_module("tinker")

        if ts.observation is None:
            return tinker.ModelInput.empty()

        messages: list[dict[str, Any]] = [
            {"role": msg["role"], "content": msg.get("content", "")} for msg in ts.observation.messages
        ]
        model_input = self._renderer.build_generation_prompt(messages)
        return self._fit_context(model_input)

    def _fit_context(self, model_input: Any) -> Any:
        """Apply middle truncation when exceeding context budget."""
        max_context = self._max_trajectory_tokens - self._max_tokens
        if model_input.length > max_context:
            model_input = _middle_truncate(model_input, max_context)
        return model_input

    @property
    def stop_condition(self) -> tinker_env.StopCondition:
        return self._renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Any, tinker_env.StopCondition]:
        await self._env.__aenter__()
        ts = await self._env.reset()
        return self._ts_to_model_input(ts), self.stop_condition

    async def step(self, action: list[int]) -> Any:
        tinker = importlib.import_module("tinker")
        step_result_cls = importlib.import_module("tinker_cookbook.rl.types").StepResult

        # Decode assistant message using renderer.
        message, parse_success = self._renderer.parse_response(action)
        assistant_text = _get_text_content(message)

        # Construct ARES LLMResponse.
        ares_action = response.LLMResponse(
            data=[response.TextData(content=assistant_text)],
            cost=0.0,
            usage=response.Usage(prompt_tokens=-1, generated_tokens=-1),
        )

        # Step the ARES environment.
        ts = await self._env.step(ares_action)

        episode_done = ts.last()
        reward = ts.reward or 0.0

        if episode_done:
            await self._env.__aexit__(None, None, None)
            self._closed = True
            next_observation = tinker.ModelInput.empty()
        else:
            next_observation = self._ts_to_model_input(ts)

        return step_result_cls(
            reward=reward,
            episode_done=episode_done,
            next_observation=next_observation,
            next_stop_condition=self.stop_condition,
            metrics={
                "parse_success": float(bool(parse_success)),
                "reward": reward,
            },
        )

    async def close(self) -> None:
        """Close the underlying ARES environment (idempotent)."""
        if self._closed:
            return
        self._closed = True
        with contextlib.suppress(Exception):
            await self._env.__aexit__(None, None, None)


class AresEnvGroupBuilder:
    """Build a group of AresCodeTinkerEnv instances for a single task index.

    Creates ``group_size`` environments for the *same* task (via ``ares.make(preset:idx)``),
    collects trajectories, and lets the RL algorithm center rewards within the group.
    """

    def __init__(
        self,
        *,
        preset_name: str,
        task_idx: int,
        group_size: int,
        renderer: tinker_env.RendererProtocol,
        container_factory: containers.ContainerFactory,
        max_trajectory_tokens: int = 32 * 1024,
        max_tokens: int = 4096,
        snapshot_template_name: str | None = None,
    ):
        self._preset_name = preset_name
        self._task_idx = int(task_idx)
        self._group_size = int(group_size)
        if self._group_size <= 0:
            raise ValueError("group_size must be positive")

        self._renderer = renderer
        self._container_factory = container_factory
        self._max_trajectory_tokens = int(max_trajectory_tokens)
        self._max_tokens = int(max_tokens)
        self._snapshot_template_name = snapshot_template_name

    async def make_envs(self) -> Sequence[AresCodeTinkerEnv]:
        importlib.import_module("tinker")
        importlib.import_module("tinker_cookbook")

        envs: list[AresCodeTinkerEnv] = []
        for _ in range(self._group_size):
            env = ares.make(
                f"{self._preset_name}:{self._task_idx}",
                container_factory=self._container_factory,
                snapshot_template_name=self._snapshot_template_name,
            )
            envs.append(
                AresCodeTinkerEnv(
                    env=env,
                    renderer=self._renderer,
                    max_trajectory_tokens=self._max_trajectory_tokens,
                    max_tokens=self._max_tokens,
                )
            )
        return envs

    async def compute_group_rewards(
        self,
        trajectory_group: list[Any],
        env_group: Sequence[AresCodeTinkerEnv],  # noqa: ARG002
    ) -> list[tuple[float, dict[str, Any]]]:
        return [(0.0, {}) for _ in range(len(trajectory_group))]

    def logging_tags(self) -> list[str]:
        return ["ares-code-agent", f"{self._preset_name}:{self._task_idx}"]


class AresRLDatasetBuilder:
    """tinker-cookbook-compatible RLDatasetBuilder for ARES CodeEnvironment tasks.

    Creates a ``TerminalRLDataset`` (reused from the terminal harness — it's generic)
    with integer task indices and ARES group builder thunks.
    """

    def __init__(
        self,
        *,
        preset_name: str,
        num_tasks: int | None,
        group_size: int,
        container_factory: containers.ContainerFactory,
        renderer: tinker_env.RendererProtocol | None = None,
        renderer_name: str | None = None,
        model_name_for_tokenizer: str | None = None,
        groups_per_batch: int = 1,
        num_batches: int = 1,
        max_trajectory_tokens: int = 32 * 1024,
        max_tokens: int = 4096,
        builder_buffer: int = 0,
        snapshot_template_name: str | None = None,
    ):
        self._preset_name = preset_name
        self._num_tasks_limit = num_tasks
        self._group_size = int(group_size)
        self._container_factory = container_factory
        self._renderer = renderer
        self._renderer_name = renderer_name
        self._model_name_for_tokenizer = model_name_for_tokenizer
        self._groups_per_batch = int(groups_per_batch)
        self._num_batches = int(num_batches)
        self._max_trajectory_tokens = int(max_trajectory_tokens)
        self._max_tokens = int(max_tokens)
        self._builder_buffer = max(0, int(builder_buffer))
        self._snapshot_template_name = snapshot_template_name

    async def __call__(self) -> tuple[dataset.TerminalRLDataset, None]:
        # Resolve renderer.
        renderer = self._renderer
        if renderer is None:
            if not self._renderer_name or not self._model_name_for_tokenizer:
                raise ValueError("Provide either renderer=... or (renderer_name=..., model_name_for_tokenizer=...)")
            tokenizer_utils = importlib.import_module("tinker_cookbook.tokenizer_utils")
            tokenizer = tokenizer_utils.get_tokenizer(self._model_name_for_tokenizer)
            renderers_mod = importlib.import_module("tinker_cookbook.renderers")
            renderer = renderers_mod.get_renderer(self._renderer_name, tokenizer=tokenizer)

        # Get task count from ARES registry.
        preset_info = ares.info(self._preset_name)
        total_tasks = preset_info.num_tasks
        if self._num_tasks_limit is not None:
            total_tasks = min(total_tasks, self._num_tasks_limit)

        _LOGGER.info(
            "AresRLDatasetBuilder: preset=%s, tasks=%d, group_size=%d",
            self._preset_name,
            total_tasks,
            self._group_size,
        )

        # Use integer task indices as the "task" objects.
        tasks: list[int] = list(range(total_tasks))

        # Capture closure variables for the thunk.
        preset_name = self._preset_name
        group_size = self._group_size
        container_factory = self._container_factory
        max_trajectory_tokens = self._max_trajectory_tokens
        max_tokens = self._max_tokens
        snapshot_template_name = self._snapshot_template_name

        def thunk(task_idx: Any) -> Any:
            return AresEnvGroupBuilder(
                preset_name=preset_name,
                task_idx=int(task_idx),
                group_size=group_size,
                renderer=renderer,
                container_factory=container_factory,
                max_trajectory_tokens=max_trajectory_tokens,
                max_tokens=max_tokens,
                snapshot_template_name=snapshot_template_name,
            )

        return (
            dataset.TerminalRLDataset(
                tasks=tasks,  # type: ignore[arg-type]
                groups_per_batch=self._groups_per_batch,
                num_batches=self._num_batches,
                group_builder_thunk=thunk,  # type: ignore[arg-type]
                builder_buffer=self._builder_buffer,
            ),
            None,
        )
