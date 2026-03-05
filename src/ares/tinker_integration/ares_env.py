"""Tinker Env adapter wrapping ARES CodeEnvironment for RL training.

Ported from examples/05_tinker_train.py (TinkerCompatibleEnv pattern). Provides the
``AresCodeTinkerEnv`` class that wraps an ARES ``CodeEnvironment`` (created via
``ares.make()``) as a Tinker-compatible RL environment, enabling training with any
ARES agent harness (Mini-SWE-Agent, Terminus2, etc.) on any preset.

Context overflow is handled by terminating the episode with ``reward=0`` and
``too_long=1.0`` (same strategy as the terminal harness) instead of middle-truncating
the token sequence.  Middle truncation was removed because it can cause infinite
rollouts that never produce a gradient step.
"""

from __future__ import annotations

import asyncio
from collections.abc import Sequence
import contextlib
import datetime
import importlib
import logging
import pathlib
import random
from typing import Any
import uuid

import ares
from ares.containers import containers
from ares.containers import daytona
from ares.containers import docker
from ares.llms import response
from ares.tinker_integration import dataset
from ares.tinker_integration import terminal_env
from ares.tinker_integration import tinker_env

_LOGGER = logging.getLogger(__name__)

# Retry config for sandbox creation (mirrors terminal_env's strategy).
_SANDBOX_START_MAX_RETRIES = 5
_SANDBOX_START_BASE_DELAY = 60.0  # seconds
_SANDBOX_START_MAX_DELAY = 300.0  # seconds


def _get_text_content(message: dict[str, Any]) -> str:
    """Extract text content from a renderer Message, stripping thinking parts."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")  # type: ignore[index]


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

    Context overflow terminates the episode with ``reward=0`` and ``too_long=1.0``
    (same strategy as the terminal harness).  This prevents infinite rollouts that
    would otherwise block gradient steps.
    """

    def __init__(
        self,
        *,
        env: ares.Environment,  # type: ignore[type-arg]
        renderer: tinker_env.RendererProtocol,
        max_trajectory_tokens: int = 32 * 1024,
        max_tokens: int = 4096,
        task_name: str = "unknown",
        trial_log_path: pathlib.Path | None = None,
    ):
        try:  # pragma: no cover
            importlib.import_module("tinker")
            importlib.import_module("tinker_cookbook")
        except ImportError as e:  # pragma: no cover
            raise ImportError("AresCodeTinkerEnv requires 'tinker' + 'tinker-cookbook' to be installed.") from e

        self._env = env
        self._renderer = renderer
        self._max_trajectory_tokens = int(max_trajectory_tokens)
        self._max_tokens = max(0, int(max_tokens))
        self._task_name = task_name
        self._step_count = 0
        self._closed = False
        self._trial_log_path = trial_log_path
        self._conversation: list[dict[str, str]] = []

    def _fits_context_window(self, prompt_tokens: int) -> bool:
        """Return True if prompt + reserved generation tokens fits context window."""
        return (int(prompt_tokens) + self._max_tokens) <= self._max_trajectory_tokens

    def _ts_to_model_input(self, ts: ares.TimeStep) -> Any:  # type: ignore[type-arg]
        """Convert a TimeStep's observation (LLMRequest) to a tinker.ModelInput."""
        tinker = importlib.import_module("tinker")

        if ts.observation is None:
            return tinker.ModelInput.empty()

        messages: list[dict[str, Any]] = [
            {"role": msg["role"], "content": msg.get("content", "")} for msg in ts.observation.messages
        ]
        return self._renderer.build_generation_prompt(messages)

    @property
    def stop_condition(self) -> tinker_env.StopCondition:
        return self._renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Any, tinker_env.StopCondition]:
        _LOGGER.info("ENV START | task=%s | harness=code-agent", self._task_name)

        # Gate sandbox creation with the shared concurrency semaphore (same one
        # the terminal harness uses) and retry transient Daytona errors.
        sem = terminal_env._sandbox_creation_semaphore
        for attempt in range(1, _SANDBOX_START_MAX_RETRIES + 1):
            try:
                if sem is not None:
                    async with sem:
                        await self._env.__aenter__()
                        ts = await self._env.reset()
                else:
                    await self._env.__aenter__()
                    ts = await self._env.reset()
                break
            except Exception as exc:
                if not terminal_env._is_transient_error(exc) or attempt == _SANDBOX_START_MAX_RETRIES:
                    raise
                delay = min(_SANDBOX_START_BASE_DELAY * (2 ** (attempt - 1)), _SANDBOX_START_MAX_DELAY)
                delay *= 0.5 + random.random()  # jitter
                _LOGGER.warning(
                    "ENV START RETRY | task=%s | attempt=%d/%d | retrying in %.0fs | %s: %s",
                    self._task_name,
                    attempt,
                    _SANDBOX_START_MAX_RETRIES,
                    delay,
                    type(exc).__name__,
                    exc,
                )
                await asyncio.sleep(delay)

        self._step_count = 0
        self._conversation = []

        # Capture initial messages (system + user prompt) for trajectory logging.
        if ts.observation is not None:
            for msg in ts.observation.messages:
                self._conversation.append({"role": msg["role"], "content": _get_text_content(msg)})

        model_input = self._ts_to_model_input(ts)
        if not self._fits_context_window(model_input.length):
            raise ValueError(
                f"Initial prompt too long for context window: "
                f"{model_input.length} prompt + {self._max_tokens} reserved "
                f"> {self._max_trajectory_tokens}"
            )
        _LOGGER.info(
            "ENV READY | task=%s | prompt_tokens=%d/%d",
            self._task_name,
            model_input.length,
            self._max_trajectory_tokens,
        )
        return model_input, self.stop_condition

    async def step(self, action: list[int]) -> Any:
        tinker = importlib.import_module("tinker")
        step_result_cls = importlib.import_module("tinker_cookbook.rl.types").StepResult
        self._step_count += 1

        # Decode assistant message using renderer.
        message, parse_success = self._renderer.parse_response(action)
        assistant_text = _get_text_content(message)

        # Track assistant turn for trajectory.
        self._conversation.append({"role": "assistant", "content": assistant_text})

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
        too_long = 0.0
        meta: dict[str, Any] = {}

        if episode_done:
            # Read meta-results from the underlying environment (if available).
            if hasattr(self._env, "get_meta_results"):
                meta = self._env.get_meta_results()
            if meta:
                _LOGGER.info(
                    "ENV DONE  | task=%s | step=%d | reason=task_complete | reward=%.3f | "
                    "patch=%s | valid=%s | frontier_solved=%s | reason=%s",
                    self._task_name,
                    self._step_count,
                    reward,
                    meta.get("produces_patch"),
                    meta.get("bug_is_valid"),
                    meta.get("strong_resolved"),
                    meta.get("failure_reason") or "ok",
                )
            else:
                _LOGGER.info(
                    "ENV DONE  | task=%s | step=%d | reason=task_complete | reward=%.3f",
                    self._task_name,
                    self._step_count,
                    reward,
                )
            await self._persist_trial_artifacts()
            await self._env.__aexit__(None, None, None)
            self._closed = True
            next_observation = tinker.ModelInput.empty()
        else:
            next_observation = self._ts_to_model_input(ts)
            if not self._fits_context_window(next_observation.length):
                # Context exceeds budget — terminate episode with reward=0.
                # This bounds rollout length and prevents infinite loops when
                # the agent never finishes within the context window.
                _LOGGER.info(
                    "ENV DONE  | task=%s | step=%d | reason=too_long | context=%d/%d "
                    "(prompt=%d + reserved_gen=%d > max=%d)",
                    self._task_name,
                    self._step_count,
                    next_observation.length,
                    self._max_trajectory_tokens,
                    next_observation.length,
                    self._max_tokens,
                    self._max_trajectory_tokens,
                )
                too_long = 1.0
                episode_done = True
                reward = 0.0
                next_observation = tinker.ModelInput.empty()
                await self._persist_trial_artifacts()
                await self._env.__aexit__(None, None, None)
                self._closed = True
            else:
                # Track new observation messages (tool results, etc.) for trajectory.
                # NOTE: Assumes ts.observation.messages is append-only (new messages
                # are appended after the existing ones).  This holds because
                # CodeEnvironment's QueueMediatedLLMClient accumulates messages.
                if ts.observation is not None:
                    existing_count = len(self._conversation)
                    all_msgs = ts.observation.messages
                    new_msgs = all_msgs[existing_count:] if len(all_msgs) > existing_count else []
                    for msg in new_msgs:
                        self._conversation.append({"role": msg["role"], "content": _get_text_content(msg)})
                _LOGGER.debug(
                    "ENV STEP  | task=%s | step=%d | tokens=%d/%d",
                    self._task_name,
                    self._step_count,
                    next_observation.length,
                    self._max_trajectory_tokens,
                )

        metrics: dict[str, float] = {
            "parse_success": float(bool(parse_success)),
            "reward": reward,
            "too_long": too_long,
        }
        if meta:
            metrics["bug_valid"] = float(meta.get("bug_is_valid", False))
            metrics["frontier_solved"] = float(meta.get("strong_resolved", False))
            metrics["produces_patch"] = float(meta.get("produces_patch", False))
            metrics["no_test_mods"] = float(meta.get("no_test_modifications", False))
            metrics["inner_task_created"] = float(meta.get("inner_task_created", False))

        return step_result_cls(
            reward=reward,
            episode_done=episode_done,
            next_observation=next_observation,
            next_stop_condition=self.stop_condition,
            metrics=metrics,
        )

    async def _persist_trial_artifacts(self) -> None:
        """Download sandbox artifacts and save teacher trajectory to disk."""
        if self._trial_log_path is None:
            return
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        trial_dir = self._trial_log_path / f"{self._task_name}_{ts}_{uuid.uuid4().hex[:8]}"
        trial_dir.mkdir(parents=True, exist_ok=True)

        # Download sandbox artifacts (inner_task, reward.txt, meta_results.json, etc.)
        if hasattr(self._env, "download_trial_artifacts"):
            await self._env.download_trial_artifacts(trial_dir)

        # Save teacher trajectory as markdown.
        trajectory_md = self._format_trajectory_markdown()
        if trajectory_md:
            (trial_dir / "trajectory.md").write_text(trajectory_md)

        _LOGGER.info("Persisted trial artifacts to %s", trial_dir)

    def _format_trajectory_markdown(self) -> str:
        """Format the captured conversation as a readable markdown document."""
        if not self._conversation:
            return ""
        lines = [f"# Teacher Trajectory: {self._task_name}", f"Steps: {self._step_count}", ""]
        for msg in self._conversation:
            role = msg["role"].upper()
            lines.append(f"## {role}")
            lines.append("")
            lines.append(msg["content"])
            lines.append("")
            lines.append("---")
            lines.append("")
        return "\n".join(lines)

    async def close(self) -> None:
        """Close the underlying ARES environment (idempotent)."""
        if self._closed:
            return
        self._closed = True
        _LOGGER.debug("ENV CLOSE | task=%s | step=%d", self._task_name, self._step_count)
        with contextlib.suppress(Exception):
            await self._persist_trial_artifacts()
        with contextlib.suppress(Exception):
            await self._env.__aexit__(None, None, None)


class AresEnvGroupBuilder:
    """Build a group of AresCodeTinkerEnv instances for a single task.

    Supports two modes:
    - **Preset mode**: Creates ``group_size`` environments via ``ares.make(preset:idx)``.
    - **Task mode**: Creates ``group_size`` environments from an injected ``harbor.Task`` object,
      wrapping it in a ``CodeEnvironment`` directly. Used by demiurge-swe to inject meta-tasks.

    Collects trajectories and lets the RL algorithm center rewards within the group.
    """

    def __init__(
        self,
        *,
        preset_name: str | None = None,
        task_idx: int = 0,
        task: Any | None = None,
        group_size: int,
        renderer: tinker_env.RendererProtocol,
        container_factory: containers.ContainerFactory,
        max_trajectory_tokens: int = 32 * 1024,
        max_tokens: int = 4096,
        snapshot_template_name: str | None = None,
        code_agent_factory: Any | None = None,
        sandbox_env: dict[str, str] | None = None,
        trial_log_path: pathlib.Path | None = None,
    ):
        if task is None and preset_name is None:
            raise ValueError("Either preset_name or task must be provided")

        self._preset_name = preset_name
        self._task_idx = int(task_idx)
        self._task = task
        self._group_size = int(group_size)
        if self._group_size <= 0:
            raise ValueError("group_size must be positive")

        self._renderer = renderer
        self._container_factory = container_factory
        self._max_trajectory_tokens = int(max_trajectory_tokens)
        self._max_tokens = int(max_tokens)
        self._snapshot_template_name = snapshot_template_name
        self._code_agent_factory = code_agent_factory
        self._sandbox_env = sandbox_env
        self._trial_log_path = trial_log_path

    async def make_envs(self) -> Sequence[AresCodeTinkerEnv]:
        importlib.import_module("tinker")
        importlib.import_module("tinker_cookbook")

        from ares.environments import code_env

        if self._task is not None:
            # Task mode: wrap injected Harbor Task in CodeEnvironment directly.
            task_name = getattr(self._task, "name", "injected-task")
            envs: list[AresCodeTinkerEnv] = []
            code_agent_kwargs: dict[str, Any] = {}
            if self._code_agent_factory is not None:
                code_agent_kwargs["code_agent_factory"] = self._code_agent_factory
            if self._sandbox_env is not None:
                code_agent_kwargs["env"] = self._sandbox_env
            for _ in range(self._group_size):
                env = code_env.CodeEnvironment(
                    tasks=[self._task],
                    container_factory=self._container_factory,
                    snapshot_template_name=self._snapshot_template_name,
                    **code_agent_kwargs,
                )
                envs.append(
                    AresCodeTinkerEnv(
                        env=env,
                        renderer=self._renderer,
                        max_trajectory_tokens=self._max_trajectory_tokens,
                        max_tokens=self._max_tokens,
                        task_name=task_name,
                        trial_log_path=self._trial_log_path,
                    )
                )
            return envs

        # Preset mode: use ares.make() registry.
        task_name = f"{self._preset_name}:{self._task_idx}"
        envs = []
        for _ in range(self._group_size):
            env = ares.make(
                task_name,
                container_factory=self._container_factory,
                snapshot_template_name=self._snapshot_template_name,
            )
            envs.append(
                AresCodeTinkerEnv(
                    env=env,
                    renderer=self._renderer,
                    max_trajectory_tokens=self._max_trajectory_tokens,
                    max_tokens=self._max_tokens,
                    task_name=task_name,
                    trial_log_path=self._trial_log_path,
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
        if self._task is not None:
            task_name = getattr(self._task, "name", "injected-task")
            return ["ares-code-agent", task_name]
        return ["ares-code-agent", f"{self._preset_name}:{self._task_idx}"]


class AresRLDatasetBuilder:
    """tinker-cookbook-compatible RLDatasetBuilder for ARES CodeEnvironment tasks.

    Supports two modes:
    - **Preset mode**: Uses ``preset_name`` to load tasks from the ARES registry.
    - **Task mode**: Uses ``tasks`` (list of ``harbor.Task`` objects) injected directly
      by the caller (e.g., demiurge-swe P1 pipeline).

    Creates a ``TerminalRLDataset`` (reused from the terminal harness — it's generic)
    with task objects and ARES group builder thunks.
    """

    def __init__(
        self,
        *,
        preset_name: str | None = None,
        tasks: list[Any] | None = None,
        num_tasks: int | None = None,
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
        code_agent_factory: Any | None = None,
        sandbox_env: dict[str, str] | None = None,
        trial_log_path: pathlib.Path | None = None,
    ):
        if preset_name is None and tasks is None:
            raise ValueError("Either preset_name or tasks must be provided")

        self._preset_name = preset_name
        self._injected_tasks = tasks
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
        self._code_agent_factory = code_agent_factory
        self._sandbox_env = sandbox_env
        self._trial_log_path = trial_log_path

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

        # Capture closure variables for the thunk.
        group_size = self._group_size
        container_factory = self._container_factory
        max_trajectory_tokens = self._max_trajectory_tokens
        max_tokens = self._max_tokens
        snapshot_template_name = self._snapshot_template_name
        code_agent_factory = self._code_agent_factory
        sandbox_env = self._sandbox_env
        trial_log_path = self._trial_log_path

        if self._injected_tasks is not None:
            # Task mode: use injected Harbor Task objects directly.
            task_list: list[Any] = self._injected_tasks

            _LOGGER.info(
                "AresRLDatasetBuilder: injected tasks=%d, group_size=%d",
                len(task_list),
                self._group_size,
            )

            def task_thunk(task: Any) -> Any:
                return AresEnvGroupBuilder(
                    task=task,
                    group_size=group_size,
                    renderer=renderer,
                    container_factory=container_factory,
                    max_trajectory_tokens=max_trajectory_tokens,
                    max_tokens=max_tokens,
                    snapshot_template_name=snapshot_template_name,
                    code_agent_factory=code_agent_factory,
                    sandbox_env=sandbox_env,
                    trial_log_path=trial_log_path,
                )

            return (
                dataset.TerminalRLDataset(
                    tasks=task_list,
                    groups_per_batch=self._groups_per_batch,
                    num_batches=self._num_batches,
                    group_builder_thunk=task_thunk,
                    builder_buffer=self._builder_buffer,
                ),
                None,
            )

        # Preset mode: load tasks from ARES registry.
        assert self._preset_name is not None
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
        idx_tasks: list[int] = list(range(total_tasks))
        preset_name = self._preset_name

        def idx_thunk(task_idx: Any) -> Any:
            return AresEnvGroupBuilder(
                preset_name=preset_name,
                task_idx=int(task_idx),
                group_size=group_size,
                renderer=renderer,
                container_factory=container_factory,
                max_trajectory_tokens=max_trajectory_tokens,
                max_tokens=max_tokens,
                snapshot_template_name=snapshot_template_name,
                trial_log_path=trial_log_path,
            )

        return (
            dataset.TerminalRLDataset(
                tasks=idx_tasks,  # type: ignore[arg-type]
                groups_per_batch=self._groups_per_batch,
                num_batches=self._num_batches,
                group_builder_thunk=idx_thunk,  # type: ignore[arg-type]
                builder_buffer=self._builder_buffer,
            ),
            None,
        )
