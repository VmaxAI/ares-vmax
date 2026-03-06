"""Privileged environment wrappers for OPSD teacher phase.

These wrappers inject reflection text (privileged information) into the initial
prompt seen by the model, turning the student into a "teacher" — same weights
but richer context.  The core harness code is NOT modified; these are thin
delegation wrappers that only intercept the initial observation.
"""

from __future__ import annotations

from collections.abc import Sequence
import importlib
import logging
from typing import Any

from ares.containers import containers
from ares.tinker_integration import ares_env
from ares.tinker_integration import dataset
from ares.tinker_integration import tinker_env

_LOGGER = logging.getLogger(__name__)

_PRIVILEGED_PREFIX = """\
## Detailed Analysis of Previous Failed Attempts

You have access to privileged information about this task from analyzing \
previous failed attempts. Pay close attention to the specific files, error \
patterns, anti-patterns to avoid, and recommended approach.

{reflection}

## Task (apply the analysis above to solve it correctly)

"""


class PrivilegedTerminalTinkerEnv:
    """Wraps ``HarborTerminalTinkerEnv`` to inject privileged context.

    After ``initial_observation()`` builds the initial prompt, we prepend the
    reflection text to the first user message and re-render the model input.
    All other methods delegate directly.
    """

    def __init__(
        self,
        *,
        inner: tinker_env.HarborTerminalTinkerEnv,
        reflection: str,
    ):
        self._inner = inner
        self._reflection = reflection

    async def initial_observation(self) -> tuple[Any, tinker_env.StopCondition]:
        model_input, stop = await self._inner.initial_observation()

        # Inject privileged context into the first user message.
        if self._inner._past_messages:
            original_content = self._inner._past_messages[0].get("content", "")
            enriched = _PRIVILEGED_PREFIX.format(reflection=self._reflection) + original_content
            self._inner._past_messages[0] = {"role": "user", "content": enriched}

            # Re-render with enriched messages.
            model_input = self._inner._renderer.build_generation_prompt(self._inner._past_messages)

            _LOGGER.info(
                "PRIVILEGED ENV | task=%s | injected %d chars of reflection | new prompt_tokens=%d",
                self._inner._task_name,
                len(self._reflection),
                model_input.length,
            )

        return model_input, stop

    async def step(self, action: list[int]) -> Any:
        return await self._inner.step(action)

    async def close(self) -> None:
        await self._inner.close()

    @property
    def stop_condition(self) -> tinker_env.StopCondition:
        return self._inner.stop_condition


class PrivilegedAresCodeTinkerEnv:
    """Wraps ``AresCodeTinkerEnv`` to inject privileged context.

    Overrides ``_ts_to_model_input`` to prepend reflection text as a prior
    analysis section in the first message.  All other methods delegate.
    """

    def __init__(
        self,
        *,
        inner: ares_env.AresCodeTinkerEnv,
        reflection: str,
    ):
        self._inner = inner
        self._reflection = reflection
        self._initial_done = False

    async def initial_observation(self) -> tuple[Any, tinker_env.StopCondition]:
        model_input, stop = await self._inner.initial_observation()

        # Intercept the initial observation to inject privileged context.
        # We need to re-render with the enriched messages.
        if not self._initial_done:
            self._initial_done = True
            tinker = importlib.import_module("tinker")

            # Get the original TimeStep by accessing the env's last state.
            # Since we just called initial_observation, the env has a valid state.
            # We inject the reflection as a prefix to the rendered prompt.
            prefix_messages = [
                {"role": "user", "content": _PRIVILEGED_PREFIX.format(reflection=self._reflection)},
                {
                    "role": "assistant",
                    "content": "I have carefully reviewed the analysis of previous failed attempts. "
                    "I understand the error patterns, root causes, key files involved, and the "
                    "recommended approach. I will apply these insights to solve the task correctly.",
                },
            ]
            prefix_input = self._inner._renderer.build_generation_prompt(prefix_messages)
            prefix_tokens = prefix_input.to_ints()

            # Prepend prefix tokens to the original model input.
            original_tokens = model_input.to_ints()
            combined = tinker.ModelInput.from_ints(prefix_tokens + original_tokens)

            _LOGGER.info(
                "PRIVILEGED ENV | task=%s | injected %d prefix tokens | total=%d",
                self._inner._task_name,
                len(prefix_tokens),
                combined.length,
            )

            model_input = combined

        return model_input, stop

    async def step(self, action: list[int]) -> Any:
        return await self._inner.step(action)

    async def close(self) -> None:
        await self._inner.close()

    @property
    def stop_condition(self) -> tinker_env.StopCondition:
        return self._inner.stop_condition


class PrivilegedTerminalEnvGroupBuilder:
    """Wraps ``TerminalEnvGroupBuilder`` to inject privileged context into all envs."""

    def __init__(
        self,
        *,
        task: Any,
        group_size: int,
        environment: Any,
        renderer: tinker_env.RendererProtocol,
        reflection: str,
        max_trajectory_tokens: int = 32 * 1024,
        max_tokens: int = 4096,
        gym_env_kwargs: dict[str, Any] | None = None,
    ):
        self._inner = dataset.TerminalEnvGroupBuilder(
            task=task,
            group_size=group_size,
            environment=environment,
            renderer=renderer,
            max_trajectory_tokens=max_trajectory_tokens,
            max_tokens=max_tokens,
            gym_env_kwargs=gym_env_kwargs,
        )
        self._reflection = reflection
        self._task = task

    async def make_envs(self) -> Sequence[PrivilegedTerminalTinkerEnv]:
        inner_envs = await self._inner.make_envs()
        return [PrivilegedTerminalTinkerEnv(inner=env, reflection=self._reflection) for env in inner_envs]

    async def compute_group_rewards(
        self,
        trajectory_group: list[Any],
        env_group: Sequence[Any],
    ) -> list[tuple[float, dict[str, Any]]]:
        return await self._inner.compute_group_rewards(trajectory_group, env_group)

    def logging_tags(self) -> list[str]:
        return self._inner.logging_tags()


class PrivilegedAresEnvGroupBuilder:
    """Wraps ``AresEnvGroupBuilder`` to inject privileged context into all envs."""

    def __init__(
        self,
        *,
        preset_name: str,
        task_idx: int,
        group_size: int,
        renderer: tinker_env.RendererProtocol,
        container_factory: containers.ContainerFactory,
        reflection: str,
        max_trajectory_tokens: int = 32 * 1024,
        max_tokens: int = 4096,
        snapshot_template_name: str | None = None,
    ):
        self._inner = ares_env.AresEnvGroupBuilder(
            preset_name=preset_name,
            task_idx=task_idx,
            group_size=group_size,
            renderer=renderer,
            container_factory=container_factory,
            max_trajectory_tokens=max_trajectory_tokens,
            max_tokens=max_tokens,
            snapshot_template_name=snapshot_template_name,
        )
        self._reflection = reflection

    async def make_envs(self) -> Sequence[PrivilegedAresCodeTinkerEnv]:
        inner_envs = await self._inner.make_envs()
        return [PrivilegedAresCodeTinkerEnv(inner=env, reflection=self._reflection) for env in inner_envs]

    async def compute_group_rewards(
        self,
        trajectory_group: list[Any],
        env_group: Sequence[Any],
    ) -> list[tuple[float, dict[str, Any]]]:
        return await self._inner.compute_group_rewards(trajectory_group, env_group)

    def logging_tags(self) -> list[str]:
        return self._inner.logging_tags()
