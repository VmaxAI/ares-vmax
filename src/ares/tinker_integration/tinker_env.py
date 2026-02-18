"""Tinker Env adapter wrapping AsyncTerminalGymEnv for RL training.

Ported from WORKING_TINKER/terminal_rl/wrapped_env.py (lines 1-273). Provides the
``HarborTerminalTinkerEnv`` class that converts between Tinker's token-level RL
interface and the terminal's keystroke-level interface via JSON command parsing.
"""

from __future__ import annotations

import contextlib
from dataclasses import dataclass
import importlib
import json
from typing import Any, Protocol, TypedDict

from ares.tinker_integration import terminal_env

_CONTEXT_LEN_BUFFER = 10


def _middle_truncate(model_input: Any, max_context_len: int) -> Any:
    """Truncate model input from the middle when exceeding max context length.

    Preserves both the beginning (task context / instruction) and end (recent
    terminal state and actions) of the conversation while removing stale middle
    history.  This is the same strategy used by the code-agent harness.
    """
    tinker = importlib.import_module("tinker")

    num_tokens_to_truncate = model_input.length - max_context_len + _CONTEXT_LEN_BUFFER
    if num_tokens_to_truncate <= 0:
        return model_input

    center_idx = model_input.length // 2
    truncate_start_idx = center_idx - num_tokens_to_truncate // 2
    truncate_end_idx = center_idx + num_tokens_to_truncate // 2

    curr_ints = model_input.to_ints()
    new_ints = curr_ints[:truncate_start_idx] + curr_ints[truncate_end_idx:]
    return tinker.ModelInput.from_ints(new_ints)


class TerminalActionParseError(Exception):
    pass


type StopCondition = list[str] | list[int]


class RendererProtocol(Protocol):
    def get_stop_sequences(self) -> StopCondition: ...

    def build_generation_prompt(self, messages: list[dict[str, Any]]) -> Any: ...

    def parse_response(self, tokens: list[int]) -> tuple[dict[str, Any], bool]: ...


class TerminusCommand(TypedDict, total=False):
    keystrokes: str
    duration: float


@dataclass(frozen=True)
class ParsedTerminalPolicyOutput:
    commands: list[TerminusCommand]
    task_complete: bool
    raw: dict[str, Any]


def _strip_code_fences(text: str) -> str:
    s = text.strip()
    if s.startswith("```"):
        lines = s.splitlines()
        if len(lines) >= 2:
            # Drop first line ``` or ```json
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _get_text_content(message: dict[str, Any]) -> str:
    """Extract text content from a renderer Message, stripping thinking parts."""
    content = message["content"]
    if isinstance(content, str):
        return content
    return "".join(p["text"] for p in content if p["type"] == "text")  # type: ignore[index]


def parse_terminus_json_plain(content: str) -> ParsedTerminalPolicyOutput:
    """Parse the JSON specified by terminus-json-plain.txt.

    Expected keys: analysis, plan, commands[{keystrokes, duration}], optional task_complete.
    """

    cleaned = _strip_code_fences(content)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise TerminalActionParseError(f"Invalid JSON output: {e}") from e

    if not isinstance(data, dict):
        raise TerminalActionParseError("Top-level output must be a JSON object")

    commands = data.get("commands", [])
    if not isinstance(commands, list):
        raise TerminalActionParseError("'commands' must be a list")

    # Normalize to the expected schema to keep pyright happy.
    commands_typed: list[TerminusCommand] = []
    for cmd in commands:
        if not isinstance(cmd, dict):
            continue

        typed: TerminusCommand = {}
        keystrokes = cmd.get("keystrokes")
        if isinstance(keystrokes, str):
            typed["keystrokes"] = keystrokes

        duration = cmd.get("duration")
        if isinstance(duration, (int, float)):
            typed["duration"] = float(duration)
        elif isinstance(duration, str):
            with contextlib.suppress(Exception):
                typed["duration"] = float(duration)

        commands_typed.append(typed)

    task_complete = bool(data.get("task_complete", False))

    return ParsedTerminalPolicyOutput(commands=commands_typed, task_complete=task_complete, raw=data)


class HarborTerminalTinkerEnv:
    """Adapt Harbor's AsyncTerminalGymEnv to tinker-cookbook's RL Env interface.

    High-level idea:
    - Observation: a tinker.ModelInput created by a renderer over a message history.
    - Action: model completion tokens (list[int]) that decode into assistant text.
    - The assistant text must be JSON (terminus-json-plain format) describing terminal keystrokes.
    - We execute those keystrokes in the sandbox via AsyncTerminalGymEnv.
    """

    def __init__(
        self,
        *,
        gym_env: terminal_env.AsyncTerminalGymEnv,
        renderer: RendererProtocol,
        max_trajectory_tokens: int = 32 * 1024,
        reserved_generation_tokens: int = 4096,
    ):
        try:  # pragma: no cover
            importlib.import_module("tinker")
            importlib.import_module("tinker_cookbook")
        except Exception as e:  # pragma: no cover
            raise ImportError("HarborTerminalTinkerEnv requires 'tinker' + 'tinker-cookbook' to be installed.") from e

        self._gym_env = gym_env
        self._renderer = renderer
        # Interpret max_trajectory_tokens as the model context window (in tokens).
        self._max_trajectory_tokens = int(max_trajectory_tokens)
        self._reserved_generation_tokens = max(0, int(reserved_generation_tokens))
        self._past_messages: list[dict[str, Any]] = []

    def _fits_context_window(self, prompt_tokens: int) -> bool:
        """Return True if prompt + reserved generation tokens fits context window."""
        return (int(prompt_tokens) + self._reserved_generation_tokens) <= self._max_trajectory_tokens

    @property
    def stop_condition(self) -> StopCondition:
        return self._renderer.get_stop_sequences()

    async def initial_observation(self) -> tuple[Any, StopCondition]:
        obs, info = await self._gym_env.reset()
        initial_prompt = str(info.get("initial_prompt") or obs)
        self._past_messages = [{"role": "user", "content": initial_prompt}]

        model_input = self._renderer.build_generation_prompt(self._past_messages)
        if not self._fits_context_window(model_input.length):
            raise ValueError(
                "Initial prompt too long for context window: "
                f"{model_input.length} prompt + {self._reserved_generation_tokens} reserved "
                f"> {self._max_trajectory_tokens}"
            )
        return model_input, self.stop_condition

    async def close(self) -> None:
        """Close the underlying gym environment (idempotent)."""
        await self._gym_env.close()

    async def step(self, action: list[int]) -> Any:
        tinker = importlib.import_module("tinker")
        step_result_cls = importlib.import_module("tinker_cookbook.rl.types").StepResult

        # 1) Decode assistant message using renderer.
        message, parse_success = self._renderer.parse_response(action)
        self._past_messages.append(message)

        assistant_text = _get_text_content(message)

        # 2) Parse JSON commands.
        try:
            parsed = parse_terminus_json_plain(assistant_text)
            json_ok = 1.0
        except TerminalActionParseError:
            # Treat invalid JSON as a terminal failure.
            # Close the sandbox to avoid resource leaks.
            await self._gym_env.close()
            return step_result_cls(
                reward=0.0,
                episode_done=True,
                next_observation=tinker.ModelInput.empty(),
                next_stop_condition=self.stop_condition,
                metrics={
                    "parse_success": float(bool(parse_success)),
                    "json_ok": 0.0,
                },
            )

        # 3) Execute commands.
        num_commands = 0
        last_terminal_obs = ""
        for cmd in parsed.commands:
            if not isinstance(cmd, dict):
                continue
            keystrokes = cmd.get("keystrokes")
            duration = cmd.get("duration", 1.0)
            if not isinstance(keystrokes, str):
                continue
            try:
                duration_f = float(duration)
            except Exception:
                duration_f = 1.0

            step_res = await self._gym_env.step(
                terminal_env.TerminalAction(
                    keys=keystrokes,
                    block=False,
                    min_timeout_sec=max(0.0, duration_f),
                    max_timeout_sec=180.0,
                    done=False,
                )
            )
            last_terminal_obs = step_res.obs
            num_commands += 1

        # 4) Determine done + reward.
        episode_done = bool(parsed.task_complete)
        reward = 0.0
        if episode_done:
            _, reduced = await self._gym_env.verify()
            reward = float(reduced) if isinstance(reduced, (int, float)) else 0.0

        # 5) Build next observation for continuing episodes.
        context_truncated = 0.0
        if episode_done:
            next_observation = tinker.ModelInput.empty()
            # Close the sandbox to avoid resource leaks.
            await self._gym_env.close()
        else:
            # New user prompt contains fresh terminal state.
            next_prompt = self._gym_env.build_initial_prompt(
                terminal_state=last_terminal_obs or "(no new terminal output)"
            )
            self._past_messages.append({"role": "user", "content": next_prompt})
            next_observation = self._renderer.build_generation_prompt(self._past_messages)
            if not self._fits_context_window(next_observation.length):
                # Context exceeds budget — middle-truncate (drop stale middle
                # history, keep instruction at start + recent state at end)
                # instead of killing the episode.  The agent keeps working.
                max_prompt_tokens = self._max_trajectory_tokens - self._reserved_generation_tokens
                next_observation = _middle_truncate(next_observation, max_prompt_tokens)
                context_truncated = 1.0

        return step_result_cls(
            reward=reward,
            episode_done=episode_done,
            next_observation=next_observation,
            next_stop_condition=self.stop_condition,
            metrics={
                "parse_success": float(bool(parse_success)),
                "json_ok": json_ok,
                "num_commands": num_commands,
                "context_truncated": context_truncated,
            },
        )
