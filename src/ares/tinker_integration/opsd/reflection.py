"""Self-reflection phase for OPSD.

For each hard task, extracts condensed interaction traces from failed rollouts
and generates compact self-reflections (hints, root-cause analysis) using the
same model via a pure LLM call (no sandboxes needed).

The generated reflections serve as the "privileged information" that transforms
the student into a teacher when injected into the context.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from ares.tinker_integration.opsd import config as opsd_config_mod
from ares.tinker_integration.opsd import evaluation as eval_mod

_LOGGER = logging.getLogger(__name__)

_REFLECTION_PROMPT_TEMPLATE = """\
You are analyzing failed attempts to solve a software engineering task. Your \
analysis will be used as privileged information to guide a future attempt, so \
be thorough and specific — include exact file paths, function names, error \
messages, and concrete steps.

## Task
{task_instruction}

{trace_sections}

Provide a detailed analysis covering:
1. **Error patterns**: Exact error messages, tracebacks, and assertion \
failures observed across attempts.
2. **Root cause analysis**: Why each attempt failed — wrong file edited, \
incorrect fix logic, missing edge case, wrong test interpretation, etc.
3. **Key files and locations**: Specific files, classes, functions, and \
line numbers that are relevant to the fix.
4. **What NOT to do**: Anti-patterns observed in the failed attempts that \
should be avoided.
5. **Recommended approach**: A concrete step-by-step plan to solve this \
task, including which files to modify, what changes to make, and how to \
verify the fix.
6. **Test expectations**: What the test suite expects and any subtleties \
in the test assertions.

Be thorough. Detail is more important than brevity."""

_TRACE_SECTION_TEMPLATE = """\
## Failed Attempt {attempt_num}
{condensed_trace}"""


def _extract_condensed_trace(
    trajectory: Any,
    tokenizer: Any,
    max_tokens: int,
) -> str:
    """Extract a condensed trace from a trajectory.

    Focuses on the last 2-3 turns (commands + outputs/tracebacks) to keep
    within context budget. Falls back to raw token decoding if message
    extraction isn't available.
    """
    # Try extracting text from trajectory transitions.
    transitions = getattr(trajectory, "transitions", [])
    if not transitions:
        # Fallback: decode the full trajectory token sequence.
        try:
            all_tokens = trajectory.model_input.to_ints()
            text = tokenizer.decode(all_tokens[-max_tokens:])
            return text[: max_tokens * 4]  # rough char estimate
        except Exception:
            return "(unable to extract trace)"

    # Take the last several transitions for richer context.  We include more
    # turns than strictly necessary because error patterns often span multiple
    # steps (e.g. the model edits a file, runs tests, sees an error, tries a
    # different fix, sees a different error).
    last_transitions = transitions[-6:]
    parts: list[str] = []

    # Budget per transition: split max_tokens across all transitions.
    per_transition_tokens = max(max_tokens // max(len(last_transitions), 1), 512)

    for t in last_transitions:
        # Each transition has observation (model_input) and action (token list).
        obs_text = ""
        action_text = ""

        # Try to decode observation.
        obs = getattr(t, "observation", None)
        if obs is not None:
            try:
                obs_tokens = obs.to_ints() if hasattr(obs, "to_ints") else []
                if obs_tokens:
                    obs_text = tokenizer.decode(obs_tokens[-per_transition_tokens:])
            except Exception:
                pass

        # Try to decode action.
        action = getattr(t, "action", None)
        if action is not None:
            try:
                if isinstance(action, list):
                    action_text = tokenizer.decode(action)
                elif hasattr(action, "to_ints"):
                    action_text = tokenizer.decode(action.to_ints())
            except Exception:
                pass

        if obs_text:
            parts.append(f"[Observation]\n{obs_text}")
        if action_text:
            parts.append(f"[Action]\n{action_text}")

    result = "\n\n".join(parts)

    # Truncate to max_tokens worth of characters (rough 4:1 char:token ratio).
    max_chars = max_tokens * 4
    if len(result) > max_chars:
        result = result[-max_chars:]
    return result


def _extract_task_instruction(task_result: eval_mod.TaskEvalResult, tokenizer: Any) -> str:
    """Extract the task instruction from the first trajectory's initial observation."""
    if not task_result.trajectories:
        return f"Task: {task_result.task_name}"

    first_traj = task_result.trajectories[0]
    transitions = getattr(first_traj, "transitions", [])
    if not transitions:
        return f"Task: {task_result.task_name}"

    # The first observation typically contains the task instruction.
    first_obs = getattr(transitions[0], "observation", None)
    if first_obs is None:
        return f"Task: {task_result.task_name}"

    try:
        obs_tokens = first_obs.to_ints() if hasattr(first_obs, "to_ints") else []
        if obs_tokens:
            # Take a generous portion as the task instruction — the full problem
            # statement is critical for high-quality reflection.
            text = tokenizer.decode(obs_tokens[:4096])
            if len(text) > 8000:
                text = text[:8000] + "..."
            return text
    except Exception:
        pass

    return f"Task: {task_result.task_name}"


def build_reflection_prompt(
    task_result: eval_mod.TaskEvalResult,
    tokenizer: Any,
    config: opsd_config_mod.OPSDConfig,
) -> str:
    """Build the reflection prompt for a single hard task.

    Selects the longest failed trajectories (for richer info), extracts
    condensed traces, and assembles the reflection prompt.
    """
    task_instruction = _extract_task_instruction(task_result, tokenizer)

    # Pick the longest trajectories for richer information.
    trajectories = list(task_result.trajectories)
    trajectories.sort(key=lambda t: len(getattr(t, "transitions", [])), reverse=True)
    selected = trajectories[: config.num_traces_for_reflection]

    trace_sections: list[str] = []
    for i, traj in enumerate(selected, 1):
        condensed = _extract_condensed_trace(traj, tokenizer, config.max_condensed_trace_tokens)
        section = _TRACE_SECTION_TEMPLATE.format(attempt_num=i, condensed_trace=condensed)
        trace_sections.append(section)

    return _REFLECTION_PROMPT_TEMPLATE.format(
        task_instruction=task_instruction,
        trace_sections="\n\n".join(trace_sections),
    )


async def _generate_single_reflection(
    sampling_client: Any,
    task_result: eval_mod.TaskEvalResult,
    tokenizer: Any,
    renderer: Any,
    config: opsd_config_mod.OPSDConfig,
) -> tuple[str, str | None]:
    """Generate reflection for a single task. Returns (task_name, reflection_text)."""
    prompt = build_reflection_prompt(task_result, tokenizer, config)

    messages = [{"role": "user", "content": prompt}]
    model_input = renderer.build_generation_prompt(messages)

    try:
        result = await sampling_client.sample_async(
            model_input,
            max_tokens=config.max_reflection_tokens,
            temperature=0.7,
        )

        # Decode the generated tokens.
        response_tokens = result.tokens if hasattr(result, "tokens") else []
        reflection_text = tokenizer.decode(response_tokens) if response_tokens else str(result)

        _LOGGER.info(
            "REFLECTION | task=%s | prompt_tokens=%d | reflection_tokens=%d",
            task_result.task_name,
            model_input.length,
            len(response_tokens) if response_tokens else 0,
        )
        return task_result.task_name, reflection_text

    except Exception as exc:
        _LOGGER.warning(
            "REFLECTION | task=%s | generation failed: %s: %s",
            task_result.task_name,
            type(exc).__name__,
            exc,
        )
        return task_result.task_name, None


async def generate_reflections(
    sampling_client: Any,
    hard_tasks: list[eval_mod.TaskEvalResult],
    config: opsd_config_mod.OPSDConfig,
    renderer: Any,
    tokenizer: Any,
) -> dict[str, str]:
    """Generate self-reflections for all hard tasks concurrently.

    Args:
        sampling_client: Tinker sampling client for LLM generation.
        hard_tasks: Tasks where student failed all rollouts.
        config: OPSD configuration.
        renderer: Tinker renderer for building prompts.
        tokenizer: Tokenizer for decoding trajectories and tokens.

    Returns:
        Mapping of task_name -> reflection_text for tasks with successful generation.
    """
    _LOGGER.info("=== REFLECTION PHASE | generating reflections for %d hard tasks ===", len(hard_tasks))

    coros = [
        _generate_single_reflection(sampling_client, task_result, tokenizer, renderer, config)
        for task_result in hard_tasks
    ]
    results = await asyncio.gather(*coros)

    reflections: dict[str, str] = {}
    for task_name, reflection in results:
        if reflection is not None:
            reflections[task_name] = reflection

    _LOGGER.info(
        "=== REFLECTION PHASE DONE | generated=%d | failed=%d ===",
        len(reflections),
        len(hard_tasks) - len(reflections),
    )

    return reflections
