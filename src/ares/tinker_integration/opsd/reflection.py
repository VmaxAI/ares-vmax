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

import tinker

from ares.tinker_integration.opsd import config as opsd_config_mod
from ares.tinker_integration.opsd import evaluation as eval_mod
from ares.tinker_integration.opsd import trajectory_logging as traj_log

_LOGGER = logging.getLogger(__name__)

_REFLECTION_SYSTEM_MESSAGE = """\
You are an expert code reviewer writing a structured analysis document. \
You do NOT write code, bash commands, or interact with any environment. \
You do NOT role-play as an agent or continue any interaction transcript. \
Your ONLY job is to produce a thorough written analysis of failed attempts."""

_REFLECTION_PROMPT_TEMPLATE = """\
Below is a software engineering task followed by interaction traces from \
multiple failed attempts at solving it. All attempts share the same task \
description (shown once). The traces show only each attempt's unique \
interaction: the model's responses and the environment's outputs.

Write a structured analysis document that will be injected as privileged \
information into a future attempt. Be maximally thorough.

IMPORTANT: You are a REVIEWER, not the agent. Do NOT generate bash commands, \
code blocks, or THOUGHT sections. Write ONLY the analysis sections below.

## Task Description
{task_instruction}

## Failed Attempt Traces
{trace_sections}

---

Produce your analysis covering EVERY section below. Do NOT abbreviate — \
length and specificity are critical.

### 1. Problem Distillation
Restate the core problem in your own words. What exactly needs to change? \
What is the expected behavior vs. the current behavior? What edge cases or \
subtleties make this problem non-trivial?

### 2. Codebase Context
Which repository is this? What framework/library version? What are the \
relevant source files, classes, methods, and line numbers? Describe the \
architecture around the code that needs to change — how do the relevant \
components interact?

### 3. Per-Attempt Failure Analysis
For EACH failed attempt above, explain in detail:
- What approach did the model take?
- What specific commands were run and what was the output?
- What exact error messages, tracebacks, or assertion failures occurred?
- WHY did this approach fail? (wrong file, wrong logic, missed edge case, \
misunderstood the test, etc.)
- What partial progress was made that could be built upon?

### 4. Error Patterns and Root Causes
Synthesize across all attempts:
- What common mistakes keep recurring?
- What is the fundamental root cause of failure?
- Are there red herrings or misleading paths in the codebase?

### 5. Critical Anti-Patterns (What NOT To Do)
List specific approaches, files, or strategies that were tried and failed. \
Be explicit about why each should be avoided.

### 6. Recommended Solution
Provide a concrete, step-by-step implementation plan:
- Which file(s) to modify and in what order
- What exact changes to make (describe the code changes precisely)
- How to handle edge cases identified above
- How to verify the fix works (which test commands to run)

### 7. Test Suite Analysis
- What tests are being run and what do they assert?
- Are there subtleties in the test expectations (e.g., exact string \
matching, specific exception types, ordering requirements)?
- What would a passing test output look like?

Be exhaustive. Write as much as you can. Every specific detail you include \
(file paths, function names, error messages, line numbers) directly \
increases the probability of success on the next attempt."""

_TRACE_SECTION_TEMPLATE = """\
### Failed Attempt {attempt_num}
{condensed_trace}"""


def _get_action_tokens(action: Any) -> list[int]:
    """Extract token list from an action object (TokensWithLogprobs)."""
    tokens = getattr(action, "tokens", None)
    if tokens:
        return list(tokens)
    if hasattr(action, "to_ints"):
        return list(action.to_ints())
    return []


def _get_obs_tokens(obs: Any) -> list[int]:
    """Extract token list from an observation object (ModelInput)."""
    if obs is not None and hasattr(obs, "to_ints"):
        return list(obs.to_ints())
    return []


def _extract_condensed_trace(
    trajectory: Any,
    tokenizer: Any,
    max_tokens: int,
) -> str:
    """Extract a condensed, deduplicated trace from a trajectory.

    Instead of dumping raw observations (which contain the full accumulated
    context including the system prompt), this extracts only the *delta* at
    each step: the model's response and the new environment output.  The
    system prompt / task description is shown once at the top of the
    reflection prompt, so it is NOT repeated inside each trace.
    """
    transitions = getattr(trajectory, "transitions", [])
    if not transitions:
        try:
            all_tokens = trajectory.model_input.to_ints()
            text = tokenizer.decode(all_tokens[-max_tokens:])
            return text[: max_tokens * 4]
        except Exception:
            return "(unable to extract trace)"

    # Show the last N steps of the interaction.
    window_size = 6
    start_idx = max(0, len(transitions) - window_size)
    window = transitions[start_idx:]

    # We need the transition just before the window to compute the first
    # delta (new env output = obs[i] minus prev_obs + prev_action).
    prev_transition = transitions[start_idx - 1] if start_idx > 0 else None

    parts: list[str] = []
    per_step_tokens = max(max_tokens // max(len(window), 1), 256)

    for step_idx, t in enumerate(window):
        # --- Model response (action tokens — always unique per step) ---
        action_text = ""
        action = getattr(t, "ac", None)
        if action is not None:
            try:
                ac_tokens = _get_action_tokens(action)
                if ac_tokens:
                    action_text = tokenizer.decode(ac_tokens[-per_step_tokens:])
            except Exception:
                pass

        # --- Environment output delta ---
        # For the first transition of the entire trajectory there is no
        # previous step — the observation is just the task prompt (already
        # shown at the top), so we skip the env output.
        env_output_text = ""
        prev_t = prev_transition if step_idx == 0 else window[step_idx - 1]

        if prev_t is not None:
            obs = getattr(t, "ob", None)
            prev_obs = getattr(prev_t, "ob", None)
            prev_ac = getattr(prev_t, "ac", None)
            try:
                obs_tokens = _get_obs_tokens(obs)
                prev_obs_tokens = _get_obs_tokens(prev_obs)
                prev_ac_tokens = _get_action_tokens(prev_ac) if prev_ac is not None else []
                delta_start = len(prev_obs_tokens) + len(prev_ac_tokens)
                if delta_start < len(obs_tokens):
                    delta_tokens = obs_tokens[delta_start : delta_start + per_step_tokens]
                    if delta_tokens:
                        env_output_text = tokenizer.decode(delta_tokens)
            except Exception:
                pass

        # --- Assemble step ---
        step_parts: list[str] = []
        if env_output_text:
            step_parts.append(f"**Environment Output:**\n{env_output_text}")
        if action_text:
            step_parts.append(f"**Model Response:**\n{action_text}")

        if step_parts:
            parts.append(f"#### Step {step_idx + 1}\n" + "\n\n".join(step_parts))

    result = "\n\n".join(parts)

    # Final length guard (rough 4:1 char-to-token ratio).
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
    first_obs = getattr(transitions[0], "ob", None)
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
    *,
    log_path: str | None = None,
    cycle: int = 0,
) -> tuple[str, str | None]:
    """Generate reflection for a single task. Returns (task_name, reflection_text)."""
    prompt = build_reflection_prompt(task_result, tokenizer, config)

    # Save reflection input (ATIF).
    if log_path:
        traj_log.save_reflection_input(
            log_path, cycle, task_result.task_name, prompt, system_message=_REFLECTION_SYSTEM_MESSAGE
        )

    messages = [
        {"role": "system", "content": _REFLECTION_SYSTEM_MESSAGE},
        {"role": "user", "content": prompt},
    ]
    model_input = renderer.build_generation_prompt(messages)

    try:
        sampling_params = tinker.SamplingParams(
            max_tokens=config.max_reflection_tokens,
            temperature=0.7,
        )
        result = await sampling_client.sample_async(
            model_input,
            num_samples=1,
            sampling_params=sampling_params,
        )

        # Decode the generated tokens from the first (only) sequence.
        response_tokens = result.sequences[0].tokens if result.sequences else []
        reflection_text = tokenizer.decode(response_tokens) if response_tokens else str(result)

        prompt_tok_count = model_input.length
        reflection_tok_count = len(response_tokens) if response_tokens else 0

        _LOGGER.info(
            "REFLECTION | task=%s | prompt_tokens=%d | reflection_tokens=%d",
            task_result.task_name,
            prompt_tok_count,
            reflection_tok_count,
        )

        # Save reflection output (ATIF).
        if log_path:
            traj_log.save_reflection_output(
                log_path,
                cycle,
                task_result.task_name,
                reflection_text,
                prompt_tokens=prompt_tok_count,
                reflection_tokens=reflection_tok_count,
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
    *,
    log_path: str | None = None,
    cycle: int = 0,
) -> dict[str, str]:
    """Generate self-reflections for all hard tasks concurrently.

    Args:
        sampling_client: Tinker sampling client for LLM generation.
        hard_tasks: Tasks where student failed all rollouts.
        config: OPSD configuration.
        renderer: Tinker renderer for building prompts.
        tokenizer: Tokenizer for decoding trajectories and tokens.
        log_path: If set, save ATIF trajectory files under this directory.
        cycle: OPSD cycle number (for ATIF filenames).

    Returns:
        Mapping of task_name -> reflection_text for tasks with successful generation.
    """
    _LOGGER.info("=== REFLECTION PHASE | generating reflections for %d hard tasks ===", len(hard_tasks))

    coros = [
        _generate_single_reflection(
            sampling_client,
            task_result,
            tokenizer,
            renderer,
            config,
            log_path=log_path,
            cycle=cycle,
        )
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
