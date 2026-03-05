"""Human-readable trajectory logging for OPSD phases.

Saves reflection and distillation inputs/outputs as readable Markdown files
(ATIF — Agent Trajectory Interchange Format) so you can inspect:

- What condensed traces are fed to the model for self-reflection
- The quality of generated self-reflections
- What privileged context is injected for the teacher/distillation phase
- Whether truncation is working correctly

Files are saved under ``{log_path}/atif/cycle_{N}/``.
"""

from __future__ import annotations

import logging
from pathlib import Path
import re

_LOGGER = logging.getLogger(__name__)


def _sanitize_filename(name: str) -> str:
    """Turn a task name into a safe filename component."""
    return re.sub(r"[^\w\-.]", "_", name)[:120]


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    _LOGGER.debug("ATIF | wrote %s (%d chars)", path, len(content))


def save_reflection_input(
    log_path: str,
    cycle: int,
    task_name: str,
    prompt: str,
    *,
    system_message: str = "",
) -> None:
    """Save the full reflection prompt (what the model sees for self-reflection)."""
    safe = _sanitize_filename(task_name)
    path = Path(log_path) / "atif" / f"cycle_{cycle}" / "reflection" / f"{safe}_input.md"
    header = f"# Reflection Input — {task_name}\ncycle: {cycle}\n\n---\n\n"
    parts = [header]
    if system_message:
        parts.append(f"## System Message\n\n{system_message}\n\n---\n\n")
    parts.append(f"## User Prompt\n\n{prompt}")
    _write(path, "".join(parts))


def save_reflection_output(
    log_path: str,
    cycle: int,
    task_name: str,
    reflection_text: str,
    prompt_tokens: int = 0,
    reflection_tokens: int = 0,
) -> None:
    """Save the generated self-reflection."""
    safe = _sanitize_filename(task_name)
    path = Path(log_path) / "atif" / f"cycle_{cycle}" / "reflection" / f"{safe}_output.md"
    header = (
        f"# Reflection Output — {task_name}\n"
        f"cycle: {cycle}  |  prompt_tokens: {prompt_tokens}  |  reflection_tokens: {reflection_tokens}\n\n"
        f"---\n\n"
    )
    _write(path, header + reflection_text)


def save_distillation_context(
    log_path: str,
    cycle: int,
    task_name: str,
    privileged_context: str,
    reflection_text: str,
) -> None:
    """Save the privileged context injected during distillation."""
    safe = _sanitize_filename(task_name)
    path = Path(log_path) / "atif" / f"cycle_{cycle}" / "distillation" / f"{safe}_privileged_context.md"
    content = (
        f"# Distillation Privileged Context — {task_name}\n"
        f"cycle: {cycle}\n\n"
        f"---\n\n"
        f"## Rendered Privileged Context (what the teacher model sees prepended)\n\n"
        f"{privileged_context}\n\n"
        f"---\n\n"
        f"## Raw Reflection Used\n\n"
        f"{reflection_text}\n"
    )
    _write(path, content)


def save_cycle_summary(
    log_path: str,
    cycle: int,
    *,
    hard_task_names: list[str],
    reflection_tasks_generated: list[str],
    reflection_tasks_cached: list[str],
    teacher_solved: list[str],
    distillable_tasks: list[str],
) -> None:
    """Save a per-cycle summary of what happened in each OPSD phase."""
    path = Path(log_path) / "atif" / f"cycle_{cycle}" / "summary.md"
    lines = [
        f"# OPSD Cycle {cycle} Summary\n",
        f"## Hard Tasks ({len(hard_task_names)})\n",
        *[f"- {n}\n" for n in hard_task_names],
        f"\n## Reflections Generated ({len(reflection_tasks_generated)})\n",
        *[f"- {n}\n" for n in reflection_tasks_generated],
        f"\n## Reflections Cached ({len(reflection_tasks_cached)})\n",
        *[f"- {n}\n" for n in reflection_tasks_cached],
        f"\n## Teacher Solved ({len(teacher_solved)})\n",
        *[f"- {n}\n" for n in teacher_solved],
        f"\n## Distillable Tasks ({len(distillable_tasks)})\n",
        *[f"- {n}\n" for n in distillable_tasks],
    ]
    _write(path, "".join(lines))
