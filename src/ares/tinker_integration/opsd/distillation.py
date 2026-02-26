"""On-policy distillation phase for OPSD.

The student generates on-policy trajectories.  The teacher (same model weights +
privileged context) provides dense per-token supervision via reverse KL.  Teacher
logprobs are computed by prepending privileged context tokens to the student's
full sequence and calling ``compute_logprobs_async``.

Adapted from ``tinker_cookbook.distillation.train_on_policy.incorporate_kl_penalty``.
"""

from __future__ import annotations

import asyncio
import importlib
import logging
from typing import Any, cast

from ares.tinker_integration.opsd import config as opsd_config_mod

_LOGGER = logging.getLogger(__name__)

_PRIVILEGED_CONTEXT_TEMPLATE = """\
You have access to a detailed analysis of previous failed attempts at this task. \
Use this privileged information to guide your approach — pay close attention to \
the specific files, error patterns, and recommended steps.

## Analysis of Previous Failed Attempts
{reflection}

Apply these insights directly. Avoid the anti-patterns identified above and \
follow the recommended approach."""

_PRIVILEGED_ACK = """\
I have carefully reviewed the analysis of previous failed attempts. I understand \
the error patterns, root causes, key files involved, and the recommended approach. \
I will apply these insights to solve the task correctly."""


async def compute_teacher_logprobs_for_datum(
    sampling_client: Any,
    datum: Any,
    reflection: str,
    renderer: Any,
    max_trajectory_tokens: int,
) -> list[float] | None:
    """Compute teacher logprobs for a single datum using privileged context.

    The teacher is the SAME model but conditions on additional privileged context
    (the self-reflection).  We prepend the privileged context as a rendered
    conversation prefix to the student's full token sequence, then call
    ``compute_logprobs_async`` on the combined sequence.

    Returns:
        Teacher logprobs aligned with the student's target tokens, or None if
        the combined sequence exceeds context length.
    """
    tinker = importlib.import_module("tinker")

    # 1. Render privileged context as a conversation prefix.
    priv_content = _PRIVILEGED_CONTEXT_TEMPLATE.format(reflection=reflection)
    priv_messages = [
        {"role": "user", "content": priv_content},
        {"role": "assistant", "content": _PRIVILEGED_ACK},
    ]
    priv_input = renderer.build_generation_prompt(priv_messages)
    priv_tokens = priv_input.to_ints()

    # 2. Reconstruct student's full sequence (prompt + generated tokens).
    student_full = datum.model_input.append_int(cast(int, datum.loss_fn_inputs["target_tokens"].data[-1]))
    student_tokens = student_full.to_ints()

    # 3. Teacher sequence = privileged prefix + student sequence.
    combined_tokens = priv_tokens + student_tokens
    teacher_seq = tinker.ModelInput.from_ints(combined_tokens)

    # 4. Context length check.
    if teacher_seq.length > max_trajectory_tokens:
        return None

    # 5. Compute logprobs.
    all_logprobs = await sampling_client.compute_logprobs_async(teacher_seq)

    # 6. Extract logprobs for student's target tokens.
    # all_logprobs[i] = log P(token_i | token_0..token_{i-1})
    # The student's target tokens start at position len(priv_tokens).
    # We need logprobs for positions [len(priv_tokens)+1 : ] because
    # logprobs[i] predicts token_i given all previous tokens.
    # The student's target_tokens correspond to the generation portion,
    # which starts after the student's prompt.
    offset = len(priv_tokens)
    teacher_logprobs_for_student = all_logprobs[offset + 1 :]

    return teacher_logprobs_for_student


async def incorporate_teacher_kl(
    data_d: list[Any],
    reflections: dict[str, str],
    task_names_d: list[str],
    sampling_client: Any,
    renderer: Any,
    config: opsd_config_mod.OPSDConfig,
) -> dict[str, float]:
    """Compute reverse KL between student and teacher, adjusting advantages.

    For each datum, computes teacher logprobs (with privileged context) and
    applies the reverse KL penalty to the advantages in-place.

    Args:
        data_d: List of tinker.Datum objects from student rollouts.
        reflections: task_name -> reflection_text mapping.
        task_names_d: task_name for each datum.
        sampling_client: Current sampling client (shared student/teacher weights).
        renderer: Tinker renderer for building prompts.
        config: OPSD config.

    Returns:
        Metrics dict with KL statistics.
    """
    tinker = importlib.import_module("tinker")

    # Compute teacher logprobs for all datums concurrently.
    coros = []
    for datum, task_name in zip(data_d, task_names_d, strict=True):
        reflection = reflections.get(task_name, "")
        if not reflection:
            _LOGGER.warning("DISTILL | no reflection for task=%s, using empty", task_name)
        coros.append(
            compute_teacher_logprobs_for_datum(
                sampling_client,
                datum,
                reflection,
                renderer,
                config.max_trajectory_tokens,
            )
        )

    teacher_logprobs_results = await asyncio.gather(*coros)

    # Import torch lazily (only needed during distillation).
    torch = importlib.import_module("torch")

    total_kl_sum = 0.0
    total_mask_sum = 0.0
    num_skipped = 0

    for datum, teacher_logprobs in zip(data_d, teacher_logprobs_results, strict=True):
        if teacher_logprobs is None:
            num_skipped += 1
            continue

        sampled_logprobs = datum.loss_fn_inputs["logprobs"].to_torch()
        mask = datum.loss_fn_inputs["mask"].to_torch().float()

        # Align lengths — teacher logprobs might be longer or shorter.
        target_len = len(sampled_logprobs)
        teacher_lp = torch.tensor(teacher_logprobs[:target_len])

        # Pad if teacher logprobs are shorter.
        if len(teacher_lp) < target_len:
            pad = torch.zeros(target_len - len(teacher_lp))
            teacher_lp = torch.cat([teacher_lp, pad])

        # Reverse KL: KL(student || teacher) = log p_student - log p_teacher
        reverse_kl = (sampled_logprobs - teacher_lp) * mask

        # Advantage adjustment: negative reverse KL encourages matching teacher.
        kl_advantages = -config.distill_kl_penalty_coef * mask * reverse_kl

        # Optional discounting (for token-level credit assignment).
        if config.distill_kl_discount_factor > 0:
            rl_metrics = importlib.import_module("tinker_cookbook.rl.metrics")
            kl_advantages = torch.tensor(
                rl_metrics.discounted_future_sum_vectorized(kl_advantages.numpy(), config.distill_kl_discount_factor)
            )

        datum.loss_fn_inputs["advantages"] = tinker.TensorData.from_torch(
            datum.loss_fn_inputs["advantages"].to_torch() + kl_advantages
        )

        # Accumulate metrics.
        total_kl_sum += reverse_kl.sum().item()
        total_mask_sum += mask.sum().item()

    avg_kl = total_kl_sum / total_mask_sum if total_mask_sum > 0 else 0.0

    metrics = {
        "opsd/distill/teacher_kl": avg_kl,
        "opsd/distill/num_skipped_long": num_skipped,
        "opsd/distill/num_datums": len(data_d),
        "opsd/distill/num_valid": len(data_d) - num_skipped,
    }

    _LOGGER.info(
        "DISTILL KL | avg_kl=%.4f | valid=%d/%d | skipped_long=%d",
        avg_kl,
        len(data_d) - num_skipped,
        len(data_d),
        num_skipped,
    )

    return metrics
