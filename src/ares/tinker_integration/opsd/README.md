# OPSD Recipe (On-Policy Self-Distillation)

The model learns from its own failures through self-reflection and knowledge distillation.

## Quick Start

```bash
# OPSD on SWE-bench Verified with Mini-SWE-Agent (recommended)
uv run python examples/07_tinker_opsd_train.py \
    --harness code-agent \
    --preset sbv-mswea \
    --num-tasks 50 \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/opsd_sbv_mswea \
    --env daytona \
    --snapshot-template "ares__{name}" \
    --num-batches 200 \
    --groups-per-batch 32 \
    --group-size 8 \
    --opsd-every 1 \
    --num-distillation-steps 5 \
    --distill-kl-penalty-coef 1.0 \
    --wandb-project ares-tinker-opsd

# Minimal smoke test
uv run python examples/07_tinker_opsd_train.py \
    --preset sbv-mswea \
    --num-tasks 5 \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/opsd_smoke \
    --env docker \
    --num-batches 2 \
    --num-distillation-steps 1
```

## How It Works

The main loop runs `num_batches` RL batches. Every `opsd_every` batches, the OPSD phases trigger:

```
for batch in range(num_batches):              # e.g., 200
    rl_batch(groups_per_batch × group_size)    # 32×8 = 256 rollouts

    if (batch + 1) % opsd_every == 0:         # default: every batch
        1. Evaluate student on ALL tasks (eval_group_size rollouts each)
           └── Identify hard tasks (0% success)
        2. Self-reflect on failed traces (pure LLM call per hard task)
           └── Extract condensed traces → generate compact hints
        3. Teacher re-attempts hard tasks (same model + reflections)
           └── teacher_group_size rollouts each
        4. Filter: keep tasks teacher solved but student couldn't
        5. Distill: num_distillation_steps gradient steps
           └── Each step: student rollouts on ALL distillable tasks
               (same group_size as RL), compute teacher KL, train
```

### Key Insight

The teacher is NOT a different or larger model. It is the **same model** conditioned on additional privileged information (self-reflections extracted from previous failures). This creates a meaningful KL signal because the teacher's token probabilities differ from the student's despite identical weights.

### Teacher Logprob Computation

Since the teacher uses the same model weights, we compute teacher logprobs by prepending the privileged context (rendered as a conversation prefix) to the student's full token sequence:

```
teacher_sequence = [privileged_prefix_tokens] + [student_full_sequence_tokens]
teacher_logprobs = compute_logprobs_async(teacher_sequence)
# Extract logprobs at offset positions aligned to student targets
```

The reverse KL is: `KL(student || teacher) = log p_student - log p_teacher`.
This is added as a negative advantage to encourage the student to match the teacher.

## Module Structure

```
opsd/
├── __init__.py
├── config.py              OPSDConfig — extends TrainingConfig with OPSD-specific fields
├── train.py               run_opsd_training() — main loop + OPSD phase orchestration
├── evaluation.py          Student/teacher evaluation: run rollouts, identify hard tasks
├── reflection.py          Self-reflection: extract failure traces, generate hints via LLM
├── privileged_env.py      Env wrappers that inject reflection text as privileged context
└── distillation.py        Reverse KL computation: teacher logprobs via token prepending
```

### Data flow between modules

```
evaluation.evaluate_tasks()     →  EvalPhaseResult (hard_tasks)
                                        │
reflection.generate_reflections()  ←────┘
    │
    ▼
reflections: dict[str, str]     →  privileged_env wrappers (teacher re-attempt)
                                        │
evaluation.filter_teacher_solved()  ←───┘
    │
    ▼
distillable task pairs          →  distillation.incorporate_teacher_kl()
                                        │
                                   train_step() (student weight update)
```

## CLI Arguments

All flags from the [shared CLI reference](../README.md#shared-cli-reference) apply, plus these OPSD-specific ones:

| Flag | Default | Description |
|------|---------|-------------|
| `--opsd-every` | `1` | Run OPSD phases every N RL batches |
| `--eval-group-size` | `16` | Rollouts per task during evaluation |
| `--teacher-group-size` | `16` | Rollouts per task for teacher re-attempt |
| `--max-reflection-tokens` | `4096` | Max tokens for generated reflection |
| `--max-condensed-trace-tokens` | `4096` | Max tokens per condensed failure trace |
| `--num-traces-for-reflection` | `4` | Failed traces used per reflection |
| `--num-distillation-steps` | `5` | Gradient steps per OPSD cycle on distillable tasks |
| `--distill-kl-penalty-coef` | `1.0` | Reverse KL penalty coefficient |
| `--distill-kl-discount-factor` | `0.0` | Discount factor for KL penalty |

Key defaults:
- `--harness code-agent` (code-agent is the default for OPSD)
- `--wandb-project ares-tinker-opsd`
- Distillation uses the same `--group-size` as RL (no separate distill group size)

## Logging

OPSD logs comprehensive metrics to WandB under the `opsd/` prefix:

```
# Per-phase (logged at the end of each phase)
opsd/eval/total_tasks, opsd/eval/num_hard, opsd/eval/solve_rate
opsd/teacher/num_solved, opsd/teacher/solve_rate_on_hard
opsd/distill/teacher_kl, opsd/distill/num_skipped_long

# Per-task
opsd/task/{task_name}/student_reward
opsd/task/{task_name}/teacher_reward
opsd/task/{task_name}/is_hard
opsd/task/{task_name}/teacher_solved

# Phase tracking
opsd/phase: "rl" | "eval" | "reflection" | "teacher" | "distill"
```

## Checkpoint & Resume

OPSD saves state to `{log_path}/opsd_state.json` after each phase.
Tinker checkpoints are saved via `save_checkpoint_async` at regular intervals.

## Edge Cases

1. **No hard tasks**: All tasks solved in evaluation. Skip OPSD phases.
2. **All tasks hard**: None solved. Still run reflection + teacher — the teacher might solve some with hints.
3. **Teacher can't solve any**: Skip distillation. Log warning.
4. **Context overflow during teacher logprobs**: Skip individual datums where `len(priv_tokens) + len(student_tokens) > max_trajectory_tokens`. Count logged in `opsd/distill/num_skipped_long`.
5. **Sandbox failures**: Reuse existing retry logic (monkey-patched). Failed tasks excluded from results.
6. **Empty distillation step**: All trajectory groups None. Skip train step (no weight update).
