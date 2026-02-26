# OPSD Recipe (On-Policy Self-Distillation)

Iterative phasic training where the model learns from its own failures through self-reflection and knowledge distillation.

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
    --max-steps-off-policy 3 \
    --num-iterations 3 \
    --rl-batches-per-iteration 10 \
    --distill-batches 5 \
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
    --num-iterations 1 \
    --rl-batches-per-iteration 1 \
    --distill-batches 1
```

## How It Works

Each OPSD iteration runs these phases:

```
ITERATION 1
├── Phase 1: Student RL (10 batches of standard RL)
├── Phase 2: Evaluate student on ALL 50 tasks (6 rollouts each)
│   └── Identify hard tasks (0% success across all 6 rollouts)
├── Phase 3: Self-reflection (pure LLM call per hard task)
│   └── Extract condensed traces → generate compact hints
├── Phase 4: Teacher re-attempts hard tasks (same model + reflections)
│   └── Identify tasks teacher solved but student couldn't
├── Phase 5: On-policy distillation (reverse KL from teacher)
│   └── Student generates on-policy, teacher logprobs via token prepending
│
ITERATION 2 (repeat with updated model weights)
├── ...
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
├── train.py               run_opsd_training() — iterative phasic orchestrator
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
| `--num-iterations` | `3` | Number of OPSD iterations |
| `--rl-batches-per-iteration` | `10` | RL batches per iteration |
| `--eval-group-size` | `6` | Rollouts per task during evaluation |
| `--teacher-group-size` | `6` | Rollouts per task for teacher re-attempt |
| `--max-reflection-tokens` | `1024` | Max tokens for generated reflection |
| `--max-condensed-trace-tokens` | `2048` | Max tokens per condensed failure trace |
| `--num-traces-for-reflection` | `2` | Failed traces used per reflection |
| `--distill-batches` | `5` | Distillation batches per iteration |
| `--distill-groups-per-batch` | `10` | Task groups per distillation batch |
| `--distill-group-size` | `4` | Rollouts per distillation group |
| `--distill-kl-penalty-coef` | `1.0` | Reverse KL penalty coefficient |
| `--distill-kl-discount-factor` | `0.0` | Discount factor for KL penalty |

Key defaults:
- `--harness code-agent` (code-agent is the default for OPSD)
- `--wandb-project ares-tinker-opsd`

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
opsd/iteration: 0, 1, 2, ...
```

## Checkpoint & Resume

OPSD saves iteration state to `{log_path}/opsd_state.json` after each phase.
Tinker checkpoints are saved via `save_checkpoint_async` at regular intervals.

## Edge Cases

1. **No hard tasks**: All tasks solved in evaluation. Skip OPSD phases, continue to next iteration.
2. **All tasks hard**: None solved. Still run reflection + teacher — the teacher might solve some with hints.
3. **Teacher can't solve any**: Skip distillation. Log warning.
4. **Context overflow during teacher logprobs**: Skip individual datums where `len(priv_tokens) + len(student_tokens) > max_trajectory_tokens`. Count logged in `opsd/distill/num_skipped_long`.
5. **Sandbox failures**: Reuse existing retry logic (monkey-patched). Failed tasks excluded from results.
6. **Empty distillation batch**: All trajectory groups None. Skip train step (no weight update).
