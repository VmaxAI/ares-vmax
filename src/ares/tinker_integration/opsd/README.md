# OPSD Recipe (On-Policy Self-Distillation)

The model learns from its own failures through self-reflection and knowledge distillation.

## Diagnostic Report: Training Collapse Analysis

### Observed Symptoms (first run)

| Metric | Behavior |
|--------|----------|
| reward | Crashed to 0 |
| LLM entropy | Crashed to 0 |
| parse_correctness | Crashed to 0 |
| Model output | Degenerate / garbage |
| `distill/num_tasks` | 3 (out of 50) |
| `distill/num_valid_groups` | 3 |
| `distill/num_valid` | 15–30 |
| `distill/too_long_skipped` | 1 |
| `distill/num_datums` | ~30 |

### Root Cause #1: Catastrophically small distillation batches (CRITICAL)

**The fundamental problem.** The RL recipe works at `32×8 = 256` rollouts per batch. The
distillation phase trains on ~15–30 datums from ~3 tasks. That's a **10–17× reduction** in
effective batch size, causing massive gradient variance.

**Why so few distillable tasks?** The filtering pipeline is a harsh funnel:

```
50 tasks in preset
 └─→ ~32 sampled for RL batch (random.choices with duplicates)
      └─→ ~20–25 unique tasks appear in batch
           └─→ ~15–20 are "hard" (all 8 rollouts failed) — for a 4B model on SWE-bench, most fail
                └─→ reflection generated for each hard task
                     └─→ teacher (same model + reflection) re-attempts each
                          └─→ ~3 tasks solved by teacher (4B model, even with hints, rarely succeeds)
                               └─→ 3 tasks × 8 rollouts = 24 datums available for distillation
```

The reference OPD implementation uses `groups_per_batch=1024` with `group_size=4` (4096 rollouts)
and a genuinely stronger teacher (Qwen3-32B). Our setup has **~100× less training data** per
distillation step.

### Root Cause #2: Multiple gradient steps on the same tiny data

With `num_distillation_steps=5`, the code does 5 gradient steps on the **same** ~24 datums.
Each step uses the full learning rate (4e-5). This causes:

1. **Catastrophic overfitting** — the model memorizes the 3 specific tasks' patterns
2. **Policy collapse** — weights shift dramatically toward a narrow distribution
3. **Entropy collapse** — the model becomes overconfident on the memorized patterns
4. **Cascading failure** — the degraded model then performs worse on RL batches,
   producing even fewer distillable tasks in subsequent cycles

The OPD blog notes that multi-epoch distillation works because KL provides O(N) bits per
episode. But with N=24 datums, even O(N) is too little to justify 5 gradient steps at full LR.

### Root Cause #3: Weak self-distillation signal

The original OPD uses a **different, larger teacher** (e.g., 32B teaching 8B). OPSD uses the
**same 4B model** conditioned on reflection text. The KL signal is fundamentally weaker:

- Same weights → same tendencies for most tokens
- Only tokens directly influenced by the reflection have meaningful KL divergence
- The signal-to-noise ratio is much lower than true cross-model distillation
- Combined with tiny batch sizes, the gradient from KL is dominated by noise

### Root Cause #4: Distillation rollouts are on tasks the student can't solve

During distillation, the student generates new rollouts on distillable tasks — but these are
tasks the student **failed on** (0% success in RL). The student will likely fail again,
producing garbage trajectories. The teacher's KL correction on garbage trajectories is noisy
and weak — the trajectory is so far from any solution path that per-token corrections don't
compose into a meaningful learning signal.

Compare to standard OPD where both student and teacher can solve the problem — the student
just takes a slightly worse path, and the teacher's corrections are fine-grained improvements
on an already reasonable trajectory.

### Root Cause #5: No minimum batch size safety threshold

The code proceeds with distillation even when `num_datums=15`. There's no guard preventing
gradient updates from dangerously small batches.

### Speed Issues

Each OPSD cycle is ~3–4× slower than a pure RL batch:

| Phase | Time estimate | Notes |
|-------|--------------|-------|
| RL batch | 10–15 min | 32×8 sandbox rollouts + train |
| Reflection | 2–3 min | N LLM calls (no sandboxes) |
| Teacher re-attempt | 10–15 min | N×8 sandbox rollouts |
| Distillation | 10–20 min | Rollouts + KL computation + 5 train steps |
| **Total per cycle** | **35–50 min** | vs ~10–15 min for pure RL |

---

## Recommended Fixes

### Fix 1: Distillation replay buffer (HIGHEST IMPACT)

Instead of training on each cycle's tiny batch, **accumulate** distillable data across
multiple OPSD cycles and only train when the buffer is large enough.

```
distill_buffer = []  # persists across OPSD cycles

for each OPSD cycle:
    distill_buffer.extend(new_distillable_data)  # add this cycle's data
    if len(distill_buffer) >= min_distill_batch_size:  # e.g., 128
        train on distill_buffer
        distill_buffer = []  # or keep fraction for stability
```

**Why this works:** Over multiple cycles, we accumulate data from different tasks, different
reflections, and different student states. This provides the diversity needed for stable
gradient updates. Even if each cycle only finds 3 distillable tasks, after 10 cycles we
have 30 tasks × 8 rollouts = 240 datums — comparable to an RL batch.

Implementation: add `distill_min_batch_size` config field (default: 128). Accumulate
`(task, reflection, trajectory_group)` tuples in the training loop. When the buffer
exceeds the threshold, run a single distillation step. Re-roll student trajectories fresh
to stay on-policy (use stored reflections for teacher KL).

### Fix 2: `num_distillation_steps=1` (ALREADY PLANNED)

Never do more than 1 gradient step per cycle on the same data. The user's next run
correctly reduces this to 1. With tiny batches, even 1 step can be risky — see Fix 1.

### Fix 3: Lower distillation learning rate

Use a reduced learning rate for distillation steps. The gradient variance from small
batches scales as `1/batch_size`, so halving the batch requires halving the LR to
maintain the same expected weight update magnitude.

Add `distill_learning_rate` config field. Suggested: `1e-5` (vs `4e-5` for RL).

### Fix 4: Scale up task diversity with available capacity

The user has 1020 sandbox slots. Current usage: `32×8 = 256`. Options:

| Config | Sandboxes | Unique tasks | Expected distillable |
|--------|-----------|-------------|---------------------|
| 32×8 (current) | 256 | ~25 | ~3 |
| 64×4 | 256 | ~40 | ~5 |
| 100×4 | 400 | ~45 | ~6 |
| 200×4 | 800 | ~50 | ~7 |

**Key insight:** `groups_per_batch=64, group_size=4` is better than `32×8` for OPSD because:
- More unique tasks sampled → more hard tasks discovered → more potential distillable tasks
- Fewer rollouts per task → less wasted compute on tasks the student can't solve anyway
- GRPO still works with group_size=4 (the reference OPD uses `group_size=4`)
- The RL signal is somewhat noisier per task, but the diversity is more valuable for OPSD

**For the teacher phase specifically**, `teacher_group_size=4` is sufficient — we only need
to determine if the task is solvable with hints, not get a precise success rate.

### Fix 5: Minimum batch size guard

Add a guard in `_run_distillation_steps`:

```python
if len(data_d) < config.distill_min_batch_size:
    _LOGGER.info("DISTILL | only %d datums (min=%d), accumulating for next cycle",
                 len(data_d), config.distill_min_batch_size)
    # store data for next cycle
    return sampling_client, global_batch
```

### Fix 6: Speed improvements

1. **Reduce teacher_group_size to 4** — we only need pass/fail, not precise success rate.
2. **Cache reflections** — if the same task is hard in consecutive cycles, reuse the
   reflection (the model hasn't changed much between adjacent RL batches).
3. **Run `opsd_every=3` instead of 1** — accumulate more RL experience before attempting
   distillation. This also amortizes the overhead of reflection + teacher phases.
4. **Parallelize reflection generation** — already done (asyncio.gather).
5. **Skip reflection for tasks with no trajectories** — edge case but saves time.

### Fix 7: Reduce KL penalty coefficient

With `distill_kl_penalty_coef=1.0` and the weak self-distillation signal, the KL penalty
can push the model in noisy directions. Reduce to `0.1–0.3` to be more conservative.
The user's next run uses `0.5`, which is a step in the right direction.

---

## Recommended Settings

### Conservative (prioritize stability)

```bash
uv run python examples/07_tinker_opsd_train.py \
    --harness code-agent \
    --preset sbv-mswea \
    --num-tasks 50 \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/opsd_conservative \
    --env daytona \
    --snapshot-template "ares__{name}" \
    --max-steps-off-policy 3 \
    --loss-fn cispo \
    --learning-rate 4e-5 \
    --lora-rank 32 \
    --group-size 4 \
    --groups-per-batch 64 \
    --num-batches 100 \
    --max-tokens 4096 \
    --opsd-every 3 \
    --num-distillation-steps 1 \
    --distill-kl-penalty-coef 0.3 \
    --teacher-group-size 4 \
    --num-traces-for-reflection 4 \
    --max-reflection-tokens 4096 \
    --max-condensed-trace-tokens 4096 \
    --wandb-project ares-tinker \
    --wandb-name "opsd-conservative"
```

Key differences from the failed run:
- `group_size=4` (was 8) — more unique tasks per batch
- `groups_per_batch=64` (was 32) — 2× more groups, same sandbox count (256)
- `opsd_every=3` (was 1) — less frequent distillation, more RL stabilization between
- `num_distillation_steps=1` (was 5) — single gradient step
- `distill_kl_penalty_coef=0.3` (was 1.0) — conservative KL coefficient
- `teacher_group_size=4` (was 8) — faster teacher phase

### After implementing replay buffer (Fix 1)

```bash
# Same as conservative, but with replay buffer enabled
    --distill-min-batch-size 128 \
    --group-size 4 \
    --groups-per-batch 100 \
    --opsd-every 1 \
    --num-distillation-steps 1 \
    --distill-kl-penalty-coef 0.3 \
```

This uses 400 sandboxes, discovers more distillable tasks per cycle, and accumulates
them until we have ≥128 datums before training. With ~5 distillable tasks per cycle at
group_size=4, we'd accumulate ~20 datums/cycle, training every ~7 cycles.

### Maximum throughput (uses all 1020 sandbox slots)

```bash
    --group-size 4 \
    --groups-per-batch 200 \
    --distill-min-batch-size 128 \
    --opsd-every 1 \
    --teacher-group-size 4 \
```

800 sandboxes for RL, ~50 unique tasks per batch, ~7+ distillable per cycle.
With the replay buffer, trains every ~4 cycles.

---

## How It Works

The main loop runs `num_batches` RL batches. Every `opsd_every` batches, the OPSD phases trigger:

```
for batch in range(num_batches):              # e.g., 200
    rl_batch(groups_per_batch × group_size)    # 32×8 = 256 rollouts

    if (batch + 1) % opsd_every == 0:         # default: every batch
        1. Use RL batch results to identify hard tasks (0% success)
        2. Self-reflect on failed traces (pure LLM call per hard task)
           └── Extract condensed traces → generate compact hints
        3. Teacher re-attempts hard tasks (same model + reflections)
           └── teacher_group_size rollouts each
        4. Filter: keep tasks teacher solved but student couldn't
        5. Distill: num_distillation_steps gradient steps
           └── Student rollouts on distillable tasks (same group_size),
               compute teacher KL, train
```

### Key Insight

The teacher is NOT a different or larger model. It is the **same model** conditioned on
additional privileged information (self-reflections extracted from previous failures). This
creates a meaningful KL signal because the teacher's token probabilities differ from the
student's despite identical weights.

### Teacher Logprob Computation

Since the teacher uses the same model weights, we compute teacher logprobs by prepending
the privileged context (rendered as a conversation prefix) to the student's full token
sequence:

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
RL batch results (from _do_rl_batch)  →  identify hard tasks (all_failed=True)
                                              │
reflection.generate_reflections()  ←──────────┘
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
| `--teacher-group-size` | `0` | Rollouts per task for teacher re-attempt (0=group_size) |
| `--max-reflection-tokens` | `4096` | Max tokens for generated reflection |
| `--max-condensed-trace-tokens` | `4096` | Max tokens per condensed failure trace |
| `--num-traces-for-reflection` | `4` | Failed traces used per reflection |
| `--num-distillation-steps` | `1` | Gradient steps per OPSD cycle on distillable tasks |
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
