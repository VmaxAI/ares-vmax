# RL Recipe

Standard GRPO-style RL training for code agents. Supports sync (on-policy) and async (off-policy with CISPO) modes.

## Quick Start

```bash
# Code-agent harness — Mini-SWE-Agent on SWE-bench Verified
uv run python examples/06_tinker_rl_train.py \
    --harness code-agent \
    --preset sbv-mswea \
    --num-tasks 20 \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/sbv_mswea \
    --env daytona

# Terminal harness — single task (for verification)
uv run python examples/06_tinker_rl_train.py \
    --task-dir path/to/harbor_task \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/verify \
    --env docker

# Async CISPO training with Daytona snapshots
uv run python examples/06_tinker_rl_train.py \
    --preset sbv-terminus2 \
    --num-tasks 50 \
    --snapshot-template "ares__{name}" \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/sbv_async \
    --env daytona \
    --max-steps-off-policy 3 \
    --loss-fn cispo \
    --group-size 6 \
    --groups-per-batch 32 \
    --num-batches 50 \
    --wandb-project ares-tinker
```

## How It Works

Each training batch:

1. Sample `groups_per_batch` tasks (with replacement).
2. Run `group_size` rollouts per task in sandboxes.
3. Compute GRPO-style advantages (reward centered within each group).
4. Forward-backward pass via Tinker, update weights with LoRA.

**Sync mode** (default): Rollouts and training are sequential — on-policy.

**Async mode** (`--max-steps-off-policy N`): Rollouts run concurrently with training. Data up to N weight updates old is accepted. Uses CISPO loss for off-policy correction. Requires snapshots for fast sandbox startup.

## Module Structure

```
rl/
├── __init__.py
└── train.py          run_training() — orchestrates setup, dataset, and tinker_cookbook.rl.train.main()
```

`run_training()` delegates to `tinker_cookbook.rl.train.main()` with monkey-patches applied via `MonkeyPatchContext` from the parent package.

## CLI Reference

All flags from the [shared CLI reference](../README.md#shared-cli-reference) apply. The RL recipe has no additional flags beyond the shared ones.

Key defaults:
- `--harness terminal` (terminal harness is the default)
- `--loss-fn importance_sampling` (sync on-policy)
- `--wandb-project ares-tinker`
