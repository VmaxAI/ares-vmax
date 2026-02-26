# ARES Tinker Integration

RL training for code agents using [Tinker](https://tinker.build) + ARES's [Harbor](https://github.com/withmartian/harbor) task system. Two recipes and two harness modes.

## Recipes

| Recipe | Entry Point | Docs | Description |
|--------|-------------|------|-------------|
| **RL** | `examples/06_tinker_rl_train.py` | [`rl/README.md`](rl/README.md) | Standard GRPO-style RL training (sync or async) |
| **OPSD** | `examples/07_tinker_opsd_train.py` | [`opsd/README.md`](opsd/README.md) | On-Policy Self-Distillation — iterative phasic training with self-reflection and reverse-KL distillation |

## Harness Modes

| Harness | Flag | Environment | Best For |
|---------|------|-------------|----------|
| Terminal | `--harness terminal` (default for RL) | tmux + JSON commands | Terminus2-style terminal/devops tasks |
| Code-agent | `--harness code-agent` (default for OPSD) | ARES CodeEnvironment + QueueMediatedLLMClient | Mini-SWE-Agent, any ARES agent on any preset |

## Prerequisites

```bash
# Install tinker dependencies
uv sync --extra tinker

# Required environment variables
export TINKER_API_KEY="..."        # Tinker service API key
export DAYTONA_API_KEY="..."       # Required when using --env daytona (default)

# Optional
export WANDB_API_KEY="..."         # Weights & Biases logging
```

## Architecture

```
src/ares/tinker_integration/
├── __init__.py                    Public API re-exports
├── config.py                      TrainingConfig dataclass (shared)
├── dataset.py                     Multi-task dataset layer (shared)
├── terminal_env.py                AsyncTerminalGymEnv — gym-like terminal wrapper
├── tinker_env.py                  HarborTerminalTinkerEnv — terminal harness adapter
├── ares_env.py                    AresCodeTinkerEnv — code-agent harness adapter
├── create_snapshots.py            Bulk Daytona snapshot creation
├── monkey_patches.py              MonkeyPatchContext — shared monkey-patches
├── train.py                       Backward-compat shim (delegates to rl.train)
│
├── rl/
│   ├── __init__.py
│   ├── train.py                   run_training() — standard RL entry point
│   └── README.md
│
└── opsd/
    ├── __init__.py
    ├── config.py                  OPSDConfig — extends TrainingConfig
    ├── train.py                   run_opsd_training() — iterative phasic orchestrator
    ├── evaluation.py              Student/teacher evaluation phases
    ├── reflection.py              Self-reflection from failed traces
    ├── privileged_env.py          Env wrappers injecting privileged context
    ├── distillation.py            Reverse KL computation + teacher logprobs
    └── README.md

examples/
├── 06_tinker_rl_train.py          RL recipe CLI entry point
└── 07_tinker_opsd_train.py        OPSD recipe CLI entry point
```

## Shared Monkey-Patches

Both recipes apply the same set of monkey-patches via `MonkeyPatchContext`:

1. **wandb config.update** — Allow value changes (duplicate-key workaround).
2. **optim_step** — Gradient clipping via `AdamParams.grad_clip_norm`.
3. **do_group_rollout_and_filter** — Retry on transient sandbox errors (exponential backoff).
4. **remove_constant_reward_groups** — Filter `None` trajectory groups.
5. **do_train_step** — Skip empty batches, filter `None` groups from builders.
6. **do_group_rollout** — Close sandboxes on failure + prevent wrapper chaining.

## Shared CLI Reference

### Task Source (mutually exclusive, one required)

| Flag | Description |
|------|-------------|
| `--task-dir PATH` | Single Harbor task directory (terminal harness only) |
| `--preset NAME` | ARES preset name (e.g., `sbv-mswea`, `tbench-terminus2`) |
| `--num-tasks N` | Limit number of tasks loaded from preset |

### Environment

| Flag | Default | Description |
|------|---------|-------------|
| `--env` | `daytona` | Sandbox backend: `daytona` or `docker` |
| `--auto-stop-minutes` | `30` | Auto-stop idle Daytona sandboxes |
| `--sandbox-cpus` | task default | CPU cores per sandbox |
| `--sandbox-memory-gb` | task default | RAM (GB) per sandbox |
| `--sandbox-disk-gb` | task default | Disk (GB) per sandbox |
| `--snapshot-template` | none | Snapshot template with `{name}` placeholder |
| `--max-concurrent-sandboxes` | `20` | Cap concurrent sandbox creations (0 = no limit) |

### Model

| Flag | Default | Description |
|------|---------|-------------|
| `--model-name` | (required) | HuggingFace model ID |
| `--renderer-name` | auto | Renderer name (e.g., `qwen3`, `llama3`) |

### Training Hyperparameters

| Flag | Default | Description |
|------|---------|-------------|
| `--learning-rate` | `4e-5` | Learning rate |
| `--lora-rank` | `32` | LoRA rank |
| `--temperature` | `1.0` | Sampling temperature |
| `--max-tokens` | `4096` | Max generation tokens per turn |
| `--group-size` | `4` | Rollouts per task (GRPO group size) |
| `--groups-per-batch` | `10` | Task groups per training batch |
| `--num-batches` | `15` | Total training batches |
| `--max-trajectory-tokens` | `32768` | Max context window tokens |

### Loss and Optimization

| Flag | Default | Description |
|------|---------|-------------|
| `--loss-fn` | `importance_sampling` | Loss function (`importance_sampling`, `ppo`, `cispo`) |
| `--grad-clip-norm` | `0.5` | Gradient clipping norm |
| `--kl-penalty-coef` | `0.0` | KL penalty coefficient (0 = disabled) |

### Async Training

| Flag | Default | Description |
|------|---------|-------------|
| `--max-steps-off-policy` | none | Enable async mode (e.g., `3` or `5`). Omit for sync. |
| `--async-rollout-retries` | `5` | Retry attempts per group rollout |
| `--async-builder-buffer` | `2` | Extra builders per batch to compensate for lost rollouts |

### Logging and Checkpointing

| Flag | Default | Description |
|------|---------|-------------|
| `--log-path` | (required) | Directory for logs and checkpoints |
| `--wandb-project` | `ares-tinker` | WandB project name |
| `--wandb-name` | none | WandB run name (auto-generated if omitted) |
| `--save-every` | `10` | Save checkpoint every N batches |
| `--eval-every` | `0` | Evaluate every N batches (0 = disabled) |
| `--base-url` | none | Tinker service URL override |
| `--load-checkpoint-path` | none | Resume from checkpoint |

## Creating Snapshots

```bash
uv run python -m ares.tinker_integration.create_snapshots \
    --preset sbv-terminus2 \
    --template "ares__{name}" \
    --num-tasks 50 \
    --concurrency 25
```

## Error Resilience

**Rollout retries**: Transient errors retried with exponential backoff (up to `--async-rollout-retries`).
**Empty batch guard**: Batches where all rollouts fail are skipped (no weight update).
**Sandbox cleanup**: Failed rollouts close sandboxes immediately.
**Context overflow**: Episodes terminate with `reward=0` and `too_long=1.0`.
**Verification resilience**: Harbor verify retried 3x with backoff, fallback to `reward=0`.
