# ARES Tinker Integration

RL training for code agents using [Tinker](https://tinker.build) + ARES's [Harbor](https://github.com/withmartian/harbor) task system. Supports two harness modes through a single CLI entry point (`examples/06_tinker_terminal_train.py`).

| Harness | Flag | Environment | Best For |
|---------|------|-------------|----------|
| Terminal | `--harness terminal` (default) | tmux + JSON commands | Terminus2-style terminal/devops tasks |
| Code-agent | `--harness code-agent` | ARES CodeEnvironment + QueueMediatedLLMClient | Mini-SWE-Agent, any ARES agent on any preset |

## Prerequisites

```bash
# Python packages (in addition to ares itself)
pip install tinker tinker-cookbook harbor

# Required environment variables
export TINKER_API_KEY="..."        # Tinker service API key
export DAYTONA_API_KEY="..."       # Required when using --env daytona (default)

# Optional
export WANDB_API_KEY="..."         # Weights & Biases logging
```

## Quick Start

### Single task verification (terminal harness)

```bash
uv run python examples/06_tinker_terminal_train.py \
    --task-dir path/to/harbor_task \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/verify \
    --env docker
```

### Multi-task training from ARES preset

```bash
uv run python examples/06_tinker_terminal_train.py \
    --preset sbv-terminus2 \
    --num-tasks 20 \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/sbv_multi \
    --env daytona \
    --wandb-project ares-tinker
```

### With Daytona snapshots (faster sandbox creation)

```bash
# Step 1: Pre-create snapshots (run once)
uv run python -m ares.tinker_integration.create_snapshots \
    --preset sbv-terminus2 \
    --template "ares__{name}" \
    --num-tasks 50 \
    --concurrency 25

# Step 2: Train with --snapshot-template
uv run python examples/06_tinker_terminal_train.py \
    --preset sbv-terminus2 \
    --num-tasks 50 \
    --snapshot-template "ares__{name}" \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/sbv_snap \
    --env daytona \
    --group-size 6 \
    --groups-per-batch 32 \
    --num-batches 50 \
    --wandb-project ares-tinker
```

### Code-agent harness (Mini-SWE-Agent)

```bash
uv run python examples/06_tinker_terminal_train.py \
    --harness code-agent \
    --preset sbv-mswea \
    --num-tasks 20 \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/sbv_mswea \
    --env daytona \
    --max-tokens 4096
```

## Creating Snapshots

Snapshots are pre-built Daytona sandbox images that skip the declarative build step, making sandbox creation much faster. They are optional — omit `--snapshot-template` to use standard image builds.

```bash
uv run python -m ares.tinker_integration.create_snapshots \
    --preset sbv-terminus2 \
    --template "ares__{name}" \
    --num-tasks 20 \
    --concurrency 5 \
    --force-recreate  # optional: re-create existing snapshots
```

| Flag | Default | Description |
|------|---------|-------------|
| `--preset` | (required) | ARES preset name |
| `--template` | (required) | Snapshot name template, must contain `{name}` |
| `--num-tasks` | all | Limit number of tasks |
| `--concurrency` | `5` | Max concurrent snapshot creations |
| `--force-recreate` | `false` | Re-create even if active snapshot exists |

Template convention: `"ares__{name}"` — double underscore separates prefix from task name. Resources (CPU, memory, disk) are baked into the snapshot at creation time from the task config.

## Training Modes

### Sync (default, recommended)

On-policy training. All rollouts in a batch complete before the gradient step. Proven to work well (~2h for single-task, smooth learning curve).

```bash
uv run python examples/06_tinker_terminal_train.py \
    --preset sbv-terminus2 \
    --num-tasks 50 \
    --snapshot-template "ares__{name}" \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/sbv_sync \
    --env daytona \
    --group-size 6 \
    --groups-per-batch 32 \
    --num-batches 50 \
    --wandb-project ares-tinker \
    --wandb-name "sbv-terminus2-sync"
```

### Async (experimental)

Off-policy training with parallel rollout workers. Pass `--max-steps-off-policy` to enable. Higher throughput but requires careful tuning.

```bash
uv run python examples/06_tinker_terminal_train.py \
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
    --wandb-project ares-tinker \
    --wandb-name "sbv-terminus2-async"
```

## Resuming Training

### From a checkpoint

```bash
uv run python examples/06_tinker_terminal_train.py \
    --preset sbv-terminus2 \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/sbv_resumed \
    --env daytona \
    --load-checkpoint-path ./runs/sbv_sync/checkpoints/batch_10
```

### Resuming a WandB run

To continue logging to a previous WandB run (e.g., after a crash or checkpoint resume), set `WANDB_RESUME` and `WANDB_RUN_ID` environment variables. The run ID is shown in the WandB UI or in the initial training logs.

```bash
WANDB_RESUME=must WANDB_RUN_ID=<previous-run-id> \
uv run python examples/06_tinker_terminal_train.py \
    --preset sbv-terminus2 \
    --model-name Qwen/Qwen3-4B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/sbv_resumed \
    --env daytona \
    --load-checkpoint-path ./runs/sbv_sync/checkpoints/batch_10 \
    --wandb-project ares-tinker \
    --wandb-name "sbv-experiment-1"
```

`WANDB_RESUME=must` requires the run to already exist (fails if not found). Use `WANDB_RESUME=allow` to create a new run if the ID doesn't exist.

## CLI Reference

### Task Source (mutually exclusive, one required)

| Flag | Description |
|------|-------------|
| `--task-dir PATH` | Single Harbor task directory (terminal harness only) |
| `--preset NAME` | ARES preset name (e.g., `sbv-terminus2`, `tbench-terminus2`) |
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
| `--remove-constant-reward-groups` | `false` | Filter groups where all rollouts got the same reward |

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

## Harness Modes

### Terminal harness (default)

The model controls a tmux terminal session directly via JSON commands:

```json
{"commands": [{"keystrokes": "ls -la\n", "duration": 3}], "task_complete": false}
```

- Each turn: model sees terminal output, generates JSON with keystrokes
- Episode ends on `task_complete: true` or context window exhaustion (`too_long`)
- Reward from Harbor Verifier at episode end
- Context overflow terminates the episode with reward=0 (prevents infinite rollouts)

### Code-agent harness

Wraps ARES's `CodeEnvironment` with `QueueMediatedLLMClient`. The agent (Mini-SWE-Agent, Terminus2, etc.) runs naturally while LLM calls are intercepted and exposed as RL observations.

- Requires `--preset` (not `--task-dir`)
- Works with any ARES agent harness
- Episode ends on step limit (250), `task_complete`, or context overflow
- Uses ARES's standard middle-truncation for context management

## Architecture

```
examples/06_tinker_terminal_train.py     CLI entry point (argparse)
    |
    v
ares.tinker_integration.train            Orchestration + monkey-patches
    |                                    - grad clipping (optim_step)
    |                                    - error-resilient rollouts (retry + cleanup)
    |                                    - None filtering + empty-batch skip
    |                                    - wandb config fix
    v
ares.tinker_integration.dataset          Multi-task dataset layer
    |                                    - TerminalRLDatasetBuilder / AresRLDatasetBuilder
    |                                    - Task loading from presets or directories
    |                                    - GRPO group builders with builder buffer
    v
ares.tinker_integration.tinker_env       Tinker adapter (terminal harness)
    |          or .ares_env              Tinker adapter (code-agent harness)
    |                                    - Token-level RL interface
    |                                    - JSON command parsing / QueueMediatedLLMClient
    v
ares.tinker_integration.terminal_env     Gym-like terminal environment
    |                                    - AsyncTerminalGymEnv
    |                                    - Tmux session management
    |                                    - Harbor Verifier for rewards
    v
Harbor EnvironmentFactory                Sandbox lifecycle (Daytona / Docker)
    v
tinker_cookbook.rl.train                  RL training loop (GRPO, PPO, IS, CISPO)
```

### RL Loop (terminal harness)

1. **Reset**: Create sandbox, start tmux, capture initial terminal state
2. **Observe**: Render message history into model input tokens via renderer
3. **Act**: Model generates JSON `{commands: [{keystrokes, duration}], task_complete}`
4. **Execute**: Send keystrokes to tmux, wait for output
5. **Repeat** until `task_complete: true` or context window exhausted
6. **Reward**: Harbor Verifier checks task completion inside the sandbox
7. **Train**: GRPO-style advantage estimation across the group, then gradient step

## Error Resilience

The training loop survives transient infrastructure failures through several mechanisms:

**Rollout retries**: Individual rollout failures (Daytona rate limits, connection errors, sandbox conflicts) are retried with exponential backoff up to `--async-rollout-retries` attempts. If all retries fail, the group returns `None` and training continues with remaining successful rollouts.

**Empty batch guard**: When all rollouts in a batch fail, the train step is skipped entirely (no weight update). Training proceeds to the next batch.

**Sandbox cleanup**: Failed rollouts close their sandboxes immediately via a tracking wrapper on `make_envs`. Prevents Daytona sandbox leaks.

**Wrapper chaining prevention**: The `make_envs` wrapper is saved/restored in a `finally` block to prevent wrapper chain growth when async mode requeues stale builders.

**Verification resilience**: Harbor's `Verifier.verify()` can throw `AddTestsDirError` / `DownloadVerifierDirError` under Daytona file API load (thundering herd with snapshots). The `_safe_verify()` method retries 3x with backoff; if all fail, returns reward=0 instead of crashing the group.

**Context overflow handling**: When the context window fills up, the terminal harness terminates the episode with `episode_done=True` and `reward=0` (`too_long=1.0` metric). This prevents infinite rollouts that would block gradient steps.

**Gradient clipping**: All gradient updates use configurable norm clipping (default 0.5) to prevent training instability from outlier gradients.

## Module Reference

| File | Purpose |
|------|---------|
| `config.py` | `TrainingConfig` dataclass — all training parameters with validation |
| `train.py` | `run_training()` entry point — orchestration, monkey-patches, config wiring |
| `dataset.py` | `TerminalRLDatasetBuilder` — multi-task dataset, task loading, GRPO group builders |
| `terminal_env.py` | `AsyncTerminalGymEnv` — gym-like wrapper over Harbor + tmux with reward verification |
| `tinker_env.py` | `HarborTerminalTinkerEnv` — tinker-cookbook RL adapter for terminal harness |
| `ares_env.py` | `AresCodeTinkerEnv` + `AresRLDatasetBuilder` — code-agent harness adapter |
| `create_snapshots.py` | `create_snapshots()` — bulk Daytona snapshot creation CLI |
| `__init__.py` | Public API re-exports |
