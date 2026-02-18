# ARES Tinker Integration

RL training for code agents using [Tinker](https://tinker.build) + ARES's Harbor task system.

Two training scripts are available:

- **`examples/06_tinker_terminal_train.py`** (argparse, `--flag` style) — Uses terminal-based envs (tmux + JSON commands) via `HarborTerminalTinkerEnv`. Best for **Terminus2** agent on terminal/devops tasks.
- **`examples/05_tinker_train.py`** (chz, `key=value` style) — Uses ARES environments (`CodeEnvironment` + `QueueMediatedLLMClient`). Best for **Mini-SWE-Agent** on code modification / bug-fixing tasks (e.g., SWE-bench).

This README documents the terminal-based integration (example 06). See example 05's docstring for its usage.

## Prerequisites

```bash
# Python packages (in addition to ares itself)
pip install tinker tinker-cookbook harbor

# Environment variables
export TINKER_API_KEY="..."        # Required: Tinker service API key
export DAYTONA_API_KEY="..."       # Required for --env daytona
export WANDB_API_KEY="..."         # Optional: Weights & Biases logging
```

## Quick Start

```bash
# Single task verification
uv run python examples/06_tinker_terminal_train.py \
    --task-dir WORKING_TINKER/terminal_rl/harbor_envs/devops_task \
    --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/devops_verify \
    --env daytona

# Multi-task from ARES preset
uv run python examples/06_tinker_terminal_train.py \
    --preset tbench-terminus2 \
    --num-tasks 20 \
    --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
    --renderer-name qwen3 \
    --log-path ./runs/tbench_multi \
    --env daytona \
    --wandb-project ares-tinker
```

## Configuration Reference

### Task Source (mutually exclusive, one required)

| Flag | Description |
|---|---|
| `--task-dir PATH` | Single Harbor task directory for verification runs |
| `--preset NAME` | ARES preset name (e.g., `tbench-terminus2`, `sbv-terminus2`) |
| `--num-tasks N` | Limit number of tasks loaded from preset |

### Environment

| Flag | Default | Description |
|---|---|---|
| `--env` | `daytona` | Sandbox backend: `daytona` (cloud) or `docker` (local) |
| `--auto-stop-minutes` | `30` | Auto-stop idle Daytona sandboxes after N minutes |
| `--sandbox-cpus` | task default | CPU cores per sandbox |
| `--sandbox-memory-gb` | task default | RAM in GB per sandbox |
| `--sandbox-disk-gb` | task default | Disk in GB per sandbox |

### Model

| Flag | Default | Description |
|---|---|---|
| `--model-name` | (required) | Tinker model ID (e.g., `Qwen/Qwen3-30B-A3B-Instruct-2507`) |
| `--renderer-name` | auto | Renderer name (e.g., `qwen3`, `llama3`) |

### Training Hyperparameters

| Flag | Default | Description |
|---|---|---|
| `--learning-rate` | `4e-5` | Learning rate |
| `--lora-rank` | `32` | LoRA rank |
| `--temperature` | `1.0` | Sampling temperature |
| `--max-tokens` | `4096` | Max generation tokens per turn |
| `--group-size` | `5` | Rollouts per task (GRPO group size) |
| `--groups-per-batch` | `10` | Task groups per training batch |
| `--num-batches` | `15` | Total number of training batches |
| `--max-trajectory-tokens` | `32768` | Max context window tokens |

### Loss and Optimization

| Flag | Default | Description |
|---|---|---|
| `--loss-fn` | `importance_sampling` | Loss function (`importance_sampling` or `ppo`) |
| `--grad-clip-norm` | `0.5` | Gradient clipping norm |
| `--kl-penalty-coef` | `0.0` | KL penalty coefficient (0 = disabled) |
| `--remove-constant-reward-groups` | `false` | Filter out groups where all rollouts got the same reward |

### Async Training

Sync mode is the default and is proven to work. Pass `--max-steps-off-policy` to enable async mode.

| Flag | Default | Description |
|---|---|---|
| `--max-steps-off-policy` | `None` | Max steps off-policy (None = sync mode, e.g. 5 for async) |
| `--async-rollout-retries` | `5` | Retry attempts per group rollout before giving up |
| `--async-builder-buffer` | `2` | Extra builders per batch to compensate for lost rollouts |
| `--max-concurrent-sandboxes` | `20` | Cap concurrent sandbox creations (0 = no limit) |

### Logging and Checkpointing

| Flag | Default | Description |
|---|---|---|
| `--log-path` | (required) | Directory for logs and checkpoints |
| `--wandb-project` | `ares-tinker` | WandB project name |
| `--wandb-name` | `None` | WandB run name (auto-generated if omitted) |
| `--save-every` | `10` | Save checkpoint every N batches |
| `--eval-every` | `0` | Run evaluation every N batches (0 = disabled) |
| `--base-url` | `None` | Tinker service URL (uses default if omitted) |
| `--load-checkpoint-path` | `None` | Resume training from a checkpoint |

## Architecture

```
examples/06_tinker_terminal_train.py    CLI entry point
    |
    v
ares.tinker_integration.train          Orchestration + monkey-patches
    |                                   - grad clipping (optim_step)
    |                                   - error-resilient rollouts
    |                                   - empty-batch skip
    |                                   - wandb config fix
    v
ares.tinker_integration.dataset         Multi-task dataset layer
    |                                   - TerminalRLDatasetBuilder
    |                                   - TerminalRLDataset (batch sampling)
    |                                   - TerminalEnvGroupBuilder (GRPO groups)
    v
ares.tinker_integration.tinker_env      Tinker adapter
    |                                   - HarborTerminalTinkerEnv
    |                                   - JSON command parsing
    |                                   - Token-level RL interface
    v
ares.tinker_integration.terminal_env    Gym-like terminal environment
    |                                   - AsyncTerminalGymEnv
    |                                   - Tmux session management
    |                                   - Reward via Harbor Verifier
    v
Harbor EnvironmentFactory               Sandbox lifecycle
    |                                   (Daytona / Docker)
    v
tinker_cookbook.rl.train                 RL training loop
                                        (GRPO, PPO, importance sampling)
```

### RL Loop

1. **Reset**: Create sandbox, start tmux, capture initial terminal state.
2. **Observe**: Render message history into model input tokens via renderer.
3. **Act**: Model generates JSON with `{commands: [{keystrokes, duration}], task_complete}`.
4. **Execute**: Send keystrokes to tmux, wait for output.
5. **Repeat** until `task_complete: true` or context window exhausted.
6. **Reward**: Harbor Verifier checks task completion inside the sandbox.
7. **Train**: GRPO-style advantage estimation across the group, then gradient step.

### Error Resilience

The training loop survives transient infrastructure failures:

- **Individual rollout failures** (Daytona rate limits, sandbox conflicts, download errors) are retried with exponential backoff (up to `async_rollout_retries` attempts). If all retries fail, the group returns `None` and training continues with remaining successful rollouts.
- **Entire batch failures** (all rollouts failed) skip the train step entirely — no weight update for that batch, training proceeds to the next.
- **Gradient clipping** prevents training instability from outlier gradients.

## Module Reference

| File | Purpose |
|---|---|
| `config.py` | `TrainingConfig` dataclass with all training parameters |
| `train.py` | Main entry point (`run_training`), monkey-patches, orchestration |
| `dataset.py` | Dataset builders, task loading from presets or directories |
| `terminal_env.py` | `AsyncTerminalGymEnv` — gym-like wrapper over Harbor environments |
| `tinker_env.py` | `HarborTerminalTinkerEnv` — tinker-cookbook RL adapter |
