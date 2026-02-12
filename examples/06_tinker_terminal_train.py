"""ARES + Tinker Terminal RL Training.

Train code agents using direct terminal control with Tinker's RL infrastructure.
This uses the proven terminal-based architecture (tmux + JSON commands) that produces
a clean learning signal, combined with ARES's Harbor task loading and preset system.

Prerequisites:
    - Set TINKER_API_KEY environment variable.
    - Set DAYTONA_API_KEY if using --env daytona (default).
    - Install tinker, tinker-cookbook, and harbor dependencies.

Usage:
    # Single task (for verification — matches working reference behavior)
    uv run python examples/06_tinker_terminal_train.py \
        --task-dir WORKING_TINKER/terminal_rl/harbor_envs/devops_task \
        --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --renderer-name qwen3 \
        --log-path ./runs/devops_verify \
        --env docker

    # Multi-task from ARES preset
    uv run python examples/06_tinker_terminal_train.py \
        --preset tbench-terminus2 \
        --num-tasks 20 \
        --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --renderer-name qwen3 \
        --log-path ./runs/tbench_multi \
        --env daytona

    # With WandB logging and async rollouts
    uv run python examples/06_tinker_terminal_train.py \
        --preset tbench-terminus2 \
        --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \
        --renderer-name qwen3 \
        --log-path ./runs/tbench_async \
        --wandb-project ares-terminal-rl \
        --max-steps-off-policy 5
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from ares.tinker_integration import config as config_mod
from ares.tinker_integration import train


def parse_args() -> config_mod.TrainingConfig:
    p = argparse.ArgumentParser(
        description="ARES + Tinker Terminal RL Training",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Task source (mutually exclusive)
    task_group = p.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task-dir", type=str, help="Single task directory path")
    task_group.add_argument("--preset", type=str, help="ARES preset name (e.g., tbench-terminus2)")

    p.add_argument("--num-tasks", type=int, default=None, help="Limit tasks from preset")

    # Environment
    p.add_argument("--env", type=str, default="daytona", choices=["daytona", "docker"], help="Environment type")

    # Model
    p.add_argument("--model-name", type=str, required=True, help="HuggingFace model ID")
    p.add_argument("--renderer-name", type=str, default=None, help="Renderer name (auto-detected if None)")

    # Training hyperparameters
    p.add_argument("--learning-rate", type=float, default=4e-5, help="Learning rate")
    p.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max generation tokens")
    p.add_argument("--group-size", type=int, default=5, help="Rollouts per task")
    p.add_argument("--groups-per-batch", type=int, default=10, help="Task groups per batch")
    p.add_argument("--num-batches", type=int, default=15, help="Number of training batches")
    p.add_argument("--max-trajectory-tokens", type=int, default=32768, help="Max context tokens")

    # Loss and training options
    p.add_argument("--loss-fn", type=str, default="importance_sampling", help="Loss function")
    p.add_argument("--remove-constant-reward-groups", action="store_true", help="Filter constant reward groups")
    p.add_argument("--grad-clip-norm", type=float, default=0.5, help="Gradient clipping norm")
    p.add_argument("--kl-penalty-coef", type=float, default=0.0, help="KL penalty coefficient")

    # Sandbox safety and resources
    p.add_argument("--auto-stop-minutes", type=int, default=30, help="Auto-stop idle sandboxes after N minutes")
    p.add_argument("--sandbox-cpus", type=int, default=None, help="CPU cores per sandbox (default: task config)")
    p.add_argument("--sandbox-memory-gb", type=int, default=None, help="RAM in GB per sandbox (default: task config)")
    p.add_argument("--sandbox-disk-gb", type=int, default=None, help="Disk in GB per sandbox (default: task config)")

    # Async
    p.add_argument("--max-steps-off-policy", type=int, default=None, help="Max steps off-policy (None=sync)")

    # Logging
    p.add_argument("--log-path", type=str, required=True, help="Path for logs and checkpoints")
    p.add_argument("--wandb-project", type=str, default="ares-tinker", help="WandB project name")
    p.add_argument("--wandb-name", type=str, default=None, help="WandB run name")
    p.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N batches")
    p.add_argument("--eval-every", type=int, default=0, help="Evaluate every N batches")
    p.add_argument("--base-url", type=str, default=None, help="Tinker service URL")
    p.add_argument("--load-checkpoint-path", type=str, default=None, help="Resume from checkpoint")

    args = p.parse_args()

    return config_mod.TrainingConfig(
        model_name=args.model_name,
        renderer_name=args.renderer_name,
        env_type=args.env,
        task_dir=args.task_dir,
        preset_name=args.preset,
        num_tasks=args.num_tasks,
        learning_rate=args.learning_rate,
        lora_rank=args.lora_rank,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        group_size=args.group_size,
        groups_per_batch=args.groups_per_batch,
        num_batches=args.num_batches,
        max_trajectory_tokens=args.max_trajectory_tokens,
        loss_fn=args.loss_fn,
        remove_constant_reward_groups=args.remove_constant_reward_groups,
        grad_clip_norm=args.grad_clip_norm,
        kl_penalty_coef=args.kl_penalty_coef,
        auto_stop_minutes=args.auto_stop_minutes,
        sandbox_cpus=args.sandbox_cpus,
        sandbox_memory_gb=args.sandbox_memory_gb,
        sandbox_disk_gb=args.sandbox_disk_gb,
        max_steps_off_policy=args.max_steps_off_policy,
        log_path=args.log_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        save_every=args.save_every,
        eval_every=args.eval_every,
        base_url=args.base_url,
        load_checkpoint_path=args.load_checkpoint_path,
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = parse_args()
    await train.run_training(cfg)


if __name__ == "__main__":
    asyncio.run(main())
