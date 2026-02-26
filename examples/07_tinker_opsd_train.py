"""ARES + Tinker OPSD Training (On-Policy Self-Distillation).

Iterative phasic training: student RL, hard-task identification, self-reflection,
teacher re-attempt, and reverse-KL distillation from a context-enriched teacher
(same model weights + privileged information).

Supports both harness modes:
- ``--harness terminal``: Direct tmux terminal control via JSON commands.
- ``--harness code-agent``: ARES CodeEnvironment with any agent harness.

Prerequisites:
    - Set TINKER_API_KEY environment variable.
    - Set DAYTONA_API_KEY if using --env daytona (default).
    - Install tinker, tinker-cookbook, and harbor dependencies.

Usage:
    # Code-agent harness — Mini-SWE-Agent on SWE-bench Verified (recommended)
    uv run python examples/07_tinker_opsd_train.py \\
        --harness code-agent \\
        --preset sbv-mswea \\
        --num-tasks 50 \\
        --model-name Qwen/Qwen3-4B-Instruct-2507 \\
        --renderer-name qwen3 \\
        --log-path ./runs/opsd_sbv_mswea \\
        --env daytona \\
        --snapshot-template "ares__{name}" \\
        --max-steps-off-policy 3 \\
        --num-iterations 3 \\
        --rl-batches-per-iteration 10 \\
        --distill-batches 5 \\
        --distill-kl-penalty-coef 1.0 \\
        --wandb-project ares-tinker-opsd \\
        --wandb-name "opsd-sbv-mswea-code-agent"

    # Terminal harness — multi-task from ARES preset
    uv run python examples/07_tinker_opsd_train.py \\
        --harness terminal \\
        --preset tbench-terminus2 \\
        --num-tasks 20 \\
        --model-name Qwen/Qwen3-30B-A3B-Instruct-2507 \\
        --renderer-name qwen3 \\
        --log-path ./runs/opsd_tbench \\
        --env daytona \\
        --num-iterations 3

    # Minimal smoke test (Docker, few tasks, 1 iteration)
    uv run python examples/07_tinker_opsd_train.py \\
        --preset sbv-mswea \\
        --num-tasks 5 \\
        --model-name Qwen/Qwen3-4B-Instruct-2507 \\
        --renderer-name qwen3 \\
        --log-path ./runs/opsd_smoke \\
        --env docker \\
        --num-iterations 1 \\
        --rl-batches-per-iteration 1 \\
        --distill-batches 1
"""

from __future__ import annotations

import argparse
import asyncio
import logging

from ares.tinker_integration.opsd import config as opsd_config_mod
from ares.tinker_integration.opsd import train as opsd_train


def parse_args() -> opsd_config_mod.OPSDConfig:
    p = argparse.ArgumentParser(
        description="ARES + Tinker OPSD Training (On-Policy Self-Distillation)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Harness mode
    p.add_argument(
        "--harness",
        type=str,
        default="code-agent",
        choices=["terminal", "code-agent"],
        help="Harness mode: 'terminal' (tmux + JSON) or 'code-agent' (ARES CodeEnvironment)",
    )

    # Task source
    task_group = p.add_mutually_exclusive_group(required=True)
    task_group.add_argument("--task-dir", type=str, help="Single task directory path (terminal harness only)")
    task_group.add_argument("--preset", type=str, help="ARES preset name (e.g., sbv-mswea, tbench-terminus2)")

    p.add_argument("--num-tasks", type=int, default=None, help="Limit tasks from preset")

    # Environment
    p.add_argument("--env", type=str, default="daytona", choices=["daytona", "docker"], help="Environment type")

    # Model
    p.add_argument("--model-name", type=str, required=True, help="HuggingFace model ID")
    p.add_argument("--renderer-name", type=str, default=None, help="Renderer name (auto-detected if None)")

    # Base training hyperparameters (for RL phase)
    p.add_argument("--learning-rate", type=float, default=4e-5, help="Learning rate")
    p.add_argument("--lora-rank", type=int, default=32, help="LoRA rank")
    p.add_argument("--temperature", type=float, default=1.0, help="Sampling temperature")
    p.add_argument("--max-tokens", type=int, default=4096, help="Max generation tokens")
    p.add_argument("--group-size", type=int, default=4, help="RL rollouts per task group")
    p.add_argument("--groups-per-batch", type=int, default=10, help="RL task groups per batch")
    p.add_argument("--num-batches", type=int, default=100, help="Total RL batches (across all iterations)")
    p.add_argument("--max-trajectory-tokens", type=int, default=32768, help="Max context tokens")

    # Loss and training options
    p.add_argument("--loss-fn", type=str, default="importance_sampling", help="Loss function")
    p.add_argument("--remove-constant-reward-groups", action="store_true", help="Filter constant reward groups")
    p.add_argument("--grad-clip-norm", type=float, default=0.5, help="Gradient clipping norm")
    p.add_argument("--kl-penalty-coef", type=float, default=0.0, help="RL KL penalty coefficient")

    # OPSD iteration control
    p.add_argument("--num-iterations", type=int, default=3, help="Number of OPSD iterations")
    p.add_argument("--rl-batches-per-iteration", type=int, default=10, help="RL batches per OPSD iteration")

    # Evaluation phase
    p.add_argument("--eval-group-size", type=int, default=6, help="Rollouts per task during evaluation")

    # Teacher phase
    p.add_argument("--teacher-group-size", type=int, default=6, help="Rollouts per task for teacher re-attempt")

    # Reflection phase
    p.add_argument("--max-reflection-tokens", type=int, default=1024, help="Max tokens for reflection generation")
    p.add_argument("--max-condensed-trace-tokens", type=int, default=2048, help="Max tokens per condensed trace")
    p.add_argument("--num-traces-for-reflection", type=int, default=2, help="Failed traces per task for reflection")

    # Distillation phase
    p.add_argument("--distill-batches", type=int, default=5, help="Distillation batches per iteration")
    p.add_argument("--distill-groups-per-batch", type=int, default=10, help="Task groups per distillation batch")
    p.add_argument("--distill-group-size", type=int, default=4, help="Rollouts per distillation task group")
    p.add_argument("--distill-kl-penalty-coef", type=float, default=1.0, help="KL penalty for distillation")
    p.add_argument("--distill-kl-discount-factor", type=float, default=0.0, help="Discount factor for KL")

    # Sandbox safety and resources
    p.add_argument("--auto-stop-minutes", type=int, default=30, help="Auto-stop idle sandboxes after N minutes")
    p.add_argument("--sandbox-cpus", type=int, default=None, help="CPU cores per sandbox")
    p.add_argument("--sandbox-memory-gb", type=int, default=None, help="RAM in GB per sandbox")
    p.add_argument("--sandbox-disk-gb", type=int, default=None, help="Disk in GB per sandbox")
    p.add_argument(
        "--snapshot-template",
        type=str,
        default=None,
        help="Daytona snapshot template with {name} placeholder (e.g., 'ares__{name}')",
    )
    p.add_argument(
        "--max-concurrent-sandboxes",
        type=int,
        default=20,
        help="Max concurrent sandbox creations (0=no limit)",
    )

    # Async mode
    p.add_argument(
        "--max-steps-off-policy",
        type=int,
        default=None,
        help="Enable async training with N max off-policy steps (e.g. 5). Omit for sync.",
    )
    p.add_argument("--async-rollout-retries", type=int, default=5, help="Max retry attempts per rollout")
    p.add_argument("--async-builder-buffer", type=int, default=2, help="Extra builders per batch in async mode")

    # Logging
    p.add_argument("--log-path", type=str, required=True, help="Path for logs and checkpoints")
    p.add_argument("--wandb-project", type=str, default="ares-tinker-opsd", help="WandB project name")
    p.add_argument("--wandb-name", type=str, default=None, help="WandB run name")
    p.add_argument("--save-every", type=int, default=10, help="Save checkpoint every N batches")
    p.add_argument("--eval-every", type=int, default=0, help="Evaluate every N batches (RL phase only)")
    p.add_argument("--base-url", type=str, default=None, help="Tinker service URL")
    p.add_argument("--load-checkpoint-path", type=str, default=None, help="Resume from checkpoint")

    args = p.parse_args()

    return opsd_config_mod.OPSDConfig(
        # Base TrainingConfig fields
        harness=args.harness,
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
        snapshot_template_name=args.snapshot_template,
        max_concurrent_sandboxes=args.max_concurrent_sandboxes or None,
        max_steps_off_policy=args.max_steps_off_policy,
        async_rollout_retries=args.async_rollout_retries,
        async_builder_buffer=args.async_builder_buffer,
        log_path=args.log_path,
        wandb_project=args.wandb_project,
        wandb_name=args.wandb_name,
        save_every=args.save_every,
        eval_every=args.eval_every,
        base_url=args.base_url,
        load_checkpoint_path=args.load_checkpoint_path,
        # OPSD-specific fields
        num_iterations=args.num_iterations,
        rl_batches_per_iteration=args.rl_batches_per_iteration,
        eval_group_size=args.eval_group_size,
        teacher_group_size=args.teacher_group_size,
        max_reflection_tokens=args.max_reflection_tokens,
        max_condensed_trace_tokens=args.max_condensed_trace_tokens,
        num_traces_for_reflection=args.num_traces_for_reflection,
        distill_batches=args.distill_batches,
        distill_groups_per_batch=args.distill_groups_per_batch,
        distill_group_size=args.distill_group_size,
        distill_kl_penalty_coef=args.distill_kl_penalty_coef,
        distill_kl_discount_factor=args.distill_kl_discount_factor,
    )


async def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
    cfg = parse_args()
    await opsd_train.run_opsd_training(cfg)


if __name__ == "__main__":
    asyncio.run(main())
