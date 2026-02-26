"""Shared monkey-patches for Tinker training recipes.

Extracted from the original ``train.py`` so that both the standard RL recipe
(``rl.train``) and OPSD recipe (``opsd.train``) can reuse the same patches.

Patches applied:
1. ``wandb.Config.update`` — allow value changes (duplicate-key workaround).
2. ``optim_step`` — add gradient clipping via ``AdamParams.grad_clip_norm``.
3. ``do_group_rollout_and_filter_constant_reward`` — retry on transient errors.
4. ``remove_constant_reward_groups`` — filter out ``None`` trajectory groups.
5. ``do_train_step_and_get_sampling_client`` — skip empty batches.
6. ``do_group_rollout`` — close sandboxes on failure + prevent wrapper chaining.
"""

from __future__ import annotations

import asyncio
import importlib
import inspect
import logging
import random
import time
from typing import Any

_LOGGER = logging.getLogger(__name__)


def _log_rollout_complete(task_label: str, result: Any, num_envs: int, elapsed: float) -> None:
    """Log a one-line summary after a successful group rollout."""
    rewards = result.get_total_rewards()
    turns = [len(t.transitions) for t in result.trajectories_G]

    mean_reward = sum(rewards) / len(rewards) if rewards else 0.0
    mean_turns = sum(turns) / len(turns) if turns else 0.0

    reward_parts = " ".join(f"{r:.2f}" for r in rewards)
    turns_parts = " ".join(str(t) for t in turns)

    _LOGGER.info(
        "Rollout done | task=%s | %.1fs | envs=%d | reward=%.3f [%s] | turns=%.1f [%s] | sandboxes: closed",
        task_label,
        elapsed,
        num_envs,
        mean_reward,
        reward_parts,
        mean_turns,
        turns_parts,
    )


class MonkeyPatchContext:
    """Context manager that applies and restores all monkey-patches.

    Usage::

        with MonkeyPatchContext(
            grad_clip_norm=0.5,
            rollout_max_retries=5,
        ):
            await tinker_train.main(cfg)
    """

    def __init__(
        self,
        *,
        grad_clip_norm: float = 0.5,
        rollout_max_retries: int = 5,
    ) -> None:
        self._grad_clip_norm = grad_clip_norm
        self._rollout_max_retries = rollout_max_retries

        # Saved originals (populated on __enter__).
        self._originals: dict[str, Any] = {}
        self._tinker_train: Any = None
        self._wbcfg: Any = None

    def __enter__(self) -> MonkeyPatchContext:
        self.apply()
        return self

    def __exit__(self, *args: Any) -> None:
        self.restore()

    def apply(self) -> None:
        tinker_mod = importlib.import_module("tinker")
        tinker_train = importlib.import_module("tinker_cookbook.rl.train")
        self._tinker_train = tinker_train

        import wandb.sdk.wandb_config as _wbcfg

        self._wbcfg = _wbcfg

        # Save originals.
        self._originals = {
            "optim_step": tinker_train.optim_step,
            "do_group_rollout_and_filter": tinker_train.do_group_rollout_and_filter_constant_reward,
            "remove_constant_reward_groups": tinker_train.remove_constant_reward_groups,
            "do_train_step": tinker_train.do_train_step_and_get_sampling_client,
            "do_group_rollout": tinker_train.do_group_rollout,
            "config_update": _wbcfg.Config.update,
        }

        # 1. wandb config.update allow value changes.
        orig_config_update = self._originals["config_update"]

        def _config_update_allow(self_wb: Any, d: Any, allow_val_change: Any = True) -> None:
            return orig_config_update(self_wb, d, allow_val_change=allow_val_change)

        _wbcfg.Config.update = _config_update_allow  # type: ignore[assignment]

        # 2. optim_step with grad clipping.
        grad_clip_norm = self._grad_clip_norm

        async def _optim_step_with_grad_clip(training_client: Any, learning_rate: float) -> None:
            adam_params_cls = tinker_mod.AdamParams
            adam_params = adam_params_cls(
                learning_rate=learning_rate,
                beta1=0.9,
                beta2=0.95,
                eps=1e-8,
                grad_clip_norm=grad_clip_norm,
            )
            optim_step_future = await training_client.optim_step_async(adam_params)
            await optim_step_future.result_async()

        tinker_train.optim_step = _optim_step_with_grad_clip  # type: ignore[assignment]

        # 3. Error-resilient rollouts with retry.
        orig_do_group_rollout_and_filter = self._originals["do_group_rollout_and_filter"]
        rollout_max_retries = self._rollout_max_retries

        async def _safe_do_group_rollout_and_filter(*args: Any, **kwargs: Any) -> Any:
            for attempt in range(1, rollout_max_retries + 1):
                try:
                    return await orig_do_group_rollout_and_filter(*args, **kwargs)
                except Exception as e:
                    if attempt < rollout_max_retries:
                        delay = min(10.0 * (2 ** (attempt - 1)), 120.0)
                        delay *= 0.5 + random.random()  # jitter
                        _LOGGER.warning(
                            "Group rollout failed (attempt %d/%d), retrying in %.0fs: %s: %s",
                            attempt,
                            rollout_max_retries,
                            delay,
                            type(e).__name__,
                            e,
                        )
                        await asyncio.sleep(delay)
                    else:
                        _LOGGER.error(
                            "Group rollout failed after %d attempts (training continues): %s: %s",
                            rollout_max_retries,
                            type(e).__name__,
                            e,
                        )
                        return None

        tinker_train.do_group_rollout_and_filter_constant_reward = _safe_do_group_rollout_and_filter  # type: ignore[assignment]

        # 4. Filter None from remove_constant_reward_groups.
        orig_remove_constant = self._originals["remove_constant_reward_groups"]

        def _safe_remove_constant_reward_groups(trajectory_groups: list[Any]) -> list[Any]:
            filtered = [g for g in trajectory_groups if g is not None]
            if not filtered:
                _LOGGER.warning("All rollouts in batch returned None — nothing to filter")
                return filtered
            return orig_remove_constant(filtered)

        tinker_train.remove_constant_reward_groups = _safe_remove_constant_reward_groups  # type: ignore[assignment]

        # 5. Safe train step that handles None trajectory groups.
        orig_do_train_step = self._originals["do_train_step"]
        do_train_step_sig = inspect.signature(orig_do_train_step)

        required_params = {"training_client", "env_group_builders_P", "trajectory_groups_P"}
        actual_params = set(do_train_step_sig.parameters)
        if not required_params.issubset(actual_params):
            missing = required_params - actual_params
            raise RuntimeError(
                f"do_train_step_and_get_sampling_client signature changed: missing {missing}. "
                f"Actual parameters: {list(do_train_step_sig.parameters)}. "
                f"The monkey-patch needs to be updated."
            )

        async def _safe_do_train_step(*args: Any, **kwargs: Any) -> Any:
            bound = do_train_step_sig.bind(*args, **kwargs)
            bound.apply_defaults()

            trajectory_groups: list[Any] = bound.arguments["trajectory_groups_P"]

            if any(g is None for g in trajectory_groups):
                original_count = len(trajectory_groups)
                env_builders: list[Any] = bound.arguments["env_group_builders_P"]
                pairs = [(b, g) for b, g in zip(env_builders, trajectory_groups, strict=False) if g is not None]
                bound.arguments["env_group_builders_P"] = [p[0] for p in pairs]
                bound.arguments["trajectory_groups_P"] = [p[1] for p in pairs]
                trajectory_groups = bound.arguments["trajectory_groups_P"]
                _LOGGER.warning(
                    "TRAIN STEP | filtered %d None groups (%d -> %d valid)",
                    original_count - len(trajectory_groups),
                    original_count,
                    len(trajectory_groups),
                )

            if not trajectory_groups:
                _LOGGER.warning("TRAIN STEP | all rollouts failed — skipping (no weight update)")
                training_client = bound.arguments["training_client"]
                sampling_client = await training_client.save_weights_and_get_sampling_client_async()
                return sampling_client, {}

            _LOGGER.info("TRAIN STEP | training on %d groups", len(trajectory_groups))
            return await orig_do_train_step(*bound.args, **bound.kwargs)

        tinker_train.do_train_step_and_get_sampling_client = _safe_do_train_step  # type: ignore[assignment]

        # 6. Sandbox cleanup on rollout failure + prevent wrapper chaining.
        orig_do_group_rollout = self._originals["do_group_rollout"]
        original_make_envs_attr = "_ares_original_make_envs"

        async def _do_group_rollout_with_cleanup(env_group_builder: Any, policy: Any) -> Any:
            original_make_envs = getattr(env_group_builder, original_make_envs_attr, env_group_builder.make_envs)
            setattr(env_group_builder, original_make_envs_attr, original_make_envs)

            created_envs: list[Any] = []

            async def _tracking_make_envs() -> Any:
                envs = await original_make_envs()
                created_envs.extend(envs)
                return envs

            env_group_builder.make_envs = _tracking_make_envs
            tags = env_group_builder.logging_tags()
            task_label = tags[-1] if tags else "unknown"
            t_start = time.monotonic()

            try:
                result = await orig_do_group_rollout(env_group_builder, policy)
                elapsed = time.monotonic() - t_start
                _log_rollout_complete(task_label, result, len(created_envs), elapsed)
                return result
            except BaseException:
                elapsed = time.monotonic() - t_start
                num_envs = len(created_envs)
                _LOGGER.warning(
                    "Rollout FAILED | task=%s | %.1fs | envs=%d | sandboxes: closing...",
                    task_label,
                    elapsed,
                    num_envs,
                )
                t_close = time.monotonic()
                for env in created_envs:
                    try:
                        await env.close()
                    except Exception as close_err:
                        _LOGGER.debug("Failed to close sandbox during cleanup: %s", close_err)
                close_elapsed = time.monotonic() - t_close
                _LOGGER.warning(
                    "Rollout FAILED | task=%s | envs=%d | sandboxes: closed! (%.1fs cleanup)",
                    task_label,
                    num_envs,
                    close_elapsed,
                )
                raise
            finally:
                env_group_builder.make_envs = original_make_envs

        tinker_train.do_group_rollout = _do_group_rollout_with_cleanup  # type: ignore[assignment]

        # Suppress noisy Harbor/Daytona debug logging.
        for noisy_logger in ("harbor", "daytona_sdk", "daytona", "httpx"):
            logging.getLogger(noisy_logger).setLevel(logging.WARNING)

    def restore(self) -> None:
        if self._tinker_train is None:
            return
        tt = self._tinker_train
        tt.optim_step = self._originals["optim_step"]  # type: ignore[assignment]
        tt.do_group_rollout_and_filter_constant_reward = self._originals["do_group_rollout_and_filter"]  # type: ignore[assignment]
        tt.remove_constant_reward_groups = self._originals["remove_constant_reward_groups"]  # type: ignore[assignment]
        tt.do_train_step_and_get_sampling_client = self._originals["do_train_step"]  # type: ignore[assignment]
        tt.do_group_rollout = self._originals["do_group_rollout"]  # type: ignore[assignment]
        if self._wbcfg is not None:
            self._wbcfg.Config.update = self._originals["config_update"]  # type: ignore[assignment]
        self._originals.clear()
        self._tinker_train = None
        self._wbcfg = None
