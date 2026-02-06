# ARES + Tinker Training Script Audit Report

**File:** `examples/05_tinker_train.py`
**Date:** 2025-02-06
**Status:** Issues identified and fixed

---

## Executive Summary

The ARES + Tinker training script (`05_tinker_train.py`) had several critical configuration gaps preventing effective RL training on SWE-bench tasks. The most impactful issues were:

1. **Wrong loss function** - defaulting to vanilla importance sampling instead of PPO
2. **Zero-advantage groups not filtered** - wasting ~66% of gradient updates on zero-signal groups
3. **Hardcoded learning rate** - ignoring Tinker's model-specific LR recommendations
4. **Evaluation disabled** - no way to track training progress

All issues have been addressed in the updated script.

---

## Findings Detail

### CRITICAL-1: Loss Function Defaults to REINFORCE (Importance Sampling)

**Severity:** Critical
**Impact:** Extreme gradient variance, unstable training

**Problem:**
The `loss_fn` parameter was not passed to `tinker_train.Config`, so it defaulted to `"importance_sampling"` (line 241 of `tinker_cookbook/rl/train.py`):

```python
# tinker_cookbook/rl/train.py:241
loss_fn: LossFnType = "importance_sampling"
```

With `max_steps_off_policy=10` (async training), the script was using **vanilla REINFORCE with importance weights on off-policy data**. This has extreme variance because the importance ratio `pi(a|s) / pi_old(a|s)` can explode without clipping.

**Fix:** Default to `"ppo"` which clips the importance ratio, dramatically reducing variance.

**Available options** (from `tinker.types.LossFnType`):
- `"importance_sampling"` - Vanilla REINFORCE with importance weights (high variance)
- `"ppo"` - Proximal Policy Optimization with clipped ratio (recommended)
- `"cispo"` - CISPO variant
- `"dro"` - Distributionally Robust Optimization
- `"cross_entropy"` - Standard cross-entropy (not for RL)

---

### CRITICAL-2: Zero-Advantage Groups Not Filtered

**Severity:** Critical
**Impact:** ~66% of training batches produce zero gradient

**Problem:**
`remove_constant_reward_groups` was not passed to `tinker_train.Config`, defaulting to `False` (line 254 of `tinker_cookbook/rl/train.py`).

**Why this matters for SWE-bench:**

With a ~10% solve rate and `group_size=4`, the probability that ALL rollouts in a group have the same reward (all 0 or all 1) is:

```
P(all_zero) = 0.9^4 = 0.6561  (65.6%)
P(all_one)  = 0.1^4 = 0.0001  (0.01%)
P(constant) = 0.6562           (65.6%)
```

When all rewards in a group are identical, `compute_advantages` (GRPO-style mean-centering) produces zero advantages for every trajectory:

```python
# tinker_cookbook/rl/data_processing.py:20-30
def compute_advantages(trajectory_groups_P):
    for traj_group in trajectory_groups_P:
        rewards_G = torch.tensor(traj_group.get_total_rewards())
        advantages_G = rewards_G - rewards_G.mean()  # All zeros if rewards are constant!
```

This means ~66% of groups contribute **zero gradient signal**, wasting compute.

**Fix:** Set `remove_constant_reward_groups=True`. The cookbook's `remove_constant_reward_groups()` function (data_processing.py:198-209) filters these out before training, ensuring every group contributes meaningful gradients.

**Note:** The cookbook has a safety fallback - if ALL groups are filtered, it returns one group to prevent empty batches:

```python
if not new_groups:
    logger.warning("All rewards are uniform. There will be no gradient")
    return trajectory_groups_P[0:1]
```

---

### CRITICAL-3: Hardcoded Learning Rate Not Model-Specific

**Severity:** Critical
**Impact:** Suboptimal or destabilizing learning rate

**Problem:**
Learning rate was hardcoded to `1e-6` regardless of model. Tinker provides `get_lr()` in `hyperparam_utils.py` which computes model-specific LRs based on hidden dimension scaling:

```python
# tinker_cookbook/hyperparam_utils.py:147-159
def get_lr(model_name: str, is_lora: bool = True) -> float:
    base_lr = 5e-05
    lora_multiplier = 10.0
    lr = base_lr * lora_multiplier if is_lora else base_lr
    if "llama" in model_name.lower():
        exponent_model = 0.781
    elif "qwen" in model_name.lower():
        exponent_model = 0.0775
    else:
        assert False, f"Unknown model: {model_name}"
    lr = lr * (2000 / hidden_size) ** exponent_model
    return lr
```

**Model-Specific Recommendations:**

| Model | Hidden Size | `get_lr()` Result | Previous (hardcoded) |
|-------|------------|-------------------|---------------------|
| Qwen/Qwen3-4B-Instruct-2507 | 2560 | ~4.7e-4 | 1e-6 (470x too low) |
| meta-llama/Llama-3.1-8B-Instruct | 4096 | ~2.8e-4 | 1e-6 (280x too low) |

The hardcoded `1e-6` was **orders of magnitude too low**, causing extremely slow learning.

**Fix:** Default `learning_rate` to `None`, which triggers auto-detection via `get_lr(model_name)`. Users can still override with an explicit value.

---

### CRITICAL-4: No Gradient Clipping (Cookbook Limitation)

**Severity:** Critical (but not fixable in our script)
**Impact:** Potential gradient explosions

**Problem:**
The cookbook hardcodes Adam parameters without gradient clipping:

```python
# tinker_cookbook/rl/train.py:143
adam_params = tinker.AdamParams(learning_rate=learning_rate, beta1=0.9, beta2=0.95, eps=1e-8)
```

There is no `grad_clip_norm` parameter available.

**Mitigation:** This is a cookbook limitation. The combination of PPO's clipped ratio + proper learning rate from `get_lr()` provides implicit gradient stabilization. Documented for awareness.

---

### HIGH-5: Evaluation Disabled

**Severity:** High
**Impact:** No training progress visibility

**Problem:**
`eval_every` was set to `0` (disabling evaluation), and no evaluators were configured. The cookbook's `main()` function (line 1084-1087) automatically creates an `RLTestSetEvaluator` from the test dataset, but with `eval_every=0` it never runs.

**Fix:** Changed `eval_every` default from `0` to `20` (matching cookbook default on line 255). The test dataset is already built by `TinkerDatasetBuilder.__call__()` (line 347-348) and automatically gets an evaluator in the cookbook's `main()`.

---

### HIGH-6: Missing `logging_tags()` on EnvGroupBuilder

**Severity:** High
**Impact:** Training metrics not properly segmented

**Problem:**
`TinkerEnvGroupBuilder` inherited the default empty `logging_tags()` from the base class, so `compute_trajectory_metrics()` couldn't segment rewards by environment.

The cookbook uses `logging_tags()` in `prepare_minibatch()` (train.py:720):

```python
taglist_P = [env_group_builder.logging_tags() for env_group_builder in env_group_builders_P]
metrics.update(compute_trajectory_metrics(trajectory_groups_P, taglist_P))
```

**Fix:** Override `logging_tags()` to return `[self.env_preset_name, f"task_{self.env_preset_idx}"]`.

---

### HIGH-7: Temperature Not Passed Through

**Severity:** High
**Impact:** Missing configurability

**Problem:**
The `temperature` parameter was not exposed in `CLIConfig` or passed to `tinker_train.Config`. The cookbook defaults to `1.0` (line 232 of train.py), which is generally fine, but users couldn't tune it.

**Fix:** Added `temperature` to `CLIConfig` (default `1.0`) and wire it to `tinker_train.Config`.

---

### HIGH-8: Middle Truncation Loses System Prompt

**Severity:** High
**Impact:** Model loses task context on long conversations

**Problem:**
The `_middle_truncate` function removed tokens symmetrically around the center:

```python
center_idx = model_input.length // 2
truncate_start_idx = center_idx - num_tokens_to_truncate // 2
truncate_end_idx = center_idx + num_tokens_to_truncate // 2
```

This could truncate the beginning (system prompt + task description) which is critical for the agent to understand what to do.

**Fix:** Replaced with "preserve ends" strategy:
- Keep first `buffer_tokens` tokens (system prompt + task context)
- Keep last `remaining` tokens (recent conversation history)
- Drop tokens in the middle

This is a common pattern in long-context LLM applications and preserves the most important information at both ends.

---

### MEDIUM-9: No `stream_minibatch_config`

**Severity:** Medium
**Impact:** Suboptimal throughput for large batches

`StreamMinibatchConfig` allows overlapping sampling and training by processing minibatches as they arrive. Currently not exposed. Documented as an advanced option for future optimization.

---

### MEDIUM-10: Container Factory Default

**Severity:** Medium
**Impact:** User confusion

The script defaults to `DaytonaContainer` in `env_make_kwargs`, but the ARES framework defaults to `DockerContainer`. This is intentional for cloud training but should be documented clearly.

---

### MEDIUM-11: Sparse StepResult Metrics

**Severity:** Medium
**Impact:** Limited per-step observability

**Fix:** Added `episode_length` to StepResult metrics alongside existing `reward` and `step_count`.

---

## Cookbook Architecture Notes

### Advantage Computation (GRPO-style)
The cookbook uses simple mean-centering within groups (`data_processing.py:20-30`), not per-step GAE. This is the GRPO approach where the advantage of each trajectory is its total reward minus the group mean.

### Adam Parameters
Hardcoded in `train.py:143`: `beta1=0.9, beta2=0.95, eps=1e-8`. No gradient clipping. This is not configurable from outside the cookbook.

### Async Training
With `AsyncConfig`, rollout workers sample asynchronously while training proceeds. Stale samples (beyond `max_steps_off_policy` steps old) are requeued. This makes PPO's clipping especially important since data can be significantly off-policy.

### Test Set Evaluation
The cookbook automatically creates an `RLTestSetEvaluator` from the test dataset returned by `dataset_builder()` (train.py:1086-1087). This evaluator runs with `group_size=1` and `eval_every` controls frequency.

---

## Recommended Configurations

### Quick Experiment (Qwen3-4B, 20 tasks)
```bash
uv run -m examples.05_tinker_train \
    model_name=Qwen/Qwen3-4B-Instruct-2507 \
    num_tasks=20 \
    group_size=4 \
    groups_per_batch=10 \
    loss_fn=ppo \
    remove_constant_reward_groups=True
    # learning_rate auto-detected via get_lr()
```

### Full Training (Llama-3.1-8B, all tasks)
```bash
uv run -m examples.05_tinker_train \
    model_name=meta-llama/Llama-3.1-8B-Instruct \
    num_tasks=None \
    group_size=4 \
    groups_per_batch=20 \
    lora_rank=64 \
    loss_fn=ppo \
    remove_constant_reward_groups=True \
    wandb_project=ares-rl \
    eval_every=20 \
    save_every=20
```
