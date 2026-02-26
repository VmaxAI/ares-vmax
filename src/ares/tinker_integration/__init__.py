"""ARES + Tinker training integration.

Provides two training recipes and two harness modes:

**Recipes:**
- ``rl``: Standard RL training (GRPO-style, sync or async).
- ``opsd``: On-Policy Self-Distillation — iterative phasic training with
  self-reflection and reverse-KL distillation from a context-enriched teacher.

**Harness modes** (shared by both recipes):
- ``terminal``: Direct tmux terminal control via JSON commands.
- ``code-agent``: ARES CodeEnvironment with any agent harness (Mini-SWE-Agent, etc.).

Shared infrastructure: monkey-patches, config, dataset batching, env adapters.
"""

from ares.tinker_integration.ares_env import AresCodeTinkerEnv
from ares.tinker_integration.ares_env import AresEnvGroupBuilder
from ares.tinker_integration.ares_env import AresRLDatasetBuilder
from ares.tinker_integration.config import TrainingConfig
from ares.tinker_integration.create_snapshots import create_snapshots
from ares.tinker_integration.dataset import TerminalEnvGroupBuilder
from ares.tinker_integration.dataset import TerminalRLDataset
from ares.tinker_integration.dataset import TerminalRLDatasetBuilder
from ares.tinker_integration.dataset import load_tasks_from_preset
from ares.tinker_integration.dataset import load_tasks_from_task_dir
from ares.tinker_integration.monkey_patches import MonkeyPatchContext
from ares.tinker_integration.rl.train import run_training
from ares.tinker_integration.terminal_env import AsyncTerminalGymEnv
from ares.tinker_integration.tinker_env import HarborTerminalTinkerEnv

__all__ = [
    "AresCodeTinkerEnv",
    "AresEnvGroupBuilder",
    "AresRLDatasetBuilder",
    "AsyncTerminalGymEnv",
    "HarborTerminalTinkerEnv",
    "MonkeyPatchContext",
    "TerminalEnvGroupBuilder",
    "TerminalRLDataset",
    "TerminalRLDatasetBuilder",
    "TrainingConfig",
    "create_snapshots",
    "load_tasks_from_preset",
    "load_tasks_from_task_dir",
    "run_training",
]
