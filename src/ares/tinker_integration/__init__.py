"""ARES + Tinker RL integration.

This module provides two harness modes for training code agents with Tinker's RL
infrastructure:

**Terminal harness** (``harness="terminal"``, default):
    Direct tmux terminal control via JSON commands. The model gets raw terminal
    access, producing a clean RL learning signal.

**Code-agent harness** (``harness="code-agent"``):
    ARES CodeEnvironment with any agent harness (Mini-SWE-Agent, Terminus2, etc.).
    LLM calls are intercepted via QueueMediatedLLMClient and exposed as RL observations.

Both share the same training infrastructure (monkey-patches, config, dataset batching).

Key components:
- AsyncTerminalGymEnv: Gym-like wrapper over Harbor environments with tmux terminal control
- HarborTerminalTinkerEnv: Tinker Env adapter with JSON command parsing (terminal harness)
- AresCodeTinkerEnv: Tinker Env adapter wrapping ARES CodeEnvironment (code-agent harness)
- TerminalRLDataset / TerminalRLDatasetBuilder: Multi-task dataset layer (terminal harness)
- AresEnvGroupBuilder / AresRLDatasetBuilder: Multi-task dataset layer (code-agent harness)
- run_training: Training entry point (auto-selects harness based on config)
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
from ares.tinker_integration.terminal_env import AsyncTerminalGymEnv
from ares.tinker_integration.tinker_env import HarborTerminalTinkerEnv
from ares.tinker_integration.train import run_training

__all__ = [
    "AresCodeTinkerEnv",
    "AresEnvGroupBuilder",
    "AresRLDatasetBuilder",
    "AsyncTerminalGymEnv",
    "HarborTerminalTinkerEnv",
    "TerminalEnvGroupBuilder",
    "TerminalRLDataset",
    "TerminalRLDatasetBuilder",
    "TrainingConfig",
    "create_snapshots",
    "load_tasks_from_preset",
    "load_tasks_from_task_dir",
    "run_training",
]
