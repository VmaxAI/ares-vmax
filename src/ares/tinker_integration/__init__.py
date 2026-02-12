"""ARES + Tinker terminal-based RL integration.

This module provides a direct terminal control architecture for training code agents
with Tinker's RL infrastructure. Instead of routing model outputs through an intermediate
agent + QueueMediatedLLMClient, the model gets direct tmux terminal control via JSON
commands, producing a clean RL learning signal.

Key components:
- AsyncTerminalGymEnv: Gym-like wrapper over Harbor environments with tmux terminal control
- HarborTerminalTinkerEnv: Tinker Env adapter with JSON command parsing
- TerminalRLDataset / TerminalRLDatasetBuilder: Multi-task dataset layer
- run_training: Training entry point
"""

from ares.tinker_integration.config import TrainingConfig
from ares.tinker_integration.dataset import TerminalEnvGroupBuilder
from ares.tinker_integration.dataset import TerminalRLDataset
from ares.tinker_integration.dataset import TerminalRLDatasetBuilder
from ares.tinker_integration.dataset import load_tasks_from_preset
from ares.tinker_integration.dataset import load_tasks_from_task_dir
from ares.tinker_integration.terminal_env import AsyncTerminalGymEnv
from ares.tinker_integration.tinker_env import HarborTerminalTinkerEnv
from ares.tinker_integration.train import run_training

__all__ = [
    "AsyncTerminalGymEnv",
    "HarborTerminalTinkerEnv",
    "TerminalEnvGroupBuilder",
    "TerminalRLDataset",
    "TerminalRLDatasetBuilder",
    "TrainingConfig",
    "load_tasks_from_preset",
    "load_tasks_from_task_dir",
    "run_training",
]
