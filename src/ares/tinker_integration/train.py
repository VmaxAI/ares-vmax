"""Backward-compatibility shim — delegates to ``rl.train``.

.. deprecated::
    Import from ``ares.tinker_integration.rl.train`` instead.
"""

from ares.tinker_integration.rl.train import run_training

__all__ = ["run_training"]
