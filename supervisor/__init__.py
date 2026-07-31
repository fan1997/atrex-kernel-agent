"""Independent campaign supervisor for the orchestrated optimization route.

The normal entry point remains ``orchestrator/optimize.py``.  Supervised runs use
``python -m supervisor.optimize`` and install a process-local session adapter without
modifying the original orchestrator module on disk.
"""

from .bridge import SupervisorConfig, SupervisorRuntime

__all__ = ["SupervisorConfig", "SupervisorRuntime"]
