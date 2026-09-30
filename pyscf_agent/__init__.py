"""Core public API for PySCF Agent.

Specialized contracts live in their own namespaces, including
``pyscf_agent.executors`` and ``pyscf_agent.registry``. Internal workflow
steps are not part of the package-root API.
"""

from __future__ import annotations

from .application import CalculationApplicationService
from .contracts import TaskReport, TaskSpec
from .registry import PlatformRegistry, default_registry


__all__ = [
    'CalculationApplicationService',
    'PlatformRegistry',
    'TaskReport',
    'TaskSpec',
    'default_registry',
]
