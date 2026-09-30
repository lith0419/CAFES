from __future__ import annotations

from typing import Any, Callable, Dict, List


def build_core_families(
    item_factory: Callable[..., Any],
    *,
    capability_design_only: str,
    capability_planned: str,
) -> Dict[str, List[Any]]:
    _capability = item_factory
    CAPABILITY_DESIGN_ONLY = capability_design_only
    CAPABILITY_PLANNED = capability_planned

    return {
        'task_types': [
            _capability('molecular', 'Molecular electronic structure', 'task_type'),
            _capability('model_hamiltonian', 'Model Hamiltonian', 'task_type'),
            _capability(
                'periodic',
                'Periodic electronic structure',
                'task_type',
                planner_allowed=False,
            ),
        ],
    }


__all__ = ['build_core_families']
