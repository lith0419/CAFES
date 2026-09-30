from __future__ import annotations

import copy
from typing import Any, Dict

from ..result_artifacts import refresh_periodic_module_artifacts
from ..runtime_context import get_runtime_context
from ..state import append_log
from ...workflow_modules.runtime import ModuleInvocation, record_module_observation
from .solver import _periodic_band_structure


def run_periodic_band_analysis(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
) -> Dict[str, Any]:
    """Evaluate the requested band path after the periodic SCF module finishes."""

    state = copy.deepcopy(state)
    results = state.get('_module_results')
    context = get_runtime_context(state.get('module_context_ref'))
    if not isinstance(results, dict) or not isinstance(context, dict):
        observation = record_module_observation(
            state,
            invocation,
            'skipped',
            reason='process_local_solver_context_unavailable',
        )
        append_log(state, 'info', 'workflow.numerical_module_skipped', observation)
        return state

    updated_results = copy.deepcopy(results)
    mean_field = context.get('mean_field')
    if not bool(getattr(mean_field, 'converged', False)):
        updated_results['band_structure_status'] = 'skipped_unconverged_scf'
        provided_paths = ('band_structure_status',)
    else:
        band_structure = _periodic_band_structure(
            mean_field,
            context['cell'],
            restricted=bool(context.get('restricted')),
            mode=str(context.get('band_path_mode') or 'auto'),
            requested_path=context.get('band_path'),
            requested_special_points=context.get('band_path_special_points') or {},
            npoints=int(context.get('band_path_npoints') or 80),
            path_definition=context.get('path_definition'),
            fermi_energy=context.get('fermi_energy'),
            fermi_by_spin=context.get('fermi_energy_by_spin') or {},
        )
        updated_results['periodic_band_structure'] = band_structure
        updated_results['band_path_mode'] = band_structure['path_mode']
        updated_results['band_path'] = band_structure['path']
        updated_results['band_path_point_count'] = band_structure['point_count']
        updated_results['band_path_labels'] = band_structure['special_point_labels']
        updated_results['band_structure_status'] = 'completed'
        provided_paths = (
            'periodic_band_structure',
            'band_path_mode',
            'band_path',
            'band_path_point_count',
            'band_path_labels',
            'band_structure_status',
        )

    state['_module_results'] = refresh_periodic_module_artifacts(state, updated_results)
    observation = record_module_observation(
        state,
        invocation,
        'completed',
        provided_result_paths=provided_paths,
    )
    append_log(state, 'info', 'workflow.numerical_module_completed', observation)
    return state


__all__ = ['run_periodic_band_analysis']
