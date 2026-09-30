from __future__ import annotations

import copy
from typing import Any, Dict

from ..result_artifacts import refresh_model_module_artifacts
from ..runtime_context import get_runtime_context
from ..state import append_log
from ...workflow_modules.runtime import ModuleInvocation, record_module_observation
from .observables import _compute_strong_correlation_diagnostics


def run_model_correlation_diagnostics(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
) -> Dict[str, Any]:
    """Compute model diagnostics from the solver payload retained by the worker."""

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

    model_spec = context.get('model_spec')
    solver_results = context.get('solver_results')
    if not isinstance(model_spec, dict) or not isinstance(solver_results, dict):
        raise ValueError('Model diagnostics require normalized model_spec and solver_results context.')

    updated_results = copy.deepcopy(results)
    diagnostics = _compute_strong_correlation_diagnostics(model_spec, solver_results)
    updated_results['strong_correlation_diagnostics'] = diagnostics
    analysis_text = str(updated_results.get('analysis_text') or '')
    summary = 'strong_correlation={0}; {1}'.format(
        diagnostics['level'],
        diagnostics['summary'],
    )
    updated_results['analysis_text'] = '; '.join(item for item in (analysis_text, summary) if item)
    state['_module_results'] = refresh_model_module_artifacts(state, updated_results)

    observation = record_module_observation(
        state,
        invocation,
        'completed',
        provided_result_paths=('strong_correlation_diagnostics',),
    )
    append_log(state, 'info', 'workflow.numerical_module_completed', observation)
    return state


__all__ = ['run_model_correlation_diagnostics']
