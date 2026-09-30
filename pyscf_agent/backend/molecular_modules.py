from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Tuple

from .correlation.strong import (
    build_correlation_diagnostics,
    build_orbital_processing_summary,
    build_scf_stability_summary,
    propose_active_space,
)
from ..contracts import task_spec_from_dict
from .result_artifacts import refresh_molecular_module_artifacts
from .runtime_context import get_runtime_context
from .state import append_log
from ..workflow_modules.runtime import ModuleInvocation, record_module_observation


def _module_inputs(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
) -> Tuple[Dict[str, Any], Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
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
        return state, None, None
    return state, copy.deepcopy(results), context


def _complete_module(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
    results: Dict[str, Any],
    provided_paths: Tuple[str, ...],
) -> Dict[str, Any]:
    state['_module_results'] = copy.deepcopy(results)
    refresh_molecular_module_artifacts(state, task_spec_from_dict(state['task_spec']), results)
    observation = record_module_observation(
        state,
        invocation,
        'completed',
        provided_result_paths=provided_paths,
    )
    append_log(state, 'info', 'workflow.numerical_module_completed', observation)
    return state


def run_molecular_correlation_diagnostics(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
) -> Dict[str, Any]:
    if not invocation.configuration.get('enabled', True):
        state = copy.deepcopy(state)
        record_module_observation(state, invocation, 'skipped', reason='disabled_by_request')
        return state
    state, results, context = _module_inputs(state, invocation)
    if results is None or context is None:
        return state
    mf = context['mf']
    scf_stability = context.get('scf_stability')
    if not isinstance(scf_stability, dict):
        scf_stability = build_scf_stability_summary(mf, task_spec=task_spec_from_dict(state['task_spec']))
        context['scf_stability'] = scf_stability
    results['scf_stability'] = scf_stability
    results['correlation_diagnostics'] = build_correlation_diagnostics(
        mf,
        context['mol'],
        post_hf=context.get('post_hf'),
        scf_stability=scf_stability,
        reference_energy=context.get('reference_energy'),
        correlation_energy=context.get('correlation_energy'),
        state_energies=context.get('state_energies'),
        state_sector=context.get('state_sector'),
        correlated_natural_occupations=context.get('correlated_natural_occupations'),
        current_method=context.get('current_method'),
    )
    return _complete_module(
        state,
        invocation,
        results,
        ('scf_stability', 'correlation_diagnostics'),
    )


def run_molecular_orbital_processing(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
) -> Dict[str, Any]:
    state, results, context = _module_inputs(state, invocation)
    if results is None or context is None:
        return state
    task_spec = task_spec_from_dict(state['task_spec'])
    results['orbital_processing'] = build_orbital_processing_summary(
        context['mf'],
        context['mol'],
        task_spec.orbital_processing,
    )
    return _complete_module(state, invocation, results, ('orbital_processing',))


def _limit_active_space_candidates(active_space: Dict[str, Any], limit: int) -> None:
    audit = active_space.get('audit')
    if not isinstance(audit, dict):
        return
    candidates = audit.get('candidate_active_spaces')
    if not isinstance(candidates, list) or len(candidates) <= limit:
        return
    selected_method = str(audit.get('selected_candidate_method') or '')
    selected = [item for item in candidates if isinstance(item, Mapping) and item.get('method') == selected_method]
    retained = selected[:1]
    for candidate in candidates:
        if candidate in retained:
            continue
        retained.append(candidate)
        if len(retained) >= limit:
            break
    audit['candidate_active_spaces'] = retained
    audit['candidate_limit'] = limit
    summary = active_space.get('audit_summary')
    if isinstance(summary, dict):
        summary['candidate_methods'] = [
            item.get('method') for item in retained if isinstance(item, Mapping)
        ]


def run_molecular_active_space_audit(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
) -> Dict[str, Any]:
    state, results, context = _module_inputs(state, invocation)
    if results is None or context is None:
        return state
    task_spec = task_spec_from_dict(state['task_spec'])
    cached_probe_space = context.pop('active_space_probe_final_active_space', None)
    active_space = (
        copy.deepcopy(cached_probe_space)
        if isinstance(cached_probe_space, dict)
        else propose_active_space(
            context['mf'],
            task_spec.active_space,
            post_hf=context.get('post_hf'),
            mol=context['mol'],
            localization_method=task_spec.orbital_processing.localization_method,
            localization_scope=task_spec.orbital_processing.localization_scope,
            orbital_ordering=task_spec.orbital_processing.orbital_ordering,
            orbital_order=task_spec.orbital_processing.orbital_order,
        )
    )
    cas_result = results.get('cas_result') if isinstance(results.get('cas_result'), dict) else {}
    if isinstance(active_space.get('audit'), dict) and cas_result:
        active_space['audit']['executed_orbital_processing'] = copy.deepcopy(
            cas_result.get('orbital_provenance') or {}
        )
        dmrg_result = cas_result.get('dmrg_result') if isinstance(cas_result.get('dmrg_result'), dict) else {}
        if dmrg_result.get('orbital_ordering'):
            active_space['audit']['orbital_ordering'] = copy.deepcopy(dmrg_result['orbital_ordering'])
    candidate_limit = int(invocation.configuration.get('candidate_limit') or 7)
    _limit_active_space_candidates(active_space, candidate_limit)
    audit = active_space.get('audit')
    if isinstance(audit, dict):
        audit['module_policy'] = {
            'candidate_limit': candidate_limit,
            'require_approval': bool(invocation.configuration.get('require_approval', True)),
        }
    results['active_space'] = active_space
    probe_decision = results.get('active_space_probe')
    if isinstance(probe_decision, dict):
        probe_decision['final_candidate_method'] = (
            active_space.get('audit', {}).get('selected_candidate_method')
            if isinstance(active_space.get('audit'), dict)
            else None
        )
        probe_decision['final_ncas'] = active_space.get('ncas')
        probe_decision['final_nelecas'] = copy.deepcopy(active_space.get('nelecas'))
    return _complete_module(state, invocation, results, ('active_space',))


__all__ = [
    'run_molecular_active_space_audit',
    'run_molecular_correlation_diagnostics',
    'run_molecular_orbital_processing',
]
