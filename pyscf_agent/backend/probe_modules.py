from __future__ import annotations

import copy
from typing import Any, Dict, Mapping

from .active_space_probe import ACTIVE_SPACE_PROBE_SCHEMA
from .correlation.strong import (
    build_correlation_diagnostics,
    build_scf_stability_summary,
    propose_active_space,
)
from .result_artifacts import refresh_molecular_module_artifacts
from .runtime_context import get_runtime_context
from ..contracts import task_spec_from_dict
from .state import append_log
from ..workflow_modules.runtime import ModuleInvocation, record_module_observation


_REVIEWABLE_CANDIDATE_STATUSES = frozenset(('available',))
_REFINEMENT_RISK_FLAGS = frozenset((
    'active_electron_count_unavailable',
    'chemical_valence_manifold_incomplete',
    'orbital_mapping_ambiguous',
))
_DIAGNOSTIC_REFINEMENT_BOUNDARIES = ((0.30, 0.40), (0.62, 0.72))


def _probe_diagnostic_refinement_reasons(diagnostics: Any) -> list[str]:
    if not isinstance(diagnostics, Mapping):
        return []
    reasons = []
    scf_stability = diagnostics.get('scf_stability')
    if isinstance(scf_stability, Mapping) and scf_stability.get('stable') is False:
        reasons.append('scf_instability')

    frontier = diagnostics.get('frontier_orbital_degeneracy')
    try:
        frontier_score = float(frontier.get('score') or 0.0)
    except (AttributeError, TypeError, ValueError):
        frontier_score = 0.0
    if (
        isinstance(frontier, Mapping)
        and frontier.get('status') == 'available'
        and frontier_score >= 0.5
    ):
        reasons.append('frontier_orbital_degeneracy')

    risk = (
        diagnostics.get('molecular_correlation_risk')
        if isinstance(diagnostics.get('molecular_correlation_risk'), Mapping)
        else diagnostics
    )
    score = risk.get('overall_score', diagnostics.get('score'))
    try:
        score_value = float(score)
    except (TypeError, ValueError):
        score_value = None
    if score_value is not None and any(
        lower <= score_value <= upper
        for lower, upper in _DIAGNOSTIC_REFINEMENT_BOUNDARIES
    ):
        reasons.append('near_correlation_decision_boundary')
    return reasons


def active_space_probe_refinement_decision(
    active_space: Any,
    *,
    diagnostics: Any = None,
) -> Dict[str, Any]:
    """Decide whether an SCF/AVAS proposal needs perturbative evidence."""

    reasons = []
    if not isinstance(active_space, Mapping):
        reasons.append('active_space_audit_unavailable')
        return {
            'status': 'refinement_required',
            'refinement_required': True,
            'reasons': reasons,
        }
    audit = active_space.get('audit') if isinstance(active_space.get('audit'), Mapping) else {}
    summary = (
        active_space.get('audit_summary')
        if isinstance(active_space.get('audit_summary'), Mapping)
        else {}
    )
    selected_method = str(
        audit.get('selected_candidate_method')
        or summary.get('selected_candidate_method')
        or 'unresolved'
    )
    candidates = audit.get('candidate_active_spaces')
    selected = next((
        candidate
        for candidate in (candidates if isinstance(candidates, list) else [])
        if isinstance(candidate, Mapping) and str(candidate.get('method')) == selected_method
    ), {})
    candidate_status = str(selected.get('status') or '')
    consistency = (
        audit.get('ncas_nelecas_consistency')
        if isinstance(audit.get('ncas_nelecas_consistency'), Mapping)
        else {}
    )
    consistent = bool(
        consistency.get('consistent', summary.get('ncas_nelecas_consistent', False))
    )
    ncas = active_space.get('ncas')
    nelecas = active_space.get('nelecas')
    risk_flags = {
        str(flag)
        for flag in ((selected.get('evaluation') or {}).get('risk_flags') or [])
    }

    if selected_method == 'unresolved' or not selected:
        reasons.append('no_executable_candidate')
    if candidate_status not in _REVIEWABLE_CANDIDATE_STATUSES:
        reasons.append('candidate_status_{0}'.format(candidate_status or 'missing'))
    if ncas is None or nelecas is None:
        reasons.append('active_space_size_or_electron_count_unavailable')
    if not consistent:
        reasons.append('ncas_nelecas_inconsistent')
    reasons.extend(sorted(risk_flags.intersection(_REFINEMENT_RISK_FLAGS)))
    reasons.extend(_probe_diagnostic_refinement_reasons(diagnostics))
    reasons = list(dict.fromkeys(reasons))
    return {
        'status': 'refinement_required' if reasons else 'review_ready',
        'refinement_required': bool(reasons),
        'reasons': reasons,
        'initial_candidate_method': selected_method,
        'initial_candidate_status': candidate_status or None,
        'initial_ncas': ncas,
        'initial_nelecas': copy.deepcopy(nelecas),
        'initial_selection_confidence': summary.get('selection_confidence'),
    }


def _run_mp2_probe(mf: Any, *, restricted: bool) -> Any:
    from pyscf import mp  # pylint: disable=import-outside-toplevel

    post_hf = mp.MP2(mf) if restricted else mp.UMP2(mf)
    post_hf.kernel()
    return post_hf


def prepare_molecular_active_space_probe(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
) -> Dict[str, Any]:
    """Record a molecular probe contract without activating the target solver."""
    state = copy.deepcopy(state)
    task_spec = task_spec_from_dict(state.get('task_spec') or {})
    target_method = task_spec.active_space.target_method
    if task_spec.task_type != 'molecular' or task_spec.method.name not in ('hf', 'mp2', 'fci'):
        raise ValueError('The active-space probe module requires a molecular HF, MP2, or FCI task.')
    if target_method not in ('casci', 'casscf'):
        raise ValueError('The active-space probe module requires a CASCI or CASSCF target method.')

    probe_contract = {
        'schema': ACTIVE_SPACE_PROBE_SCHEMA,
        'requested_strategy': str(
            invocation.configuration.get('requested_strategy') or task_spec.method.name
        ),
        'probe_method': task_spec.method.name,
        'refinement_method': invocation.configuration.get('refinement_method'),
        'target_method': target_method,
        'target_solver': task_spec.active_space.target_solver,
        'target_solver_options': copy.deepcopy(
            task_spec.active_space.target_solver_options or {}
        ),
        'selection_method': task_spec.active_space.selection_method,
        'occupation_window': list(task_spec.active_space.occupation_window),
    }
    state['calculation_role'] = 'active_space_probe'
    state['active_space_probe'] = probe_contract
    observation = record_module_observation(
        state,
        invocation,
        'completed',
        role='active_space_probe',
        probe_contract=probe_contract,
    )
    append_log(state, 'info', 'workflow.active_space_probe_prepared', observation)
    return state


def refine_molecular_active_space_probe(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
) -> Dict[str, Any]:
    """Run MP2 when the SCF/AVAS proposal or its diagnostics need refinement."""

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
        append_log(state, 'info', 'workflow.active_space_probe_refinement_skipped', observation)
        return state

    task_spec = task_spec_from_dict(state.get('task_spec') or {})
    refinement_method = str(
        invocation.configuration.get('refinement_method') or ''
    ).strip().lower()
    preliminary = propose_active_space(
        context['mf'],
        task_spec.active_space,
        post_hf=None,
        mol=context['mol'],
        localization_method=task_spec.orbital_processing.localization_method,
        localization_scope=task_spec.orbital_processing.localization_scope,
        orbital_ordering=task_spec.orbital_processing.orbital_ordering,
        orbital_order=task_spec.orbital_processing.orbital_order,
    )
    scf_stability = context.get('scf_stability')
    if not isinstance(scf_stability, dict):
        scf_stability = build_scf_stability_summary(context['mf'], task_spec=task_spec)
        context['scf_stability'] = scf_stability
    preliminary_diagnostics = build_correlation_diagnostics(
        context['mf'],
        context['mol'],
        scf_stability=scf_stability,
        reference_energy=context.get('reference_energy'),
        correlation_energy=context.get('correlation_energy'),
        current_method=context.get('current_method'),
    )
    decision = active_space_probe_refinement_decision(
        preliminary,
        diagnostics=preliminary_diagnostics,
    )
    decision.update({
        'schema': 'pyscf-agent.active-space-probe-refinement.v1',
        'initial_method': task_spec.method.name,
        'refinement_method': refinement_method or None,
        'executed_methods': [task_spec.method.name],
    })

    if not decision['refinement_required']:
        context['active_space_probe_final_active_space'] = preliminary
    elif refinement_method == 'mp2':
        try:
            post_hf = _run_mp2_probe(
                context['mf'],
                restricted=bool(task_spec.method.restricted),
            )
            context['post_hf'] = post_hf
            context['correlation_energy'] = float(post_hf.e_corr)
            decision.update({
                'status': 'refined_with_mp2',
                'refinement_completed': True,
                'executed_methods': [task_spec.method.name, 'mp2'],
                'refinement_energy': float(post_hf.e_tot),
                'refinement_correlation_energy': float(post_hf.e_corr),
                'refinement_converged': bool(getattr(post_hf, 'converged', True)),
            })
        except Exception as exc:  # pragma: no cover - depends on PySCF numerical behavior
            context['active_space_probe_final_active_space'] = preliminary
            decision.update({
                'status': 'refinement_failed',
                'refinement_completed': False,
                'refinement_error': str(exc),
            })
    else:
        context['active_space_probe_final_active_space'] = preliminary
        decision['status'] = 'refinement_unavailable'

    results['active_space_probe'] = decision
    state['_module_results'] = copy.deepcopy(results)
    refresh_molecular_module_artifacts(state, task_spec, results)
    observation = record_module_observation(
        state,
        invocation,
        'completed',
        decision=copy.deepcopy(decision),
    )
    append_log(state, 'info', 'workflow.active_space_probe_refinement_completed', observation)
    return state


__all__ = [
    'active_space_probe_refinement_decision',
    'prepare_molecular_active_space_probe',
    'refine_molecular_active_space_probe',
]
