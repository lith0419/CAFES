from __future__ import annotations

from pyscf_agent.serialization import json_default

import copy
import hashlib
import json
from typing import Any, Dict, List, Optional

from dataclasses import asdict, fields
from pyscf_agent.contracts import RuntimeSpec
from pyscf_agent.backend.parsing import task_spec_from_partial

from pyscf_agent.backend.active_space_probe import build_molecular_active_space_probe_request

from .active_space_contracts import (
    active_space_contract_from_case_report,
    active_space_solver_contract_from_request,
)
from .initial_scan import (
    DEFAULT_METHOD_POLICY as DEFAULT_METHOD_POLICY,
    DEFAULT_MOLECULAR_INITIAL_SCAN_OBSERVABLES,
    MOLECULAR_SINGLE_REFERENCE_METHODS,
    RECOVERY_SOLVER_ORDER,
    _as_study_spec_payload,
    _dedupe,
    _normalize_options,
    _set_model_solver,
    initial_scan_method_for_options,
)
from .study_active_space import resolve_study_active_space_contracts
from ..planner import build_study_plan
from ..costing import ensure_plan_cost_estimate
from ..normalization import normalize_system_type
from ..schema import StudyCase, StudyPlan

CAS_METHODS = ('casci', 'casscf')
DIRECT_ACTIVE_SPACE_PROBE_SCHEMA = 'pyscf-agent.direct-active-space-probe.v1'


def _recovery_solver(current_solver: Any, site_count: Optional[int], options: Dict[str, Any]) -> str:
    current = str(current_solver or '').strip().lower()
    policy = options['method_policy']
    preferred = []
    if site_count is not None and site_count <= options['max_fci_sites']:
        preferred.append(policy['strong_small'])
    else:
        preferred.append(policy['strong_large'])
    preferred.extend(RECOVERY_SOLVER_ORDER)
    for solver in preferred:
        normalized = str(solver or '').strip().lower()
        if normalized and normalized != current:
            return normalized
    return policy['strong_small'] if current != policy['strong_small'] else policy['weak']


def _active_space_is_approved(active_space: Any) -> bool:
    if not isinstance(active_space, dict):
        return False
    if not active_space.get('approved'):
        return False
    return active_space.get('ncas') is not None and active_space.get('nelecas') is not None


def _active_space_is_candidate(active_space: Any) -> bool:
    if not isinstance(active_space, dict):
        return False
    if active_space.get('ncas') is None or active_space.get('nelecas') is None:
        return False
    return bool(active_space.get('orbital_indices'))


def _approved_active_space_from_case_or_decision(case: StudyCase, decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    request_active_space = case.request.get('active_space') if isinstance(case.request, dict) else None
    if _active_space_is_approved(request_active_space):
        return copy.deepcopy(request_active_space)
    decision_active_space = decision.get('active_space_contract') if isinstance(decision, dict) else None
    if _active_space_is_approved(decision_active_space):
        return copy.deepcopy(decision_active_space)
    return None


def _active_space_candidate_from_case_or_decision(case: StudyCase, decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    request_active_space = case.request.get('active_space') if isinstance(case.request, dict) else None
    if _active_space_is_candidate(request_active_space):
        return copy.deepcopy(request_active_space)
    decision_active_space = decision.get('active_space_contract') if isinstance(decision, dict) else None
    if _active_space_is_candidate(decision_active_space):
        return copy.deepcopy(decision_active_space)
    return None


def _molecular_size_for_case(case: StudyCase) -> Dict[str, Optional[int]]:
    request = case.request if isinstance(case.request, dict) else {}
    atom = request.get('atom')
    basis = request.get('basis')
    if atom and basis:
        try:
            from pyscf import gto  # pylint: disable=import-outside-toplevel

            mol = gto.M(
                atom=atom,
                basis=basis,
                unit=request.get('unit') or 'Angstrom',
                charge=int(request.get('charge') or 0),
                spin=int(request.get('spin') or 0),
                verbose=0,
            )
            return {
                'norb': int(mol.nao_nr()),
                'nelec': int(mol.nelectron),
            }
        except Exception:
            pass
    return {
        'norb': None,
        'nelec': None,
    }


def _recovery_active_space_candidate(
    case: StudyCase,
    decision: Dict[str, Any],
    options: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    del options
    return _active_space_candidate_from_case_or_decision(case, decision)


def _molecular_fci_recovery_allowed(case: StudyCase, options: Dict[str, Any]) -> bool:
    size = _molecular_size_for_case(case)
    norb = size.get('norb')
    nelec = size.get('nelec')
    return bool(
        norb is not None
        and nelec is not None
        and norb <= options['max_molecular_fci_orbitals']
        and nelec <= options['max_molecular_fci_electrons']
    )


def _molecular_recovery_method(
    current_method: Any,
    case: StudyCase,
    decision: Dict[str, Any],
    options: Dict[str, Any],
) -> Optional[str]:
    current = str(current_method or '').strip().lower()
    policy = options['molecular_method_policy']
    approved_active_space = _approved_active_space_from_case_or_decision(case, decision)
    active_space_candidate = _recovery_active_space_candidate(case, decision, options)
    strong_method = str(policy.get('strong_small') or 'casscf').strip().lower()
    moderate_method = str(policy.get('moderate') or 'ccsd').strip().lower()
    if current in MOLECULAR_SINGLE_REFERENCE_METHODS and moderate_method and moderate_method != current:
        return moderate_method
    if approved_active_space and strong_method and strong_method != current:
        return strong_method
    if active_space_candidate and strong_method and strong_method != current:
        return strong_method
    if current != 'fci' and _molecular_fci_recovery_allowed(case, options):
        return 'fci'
    if approved_active_space and current != strong_method:
        return strong_method
    return None


def _molecular_reference_label(request: Dict[str, Any], method: Any = None) -> str:
    method_name = str(method or request.get('method') or '').strip().lower()
    restricted = request.get('restricted')
    try:
        spin = int(request.get('spin') or 0)
    except (TypeError, ValueError):
        spin = 0
    if method_name in CAS_METHODS and restricted is not False:
        return 'rohf_spin_adapted' if spin else 'rhf_spin_adapted'
    if restricted is False:
        return 'unrestricted'
    if restricted is True:
        return 'restricted'
    return 'automatic'


def _configure_molecular_method_request(request: Dict[str, Any], method: str) -> Dict[str, Any]:
    """Apply method/reference compatibility rules to an adaptive molecular request."""
    method_name = str(method or '').strip().lower()
    request['method'] = method_name
    if method_name != 'dft':
        request['xc'] = None

    post_cas = request.get('post_cas')
    if not isinstance(post_cas, dict):
        post_cas = {}
        request['post_cas'] = post_cas
    sc_nevpt2 = post_cas.get('sc_nevpt2')
    if not isinstance(sc_nevpt2, dict):
        sc_nevpt2 = {}
        post_cas['sc_nevpt2'] = sc_nevpt2

    if method_name in CAS_METHODS:
        # Adaptive diagnostics may use UHF/UMP2, but the executable CAS route is
        # spin adapted so open-shell cases use ROHF and can support SC-NEVPT2.
        request['restricted'] = True
        sc_nevpt2.setdefault('enabled', False)
    else:
        # A CAS eigensolver belongs to the CAS task only.  Keeping it on an
        # adaptively selected MP2/CC request misstates what actually ran and
        # can leak solver-specific options into unrelated execution paths.
        request.pop('solver', None)
        sc_nevpt2['enabled'] = False
    return request


def _molecular_reference_transition(source_request: Dict[str, Any], target_request: Dict[str, Any]) -> Dict[str, Any]:
    source_method = str(source_request.get('method') or '').strip().lower()
    target_method = str(target_request.get('method') or '').strip().lower()
    source_reference = _molecular_reference_label(source_request, source_method)
    target_reference = _molecular_reference_label(target_request, target_method)
    return {
        'source_method': source_method,
        'source_reference': source_reference,
        'target_method': target_method,
        'target_reference': target_reference,
        'changed': source_method != target_method or source_reference != target_reference,
        'reason': (
            'Adaptive CAS execution uses an RHF/ROHF spin-adapted reference; '
            'unrestricted CAS is not routed to SC-NEVPT2.'
            if target_method in CAS_METHODS
            else 'The selected single-reference method retains its requested reference policy.'
        ),
    }


def _molecular_recovery_request(
    case: StudyCase,
    method: str,
    decision: Dict[str, Any],
    observables: List[str],
    active_space_candidate: Optional[Dict[str, Any]] = None,
    adaptive_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    request = copy.deepcopy(case.request)
    _configure_molecular_method_request(request, method)
    request['analysis'] = {'outputs': _case_outputs_for_molecular_method(observables, method)}
    if method in CAS_METHODS:
        active_space = copy.deepcopy(active_space_candidate) if active_space_candidate else _active_space_candidate_from_case_or_decision(case, decision)
        if active_space:
            active_space['enabled'] = True
            active_space['selection_method'] = 'manual'
            active_space['approved'] = bool(active_space.get('approved'))
            request['active_space'] = active_space
        method, _solver = _configure_active_space_solver_request(
            request,
            method,
            decision,
            adaptive_options or _normalize_options(None),
        )
    else:
        request['active_space'] = {'enabled': False}
    return request


def _runtime_recovery_request(request: Dict[str, Any], *, minimum_max_cycle: int = 200) -> Dict[str, Any]:
    next_request = copy.deepcopy(request)
    runtime_fields = {field.name for field in fields(RuntimeSpec)} | {'runtime'}
    runtime_input = {key: value for key, value in request.items() if key in runtime_fields}
    runtime = asdict(task_spec_from_partial(runtime_input).runtime)
    runtime['max_cycle'] = max(runtime['max_cycle'] * 2, minimum_max_cycle)
    # Remove shorthand runtime aliases so they cannot override this canonical update.
    for field in runtime:
        next_request.pop(field, None)
    next_request['runtime'] = runtime
    return next_request


def _recovery_next_step(status: Any, recovery_action: str, recovery_method: Any, active_space: Any = None) -> str:
    status_text = str(status or '').strip().lower()
    method_label = str(recovery_method or '').strip().upper() or 'the selected method'
    if status_text == 'blocked' and isinstance(active_space, dict) and active_space.get('enabled') and not active_space.get('approved'):
        return 'Approve or edit the proposed ActiveSpaceAudit, then rerun the recovery calculation.'
    if status_text == 'blocked':
        return 'Open the case validation details, fix the blocked request fields, then rerun the affected recovery case.'
    if status_text == 'unconverged':
        if recovery_action == 'increase_max_cycle':
            return 'The higher max_cycle retry still did not converge; switch to a different method or review active-space treatment for this case.'
        return 'The fallback {0} calculation still did not converge; increase max_cycle further or choose a stronger fallback method.'.format(method_label)
    if status_text and status_text != 'succeeded':
        return 'Inspect the case error details, adjust runtime or method settings, then rerun this affected case.'
    return ''


def _numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, str):
        token = value.strip().split()[0] if value.strip() else ''
        try:
            return float(token)
        except ValueError:
            return None
    return None


def _decision_distance(variables: Dict[str, Any], decision: Dict[str, Any]) -> float:
    decision_variables = decision.get('variables') if isinstance(decision.get('variables'), dict) else {}
    keys = sorted(set(variables) | set(decision_variables))
    if not keys:
        return 0.0
    distance = 0.0
    comparable = 0
    for key in keys:
        left = _numeric_value(variables.get(key))
        right = _numeric_value(decision_variables.get(key))
        if left is not None and right is not None:
            scale = max(abs(left), abs(right), 1.0)
            distance += abs(left - right) / scale
            comparable += 1
            continue
        if variables.get(key) != decision_variables.get(key):
            distance += 1.0
        comparable += 1
    return distance / max(comparable, 1)


def _nearest_decision(variables: Dict[str, Any], decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not decisions:
        raise ValueError('Cannot refine a study without initial-scan diagnostic decisions')
    return min(decisions, key=lambda item: _decision_distance(variables, item))


def _case_outputs_for_solver(original_outputs: List[str], solver: str) -> List[str]:
    outputs = _dedupe(list(original_outputs or ['energy']) + ['energy'])
    if solver in ('mp2', 'ccsd', 'ccsd_t', 'fci'):
        outputs = _dedupe(outputs + ['strong_correlation_diagnostics'])
    return outputs


def _case_outputs_for_molecular_method(original_outputs: List[str], method: str) -> List[str]:
    return _dedupe(list(original_outputs or ['energy']) + list(DEFAULT_MOLECULAR_INITIAL_SCAN_OBSERVABLES))


def _active_space_solver_for_decision(
    selected_decision: Dict[str, Any],
    adaptive_options: Dict[str, Any],
) -> str:
    configured = str(adaptive_options.get('active_space_solver') or 'auto').strip().lower().replace('-', '_')
    if configured in ('block2', 'dmrg'):
        configured = 'block2_dmrg'
    if configured in ('fci', 'block2_dmrg'):
        return configured
    contract = selected_decision.get('active_space_contract')
    contract = contract if isinstance(contract, dict) else {}
    recommendation = contract.get('method_recommendation')
    recommendation = recommendation if isinstance(recommendation, dict) else {}
    next_step = str(
        recommendation.get('recommended_next_step')
        or (contract.get('audit_summary') or {}).get('recommended_next_step')
        or ''
    ).strip().lower()
    if 'block2' in next_step or 'dmrg' in next_step:
        return 'block2_dmrg'
    try:
        ncas = int(contract.get('ncas'))
    except (TypeError, ValueError):
        ncas = None
    if ncas is not None and ncas > int(adaptive_options.get('max_cas_orbitals') or 14):
        return 'block2_dmrg'
    return 'fci'


def _configure_active_space_solver_request(
    request: Dict[str, Any],
    method: str,
    selected_decision: Dict[str, Any],
    adaptive_options: Dict[str, Any],
) -> tuple[str, str]:
    if method not in CAS_METHODS:
        return method, method
    requested_solver, requested_options = active_space_solver_contract_from_request(request)
    solver = requested_solver or _active_space_solver_for_decision(
        selected_decision,
        adaptive_options,
    )
    if solver == 'block2_dmrg':
        method = 'casscf'
        _configure_molecular_method_request(request, method)
        options = copy.deepcopy(requested_options)
        if not options:
            options = copy.deepcopy(adaptive_options.get('block2_dmrg_options') or {})
        options.setdefault('preset', 'balanced')
        options.setdefault('save_mps', True)
        request['solver'] = {'name': 'block2_dmrg', 'options': options}
        orbital_processing = adaptive_options.get('active_space_orbital_processing')
        if isinstance(orbital_processing, dict) and orbital_processing:
            request['orbital_processing'] = copy.deepcopy(orbital_processing)
        current_analysis = request.get('analysis')
        current_outputs = (
            list(current_analysis.get('outputs') or [])
            if isinstance(current_analysis, dict)
            else []
        )
        request['analysis'] = {
            'outputs': _dedupe(
                _case_outputs_for_molecular_method(current_outputs, method)
                + ['entanglement_diagnostics', 'symmetry_analysis']
            ),
        }
    else:
        request['solver'] = {'name': 'fci', 'options': {}}
        orbital_processing = request.get('orbital_processing')
        if isinstance(orbital_processing, dict):
            orbital_processing['localization_scope'] = 'analysis'
            orbital_processing['orbital_ordering'] = 'canonical'
            orbital_processing['orbital_order'] = []
    return method, solver


def _casscf_review_active_space(selected_decision: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    contract = selected_decision.get('active_space_contract')
    if isinstance(contract, dict) and contract:
        active_space = copy.deepcopy(contract)
        active_space['enabled'] = True
        active_space['selection_method'] = 'manual'
        active_space['approved'] = False
        return active_space
    return None


def _build_model_refined_plan_from_decisions(
    base_plan: StudyPlan,
    decisions: List[Dict[str, Any]],
    adaptive_options: Dict[str, Any],
) -> Dict[str, Any]:
    original_outputs = list(base_plan.observables or ['energy'])
    refined_cases: List[StudyCase] = []
    decision_log = []
    for case in base_plan.cases:
        selected_decision = _nearest_decision(case.variables, decisions)
        solver = str(selected_decision.get('recommended_solver') or adaptive_options['method_policy']['weak']).lower()
        configured_solver_options = adaptive_options.get('model_solver_options')
        configured_solver_options = configured_solver_options if isinstance(configured_solver_options, dict) else {}
        solver_options = configured_solver_options.get(solver)
        solver_options = copy.deepcopy(solver_options) if isinstance(solver_options, dict) else {}
        request = copy.deepcopy(case.request)
        request['solver'] = {
            'name': solver,
            'options': solver_options,
        } if solver_options else solver
        request['analysis'] = {'outputs': _case_outputs_for_solver(original_outputs, solver)}
        refined_cases.append(StudyCase(
            case_id=case.case_id,
            label=case.label,
            request=request,
            variables=copy.deepcopy(case.variables),
            operations=copy.deepcopy(case.operations),
            model_spec=copy.deepcopy(case.model_spec),
        ))
        decision_log.append({
            'case_id': case.case_id,
            'label': case.label,
            'variables': copy.deepcopy(case.variables),
            'source_initial_scan_case_id': selected_decision.get('case_id'),
            'initial_scan_status': selected_decision.get('initial_scan_status'),
            'initial_scan_method': selected_decision.get('initial_scan_method'),
            'initial_scan_run_dir': selected_decision.get('initial_scan_run_dir'),
            'level': selected_decision.get('level'),
            'diagnostic_level': selected_decision.get('diagnostic_level'),
            'physics_level': selected_decision.get('physics_level'),
            'solver_stress_level': selected_decision.get('solver_stress_level'),
            'routing_level': selected_decision.get('routing_level'),
            'score': selected_decision.get('score'),
            'recommended_solver': solver,
            'recommended_method': solver,
            'tags': copy.deepcopy(selected_decision.get('tags') or []),
            'reasons': copy.deepcopy(selected_decision.get('reasons') or []),
        })
    return {'cases': refined_cases, 'decision_log': decision_log, 'observables': original_outputs}


def _build_molecular_refined_plan_from_decisions(
    base_plan: StudyPlan,
    decisions: List[Dict[str, Any]],
    adaptive_options: Dict[str, Any],
) -> Dict[str, Any]:
    original_outputs = list(base_plan.observables or ['energy'])
    study_decisions, study_active_space_policy = resolve_study_active_space_contracts(decisions)
    refined_cases: List[StudyCase] = []
    decision_log = []
    for case in base_plan.cases:
        selected_decision = _nearest_decision(case.variables, study_decisions)
        method = str(
            selected_decision.get('recommended_method')
            or selected_decision.get('recommended_solver')
            or adaptive_options['molecular_method_policy']['weak']
        ).lower()
        request = copy.deepcopy(case.request)
        _configure_molecular_method_request(request, method)
        request['analysis'] = {'outputs': _case_outputs_for_molecular_method(original_outputs, method)}
        if method in CAS_METHODS:
            review_active_space = _casscf_review_active_space(selected_decision)
            if not review_active_space:
                if 'active_space_choice_review_required' in selected_decision.get('tags', []):
                    raise ValueError(
                        'Case {0} has an explicit manual/approved active-space conflict; '
                        'review the preserved choices before multireference execution.'.format(case.case_id)
                    )
                raise ValueError(
                    'Case {0} requires {1}, but it has no case-specific orbital mapping compatible '
                    'with the shared study active space. Rerun the active-space probe for this case; '
                    'a fixed frontier CAS fallback is not allowed.'.format(
                        case.case_id,
                        method.upper(),
                    )
                )
            request['active_space'] = review_active_space
            method, active_space_solver = _configure_active_space_solver_request(
                request,
                method,
                selected_decision,
                adaptive_options,
            )
            request['analysis']['outputs'] = _dedupe(
                list(request['analysis'].get('outputs') or []) + list(original_outputs or ['energy'])
            )
        else:
            request['active_space'] = {'enabled': False}
            active_space_solver = method
        refined_cases.append(StudyCase(
            case_id=case.case_id,
            label=case.label,
            request=request,
            variables=copy.deepcopy(case.variables),
            operations=copy.deepcopy(case.operations),
            model_spec=copy.deepcopy(case.model_spec),
        ))
        decision_log.append({
            'case_id': case.case_id,
            'label': case.label,
            'variables': copy.deepcopy(case.variables),
            'source_initial_scan_case_id': selected_decision.get('case_id'),
            'initial_scan_status': selected_decision.get('initial_scan_status'),
            'initial_scan_method': selected_decision.get('initial_scan_method'),
            'initial_scan_run_dir': selected_decision.get('initial_scan_run_dir'),
            'level': selected_decision.get('level'),
            'diagnostic_level': selected_decision.get('diagnostic_level'),
            'physics_level': selected_decision.get('physics_level'),
            'solver_stress_level': selected_decision.get('solver_stress_level'),
            'routing_level': selected_decision.get('routing_level'),
            'score': selected_decision.get('score'),
            'recommended_solver': active_space_solver,
            'recommended_method': method,
            'reference_transition': _molecular_reference_transition(case.request, request),
            'active_space_contract': copy.deepcopy(selected_decision.get('active_space_contract')),
            'study_active_space_status': selected_decision.get('study_active_space_status'),
            'study_active_space_policy': copy.deepcopy(study_active_space_policy),
            'diagnostics': copy.deepcopy(selected_decision.get('diagnostics') or {}),
            'tags': copy.deepcopy(selected_decision.get('tags') or []),
            'reasons': copy.deepcopy(selected_decision.get('reasons') or []),
        })
    return {
        'cases': refined_cases,
        'decision_log': decision_log,
        'observables': original_outputs,
        'study_active_space_policy': study_active_space_policy,
    }


def build_refined_plan_from_decisions(
    spec: Any,
    decisions: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    if any('initial_scan_blocked' in (item.get('tags') or []) for item in decisions if isinstance(item, dict)):
        raise ValueError('Cannot build a refined adaptive plan because one or more initial-scan cases were blocked')
    adaptive_options = _normalize_options(options)
    base_spec_payload = _as_study_spec_payload(spec)
    system_type = normalize_system_type(base_spec_payload.get('system_type'))
    if system_type == 'model_hamiltonian':
        _set_model_solver(
            base_spec_payload,
            initial_scan_method_for_options(adaptive_options, system_type=system_type),
        )
    base_plan = build_study_plan(base_spec_payload)
    if base_plan.system_type == 'model_hamiltonian':
        refined_payload = _build_model_refined_plan_from_decisions(base_plan, decisions, adaptive_options)
    elif base_plan.system_type == 'molecular':
        refined_payload = _build_molecular_refined_plan_from_decisions(base_plan, decisions, adaptive_options)
    else:
        raise ValueError('Adaptive scan currently supports molecular or model_hamiltonian studies only')
    refined_cases = refined_payload['cases']
    decision_log = refined_payload['decision_log']
    original_outputs = refined_payload['observables']
    refined_plan = StudyPlan(
        study_id='refined',
        name='{0}-refined'.format(base_plan.name),
        objective='Adaptive refined calculation for {0}'.format(base_plan.objective),
        system_type=base_plan.system_type,
        cases=refined_cases,
        observables=original_outputs,
        comparison=copy.deepcopy(base_plan.comparison),
        capability_snapshot=copy.deepcopy(base_plan.capability_snapshot),
        resource_policy=copy.deepcopy(base_plan.resource_policy),
    )
    # Refined methods can be much more expensive than the initial scan, so
    # preserve the caller's policy and estimate this exact set of cases now.
    ensure_plan_cost_estimate(refined_plan)
    return {
        'refined_plan': refined_plan.to_dict(),
        'decision_log': decision_log,
        'adaptive_options': adaptive_options,
        'study_active_space_policy': copy.deepcopy(
            refined_payload.get('study_active_space_policy')
        ),
    }


def _unify_active_space_review_cases(
    review_cases: List[StudyCase],
    review_decisions: List[Dict[str, Any]],
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    resolved_decisions, policy = resolve_study_active_space_contracts(review_decisions)
    if policy.get('status') in ('unavailable', 'incomplete', 'conflict'):
        unresolved = _dedupe(
            list(policy.get('missing_case_ids') or [])
            + list(policy.get('conflicting_case_ids') or [])
        )
        if any('active_space_choice_review_required' in decision.get('tags', []) for decision in resolved_decisions):
            raise ValueError(
                'Explicit manual/approved active spaces conflict across the study. '
                'Review the preserved choices before execution: {0}.'.format(', '.join(unresolved))
            )
        raise ValueError(
            'The study does not yet have one consistent evidence-backed active space. '
            'Rerun the active-space probe for: {0}.'.format(', '.join(unresolved) or 'the requested cases')
        )
    decisions_by_id = {
        str(item.get('case_id')): item
        for item in resolved_decisions
        if isinstance(item, dict) and item.get('case_id') is not None
    }
    for case in review_cases:
        decision = decisions_by_id.get(case.case_id, {})
        contract = decision.get('active_space_contract')
        if not isinstance(contract, dict):
            continue
        active_space = copy.deepcopy(contract)
        active_space['enabled'] = True
        active_space['selection_method'] = 'manual'
        active_space['approved'] = False
        case.request['active_space'] = active_space
    return resolved_decisions, policy


def build_casscf_active_space_review_plan(
    spec: Any,
    options: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Turn a direct molecular CAS request into an ActiveSpaceAudit review plan.

    Only complete, evidence-backed candidates can enter review directly.  If
    any requested CAS case is missing a candidate, the caller must run the
    shared molecular active-space probe first so no case receives a fabricated
    frontier-space fallback.
    """
    adaptive_options = _normalize_options(options)
    base_spec_payload = _as_study_spec_payload(spec)
    if normalize_system_type(base_spec_payload.get('system_type')) != 'molecular':
        return None
    base_plan = build_study_plan(base_spec_payload)
    observables = list(base_plan.observables or ['energy'])
    review_cases: List[StudyCase] = []
    review_decisions: List[Dict[str, Any]] = []
    target_case_count = 0

    for case in base_plan.cases:
        requested_method = str(case.request.get('method') or '').strip().lower()
        if requested_method not in CAS_METHODS:
            continue
        target_case_count += 1
        candidate = _active_space_candidate_from_case_or_decision(case, {})
        if not candidate:
            return None
        request = _molecular_recovery_request(
            case,
            requested_method,
            {},
            observables,
            candidate,
            adaptive_options,
        )
        active_space = request.get('active_space')
        if not isinstance(active_space, dict):
            continue
        active_space['enabled'] = True
        active_space['selection_method'] = 'manual'
        active_space['approved'] = False
        request['active_space'] = active_space
        review_cases.append(StudyCase(
            case_id=case.case_id,
            label=case.label,
            request=request,
            variables=copy.deepcopy(case.variables),
            operations=copy.deepcopy(case.operations),
            model_spec=copy.deepcopy(case.model_spec),
        ))
        review_decision = {
            'case_id': case.case_id,
            'label': case.label,
            'variables': copy.deepcopy(case.variables),
            'recommended_method': requested_method,
            'active_space_contract': copy.deepcopy(active_space),
            'tags': [
                'direct_casscf_request',
                'review_active_space',
                'requires_active_space_approval',
            ],
            'reason': (
                'Direct {0} request requires approval of the proposed '
                'ActiveSpaceAudit before execution.'
            ).format(requested_method.upper()),
            'next_step': 'Approve or edit the proposed ActiveSpaceAudit, then run this CASSCF/CASCI calculation.',
        }
        solver, _solver_options = _request_solver_contract(case.request)
        if solver:
            review_decision['recommended_solver'] = solver
        review_decisions.append(review_decision)

    if not target_case_count or len(review_cases) != target_case_count:
        return None
    review_decisions, study_active_space_policy = _unify_active_space_review_cases(
        review_cases,
        review_decisions,
    )
    review_plan = StudyPlan(
        study_id='casscf-active-space-review',
        name='{0}-active-space-review'.format(base_plan.name),
        objective='Active-space review for {0}'.format(base_plan.objective),
        system_type='molecular',
        cases=review_cases,
        observables=observables,
        comparison=copy.deepcopy(base_plan.comparison),
        capability_snapshot=copy.deepcopy(base_plan.capability_snapshot),
        resource_policy=copy.deepcopy(base_plan.resource_policy),
    )
    ensure_plan_cost_estimate(review_plan)
    return {
        'active_space_review_plan': review_plan.to_dict(),
        'active_space_review_decisions': review_decisions,
        'study_active_space_policy': study_active_space_policy,
        'adaptive_options': adaptive_options,
    }


def _request_solver_contract(request: Dict[str, Any]) -> tuple[Optional[str], Dict[str, Any]]:
    solver = request.get('solver') if isinstance(request, dict) else None
    if isinstance(solver, dict):
        name = str(solver.get('name') or '').strip().lower().replace('-', '_') or None
        options = solver.get('options') if isinstance(solver.get('options'), dict) else {}
        return name, copy.deepcopy(options)
    if isinstance(solver, str) and solver.strip():
        return solver.strip().lower().replace('-', '_'), {}
    return None, {}


def build_casscf_active_space_probe_plan(
    spec: Any,
    options: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build per-case numerical probes for direct CAS requests without evidence."""

    adaptive_options = _normalize_options(options)
    base_spec_payload = _as_study_spec_payload(spec)
    if normalize_system_type(base_spec_payload.get('system_type')) != 'molecular':
        return None
    base_plan = build_study_plan(base_spec_payload)
    target_cases: List[Dict[str, Any]] = []
    probe_cases: List[StudyCase] = []
    strategy = adaptive_options.get('initial_scan_strategy') or 'auto'

    for case in base_plan.cases:
        requested_method = str(case.request.get('method') or '').strip().lower()
        if requested_method not in CAS_METHODS:
            continue
        target_cases.append(case.to_dict())
        if _active_space_candidate_from_case_or_decision(case, {}):
            continue
        target_solver, target_solver_options = _request_solver_contract(case.request)
        request_active_space = case.request.get('active_space')
        selection_method = (
            request_active_space.get('selection_method')
            if isinstance(request_active_space, dict)
            else None
        )
        probe = build_molecular_active_space_probe_request(
            case.request,
            strategy=strategy,
            target_method=requested_method,
            target_solver=target_solver,
            target_solver_options=target_solver_options,
            selection_method=selection_method,
        )
        probe_cases.append(StudyCase(
            case_id=case.case_id,
            label=case.label,
            request=probe['request'],
            variables=copy.deepcopy(case.variables),
            operations=copy.deepcopy(case.operations),
            model_spec=copy.deepcopy(case.model_spec),
        ))

    if not probe_cases:
        return None
    provenance = copy.deepcopy(base_plan.workflow_provenance)
    provenance['direct_active_space_probe'] = {
        'schema': DIRECT_ACTIVE_SPACE_PROBE_SCHEMA,
        'target_cases': target_cases,
        'target_observables': list(base_plan.observables or ['energy']),
        'probe_case_ids': [case.case_id for case in probe_cases],
    }
    probe_plan = StudyPlan(
        study_id='casscf-active-space-probe',
        name='{0}-active-space-probe'.format(base_plan.name),
        objective='Active-space probe for {0}'.format(base_plan.objective),
        system_type='molecular',
        cases=probe_cases,
        observables=_dedupe(
            list(base_plan.observables or ['energy'])
            + list(DEFAULT_MOLECULAR_INITIAL_SCAN_OBSERVABLES)
        ),
        comparison=copy.deepcopy(base_plan.comparison),
        capability_snapshot=copy.deepcopy(base_plan.capability_snapshot),
        resource_policy=copy.deepcopy(base_plan.resource_policy),
        workflow_provenance=provenance,
    )
    ensure_plan_cost_estimate(probe_plan)
    return {
        'active_space_probe_plan': probe_plan.to_dict(),
        'adaptive_options': adaptive_options,
    }


def build_casscf_active_space_review_from_probe(
    probe_plan: Any,
    probe_report: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Convert executed shared probes into the requested per-case CAS review."""

    adaptive_options = _normalize_options(options)
    plan = probe_plan if isinstance(probe_plan, StudyPlan) else StudyPlan.from_dict(probe_plan)
    metadata = plan.workflow_provenance.get('direct_active_space_probe')
    if not isinstance(metadata, dict):
        return None
    target_payloads = metadata.get('target_cases')
    if not isinstance(target_payloads, list) or not target_payloads:
        return None
    target_case_count = sum(isinstance(payload, dict) for payload in target_payloads)
    report_cases = {
        str(item.get('case_id')): item
        for item in (probe_report.get('cases') or [])
        if isinstance(item, dict) and item.get('case_id') is not None
    }
    review_cases: List[StudyCase] = []
    review_decisions: List[Dict[str, Any]] = []
    target_observables = metadata.get('target_observables')
    observables = (
        list(target_observables)
        if isinstance(target_observables, list) and target_observables
        else list(plan.observables or ['energy'])
    )

    for payload in target_payloads:
        if not isinstance(payload, dict):
            continue
        target_case = StudyCase.from_dict(payload)
        candidate = _active_space_candidate_from_case_or_decision(target_case, {})
        source = 'user-provided active-space contract'
        if not candidate:
            candidate = active_space_contract_from_case_report(
                report_cases.get(target_case.case_id, {})
            )
            source = 'per-case molecular active-space probe'
        if not candidate:
            continue
        request = copy.deepcopy(target_case.request)
        method = str(request.get('method') or 'casscf').strip().lower()
        _configure_molecular_method_request(request, method)
        candidate['enabled'] = True
        candidate['selection_method'] = 'manual'
        candidate['approved'] = False
        request['active_space'] = candidate
        request['analysis'] = {
            'outputs': _case_outputs_for_molecular_method(observables, method),
        }
        method, _active_space_solver = _configure_active_space_solver_request(
            request,
            method,
            {'active_space_contract': candidate},
            adaptive_options,
        )
        solver, _solver_options = _request_solver_contract(request)
        review_cases.append(StudyCase(
            case_id=target_case.case_id,
            label=target_case.label,
            request=request,
            variables=copy.deepcopy(target_case.variables),
            operations=copy.deepcopy(target_case.operations),
            model_spec=copy.deepcopy(target_case.model_spec),
        ))
        review_decision = {
            'case_id': target_case.case_id,
            'label': target_case.label,
            'variables': copy.deepcopy(target_case.variables),
            'recommended_method': method,
            'active_space_contract': copy.deepcopy(candidate),
            'tags': [
                'direct_casscf_request',
                'review_active_space',
                'requires_active_space_approval',
                'active_space_probe_completed',
            ],
            'reason': (
                '{0} produced a case-specific CAS({1}, {2}) candidate; '
                'ActiveSpaceAudit approval is required before execution.'
            ).format(source.capitalize(), candidate.get('nelecas'), candidate.get('ncas')),
            'next_step': 'Approve or edit the proposed ActiveSpaceAudit, then run the requested CASSCF/CASCI calculation.',
        }
        if solver:
            review_decision['recommended_solver'] = solver
        review_decisions.append(review_decision)

    if not target_case_count or len(review_cases) != target_case_count:
        return None
    review_decisions, study_active_space_policy = _unify_active_space_review_cases(
        review_cases,
        review_decisions,
    )
    review_plan = StudyPlan(
        study_id='casscf-active-space-review',
        name='{0}-review'.format(plan.name),
        objective='Review probe-selected active spaces for {0}'.format(plan.objective),
        system_type='molecular',
        cases=review_cases,
        observables=observables,
        comparison=copy.deepcopy(plan.comparison),
        capability_snapshot=copy.deepcopy(plan.capability_snapshot),
        resource_policy=copy.deepcopy(plan.resource_policy),
    )
    ensure_plan_cost_estimate(review_plan)
    return {
        'active_space_review_plan': review_plan.to_dict(),
        'active_space_review_decisions': review_decisions,
        'study_active_space_policy': study_active_space_policy,
        'adaptive_options': adaptive_options,
    }


def _case_model_site_count(case: StudyCase) -> Optional[int]:
    sites = case.model_spec.get('sites') if isinstance(case.model_spec, dict) else None
    if isinstance(sites, list):
        return len(sites)
    model_hamiltonian = case.request.get('model_hamiltonian') if isinstance(case.request, dict) else {}
    spec = model_hamiltonian.get('spec') if isinstance(model_hamiltonian, dict) else {}
    sites = spec.get('sites') if isinstance(spec, dict) else None
    return len(sites) if isinstance(sites, list) else None


def _case_reports_by_id(report_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get('case_id')): item
        for item in (report_payload.get('cases') or [])
        if isinstance(item, dict) and item.get('case_id') is not None
    }


def _comparison_rows_by_id(report_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get('case_id')): item
        for item in (report_payload.get('comparison_table') or [])
        if isinstance(item, dict) and item.get('case_id') is not None
    }


def build_recovery_plan_for_unresolved_refined_cases(
    refined_plan: StudyPlan,
    refined_report_payload: Dict[str, Any],
    decision_log: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    if refined_plan.system_type not in ('model_hamiltonian', 'molecular'):
        return None
    adaptive_options = _normalize_options(options)
    rows_by_id = _comparison_rows_by_id(refined_report_payload)
    decision_by_id = {str(item.get('case_id')): item for item in decision_log if isinstance(item, dict) and item.get('case_id') is not None}
    recovery_cases: List[StudyCase] = []
    recovery_decisions = []
    for case in refined_plan.cases:
        row = rows_by_id.get(case.case_id, {})
        initial_status = str(row.get('status') or '').strip().lower()
        if initial_status not in ('unconverged', 'failed'):
            continue
        failure_description = 'did not converge' if initial_status == 'unconverged' else 'failed during execution'
        decision = decision_by_id.get(case.case_id, {})
        if refined_plan.system_type == 'model_hamiltonian':
            current_solver = row.get('solver') or case.request.get('solver')
            solver = _recovery_solver(current_solver, _case_model_site_count(case), adaptive_options)
            request = _runtime_recovery_request(case.request)
            request['solver'] = solver
            current_method = current_solver
            recovery_method = solver
            recovery_action = 'switch_solver' if str(solver).lower() != str(current_solver or '').lower() else 'increase_max_cycle'
            reason = 'Refined solver {0}; retrying with {1} and max_cycle={2}.'.format(
                failure_description,
                solver.upper(),
                request.get('runtime', {}).get('max_cycle'),
            )
        else:
            current_method = row.get('method') or case.request.get('method') or row.get('solver')
            recovery_method = _molecular_recovery_method(current_method, case, decision, adaptive_options)
            if not recovery_method:
                recovery_method = str(current_method or case.request.get('method') or 'hf').strip().lower()
            active_space_candidate = _recovery_active_space_candidate(case, decision, adaptive_options)
            request = _runtime_recovery_request(_molecular_recovery_request(
                case,
                recovery_method,
                decision,
                list(refined_plan.observables or ['energy']),
                active_space_candidate,
                adaptive_options,
            ))
            configured_solver, _configured_solver_options = _request_solver_contract(request)
            solver = configured_solver or recovery_method
            recovery_action = 'increase_max_cycle' if str(recovery_method).lower() == str(current_method or '').lower() else 'switch_method'
            if recovery_method in ('casci', 'casscf') and active_space_candidate:
                reason = 'Refined molecular method {0}; proposing {1} with ActiveSpaceAudit for user approval and max_cycle={2}.'.format(
                    failure_description,
                    recovery_method.upper(),
                    request.get('runtime', {}).get('max_cycle'),
                )
            elif recovery_action == 'increase_max_cycle':
                reason = 'Refined molecular method {0}; retrying {1} with increased max_cycle={2}.'.format(
                    failure_description,
                    recovery_method.upper(),
                    request.get('runtime', {}).get('max_cycle'),
                )
            else:
                reason = 'Refined molecular method {0}; retrying with {1} and max_cycle={2}.'.format(
                    failure_description,
                    recovery_method.upper(),
                    request.get('runtime', {}).get('max_cycle'),
                )
        recovery_cases.append(StudyCase(
            case_id=case.case_id,
            label=case.label,
            request=request,
            variables=copy.deepcopy(case.variables),
            operations=copy.deepcopy(case.operations),
            model_spec=copy.deepcopy(case.model_spec),
        ))
        diagnostic_level = decision.get('diagnostic_level') or decision.get('level')
        physics_level = decision.get('physics_level') or diagnostic_level
        recovery_tags = _dedupe(
            list(decision.get('tags') or [])
            + ['solver_failure_strong_correlation_evidence']
        )
        recovery_decisions.append({
            'case_id': case.case_id,
            'label': case.label,
            'variables': copy.deepcopy(case.variables),
            'initial_solver': current_method,
            'initial_method': current_method,
            'recovery_solver': solver,
            'recovery_method': recovery_method,
            'recovery_action': recovery_action,
            'runtime': copy.deepcopy(request.get('runtime') or {}),
            'initial_status': row.get('status'),
            'level': 'strong',
            'diagnostic_level': diagnostic_level,
            'physics_level': physics_level,
            'solver_stress_level': 'strong',
            'level_reason': 'refined_solver_{0}'.format(initial_status),
            'tags': recovery_tags,
            'reason': reason,
            'correlation_reason': (
                'The refined {0} status is strong solver-stress evidence. '
                'The independent physics level remains {1}.'
            ).format(initial_status, physics_level or 'unknown'),
            'next_step': _recovery_next_step(
                'blocked' if (
                    recovery_method in ('casci', 'casscf')
                    and isinstance(request.get('active_space'), dict)
                    and not request['active_space'].get('approved')
                ) else None,
                recovery_action,
                recovery_method,
                request.get('active_space') if isinstance(request.get('active_space'), dict) else None,
            ) or 'Run the generated recovery case; if it remains unresolved, inspect the returned status-specific guidance.',
            'source_initial_scan_case_id': decision.get('source_initial_scan_case_id'),
            'active_space_contract': (
                copy.deepcopy(request.get('active_space'))
                if isinstance(request.get('active_space'), dict)
                and request['active_space'].get('enabled')
                else None
            ),
            'reference_transition': (
                _molecular_reference_transition(case.request, request)
                if refined_plan.system_type == 'molecular'
                else None
            ),
        })
    if not recovery_cases:
        return None
    recovery_plan = StudyPlan(
        study_id='refined-recovery',
        name='{0}-recovery'.format(refined_plan.name),
        objective='Recovery calculations for unresolved refined cases',
        system_type=refined_plan.system_type,
        cases=recovery_cases,
        observables=copy.deepcopy(refined_plan.observables),
        comparison=copy.deepcopy(refined_plan.comparison),
        capability_snapshot=copy.deepcopy(refined_plan.capability_snapshot),
        resource_policy=copy.deepcopy(refined_plan.resource_policy),
    )
    ensure_plan_cost_estimate(recovery_plan)
    return {
        'recovery_plan': recovery_plan.to_dict(),
        'recovery_decisions': recovery_decisions,
        'adaptive_options': adaptive_options,
    }


def build_recovery_plan_for_unconverged_refined_cases(
    refined_plan: StudyPlan,
    refined_report_payload: Dict[str, Any],
    decision_log: List[Dict[str, Any]],
    options: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Backward-compatible name for unresolved refined-case recovery."""
    return build_recovery_plan_for_unresolved_refined_cases(
        refined_plan,
        refined_report_payload,
        decision_log,
        options,
    )


def _molecular_atom_symbols(atom: Any) -> List[str]:
    if not isinstance(atom, str):
        return []
    symbols: List[str] = []
    for entry in atom.replace('\n', ';').split(';'):
        fields = entry.strip().split()
        if fields:
            symbols.append(fields[0])
    return symbols


def _effective_restricted_reference(request: Dict[str, Any]) -> bool:
    explicit = request.get('restricted')
    if explicit is not None:
        return bool(explicit)
    method = str(request.get('method') or '').strip().lower()
    if method in CAS_METHODS:
        return True
    try:
        return int(request.get('spin') or 0) == 0
    except (TypeError, ValueError):
        return True


def _reference_label(request: Dict[str, Any]) -> str:
    restricted = _effective_restricted_reference(request)
    try:
        spin = int(request.get('spin') or 0)
    except (TypeError, ValueError):
        spin = 0
    if restricted and spin:
        return 'rohf'
    return 'rhf' if restricted else 'uhf'


def _continuation_contract(
    request: Dict[str, Any],
    *,
    include_method: bool = True,
) -> Optional[Dict[str, Any]]:
    if not isinstance(request, dict):
        return None
    symbols = _molecular_atom_symbols(request.get('atom'))
    basis = str(request.get('basis') or '').strip().lower()
    method = str(request.get('method') or '').strip().lower()
    if not symbols or not basis or not method:
        return None
    try:
        charge = int(request.get('charge') or 0)
        spin = int(request.get('spin') or 0)
    except (TypeError, ValueError):
        return None
    contract = {
        'atom_symbols': symbols,
        'basis': basis,
        'charge': charge,
        'spin': spin,
        'restricted': _effective_restricted_reference(request),
    }
    if include_method:
        contract['method'] = method
    return contract


def _one_particle_state_artifact(case_report: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    task_report = case_report.get('task_report') if isinstance(case_report, dict) else None
    artifacts = task_report.get('artifacts') if isinstance(task_report, dict) else None
    if not isinstance(artifacts, list):
        return None
    for artifact in artifacts:
        if not isinstance(artifact, dict):
            continue
        if artifact.get('kind') == 'one_particle_state' and artifact.get('path'):
            return copy.deepcopy(artifact)
    return None


def _case_request_from_report(case_report: Dict[str, Any], fallback: StudyCase) -> Dict[str, Any]:
    task_report = case_report.get('task_report') if isinstance(case_report, dict) else None
    executed_spec = task_report.get('task_spec') if isinstance(task_report, dict) else None
    if isinstance(executed_spec, dict):
        system = executed_spec.get('system') if isinstance(executed_spec.get('system'), dict) else {}
        method = executed_spec.get('method') if isinstance(executed_spec.get('method'), dict) else {}
        return {
            'atom': system.get('atom'),
            'basis': system.get('basis'),
            'charge': system.get('charge'),
            'spin': system.get('spin'),
            'method': method.get('name'),
            'restricted': method.get('restricted'),
        }
    return fallback.request if isinstance(fallback.request, dict) else {}


def _case_coordinate(case: StudyCase, coordinate: Optional[str]) -> Optional[float]:
    if not coordinate or not isinstance(case.variables, dict):
        return None
    try:
        value = float(case.variables.get(coordinate))
    except (TypeError, ValueError):
        return None
    return value


PATH_WINDOW_RESTART_ENERGY_TOLERANCE_HA = 1.0e-4
PATH_WINDOW_RESTART_REFERENCE_TOLERANCE_HA = 1.0e-5
PATH_WINDOW_RESTART_SPIN_TOLERANCE = 2.0e-2


def _path_window_case_ids(anomaly: Dict[str, Any]) -> List[str]:
    """Return the local overlap window associated with one path anomaly."""
    ordered: List[str] = []
    transition = anomaly.get('transition') if isinstance(anomaly.get('transition'), dict) else {}
    for case_id in (
        transition.get('left_case_id'),
        *(anomaly.get('target_case_ids') or []),
        transition.get('right_case_id'),
    ):
        normalized = str(case_id or '').strip()
        if normalized and normalized not in ordered:
            ordered.append(normalized)
    return ordered


def _ordered_path_window_case_ids(
    case_ids: List[str],
    cases_by_id: Dict[str, StudyCase],
    coordinate: Optional[str],
) -> List[str]:
    unique_ids = []
    for case_id in case_ids:
        normalized = str(case_id or '').strip()
        if normalized and normalized not in unique_ids:
            unique_ids.append(normalized)
    if not coordinate:
        return unique_ids
    positions = {
        case_id: _case_coordinate(cases_by_id[case_id], coordinate)
        for case_id in unique_ids
        if case_id in cases_by_id
    }
    return sorted(
        unique_ids,
        key=lambda case_id: (
            positions.get(case_id) is None,
            positions.get(case_id) if positions.get(case_id) is not None else 0.0,
            case_id,
        ),
    )


def _path_restart_anchor(
    target_case: StudyCase,
    target_row: Dict[str, Any],
    cases_by_id: Dict[str, StudyCase],
    rows_by_id: Dict[str, Dict[str, Any]],
    reports_by_id: Dict[str, Dict[str, Any]],
    coordinate: Optional[str],
    excluded_case_ids: set[str],
    *,
    direction: str,
) -> Optional[Dict[str, Any]]:
    """Find a compatible, trusted source outside a local path window.

    The saved density is an SCF reference 1RDM rather than a correlated wave
    function.  It can therefore seed the *same target method* from either side
    of a method boundary, provided the molecular/reference contract matches.
    """
    target_request = _case_request_from_report(
        reports_by_id.get(target_case.case_id) or {},
        target_case,
    )
    target_contract = _continuation_contract(target_request, include_method=False)
    target_coordinate = _case_coordinate(target_case, coordinate)
    target_method = str(target_row.get('method') or target_request.get('method') or '').strip().lower()
    target_reference = _reference_label(target_request)
    if not target_contract or target_coordinate is None or not target_method:
        return None

    candidates: List[Dict[str, Any]] = []
    for source_case_id, source_case in cases_by_id.items():
        if source_case_id == target_case.case_id or source_case_id in excluded_case_ids:
            continue
        source_row = rows_by_id.get(source_case_id) or {}
        if str(source_row.get('status') or '').strip().lower() != 'succeeded':
            continue
        source_coordinate = _case_coordinate(source_case, coordinate)
        if source_coordinate is None:
            continue
        if direction == 'left' and source_coordinate >= target_coordinate - 1.0e-12:
            continue
        if direction == 'right' and source_coordinate <= target_coordinate + 1.0e-12:
            continue
        source_request = _case_request_from_report(reports_by_id.get(source_case_id) or {}, source_case)
        source_contract = _continuation_contract(source_request, include_method=False)
        if source_contract != target_contract:
            continue
        artifact = _one_particle_state_artifact(reports_by_id.get(source_case_id) or {})
        if not artifact:
            continue
        candidates.append({
            'case_id': source_case_id,
            'coordinate': source_coordinate,
            'method': str(source_row.get('method') or source_request.get('method') or '').strip().lower(),
            'reference': _reference_label(source_request),
            'target_method': target_method,
            'target_reference': target_reference,
            'artifact': artifact,
        })
    if not candidates:
        return None
    if direction == 'left':
        candidates.sort(key=lambda item: (-float(item['coordinate']), str(item['case_id'])))
    else:
        candidates.sort(key=lambda item: (float(item['coordinate']), str(item['case_id'])))
    return candidates[0]


def _path_restart_case(
    case: StudyCase,
    source: Dict[str, Any],
) -> StudyCase:
    request = copy.deepcopy(case.request)
    request['initial_state'] = {
        'mode': 'projected_1rdm',
        'source_case_id': source['case_id'],
        'source_artifact': copy.deepcopy(source['artifact']),
    }
    return StudyCase(
        case_id=case.case_id,
        label=case.label,
        request=request,
        variables=copy.deepcopy(case.variables),
        operations=copy.deepcopy(case.operations),
        model_spec=copy.deepcopy(case.model_spec),
    )


def _path_restart_self_source(
    case: StudyCase,
    row: Dict[str, Any],
    reports_by_id: Dict[str, Dict[str, Any]],
    coordinate: Optional[str],
) -> Optional[Dict[str, Any]]:
    artifact = _one_particle_state_artifact(reports_by_id.get(case.case_id) or {})
    request = _case_request_from_report(reports_by_id.get(case.case_id) or {}, case)
    position = _case_coordinate(case, coordinate)
    method = str(row.get('method') or request.get('method') or '').strip().lower()
    if not artifact or position is None or not method:
        return None
    return {
        'case_id': case.case_id,
        'coordinate': position,
        'method': method,
        'reference': _reference_label(request),
        'target_method': method,
        'target_reference': _reference_label(request),
        'artifact': artifact,
        'source_mode': 'endpoint_baseline',
    }


def _path_restart_deferred_source(
    source_case: StudyCase,
    source_row: Dict[str, Any],
    reports_by_id: Dict[str, Dict[str, Any]],
    coordinate: Optional[str],
) -> Optional[Dict[str, Any]]:
    """Describe a 1RDM that the preceding case will produce in this branch."""
    request = _case_request_from_report(reports_by_id.get(source_case.case_id) or {}, source_case)
    position = _case_coordinate(source_case, coordinate)
    method = str(source_row.get('method') or request.get('method') or '').strip().lower()
    if position is None or not method:
        return None
    return {
        'case_id': source_case.case_id,
        'coordinate': position,
        'method': method,
        'reference': _reference_label(request),
        'target_method': method,
        'target_reference': _reference_label(request),
        'artifact': {
            'kind': 'one_particle_state',
            'deferred_case_id': source_case.case_id,
        },
        'source_mode': 'branch_continuation',
    }


def _path_restart_approval_branch(
    item: Dict[str, Any],
    direction: str,
) -> Dict[str, Any]:
    source_artifact = copy.deepcopy(item.get('{0}_source_artifact'.format(direction)) or {})
    return {
        'direction': 'left_to_right' if direction == 'left' else 'right_to_left',
        'target_case_id': item.get('case_id'),
        'target_method': item.get('target_method'),
        'target_reference': item.get('target_reference'),
        'source_case_id': item.get('{0}_source_case_id'.format(direction)),
        'source_coordinate': item.get('{0}_source_coordinate'.format(direction)),
        'source_method': item.get('{0}_source_method'.format(direction)),
        'source_reference': item.get('{0}_source_reference'.format(direction)),
        'source_mode': item.get('{0}_source_mode'.format(direction)),
        'initial_state_kind': item.get('initial_state_kind'),
        'source_artifact': source_artifact,
    }


def path_window_restart_approval(
    restart_decisions: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Build the review contract for explicit bidirectional continuation."""
    review_items = [
        item
        for item in restart_decisions
        if isinstance(item, dict) and item.get('approval_required')
    ]
    if not review_items:
        return None
    contract = [
        {
            key: item.get(key)
            for key in (
                'case_id',
                'target_method',
                'target_reference',
                'window_case_ids',
                'left_source_case_id',
                'left_source_coordinate',
                'left_source_method',
                'left_source_reference',
                'left_source_mode',
                'right_source_case_id',
                'right_source_coordinate',
                'right_source_method',
                'right_source_reference',
                'right_source_mode',
                'left_source_artifact',
                'right_source_artifact',
                'approval_reason',
            )
        }
        for item in review_items
    ]
    token = hashlib.sha256(
        json.dumps(contract, ensure_ascii=False, sort_keys=True, default=json_default, allow_nan=False).encode('utf-8')
    ).hexdigest()
    return {
        'schema': 'pyscf-agent.path-restart-approval.v1',
        'kind': 'path_restart_approval',
        'status': 'approval_required',
        'approval_token': token,
        'case_ids': [str(item.get('case_id')) for item in review_items],
        'items': [
            {
                'case_id': item.get('case_id'),
                'label': item.get('label'),
                'variables': copy.deepcopy(item.get('variables') or {}),
                'target_method': item.get('target_method'),
                'target_reference': item.get('target_reference'),
                'window_case_ids': copy.deepcopy(item.get('window_case_ids') or []),
                'left_source_case_id': item.get('left_source_case_id'),
                'left_source_coordinate': item.get('left_source_coordinate'),
                'left_source_method': item.get('left_source_method'),
                'left_source_reference': item.get('left_source_reference'),
                'left_source_mode': item.get('left_source_mode'),
                'right_source_case_id': item.get('right_source_case_id'),
                'right_source_coordinate': item.get('right_source_coordinate'),
                'right_source_method': item.get('right_source_method'),
                'right_source_reference': item.get('right_source_reference'),
                'right_source_mode': item.get('right_source_mode'),
                'left_source_artifact': copy.deepcopy(item.get('left_source_artifact') or {}),
                'right_source_artifact': copy.deepcopy(item.get('right_source_artifact') or {}),
                'branches': [
                    _path_restart_approval_branch(item, 'left'),
                    _path_restart_approval_branch(item, 'right'),
                ],
                'cross_method_restart': bool(item.get('cross_method_restart')),
                'cross_reference_restart': bool(item.get('cross_reference_restart')),
                'approval_reason': item.get('approval_reason'),
                'initial_state_kind': item.get('initial_state_kind'),
            }
            for item in review_items
        ],
        'summary': (
            'The local window will be recomputed in both coordinate directions by '
            'passing the converged SCF-reference AO 1RDM between adjacent tasks. '
            'Review the exact seed and propagation chain before execution.'
        ),
    }


def build_path_window_restart_plans(
    refined_plan: StudyPlan,
    refined_report_payload: Dict[str, Any],
    decision_log: List[Dict[str, Any]],
    path_diagnostics: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build independent left/right 1RDM restart candidates for a path window.

    A single neighboring result is not trusted near a method boundary.  Each
    target is re-run from two anchors outside the flagged local window.  The
    executor keeps both branch reports and accepts a replacement only after
    ``evaluate_path_window_restarts`` finds them consistent.
    """
    del decision_log  # The restart contract is defined by executed reports.
    if refined_plan.system_type != 'molecular':
        return None
    anomalies = path_diagnostics.get('anomalies') if isinstance(path_diagnostics, dict) else None
    coordinate = path_diagnostics.get('coordinate') if isinstance(path_diagnostics, dict) else None
    if not isinstance(anomalies, list) or not anomalies or not coordinate:
        return None

    cases_by_id = {case.case_id: case for case in refined_plan.cases}
    rows_by_id = _comparison_rows_by_id(refined_report_payload)
    reports_by_id = _case_reports_by_id(refined_report_payload)
    branch_cases: Dict[str, List[StudyCase]] = {'left': [], 'right': []}
    branch_decisions: List[Dict[str, Any]] = []
    seen_case_ids = set()

    for anomaly in anomalies:
        if not isinstance(anomaly, dict):
            continue
        window_case_ids = _ordered_path_window_case_ids(
            _path_window_case_ids(anomaly),
            cases_by_id,
            str(coordinate),
        )
        target_case_ids = _ordered_path_window_case_ids(
            [
                str(case_id)
                for case_id in anomaly.get('target_case_ids') or []
                if str(case_id) not in seen_case_ids
            ],
            cases_by_id,
            str(coordinate),
        )
        target_case_ids = [
            case_id
            for case_id in target_case_ids
            if (
                case_id in cases_by_id
                and str((rows_by_id.get(case_id) or {}).get('status') or '').strip().lower() == 'succeeded'
            )
        ]
        if not window_case_ids or not target_case_ids:
            continue

        excluded_case_ids = set(window_case_ids)
        endpoint_side = str(anomaly.get('endpoint_side') or '').strip().lower()
        local_cases: Dict[str, List[StudyCase]] = {'left': [], 'right': []}
        sources_by_direction: Dict[str, Dict[str, Dict[str, Any]]] = {'left': {}, 'right': {}}
        branch_complete = True
        for direction in ('left', 'right'):
            ordered_targets = target_case_ids if direction == 'left' else list(reversed(target_case_ids))
            first_case_id = ordered_targets[0]
            first_case = cases_by_id[first_case_id]
            first_row = rows_by_id.get(first_case_id) or {}
            source = _path_restart_anchor(
                first_case,
                first_row,
                cases_by_id,
                rows_by_id,
                reports_by_id,
                str(coordinate),
                excluded_case_ids,
                direction=direction,
            )
            touches_endpoint = (
                (endpoint_side == 'left' and direction == 'left')
                or (endpoint_side == 'right' and direction == 'right')
            )
            if not source and touches_endpoint:
                source = _path_restart_self_source(
                    first_case,
                    first_row,
                    reports_by_id,
                    str(coordinate),
                )
            if not source:
                branch_complete = False
                break

            previous_case: Optional[StudyCase] = None
            for target_case_id in ordered_targets:
                target_case = cases_by_id[target_case_id]
                target_row = rows_by_id.get(target_case_id) or {}
                if previous_case is not None:
                    source = _path_restart_deferred_source(
                        previous_case,
                        rows_by_id.get(previous_case.case_id) or {},
                        reports_by_id,
                        str(coordinate),
                    )
                if not source:
                    branch_complete = False
                    break
                local_cases[direction].append(_path_restart_case(target_case, source))
                sources_by_direction[direction][target_case_id] = source
                previous_case = target_case
            if not branch_complete:
                break
        if not branch_complete:
            continue

        local_decisions: List[Dict[str, Any]] = []
        for case_id in target_case_ids:
            case = cases_by_id[case_id]
            row = rows_by_id.get(case_id) or {}
            left_source = sources_by_direction['left'][case_id]
            right_source = sources_by_direction['right'][case_id]
            target_method = str(row.get('method') or case.request.get('method') or '').strip().lower()
            target_reference = _reference_label(
                _case_request_from_report(reports_by_id.get(case.case_id) or {}, case)
            )
            cross_method_restart = any(
                str(source.get('method') or '').strip().lower() != target_method
                for source in (left_source, right_source)
            )
            cross_reference_restart = any(
                str(source.get('reference') or '').strip().lower() != target_reference
                for source in (left_source, right_source)
            )

            def artifact_summary(source: Dict[str, Any]) -> Dict[str, Any]:
                artifact = source.get('artifact') if isinstance(source.get('artifact'), dict) else {}
                return {
                    key: artifact.get(key)
                    for key in ('kind', 'path', 'deferred_case_id')
                    if artifact.get(key) is not None
                }

            local_decisions.append({
                'case_id': case.case_id,
                'label': case.label,
                'variables': copy.deepcopy(case.variables),
                'method': row.get('method') or case.request.get('method'),
                'target_method': target_method,
                'target_reference': target_reference,
                'initial_state_kind': 'scf_reference_1rdm',
                'approval_required': False,
                'cross_method_restart': cross_method_restart,
                'cross_reference_restart': cross_reference_restart,
                'restart_action': 'bidirectional_projected_1rdm',
                'path_anomaly_id': anomaly.get('id'),
                'path_anomaly_kind': anomaly.get('kind'),
                'path_anomaly_reason': anomaly.get('reason'),
                'path_transition': copy.deepcopy(anomaly.get('transition') or {}),
                'window_case_ids': window_case_ids,
                'left_source_case_id': left_source['case_id'],
                'left_source_coordinate': left_source['coordinate'],
                'left_source_method': left_source['method'],
                'left_source_reference': left_source['reference'],
                'left_source_mode': left_source.get('source_mode') or 'external_anchor',
                'left_source_artifact': artifact_summary(left_source),
                'right_source_case_id': right_source['case_id'],
                'right_source_coordinate': right_source['coordinate'],
                'right_source_method': right_source['method'],
                'right_source_reference': right_source['reference'],
                'right_source_mode': right_source.get('source_mode') or 'external_anchor',
                'right_source_artifact': artifact_summary(right_source),
                'tags': [
                    'path_continuity_anomaly',
                    'bidirectional_restart',
                    'projected_1rdm',
                    'window_validation',
                ],
                'reason': (
                    'Recompute {0} with the same target method while propagating the AO-projected '
                    'SCF-reference 1RDM through adjacent cases from both coordinate directions. '
                    'Accept a replacement only when the branches agree.'
                ).format(str(row.get('method') or case.request.get('method') or 'method').upper()),
                'next_step': (
                    'Compare the two same-method continuation branches. If they disagree or the path '
                    'remains non-smooth, review a unified local CASSCF ActiveSpaceAudit.'
                ),
            })

        anomaly_requires_approval = (
            anomaly.get('kind') == 'endpoint_continuity_risk'
            or any(
                item.get('cross_method_restart') or item.get('cross_reference_restart')
                for item in local_decisions
            )
        )
        approval_reason = (
            'endpoint_bidirectional_validation'
            if anomaly.get('kind') == 'endpoint_continuity_risk'
            else 'cross_method_or_reference_1rdm'
        )
        for item in local_decisions:
            item['approval_required'] = bool(anomaly_requires_approval)
            item['approval_reason'] = approval_reason

        branch_cases['left'].extend(local_cases['left'])
        branch_cases['right'].extend(local_cases['right'])
        branch_decisions.extend(local_decisions)
        seen_case_ids.update(target_case_ids)

    if not branch_decisions:
        return None
    plans: Dict[str, Dict[str, Any]] = {}
    for direction in ('left', 'right'):
        restart_plan = StudyPlan(
            study_id='path-window-restart-{0}'.format(direction),
            name='{0}-path-window-restart-{1}'.format(refined_plan.name, direction),
            objective='{0}-to-{1} adjacent-1RDM continuity retry for {2}'.format(
                direction,
                'right' if direction == 'left' else 'left',
                refined_plan.objective,
            ),
            system_type=refined_plan.system_type,
            cases=branch_cases[direction],
            observables=copy.deepcopy(refined_plan.observables),
            comparison=copy.deepcopy(refined_plan.comparison),
            capability_snapshot=copy.deepcopy(refined_plan.capability_snapshot),
            resource_policy=copy.deepcopy(refined_plan.resource_policy),
        )
        ensure_plan_cost_estimate(restart_plan)
        plans[direction] = restart_plan.to_dict()
    approval = path_window_restart_approval(branch_decisions)
    return {
        'path_window_restart_plans': plans,
        'path_window_restart_decisions': branch_decisions,
        'approval_required': approval is not None,
        'path_window_restart_approval': approval,
        'adaptive_options': _normalize_options(options),
    }


def _case_structured_results(case_report: Dict[str, Any]) -> Dict[str, Any]:
    task_report = case_report.get('task_report') if isinstance(case_report, dict) else None
    structured = task_report.get('structured_results') if isinstance(task_report, dict) else None
    return structured if isinstance(structured, dict) else {}


def _spin_square_from_case(case_report: Dict[str, Any]) -> Optional[float]:
    diagnostics = _case_structured_results(case_report).get('correlation_diagnostics')
    if not isinstance(diagnostics, dict):
        return None
    direct = _numeric_value(diagnostics.get('spin_square'))
    if direct is not None:
        return direct
    symmetry = diagnostics.get('reference_spin_symmetry')
    return _numeric_value(symmetry.get('spin_square')) if isinstance(symmetry, dict) else None


def _scf_stability_state(case_report: Dict[str, Any]) -> Optional[bool]:
    stability = _case_structured_results(case_report).get('scf_stability')
    if not isinstance(stability, dict):
        return None
    for key in ('stable', 'is_stable', 'internal_stable'):
        if stability.get(key) is not None:
            return bool(stability.get(key))
    return None


def _branch_quality_summary(
    row: Dict[str, Any],
    case_report: Dict[str, Any],
) -> Dict[str, Any]:
    structured = _case_structured_results(case_report)
    return {
        'status': row.get('status'),
        'final_energy': _numeric_value(row.get('final_energy') or row.get('energy')),
        'reference_energy': _numeric_value(structured.get('reference_energy')),
        'spin_square': _spin_square_from_case(case_report),
        'scf_converged': structured.get('converged'),
        'scf_stable': _scf_stability_state(case_report),
    }


def evaluate_path_window_restarts(
    left_report_payload: Dict[str, Any],
    right_report_payload: Dict[str, Any],
    restart_decisions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Decide whether independently restarted candidates agree sufficiently."""
    left_rows = _comparison_rows_by_id(left_report_payload)
    right_rows = _comparison_rows_by_id(right_report_payload)
    left_cases = _case_reports_by_id(left_report_payload)
    right_cases = _case_reports_by_id(right_report_payload)
    comparisons: List[Dict[str, Any]] = []

    for decision in restart_decisions:
        if not isinstance(decision, dict):
            continue
        case_id = str(decision.get('case_id') or '')
        left_row = left_rows.get(case_id) or {}
        right_row = right_rows.get(case_id) or {}
        left_quality = _branch_quality_summary(left_row, left_cases.get(case_id) or {})
        right_quality = _branch_quality_summary(right_row, right_cases.get(case_id) or {})
        checks: List[Dict[str, Any]] = []
        both_succeeded = (
            str(left_quality.get('status') or '').strip().lower() == 'succeeded'
            and str(right_quality.get('status') or '').strip().lower() == 'succeeded'
        )
        checks.append({'name': 'both_succeeded', 'passed': both_succeeded})

        left_energy = left_quality.get('final_energy')
        right_energy = right_quality.get('final_energy')
        energy_difference = None if left_energy is None or right_energy is None else abs(left_energy - right_energy)
        checks.append({
            'name': 'final_energy_agreement',
            'passed': energy_difference is not None and energy_difference <= PATH_WINDOW_RESTART_ENERGY_TOLERANCE_HA,
            'difference_ha': energy_difference,
            'tolerance_ha': PATH_WINDOW_RESTART_ENERGY_TOLERANCE_HA,
        })

        left_reference = left_quality.get('reference_energy')
        right_reference = right_quality.get('reference_energy')
        reference_difference = None if left_reference is None or right_reference is None else abs(left_reference - right_reference)
        if reference_difference is not None:
            checks.append({
                'name': 'reference_energy_agreement',
                'passed': reference_difference <= PATH_WINDOW_RESTART_REFERENCE_TOLERANCE_HA,
                'difference_ha': reference_difference,
                'tolerance_ha': PATH_WINDOW_RESTART_REFERENCE_TOLERANCE_HA,
            })

        left_spin = left_quality.get('spin_square')
        right_spin = right_quality.get('spin_square')
        spin_difference = None if left_spin is None or right_spin is None else abs(left_spin - right_spin)
        if spin_difference is not None:
            checks.append({
                'name': 'spin_square_agreement',
                'passed': spin_difference <= PATH_WINDOW_RESTART_SPIN_TOLERANCE,
                'difference': spin_difference,
                'tolerance': PATH_WINDOW_RESTART_SPIN_TOLERANCE,
            })

        left_stability = left_quality.get('scf_stable')
        right_stability = right_quality.get('scf_stable')
        if left_stability is not None and right_stability is not None:
            checks.append({
                'name': 'scf_stability_agreement',
                'passed': bool(left_stability) and bool(right_stability),
            })
        accepted = all(bool(check.get('passed')) for check in checks)
        comparisons.append({
            'case_id': case_id,
            'accepted': accepted,
            'left': left_quality,
            'right': right_quality,
            'checks': checks,
            'reason': (
                'Both anchor branches converged to consistent results.'
                if accepted
                else 'The two anchor branches did not establish a unique same-method result; retain the original point and promote the full local window for review.'
            ),
        })

    accepted_count = sum(1 for item in comparisons if item.get('accepted'))
    return {
        'schema': 'pyscf-agent.path-window-restart-validation.v1',
        'kind': 'path_window_restart_validation',
        'status': 'validated' if comparisons and accepted_count == len(comparisons) else 'review_required',
        'accepted_case_ids': [item['case_id'] for item in comparisons if item.get('accepted')],
        'unresolved_case_ids': [item['case_id'] for item in comparisons if not item.get('accepted')],
        'comparisons': comparisons,
        'summary': (
            'Bidirectional restart validated {0} of {1} local path candidate(s).'
        ).format(accepted_count, len(comparisons)),
    }


def merge_path_window_restart_reports(
    refined_report_payload: Dict[str, Any],
    left_report_payload: Dict[str, Any],
    right_report_payload: Dict[str, Any],
    decision_log: List[Dict[str, Any]],
    restart_decisions: List[Dict[str, Any]],
    validation: Dict[str, Any],
) -> Dict[str, Any]:
    """Merge only validated path candidates while retaining both branch records."""
    merged = copy.deepcopy(refined_report_payload)
    left_rows = _comparison_rows_by_id(left_report_payload)
    right_rows = _comparison_rows_by_id(right_report_payload)
    left_cases = _case_reports_by_id(left_report_payload)
    right_cases = _case_reports_by_id(right_report_payload)
    decisions_by_id = {
        str(item.get('case_id')): item
        for item in restart_decisions
        if isinstance(item, dict) and item.get('case_id') is not None
    }
    validation_by_id = {
        str(item.get('case_id')): item
        for item in validation.get('comparisons') or []
        if isinstance(item, dict) and item.get('case_id') is not None
    }

    merged_rows: List[Dict[str, Any]] = []
    for row in merged.get('comparison_table') or []:
        if not isinstance(row, dict):
            merged_rows.append(row)
            continue
        case_id = str(row.get('case_id') or '')
        comparison = validation_by_id.get(case_id)
        if not comparison:
            merged_rows.append(row)
            continue
        decision = decisions_by_id.get(case_id, {})
        candidate_summary = {
            'original_energy': row.get('final_energy') or row.get('energy'),
            'left_energy': (left_rows.get(case_id) or {}).get('final_energy') or (left_rows.get(case_id) or {}).get('energy'),
            'right_energy': (right_rows.get(case_id) or {}).get('final_energy') or (right_rows.get(case_id) or {}).get('energy'),
            'left_source_case_id': decision.get('left_source_case_id'),
            'right_source_case_id': decision.get('right_source_case_id'),
            'checks': copy.deepcopy(comparison.get('checks') or []),
        }
        if comparison.get('accepted'):
            canonical = copy.deepcopy(left_rows.get(case_id) or row)
            canonical['path_restart_applied'] = True
            canonical['path_restart_validation'] = 'bidirectional_validated'
            canonical['path_restart_anomaly_id'] = decision.get('path_anomaly_id')
            canonical['path_restart_anomaly_kind'] = decision.get('path_anomaly_kind')
            canonical['path_restart_endpoint_side'] = (
                decision.get('path_transition', {}).get('endpoint_side')
                if isinstance(decision.get('path_transition'), dict)
                else None
            )
            canonical['path_restart_candidates'] = candidate_summary
            canonical['path_restart_original_energy'] = candidate_summary['original_energy']
            merged_rows.append(canonical)
        else:
            retained = copy.deepcopy(row)
            retained['path_restart_applied'] = False
            retained['path_restart_validation'] = 'inconclusive'
            retained['path_restart_candidates'] = candidate_summary
            retained['path_review'] = 'required'
            retained['path_recommendation'] = comparison.get('reason')
            merged_rows.append(retained)
    merged['comparison_table'] = merged_rows

    merged_cases: List[Dict[str, Any]] = []
    for item in merged.get('cases') or []:
        if not isinstance(item, dict):
            merged_cases.append(item)
            continue
        case_id = str(item.get('case_id') or '')
        comparison = validation_by_id.get(case_id)
        if not comparison:
            merged_cases.append(item)
            continue
        candidate_reports = {
            'original_task_report': copy.deepcopy(item.get('task_report')),
            'left_task_report': copy.deepcopy((left_cases.get(case_id) or {}).get('task_report')),
            'right_task_report': copy.deepcopy((right_cases.get(case_id) or {}).get('task_report')),
        }
        if comparison.get('accepted'):
            canonical = copy.deepcopy(left_cases.get(case_id) or item)
            canonical['pre_path_restart_task_report'] = candidate_reports['original_task_report']
            canonical['path_restart_branch_reports'] = candidate_reports
            canonical['path_restart_validation'] = 'bidirectional_validated'
            canonical['path_restart_anomaly_id'] = (
                decisions_by_id.get(case_id, {}).get('path_anomaly_id')
            )
            canonical['path_restart_anomaly_kind'] = (
                decisions_by_id.get(case_id, {}).get('path_anomaly_kind')
            )
            merged_cases.append(canonical)
        else:
            retained = copy.deepcopy(item)
            retained['path_restart_branch_reports'] = candidate_reports
            retained['path_restart_validation'] = 'inconclusive'
            merged_cases.append(retained)
    merged['cases'] = merged_cases
    merged['path_window_restart_validation'] = copy.deepcopy(validation)

    for decision in decision_log:
        if not isinstance(decision, dict):
            continue
        case_id = str(decision.get('case_id') or '')
        comparison = validation_by_id.get(case_id)
        restart_decision = decisions_by_id.get(case_id)
        if not comparison or not restart_decision:
            continue
        decision['path_restart_applied'] = bool(comparison.get('accepted'))
        decision['path_restart_action'] = 'bidirectional_projected_1rdm'
        decision['path_restart_validation'] = 'validated' if comparison.get('accepted') else 'inconclusive'
        decision['path_restart_left_source_case_id'] = restart_decision.get('left_source_case_id')
        decision['path_restart_right_source_case_id'] = restart_decision.get('right_source_case_id')
        decision['path_restart_anomaly_id'] = restart_decision.get('path_anomaly_id')
        decision['path_restart_anomaly_kind'] = restart_decision.get('path_anomaly_kind')
        decision['path_restart_reason'] = comparison.get('reason')
        tags = list(decision.get('tags') or [])
        for tag in ('path_continuity_anomaly', 'bidirectional_restart', 'projected_1rdm', 'window_validation'):
            if tag not in tags:
                tags.append(tag)
        if not comparison.get('accepted') and 'path_restart_inconclusive' not in tags:
            tags.append('path_restart_inconclusive')
        decision['tags'] = tags
    merged['status'] = (
        'succeeded'
        if all(row.get('status') == 'succeeded' for row in merged_rows if isinstance(row, dict))
        else 'completed_with_issues'
    )
    return merged
def _active_space_electron_count(nelecas: Any) -> Optional[int]:
    if isinstance(nelecas, (list, tuple)):
        try:
            return sum(int(value) for value in nelecas)
        except (TypeError, ValueError):
            return None
    try:
        return int(nelecas)
    except (TypeError, ValueError):
        return None


def _valid_active_space_candidate(active_space: Any) -> bool:
    if not _active_space_is_candidate(active_space):
        return False
    try:
        ncas = int(active_space.get('ncas'))
    except (AttributeError, TypeError, ValueError):
        return False
    active_electrons = _active_space_electron_count(active_space.get('nelecas'))
    return bool(ncas > 0 and active_electrons is not None and 0 < active_electrons <= 2 * ncas)


def _window_active_space_template(
    candidates: List[Dict[str, Any]],
) -> Optional[Dict[str, Any]]:
    """Pick a valid, largest CAS dimension to keep a local window comparable."""
    valid = [copy.deepcopy(candidate) for candidate in candidates if _valid_active_space_candidate(candidate)]
    if not valid:
        return None
    valid.sort(key=lambda candidate: (
        int(candidate.get('ncas') or 0),
        _active_space_electron_count(candidate.get('nelecas')) or 0,
    ), reverse=True)
    return valid[0]


def _window_orbital_indices(
    case: StudyCase,
    candidate: Optional[Dict[str, Any]],
    template: Dict[str, Any],
) -> List[int]:
    del case
    target_ncas = int(template.get('ncas') or 0)
    if target_ncas <= 0 or not isinstance(candidate, dict):
        return []
    source_indices = candidate.get('orbital_indices')
    indices: List[int] = []
    if isinstance(source_indices, list):
        for value in source_indices:
            try:
                index = int(value)
            except (TypeError, ValueError):
                continue
            if index >= 0 and index not in indices:
                indices.append(index)
    # Orbital identities are geometry-specific.  A neighboring case may
    # define the shared CAS dimension, but its canonical indices must never be
    # copied or used to fabricate a frontier window for this case.
    return sorted(indices) if len(indices) == target_ncas else []


def _window_active_space_for_case(
    case: StudyCase,
    candidate: Optional[Dict[str, Any]],
    template: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    indices = _window_orbital_indices(case, candidate, template)
    target_ncas = int(template.get('ncas') or 0)
    if len(indices) != target_ncas:
        return None
    active_space = copy.deepcopy(candidate) if isinstance(candidate, dict) else {}
    active_space.update({
        'enabled': True,
        'selection_method': 'manual',
        'ncas': target_ncas,
        'nelecas': copy.deepcopy(template.get('nelecas')),
        'orbital_indices': indices,
        'approved': False,
        'selection_reason': 'Local path-window consistency uses a shared CAS size and electron count across the overlap region.',
    })
    return active_space if _valid_active_space_candidate(active_space) else None


def build_path_refinement_plan(
    refined_plan: StudyPlan,
    refined_report_payload: Dict[str, Any],
    decision_log: List[Dict[str, Any]],
    path_diagnostics: Dict[str, Any],
    options: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    """Build a unified local CASSCF review plan for non-smooth molecular scans.

    The path diagnostic does not reinterpret a single point as a phase label. It
    identifies a discontinuity at a method boundary, then promotes both sides
    of the local overlap window.  This prevents a suspect point from being
    judged solely against a potentially unreliable neighboring method branch.
    """
    if refined_plan.system_type != 'molecular':
        return None
    anomalies = path_diagnostics.get('anomalies') if isinstance(path_diagnostics, dict) else None
    if not isinstance(anomalies, list) or not anomalies:
        return None

    adaptive_options = _normalize_options(options)
    cases_by_id = {case.case_id: case for case in refined_plan.cases}
    rows_by_id = _comparison_rows_by_id(refined_report_payload)
    decisions_by_id = {
        str(item.get('case_id')): item
        for item in decision_log
        if isinstance(item, dict) and item.get('case_id') is not None
    }
    observables = list(refined_plan.observables or ['energy'])
    path_cases: List[StudyCase] = []
    path_decisions: List[Dict[str, Any]] = []
    seen_case_ids = set()
    active_space_probe_case_ids = set()
    coordinate = path_diagnostics.get('coordinate') if isinstance(path_diagnostics, dict) else None

    for anomaly in anomalies:
        if not isinstance(anomaly, dict):
            continue
        window_case_ids = _ordered_path_window_case_ids(
            _path_window_case_ids(anomaly),
            cases_by_id,
            str(coordinate) if coordinate else None,
        )
        candidates_by_id: Dict[str, Optional[Dict[str, Any]]] = {}
        template_candidates: List[Dict[str, Any]] = []
        for case_id in window_case_ids:
            normalized_case_id = str(case_id)
            case = cases_by_id.get(normalized_case_id)
            row = rows_by_id.get(normalized_case_id, {})
            if case is None or str(row.get('status') or '').strip().lower() != 'succeeded':
                continue
            decision = copy.deepcopy(decisions_by_id.get(normalized_case_id) or {})
            active_space_candidate = _recovery_active_space_candidate(case, decision, adaptive_options)
            candidates_by_id[normalized_case_id] = active_space_candidate
            if _valid_active_space_candidate(active_space_candidate):
                template_candidates.append(active_space_candidate)
        template = _window_active_space_template(template_candidates)
        if not template:
            continue

        for case_id in window_case_ids:
            normalized_case_id = str(case_id)
            if normalized_case_id in seen_case_ids:
                continue
            case = cases_by_id.get(normalized_case_id)
            row = rows_by_id.get(normalized_case_id, {})
            if case is None or str(row.get('status') or '').strip().lower() != 'succeeded':
                continue
            decision = copy.deepcopy(decisions_by_id.get(normalized_case_id) or {})
            active_space = _window_active_space_for_case(
                case,
                candidates_by_id.get(normalized_case_id),
                template,
            )
            if not active_space:
                active_space_probe_case_ids.add(normalized_case_id)
                continue
            request = _runtime_recovery_request(_molecular_recovery_request(
                case,
                'casscf',
                decision,
                observables,
                active_space,
                adaptive_options,
            ))
            reviewed_active_space = request.get('active_space') if isinstance(request.get('active_space'), dict) else None
            if not reviewed_active_space:
                continue
            reviewed_active_space['approved'] = False
            reviewed_active_space['enabled'] = True
            reviewed_active_space['selection_method'] = 'manual'
            request['active_space'] = reviewed_active_space
            current_method = row.get('method') or case.request.get('method') or 'unknown'
            path_cases.append(StudyCase(
                case_id=case.case_id,
                label=case.label,
                request=request,
                variables=copy.deepcopy(case.variables),
                operations=copy.deepcopy(case.operations),
                model_spec=copy.deepcopy(case.model_spec),
            ))
            path_decisions.append({
                'case_id': case.case_id,
                'label': case.label,
                'variables': copy.deepcopy(case.variables),
                'initial_method': current_method,
                'diagnostic_level': decision.get('diagnostic_level') or decision.get('level'),
                'physics_level': decision.get('physics_level'),
                'solver_stress_level': decision.get('solver_stress_level'),
                'diagnostics': copy.deepcopy(decision.get('diagnostics') or {}),
                'recommended_method': 'casscf',
                'recommended_solver': 'casscf',
                'active_space_contract': copy.deepcopy(reviewed_active_space),
                'path_anomaly_id': anomaly.get('id'),
                'path_anomaly_reason': anomaly.get('reason'),
                'path_transition': copy.deepcopy(anomaly.get('transition') or {}),
                'path_window_case_ids': window_case_ids,
                'path_window_active_space_template': {
                    'ncas': template.get('ncas'),
                    'nelecas': copy.deepcopy(template.get('nelecas')),
                },
                'tags': _dedupe([
                    'path_continuity_anomaly',
                    'path_window_refinement',
                    'needs_refined_method',
                    'review_active_space',
                    'requires_active_space_approval',
                ]),
                'reason': (
                    'Scan-path continuity review promotes the local overlap window from {0} '
                    'to a shared CAS-size CASSCF treatment. {1}'
                ).format(str(current_method).upper(), anomaly.get('reason') or ''),
                'next_step': 'Approve or edit the shared-size ActiveSpaceAudit batch, then rerun the full local continuity window.',
            })
            seen_case_ids.add(normalized_case_id)

    if not path_cases:
        return None
    path_plan = StudyPlan(
        study_id='path-refinement',
        name='{0}-path-refinement'.format(refined_plan.name),
        objective='Local method-continuity refinement for {0}'.format(refined_plan.objective),
        system_type=refined_plan.system_type,
        cases=path_cases,
        observables=copy.deepcopy(refined_plan.observables),
        comparison=copy.deepcopy(refined_plan.comparison),
        capability_snapshot=copy.deepcopy(refined_plan.capability_snapshot),
        resource_policy=copy.deepcopy(refined_plan.resource_policy),
    )
    ensure_plan_cost_estimate(path_plan)
    return {
        'path_refinement_plan': path_plan.to_dict(),
        'path_refinement_decisions': path_decisions,
        'active_space_probe_case_ids': sorted(active_space_probe_case_ids),
        'adaptive_options': adaptive_options,
    }


def _merge_recovery_report(
    refined_report_payload: Dict[str, Any],
    recovery_report_payload: Dict[str, Any],
    decision_log: List[Dict[str, Any]],
    recovery_decisions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = copy.deepcopy(refined_report_payload)
    recovery_rows = _comparison_rows_by_id(recovery_report_payload)
    recovery_cases = _case_reports_by_id(recovery_report_payload)
    recovery_decision_by_id = {str(item.get('case_id')): item for item in recovery_decisions if isinstance(item, dict)}
    merged_rows = []
    for row in merged.get('comparison_table') or []:
        row_case_id = str(row.get('case_id')) if isinstance(row, dict) and row.get('case_id') is not None else None
        if row_case_id in recovery_rows:
            next_row = copy.deepcopy(recovery_rows[row_case_id])
            recovery_decision = recovery_decision_by_id.get(row_case_id, {})
            next_row['initial_status'] = row.get('status')
            next_row['initial_solver'] = row.get('solver')
            next_row['initial_method'] = row.get('method')
            next_row['recovery_applied'] = True
            next_row['recovery_action'] = recovery_decision.get('recovery_action')
            next_row['recovery_reason'] = recovery_decision.get('reason')
            runtime = recovery_decision.get('runtime') if isinstance(recovery_decision.get('runtime'), dict) else {}
            if runtime.get('max_cycle') is not None:
                next_row['recovery_max_cycle'] = runtime.get('max_cycle')
            next_step = _recovery_next_step(
                next_row.get('status'),
                str(recovery_decision.get('recovery_action') or ''),
                recovery_decision.get('recovery_method') or recovery_decision.get('recovery_solver'),
                recovery_decision.get('active_space_contract'),
            )
            if next_step:
                next_row['next_step'] = next_step
            merged_rows.append(next_row)
        else:
            merged_rows.append(row)
    merged['comparison_table'] = merged_rows
    merged_cases = []
    for item in merged.get('cases') or []:
        item_case_id = str(item.get('case_id')) if isinstance(item, dict) and item.get('case_id') is not None else None
        if item_case_id in recovery_cases:
            next_case = copy.deepcopy(recovery_cases[item_case_id])
            next_case['initial_task_report'] = copy.deepcopy(item.get('task_report'))
            next_case['recovery_applied'] = True
            merged_cases.append(next_case)
        else:
            merged_cases.append(item)
    merged['cases'] = merged_cases
    merged['status'] = 'succeeded' if all(row.get('status') == 'succeeded' for row in merged_rows if isinstance(row, dict)) else 'completed_with_issues'
    for decision in decision_log:
        recovery_decision = recovery_decision_by_id.get(str(decision.get('case_id'))) if isinstance(decision, dict) else None
        if not recovery_decision:
            continue
        decision['initial_refined_solver'] = recovery_decision.get('initial_solver')
        decision['initial_refined_method'] = recovery_decision.get('initial_method')
        decision['initial_refined_status'] = recovery_decision.get('initial_status')
        decision['recovery_solver'] = recovery_decision.get('recovery_solver')
        decision['recovery_method'] = recovery_decision.get('recovery_method')
        decision['recovery_action'] = recovery_decision.get('recovery_action')
        decision['recovery_reason'] = recovery_decision.get('reason')
        decision['recovery_runtime'] = copy.deepcopy(recovery_decision.get('runtime') or {})
        decision['recommended_solver'] = recovery_decision.get('recovery_solver') or decision.get('recommended_solver')
        decision['recommended_method'] = recovery_decision.get('recovery_method') or decision.get('recommended_method')
        decision['physics_level'] = recovery_decision.get('physics_level') or decision.get('physics_level') or decision.get('level')
        decision['solver_stress_level'] = recovery_decision.get('solver_stress_level') or decision.get('solver_stress_level')
        decision['diagnostic_level'] = recovery_decision.get('diagnostic_level') or decision.get('diagnostic_level') or decision.get('level')
        decision['level'] = recovery_decision.get('level') or decision.get('level')
        decision['level_reason'] = recovery_decision.get('level_reason') or decision.get('level_reason')
        if recovery_decision.get('correlation_reason'):
            decision['correlation_reason'] = recovery_decision['correlation_reason']
            reasons = list(decision.get('reasons') or [])
            if recovery_decision['correlation_reason'] not in reasons:
                reasons.insert(0, recovery_decision['correlation_reason'])
            decision['reasons'] = reasons[:6]
        recovery_row = recovery_rows.get(str(decision.get('case_id')))
        next_step = _recovery_next_step(
            recovery_row.get('status') if isinstance(recovery_row, dict) else None,
            str(recovery_decision.get('recovery_action') or ''),
            recovery_decision.get('recovery_method') or recovery_decision.get('recovery_solver'),
            recovery_decision.get('active_space_contract'),
        )
        if next_step:
            decision['next_step'] = next_step
        if isinstance(recovery_decision.get('active_space_contract'), dict):
            decision['active_space_contract'] = copy.deepcopy(recovery_decision['active_space_contract'])
        tags = list(decision.get('tags') or [])
        initial_status = str(recovery_decision.get('initial_status') or '').strip().lower()
        initial_status_tag = 'refined_failed' if initial_status == 'failed' else 'refined_unconverged'
        for tag in list(recovery_decision.get('tags') or []) + [initial_status_tag, 'method_recovery_applied']:
            if tag not in tags:
                tags.append(tag)
        if recovery_decision.get('recovery_action') == 'increase_max_cycle' and 'runtime_recovery_applied' not in tags:
            tags.append('runtime_recovery_applied')
        if isinstance(recovery_row, dict) and recovery_row.get('status') != 'succeeded' and 'recovery_unresolved' not in tags:
            tags.append('recovery_unresolved')
        if (
            recovery_decision.get('recovery_method') in ('casci', 'casscf')
            and isinstance(recovery_decision.get('active_space_contract'), dict)
            and not recovery_decision['active_space_contract'].get('approved')
            and 'requires_active_space_approval' not in tags
        ):
            tags.append('requires_active_space_approval')
        decision['tags'] = tags
    return merged
