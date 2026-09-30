from __future__ import annotations

import copy
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from pyscf_agent.lifecycle import ensure_lifecycle, transition_lifecycle
from pyscf_agent.providers.libdmet.recovery import dmet_task_recovery

from ..adaptive.active_space_contracts import active_space_solver_contract_from_request
from ..adaptive.refinement import _runtime_recovery_request
from .actions import gate_action


REVIEW_ACTION_RESULT_SCHEMA = 'pyscf-agent.study-review-action-result.v1'


class StudyReviewActionError(ValueError):
    """Raised when a displayed study review action cannot be applied safely."""


_PLAN_KEY_BY_KIND = {
    'static': 'retry_plan',
    'direct_casscf_review': 'direct_active_space_review_plan',
    'entanglement_active_space': 'entanglement_active_space_plan',
    'recovery': 'recovery_plan',
    'path_refinement': 'path_refinement_plan',
    'refined': 'refined_plan',
    'initial': 'initial_scan_plan',
}


def _case_ids(values: Optional[Sequence[Any]]) -> List[str]:
    result: List[str] = []
    for value in values or ():
        item = str(value or '').strip()
        if item and item not in result:
            result.append(item)
    return result


def _adaptive_container(state: Dict[str, Any]) -> Tuple[Dict[str, Any], bool]:
    adaptive = state.get('adaptive')
    if isinstance(adaptive, dict):
        return adaptive, True
    if any(key in state for key in tuple(_PLAN_KEY_BY_KIND.values()) + ('workflow',)):
        return state, False
    adaptive = {}
    state['adaptive'] = adaptive
    return adaptive, True


def _plan_kind(plan: Optional[Mapping[str, Any]]) -> str:
    if not isinstance(plan, Mapping):
        return 'refined'
    explicit = str(plan.get('_review_kind') or '').strip()
    if explicit:
        return explicit
    study_id = str(plan.get('study_id') or '').strip()
    if study_id == 'casscf-active-space-review':
        return 'direct_casscf_review'
    if study_id == 'entanglement-active-space-review':
        return 'entanglement_active_space'
    if study_id == 'refined-recovery':
        return 'recovery'
    if study_id == 'path-refinement':
        return 'path_refinement'
    if study_id.startswith('initial-scan'):
        return 'initial'
    workflow = plan.get('workflow_configuration')
    nodes = workflow.get('nodes') if isinstance(workflow, Mapping) else []
    if any(
        isinstance(node, Mapping)
        and str(node.get('module_id') or '').strip() == 'study.static_execution'
        for node in (nodes or [])
    ):
        return 'static'
    return 'refined'


def _plan_has_cases(plan: Any, selected: Sequence[str]) -> bool:
    if not isinstance(plan, dict) or not isinstance(plan.get('cases'), list):
        return False
    selected_set = set(selected)
    return any(
        isinstance(item, dict) and str(item.get('case_id')) in selected_set
        for item in plan['cases']
    )


def _plan_candidates(
    adaptive: Mapping[str, Any],
    current_plan: Optional[Mapping[str, Any]],
) -> List[Tuple[str, Dict[str, Any]]]:
    candidates: List[Tuple[str, Dict[str, Any]]] = []
    for kind in (
        'static',
        'direct_casscf_review',
        'entanglement_active_space',
        'recovery',
        'path_refinement',
    ):
        value = adaptive.get(_PLAN_KEY_BY_KIND[kind])
        if isinstance(value, dict):
            candidates.append((kind, value))
    if isinstance(current_plan, Mapping):
        candidates.append((_plan_kind(current_plan), dict(current_plan)))
    value = adaptive.get('refined_plan')
    if isinstance(value, dict):
        candidates.append(('refined', value))
    value = adaptive.get('initial_scan_plan')
    if isinstance(value, dict):
        candidates.append(('initial', value))
    return candidates


def _active_space(plan: Mapping[str, Any], selected: Sequence[str]) -> bool:
    selected_set = set(selected)
    for item in plan.get('cases') or []:
        if not isinstance(item, dict) or str(item.get('case_id')) not in selected_set:
            continue
        request = item.get('request') if isinstance(item.get('request'), dict) else {}
        method = str(request.get('method') or request.get('solver') or '').strip().lower()
        active_space = request.get('active_space') if isinstance(request.get('active_space'), dict) else {}
        if method in ('casci', 'casscf') and active_space.get('enabled') is not False and not active_space.get('approved'):
            return True
    return False


def _source_plan(
    adaptive: Mapping[str, Any],
    current_plan: Optional[Mapping[str, Any]],
    selected: Sequence[str],
    *,
    require_active_space: bool = False,
    requested_kind: Optional[str] = None,
) -> Tuple[str, Dict[str, Any]]:
    candidates = _plan_candidates(adaptive, current_plan)
    if requested_kind:
        candidates.sort(key=lambda item: item[0] != requested_kind)
    for kind, plan in candidates:
        if not _plan_has_cases(plan, selected):
            continue
        if require_active_space and not _active_space(plan, selected):
            continue
        return kind, copy.deepcopy(plan)
    raise StudyReviewActionError('No compatible study plan contains the selected review cases.')


def _selected_plan(
    source: Mapping[str, Any],
    selected: Sequence[str],
    *,
    kind: str,
    label: str,
    study_id: str,
) -> Dict[str, Any]:
    selected_set = set(selected)
    plan = copy.deepcopy(dict(source))
    plan['study_id'] = study_id or source.get('study_id') or 'study'
    plan['_review_kind'] = kind
    plan['name'] = '{0}-{1}'.format(source.get('name') or source.get('study_id') or 'adaptive-plan', label)
    plan['objective'] = '{0} ({1})'.format(source.get('objective') or 'Adaptive recovery', label.replace('-', ' '))
    plan['cases'] = [
        item for item in plan.get('cases') or []
        if isinstance(item, dict) and str(item.get('case_id')) in selected_set
    ]
    return plan


def _merge_plan_cases(base: Optional[Mapping[str, Any]], update: Mapping[str, Any]) -> Dict[str, Any]:
    if not isinstance(base, Mapping) or not isinstance(base.get('cases'), list):
        result = copy.deepcopy(dict(update))
        for key in ('_review_kind',):
            result.pop(key, None)
        return result
    result = copy.deepcopy(dict(base))
    replacements = {
        str(item.get('case_id')): item
        for item in update.get('cases') or []
        if isinstance(item, dict) and item.get('case_id') is not None
    }
    result['cases'] = [
        copy.deepcopy(replacements.get(str(item.get('case_id')), item))
        for item in result.get('cases') or []
        if isinstance(item, dict)
    ]
    known = {str(item.get('case_id')) for item in result['cases'] if item.get('case_id') is not None}
    result['cases'].extend(
        copy.deepcopy(item)
        for case_id, item in replacements.items()
        if case_id not in known
    )
    return result


def _refresh_cost_estimate(plan: Dict[str, Any]) -> None:
    from ..costing import ensure_plan_cost_estimate
    from ..schema import StudyPlan

    typed_plan = StudyPlan.from_dict(plan)
    plan['cost_estimate'] = ensure_plan_cost_estimate(typed_plan)
    plan['resource_policy'] = typed_plan.resource_policy


def _sync_plan(adaptive: Dict[str, Any], kind: str, plan: Dict[str, Any]) -> None:
    _refresh_cost_estimate(plan)
    key = _PLAN_KEY_BY_KIND.get(kind, 'refined_plan')
    adaptive[key] = _merge_plan_cases(adaptive.get(key), plan)


def _decision_entries(adaptive: Mapping[str, Any]) -> List[Dict[str, Any]]:
    entries: List[Dict[str, Any]] = []
    for key in (
        'decision_log',
        'recovery_decisions',
        'path_refinement_decisions',
        'direct_active_space_review_decisions',
        'entanglement_active_space_decisions',
    ):
        entries.extend(item for item in adaptive.get(key) or [] if isinstance(item, dict))
    return entries


def _decision_map(adaptive: Mapping[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get('case_id')): item
        for item in _decision_entries(adaptive)
        if item.get('case_id') is not None
    }


def _lifecycle(state: Mapping[str, Any], plan: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
    value = state.get('lifecycle')
    if not isinstance(value, dict) and isinstance(plan, Mapping):
        value = plan.get('lifecycle')
    return ensure_lifecycle(
        value,
        'study',
        entity_id=state.get('study_id') or (plan or {}).get('study_id'),
    )


def _request_review(lifecycle: Dict[str, Any], details: Mapping[str, Any]) -> Dict[str, Any]:
    if lifecycle['stage'] == 'review_required':
        return lifecycle
    if lifecycle['stage'] in ('planned', 'executing', 'refining', 'completed', 'analyzing'):
        return transition_lifecycle(lifecycle, 'review_requested', details=details)
    return lifecycle


def _approve(lifecycle: Dict[str, Any], details: Mapping[str, Any], *, refining: bool) -> Dict[str, Any]:
    lifecycle = _request_review(lifecycle, details)
    if lifecycle['stage'] == 'review_required':
        lifecycle = transition_lifecycle(lifecycle, 'approval_granted', details=details)
    if refining and lifecycle['stage'] == 'retry_ready':
        lifecycle = transition_lifecycle(lifecycle, 'refinement_started', details=details)
    return lifecycle


def _set_lifecycle(state: Dict[str, Any], plan: Optional[Dict[str, Any]], lifecycle: Mapping[str, Any]) -> None:
    state['lifecycle'] = copy.deepcopy(dict(lifecycle))
    if isinstance(plan, dict):
        plan['lifecycle'] = copy.deepcopy(dict(lifecycle))


def _ready_workflow(
    *,
    stage: str,
    plan_kind: str,
    case_ids: Sequence[str],
    message: str,
) -> Dict[str, Any]:
    return {
        'schema': 'pyscf-agent.adaptive-workflow.v1',
        'stage': stage,
        'active_plan_kind': plan_kind,
        'scope': 'subset' if case_ids else 'full',
        'pending_case_ids': list(case_ids),
        'completed_case_ids': [],
        'failed_case_ids': [],
        'allowed_actions': [],
        'message': message,
        'gate_status': 'passed',
    }


def _review_workflow(
    *,
    plan_kind: str,
    case_ids: Sequence[str],
    message: str,
) -> Dict[str, Any]:
    singular = len(case_ids) == 1
    return {
        'schema': 'pyscf-agent.adaptive-workflow.v1',
        'stage': 'active_space_review_required',
        'active_plan_kind': plan_kind,
        'scope': 'subset',
        'pending_case_ids': list(case_ids),
        'completed_case_ids': [],
        'failed_case_ids': [],
        'review_mode': 'batch',
        'current_case_id': None,
        'current_case_ids': [],
        'allowed_actions': [
            gate_action('approve_active_space', 'Approve Active Space' if singular else 'Approve Active Spaces', case_ids=list(case_ids)),
            gate_action('expand_active_space', 'Expand Active Space' if singular else 'Expand Active Spaces', secondary=True, case_ids=list(case_ids)),
            gate_action('cancel_active_space_review', 'Cancel', secondary=True, case_ids=list(case_ids)),
        ],
        'message': message,
        'gate_status': 'review_required',
    }


def _mark_cost_approved(plan: Dict[str, Any]) -> None:
    _refresh_cost_estimate(plan)
    if plan['cost_estimate'].get('resource_limit_exceeded') or plan['cost_estimate'].get('approval_allowed') is False:
        raise StudyReviewActionError(
            'The estimate exceeds the selected Slurm memory allocation and cannot be approved. '
            'Choose a larger resource profile or reduce the calculation size.'
        )
    policy = plan.get('resource_policy') if isinstance(plan.get('resource_policy'), dict) else {}
    policy['approved'] = True
    plan['resource_policy'] = policy
    estimate = plan.get('cost_estimate') if isinstance(plan.get('cost_estimate'), dict) else None
    if estimate is not None:
        estimate['approved'] = True
        estimate['can_execute'] = True
        estimate['status'] = 'ready'
        if isinstance(estimate.get('policy'), dict):
            estimate['policy']['approved'] = True


def _integer_list(values: Any) -> List[int]:
    if not isinstance(values, list):
        return []
    result = []
    for value in values:
        try:
            number = int(value)
        except (TypeError, ValueError):
            continue
        if number not in result:
            result.append(number)
    return sorted(result)


def _electron_increment(row: Mapping[str, Any], spin_resolved: bool) -> Optional[Dict[str, int]]:
    try:
        occupation = float(row.get('occupation'))
    except (TypeError, ValueError):
        return None
    if occupation < -1e-6 or occupation > 2.0 + 1e-6:
        return None
    if not spin_resolved:
        return {'total': max(0, min(2, int(round(occupation))))}
    try:
        alpha = float(row.get('occupation_alpha'))
        beta = float(row.get('occupation_beta'))
    except (TypeError, ValueError):
        return None
    return {
        'alpha': max(0, min(1, int(round(alpha)))),
        'beta': max(0, min(1, int(round(beta)))),
    }


def _increment_nelecas(nelecas: Any, increments: Iterable[Mapping[str, int]]) -> Any:
    total = sum(int(item.get('total') or 0) for item in increments)
    alpha = sum(int(item.get('alpha') or 0) for item in increments)
    beta = sum(int(item.get('beta') or 0) for item in increments)
    if isinstance(nelecas, list) and len(nelecas) >= 2:
        return [int(nelecas[0]) + alpha, int(nelecas[1]) + beta]
    if isinstance(nelecas, str) and ',' in nelecas:
        parts = [int(item.strip()) for item in nelecas.split(',')]
        return [parts[0] + alpha, parts[1] + beta]
    try:
        return int(nelecas) + total
    except (TypeError, ValueError):
        return nelecas


def expand_active_space_contract(active_space: Mapping[str, Any]) -> Optional[Dict[str, Any]]:
    value = copy.deepcopy(dict(active_space))
    evidence = value.get('expansion_evidence') if isinstance(value.get('expansion_evidence'), dict) else {}
    rows = [item for item in evidence.get('orbitals') or [] if isinstance(item, dict)]
    try:
        orbital_count = int(evidence.get('orbital_count'))
    except (TypeError, ValueError):
        return None
    by_index = {}
    for row in rows:
        try:
            by_index[int(row.get('index'))] = row
        except (TypeError, ValueError):
            continue
    indices = _integer_list(value.get('orbital_indices'))
    if not indices or orbital_count <= 0:
        return None
    spin_resolved = isinstance(value.get('nelecas'), list) or (
        isinstance(value.get('nelecas'), str) and ',' in value['nelecas']
    )
    if spin_resolved and evidence.get('orbital_mapping') != 'spin_resolved_mapped':
        return None
    added = []
    if indices[0] > 0:
        added.append(indices[0] - 1)
    if indices[-1] + 1 < orbital_count:
        added.append(indices[-1] + 1)
    increments = [_electron_increment(by_index.get(index, {}), spin_resolved) for index in added]
    if not added or any(item is None for item in increments):
        return None
    value['orbital_indices'] = sorted(set(indices + added))
    value['ncas'] = len(value['orbital_indices'])
    value['nelecas'] = _increment_nelecas(value.get('nelecas'), [item for item in increments if item])
    value['enabled'] = True
    value['selection_method'] = 'manual'
    value['approved'] = False
    value['expansion'] = {
        'source': evidence.get('source') or 'ActiveSpaceAudit reference occupations',
        'added_orbital_indices': added,
    }
    return value


def _active_space_for_case(
    request: Mapping[str, Any],
    decision: Optional[Mapping[str, Any]],
) -> Optional[Dict[str, Any]]:
    value = request.get('active_space')
    if isinstance(value, dict) and value.get('ncas') is not None:
        return copy.deepcopy(value)
    candidate = decision.get('active_space_contract') if isinstance(decision, Mapping) else None
    return copy.deepcopy(candidate) if isinstance(candidate, dict) else None


def _configure_casscf(request: Dict[str, Any], active_space: Mapping[str, Any], *, approved: bool) -> None:
    request['method'] = 'casscf'
    request['restricted'] = True
    request['xc'] = None
    request['active_space'] = copy.deepcopy(dict(active_space))
    request['active_space']['enabled'] = True
    request['active_space']['selection_method'] = request['active_space'].get('selection_method') or 'manual'
    request['active_space']['approved'] = approved
    solver, solver_options = active_space_solver_contract_from_request(request)
    if solver == 'block2_dmrg':
        options = copy.deepcopy(solver_options)
        options.setdefault('preset', 'balanced')
        options.setdefault('save_mps', True)
        request['solver'] = {'name': solver, 'options': options}
    elif solver == 'fci':
        request['solver'] = {'name': solver, 'options': copy.deepcopy(solver_options)}
    post_cas = request.get('post_cas') if isinstance(request.get('post_cas'), dict) else {}
    sc_nevpt2 = post_cas.get('sc_nevpt2') if isinstance(post_cas.get('sc_nevpt2'), dict) else {}
    sc_nevpt2['enabled'] = False
    post_cas['sc_nevpt2'] = sc_nevpt2
    request['post_cas'] = post_cas


def _result(
    *,
    action_id: str,
    status: str,
    message: str,
    state: Dict[str, Any],
    plan: Optional[Dict[str, Any]],
    workflow: Dict[str, Any],
    lifecycle: Dict[str, Any],
    case_ids: Sequence[str],
    can_run: bool,
) -> Dict[str, Any]:
    return {
        'schema': REVIEW_ACTION_RESULT_SCHEMA,
        'action_id': action_id,
        'status': status,
        'message': message,
        'study_state': state,
        'plan': plan,
        'workflow': workflow,
        'lifecycle': lifecycle,
        'case_ids': list(case_ids),
        'can_run': bool(can_run),
    }


def apply_study_review_action(
    action_id: str,
    *,
    study_state: Optional[Mapping[str, Any]] = None,
    plan: Optional[Mapping[str, Any]] = None,
    case_ids: Optional[Sequence[Any]] = None,
    approval_token: Optional[str] = None,
    plan_kind: Optional[str] = None,
) -> Dict[str, Any]:
    action = str(action_id or '').strip().lower()
    state = copy.deepcopy(dict(study_state or {}))
    current_plan = copy.deepcopy(dict(plan)) if isinstance(plan, Mapping) else None
    selected = _case_ids(case_ids)
    adaptive, _ = _adaptive_container(state)
    lifecycle = _lifecycle(state, current_plan)
    details = {'source': action, 'case_ids': selected}

    if action == 'approve_cost_estimate':
        kind = str(plan_kind or _plan_kind(current_plan)).strip() or 'refined'
        target = current_plan
        if kind == 'initial':
            initial_plan = adaptive.get('initial_scan_plan')
            if isinstance(initial_plan, dict):
                target = copy.deepcopy(initial_plan)
        if target is None:
            candidate = adaptive.get(_PLAN_KEY_BY_KIND.get(kind, 'refined_plan'))
            target = copy.deepcopy(candidate) if isinstance(candidate, dict) else None
        if target is None:
            raise StudyReviewActionError('The reviewed cost estimate has no executable plan.')
        if kind != 'initial':
            target['study_id'] = state.get('study_id') or target['study_id']
            target['_review_kind'] = kind
        estimate = target.get('cost_estimate') if isinstance(target.get('cost_estimate'), dict) else {}
        if estimate.get('resource_limit_exceeded') or estimate.get('approval_allowed') is False:
            raise StudyReviewActionError(
                'The estimate exceeds the selected Slurm memory allocation and cannot be approved. '
                'Choose a larger resource profile or reduce the calculation size.'
            )
        _mark_cost_approved(target)
        if kind == 'initial':
            estimate = adaptive.get('initial_scan_cost_estimate')
            if isinstance(estimate, dict):
                estimate['approved'] = True
                estimate['can_execute'] = True
                estimate['status'] = 'ready'
                if isinstance(estimate.get('policy'), dict):
                    estimate['policy']['approved'] = True
        lifecycle = _approve(lifecycle, details, refining=kind not in ('initial', 'static'))
        _set_lifecycle(state, target, lifecycle)
        if kind in _PLAN_KEY_BY_KIND:
            _sync_plan(adaptive, kind, target)
        message = 'Estimated cost approved. The selected plan is ready to run.'
        workflow = _ready_workflow(
            stage='refinement_ready' if lifecycle['stage'] == 'refining' else 'plan_ready',
            plan_kind=kind,
            case_ids=selected,
            message=message,
        )
        adaptive['workflow'] = copy.deepcopy(workflow)
        result = _result(action_id=action, status='prepared', message=message, state=state, plan=target, workflow=workflow, lifecycle=lifecycle, case_ids=selected, can_run=True)
        if kind == 'initial':
            result['study_spec_patch'] = {'resource_policy': {'approved': True}}
        return result

    if action in ('increase_recovery_max_cycle', 'increase_dmet_iterations', 'try_fci_recovery', 'retry_dmet_without_impurity_diis'):
        kind, source = _source_plan(adaptive, current_plan, selected, requested_kind=plan_kind)
        label = {
            'increase_recovery_max_cycle': 'max-cycle-retry',
            'increase_dmet_iterations': 'dmet-iteration-retry',
            'try_fci_recovery': 'fci-benchmark',
            'retry_dmet_without_impurity_diis': 'impurity-scf-retry',
        }[action]
        target = _selected_plan(source, selected, kind=kind, label=label, study_id=state.get('study_id'))
        statuses = {
            str(row['case_id']): row.get('status')
            for row in state.get('comparison_table') or []
            if isinstance(row, dict) and row.get('case_id')
        }
        statuses.update({
            str(case['case_id']): case['task_report']['execution_status']
            for case in state.get('cases') or []
            if isinstance(case, dict) and case.get('case_id')
            and isinstance(case.get('task_report'), dict)
            and case['task_report'].get('execution_status')
        })
        for item in target.get('cases') or []:
            request = item.get('request') if isinstance(item.get('request'), dict) else {}
            if action in ('increase_recovery_max_cycle', 'increase_dmet_iterations') and statuses.get(str(item.get('case_id'))) == 'failed':
                raise StudyReviewActionError(
                    '{0}: execution failed with an error, not an iteration-limit result. '
                    'Inspect the task error and provider log before choosing recovery.'.format(item['case_id'])
                )
            if action == 'retry_dmet_without_impurity_diis':
                task_report = next((
                    case.get('task_report') or {} for case in state.get('cases') or []
                    if str(case.get('case_id')) == str(item.get('case_id'))
                ), {})
                recommendation = dmet_task_recovery(task_report, request)
                if not recommendation:
                    raise StudyReviewActionError(
                        'This retry needs a reported impurity SCF DIIS failure with DIIS still enabled.'
                    )
                if not isinstance(request.get('solver'), dict):
                    request['solver'] = {'name': 'dmet', 'options': {}}
                request['solver'].setdefault('options', {}).update(recommendation['solver_options_patch'])
            elif action == 'increase_recovery_max_cycle':
                solver = request.get('solver')
                solver_name = solver.get('name') if isinstance(solver, dict) else solver
                if str(solver_name or '').strip().lower() == 'dmet':
                    raise StudyReviewActionError(
                        'runtime.max_cycle does not control DMET iterations or its impurity solver. '
                        'Use increase_dmet_iterations for an unconverged DMET outer loop; '
                        'inspect impurity solver failures separately.'
                    )
                request = _runtime_recovery_request(request, minimum_max_cycle=400)
            elif action == 'increase_dmet_iterations':
                solver = request.get('solver') if isinstance(request.get('solver'), dict) else {}
                if str(solver.get('name') or '').strip().lower() != 'dmet':
                    raise StudyReviewActionError(
                        'The selected case is not a DMET request and cannot use the DMET iteration retry.'
                    )
                options = solver.get('options') if isinstance(solver.get('options'), dict) else {}
                try:
                    current = int(options.get('max_iterations') or 50)
                except (TypeError, ValueError):
                    current = 50
                options['max_iterations'] = max(current * 2, 100)
                solver['options'] = options
                request['solver'] = solver
            elif request.get('task_type') == 'model_hamiltonian' or request.get('model_hamiltonian'):
                request['solver'] = 'fci'
            else:
                request['method'] = 'fci'
                request['xc'] = None
                request['active_space'] = {'enabled': False}
                post_cas = request.get('post_cas') if isinstance(request.get('post_cas'), dict) else {}
                post_cas['sc_nevpt2'] = {'enabled': False}
                request['post_cas'] = post_cas
            item['request'] = request
        lifecycle = _approve(lifecycle, details, refining=True)
        _set_lifecycle(state, target, lifecycle)
        _sync_plan(adaptive, kind, target)
        message = '{0} prepared for {1} selected case(s).'.format(
            {
                'increase_recovery_max_cycle': 'Max-cycle retry',
                'increase_dmet_iterations': 'DMET iteration retry',
                'try_fci_recovery': 'FCI benchmark',
                'retry_dmet_without_impurity_diis': 'Impurity SCF retry (impurity_scf_diis: true → false)',
            }[action],
            len(target.get('cases') or []),
        )
        workflow = _ready_workflow(stage='refinement_ready', plan_kind=kind, case_ids=selected, message=message)
        adaptive['workflow'] = copy.deepcopy(workflow)
        return _result(action_id=action, status='prepared', message=message, state=state, plan=target, workflow=workflow, lifecycle=lifecycle, case_ids=selected, can_run=True)

    if action in ('expand_active_space', 'promote_casscf_recovery', 'approve_active_space'):
        require_active = action == 'approve_active_space'
        kind, source = _source_plan(
            adaptive,
            current_plan,
            selected,
            require_active_space=require_active,
            requested_kind=plan_kind,
        )
        label = {
            'expand_active_space': 'expanded-active-space',
            'promote_casscf_recovery': 'casscf-review',
            'approve_active_space': 'approved-active-space',
        }[action]
        target = _selected_plan(source, selected, kind=kind, label=label, study_id=state.get('study_id'))
        decisions = _decision_map(adaptive)
        changed = 0
        for item in target.get('cases') or []:
            request = item.get('request') if isinstance(item.get('request'), dict) else {}
            active_space = _active_space_for_case(request, decisions.get(str(item.get('case_id'))))
            if not isinstance(active_space, dict):
                continue
            if action == 'expand_active_space':
                active_space = expand_active_space_contract(active_space)
                if active_space is None:
                    continue
                _configure_casscf(request, active_space, approved=False)
            elif action == 'promote_casscf_recovery':
                if active_space.get('ncas') is None or active_space.get('nelecas') is None:
                    continue
                _configure_casscf(request, active_space, approved=False)
            else:
                _configure_casscf(request, active_space, approved=True)
            item['request'] = request
            changed += 1
        if not changed:
            raise StudyReviewActionError(
                'The selected ActiveSpaceAudit does not contain enough bounded orbital evidence for this action.'
            )
        if action == 'approve_active_space':
            for decision in _decision_entries(adaptive):
                if str(decision.get('case_id')) not in set(selected):
                    continue
                if isinstance(decision.get('active_space_contract'), dict):
                    decision['active_space_contract']['approved'] = True
                tags = [tag for tag in decision.get('tags') or [] if tag != 'requires_active_space_approval']
                if 'active_space_approved' not in tags:
                    tags.append('active_space_approved')
                decision['tags'] = tags
            if isinstance(target.get('cost_estimate'), dict) and target['cost_estimate'].get('approval_required'):
                target.setdefault('resource_policy', {})['approved'] = True
                target.pop('cost_estimate', None)
            lifecycle = _approve(lifecycle, details, refining=True)
            _set_lifecycle(state, target, lifecycle)
            _sync_plan(adaptive, kind, target)
            message = 'Approved ActiveSpaceAudit for {0} selected case(s).'.format(changed)
            workflow = _ready_workflow(stage='refinement_ready', plan_kind=kind, case_ids=selected, message=message)
            adaptive['workflow'] = copy.deepcopy(workflow)
            return _result(action_id=action, status='prepared', message=message, state=state, plan=target, workflow=workflow, lifecycle=lifecycle, case_ids=selected, can_run=True)
        lifecycle = _request_review(lifecycle, details)
        _set_lifecycle(state, target, lifecycle)
        _sync_plan(adaptive, kind, target)
        message = (
            'Expanded ActiveSpaceAudit prepared; review the larger active space before running.'
            if action == 'expand_active_space'
            else 'CASSCF ActiveSpaceAudit candidate prepared; approval is required before running.'
        )
        workflow = _review_workflow(plan_kind=kind, case_ids=selected, message=message)
        adaptive['workflow'] = copy.deepcopy(workflow)
        return _result(action_id=action, status='review_required', message=message, state=state, plan=target, workflow=workflow, lifecycle=lifecycle, case_ids=selected, can_run=False)

    if action in (
        'approve_mps_continuation',
        'skip_mps_continuation',
        'cancel_mps_continuation_review',
    ):
        approval = adaptive.get('mps_continuation_approval')
        if not isinstance(approval, dict):
            raise StudyReviewActionError('The MPS continuation approval contract is unavailable.')
        expected = str(approval.get('approval_token') or '')
        if not approval_token or str(approval_token) != expected:
            raise StudyReviewActionError('The MPS continuation approval token is invalid or stale.')
        decision = {
            'approve_mps_continuation': 'approve',
            'skip_mps_continuation': 'skip',
            'cancel_mps_continuation_review': 'cancel',
        }[action]
        adaptive['mps_continuation_review'] = {
            'schema': approval.get('schema'),
            'approval_token': expected,
            'decision': decision,
        }
        approval['status'] = {
            'approve': 'approved',
            'skip': 'skipped',
            'cancel': 'cancelled',
        }[decision]
        approval['decision'] = decision
        if decision == 'approve':
            lifecycle = _approve(lifecycle, details, refining=True)
            message = 'Same-method MPS continuation approved and ready to run.'
            can_run = True
            stage = 'mps_continuation_retry_ready'
        elif decision == 'skip':
            lifecycle = _approve(lifecycle, details, refining=False)
            message = 'MPS continuation skipped; ordinary path and method review may continue.'
            can_run = False
            stage = 'mps_continuation_skipped'
        else:
            lifecycle = _request_review(lifecycle, details)
            if lifecycle['stage'] == 'review_required':
                lifecycle = transition_lifecycle(lifecycle, 'approval_rejected', details=details)
            message = 'MPS continuation review canceled. No retry was launched.'
            can_run = False
            stage = 'mps_continuation_cancelled'
        _set_lifecycle(state, current_plan, lifecycle)
        workflow = _ready_workflow(
            stage=stage,
            plan_kind='mps_continuation',
            case_ids=selected,
            message=message,
        )
        adaptive['workflow'] = copy.deepcopy(workflow)
        result_status = 'prepared' if can_run else ('skipped' if decision == 'skip' else 'cancelled')
        result = _result(
            action_id=action,
            status=result_status,
            message=message,
            state=state,
            plan=current_plan,
            workflow=workflow,
            lifecycle=lifecycle,
            case_ids=selected,
            can_run=can_run,
        )
        result['analysis_approval'] = {
            'decision': decision,
            'approval_token': expected,
        }
        return result

    if action in ('approve_path_restart', 'skip_path_restart', 'cancel_path_restart_review'):
        approval = adaptive.get('path_window_restart_approval')
        if not isinstance(approval, dict):
            raise StudyReviewActionError('The one-particle-state continuation approval contract is unavailable.')
        expected = str(approval.get('approval_token') or '')
        if not approval_token or str(approval_token) != expected:
            raise StudyReviewActionError('The one-particle-state continuation approval token is invalid or stale.')
        decision = {
            'approve_path_restart': 'approve',
            'skip_path_restart': 'skip',
            'cancel_path_restart_review': 'cancel',
        }[action]
        adaptive['path_window_restart_review'] = {
            'schema': approval.get('schema'),
            'approval_token': expected,
            'decision': decision,
        }
        approval['status'] = {
            'approve': 'approved',
            'skip': 'skipped',
            'cancel': 'cancelled',
        }[decision]
        approval['decision'] = decision
        validation = adaptive.get('path_window_restart_validation')
        if isinstance(validation, dict):
            validation['status'] = {
                'approve': 'approved',
                'skip': 'skipped',
                'cancel': 'cancelled',
            }[decision]
        if decision == 'approve':
            lifecycle = _approve(lifecycle, details, refining=True)
            message = 'Bidirectional one-particle-state continuation approved and ready to run.'
            can_run = True
            stage = 'continuation_retry_ready'
        elif decision == 'skip':
            lifecycle = _approve(lifecycle, details, refining=False)
            message = 'One-particle-state continuation skipped; method-level analysis may continue.'
            can_run = False
            stage = 'continuation_skipped'
        else:
            lifecycle = _request_review(lifecycle, details)
            if lifecycle['stage'] == 'review_required':
                lifecycle = transition_lifecycle(lifecycle, 'approval_rejected', details=details)
            message = 'One-particle-state continuation review canceled. No restart was launched.'
            can_run = False
            stage = 'continuation_cancelled'
        _set_lifecycle(state, current_plan, lifecycle)
        workflow = _ready_workflow(stage=stage, plan_kind='path_restart', case_ids=selected, message=message)
        adaptive['workflow'] = copy.deepcopy(workflow)
        result_status = 'prepared' if can_run else ('skipped' if decision == 'skip' else 'cancelled')
        result = _result(action_id=action, status=result_status, message=message, state=state, plan=current_plan, workflow=workflow, lifecycle=lifecycle, case_ids=selected, can_run=can_run)
        result['analysis_approval'] = {'decision': decision, 'approval_token': expected}
        return result

    if action == 'cancel_active_space_review':
        lifecycle = _request_review(lifecycle, details)
        if lifecycle['stage'] == 'review_required':
            lifecycle = transition_lifecycle(lifecycle, 'approval_rejected', details=details)
        _set_lifecycle(state, current_plan, lifecycle)
        message = 'Active-space review canceled. No candidate was approved.'
        workflow = _ready_workflow(stage='cancelled', plan_kind=str(plan_kind or 'refined'), case_ids=selected, message=message)
        adaptive['workflow'] = copy.deepcopy(workflow)
        return _result(action_id=action, status='cancelled', message=message, state=state, plan=current_plan, workflow=workflow, lifecycle=lifecycle, case_ids=selected, can_run=False)

    raise StudyReviewActionError('Unsupported study review action: {0}'.format(action or '<empty>'))


__all__ = [
    'REVIEW_ACTION_RESULT_SCHEMA',
    'StudyReviewActionError',
    'apply_study_review_action',
    'expand_active_space_contract',
]
