from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple

from .plan_utils import case_ids_from_plan as _case_ids_from_plan

from pyscf_agent.providers.libdmet.recovery import dmet_task_recovery

from pyscf_agent.workflow_gates import (
    GATE_STATUS_BLOCKED,
    GATE_STATUS_PASSED,
    GATE_STATUS_REVIEW_REQUIRED,
    GATE_STATUS_SKIPPED,
    GateDecision,
    GateInvocation,
)

from .actions import gate_action as _action


def _decision(
    invocation: GateInvocation,
    status: str,
    summary: str,
    *,
    checks: Any = (),
    actions: Any = (),
    evidence: Any = (),
    case_ids: Any = (),
    details: Any = None,
) -> GateDecision:
    return GateDecision(
        gate_id=invocation.gate_id,
        scope=invocation.scope,
        hook=invocation.hook,
        status=status,
        summary=summary,
        checks=tuple(copy.deepcopy(checks or ())),
        recommended_actions=tuple(copy.deepcopy(actions or ())),
        evidence=tuple(copy.deepcopy(evidence or ())),
        affected_case_ids=tuple(str(item) for item in (case_ids or ()) if str(item).strip()),
        details=copy.deepcopy(details or {}),
        provenance={
            'gate_set_id': invocation.gate_set_id,
            'evaluator_id': invocation.evaluator_id,
            'configuration': copy.deepcopy(invocation.configuration),
        },
    )


def _report(context: Dict[str, Any]) -> Dict[str, Any]:
    value = context.get('study_report')
    return value if isinstance(value, dict) else {}


def _plan(context: Dict[str, Any]) -> Dict[str, Any]:
    value = context.get('study_plan')
    return value if isinstance(value, dict) else {}


def _adaptive(report: Dict[str, Any]) -> Dict[str, Any]:
    value = report.get('adaptive')
    return value if isinstance(value, dict) else {}


def _normalize_method(value: Any) -> str:
    return str(value or '').strip().lower()


def _request_solver(request: Dict[str, Any]) -> str:
    value = request.get('solver') if isinstance(request, dict) else None
    if isinstance(value, dict):
        value = value.get('name')
    normalized = _normalize_method(value).replace('-', '_')
    if normalized in ('block2', 'dmrg'):
        return 'block2_dmrg'
    return normalized


def _enabled_unapproved_active_space(request: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    active_space = request.get('active_space') if isinstance(request, dict) else None
    if not isinstance(active_space, dict):
        return None
    if active_space.get('enabled') is False or active_space.get('approved') is True:
        return None
    return active_space


def _decision_map(report: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    adaptive = _adaptive(report)
    decisions: List[Dict[str, Any]] = []
    for key in (
        'decision_log',
        'recovery_decisions',
        'path_refinement_decisions',
        'direct_active_space_review_decisions',
        'entanglement_active_space_decisions',
    ):
        values = adaptive.get(key)
        if isinstance(values, list):
            decisions.extend(item for item in values if isinstance(item, dict))
    return {
        str(item.get('case_id')): item
        for item in decisions
        if item.get('case_id') is not None
    }


_ACTIVE_SPACE_PLAN_KEYS = {
    'direct_casscf_review': 'direct_active_space_review_plan',
    'entanglement_active_space': 'entanglement_active_space_plan',
    'recovery': 'recovery_plan',
    'refined': 'refined_plan',
    'path_refinement': 'path_refinement_plan',
}


def _preferred_active_plan_kind(report: Dict[str, Any]) -> str:
    adaptive = _adaptive(report)
    workflow = adaptive.get('workflow') if isinstance(adaptive.get('workflow'), dict) else {}
    if workflow.get('stage') in ('active_space_review_required', 'path_refinement_review_required'):
        plan_kind = str(workflow.get('active_plan_kind') or '').strip()
        if plan_kind in _ACTIVE_SPACE_PLAN_KEYS:
            return plan_kind
    return {
        'direct_casscf_review': 'direct_casscf_review',
        'entanglement_active_space_review': 'entanglement_active_space',
        'result_path_review': 'path_refinement',
    }.get(str(adaptive.get('mode') or '').strip(), '')


def active_space_approval_items(
    report: Dict[str, Any],
    plan_kind: Optional[str] = None,
) -> List[Dict[str, Any]]:
    adaptive = _adaptive(report)
    decisions = _decision_map(report)
    items: List[Dict[str, Any]] = []
    seen = set()
    selected_kind = str(plan_kind or _preferred_active_plan_kind(report) or '').strip()
    candidate_plans = (
        ('direct_casscf_review', adaptive.get('direct_active_space_review_plan')),
        ('entanglement_active_space', adaptive.get('entanglement_active_space_plan')),
        ('recovery', adaptive.get('recovery_plan')),
        ('refined', adaptive.get('refined_plan')),
        ('path_refinement', adaptive.get('path_refinement_plan')),
    )
    if selected_kind:
        candidate_plans = tuple(item for item in candidate_plans if item[0] == selected_kind)
    for current_plan_kind, plan in candidate_plans:
        cases = plan.get('cases') if isinstance(plan, dict) else []
        if not isinstance(cases, list):
            continue
        for case in cases:
            if not isinstance(case, dict):
                continue
            case_id = str(case.get('case_id') or '')
            if not case_id or case_id in seen:
                continue
            request = case.get('request') if isinstance(case.get('request'), dict) else {}
            method = _normalize_method(request.get('method') or request.get('solver'))
            solver = _request_solver(request) or 'fci'
            active_space = _enabled_unapproved_active_space(request)
            if method not in ('casci', 'casscf') or not active_space:
                continue
            seen.add(case_id)
            decision = decisions.get(case_id, {})
            detail = '{0} CAS({1}, {2}) active space awaits approval.'.format(
                method.upper(),
                active_space.get('nelecas', 'n/a'),
                active_space.get('ncas', 'n/a'),
            )
            if solver == 'block2_dmrg':
                solver_options = request.get('solver', {}).get('options', {}) if isinstance(request.get('solver'), dict) else {}
                dimensions = solver_options.get('bond_dimensions')
                max_bond_dimension = max(dimensions) if isinstance(dimensions, list) and dimensions else {
                    'screening': 200,
                    'balanced': 500,
                    'high_accuracy': 1000,
                }.get(str(solver_options.get('preset') or 'balanced').lower(), 500)
                detail = '{0} Solver=block2 DMRG; max bond dimension={1}; sweeps={2}.'.format(
                    detail,
                    max_bond_dimension,
                    solver_options.get('sweeps') or 10,
                )
            if current_plan_kind == 'path_refinement' and decision.get('path_anomaly_reason'):
                detail = '{0} {1}'.format(detail, decision['path_anomaly_reason'])
            items.append({
                'case_id': case_id,
                'label': case.get('label') or case_id,
                'plan_kind': current_plan_kind,
                'title': '{0} · {1}'.format(case_id, case.get('label') or case_id),
                'detail': detail,
                'status': 'approval required',
                'active_space': copy.deepcopy(active_space),
                'method': method,
                'solver': solver,
            })
    return items


def _active_space_continuity(items: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Summarize cross-case CAS contract changes without forcing one CAS size."""

    contracts = []
    for item in items:
        active_space = item.get('active_space') if isinstance(item, dict) else None
        if not isinstance(active_space, dict):
            continue
        contracts.append({
            'case_id': item.get('case_id'),
            'label': item.get('label'),
            'ncas': active_space.get('ncas'),
            'nelecas': copy.deepcopy(active_space.get('nelecas')),
            'orbital_indices': copy.deepcopy(active_space.get('orbital_indices') or []),
        })
    signatures = {
        (
            str(contract.get('ncas')),
            repr(contract.get('nelecas')),
        )
        for contract in contracts
    }
    consistent = len(signatures) <= 1
    return {
        'status': 'consistent' if consistent else 'review_required',
        'consistent': consistent,
        'contract_count': len(contracts),
        'variant_count': len(signatures),
        'contracts': contracts,
        'interpretation': (
            'The reviewed cases use one CAS size and electron-count contract.'
            if consistent
            else 'The reviewed path changes CAS size or active-electron count between cases; '
            'verify orbital identity and curve continuity before approving the batch.'
        ),
    }


def _comparison_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = report.get('comparison_table') if isinstance(report, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _unresolved_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    return [
        row for row in _comparison_rows(report)
        if (
            row.get('status')
            and str(row.get('status')).strip().lower() != 'succeeded'
        ) or row.get('publication_eligible') is False
    ]


def _dedupe_actions(actions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    result = []
    for action in actions:
        if action.get('id') and action.get('id') not in seen:
            seen.add(action['id'])
            result.append(action)
    return result


def _unresolved_actions(report: Dict[str, Any], rows: List[Dict[str, Any]], case_ids: List[str]) -> List[Dict[str, Any]]:
    system_type = str(report.get('system_type') or '').strip().lower()
    decisions = _decision_map(report)
    current_case_id = case_ids[0] if case_ids else None
    current_case_ids = [current_case_id] if current_case_id else []
    current_rows = [
        row for row in rows
        if current_case_id is None or str(row.get('case_id') or '') == current_case_id
    ]
    has_cas_unconverged = False
    has_non_cas_unconverged = False
    has_block2_unresolved = False
    has_dmet_unconverged = False
    has_execution_failed = False
    has_blocked = False
    has_quality_unresolved = False
    for row in current_rows:
        decision = decisions.get(str(row.get('case_id') or ''), {})
        method = _normalize_method(
            row.get('method')
            or row.get('solver')
            or decision.get('recovery_method')
            or decision.get('recovery_solver')
            or decision.get('recommended_method')
            or decision.get('recommended_solver')
        )
        status = str(row.get('status') or '').strip().lower()
        has_blocked = has_blocked or status == 'blocked'
        has_quality_unresolved = has_quality_unresolved or (
            status == 'succeeded' and row.get('publication_eligible') is False
        )
        if status in ('unconverged', 'failed'):
            solver = _normalize_method(row.get('solver')).replace('-', '_')
            if solver in ('block2', 'dmrg', 'block2_dmrg'):
                has_block2_unresolved = True
            elif status == 'failed':
                has_execution_failed = True
            elif solver == 'dmet':
                has_dmet_unconverged = True
            elif method in ('casci', 'casscf'):
                has_cas_unconverged = True
            else:
                has_non_cas_unconverged = True
    actions: List[Dict[str, Any]] = []
    if has_execution_failed:
        for case in report.get('cases') or []:
            if str(case.get('case_id') or '') not in current_case_ids:
                continue
            recommendation = dmet_task_recovery(case.get('task_report') or {})
            if recommendation:
                action = _action(
                    recommendation['recommended_action'],
                    'Prepare Retry without Impurity SCF DIIS', case_ids=current_case_ids,
                )
                action['description'] = recommendation['summary']
                actions.append(action)
        actions.append(_action('show_case_guidance', 'Show Failure Details', case_ids=current_case_ids))
    if has_dmet_unconverged:
        actions.append(_action(
            'increase_dmet_iterations',
            'Increase DMET Outer Iterations (max_iterations)',
            case_ids=current_case_ids,
        ))
    if has_block2_unresolved:
        actions.append(_action('show_case_guidance', 'Show DMRG Recovery', case_ids=current_case_ids))
    if has_cas_unconverged:
        actions.append(_action('expand_active_space', 'Expand Active Space', case_ids=current_case_ids))
        actions.append(_action('increase_recovery_max_cycle', 'Increase CAS max_cycle', secondary=True, case_ids=current_case_ids))
        if system_type == 'molecular':
            actions.append(_action('try_fci_recovery', 'Try FCI Benchmark', secondary=True, case_ids=current_case_ids))
    if has_non_cas_unconverged and not has_dmet_unconverged:
        if system_type == 'molecular':
            actions.append(_action('promote_casscf_recovery', 'Promote to CASSCF', secondary=bool(actions), case_ids=current_case_ids))
        actions.append(_action('increase_recovery_max_cycle', 'Increase max_cycle', secondary=bool(actions), case_ids=current_case_ids))
    if has_blocked and not has_block2_unresolved:
        actions.append(_action('show_case_guidance', 'Show Fix Details', secondary=bool(actions), case_ids=current_case_ids))
    if has_quality_unresolved:
        actions.append(_action('show_case_guidance', 'Show Quality Details', secondary=bool(actions), case_ids=current_case_ids))
    actions.append(_action('acknowledge', 'Acknowledge', secondary=True, case_ids=current_case_ids))
    return _dedupe_actions(actions)


def _cost_items(cost_estimate: Dict[str, Any]) -> List[Dict[str, Any]]:
    items = []
    for estimate in cost_estimate.get('cases') or []:
        if not isinstance(estimate, dict):
            continue
        determinant_count = estimate.get('determinant_count')
        memory_mb = estimate.get('memory_mb')
        if determinant_count is None and memory_mb is None:
            continue
        details = []
        if determinant_count is not None:
            details.append('determinants={0:,}'.format(int(determinant_count)))
        if memory_mb is not None:
            details.append('memory~{0:.1f} MB'.format(float(memory_mb)))
        if estimate.get('solver') == 'block2_dmrg' or estimate.get('impurity_solver') == 'block2_dmrg':
            details.append('M={0}'.format(estimate.get('bond_dimension') or 'n/a'))
            details.append('sweeps={0}'.format(estimate.get('sweeps') or 'n/a'))
        items.append({
            'case_id': estimate.get('case_id'),
            'label': estimate.get('label'),
            'title': '{0} · {1}'.format(estimate.get('case_id') or 'case', estimate.get('method') or 'method'),
            'detail': ', '.join(details),
            'status': 'cost review',
        })
    return items


def _active_plan_and_kind(context: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    report = _report(context)
    adaptive = _adaptive(report)
    requested_kind = str(context.get('plan_kind') or '').strip()
    if requested_kind:
        plan = _plan(context)
        return plan, requested_kind
    preferred_kind = _preferred_active_plan_kind(report)
    preferred_key = _ACTIVE_SPACE_PLAN_KEYS.get(preferred_kind)
    preferred_plan = adaptive.get(preferred_key) if preferred_key else None
    if isinstance(preferred_plan, dict):
        return preferred_plan, preferred_kind
    for plan_kind, plan_key in (
        ('direct_casscf_review', 'direct_active_space_review_plan'),
        ('entanglement_active_space', 'entanglement_active_space_plan'),
        ('path_refinement', 'path_refinement_plan'),
        ('recovery', 'recovery_plan'),
        ('refined', 'refined_plan'),
    ):
        plan = adaptive.get(plan_key)
        if isinstance(plan, dict):
            return plan, plan_kind
    return _plan(context), 'initial'


def compilation_gate(context: Dict[str, Any], invocation: GateInvocation) -> GateDecision:
    plan = _plan(context)
    if not plan:
        return _decision(invocation, GATE_STATUS_SKIPPED, 'No compiled study plan is present in this review context.')
    configuration = plan.get('workflow_configuration')
    provenance = plan.get('workflow_provenance')
    if not configuration and not provenance:
        return _decision(
            invocation,
            GATE_STATUS_SKIPPED,
            'This legacy or partial study-plan view does not carry compilation provenance.',
        )
    compiled = bool(
        isinstance(configuration, dict)
        and configuration.get('workflow_id')
        and isinstance(provenance, dict)
        and provenance.get('status') == 'compiled'
    )
    return _decision(
        invocation,
        GATE_STATUS_PASSED if compiled else GATE_STATUS_BLOCKED,
        'Study workflow compilation is complete.' if compiled else 'Study workflow compilation is incomplete or rejected.',
        checks=({
            'id': 'study_module_compilation',
            'status': 'passed' if compiled else 'failed',
            'message': 'Compiled study workflow and provenance are available.' if compiled else 'Compiled study workflow is missing.',
        },),
    )


def active_space_approval_gate(context: Dict[str, Any], invocation: GateInvocation) -> GateDecision:
    report = _report(context)
    preferred_kind = _preferred_active_plan_kind(report)
    items = active_space_approval_items(report, preferred_kind or None)
    if not items:
        return _decision(invocation, GATE_STATUS_SKIPPED, 'No active-space approval is pending.')
    plan_kind = next(
        (
            candidate
            for candidate in (
                'direct_casscf_review',
                'entanglement_active_space',
                'recovery',
                'refined',
                'path_refinement',
            )
            if any(item['plan_kind'] == candidate for item in items)
        ),
        'refined',
    )
    items = [item for item in items if item['plan_kind'] == plan_kind]
    continuity = _active_space_continuity(items)
    case_ids = [item['case_id'] for item in items]
    plan_key = (
        'direct_active_space_review_plan'
        if plan_kind == 'direct_casscf_review'
        else '{0}_plan'.format(plan_kind)
    )
    plan = _adaptive(report).get(plan_key)
    cost_estimate = plan.get('cost_estimate') if isinstance(plan, dict) else None
    singular = len(case_ids) == 1
    message = (
        'A scan-path discontinuity needs local CASSCF refinement. Approve the shared-size ActiveSpaceAudit before rerunning the full affected overlap window.'
        if plan_kind == 'path_refinement'
        else 'Entanglement diagnostics recommend a larger ActiveSpaceAudit before rerunning the selected subset.'
        if plan_kind == 'entanglement_active_space'
        else 'The requested CASSCF/CASCI calculation needs explicit ActiveSpaceAudit approval before execution.'
        if plan_kind == 'direct_casscf_review'
        else 'CASSCF/CASCI active-space candidates need approval before the selected subset can run.'
    )
    if not continuity['consistent']:
        message = '{0} The proposed cases do not all use the same CAS size/electron-count contract.'.format(
            message
        )
    return _decision(
        invocation,
        GATE_STATUS_REVIEW_REQUIRED,
        message,
        checks=(
            {
                'id': 'active_space_approval',
                'status': 'review_required',
                'message': '{0} active-space candidate(s) await approval.'.format(len(case_ids)),
            },
            {
                'id': 'active_space_continuity',
                'status': 'passed' if continuity['consistent'] else 'review_required',
                'message': continuity['interpretation'],
            },
        ),
        actions=(
            _action('approve_active_space', 'Approve Active Space' if singular else 'Approve Active Spaces', case_ids=case_ids),
            _action('expand_active_space', 'Expand Active Space' if singular else 'Expand Active Spaces', secondary=True, case_ids=case_ids),
            _action('cancel_active_space_review', 'Cancel', secondary=True, case_ids=case_ids),
        ),
        evidence=tuple(items),
        case_ids=case_ids,
        details={
            'workflow_stage': 'path_refinement_review_required' if plan_kind == 'path_refinement' else 'active_space_review_required',
            'active_plan_kind': plan_kind,
            'scope': 'subset',
            'review_mode': 'batch',
            'items': items,
            'active_space_continuity': continuity,
            'cost_estimate': copy.deepcopy(cost_estimate) if isinstance(cost_estimate, dict) else None,
        },
    )


def resource_feasibility_gate(context: Dict[str, Any], invocation: GateInvocation) -> GateDecision:
    plan, plan_kind = _active_plan_and_kind(context)
    cost_estimate = plan.get('cost_estimate') if isinstance(plan, dict) else None
    if not isinstance(cost_estimate, dict):
        return _decision(invocation, GATE_STATUS_SKIPPED, 'No cost estimate is available in this context.')
    if cost_estimate.get('resource_limit_exceeded'):
        case_ids = _case_ids_from_plan(plan)
        return _decision(
            invocation,
            GATE_STATUS_BLOCKED,
            'The planned calculation exceeds the memory allocated by the selected Slurm profile.',
            checks=({
                'id': 'resource_policy',
                'status': 'failed',
                'message': cost_estimate.get('review_reason') or 'Estimated memory exceeds the scheduler allocation.',
            },),
            actions=(
                _action('acknowledge', 'Close', secondary=True, case_ids=case_ids),
            ),
            evidence=({'cost_estimate': copy.deepcopy(cost_estimate)},),
            case_ids=case_ids,
            details={
                'workflow_stage': 'resource_limit_exceeded',
                'active_plan_kind': plan_kind,
                'scope': 'full',
                'review_mode': 'batch',
                'items': _cost_items(cost_estimate)[:8],
                'cost_estimate': copy.deepcopy(cost_estimate),
            },
        )
    if not cost_estimate.get('approval_required') or cost_estimate.get('approved'):
        return _decision(
            invocation,
            GATE_STATUS_PASSED,
            'The study is within the configured resource policy or its estimate was approved.',
            evidence=({'cost_estimate': copy.deepcopy(cost_estimate)},),
        )
    case_ids = _case_ids_from_plan(plan)
    return _decision(
        invocation,
        GATE_STATUS_REVIEW_REQUIRED,
        'The planned calculation crosses the configured resource-review threshold.',
        checks=({
            'id': 'resource_policy',
            'status': 'review_required',
            'message': cost_estimate.get('review_reason') or 'Estimated resources require review.',
        },),
        actions=(
            _action('approve_cost_estimate', 'Approve Estimated Cost', case_ids=case_ids),
            _action('acknowledge', 'Acknowledge', secondary=True, case_ids=case_ids),
        ),
        evidence=({'cost_estimate': copy.deepcopy(cost_estimate)},),
        case_ids=case_ids,
        details={
            'workflow_stage': 'cost_review_required',
            'active_plan_kind': plan_kind,
            'scope': 'full',
            'review_mode': 'batch',
            'items': _cost_items(cost_estimate)[:8],
            'cost_estimate': copy.deepcopy(cost_estimate),
        },
    )


def initial_scan_quality_gate(context: Dict[str, Any], invocation: GateInvocation) -> GateDecision:
    report = _report(context)
    issues = _adaptive(report).get('initial_scan_issues')
    issues = [item for item in issues if isinstance(item, dict)] if isinstance(issues, list) else []
    if not issues:
        return _decision(invocation, GATE_STATUS_SKIPPED, 'No blocked initial-scan cases are present.')
    case_ids = [str(item.get('case_id')) for item in issues if item.get('case_id') is not None]
    current_case_id = case_ids[0] if case_ids else None
    current_ids = [current_case_id] if current_case_id else []
    current_items = [item for item in issues if str(item.get('case_id')) == current_case_id] or issues[:1]
    return _decision(
        invocation,
        GATE_STATUS_BLOCKED,
        'Initial scan did not produce valid diagnostics for every case.',
        checks=({
            'id': 'initial_scan_results',
            'status': 'failed',
            'message': '{0} initial-scan case(s) are blocked.'.format(len(case_ids)),
        },),
        actions=(
            _action('show_case_guidance', 'Show Fix Details', case_ids=current_ids),
            _action('acknowledge', 'Acknowledge', secondary=True, case_ids=current_ids),
        ),
        evidence=tuple(copy.deepcopy(current_items[:1])),
        case_ids=case_ids,
        details={
            'workflow_stage': 'initial_scan_blocked',
            'active_plan_kind': 'initial',
            'scope': 'subset',
            'review_mode': 'case_queue',
            'items': copy.deepcopy(current_items[:1]),
        },
    )


def execution_quality_gate(context: Dict[str, Any], invocation: GateInvocation) -> GateDecision:
    report = _report(context)
    unresolved = _unresolved_rows(report)
    if not unresolved:
        return _decision(invocation, GATE_STATUS_PASSED, 'Every reported study case completed successfully.')
    case_ids = [str(row.get('case_id')) for row in unresolved if row.get('case_id') is not None]
    current_case_id = case_ids[0] if case_ids else None
    completed = [
        str(row.get('case_id'))
        for row in _comparison_rows(report)
        if row.get('case_id') is not None
        and str(row.get('status') or '').strip().lower() == 'succeeded'
        and row.get('publication_eligible') is not False
    ]
    adaptive = _adaptive(report)
    active_plan_kind = (
        'recovery'
        if adaptive.get('recovery_plan')
        else 'refined'
        if adaptive.get('refined_plan')
        else 'static'
    )
    return _decision(
        invocation,
        GATE_STATUS_REVIEW_REQUIRED,
        'Some study cases remain unresolved or failed required quality checks; unresolved cases must be rerun and failed quality evidence must be reviewed before results are used.',
        checks=({
            'id': 'case_execution_or_quality_status',
            'status': 'review_required',
            'message': '{0} case(s) require case-specific recovery.'.format(len(case_ids)),
        },),
        actions=tuple(_unresolved_actions(report, unresolved, case_ids)),
        evidence=tuple(copy.deepcopy(unresolved[:1])),
        case_ids=case_ids,
        details={
            'workflow_stage': 'recovery_review_required',
            'active_plan_kind': active_plan_kind,
            'scope': 'subset',
            'review_mode': 'case_queue',
            'items': copy.deepcopy(unresolved[:1]),
            'completed_case_ids': completed,
            'current_case_id': current_case_id,
        },
    )


def continuation_approval_gate(context: Dict[str, Any], invocation: GateInvocation) -> GateDecision:
    report = _report(context)
    adaptive = _adaptive(report)
    mps_approval = adaptive.get('mps_continuation_approval')
    mps_approval = mps_approval if isinstance(mps_approval, dict) else {}
    if mps_approval.get('status') == 'approval_required':
        case_ids = [str(item) for item in mps_approval.get('case_ids') or [] if str(item).strip()]
        return _decision(
            invocation,
            GATE_STATUS_REVIEW_REQUIRED,
            str(mps_approval.get('summary') or 'Approve the proposed block2 MPS continuation.'),
            checks=({
                'id': 'mps_continuation_approval',
                'status': 'review_required',
                'message': 'Cross-task MPS checkpoint reuse requires an explicit decision.',
            },),
            actions=(
                _action('approve_mps_continuation', 'Approve MPS Retry', case_ids=case_ids),
                _action('skip_mps_continuation', 'Skip MPS Retry', secondary=True, case_ids=case_ids),
                _action('cancel_mps_continuation_review', 'Cancel', secondary=True, case_ids=case_ids),
            ),
            evidence=tuple(copy.deepcopy(mps_approval.get('items') or ())),
            case_ids=case_ids,
            details={
                'workflow_stage': 'mps_continuation_review_required',
                'active_plan_kind': 'mps_continuation',
                'scope': 'subset',
                'review_mode': 'batch',
                'items': copy.deepcopy(mps_approval.get('items') or []),
                'approval_token': mps_approval.get('approval_token'),
            },
        )
    validation = adaptive.get('path_window_restart_validation')
    approval = adaptive.get('path_window_restart_approval')
    validation = validation if isinstance(validation, dict) else {}
    approval = approval if isinstance(approval, dict) else {}
    if validation.get('status') != 'approval_required' or not approval:
        return _decision(invocation, GATE_STATUS_SKIPPED, 'No one-particle-state continuation approval is pending.')
    case_ids = [str(item) for item in approval.get('case_ids') or [] if str(item).strip()]
    return _decision(
        invocation,
        GATE_STATUS_REVIEW_REQUIRED,
        str(approval.get('summary') or 'Approve the proposed one-particle-state continuation restart.'),
        checks=({
            'id': 'continuation_approval',
            'status': 'review_required',
            'message': 'Projected 1RDM reuse requires an explicit decision.',
        },),
        actions=(
            _action('approve_path_restart', 'Approve 1RDM Restart', case_ids=case_ids),
            _action('skip_path_restart', 'Skip to Method Review', secondary=True, case_ids=case_ids),
            _action('cancel_path_restart_review', 'Cancel', secondary=True, case_ids=case_ids),
        ),
        evidence=tuple(copy.deepcopy(approval.get('items') or ())),
        case_ids=case_ids,
        details={
            'workflow_stage': 'continuation_review_required',
            'active_plan_kind': 'path_restart',
            'scope': 'subset',
            'review_mode': 'batch',
            'items': copy.deepcopy(approval.get('items') or []),
            'approval_token': approval.get('approval_token'),
        },
    )


def path_consistency_gate(context: Dict[str, Any], invocation: GateInvocation) -> GateDecision:
    if not context.get('analysis_requested'):
        return _decision(invocation, GATE_STATUS_SKIPPED, 'Cross-case continuity review is deferred until result analysis is requested.')
    report = _report(context)
    diagnostics = report.get('scan_path_diagnostics')
    if not isinstance(diagnostics, dict):
        diagnostics = _adaptive(report).get('path_diagnostics')
    diagnostics = diagnostics if isinstance(diagnostics, dict) else {}
    if diagnostics.get('status') != 'review_required':
        return _decision(invocation, GATE_STATUS_PASSED, 'No unresolved cross-case continuity anomaly was detected.')
    anomalies = [item for item in diagnostics.get('anomalies') or [] if isinstance(item, dict)]
    case_ids = []
    for anomaly in anomalies:
        for key in ('case_id', 'left_case_id', 'right_case_id'):
            value = anomaly.get(key)
            if value is not None and str(value) not in case_ids:
                case_ids.append(str(value))
    return _decision(
        invocation,
        GATE_STATUS_REVIEW_REQUIRED,
        str(diagnostics.get('summary') or 'Cross-case continuity requires local refinement.'),
        checks=({
            'id': 'path_continuity',
            'status': 'review_required',
            'message': '{0} path anomaly or anomaly window(s) require review.'.format(len(anomalies)),
        },),
        actions=(
            _action('prepare_path_restart', 'Prepare 1RDM Restart', case_ids=case_ids),
            _action('promote_path_method', 'Promote Local Method', secondary=True, case_ids=case_ids),
            _action('acknowledge', 'Acknowledge', secondary=True, case_ids=case_ids),
        ),
        evidence=tuple(copy.deepcopy(anomalies)),
        case_ids=case_ids,
        details={
            'workflow_stage': 'path_consistency_review_required',
            'active_plan_kind': 'path_refinement',
            'scope': 'subset',
            'review_mode': 'batch',
            'items': copy.deepcopy(anomalies),
        },
    )


__all__ = [
    'active_space_approval_gate',
    'active_space_approval_items',
    'compilation_gate',
    'continuation_approval_gate',
    'execution_quality_gate',
    'initial_scan_quality_gate',
    'path_consistency_gate',
    'resource_feasibility_gate',
]
