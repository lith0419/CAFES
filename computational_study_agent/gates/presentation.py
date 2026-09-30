from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from .plan_utils import case_ids_from_plan as _case_ids_from_plan

from pyscf_agent.workflow_gates import (
    GATE_STATUS_BLOCKED,
    GATE_STATUS_REVIEW_REQUIRED,
)
from pyscf_agent.lifecycle import (
    ensure_lifecycle,
    synchronize_study_report_lifecycle,
    transition_lifecycle,
)

from .actions import gate_action as _action
from .runtime import evaluate_study_gates


WORKFLOW_SCHEMA = 'pyscf-agent.adaptive-workflow.v1'

ACTIONABLE_GATE_PRIORITY = (
    'core.study_compilation',
    'study.continuation_approval',
    'study.active_space_approval',
    'study.resource_feasibility',
    'study.initial_scan_quality',
    'study.execution_quality',
    'study.path_consistency',
)


def _comparison_rows(report: Dict[str, Any]) -> List[Dict[str, Any]]:
    rows = report.get('comparison_table') if isinstance(report, dict) else []
    return [row for row in rows if isinstance(row, dict)] if isinstance(rows, list) else []


def _completed_case_ids(report: Dict[str, Any]) -> List[str]:
    return [
        str(row.get('case_id'))
        for row in _comparison_rows(report)
        if row.get('case_id') is not None and str(row.get('status') or '').strip().lower() == 'succeeded'
    ]


def _decision_map(decisions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    return {
        str(item.get('gate_id')): item
        for item in decisions
        if isinstance(item, dict) and item.get('gate_id')
    }


def _first_actionable(decisions: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    by_gate = _decision_map(decisions)
    for gate_id in ACTIONABLE_GATE_PRIORITY:
        decision = by_gate.get(gate_id)
        if decision and decision.get('status') in (GATE_STATUS_BLOCKED, GATE_STATUS_REVIEW_REQUIRED):
            return decision
    return None


def _workflow_from_decision(
    decision: Dict[str, Any],
    decisions: List[Dict[str, Any]],
    *,
    report: Optional[Dict[str, Any]] = None,
    lifecycle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    details = decision.get('details') if isinstance(decision.get('details'), dict) else {}
    case_ids = [str(item) for item in decision.get('affected_case_ids') or [] if str(item).strip()]
    review_mode = str(details.get('review_mode') or ('case_queue' if len(case_ids) > 1 else 'batch'))
    current_case_id = details.get('current_case_id')
    if review_mode == 'case_queue' and not current_case_id:
        current_case_id = case_ids[0] if case_ids else None
    current_case_ids = [current_case_id] if current_case_id else []
    completed = list(details.get('completed_case_ids') or [])
    failed = case_ids if decision.get('gate_id') in ('study.initial_scan_quality', 'study.execution_quality') else []
    workflow = {
        'schema': WORKFLOW_SCHEMA,
        'stage': details.get('workflow_stage') or 'review_required',
        'active_plan_kind': details.get('active_plan_kind') or 'refined',
        'scope': details.get('scope') or ('subset' if case_ids else 'full'),
        'pending_case_ids': case_ids,
        'completed_case_ids': completed,
        'failed_case_ids': failed,
        'review_mode': review_mode,
        'current_case_id': current_case_id if review_mode == 'case_queue' else None,
        'current_case_ids': current_case_ids if review_mode == 'case_queue' else [],
        'queue_position': 1 if review_mode == 'case_queue' and current_case_id else 0,
        'queue_total': len(case_ids) if review_mode == 'case_queue' else 0,
        'items': copy.deepcopy(details.get('items') or decision.get('evidence') or []),
        'allowed_actions': copy.deepcopy(decision.get('recommended_actions') or []),
        'message': decision.get('summary'),
        'gate_status': decision.get('status'),
        'gate_decision': copy.deepcopy(decision),
        'gate_decisions': copy.deepcopy(decisions),
        'lifecycle': copy.deepcopy(lifecycle or {}),
    }
    if details.get('cost_estimate') is not None:
        workflow['cost_estimate'] = copy.deepcopy(details['cost_estimate'])
    if details.get('approval_token') is not None:
        workflow['approval_token'] = details['approval_token']
    if report is not None and not completed:
        workflow['completed_case_ids'] = _completed_case_ids(report)
    return workflow


def build_initial_plan_workflow(
    initial_scan_plan: Dict[str, Any],
    cost_estimate: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    plan = copy.deepcopy(initial_scan_plan)
    if isinstance(cost_estimate, dict):
        plan['cost_estimate'] = copy.deepcopy(cost_estimate)
    context = evaluate_study_gates(
        {
            'study_plan': plan,
            'plan_kind': 'initial',
        },
        hooks=('study.after_compile', 'study.before_execute'),
    )
    decisions = copy.deepcopy(context.get('gate_decisions') or [])
    actionable = _first_actionable(decisions)
    lifecycle = ensure_lifecycle(
        plan.get('lifecycle'),
        'study',
        entity_id=plan.get('study_id'),
    )
    if actionable:
        if lifecycle.get('stage') != 'review_required':
            lifecycle = transition_lifecycle(
                lifecycle,
                'review_requested',
                details={'source': 'study_plan_gates'},
            )
        plan['lifecycle'] = copy.deepcopy(lifecycle)
        return _workflow_from_decision(
            actionable,
            decisions,
            lifecycle=lifecycle,
        )
    case_ids = _case_ids_from_plan(plan)
    return {
        'schema': WORKFLOW_SCHEMA,
        'stage': 'initial_plan_ready',
        'active_plan_kind': 'initial',
        'scope': 'full',
        'pending_case_ids': case_ids,
        'completed_case_ids': [],
        'failed_case_ids': [],
        'allowed_actions': [
            _action('run_initial_scan', 'Run Initial Scan', case_ids=case_ids),
        ],
        'message': 'Adaptive initial-scan plan is ready. Run the initial scan to compute diagnostics and choose refined methods.',
        'gate_status': 'passed',
        'gate_decisions': decisions,
        'lifecycle': lifecycle,
    }


def build_report_workflow(
    report: Dict[str, Any],
    *,
    analysis_requested: bool = False,
) -> Dict[str, Any]:
    context = evaluate_study_gates({
        'study_report': report,
        'analysis_requested': bool(analysis_requested),
    })
    decisions = copy.deepcopy(context.get('gate_decisions') or [])
    report['gate_configuration'] = copy.deepcopy(context.get('gate_configuration') or {})
    report['gate_provenance'] = copy.deepcopy(
        report['gate_configuration'].get('provenance') or report.get('gate_provenance') or {}
    )
    report['gate_decisions'] = decisions
    report['gate_execution_trace'] = copy.deepcopy(context.get('gate_execution_trace') or [])
    report['lifecycle'] = synchronize_study_report_lifecycle(
        report.get('lifecycle'),
        entity_id=report.get('study_id'),
        decisions=decisions,
        details={'source': 'study_report_gates'},
    )
    return build_report_workflow_from_gate_decisions(report, decisions)


def build_report_workflow_from_gate_decisions(
    report: Dict[str, Any],
    decisions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Present already-evaluated study gates as one workflow state."""

    actionable = _first_actionable(decisions)
    if actionable:
        return _workflow_from_decision(
            actionable,
            decisions,
            report=report,
            lifecycle=report['lifecycle'],
        )

    adaptive = report.get('adaptive') or {}
    pending_kind = adaptive.get('execution_pending_plan_kind')
    if pending_kind:
        plan = adaptive.get(pending_kind + '_plan') or {}
        return {
            'schema': WORKFLOW_SCHEMA, 'stage': 'plan_ready',
            'active_plan_kind': pending_kind, 'scope': 'full',
            'pending_case_ids': _case_ids_from_plan(plan),
            'completed_case_ids': _completed_case_ids(report), 'failed_case_ids': [],
            'allowed_actions': [], 'gate_status': 'passed', 'gate_decisions': decisions,
            'message': 'Results collected. Use Run Plan to execute the next plan.',
            'lifecycle': copy.deepcopy(report['lifecycle']),
        }

    status = str(report.get('status') or '').strip().lower()
    completed = _completed_case_ids(report)
    stage = 'completed' if status == 'succeeded' else 'completed_with_issues'
    return {
        'schema': WORKFLOW_SCHEMA,
        'stage': stage,
        'active_plan_kind': 'refined' if isinstance(report.get('adaptive'), dict) and report['adaptive'].get('refined_plan') else 'initial',
        'scope': 'full',
        'pending_case_ids': [],
        'completed_case_ids': completed,
        'failed_case_ids': [],
        'allowed_actions': [
            _action('suggest_plots', 'Suggest Plots', secondary=True),
            _action('acknowledge', 'Acknowledge', secondary=True),
        ],
        'message': 'Study execution is complete.' if stage == 'completed' else 'Study execution completed with nonblocking issues.',
        'gate_status': 'passed',
        'gate_decisions': decisions,
        'lifecycle': copy.deepcopy(report['lifecycle']),
    }


__all__ = [
    'WORKFLOW_SCHEMA',
    'build_initial_plan_workflow',
    'build_report_workflow',
    'build_report_workflow_from_gate_decisions',
]
