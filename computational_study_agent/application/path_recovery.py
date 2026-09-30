"""Path recovery use cases, composed by StudyApplicationService."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..gates.presentation import build_report_workflow
from ..adaptive.refinement import path_window_restart_approval
from ..schema import StudyPlan


def run_result_path_restarts(
    self,
    report: Dict[str, Any],
    plan: Any,
    path_diagnostics: Dict[str, Any],
    *,
    locale: str = 'en',
    approval: Optional[Dict[str, Any]] = None,
    resource_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Retry a non-smooth molecular path before proposing method promotion."""
    report_payload = copy.deepcopy(self._require_report(report))
    result = {
        'status': 'not_required',
        'summary': 'No projected-1RDM continuation restart was required.',
        'report': report_payload,
        'path_diagnostics': copy.deepcopy(path_diagnostics),
        'plans': None,
        'decisions': [],
        'reports': {},
        'validation': None,
        'approval': None,
    }
    if path_diagnostics.get('status') != 'review_required' or not isinstance(
        plan, dict
    ):
        return result
    try:
        study_plan = StudyPlan.from_dict(plan)
    except (TypeError, ValueError):
        result['status'] = 'unavailable'
        result['summary'] = (
            'Continuation restart requires the executable refined StudyPlan.'
        )
        return result
    system_type = str(study_plan.system_type or '').strip().lower().replace('-', '_')
    if system_type not in ('molecular', 'molecule', 'molecular_electronic_structure'):
        return result

    adaptive = (
        report_payload.get('adaptive')
        if isinstance(report_payload.get('adaptive'), dict)
        else {}
    )
    existing_decisions = (
        adaptive.get('path_window_restart_decisions')
        if isinstance(adaptive.get('path_window_restart_decisions'), list)
        else []
    )
    existing_validation = adaptive.get('path_window_restart_validation')
    existing_reports = (
        adaptive.get('path_window_restart_reports')
        if isinstance(adaptive.get('path_window_restart_reports'), dict)
        else {}
    )
    supplied_decision = (
        str((approval or {}).get('decision') or '').strip().lower()
        if isinstance(approval, dict)
        else ''
    )
    pending_approval = (
        isinstance(existing_validation, dict)
        and existing_validation.get('status') in ('approval_required', 'approved')
    ) or supplied_decision in ('approve', 'skip')
    if (existing_decisions and existing_reports) or (
        isinstance(existing_validation, dict) and not pending_approval
    ):
        result['status'] = 'already_attempted'
        result['summary'] = (
            'Projected-1RDM continuation restart was already attempted for this path anomaly; '
            'the stored validation is reused.'
        )
        result['decisions'] = copy.deepcopy(existing_decisions)
        result['validation'] = copy.deepcopy(existing_validation)
        return result

    decisions = (
        adaptive.get('decision_log')
        if isinstance(adaptive.get('decision_log'), list)
        else []
    )
    options = (
        adaptive.get('options') if isinstance(adaptive.get('options'), dict) else None
    )
    restart_payload = self._path_restart_builder(
        study_plan,
        report_payload,
        decisions,
        path_diagnostics,
        options,
    )
    if not isinstance(restart_payload, dict):
        result['status'] = 'unavailable'
        result['summary'] = (
            'The path anomaly has no compatible starting 1RDM for both propagation directions; '
            'method-refinement review is required.'
        )
        return result

    plans = restart_payload.get('path_window_restart_plans')
    restart_decisions = restart_payload.get('path_window_restart_decisions') or []
    if not isinstance(plans, dict) or not restart_decisions:
        result['status'] = 'unavailable'
        result['summary'] = (
            'No executable projected-1RDM continuation candidates were produced.'
        )
        return result
    approval_contract = (
        copy.deepcopy(restart_payload.get('path_window_restart_approval'))
        if isinstance(restart_payload.get('path_window_restart_approval'), dict)
        else path_window_restart_approval(restart_decisions)
    )
    if approval_contract is not None:
        supplied_approval = approval if isinstance(approval, dict) else {}
        supplied_token = str(supplied_approval.get('approval_token') or '')
        expected_token = str(approval_contract.get('approval_token') or '')
        decision = str(supplied_approval.get('decision') or '').strip().lower()
        token_matches = bool(supplied_token) and supplied_token == expected_token
        stored_review = (
            adaptive.get('path_window_restart_review')
            if isinstance(adaptive.get('path_window_restart_review'), dict)
            else {}
        )
        stored_skip = (
            stored_review.get('approval_token') == expected_token
            and stored_review.get('decision') == 'skip'
        )
        if decision == 'skip' and token_matches:
            skipped_report = copy.deepcopy(report_payload)
            skipped_adaptive = (
                copy.deepcopy(skipped_report.get('adaptive'))
                if isinstance(skipped_report.get('adaptive'), dict)
                else {}
            )
            skipped_adaptive['path_window_restart_review'] = {
                'schema': approval_contract['schema'],
                'approval_token': expected_token,
                'decision': 'skip',
            }
            skipped_adaptive['path_window_restart_validation'] = {
                'schema': 'pyscf-agent.path-window-restart-validation.v1',
                'kind': 'path_window_restart_validation',
                'status': 'skipped',
                'accepted_case_ids': [],
                'unresolved_case_ids': approval_contract['case_ids'],
                'comparisons': [],
                'summary': 'Projected-1RDM continuation was skipped by review.',
            }
            skipped_adaptive['path_window_restart_approval'] = {
                **copy.deepcopy(approval_contract),
                'status': 'skipped',
                'decision': 'skip',
            }
            skipped_report['adaptive'] = skipped_adaptive
            skipped_report['scan_path_diagnostics'] = copy.deepcopy(path_diagnostics)
            approval_contract['status'] = 'skipped'
            approval_contract['decision'] = 'skip'
            result.update(
                {
                    'status': 'skipped',
                    'summary': (
                        'Projected-1RDM continuation was skipped by review; '
                        'method-refinement review is the next step.'
                    ),
                    'report': skipped_report,
                    'plans': copy.deepcopy(plans),
                    'decisions': copy.deepcopy(restart_decisions),
                    'approval': approval_contract,
                }
            )
            return result
        if stored_skip and not decision:
            approval_contract['status'] = 'skipped'
            approval_contract['decision'] = 'skip'
            result.update(
                {
                    'status': 'skipped',
                    'summary': (
                        'Projected-1RDM continuation remains skipped by the stored review decision.'
                    ),
                    'plans': copy.deepcopy(plans),
                    'decisions': copy.deepcopy(restart_decisions),
                    'approval': approval_contract,
                }
            )
            return result
        if decision != 'approve' or not token_matches:
            pending_report = copy.deepcopy(report_payload)
            pending_adaptive = (
                copy.deepcopy(pending_report.get('adaptive'))
                if isinstance(pending_report.get('adaptive'), dict)
                else {}
            )
            pending_adaptive['path_window_restart_plans'] = copy.deepcopy(plans)
            pending_adaptive['path_window_restart_decisions'] = copy.deepcopy(
                restart_decisions
            )
            pending_adaptive['path_window_restart_approval'] = copy.deepcopy(
                approval_contract
            )
            pending_adaptive['path_window_restart_validation'] = {
                'schema': 'pyscf-agent.path-window-restart-validation.v1',
                'kind': 'path_window_restart_validation',
                'status': 'approval_required',
                'accepted_case_ids': [],
                'unresolved_case_ids': approval_contract['case_ids'],
                'comparisons': [],
                'summary': approval_contract['summary'],
            }
            pending_adaptive['path_window_restart_review'] = {
                'schema': approval_contract['schema'],
                'approval_token': expected_token,
                'decision': 'pending',
            }
            pending_report['adaptive'] = pending_adaptive
            pending_report['scan_path_diagnostics'] = copy.deepcopy(path_diagnostics)
            pending_report['adaptive']['workflow'] = build_report_workflow(
                pending_report,
                analysis_requested=True,
            )
            result.update(
                {
                    'status': 'approval_required',
                    'summary': approval_contract['summary'],
                    'report': pending_report,
                    'plans': copy.deepcopy(plans),
                    'decisions': copy.deepcopy(restart_decisions),
                    'validation': copy.deepcopy(
                        pending_adaptive['path_window_restart_validation']
                    ),
                    'approval': approval_contract,
                }
            )
            return result
        approval_contract['status'] = 'approved'
        approval_contract['decision'] = 'approve'

    refinement_lifecycle = self._begin_approved_refinement(
        report_payload,
        source='path_window_restart_review',
        case_ids=[
            str(item.get('case_id'))
            for item in restart_decisions
            if isinstance(item, dict) and item.get('case_id') is not None
        ],
    )

    report_work_dir = report_payload.get('work_dir')
    if not isinstance(report_work_dir, str) or not report_work_dir.strip():
        result['status'] = 'unavailable'
        result['summary'] = (
            'Continuation restart requires the parent StudyReport work_dir.'
        )
        return result
    restart_root = (
        Path(report_work_dir).expanduser().resolve() / 'continuation-restarts'
    )
    restart_reports: Dict[str, Dict[str, Any]] = {}
    for direction in ('left', 'right'):
        plan_payload = plans.get(direction)
        if not isinstance(plan_payload, dict):
            continue
        runner_kwargs = {
            'work_dir': str(restart_root),
            'locale': locale,
            'resume': False,
            'task_executor': self._task_executor,
            'batch_independent': False,
        }
        if resource_profile:
            runner_kwargs['resource_profile'] = resource_profile
        branch_report = self._study_runner(
            StudyPlan.from_dict(
                self._attach_parent_lifecycle(
                    plan_payload,
                    refinement_lifecycle,
                )
            ),
            **runner_kwargs,
        )
        restart_reports[direction] = self._report_payload(branch_report)

    if {'left', 'right'} <= set(restart_reports):
        validation = self._path_restart_evaluator(
            restart_reports['left'],
            restart_reports['right'],
            restart_decisions,
        )
        merged_report = self._path_restart_merger(
            report_payload,
            restart_reports['left'],
            restart_reports['right'],
            decisions,
            restart_decisions,
            validation,
        )
    else:
        validation = {
            'schema': 'pyscf-agent.path-window-restart-validation.v1',
            'kind': 'path_window_restart_validation',
            'status': 'review_required',
            'accepted_case_ids': [],
            'unresolved_case_ids': [
                str(item.get('case_id'))
                for item in restart_decisions
                if isinstance(item, dict) and item.get('case_id') is not None
            ],
            'comparisons': [],
            'summary': 'Both continuation branches must complete before a replacement can be accepted.',
        }
        merged_report = report_payload

    merged_adaptive = (
        copy.deepcopy(merged_report.get('adaptive'))
        if isinstance(merged_report.get('adaptive'), dict)
        else {}
    )
    merged_adaptive['decision_log'] = decisions
    merged_adaptive['path_window_restart_plans'] = copy.deepcopy(plans)
    merged_adaptive['path_window_restart_decisions'] = copy.deepcopy(restart_decisions)
    merged_adaptive['path_window_restart_reports'] = copy.deepcopy(restart_reports)
    merged_adaptive['path_window_restart_validation'] = copy.deepcopy(validation)
    if approval_contract is not None:
        merged_adaptive['path_window_restart_review'] = {
            'schema': approval_contract['schema'],
            'approval_token': approval_contract['approval_token'],
            'decision': 'approve',
        }
        merged_adaptive['path_window_restart_approval'] = {
            **copy.deepcopy(approval_contract),
            'status': 'completed',
            'decision': 'approve',
        }
    merged_report['adaptive'] = merged_adaptive
    merged_report['lifecycle'] = copy.deepcopy(refinement_lifecycle)

    plan_artifact = self._write_result_analysis_artifact(
        restart_root / 'adaptive-path-window-restart-plan.json',
        restart_payload,
        kind='adaptive-path-window-restart-plan',
        description='Bidirectional same-method continuation candidates using projected one-particle density matrices',
    )
    validation_artifact = self._write_result_analysis_artifact(
        restart_root / 'adaptive-path-window-restart-validation.json',
        validation,
        kind='adaptive-path-window-restart-validation',
        description='Consistency checks for independently restarted local path candidates',
    )
    merged_artifacts = list(merged_report.get('artifacts') or [])
    for branch_report in restart_reports.values():
        merged_artifacts.extend(branch_report.get('artifacts') or [])
    merged_artifacts.extend((plan_artifact, validation_artifact))
    merged_report['artifacts'] = merged_artifacts

    updated_diagnostics = self._path_analyzer(merged_report)
    merged_report['adaptive']['path_diagnostics'] = copy.deepcopy(updated_diagnostics)
    merged_report['scan_path_diagnostics'] = copy.deepcopy(updated_diagnostics)
    merged_report['adaptive']['workflow'] = build_report_workflow(
        merged_report,
        analysis_requested=True,
    )
    current_validation_artifact = self._write_result_analysis_artifact(
        Path(report_work_dir).expanduser().resolve()
        / 'adaptive-path-window-restart-validation.json',
        validation,
        kind='adaptive-path-window-restart-validation',
        description='Current bidirectional continuation validation state',
    )
    merged_report['artifacts'] = [
        artifact
        for artifact in merged_report.get('artifacts') or []
        if not (
            isinstance(artifact, dict)
            and artifact.get('path') == current_validation_artifact['path']
        )
    ] + [current_validation_artifact]
    self._write_result_analysis_artifact(
        Path(report_work_dir).expanduser().resolve() / 'adaptive-study-report.json',
        merged_report,
        kind='adaptive-study-report',
        description='Adaptive study report updated after continuation review',
    )
    result.update(
        {
            'status': validation.get('status') or 'review_required',
            'summary': validation.get('summary')
            or 'Projected-1RDM continuation restart completed.',
            'report': merged_report,
            'path_diagnostics': updated_diagnostics,
            'plans': copy.deepcopy(plans),
            'decisions': copy.deepcopy(restart_decisions),
            'reports': copy.deepcopy(restart_reports),
            'validation': copy.deepcopy(validation),
            'approval': copy.deepcopy(approval_contract),
        }
    )
    return result


def build_result_path_refinement(
    self,
    report: Dict[str, Any],
    plan: Any,
    path_diagnostics: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    if path_diagnostics.get('status') != 'review_required' or not isinstance(
        plan, dict
    ):
        return None
    adaptive = (
        report.get('adaptive') if isinstance(report.get('adaptive'), dict) else {}
    )
    full_plan = (
        adaptive.get('refined_plan')
        if isinstance(adaptive.get('refined_plan'), dict)
        else None
    )
    analysis_plan = full_plan or plan
    try:
        study_plan = StudyPlan.from_dict(analysis_plan)
    except (TypeError, ValueError):
        return None
    system_type = str(study_plan.system_type or '').strip().lower().replace('-', '_')
    if system_type not in ('molecular', 'molecule', 'molecular_electronic_structure'):
        return None
    decisions = (
        adaptive.get('decision_log')
        if isinstance(adaptive.get('decision_log'), list)
        else []
    )
    options = (
        adaptive.get('options') if isinstance(adaptive.get('options'), dict) else None
    )
    try:
        result = self._path_refinement_builder(
            study_plan,
            report,
            decisions,
            path_diagnostics,
            options,
        )
    except (TypeError, ValueError):
        return None
    if self._completed_path_refinement_matches(report, result):
        return None
    return result


def _completed_path_refinement_matches(
    report: Dict[str, Any],
    candidate: Any,
) -> bool:
    """Return true when the same local CASSCF window already succeeded."""

    if not isinstance(candidate, dict):
        return False
    adaptive = (
        report.get('adaptive') if isinstance(report.get('adaptive'), dict) else {}
    )
    previous_plan = adaptive.get('path_refinement_plan')
    previous_report = adaptive.get('path_refinement_report')
    candidate_plan = candidate.get('path_refinement_plan')
    if not all(
        isinstance(item, dict)
        for item in (previous_plan, previous_report, candidate_plan)
    ):
        return False

    def contract(plan_payload: Dict[str, Any]) -> List[Any]:
        rows = []
        for case in plan_payload.get('cases') or []:
            if not isinstance(case, dict):
                continue
            request = (
                case.get('request') if isinstance(case.get('request'), dict) else {}
            )
            active_space = (
                request.get('active_space')
                if isinstance(request.get('active_space'), dict)
                else {}
            )
            rows.append(
                (
                    str(case.get('case_id') or ''),
                    str(request.get('method') or request.get('solver') or '')
                    .strip()
                    .lower(),
                    active_space.get('ncas'),
                    active_space.get('nelecas'),
                )
            )
        return rows

    if contract(previous_plan) != contract(candidate_plan):
        return False
    completed_rows = {
        str(row.get('case_id')): str(row.get('status') or '').strip().lower()
        for row in previous_report.get('comparison_table') or []
        if isinstance(row, dict) and row.get('case_id') is not None
    }
    case_ids = [item[0] for item in contract(candidate_plan)]
    return bool(case_ids) and all(
        completed_rows.get(case_id) == 'succeeded' for case_id in case_ids
    )


def build_result_entanglement_active_space(
    self,
    report: Dict[str, Any],
    plan: Any,
) -> Optional[Dict[str, Any]]:
    if not isinstance(plan, dict):
        return None
    try:
        return self._entanglement_active_space_builder(plan, report)
    except (TypeError, ValueError):
        return None
