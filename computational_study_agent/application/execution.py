"""Execution use cases, composed by StudyApplicationService."""

from __future__ import annotations

from pyscf_agent.schema_contracts import STUDY_EXECUTION_STATUS_SCHEMA

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional
from ..gates.review_actions import _plan_kind
from ..executor import (
    collect_study as _backend_collect_study,
    reconcile_study_execution,
)
from ..study_state import read_json
from ..adaptive.executor import (
    collect_adaptive_study as _backend_collect_adaptive_study,
)
from ..execution_receipts import inspect_execution_tree
from ..schema import StudyPlan, StudyReport
from pyscf_agent.artifacts import default_artifact_repository



def run_study(
    self,
    plan_or_spec: Any,
    *,
    work_dir: Optional[str] = None,
    locale: str = 'en',
    postprocess: bool = False,
    postprocess_specs: Optional[List[Dict[str, Any]]] = None,
    resume: bool = True,
    rerun_case_ids: Optional[List[str]] = None,
    rerun_statuses: Optional[List[str]] = None,
    max_case_attempts: int = 2,
    batch_independent: bool = True,
    resource_profile: Optional[str] = None,
    study_report: Optional[Dict[str, Any]] = None,
    retry_guard: Optional[Dict[str, Any]] = None,
    grid_round_target: Optional[int] = None,
) -> StudyReport:
    if isinstance(plan_or_spec, StudyPlan):
        plan, issues = self.validate_plan(plan_or_spec)
        self._raise_for_issues('study_plan', issues)
    elif isinstance(plan_or_spec, dict) and isinstance(plan_or_spec.get('cases'), list):
        plan, issues = self.validate_plan(plan_or_spec)
        self._raise_for_issues('study_plan', issues)
    else:
        plan = self.build_plan(plan_or_spec, resource_profile=resource_profile)
    plan = self._bind_plan_resources(plan, resource_profile)
    if isinstance(plan.workflow_provenance.get('direct_active_space_probe'), dict):
        study_report = None
    runner_kwargs = {
        'work_dir': work_dir,
        'locale': locale,
        'postprocess': postprocess,
        'postprocess_specs': postprocess_specs,
        'resume': resume,
        'rerun_case_ids': rerun_case_ids,
        'rerun_statuses': rerun_statuses,
        'max_case_attempts': max_case_attempts,
        'task_executor': self._task_executor,
        'batch_independent': batch_independent,
    }
    if resource_profile:
        runner_kwargs['resource_profile'] = resource_profile
    if retry_guard is not None:
        runner_kwargs['retry_guard'] = retry_guard
    if grid_round_target is not None:
        runner_kwargs['grid_round_target'] = grid_round_target
    if study_report is not None:
        runner_kwargs['study_report'] = copy.deepcopy(study_report)
    if study_report is not None or (
        isinstance(plan_or_spec, dict) and plan_or_spec.get('_review_kind')
    ):
        runner_kwargs['review_kind'] = _plan_kind(
            plan_or_spec if isinstance(plan_or_spec, dict) else plan.to_dict(),
        )
    report = self._study_runner(plan, **runner_kwargs)
    return self._finish_probe_review(plan, report)


def _finish_probe_review(self, plan: StudyPlan, report: StudyReport) -> StudyReport:
    """Install the existing probe review consistently for every application caller."""
    if not isinstance(plan.workflow_provenance.get('direct_active_space_probe'), dict):
        return report
    if (report.adaptive or {}).get('direct_active_space_review_report'):
        return report
    review = self.build_active_space_review_from_probe(plan.to_dict(), report.to_dict())
    if isinstance(review, dict) and isinstance(review.get('report'), dict):
        report = StudyReport.from_dict(review['report'])
        report.workflow = copy.deepcopy(report.adaptive['workflow'])
        directory = Path(report.work_dir)
        path = directory / 'study-report.json'
        if (directory / 'adaptive-study-report.json').exists():
            path = directory / 'adaptive-study-report.json'
        default_artifact_repository().write_json(
            path, report.to_dict(), kind=path.stem, atomic=True
        )
    return report


def run_adaptive_study(
    self,
    spec: Any,
    *,
    options: Optional[Dict[str, Any]] = None,
    work_dir: Optional[str] = None,
    locale: str = 'en',
    resume_study_id: Optional[str] = None,
    requested_study_id: Optional[str] = None,
    resource_profile: Optional[str] = None,
    lifecycle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    study_spec = self._bind_spec_resources(self._coerce_spec(spec), resource_profile)
    self.build_adaptive_plan(
        study_spec,
        options=options,
        resource_profile=resource_profile,
    )
    runner_kwargs = {
        'options': options or {},
        'work_dir': work_dir,
        'locale': locale,
        'resume_study_id': resume_study_id,
        'requested_study_id': requested_study_id,
        'task_executor': self._task_executor,
        'lifecycle': lifecycle,
    }
    if resource_profile:
        runner_kwargs['resource_profile'] = resource_profile
    return self._adaptive_runner(study_spec, **runner_kwargs)


def reconcile_execution(
    self, plan_or_spec: Any, *, submission_path: str, work_dir=None, receipt_file=None
) -> Dict[str, Any]:
    plan, issues = self.validate_plan(plan_or_spec)
    self._raise_for_issues('study_plan', issues)
    return reconcile_study_execution(
        plan,
        submission_path=submission_path,
        work_dir=work_dir,
        receipt_file=receipt_file,
        task_executor=self._task_executor,
    )


def collect_study(
    self, plan_or_spec: Any, *, work_dir=None, locale='en'
) -> StudyReport:
    """Fetch an existing execution; this use case cannot submit calculations."""
    if isinstance(plan_or_spec, str):
        work_dir = str(self._study_directory(plan_or_spec, work_dir).parent)
        plan_or_spec = self.load_plan(plan_or_spec, work_dir=work_dir)
    plan, issues = self.validate_plan(plan_or_spec)
    self._raise_for_issues('study_plan', issues)
    report = _backend_collect_study(
        plan, work_dir=work_dir, locale=locale, task_executor=self._task_executor
    )
    return self._finish_probe_review(plan, report)


def collect_adaptive_study(
    self,
    spec: Any,
    *,
    study_id: str,
    options=None,
    work_dir=None,
    locale='en',
    resource_profile=None,
    lifecycle=None,
):
    study_spec = self._bind_spec_resources(self._coerce_spec(spec), resource_profile)
    return _backend_collect_adaptive_study(
        study_spec,
        study_id=study_id,
        options=options,
        work_dir=work_dir,
        locale=locale,
        task_executor=self._task_executor,
        resource_profile=resource_profile,
        lifecycle=lifecycle,
    )


def inspect_execution(
    self,
    study_id: str,
    *,
    work_dir: Optional[str] = None,
) -> Dict[str, Any]:
    normalized_study_id = str(study_id or '').strip()
    if not normalized_study_id:
        raise ValueError('study_id is required to inspect an execution')
    study_root = self._study_directory(normalized_study_id, work_dir)
    root = study_root.parent
    try:
        study_root.relative_to(root)
    except ValueError as exc:
        raise ValueError('study_id escapes the configured work directory') from exc
    if self._task_executor is None:
        result = {
            'schema': STUDY_EXECUTION_STATUS_SCHEMA,
            'status': 'not_found',
            'expected': 0,
            'terminal': 0,
            'report_available': 0,
            'receipt_count': 0,
            'receipts': [],
            'can_collect': False,
        }
    else:
        result = inspect_execution_tree(study_root, self._task_executor)
    result['study_id'] = normalized_study_id
    result['work_dir'] = str(study_root)
    result['executor'] = (
        self._task_executor.describe() if self._task_executor is not None else {}
    )
    if (study_root / 'study-plan.json').is_file():
        plan = self.load_plan(normalized_study_id, work_dir=work_dir)
        state_path = study_root / 'study-state.json'
        state = read_json(state_path) if state_path.is_file() else {}
        counts = {}
        for case in plan['cases']:
            checkpoint = (state.get('cases') or {}).get(case['case_id'], {})
            status = (
                'pending'
                if (checkpoint.get('execution') or {}).get('pending')
                else (
                    checkpoint.get('task_summary')
                    or checkpoint.get('task_report')
                    or {}
                ).get('execution_status', 'not_executed')
            )
            counts[status] = counts.get(status, 0) + 1
        result['task_count'] = len(plan['cases'])
        result['task_status_counts'] = counts
        if (plan.get('grid_refinement') or {}).get('enabled'):
            from computational_study_agent.grid.execution import live_grid_progress

            progress = live_grid_progress(plan, study_root)
            result['progress'] = progress
            result['task_count'] = progress['task_count']
            result['task_status_counts'] = progress['task_status_counts']
        if result['status'] == 'not_found' and not state:
            result['status'] = 'prepared'
    from .background import inspect_invocation

    invocation = inspect_invocation(study_root) if study_root.is_dir() else None
    if invocation:
        result['agent'] = invocation
        if invocation['running']:
            result.update(status='running', can_collect=False)
            grid = (result.get('progress') or {}).get('grid_refinement')
            if grid:
                grid.update(status='running', sampling_satisfied=False)
        elif invocation['error'] or invocation['interrupted']:
            result['status'] = (
                'review_required'
                if (invocation.get('error') or {}).get('code') == 'CostApprovalRequired'
                else 'interrupted'
            )
    if not invocation or not invocation['running']:
        adaptive_report = study_root / 'adaptive-study-report.json'
        report_path = (
            adaptive_report
            if adaptive_report.exists()
            else study_root / 'study-report.json'
        )
        if report_path.is_file():
            report = read_json(report_path)
            result['study_status'] = report.get('status')
            if report.get('pending_review'):
                result['pending_review'] = {
                    key: report['pending_review'][key]
                    for key in (
                        'action_id',
                        'status',
                        'can_run',
                        'case_ids',
                        'workflow',
                    )
                    if key in report['pending_review']
                }
            result['workflow'] = report.get('workflow') or (
                report.get('adaptive') or {}
            ).get('workflow')
            if invocation and invocation['finished_at'] and not invocation['error']:
                result.update(status='results_available', can_collect=True)
        elif (study_root / 'adaptive-study-request.json').exists() and not invocation:
            result['status'] = 'prepared'
    if invocation and invocation.get('cancelled') and not invocation['running']:
        result['status'] = 'cancelled'
    active_jobs = any(
        (receipt.get('summary', {}).get('terminal', 0)
         < receipt.get('summary', {}).get('expected', 0))
        for receipt in result.get('receipts', [])
    )
    result['can_stop'] = bool(
        result['executor'].get('executor_id') == 'local-process'
        and invocation and (invocation['running'] or active_jobs
                            or (invocation.get('error') or {}).get('code') == 'StudyCancellationIncomplete')
    )
    return result
