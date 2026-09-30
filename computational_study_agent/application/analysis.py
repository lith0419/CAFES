"""Analysis use cases, composed by StudyApplicationService."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional
from ..gates.presentation import build_report_workflow
from .types import StudyFeatureUnavailableError


def prepare_result_analysis(
    self,
    report: Dict[str, Any],
    *,
    path_diagnostics: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    report_payload = self._require_report(report)
    cases = (
        report_payload.get('cases')
        if isinstance(report_payload.get('cases'), list)
        else []
    )
    comparison_table = (
        report_payload.get('comparison_table')
        if isinstance(report_payload.get('comparison_table'), list)
        else []
    )
    # A new Analyze action must inspect the current integrated study.
    # Approval callbacks may pass the diagnostic contract that created the
    # approval so the action cannot drift while it is being resolved.
    if not isinstance(path_diagnostics, dict):
        path_diagnostics = self._path_analyzer(report_payload)
    adaptive = (
        report_payload.get('adaptive')
        if isinstance(report_payload.get('adaptive'), dict)
        else {}
    )
    state_tracking = report_payload.get('dmrg_state_tracking')
    if not isinstance(state_tracking, dict):
        state_tracking = adaptive.get('dmrg_state_tracking')
    if not isinstance(state_tracking, dict):
        state_tracking = self._dmrg_state_tracker(report_payload)
    return {
        'name': report_payload.get('name'),
        'objective': report_payload.get('objective'),
        'system_type': report_payload.get('system_type'),
        'status': report_payload.get('status'),
        'summary': report_payload.get('summary'),
        'case_count': len(cases),
        'comparison_table': comparison_table,
        'scan_path_diagnostics': path_diagnostics,
        'dmrg_state_tracking': state_tracking,
    }


def _attach_dmrg_state_tracking(self, report: Dict[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(report)
    tracking = self._dmrg_state_tracker(payload)
    payload['dmrg_state_tracking'] = copy.deepcopy(tracking)
    adaptive = copy.deepcopy(payload.get('adaptive') or {})
    adaptive['dmrg_state_tracking'] = copy.deepcopy(tracking)
    payload['adaptive'] = adaptive
    work_dir = payload.get('work_dir')
    if tracking.get('case_ids') and isinstance(work_dir, str) and work_dir.strip():
        artifact = self._write_result_analysis_artifact(
            Path(work_dir).expanduser().resolve() / 'adaptive-dmrg-state-tracking.json',
            tracking,
            kind='adaptive-dmrg-state-tracking',
            description='Cross-task identity tracking for targeted low-energy block2 roots',
        )
        payload['artifacts'] = [
            item
            for item in (payload.get('artifacts') or [])
            if not (isinstance(item, dict) and item.get('path') == artifact['path'])
        ] + [artifact]
    return payload


def diagnose_results(
    self,
    report: Dict[str, Any],
    *,
    plan: Any = None,
    locale: str = 'en',
    path_restart_approval: Optional[Dict[str, Any]] = None,
    mps_continuation_approval: Optional[Dict[str, Any]] = None,
    resource_profile: Optional[str] = None,
) -> Dict[str, Any]:
    report_payload = copy.deepcopy(self._require_report(report))
    report_payload = self._attach_dmrg_state_tracking(report_payload)
    pending_path_diagnostics = None
    if isinstance(path_restart_approval, dict):
        stored_path_diagnostics = report_payload.get('scan_path_diagnostics')
        if isinstance(stored_path_diagnostics, dict):
            pending_path_diagnostics = stored_path_diagnostics
    analysis_package = self.prepare_result_analysis(
        report_payload,
        path_diagnostics=pending_path_diagnostics,
    )
    report_payload['scan_path_diagnostics'] = copy.deepcopy(
        analysis_package['scan_path_diagnostics']
    )
    adaptive = (
        copy.deepcopy(report_payload.get('adaptive'))
        if isinstance(report_payload.get('adaptive'), dict)
        else {}
    )
    adaptive['path_diagnostics'] = copy.deepcopy(
        analysis_package['scan_path_diagnostics']
    )
    report_payload['adaptive'] = adaptive
    adaptive['workflow'] = build_report_workflow(
        report_payload, analysis_requested=True
    )
    mps_continuation = self.run_result_mps_continuation(
        report_payload,
        plan,
        locale=locale,
        approval=mps_continuation_approval,
        resource_profile=resource_profile,
    )
    report_payload = mps_continuation['report']
    if mps_continuation.get('status') in ('validated', 'review_required'):
        report_payload = self._attach_dmrg_state_tracking(report_payload)
        analysis_package = self.prepare_result_analysis(report_payload)
    if mps_continuation.get('status') == 'approval_required':
        path_restart = {
            'status': 'not_required',
            'summary': 'Path continuation is deferred until the MPS continuation review is resolved.',
            'report': report_payload,
            'decisions': [],
            'validation': None,
            'approval': None,
        }
    else:
        path_restart = self.run_result_path_restarts(
            report_payload,
            plan,
            analysis_package['scan_path_diagnostics'],
            locale=locale,
            approval=path_restart_approval,
            resource_profile=resource_profile,
        )
    report_payload = path_restart['report']
    if path_restart.get('status') in ('validated', 'review_required'):
        analysis_package = self.prepare_result_analysis(
            report_payload,
            path_diagnostics=path_restart.get('path_diagnostics'),
        )
    path_refinement = None
    if (
        mps_continuation.get('status') != 'approval_required'
        and path_restart.get('status') != 'approval_required'
    ):
        path_refinement = self.build_result_path_refinement(
            report_payload,
            plan,
            analysis_package['scan_path_diagnostics'],
        )
    entanglement_active_space = None
    if (
        mps_continuation.get('status') != 'approval_required'
        and path_restart.get('status') != 'approval_required'
        and path_refinement is None
    ):
        entanglement_active_space = self.build_result_entanglement_active_space(
            report_payload,
            plan,
        )
    if isinstance(path_refinement, dict):
        path_plan = path_refinement.get('path_refinement_plan')
        if isinstance(path_plan, dict):
            report_payload = self._install_review_plan(
                report_payload,
                plan=path_plan,
                decisions=path_refinement.get('path_refinement_decisions'),
                plan_key='path_refinement_plan',
                decision_key='path_refinement_decisions',
                mode='result_path_review',
            )
    elif isinstance(entanglement_active_space, dict):
        entanglement_plan = entanglement_active_space.get(
            'entanglement_active_space_plan'
        )
        if isinstance(entanglement_plan, dict):
            report_payload = self._install_review_plan(
                report_payload,
                plan=entanglement_plan,
                decisions=entanglement_active_space.get(
                    'entanglement_active_space_decisions'
                ),
                plan_key='entanglement_active_space_plan',
                decision_key='entanglement_active_space_decisions',
                mode='entanglement_active_space_review',
            )
    report_payload['scan_path_diagnostics'] = copy.deepcopy(
        analysis_package['scan_path_diagnostics']
    )
    report_payload.setdefault('adaptive', {})['path_diagnostics'] = copy.deepcopy(
        analysis_package['scan_path_diagnostics']
    )
    public_path_restart = {
        key: copy.deepcopy(path_restart.get(key))
        for key in ('status', 'summary', 'decisions', 'validation', 'approval')
    }
    public_mps_continuation = {
        key: copy.deepcopy(mps_continuation.get(key))
        for key in ('status', 'summary', 'decisions', 'approval')
    }
    return {
        'analysis_package': analysis_package,
        'plot_specs': self.suggest_postprocessing(report_payload),
        'scan_path_diagnostics': analysis_package['scan_path_diagnostics'],
        'dmrg_state_tracking': analysis_package['dmrg_state_tracking'],
        'path_restart': public_path_restart,
        'mps_continuation': public_mps_continuation,
        'report': report_payload,
        'path_refinement': path_refinement,
        'entanglement_active_space': entanglement_active_space,
    }


def analyze_results(
    self,
    report,
    *,
    plan=None,
    llm_request_builder=None,
    locale='en',
    path_restart_approval=None,
    mps_continuation_approval=None,
    resource_profile=None,
):
    """Add optional language interpretation to the same deterministic scientific analysis."""
    if llm_request_builder is None or not hasattr(
        llm_request_builder, 'build_result_analysis'
    ):
        raise StudyFeatureUnavailableError('LLM result analysis is unavailable')
    result = self.diagnose_results(
        report,
        plan=plan,
        locale=locale,
        path_restart_approval=path_restart_approval,
        mps_continuation_approval=mps_continuation_approval,
        resource_profile=resource_profile,
    )
    return self._interpret_result_analysis(result, llm_request_builder, locale)


def _interpret_result_analysis(self, result, llm_request_builder, locale):
    report_payload = result['report']
    analysis_package = result['analysis_package']
    request_text = json.dumps(
        {
            'planner_study_result': analysis_package,
            'analysis_scope': 'Analyze the multi-case planner result, emphasizing trends in the comparison table.',
        },
        ensure_ascii=False,
    )
    execution_report = {
        'execution_status': report_payload.get('status'),
        'analysis_summary': report_payload.get('summary'),
    }
    analysis = llm_request_builder.build_result_analysis(
        request_text,
        execution_report,
        locale=locale,
    )
    if not analysis:
        raise StudyFeatureUnavailableError('LLM result analysis is unavailable')
    return {**result, 'result_analysis': analysis}


def analyze_study(
    self,
    study_id,
    *,
    work_dir=None,
    locale='en',
    postprocess=False,
    llm_request_builder=None,
    resource_profile=None,
):
    """Diagnose saved results, with optional language interpretation; no new calculations."""
    from .background import invocation_lock

    with invocation_lock(self._study_directory(study_id, work_dir)):
        result = self._analyze_saved_results(
            study_id,
            work_dir=work_dir,
            locale=locale,
            postprocess=postprocess,
            resource_profile=resource_profile,
        )
        if llm_request_builder is not None:
            return self._interpret_result_analysis(result, llm_request_builder, locale)
        return result


def _analyze_saved_results(
    self,
    study_id,
    *,
    work_dir=None,
    locale='en',
    postprocess=False,
    review=None,
    resource_profile=None,
):
    report = self.load_report(study_id, work_dir=work_dir)
    if any(
        (case.get('execution') or {}).get('pending')
        for case in report.get('cases') or []
    ) or any(
        (case.get('execution') or {}).get('pending')
        for case in self._saved_case_checkpoints(study_id, work_dir).values()
    ):
        raise ValueError('Collect pending task results before analysis')
    if not report.get('cases'):
        raise ValueError('No executed Study results are available for analysis')
    prepared = self.load_study_preparation(study_id, work_dir=work_dir)
    adaptive = report.get('adaptive') or {}
    plan = (
        prepared.get('plan')
        or adaptive.get('refined_plan')
        or adaptive.get('initial_scan_plan')
    )
    kwargs = {}
    if review is None:
        saved = report.get('pending_review') or {}
        if (saved.get('analysis_approval') or {}).get('decision') == 'skip':
            review = saved
    if review and review.get('analysis_approval'):
        key = (
            'mps_continuation_approval'
            if 'mps_continuation' in review['action_id']
            else 'path_restart_approval'
        )
        kwargs[key] = review['analysis_approval']
    result = self.diagnose_results(
        report, plan=plan, locale=locale, resource_profile=resource_profile, **kwargs
    )
    result['report']['workflow'] = copy.deepcopy(
        (result['report'].get('adaptive') or {}).get('workflow') or {}
    )
    if postprocess:
        result['postprocessing'] = self.run_postprocessing(result['report'])
        result['report']['postprocessing'] = copy.deepcopy(result['postprocessing'])
    self._save_report(result['report'], work_dir=work_dir)
    return result
