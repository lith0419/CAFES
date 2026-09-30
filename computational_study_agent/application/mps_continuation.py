"""Mps continuation use cases, composed by StudyApplicationService."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, Optional
from ..gates.presentation import build_report_workflow
from ..schema import StudyPlan


def run_result_mps_continuation(
    self,
    report: Dict[str, Any],
    plan: Any,
    *,
    locale: str = 'en',
    approval: Optional[Dict[str, Any]] = None,
    resource_profile: Optional[str] = None,
) -> Dict[str, Any]:
    """Retry unresolved block2 cases from compatible succeeded MPS checkpoints."""

    report_payload = copy.deepcopy(self._require_report(report))
    result = {
        'status': 'not_required',
        'summary': 'No compatible block2 MPS continuation was required.',
        'report': report_payload,
        'plan': None,
        'decisions': [],
        'approval': None,
    }
    adaptive = (
        report_payload.get('adaptive')
        if isinstance(report_payload.get('adaptive'), dict)
        else {}
    )
    existing_report = adaptive.get('mps_continuation_report')
    if isinstance(existing_report, dict):
        result.update(
            {
                'status': 'already_attempted',
                'summary': 'The stored block2 MPS continuation result is reused.',
                'decisions': copy.deepcopy(
                    adaptive.get('mps_continuation_decisions') or []
                ),
                'approval': copy.deepcopy(adaptive.get('mps_continuation_approval')),
            }
        )
        return result
    payload = self._mps_continuation_builder(plan, report_payload)
    if not isinstance(payload, dict):
        return result
    continuation_plan = payload.get('mps_continuation_plan')
    decisions = payload.get('mps_continuation_decisions') or []
    approval_contract = payload.get('mps_continuation_approval')
    if (
        not isinstance(continuation_plan, dict)
        or not decisions
        or not isinstance(approval_contract, dict)
    ):
        return result
    source_executor = (
        decisions[0].get('source_executor') if isinstance(decisions[0], dict) else None
    )
    describe_executor = getattr(self._task_executor, 'describe', None)
    current_executor = describe_executor() if callable(describe_executor) else None
    identity_fields = (
        'executor_id',
        'execution_mode',
        'location',
        'transport',
        'remote_profile',
    )
    if (
        isinstance(source_executor, dict)
        and source_executor
        and isinstance(current_executor, dict)
    ):
        source_identity = {
            key: source_executor.get(key)
            for key in identity_fields
            if source_executor.get(key) is not None
        }
        current_identity = {
            key: current_executor.get(key)
            for key in identity_fields
            if current_executor.get(key) is not None
        }
        if source_identity != current_identity:
            result.update(
                {
                    'status': 'unavailable',
                    'summary': 'block2 MPS continuation must use the same execution target and remote profile as the source checkpoint.',
                    'plan': copy.deepcopy(continuation_plan),
                    'decisions': copy.deepcopy(decisions),
                    'approval': copy.deepcopy(approval_contract),
                }
            )
            return result
    supplied = approval if isinstance(approval, dict) else {}
    expected_token = str(approval_contract.get('approval_token') or '')
    supplied_token = str(supplied.get('approval_token') or '')
    decision = str(supplied.get('decision') or '').strip().lower()
    token_matches = bool(supplied_token) and supplied_token == expected_token
    if decision == 'skip' and token_matches:
        adaptive = copy.deepcopy(adaptive)
        adaptive['mps_continuation_plan'] = copy.deepcopy(continuation_plan)
        adaptive['mps_continuation_decisions'] = copy.deepcopy(decisions)
        adaptive['mps_continuation_approval'] = {
            **copy.deepcopy(approval_contract),
            'status': 'skipped',
            'decision': 'skip',
        }
        report_payload['adaptive'] = adaptive
        adaptive['workflow'] = build_report_workflow(
            report_payload,
            analysis_requested=True,
        )
        result.update(
            {
                'status': 'skipped',
                'summary': 'block2 MPS continuation was skipped; ordinary path and method review remain available.',
                'report': report_payload,
                'plan': copy.deepcopy(continuation_plan),
                'decisions': copy.deepcopy(decisions),
                'approval': copy.deepcopy(adaptive['mps_continuation_approval']),
            }
        )
        return result
    if decision != 'approve' or not token_matches:
        adaptive = copy.deepcopy(adaptive)
        adaptive['mps_continuation_plan'] = copy.deepcopy(continuation_plan)
        adaptive['mps_continuation_decisions'] = copy.deepcopy(decisions)
        adaptive['mps_continuation_approval'] = copy.deepcopy(approval_contract)
        report_payload['adaptive'] = adaptive
        adaptive['workflow'] = build_report_workflow(
            report_payload, analysis_requested=True
        )
        result.update(
            {
                'status': 'approval_required',
                'summary': approval_contract['summary'],
                'report': report_payload,
                'plan': copy.deepcopy(continuation_plan),
                'decisions': copy.deepcopy(decisions),
                'approval': copy.deepcopy(approval_contract),
            }
        )
        return result

    refinement_lifecycle = self._begin_approved_refinement(
        report_payload,
        source='mps_continuation_review',
        case_ids=[
            str(item.get('case_id'))
            for item in decisions
            if isinstance(item, dict) and item.get('case_id') is not None
        ],
    )

    work_dir = report_payload.get('work_dir')
    if not isinstance(work_dir, str) or not work_dir.strip():
        result.update(
            {
                'status': 'unavailable',
                'summary': 'block2 MPS continuation requires the parent StudyReport work_dir.',
                'plan': copy.deepcopy(continuation_plan),
                'decisions': copy.deepcopy(decisions),
                'approval': copy.deepcopy(approval_contract),
            }
        )
        return result
    restart_root = Path(work_dir).expanduser().resolve() / 'mps-continuation-restarts'
    runner_kwargs = {
        'work_dir': str(restart_root),
        'locale': locale,
        'resume': False,
        'task_executor': self._task_executor,
        'batch_independent': False,
    }
    if resource_profile:
        runner_kwargs['resource_profile'] = resource_profile
    continuation_report = self._study_runner(
        StudyPlan.from_dict(
            self._attach_parent_lifecycle(
                continuation_plan,
                refinement_lifecycle,
            )
        ),
        **runner_kwargs,
    )
    continuation_report_payload = self._report_payload(continuation_report)
    merged = self._mps_continuation_merger(
        report_payload,
        continuation_report_payload,
        decisions,
    )
    merged_adaptive = copy.deepcopy(merged.get('adaptive') or {})
    merged_adaptive['mps_continuation_plan'] = copy.deepcopy(continuation_plan)
    merged_adaptive['mps_continuation_decisions'] = copy.deepcopy(decisions)
    merged_adaptive['mps_continuation_approval'] = {
        **copy.deepcopy(approval_contract),
        'status': 'completed',
        'decision': 'approve',
    }
    merged['adaptive'] = merged_adaptive
    merged['lifecycle'] = copy.deepcopy(refinement_lifecycle)
    merged_adaptive['workflow'] = build_report_workflow(
        merged,
        analysis_requested=True,
    )
    plan_artifact = self._write_result_analysis_artifact(
        restart_root / 'adaptive-mps-continuation-plan.json',
        payload,
        kind='adaptive-mps-continuation-plan',
        description='Approved same-method block2 MPS continuation plan and provenance',
    )
    merged['artifacts'] = list(merged.get('artifacts') or []) + [plan_artifact]
    self._write_result_analysis_artifact(
        Path(work_dir).expanduser().resolve() / 'adaptive-study-report.json',
        merged,
        kind='adaptive-study-report',
        description='Adaptive study report updated after block2 MPS continuation',
    )
    succeeded = all(
        str(item.get('status') or '').lower() == 'succeeded'
        for item in continuation_report_payload.get('comparison_table') or []
    )
    result.update(
        {
            'status': 'validated' if succeeded else 'review_required',
            'summary': (
                'Approved block2 MPS continuation completed successfully.'
                if succeeded
                else 'block2 MPS continuation completed, but one or more retries remain unresolved.'
            ),
            'report': merged,
            'plan': copy.deepcopy(continuation_plan),
            'decisions': copy.deepcopy(decisions),
            'approval': copy.deepcopy(merged_adaptive['mps_continuation_approval']),
        }
    )
    return result
