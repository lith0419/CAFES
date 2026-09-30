from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional

from ..costing import ensure_plan_cost_estimate
from ..schema import StudyCase, StudyPlan


def _task_report(case_record: Dict[str, Any]) -> Dict[str, Any]:
    value = case_record.get('task_report') if isinstance(case_record, dict) else None
    return value if isinstance(value, dict) else {}


def _structured_results(case_record: Dict[str, Any]) -> Dict[str, Any]:
    value = _task_report(case_record).get('structured_results')
    return value if isinstance(value, dict) else {}


def _recommendation(case_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    structured = _structured_results(case_record)
    value = structured.get('entanglement_active_space_recommendation')
    if not isinstance(value, dict):
        cas_result = structured.get('cas_result')
        cas_result = cas_result if isinstance(cas_result, dict) else {}
        value = cas_result.get('entanglement_active_space_recommendation')
    if not isinstance(value, dict):
        return None
    return copy.deepcopy(value) if value.get('status') == 'approval_recommended' else None


def _checkpoint_manifest(case_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    structured = _structured_results(case_record)
    cas_result = structured.get('cas_result')
    cas_result = cas_result if isinstance(cas_result, dict) else {}
    dmrg_result = cas_result.get('dmrg_result')
    dmrg_result = dmrg_result if isinstance(dmrg_result, dict) else {}
    manifest = dmrg_result.get('checkpoint_manifest')
    context = manifest.get('orbital_context') if isinstance(manifest, dict) else None
    if (
        isinstance(manifest, dict)
        and isinstance(context, dict)
        and context.get('external_restart_supported') is True
    ):
        return copy.deepcopy(manifest)
    return None


def build_entanglement_active_space_review_plan(
    plan: Any,
    report: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build review-only expanded-CAS tasks from block2 entanglement evidence."""

    try:
        study_plan = plan if isinstance(plan, StudyPlan) else StudyPlan.from_dict(plan)
    except (TypeError, ValueError):
        return None
    if str(study_plan.system_type or '').strip().lower() != 'molecular':
        return None
    plan_cases = {case.case_id: case for case in study_plan.cases}
    review_cases: List[StudyCase] = []
    decisions: List[Dict[str, Any]] = []
    for record in report.get('cases') or ():
        if not isinstance(record, dict):
            continue
        case_id = str(record.get('case_id') or '')
        source_case = plan_cases.get(case_id)
        recommendation = _recommendation(record)
        manifest = _checkpoint_manifest(record)
        if not source_case or not recommendation or not manifest:
            continue
        candidate = recommendation.get('candidate_active_space')
        if not isinstance(candidate, dict):
            continue
        request = copy.deepcopy(source_case.request)
        raw_method = request.get('method')
        method = str(
            raw_method.get('name')
            if isinstance(raw_method, dict)
            else raw_method or ''
        ).strip().lower()
        if method not in ('casci', 'casscf'):
            continue
        request['active_space'] = copy.deepcopy(candidate)
        request['active_space']['approved'] = False
        solver = request.get('solver')
        if not isinstance(solver, dict):
            solver = {'name': 'block2_dmrg', 'options': {}}
        solver['name'] = 'block2_dmrg'
        options = copy.deepcopy(solver.get('options') or {})
        options.pop('restart_manifest', None)
        options.pop('restart_required', None)
        options['orbital_restart_manifest'] = copy.deepcopy(manifest)
        options['orbital_restart_provenance'] = {
            'source_case_id': case_id,
            'selection': 'entanglement_active_space_expansion',
            'source_ncas': recommendation.get('current_active_space', {}).get('ncas'),
            'target_ncas': candidate.get('ncas'),
        }
        solver['options'] = options
        request['solver'] = solver
        review_cases.append(StudyCase(
            case_id=source_case.case_id,
            label=source_case.label,
            request=request,
            variables=copy.deepcopy(source_case.variables),
            operations=copy.deepcopy(source_case.operations),
            model_spec=copy.deepcopy(source_case.model_spec),
        ))
        decisions.append({
            'case_id': source_case.case_id,
            'label': source_case.label,
            'variables': copy.deepcopy(source_case.variables),
            'recommended_method': method,
            'recommended_solver': 'block2_dmrg',
            'active_space_contract': copy.deepcopy(candidate),
            'entanglement_active_space_recommendation': copy.deepcopy(recommendation),
            'source_checkpoint_manifest': copy.deepcopy(manifest),
            'source_active_space': copy.deepcopy(recommendation.get('current_active_space') or {}),
            'tags': [
                'entanglement_active_space_expansion',
                'review_active_space',
                'requires_active_space_approval',
            ],
            'reason': recommendation.get('reason'),
            'next_step': (
                'Approve or edit the entanglement-driven ActiveSpaceAudit. The expanded '
                'calculation will reuse optimized orbitals but initialize a fresh MPS.'
            ),
        })
    if not review_cases:
        return None
    review_plan = StudyPlan(
        study_id='entanglement-active-space-review',
        name='{0}-entanglement-active-space-review'.format(study_plan.name),
        objective='Entanglement-driven active-space review for {0}'.format(study_plan.objective),
        system_type='molecular',
        cases=review_cases,
        observables=copy.deepcopy(study_plan.observables),
        comparison=copy.deepcopy(study_plan.comparison),
        capability_snapshot=copy.deepcopy(study_plan.capability_snapshot),
        resource_policy=copy.deepcopy(study_plan.resource_policy),
    )
    ensure_plan_cost_estimate(review_plan)
    return {
        'entanglement_active_space_plan': review_plan.to_dict(),
        'entanglement_active_space_decisions': decisions,
    }


__all__ = ['build_entanglement_active_space_review_plan']
