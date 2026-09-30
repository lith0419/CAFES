"""Build the Study report from current task results and saved scientific evidence."""

from __future__ import annotations

import copy

from .gates.presentation import build_report_workflow
from .gates.review_actions import _PLAN_KEY_BY_KIND


def merge_case_records(previous, updates):
    records = {item['case_id']: copy.deepcopy(item) for item in previous or []}
    records.update({item['case_id']: copy.deepcopy(item) for item in updates or []})
    return list(records.values())


def update_study_report(previous, current, *, review_plan=None, review_kind=None):
    """Keep analysis extensions while replacing the current task-result view."""
    previous = previous or {}
    merged = {**copy.deepcopy(previous), **copy.deepcopy(current)}
    for key in ('schema', 'adaptive', 'postprocessing'):
        if previous.get(key) is not None:
            merged[key] = copy.deepcopy(previous[key])
    merged['artifacts'] = copy.deepcopy(previous.get('artifacts') or [])
    for artifact in current.get('artifacts') or []:
        if artifact not in merged['artifacts']:
            merged['artifacts'].append(copy.deepcopy(artifact))
    case_ids = {item['case_id'] for item in merged['cases']}
    row_ids = {item['case_id'] for item in merged['comparison_table']}
    merged['status'] = 'succeeded' if case_ids and case_ids == row_ids and all(
        row.get('status') == 'succeeded' for row in merged['comparison_table']
    ) and all(
        (case.get('task_report') or {}).get('execution_status') == 'succeeded' for case in merged['cases']
    ) else 'completed_with_issues'
    if not review_kind:
        return merged

    selected = {case['case_id'] for case in review_plan['cases']}
    stage_report = copy.deepcopy(current)
    stage_report['cases'] = [copy.deepcopy(case) for case in merged['cases'] if case['case_id'] in selected]
    stage_report['comparison_table'] = [copy.deepcopy(row) for row in merged['comparison_table'] if row['case_id'] in selected]
    stage_report['status'] = 'succeeded' if all(
        row.get('status') == 'succeeded' for row in stage_report['comparison_table']
    ) else 'completed_with_issues'
    # This is scientific stage evidence in the same Study, without execution ownership.
    stage_report.pop('adaptive', None)
    stage_report.pop('retry_report', None)
    plan_key = _PLAN_KEY_BY_KIND.get(review_kind, 'refined_plan')
    report_key = plan_key.replace('_plan', '_report')
    container = merged if review_kind == 'static' else merged.setdefault('adaptive', {})
    if container is None:
        container = merged['adaptive'] = {}
    stored_plan = copy.deepcopy(container.get(plan_key) or review_plan)
    stored_plan['study_id'] = merged['study_id']
    stored_plan['cases'] = merge_case_records(stored_plan.get('cases'), review_plan['cases'])
    stored_plan.pop('_review_kind', None)
    container[plan_key], container[report_key] = stored_plan, stage_report
    succeeded = sum(row.get('status') == 'succeeded' for row in merged['comparison_table'])
    merged['summary'] = 'Study tasks updated; cases={0}; succeeded={1}; unresolved={2}.'.format(
        len(merged['cases']), succeeded, len(merged['cases']) - succeeded,
    )
    if review_kind == 'path_refinement':
        merged.pop('scan_path_diagnostics', None)
        container.pop('path_diagnostics', None)
        container['mode'] = 'adaptive_scan'
    if review_kind not in ('static', 'direct_casscf_review', 'refined'):
        refined = copy.deepcopy(container.get('refined_report') or {})
        refined.update(status=merged['status'], cases=copy.deepcopy(merged['cases']),
                       comparison_table=copy.deepcopy(merged['comparison_table']))
        container['refined_report'] = refined
    container['workflow'] = build_report_workflow(merged, analysis_requested=False)
    return merged
