"""One Study checkpoint owns its tasks and their current runs."""

from __future__ import annotations

from pyscf_agent.schema_contracts import TASK_REPORT_SCHEMA

import json
import hashlib
import threading
from pathlib import Path
from typing import Any, Dict


REPORT_LOCK = threading.RLock()


def case_index(records: Any, label: str) -> Dict[str, Dict[str, Any]]:
    if not isinstance(records, list):
        raise ValueError('{0} must be a case list'.format(label))
    result = {}
    for record in records:
        case_id = record.get('case_id') if isinstance(record, dict) else None
        if not isinstance(case_id, str) or not case_id.strip() or case_id in result:
            raise ValueError('{0} has a missing or duplicate case_id'.format(label))
        result[case_id] = record
    return result


def read_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('Cannot read Study execution record: {0}'.format(path)) from exc
    if not isinstance(payload, dict):
        raise ValueError('Invalid Study execution record: {0}'.format(path))
    return payload


def checkpoint_payload(path: Path, state: Dict[str, Any]) -> Dict[str, Any]:
    """Write each result first; the checkpoint contains only its ref and status."""
    from pyscf_agent.artifacts import default_artifact_repository
    from pyscf_agent.artifacts.task_storage import compact_task_payload

    repository = default_artifact_repository()
    def compact_record(case_id, record):
        entry = dict(record)
        report = entry.pop('task_report', None)
        if isinstance(report, dict):
            identity = '{0}/{1}/{2}/{3}'.format(case_id, report.get('run_id'), entry.get('attempt_count'), report.get('execution_status'))
            filename = hashlib.sha256(identity.encode()).hexdigest()[:24] + '.json'
            target = path.parent / 'case-reports' / filename
            report = compact_task_payload(report, target.parent)
            report.setdefault('schema', TASK_REPORT_SCHEMA)
            # The in-memory Study view must equal the persisted result; otherwise
            # a reload looks like a changed calculation and loses annotations.
            record['task_report'] = report
            entry['task_report_ref'] = repository.write_json(target, report, kind='task-report')
            entry['task_summary'] = {
                'run_id': report.get('run_id'),
                'execution_status': report.get('execution_status'),
            }
        if 'candidate_runs' in entry:
            entry['candidate_runs'] = [compact_record(case_id, candidate) for candidate in entry['candidate_runs']]
        return entry
    cases = {case_id: compact_record(case_id, record) for case_id, record in state['cases'].items()}
    return {**state, 'cases': cases}


def load_checkpoint(path: Path, study_id: str, *, include_reports: bool = True) -> Dict[str, Any]:
    """Validate and normalize persisted execution data at the loading boundary."""
    try:
        payload = read_json(path)
    except ValueError as exc:
        raise ValueError('Unreadable study checkpoint: {0}'.format(path)) from exc
    if payload.get('schema') != 'pyscf-agent.study-state.v1':
        raise ValueError('Invalid study checkpoint schema: {0}'.format(path))
    if payload.get('study_id') != study_id:
        raise ValueError('Study checkpoint belongs to a different study')
    cases = payload.get('cases')
    if not isinstance(cases, dict):
        raise ValueError('Invalid study checkpoint cases: {0}'.format(path))
    for case_id, checkpoint in cases.items():
        if not isinstance(checkpoint, dict) or not isinstance(checkpoint.get('attempt_count'), int):
            raise ValueError('Invalid case checkpoint: {0}'.format(case_id))
        if checkpoint.setdefault('case_id', case_id) != case_id:
            raise ValueError('Checkpoint case identity differs from its key')
        execution = checkpoint.get('execution')
        if execution is not None and (
            not isinstance(execution, dict) or not isinstance(execution.get('pending'), bool)
            or not isinstance(execution.get('run_id'), str) or not execution['run_id']
            or not isinstance(execution.get('attempt_count'), int) or execution['attempt_count'] < 1
        ):
            raise ValueError('Invalid execution association: {0}'.format(case_id))
        for candidate in checkpoint.get('candidate_runs') or []:
            reference = candidate.get('task_report_ref')
            if include_reports and not candidate.get('task_report') and isinstance(reference, dict):
                candidate_path = Path(reference['path'])
                candidate['task_report'] = read_json(candidate_path if candidate_path.is_absolute() else path.parent / candidate_path)
                candidate.pop('task_report_ref', None)
                candidate.pop('task_summary', None)
        report = checkpoint.get('task_report')
        reference = checkpoint.get('task_report_ref')
        if report is None and isinstance(reference, dict):
            if include_reports:
                report_path = Path(reference['path'])
                if not report_path.is_absolute():
                    report_path = path.parent / report_path
                report = read_json(report_path)
                checkpoint['task_report'] = report
                checkpoint.pop('task_report_ref', None)
                checkpoint.pop('task_summary', None)
            else:
                report = checkpoint.get('task_summary')
        if not (execution or {}).get('pending'):
            if not isinstance(report, dict):
                raise ValueError('Case checkpoint has neither a result nor a pending execution: {0}'.format(case_id))
            if execution:
                if report.get('run_id') is None:
                    report['run_id'] = execution['run_id']
                if report['run_id'] != execution['run_id']:
                    raise ValueError('TaskReport does not match the current task run: {0}'.format(case_id))
    return payload
