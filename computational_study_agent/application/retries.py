"""Preview and launch explicit retry actions against one saved Study revision."""

from __future__ import annotations

import copy
import uuid

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.serialization import json_fingerprint
from ..costing import ensure_plan_cost_estimate, require_plan_cost_approval
from ..retry import (
    RetryAction,
    compile_retry,
    preview_contexts,
    saved_revision,
    validate_execution_guard,
    case_field_fingerprints,
)
from ..schema import StudyPlan
from ..study_state import read_json, load_checkpoint
from .background import invocation_lock, launch_study


def _write(directory, record):
    default_artifact_repository().write_json(
        directory / 'retry-actions' / (record['action_id'] + '.json'),
        record,
        kind='retry-action',
        atomic=True,
    )


def _read(directory, action_id):
    if (
        not isinstance(action_id, str)
        or len(action_id) != 32
        or any(c not in '0123456789abcdef' for c in action_id)
    ):
        raise ValueError('Invalid retry action_id')
    return read_json(directory / 'retry-actions' / (action_id + '.json'))


def retry_context(self, study_id, *, work_dir=None):
    directory = self._study_directory(study_id, work_dir)
    with invocation_lock(directory):
        return {
            'study_id': study_id,
            'base_study_fingerprint': saved_revision(directory),
        }


def prepare_retry(self, payload, *, work_dir=None):
    action = RetryAction.from_dict(payload)
    directory = self._study_directory(action.study_id, work_dir)
    with invocation_lock(directory):
        return _prepare_retry(self, action, directory)


def _prepare_retry(self, action, directory):
    if saved_revision(directory) != action.base_study_fingerprint:
        raise ValueError(
            'Study changed since retry selection; refresh and preview again'
        )
    plan = StudyPlan.from_dict(read_json(directory / 'study-plan.json'))
    if plan.study_id != action.study_id:
        raise ValueError('Retry belongs to a different Study')
    state = load_checkpoint(directory / 'study-state.json', action.study_id)
    if any((c.get('execution') or {}).get('pending') for c in state['cases'].values()):
        raise ValueError('Collect pending runs before preparing a retry')
    if any(
        not isinstance(state['cases'].get(i, {}).get('task_report'), dict)
        for i in action.case_ids
    ):
        raise ValueError('Retry requires saved results for each selected case')
    subset, changes, dependents = compile_retry(plan, action)
    subset = self._bind_plan_resources(
        subset, subset.resource_policy.get('resource_profile')
    )
    subset, issues = self.validate_plan(subset)
    self._raise_for_issues('retry_action', issues)
    cost = ensure_plan_cost_estimate(subset)
    contexts = preview_contexts(subset, state)
    # A deferred child can become runnable only after its selected parent succeeds.
    from ..retry import live_dependencies

    scope = {c.case_id for c in subset.cases}
    for case, context in zip(subset.cases, contexts):
        context['awaits_selected_parent'] = bool(live_dependencies(case) & scope)
    record = {
        'action_id': uuid.uuid4().hex,
        'kind': 'retry_cases',
        'status': 'prepared',
        'action': action.to_dict(),
        'execution_case_ids': [c.case_id for c in subset.cases],
        'total_case_count': len(plan.cases),
        'dependent_case_ids': dependents,
        'changes': changes,
        'contexts': contexts,
        'cost_estimate': cost,
        'base_case_fields': case_field_fingerprints(plan),
        'plan': subset.to_dict(),
        'execution_config_fingerprint': json_fingerprint(self._execution_config or {}),
    }
    _write(directory, record)
    report = self.load_report(action.study_id, work_dir=str(directory.parent))
    report['pending_review'] = {
        'kind': 'retry_cases',
        'action_id': record['action_id'],
        'case_ids': record['execution_case_ids'],
        'can_run': True,
        'retry_action': record,
    }
    self._save_report(report, work_dir=str(directory.parent))
    return record


def start_retry(
    self, study_id, action_id, *, work_dir=None, approve_cost=False, locale='en'
):
    directory = self._study_directory(study_id, work_dir)
    with invocation_lock(directory) as lock:
        return _start_retry(
            self, directory, action_id, lock, approve_cost=approve_cost, locale=locale
        )


def _start_retry(self, directory, action_id, lock, *, approve_cost=False, locale='en'):
    record = _read(directory, action_id)
    if record['status'] != 'prepared':
        return {
            'study_id': directory.name,
            'action_id': action_id,
            'started': False,
            'status': record['status'],
            'message': 'This retry action has already been submitted.',
        }
    if (
        json_fingerprint(self._execution_config or {})
        != record['execution_config_fingerprint']
    ):
        raise ValueError('Execution target changed since retry preview; preview again')
    plan = StudyPlan.from_dict(record['plan'])
    validate_execution_guard(directory, plan, record, record['execution_case_ids'])
    if approve_cost is True:
        plan.resource_policy['approved'] = True
        plan.cost_estimate['approved'] = True
        plan.cost_estimate.setdefault('policy', {})['approved'] = True
    require_plan_cost_approval(plan)
    request = {
        'mode': 'retry',
        'action_id': action_id,
        'plan': plan.to_dict(),
        'retry_guard': {
            k: copy.deepcopy(record[k])
            for k in ('action', 'execution_case_ids', 'base_case_fields')
        },
        'kwargs': {
            'locale': locale,
            'rerun_case_ids': record['execution_case_ids'],
            'resource_profile': plan.resource_policy.get('resource_profile'),
        },
    }
    record['status'] = 'submitted'
    _write(directory, record)
    launcher = self._study_launcher or launch_study
    config = self._execution_config
    if config is None:
        description = self._task_executor.describe() if self._task_executor else {}
        if description.get('executor_id') not in (None, 'local', 'local-process'):
            record['status'] = 'prepared'
            _write(directory, record)
            raise ValueError(
                'Background retry requires configured executor connection options'
            )
        config = {
            'execution_target': 'local',
            'local_wall_time_seconds': description.get('wall_time_seconds'),
        }
    try:
        return dict(
            launcher(directory, request, config, lock=lock), action_id=action_id
        )
    except Exception as exc:
        # An uncertain launch must not silently become a second submission.
        record.update(status='launch_failed', error=str(exc))
        _write(directory, record)
        raise


def finish_retry(directory, action_id, error=None):
    record = _read(directory, action_id)
    record.update(status='failed' if error else 'completed', error=error)
    _write(directory, record)
    report_path = directory / (
        'adaptive-study-report.json'
        if (directory / 'adaptive-study-report.json').exists()
        else 'study-report.json'
    )
    report = read_json(report_path)
    if (report.get('pending_review') or {}).get('action_id') == action_id:
        report.pop('pending_review')
        default_artifact_repository().write_json(
            report_path, report, kind=report_path.stem, atomic=True
        )


def run_retry_invocation(service, directory, request):
    # The ledger claim survives coordinator crashes. Even a duplicate worker
    # cannot replay an action that already entered execution.
    with invocation_lock(directory / 'retry-actions'):
        record = _read(directory, request['action_id'])
        if record['status'] != 'submitted':
            return
        if record['action'] != request['retry_guard']['action']:
            raise ValueError('Retry invocation differs from its saved action')
        record['status'] = 'running'
        _write(directory, record)
        error = None
        try:
            service.run_study(
                request['plan'],
                work_dir=str(directory.parent),
                retry_guard=request['retry_guard'],
                **request['kwargs'],
            )
        except Exception as exc:
            error = {'code': type(exc).__name__, 'message': str(exc)}
            raise
        finally:
            finish_retry(directory, request['action_id'], error)
