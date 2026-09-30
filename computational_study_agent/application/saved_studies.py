"""Saved studies use cases, composed by StudyApplicationService."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence
from pyscf_agent.paths import study_search_roots
from ..gates.review_actions import apply_study_review_action
from ..study_state import read_json
from ..schema import StudyPlan
from pyscf_agent.artifacts import default_artifact_repository



def _study_directory(self, study_id: str, work_dir=None) -> Path:
    if not study_id or Path(study_id).name != study_id or study_id in ('.', '..'):
        raise ValueError('study_id must be a single directory name')
    roots = study_search_roots(work_dir)
    root = next(
        (path for path in roots if (path / study_id).is_dir()), roots[0]
    ).resolve()
    directory = (root / study_id).resolve()
    directory.relative_to(root)
    if directory == root:
        raise ValueError('study_id must identify a Study below the work directory')
    return directory


def load_plan(self, study_id: str, *, work_dir=None) -> Dict[str, Any]:
    plan = read_json(self._study_directory(study_id, work_dir) / 'study-plan.json')
    if plan.get('study_id') != study_id:
        raise ValueError('Saved plan belongs to a different Study')
    return plan


def load_report(self, study_id: str, *, work_dir=None) -> Dict[str, Any]:
    """Read a saved report projection without fetching or submitting jobs."""
    directory = self._study_directory(study_id, work_dir)
    adaptive = directory / 'adaptive-study-report.json'
    report = read_json(
        adaptive if adaptive.exists() else directory / 'study-report.json'
    )
    if report.get('study_id') != study_id:
        raise ValueError('Saved report belongs to a different Study')
    return report


def _save_prepared_plan(self, plan, *, work_dir=None):
    directory = self._study_directory(plan.study_id, work_dir)
    directory.mkdir(parents=True, exist_ok=False)
    default_artifact_repository().write_json(
        directory / 'study-plan.json', plan.to_dict(), kind='study-plan', atomic=True
    )
    default_artifact_repository().write_json(
        directory / 'cost-estimate.json',
        plan.cost_estimate,
        kind='cost-estimate',
        atomic=True,
    )
    return directory


def _saved_report_path(self, study_id, work_dir=None):
    directory = self._study_directory(study_id, work_dir)
    return directory / (
        'adaptive-study-report.json'
        if (directory / 'adaptive-study-request.json').exists()
        or (directory / 'adaptive-study-report.json').exists()
        else 'study-report.json'
    )


def _save_report(self, report, *, work_dir=None):
    path = self._saved_report_path(report['study_id'], work_dir)
    default_artifact_repository().write_json(path, report, kind=path.stem, atomic=True)


def complete_saved_review(self, study_id, *, work_dir=None):
    """The runner has accepted this review; subsequent starts use its saved results."""
    report = self.load_report(study_id, work_dir=work_dir)
    if report.pop('pending_review', None) is not None:
        self._save_report(report, work_dir=work_dir)


def load_study_preparation(self, study_id: str, *, work_dir=None) -> Dict[str, Any]:
    directory = self._study_directory(study_id, work_dir)
    if not directory.is_dir():
        raise FileNotFoundError('Study not found: {0}'.format(study_id))
    if (directory / 'adaptive-study-request.json').is_file():
        request = read_json(directory / 'adaptive-study-request.json')
        result = {
            'study_id': study_id,
            'mode': 'adaptive',
            'study_spec': request.get('study_spec'),
            'options': request.get('adaptive_options'),
            'initial_scan_plan': read_json(
                directory / 'adaptive-initial-scan-plan.json'
            ),
        }
        if (directory / 'study-plan.json').is_file():
            result['plan'] = self.load_plan(study_id, work_dir=work_dir)
        return result
    return {
        'study_id': study_id,
        'mode': 'static',
        'plan': self.load_plan(study_id, work_dir=work_dir),
    }


def start_study(
    self,
    study_id: str,
    *,
    work_dir=None,
    prepared_plan=None,
    locale='en',
    rerun_case_ids=None,
    rerun_statuses=None,
    max_case_attempts=2,
    resource_profile=None,
    grid_round_target=None,
):
    """Start one complete agent invocation; the ordinary runners own every task and stage."""
    from .background import invocation_lock
    from ..execution_receipts import StudyExecutionBusy

    directory = self._study_directory(study_id, work_dir)
    try:
        with invocation_lock(directory) as lock:
            return self._start_saved_study(
                study_id,
                work_dir=work_dir,
                prepared_plan=prepared_plan,
                locale=locale,
                rerun_case_ids=rerun_case_ids,
                rerun_statuses=rerun_statuses,
                max_case_attempts=max_case_attempts,
                resource_profile=resource_profile,
                grid_round_target=grid_round_target,
                lock=lock,
            )
    except StudyExecutionBusy:
        return {'study_id': study_id, 'started': False, 'running': True}


def open_study(self, study_id: str, *, work_dir=None) -> Dict[str, Any]:
    """Read the saved preparation and report without executing or collecting tasks."""
    prepared = self.load_study_preparation(study_id, work_dir=work_dir)
    directory = self._study_directory(study_id, work_dir)
    report = (
        self.load_report(study_id, work_dir=work_dir)
        if self._saved_report_path(study_id, work_dir).is_file()
        else None
    )
    pending = (report or {}).get('pending_review') or {}
    if pending.get('kind') == 'retry_cases':
        from .retries import _read
        record = _read(directory, pending['action_id'])
        pending.update(retry_action=record, status=record['status'], can_run=record['status'] == 'prepared')
    return {**prepared, 'report': report, 'work_dir': str(directory.parent)}


def list_studies(self, *, work_dir=None) -> List[Dict[str, Any]]:
    """Discover saved Studies on disk; do not poll schedulers or create an index."""
    roots = study_search_roots(work_dir)
    if len(roots) > 1:
        # Search each root explicitly so duplicate IDs retain their location.
        entries = []
        for root in roots:
            entries.extend(self._list_studies_in_root(root))
        return sorted(entries, key=lambda entry: entry['study_id'], reverse=True)
    return self._list_studies_in_root(roots[0])


def _list_studies_in_root(self, root: Path) -> List[Dict[str, Any]]:
    root = root.resolve()
    if not root.is_dir():
        return []
    entries = []
    for directory in sorted(root.iterdir(), key=lambda path: path.name, reverse=True):
        if not any(
            (directory / filename).is_file()
            for filename in ('study-plan.json', 'adaptive-study-request.json')
        ):
            continue
        try:
            saved = self.open_study(directory.name, work_dir=str(root))
            plan = saved.get('plan') or saved.get('initial_scan_plan') or {}
            report = saved['report'] or {}
            entries.append(
                {
                    'study_id': directory.name,
                    'name': plan.get('name', directory.name),
                    'mode': saved['mode'],
                    'status': report.get('status', 'prepared'),
                    'work_dir': str(root),
                }
            )
        except (OSError, ValueError, KeyError) as exc:
            entries.append(
                {
                    'study_id': directory.name,
                    'status': 'unreadable',
                    'work_dir': str(root),
                    'error': str(exc),
                }
            )
    return entries


def _start_saved_study(
    self,
    study_id,
    *,
    work_dir,
    prepared_plan,
    locale,
    rerun_case_ids,
    rerun_statuses,
    max_case_attempts,
    resource_profile,
    lock,
    grid_round_target=None,
):
    from .background import launch_study
    from ..costing import require_plan_cost_approval

    directory = self._study_directory(study_id, work_dir)
    work_dir = str(directory.parent)
    prepared = self.load_study_preparation(study_id, work_dir=work_dir)
    if max_case_attempts < 1:
        raise ValueError('max_case_attempts must be at least 1')
    config = self._execution_config
    if config is None:
        description = (
            self._task_executor.describe() if self._task_executor is not None else {}
        )
        if description.get('executor_id') not in (None, 'local', 'local-process'):
            raise ValueError(
                'Background Study execution requires the configured executor connection options'
            )
        config = {
            'execution_target': 'local',
            'local_wall_time_seconds': description.get('wall_time_seconds'),
        }
    report_path = self._saved_report_path(study_id, work_dir)
    saved_review = (
        (read_json(report_path).get('pending_review') or {})
        if report_path.exists()
        else {}
    )
    use_saved_review = prepared_plan is None and bool(saved_review)
    if use_saved_review and saved_review.get('kind') == 'retry_cases':
        from .retries import _start_retry
        return _start_retry(self, directory, saved_review['action_id'], lock, locale=locale)
    if use_saved_review:
        if not saved_review['can_run']:
            raise ValueError('The saved review requires approval before execution')
        prepared_plan = saved_review.get('plan')
    adaptive_approval = (
        isinstance(prepared_plan, dict) and prepared_plan.get('mode') == 'adaptive'
    )
    if adaptive_approval:
        if prepared_plan.get('study_id') != study_id:
            raise ValueError('Prepared plan belongs to a different Study')
        prepared = prepared_plan
    policy = (prepared.get('study_spec') or prepared.get('plan') or {}).get(
        'resource_policy'
    ) or {}
    resource_profile = resource_profile or policy.get('resource_profile')
    kwargs = {'locale': locale, 'resource_profile': resource_profile}
    if grid_round_target is not None:
        kwargs['grid_round_target'] = grid_round_target
    if use_saved_review and saved_review.get('analysis_approval'):
        request = {'mode': 'analysis', 'kwargs': kwargs, 'review': saved_review}
    elif prepared['mode'] == 'adaptive' and (
        adaptive_approval or (prepared_plan is None and 'plan' not in prepared)
    ):
        if rerun_case_ids or rerun_statuses:
            raise ValueError(
                'Use a reviewed task plan to retry selected adaptive cases'
            )
        if not prepared.get('study_spec'):
            raise ValueError(
                'Historical adaptive input is unavailable; resume with the original StudySpec through the agent API'
            )
        bound = self._bind_spec_resources(
            self._coerce_spec(prepared['study_spec']), resource_profile
        )
        if bound.to_dict() != prepared['study_spec']:
            raise ValueError(
                'Prepare a new adaptive Study to change its saved resource profile'
            )
        request = {
            'mode': 'adaptive',
            'spec': prepared['study_spec'],
            'options': prepared['options'],
            'kwargs': kwargs,
        }
        first_payload = self.build_adaptive_plan(
            prepared['study_spec'],
            options=prepared['options'],
            resource_profile=resource_profile,
        )
        first = StudyPlan.from_dict(first_payload['initial_scan_plan'])
        require_plan_cost_approval(self._bind_plan_resources(first, resource_profile))
        if adaptive_approval:
            request['initial_cost_approved'] = True
    else:
        payload = prepared_plan if prepared_plan is not None else prepared['plan']
        plan, issues = self.validate_plan(payload)
        self._raise_for_issues('study_plan', issues)
        if plan.study_id != study_id:
            raise ValueError('Prepared plan belongs to a different Study')
        require_plan_cost_approval(self._bind_plan_resources(plan, resource_profile))
        normalized = plan.to_dict()
        if isinstance(payload, dict) and payload.get('_review_kind'):
            normalized['_review_kind'] = payload['_review_kind']
        kwargs.update(
            rerun_case_ids=rerun_case_ids,
            rerun_statuses=rerun_statuses,
            max_case_attempts=max_case_attempts,
        )
        request = {'mode': 'static', 'plan': normalized, 'kwargs': kwargs}
    request['consume_review'] = use_saved_review
    launcher = self._study_launcher or launch_study
    return launcher(directory, request, config, lock=lock)


def collect_saved_study(
    self, study_id: str, *, work_dir=None, locale='en'
) -> Dict[str, Any]:
    """Read a completed agent invocation, or recover existing handles after interruption."""
    from .background import invocation_lock

    directory = self._study_directory(study_id, work_dir)
    work_dir = str(directory.parent)
    with invocation_lock(directory):
        prepared = self.load_study_preparation(study_id, work_dir=work_dir)
        # Completed adaptive reports also contain reviewed subset updates.
        # Refreshing them through an older stage plan could replace those results.
        if prepared['mode'] == 'adaptive':
            if (directory / 'study-state.json').exists():
                return self.collect_study(
                    study_id, work_dir=work_dir, locale=locale
                ).to_dict()
            invocation = (
                read_json(directory / 'study-invocation.json')
                if (directory / 'study-invocation.json').exists()
                else {}
            )
            if invocation.get('finished_at') and not invocation.get('error'):
                return self.load_report(study_id, work_dir=work_dir)
            return self.collect_adaptive_study(
                prepared['study_spec'],
                study_id=study_id,
                options=prepared['options'],
                work_dir=work_dir,
                locale=locale,
            )
        return self.collect_study(study_id, work_dir=work_dir, locale=locale).to_dict()


def review_study(
    self,
    study_id: str,
    action_id: str,
    *,
    work_dir=None,
    case_ids=None,
    approval_token=None,
    plan_kind=None,
) -> Dict[str, Any]:
    """Save a review decision without executing it; start later using the Study ID."""
    from .background import invocation_lock

    directory = self._study_directory(study_id, work_dir)
    work_dir = str(directory.parent)
    with invocation_lock(directory):
        return self._review_saved_study(
            study_id,
            action_id,
            work_dir=work_dir,
            case_ids=case_ids,
            approval_token=approval_token,
            plan_kind=plan_kind,
        )


def _review_saved_study(
    self, study_id, action_id, *, work_dir, case_ids, approval_token, plan_kind
):
    prepared = self.load_study_preparation(study_id, work_dir=work_dir)
    directory = self._study_directory(study_id, work_dir)
    plan = prepared.get('plan')
    checkpoint = (
        read_json(directory / 'study-state.json')
        if (directory / 'study-state.json').exists()
        else {}
    )
    if any(
        (case.get('execution') or {}).get('pending')
        for case in (checkpoint.get('cases') or {}).values()
    ):
        raise ValueError('Collect pending runs before preparing a review')
    if (directory / 'study-report.json').exists() or (
        directory / 'adaptive-study-report.json'
    ).exists():
        state = self.load_report(study_id, work_dir=work_dir)
    else:
        state = {
            'study_id': study_id,
            'lifecycle': copy.deepcopy((plan or {}).get('lifecycle') or {}),
        }
    previous_review = state.pop('pending_review', None) or {}
    if (
        previous_review.get('plan')
        and previous_review['plan'].get('mode') != 'adaptive'
    ):
        plan = previous_review['plan']
    if prepared['mode'] == 'adaptive':
        adaptive = state.get('adaptive') or {}
        plan_kind = plan_kind or adaptive.get('cost_review_plan_kind')
        if not adaptive or plan_kind == 'initial':
            plan = prepared['initial_scan_plan']
            plan_kind = 'initial'
        elif plan_kind:
            from ..gates.review_actions import _PLAN_KEY_BY_KIND

            plan = adaptive.get(_PLAN_KEY_BY_KIND.get(plan_kind, 'refined_plan'))
    result = self.apply_review_action(
        action_id,
        study_state=state,
        plan=plan,
        case_ids=case_ids,
        approval_token=approval_token,
        plan_kind=plan_kind,
    )
    if action_id in ('increase_recovery_max_cycle', 'increase_dmet_iterations',
                     'try_fci_recovery', 'retry_dmet_without_impurity_diis'):
        from .retries import _prepare_retry
        from ..retry import RetryAction, saved_revision
        # Existing structured recovery buttons use the same typed retry boundary.
        source = {c['case_id']: c for c in self.load_plan(study_id, work_dir=work_dir)['cases']}
        overrides = {}
        for case in result['plan']['cases']:
            old = source[case['case_id']]['request']
            overrides[case['case_id']] = {key: value for key, value in case['request'].items()
                                         if old.get(key) != value}
        action = RetryAction.from_dict({'study_id': study_id, 'case_ids': list(overrides),
            'case_overrides': overrides, 'reason': result['message'],
            'base_study_fingerprint': saved_revision(directory)})
        retry = _prepare_retry(self, action, directory)
        return dict(result, retry_action=retry, kind='retry_cases')
    if prepared['mode'] == 'adaptive' and result.get('study_spec_patch'):
        # Initial cost approval changes the saved adaptive request, not a
        # synthetic static child Study. It is still a draft until started.
        result['plan'] = {
            **prepared,
            'study_spec': copy.deepcopy(prepared['study_spec']),
        }
        for key, value in result['study_spec_patch'].items():
            result['plan']['study_spec'][key] = {
                **result['plan']['study_spec'].get(key, {}),
                **value,
            }
    saved = {
        **self._review_report_scaffold(plan or prepared.get('initial_scan_plan') or {}),
        **result['study_state'],
        'study_id': study_id,
        'work_dir': str(directory),
    }
    saved['pending_review'] = {
        key: copy.deepcopy(result[key])
        for key in (
            'action_id',
            'status',
            'message',
            'plan',
            'can_run',
            'case_ids',
            'workflow',
            'lifecycle',
            'analysis_approval',
        )
        if key in result
    }
    saved['workflow'] = copy.deepcopy(result['workflow'])
    self._save_report(saved, work_dir=work_dir)
    return result


def apply_review_action(
    self,
    action_id: str,
    *,
    study_state: Optional[Dict[str, Any]] = None,
    plan: Optional[Dict[str, Any]] = None,
    case_ids: Optional[Sequence[str]] = None,
    approval_token: Optional[str] = None,
    plan_kind: Optional[str] = None,
) -> Dict[str, Any]:
    """Apply a displayed Planner review action without UI-owned state changes."""

    return apply_study_review_action(
        action_id,
        study_state=study_state,
        plan=plan,
        case_ids=case_ids,
        approval_token=approval_token,
        plan_kind=plan_kind,
    )


def _saved_case_checkpoints(self, study_id, work_dir=None):
    path = self._study_directory(study_id, work_dir) / 'study-state.json'
    return (read_json(path).get('cases') or {}) if path.exists() else {}
