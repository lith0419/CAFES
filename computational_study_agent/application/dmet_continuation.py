"""Bounded branch sweeps composed from the existing saved-Study retry lifecycle."""

from __future__ import annotations

import copy
import uuid

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.serialization import json_fingerprint
from ..costing import ensure_plan_cost_estimate, require_plan_cost_approval
from ..dmet_branches import (
    analyze_branches,
    candidates,
    case_identity,
    is_dmet,
    normalize_policy,
    select_donor,
)
from ..dmet_phase_plots import write_phase_plots
from ..locking import study_lock
from ..retry import RetryAction, saved_revision
from ..schema import StudyPlan
from ..study_state import load_checkpoint, read_json
from .background import invocation_lock, launch_study
from .retries import _prepare_retry, _write as write_retry, run_retry_invocation


def _write(directory, record):
    default_artifact_repository().write_json(
        directory / 'dmet-continuations' / (record['action_id'] + '.json'),
        record,
        kind='dmet-continuation-action',
        atomic=True,
    )


def _read(directory, action_id):
    if (
        not isinstance(action_id, str)
        or len(action_id) != 32
        or any(c not in '0123456789abcdef' for c in action_id)
    ):
        raise ValueError('Invalid DMET continuation action_id')
    return read_json(directory / 'dmet-continuations' / (action_id + '.json'))


def _state(directory):
    return load_checkpoint(directory / 'study-state.json', directory.name)


def _analyze(self, directory, policy):
    analysis = analyze_branches(_state(directory)['cases'].values(), policy)
    report = self.load_report(directory.name, work_dir=str(directory.parent))
    report['dmet_phase_analysis'] = analysis
    artifacts = write_phase_plots(analysis, directory / 'dmet-phases')
    artifacts.append(
        default_artifact_repository().write_json(
            directory / 'dmet-phase-analysis.json',
            analysis,
            kind='dmet-phase-analysis',
            atomic=True,
        )
    )
    paths = {a['path'] for a in artifacts}
    report['artifacts'] = [
        a for a in report.get('artifacts', []) if a.get('path') not in paths
    ] + artifacts
    self._save_report(report, work_dir=str(directory.parent))
    return analysis


def analyze_dmet_branches(self, study_id, *, policy=None, work_dir=None):
    directory = self._study_directory(study_id, work_dir)
    with invocation_lock(directory):
        return _analyze(self, directory, normalize_policy(policy))


def _options(case, mixing, iterations):
    if not isinstance(case.request.get('solver'), dict):
        case.request['solver'] = {'name': case.request.get('solver'), 'options': {}}
    options = case.request['solver'].setdefault('options', {})
    options.update(
        correlation_potential_mixing=mixing,
        max_iterations=iterations,
        diis_enabled=False,
        bath_spin_dimension_policy='max',
        initial_correlation_potential='zero',
    )
    return options


def prepare_dmet_continuation(
    self,
    study_id,
    *,
    case_ids=None,
    branches=None,
    mode='bidirectional',
    policy=None,
    mixing=0.2,
    max_iterations=200,
    work_dir=None,
):
    """Preview at most one Run per selected physical point and requested branch.

    At execution each step selects from the then-current qualified candidates.
    No compatible donor is a recorded skip, never an analytic-seed fallback.
    """
    policy = normalize_policy(policy)
    branches = ['afm', 'cdw'] if branches is None else branches
    if (
        not isinstance(branches, list)
        or not branches
        or any(b not in ('afm', 'cdw', 'mixed', 'near_unordered') for b in branches)
        or len(set(branches)) != len(branches)
    ):
        raise ValueError('branches must be distinct classified branch names')
    if mode not in ('bidirectional', 'nearest') or (
        mode == 'bidirectional' and set(branches) - {'afm', 'cdw'}
    ):
        raise ValueError(
            'Bidirectional sweeps support AFM increasing V and CDW decreasing V; use nearest for other branches'
        )
    directory = self._study_directory(study_id, work_dir)
    with invocation_lock(directory):
        plan = StudyPlan.from_dict(read_json(directory / 'study-plan.json'))
        state = _state(directory)
        if any(
            (c.get('execution') or {}).get('pending') for c in state['cases'].values()
        ):
            raise ValueError('Collect pending Runs before preparing continuation')
        if case_ids is not None and (
            not isinstance(case_ids, list)
            or not case_ids
            or any(not isinstance(i, str) for i in case_ids)
            or len(set(case_ids)) != len(case_ids)
        ):
            raise ValueError(
                'case_ids must be a nonempty list of distinct saved case IDs'
            )
        selected = (
            set(case_ids)
            if case_ids is not None
            else {c.case_id for c in plan.cases if is_dmet(c.to_dict())}
        )
        if not selected or selected - {c.case_id for c in plan.cases}:
            raise ValueError('Select saved DMET cases for continuation')
        unique = {}
        for case in plan.cases:
            if case.case_id not in selected:
                continue
            if not (state['cases'].get(case.case_id) or {}).get('task_report'):
                raise ValueError(
                    'Continuation requires a saved result for each selected case'
                )
            case = copy.deepcopy(case)
            # Compare target and donor using the actual intended bath policy.
            _options(case, mixing, max_iterations)
            identity = case_identity(case)
            unique.setdefault(identity['point_id'], (case, identity))
        available = candidates(state['cases'].values(), policy)
        steps, budget_cases = [], []
        for branch in branches:
            direction = (
                'nearest'
                if mode == 'nearest'
                else 'increasing'
                if branch == 'afm'
                else 'decreasing'
            )
            ordered = sorted(
                unique.values(),
                key=lambda pair: (
                    pair[1]['family'],
                    pair[1]['coordinates']['U'],
                    pair[1]['coordinates']['V']
                    * (-1 if direction == 'decreasing' else 1),
                    pair[0].case_id,
                ),
            )
            for case, identity in ordered:
                donor = select_donor(
                    case, available, branch, direction=direction, policy=policy
                )
                steps.append(
                    {
                        'case_id': case.case_id,
                        'point_id': identity['point_id'],
                        'coordinates': identity['coordinates'],
                        'branch': branch,
                        'direction': direction,
                        'status': 'planned',
                        'preview_donor': donor,
                        'selection_timing': 'reselect_from_saved_results_before_each_step',
                    }
                )
                budget_case = copy.deepcopy(case)
                budget_case.case_id += '-continuation-' + branch
                # A cost preview never resolves an obsolete previous source.
                budget_case.request['solver']['options'].pop(
                    'reference_density_source', None
                )
                budget_cases.append(budget_case)
        budget = copy.deepcopy(plan)
        budget.cases, budget.grid_refinement, budget.cost_estimate = (
            budget_cases,
            {},
            {},
        )
        budget, issues = self.validate_plan(budget)
        self._raise_for_issues('dmet_continuation', issues)
        record = {
            'schema': 'pyscf-agent.dmet-continuation-action.v1',
            'action_id': uuid.uuid4().hex,
            'study_id': study_id,
            'status': 'prepared',
            'base_study_fingerprint': saved_revision(directory),
            'execution_config_fingerprint': json_fingerprint(
                self._execution_config or {}
            ),
            'steps': steps,
            'max_runs': len(steps),
            'policy': policy,
            'mixing': mixing,
            'max_iterations': max_iterations,
            'transfer': 'mean_field_density_only',
            'cost_estimate': ensure_plan_cost_estimate(budget),
            'budget_plan': budget.to_dict(),
        }
        _write(directory, record)
        return record


def start_dmet_continuation(
    self, study_id, action_id, *, work_dir=None, approve_cost=False, locale='en'
):
    directory = self._study_directory(study_id, work_dir)
    with invocation_lock(directory) as lock:
        record = _read(directory, action_id)
        if record['status'] != 'prepared':
            return {
                'study_id': study_id,
                'action_id': action_id,
                'started': False,
                'status': record['status'],
            }
        if saved_revision(directory) != record['base_study_fingerprint']:
            raise ValueError('Study changed since continuation preview; preview again')
        if (
            json_fingerprint(self._execution_config or {})
            != record['execution_config_fingerprint']
        ):
            raise ValueError('Execution target changed since continuation preview')
        budget = StudyPlan.from_dict(record['budget_plan'])
        if approve_cost is True:
            budget.resource_policy['approved'] = True
            budget.cost_estimate['approved'] = True
            budget.cost_estimate.setdefault('policy', {})['approved'] = True
        require_plan_cost_approval(budget)
        config = self._execution_config
        if config is None:
            description = self._task_executor.describe() if self._task_executor else {}
            if description.get('executor_id') not in (None, 'local', 'local-process'):
                raise ValueError(
                    'Background continuation requires configured executor connection options'
                )
            config = {
                'execution_target': 'local',
                'local_wall_time_seconds': description.get('wall_time_seconds'),
            }
        record['status'] = 'submitted'
        _write(directory, record)
        try:
            return dict(
                (self._study_launcher or launch_study)(
                    directory,
                    {
                        'mode': 'dmet_continuation',
                        'action_id': action_id,
                        'locale': locale,
                    },
                    config,
                    lock=lock,
                ),
                action_id=action_id,
            )
        except Exception as exc:
            record.update(status='launch_failed', error=str(exc))
            _write(directory, record)
            raise


def run_continuation_invocation(self, directory, request):
    with study_lock(directory / 'dmet-continuations'):
        record = _read(directory, request['action_id'])
        if record['status'] != 'submitted':
            return
        if saved_revision(directory) != record['base_study_fingerprint']:
            record.update(
                status='failed', error='Study changed before continuation execution'
            )
            _write(directory, record)
            raise ValueError(record['error'])
        record['status'] = 'running'
        _write(directory, record)
        try:
            _analyze(self, directory, record['policy'])
            for step in record['steps']:
                plan = StudyPlan.from_dict(read_json(directory / 'study-plan.json'))
                target = next(c for c in plan.cases if c.case_id == step['case_id'])
                options = _options(target, record['mixing'], record['max_iterations'])
                donor = select_donor(
                    target,
                    candidates(_state(directory)['cases'].values(), record['policy']),
                    step['branch'],
                    direction=step['direction'],
                    policy=record['policy'],
                )
                if donor is None:
                    step.update(
                        status='skipped',
                        reason='no_qualified_compatible_directional_donor',
                    )
                    _write(directory, record)
                    continue
                source = {
                    'source_case_id': donor['case_id'],
                    'source_run_id': donor['run_id'],
                    'path': donor['density_artifact']['path'],
                    'sha256': donor['density_artifact']['sha256'],
                }
                overrides = {
                    'solver.options.' + k: options[k]
                    for k in (
                        'correlation_potential_mixing',
                        'max_iterations',
                        'diis_enabled',
                        'bath_spin_dimension_policy',
                        'initial_correlation_potential',
                    )
                }
                overrides['solver.options.reference_density_source'] = source
                action = RetryAction(
                    directory.name,
                    [target.case_id],
                    {target.case_id: overrides},
                    saved_revision(directory),
                    reason=f'{step["branch"]} density continuation ({step["direction"]})',
                )
                retry = _prepare_retry(self, action, directory)
                # This step is covered by the reviewed finite maximum-run budget.
                retry['plan']['resource_policy']['approved'] = True
                retry['plan']['cost_estimate']['approved'] = True
                retry['plan']['cost_estimate'].setdefault('policy', {})['approved'] = (
                    True
                )
                retry['status'] = 'submitted'
                write_retry(directory, retry)
                step.update(
                    status='running', donor=donor, retry_action_id=retry['action_id']
                )
                _write(directory, record)
                run_retry_invocation(
                    self,
                    directory,
                    {
                        'action_id': retry['action_id'],
                        'plan': retry['plan'],
                        'retry_guard': {
                            k: copy.deepcopy(retry[k])
                            for k in (
                                'action',
                                'execution_case_ids',
                                'base_case_fields',
                            )
                        },
                        'kwargs': {
                            'rerun_case_ids': [target.case_id],
                            'locale': request.get('locale', 'en'),
                        },
                    },
                )
                current = _state(directory)['cases'][target.case_id]['task_report']
                step.update(
                    status='completed',
                    run_id=current.get('run_id'),
                    execution_status=current.get('execution_status'),
                )
                _write(directory, record)
            record['status'] = 'completed'
        except Exception as exc:
            record.update(
                status='interrupted',
                error=str(exc),
                recovery='Collect existing Runs before preparing another sweep; this action is never replayed automatically.',
            )
            raise
        finally:
            _write(directory, record)
            _analyze(self, directory, record['policy'])
