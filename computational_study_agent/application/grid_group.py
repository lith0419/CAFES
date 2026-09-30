"""Coordinate saved Studies at a shared scientific convergence barrier.

Every numerical submission, retry and receipt remains owned by the ordinary
Study service. The group only grants the next absolute sampling generation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import time

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.executors.factory import create_task_executor

from ..grid.execution import unconverged_cases
from ..locking import study_lock
from ..schema import StudyPlan, StudyReport
from ..study_state import read_json
from .background import inspect_invocation

TERMINAL = {'satisfied', 'budget_exhausted', 'max_rounds_reached', 'min_spacing_reached'}


def group_step(service, group):
    """Advance at most one shared barrier; safe to repeat after interruption."""
    root = Path(group['work_dir'])
    target = group.get('target_generation', 0)
    if not group['study_ids'] or len(set(group['study_ids'])) != len(group['study_ids']):
        raise ValueError('A group requires distinct Study IDs')
    observations, ready, terminal = {}, [], []
    for study_id in group['study_ids']:
        directory = root / study_id
        plan = StudyPlan.from_dict(service.load_plan(study_id, work_dir=str(root)))
        if not plan.grid_refinement.get('require_converged'):
            raise ValueError('Every grouped Study must require full convergence')
        invocation = inspect_invocation(directory)
        if invocation and invocation['running']:
            observations[study_id] = 'running'
            continue
        if invocation and (invocation.get('error') or invocation.get('interrupted') or invocation.get('cancelled')):
            observations[study_id] = invocation
            group.update(status='needs_attention', observations=observations)
            return group
        report_path = directory / 'study-report.json'
        if report_path.exists():
            report = StudyReport.from_dict(read_json(report_path))
            grid = report.grid_refinement or {}
            status = grid.get('status')
            if status == 'awaiting_convergence':
                observations[study_id] = {'status': status, 'case_ids': grid.get('awaiting_convergence')}
                group.update(status='awaiting_convergence', observations=observations)
                return group
            if (status in TERMINAL or
                    (status == 'awaiting_round_barrier' and grid.get('generation', 0) >= target)):
                blocked = unconverged_cases(report, plan)
                if blocked:
                    group.update(status='awaiting_convergence', observations={study_id: blocked})
                    return group
                ready.append(study_id)
                observations[study_id] = status
                if status in TERMINAL:
                    terminal.append(study_id)
                continue
        started = service.start_study(
            study_id, work_dir=str(root), resource_profile=group['resource_profile'],
            max_case_attempts=1, grid_round_target=target, locale=group.get('locale', 'en'),
        )
        observations[study_id] = started
    group['observations'] = observations
    if len(terminal) == len(group['study_ids']):
        group['status'] = 'completed'
    elif len(ready) == len(group['study_ids']):
        group.setdefault('passed_barriers', []).append({'generation': target, 'study_ids': ready})
        group.update(target_generation=target + 1, status='running')
    else:
        group['status'] = 'running'
    return group


def run_group(path):
    from .service import StudyApplicationService
    path = Path(path).resolve()
    with study_lock(path.parent, reentrant=False):
        group = read_json(path)
        config = group['execution_config']
        service = StudyApplicationService(task_executor=create_task_executor(**config),
                                          execution_config=config)
        while True:
            try:
                group = group_step(service, group)
            except Exception as exc:
                group.update(status='needs_attention', error={'code': type(exc).__name__, 'message': str(exc)})
            default_artifact_repository().write_json(path, group, kind='study-group', atomic=True)
            if group['status'] != 'running':
                return group
            time.sleep(15)


def main():
    parser = argparse.ArgumentParser(description='Coordinate saved grid Studies at shared round barriers')
    parser.add_argument('--group', required=True)
    args = parser.parse_args()
    run_group(args.group)


if __name__ == '__main__':
    main()
