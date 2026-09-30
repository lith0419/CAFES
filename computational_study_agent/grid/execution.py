"""Persist bounded one-dimensional or local two-dimensional grid refinement."""
from __future__ import annotations

import copy
from pathlib import Path

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.paths import resolve_work_dir

from computational_study_agent.costing import CostApprovalRequired, ensure_plan_cost_estimate
from computational_study_agent.grid.refinement import (assess_grid, build_point, fingerprint, groups, key,
                              normalize_policy, validate_plan_contract, metric_sample)
from computational_study_agent.schema import StudyCase, StudyPlan, StudySpec, StudyReport
from computational_study_agent.study_state import read_json

STATE_FILE = 'grid-refinement-state.json'
STATE_SCHEMA = 'pyscf-agent.grid-refinement-state.v1'


def unconverged_cases(report, plan):
    """Require affirmative convergence and passing required provider checks."""
    from pyscf_agent.result_quality import evaluate_quality_checks
    records = {c['case_id']: c.get('task_report') or {} for c in report.cases}
    blocked = []
    for case in plan.cases:
        task = records.get(case.case_id, {})
        results = task.get('structured_results') or {}
        quality = evaluate_quality_checks(results.get('quality_checks'))
        if (task.get('execution_status') != 'succeeded' or results.get('converged') is not True
                or quality['publication_eligible'] is not True):
            blocked.append(case.case_id)
    return blocked


def execution_case(case, policy, overrides):
    """Apply a persisted numerical fallback without changing the sampling contract."""
    if case.case_id not in overrides:
        return case
    retry = overrides[case.case_id]
    if not policy.get('require_converged') or retry != policy.get('convergence_retry'):
        raise ValueError('Saved convergence retry differs from the grid policy')
    resolved = copy.deepcopy(case)
    solver = resolved.request.get('solver')
    if not isinstance(solver, dict) or solver.get('name') != 'dmet':
        raise ValueError('Grid convergence retries require a DMET solver')
    solver.setdefault('options', {}).update(retry)
    return resolved


def _write(path, payload, kind):
    return default_artifact_repository().write_json(path, payload, kind=kind, atomic=True)


def _result_identity(report):
    return fingerprint([{
        'case_id': c['case_id'], 'task_report': {
            k: (c.get('task_report') or {}).get(k)
            for k in ('run_id', 'execution_status', 'structured_results', 'gate_decisions')},
    } for c in report.cases])


def attach_grid_report(report, plan, state=None):
    """Report decoration is read-only with respect to calculations and decisions."""
    path = Path(report.work_dir) / STATE_FILE
    if not plan.grid_refinement or (state is None and not path.exists()):
        return report
    state = state if state is not None else read_json(path)
    stale = bool(state.get('result_identity') and state['result_identity'] != _result_identity(report))
    status = 'needs_resume' if stale else state['status']
    additions = {item['case']['case_id']: item for item in state['additions']}
    report.grid_refinement = {
        'policy': copy.deepcopy(plan.grid_refinement), 'status': status,
        'strategy': state.get('strategy', 'axis_alternating'),
        'generation': state.get('generation', 0),
        'sampling_satisfied': status == 'satisfied',
        'seed_point_count': len(state['seed_cases']), 'new_point_count': len(additions),
        'axis_rounds': copy.deepcopy(state['axis_rounds']),
        'decisions': copy.deepcopy(state['decisions']),
        'intervals': copy.deepcopy(state.get('intervals', [])),
        'remaining_candidates': copy.deepcopy(state.get('remaining_candidates', [])),
        'awaiting_convergence': copy.deepcopy(state.get('awaiting_convergence', [])),
        'convergence_retry_overrides': copy.deepcopy(state.get('convergence_retry_overrides', {})),
        'state_path': str(path),
        'interpretation': 'Local midpoint sampling criteria; not a bound on solver error or proof of a phase boundary.',
    }
    if state.get('strategy') == 'local_cells':
        from computational_study_agent.grid.local import plot_cells
        report.grid_refinement['cells'] = plot_cells(plan, state['cells'])
    from computational_study_agent.executor import _with_energy_unit
    cases = {c.case_id: c for c in plan.cases}
    records = {c['case_id']: c for c in report.cases}
    for row in report.comparison_table:
        case = cases.get(row['case_id'])
        if case:
            sample = metric_sample(records.get(case.case_id), {'name': 'energy_per_site'}, case, row)
            if sample is not None:
                row['energy_per_site'] = _with_energy_unit(sample[0], case.model_spec.get('energy_unit', 'a.u.'))
        addition = additions.get(row['case_id'])
        row['grid_origin'] = 'refined' if addition else 'seed'
        row['grid_round'] = addition['round'] if addition else 0
        row['grid_axis'] = addition['axis'] if addition else ''
    # Existing summaries can already contain a previous sampling status on Collect.
    report.summary = report.summary.split('; grid_refinement=')[0] + '; grid_refinement={0}; added_points={1}'.format(status, len(additions))
    return report


def live_grid_progress(plan, directory):
    """Read current case checkpoints without collecting, submitting or saving."""
    from computational_study_agent.executor import _complete_task_view
    from computational_study_agent.study_state import load_checkpoint

    directory = Path(directory)
    plan = StudyPlan.from_dict(plan) if isinstance(plan, dict) else plan
    checkpoint_path = directory / 'study-state.json'
    checkpoint = (load_checkpoint(checkpoint_path, plan.study_id)
                  if checkpoint_path.exists() else {'cases': {}})
    report_path = directory / 'study-report.json'
    previous = read_json(report_path) if report_path.exists() else {}
    cases, rows = _complete_task_view(plan, checkpoint, [], previous, directory)
    report = StudyReport(
        study_id=plan.study_id, name=plan.name, objective=plan.objective,
        system_type=plan.system_type, status='running', work_dir=str(directory),
        cases=cases, comparison_table=rows,
    )
    attach_grid_report(report, plan)
    counts = {}
    for row in rows:
        counts[row['status']] = counts.get(row['status'], 0) + 1
    # Keep the polling payload tabular; full diagnostic objects remain in reports.
    rows = [{key: value for key, value in row.items()
             if not isinstance(value, (dict, list)) or
             (isinstance(value, dict) and set(value) <= {'value', 'unit'})}
            for row in rows]
    return {'comparison_table': rows, 'grid_refinement': report.grid_refinement,
            'task_count': len(rows), 'task_status_counts': counts}


def run_grid_study(plan, *, runner, grid_round_target=None, **kwargs):
    from computational_study_agent.planner import _model_cases
    from computational_study_agent.capabilities import default_study_capabilities
    from computational_study_agent.validation import validate_study_plan, has_errors

    if grid_round_target is not None and (isinstance(grid_round_target, bool)
            or not isinstance(grid_round_target, int) or grid_round_target < 0):
        raise ValueError('grid_round_target must be a nonnegative integer')
    plan = copy.deepcopy(plan)
    plan.grid_refinement = normalize_policy(plan.grid_refinement)
    validate_plan_contract(plan)
    directory = Path(resolve_work_dir(kwargs.get('work_dir'))).resolve() / plan.study_id
    directory.resolve().relative_to(Path(resolve_work_dir(kwargs.get('work_dir'))).resolve())
    directory.mkdir(parents=True, exist_ok=True)
    state_path, plan_path = directory / STATE_FILE, directory / 'study-plan.json'
    contract = fingerprint({'policy': plan.grid_refinement, 'source': plan.grid_refinement_source})
    if state_path.exists():
        state = read_json(state_path)
        if (state.get('schema') != STATE_SCHEMA or state.get('study_id') != plan.study_id
                or state.get('contract') != contract):
            raise ValueError('Saved grid refinement contract differs; prepare a new Study to change it')
        known = {c['case_id']: c for c in state['seed_cases']}
        known.update({a['case']['case_id']: a['case'] for a in state['additions']})
        for case in plan.cases:
            expected = known.get(case.case_id)
            if not expected or key(case.to_dict()) != key(expected):
                raise ValueError('A refined Study case was changed; prepare a new Study for a different contract')
        plan.cases = [StudyCase.from_dict(case) for case in known.values()]
    else:
        source = StudySpec.from_dict(plan.grid_refinement_source)
        seeds = _model_cases(source, default_study_capabilities())
        if key([c.to_dict() for c in plan.cases]) != key([c.to_dict() for c in seeds]):
            raise ValueError('Grid seed cases must match the frozen source specification')
        state = {'schema': STATE_SCHEMA, 'study_id': plan.study_id, 'contract': contract,
                 'seed_cases': [c.to_dict() for c in seeds], 'additions': [], 'decisions': [],
                 'axis_rounds': {a: 0 for a in plan.grid_refinement['axes']},
                 'axis_cursor': 0, 'status': 'running'}
        state['strategy'] = 'local_cells' if len(plan.grid_refinement['axes']) == 2 else 'axis_alternating'
        if state['strategy'] == 'local_cells':
            from computational_study_agent.grid.local import seed_cells
            state['cells'] = seed_cells(seeds, plan.grid_refinement['axes'])
        _write(state_path, state, 'grid-refinement-state')
    local = state.get('strategy') == 'local_cells'
    seeds = [StudyCase.from_dict(c) for c in state['seed_cases']]
    policy, axes = plan.grid_refinement, plan.grid_refinement['axes']
    # Postprocessing is performed once, after all sampling rounds.
    postprocess = kwargs.pop('postprocess', False)
    plot_specs = kwargs.pop('postprocess_specs', None)
    first_run = True
    while True:
        ensure_plan_cost_estimate(plan)
        _write(plan_path, plan.to_dict(), 'study-plan')
        try:
            report = runner(plan, **kwargs)
        except CostApprovalRequired:
            state['status'] = 'cost_approval_required'
            _write(state_path, state, 'grid-refinement-state')
            report_path = directory / 'study-report.json'
            if report_path.exists():
                saved_report = StudyReport.from_dict(read_json(report_path))
                attach_grid_report(saved_report, plan, state)
                _write(report_path, saved_report.to_dict(), 'study-report')
            raise
        # Explicit resume=False applies to the initial invocation only, never to
        # the completed points on subsequent sampling rounds.
        kwargs['resume'] = True
        if first_run:
            kwargs.pop('study_report', None)
            first_run = False
        checkpoint = read_json(directory / 'study-state.json')
        if any((c.get('execution') or {}).get('pending') for c in checkpoint['cases'].values()):
            state['status'] = 'awaiting_results'
            break
        if policy.get('require_converged'):
            blocked = unconverged_cases(report, plan)
            state['awaiting_convergence'] = blocked
            state['result_identity'] = _result_identity(report)
            if blocked:
                overrides = state.setdefault('convergence_retry_overrides', {})
                retry = policy.get('convergence_retry')
                retry_ids = [case_id for case_id in blocked if retry and case_id not in overrides]
                for case_id in retry_ids:
                    overrides[case_id] = copy.deepcopy(retry)
                state['status'] = 'retrying_convergence' if retry_ids else 'awaiting_convergence'
                _write(state_path, state, 'grid-refinement-state')
                if retry_ids:
                    continue
                break
        if grid_round_target is not None and state.get('generation', 0) >= grid_round_target:
            state['status'] = 'awaiting_round_barrier'
            state['result_identity'] = _result_identity(report)
            break
        if local:
            from computational_study_agent.grid.local import assess_cells, choose_cells, commit_cells
            candidates, intervals = assess_cells(plan, report, state['cells'])
        else:
            candidates, intervals = assess_grid(plan, report, seeds)
        state['intervals'] = intervals
        state['remaining_candidates'] = [{k: v for k, v in c.items() if k != 'variables'} for c in candidates]
        state['result_identity'] = _result_identity(report)
        remaining = policy['max_new_points'] - len(state['additions'])
        chosen, chosen_axis = [], None
        if local:
            chosen = choose_cells(candidates, remaining)
            chosen_axis = ','.join(axes)
        else:
            for offset in range(len(axes)):
                axis = axes[(state['axis_cursor'] + offset) % len(axes)]
                if state['axis_rounds'][axis] >= policy['max_rounds']:
                    continue
                options = sorted((c for c in candidates if c['axis'] == axis),
                                 key=lambda c: (-c['max_error_ratio'], key(c['group']), c['interval']))
                available = remaining
                for candidate in options:
                    if candidate['point_count'] <= available:
                        chosen.append(candidate)
                        available -= candidate['point_count']
                if chosen:
                    chosen_axis = axis
                    break
        if not chosen:
            if candidates:
                eligible = candidates if local else [c for c in candidates if state['axis_rounds'][c['axis']] < policy['max_rounds']]
                state['status'] = 'budget_exhausted' if eligible else 'max_rounds_reached'
            elif any(i['status'] == 'insufficient_evidence' for i in intervals):
                state['status'] = 'insufficient_evidence'
            elif any(i['status'] == 'max_rounds_reached' for i in intervals):
                state['status'] = 'max_rounds_reached'
            elif any(i['status'] == 'min_spacing_reached' for i in intervals):
                state['status'] = 'min_spacing_reached'
            else:
                state['status'] = 'satisfied'
            break
        round_number = max(c['round'] for c in chosen) if local else state['axis_rounds'][chosen_axis] + 1
        additions, decisions = [], []
        # Refined points have their own sequence; reuse persisted IDs on resume.
        numbered = [int(c.case_id[5:]) for c in plan.cases
                    if c.case_id.startswith('grid-') and c.case_id[5:].isdigit()]
        next_number = max([0] + numbered) + 1
        known_points = {key(case.variables): case for case in plan.cases}
        for candidate in chosen:
            candidate_round = candidate['round'] if local else round_number
            addition_count = len(additions)
            ids = []
            for variables in candidate['variables']:
                if local and key(variables) in known_points:
                    ids.append(known_points[key(variables)].case_id)
                    continue
                case = build_point(plan.grid_refinement_source, variables)
                case.case_id = 'grid-{0:04d}'.format(next_number)
                next_number += 1
                if any(c.case_id == case.case_id or key(c.variables) == key(variables) for c in plan.cases):
                    raise ValueError('Grid proposal duplicates an existing calculation')
                ids.append(case.case_id)
                plan.cases.append(case)
                known_points[key(variables)] = case
                additions.append({'case': case.to_dict(), 'axis': chosen_axis, 'round': candidate_round})
            decisions.append({**{k: v for k, v in candidate.items() if k != 'variables'},
                              'round': candidate_round, 'case_ids': ids,
                              'point_count': len(additions) - addition_count})
        issues = validate_study_plan(plan)
        if has_errors(issues):
            raise ValueError('Generated grid points are invalid: ' + '; '.join(i.message for i in issues if i.severity == 'error'))
        groups(plan.cases, axes, rectangular=not local)
        state['additions'].extend(additions)
        state['decisions'].extend(decisions)
        if local:
            state['cells'] = commit_cells(state['cells'], chosen, axes)
            for axis in axes:
                state['axis_rounds'][axis] = max(state['axis_rounds'][axis], round_number)
        else:
            state['axis_rounds'][chosen_axis] = round_number
            state['axis_cursor'] = (axes.index(chosen_axis) + 1) % len(axes)
        state['status'] = 'running'
        state['generation'] = state.get('generation', 0) + 1
        # Persist the complete insertion transaction before submission. If the
        # coordinator dies before the next plan write, the cases are recovered here.
        _write(state_path, state, 'grid-refinement-state')
    _write(state_path, state, 'grid-refinement-state')
    attach_grid_report(report, plan, state)
    artifact = default_artifact_repository().register_existing(state_path, kind='grid-refinement-state', mime_type='application/json')
    if artifact and artifact not in report.artifacts:
        report.artifacts.append(artifact)
    if postprocess and state['status'] != 'awaiting_results':
        from computational_study_agent.postprocessing import run_postprocessing
        result = run_postprocessing(report, specs=plot_specs, output_dir=str(directory / 'postprocessing'))
        report.artifacts.extend(result.get('artifacts') or [])
        report.summary += '; postprocessing=' + result['status']
    _write(directory / 'study-report.json', report.to_dict(), 'study-report')
    return report
