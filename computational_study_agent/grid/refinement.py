"""Deterministic midpoint sampling of one or two model-parameter axes.

This module proposes calculations; it never runs a solver or infers a phase.
Seed coordinates form a rectangular grid. Two-dimensional refinement may be local.
"""
from __future__ import annotations

from pyscf_agent.serialization import json_fingerprint, canonical_json

import itertools
import math
import re
from decimal import Decimal


AXES = ('U', 'V', 't', 'epsilon')
METRICS = {
    'energy_per_site': None,
    'mean_double_occupancy': ('double_occupancy_suppression', 'mean_double_occupancy'),
    'nearest_neighbor_spin_correlation': ('nearest_neighbor_spin_correlation', 'mean'),
    'nearest_neighbor_charge_correlation': ('nearest_neighbor_charge_correlation', 'mean'),
    'staggered_magnetization': ('sublattice_order_parameters', 'staggered_magnetization'),
    'sublattice_charge_imbalance': ('sublattice_order_parameters', 'charge_imbalance'),
}
DEFAULT_METRICS = [
    {'name': 'energy_per_site', 'atol': 1e-3, 'rtol': 0.0, 'required': True},
    {'name': 'mean_double_occupancy', 'atol': 1e-3, 'rtol': 0.0, 'required': True},
]


def canonical(value):
    # 1 and 1.0 identify the same coordinate, including fixed numeric variables.
    if isinstance(value, dict):
        return {key: canonical(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [canonical(item) for item in value]
    if isinstance(value, (float, int)) and not isinstance(value, bool):
        return float(value)
    return value


def key(value):
    return canonical_json(canonical(value), ensure_ascii=True)


def fingerprint(value):
    return json_fingerprint(canonical(value), ensure_ascii=True)


def number(value):
    return (isinstance(value, (float, int)) and not isinstance(value, bool)
            and math.isfinite(value))


def normalize_policy(raw):
    if not isinstance(raw, dict):
        raise ValueError('grid_refinement must be an object')
    if not raw:
        return {}
    allowed = {'enabled', 'axes', 'metrics', 'max_new_points', 'max_rounds', 'min_spacing', 'max_interval',
               'require_converged', 'convergence_retry'}
    if set(raw) - allowed:
        raise ValueError('Unknown grid_refinement fields: ' + ', '.join(sorted(set(raw) - allowed)))
    enabled = raw.get('enabled', True)
    if not isinstance(enabled, bool):
        raise ValueError('grid_refinement.enabled must be a boolean')
    if not enabled:
        return {}
    axes = raw.get('axes')
    if (not isinstance(axes, list) or not 1 <= len(axes) <= 2
            or any(not isinstance(axis, str) or not re.fullmatch(r'[A-Za-z_][A-Za-z0-9_]*', axis) for axis in axes) or len(set(axes)) != len(axes)):
        raise ValueError('grid_refinement.axes requires one or two distinct continuous grid variable names')
    result = {'enabled': True, 'axes': axes[:]}  # axis order is the sweep order
    if 'require_converged' in raw:
        if not isinstance(raw['require_converged'], bool):
            raise ValueError('grid_refinement.require_converged must be a boolean')
        result['require_converged'] = raw['require_converged']
    if 'convergence_retry' in raw:
        retry = raw['convergence_retry']
        if not result.get('require_converged') or not isinstance(retry, dict) or not retry:
            raise ValueError('convergence_retry requires require_converged and numerical retry settings')
        if set(retry) - {'correlation_potential_mixing', 'max_iterations', 'diis_enabled'}:
            raise ValueError('convergence_retry only supports DMET damping and iteration settings')
        if 'correlation_potential_mixing' in retry and (not number(retry['correlation_potential_mixing'])
                or not 0 < retry['correlation_potential_mixing'] <= 1):
            raise ValueError('convergence_retry mixing must be in (0, 1]')
        if 'max_iterations' in retry and (isinstance(retry['max_iterations'], bool)
                or not isinstance(retry['max_iterations'], int) or retry['max_iterations'] < 1):
            raise ValueError('convergence_retry max_iterations must be a positive integer')
        if 'diis_enabled' in retry and not isinstance(retry['diis_enabled'], bool):
            raise ValueError('convergence_retry diis_enabled must be a boolean')
        result['convergence_retry'] = dict(retry)
    for name, default in (('max_new_points', 64), ('max_rounds', 4)):
        value = raw.get(name, default)
        if isinstance(value, bool) or not isinstance(value, int) or value < 1:
            raise ValueError('grid_refinement.' + name + ' must be a positive integer')
        result[name] = value
    for name, default in (('min_spacing', 1e-4), ('max_interval', None)):
        value = raw.get(name, {})
        if not isinstance(value, dict) or set(value) - set(axes):
            raise ValueError('grid_refinement.' + name + ' must map selected axes to positive numbers')
        result[name] = {}
        for axis in axes:
            spacing = value.get(axis, default)
            if spacing is not None:
                if not number(spacing) or spacing <= 0:
                    raise ValueError('grid_refinement.' + name + ' values must be positive and finite')
                result[name][axis] = float(spacing)
    metrics = raw.get('metrics', DEFAULT_METRICS)
    if not isinstance(metrics, list) or not metrics:
        raise ValueError('grid_refinement.metrics must be a nonempty list')
    normalized, seen = [], set()
    for metric in metrics:
        if not isinstance(metric, dict) or set(metric) - {'name', 'atol', 'rtol', 'required'}:
            raise ValueError('Each grid metric requires name, atol, rtol and optional required')
        name = metric.get('name')
        if name not in METRICS or name in seen:
            raise ValueError('Unsupported or duplicate grid metric: ' + str(name))
        item = {'name': name, 'atol': metric.get('atol', 1e-3),
                'rtol': metric.get('rtol', 0.0), 'required': metric.get('required', True)}
        if (any(not number(item[k]) or item[k] < 0 for k in ('atol', 'rtol'))
                or item['atol'] + item['rtol'] <= 0 or not isinstance(item['required'], bool)):
            raise ValueError('Grid metric tolerances must be finite, nonnegative and not both zero')
        normalized.append(item)
        seen.add(name)
    if not any(item['required'] for item in normalized):
        raise ValueError('At least one grid metric must be required')
    result['metrics'] = normalized
    return result


def validate_source(spec, policy):
    if not policy:
        return
    if spec.system_type != 'model_hamiltonian' or spec.study_mode not in ('static', 'static_scan'):
        raise ValueError('Grid refinement currently requires a static model-Hamiltonian Study with explicit solvers')
    if policy.get('convergence_retry'):
        solver = spec.base_task.get('solver')
        if not isinstance(solver, dict) or solver.get('name') != 'dmet':
            raise ValueError('Grid convergence retries require a DMET solver')
    design = spec.case_design
    if design:
        if design.get('mode', 'grid') != 'grid' or design.get('overrides'):
            raise ValueError('Grid refinement requires a grid template without case-specific overrides')
        variables = design.get('variables') or {}
        # Axis variables may change Hamiltonian coefficients, never topology,
        # particle sectors or solver settings. Fixed variables can still do so.
        template = design.get('template') or {}
        def dependencies(value):
            if isinstance(value, dict):
                return set().union(*(dependencies(v) for v in value.values()), set())
            if isinstance(value, list):
                return set().union(*(dependencies(v) for v in value), set())
            if isinstance(value, str) and '$' in value:
                return {axis for axis in policy['axes'] if re.search(r'\b' + re.escape(axis) + r'\b', value)}
            return set()
        def uses_axis(value):
            return bool(dependencies(value))
        if uses_axis(template.get('request_updates')):
            raise ValueError('Refined axes cannot control request or solver settings')
        used = set()
        for operation in template.get('operations') or []:
            if uses_axis(operation):
                rest = {k: v for k, v in operation.items() if k != 'value'}
                if (uses_axis(rest) or operation.get('parameter') not in AXES
                        or operation.get('op') not in ('set_global_parameter', 'set_site_parameter',
                            'shift_site_parameter', 'scale_site_parameter', 'set_bond_parameter',
                            'shift_bond_parameter', 'scale_bond_parameter')):
                    raise ValueError('Refined axes may only control continuous coefficient values')
                used.update(dependencies(operation.get('value')))
        if set(policy['axes']) - used:
            raise ValueError('Each refined axis must control a coefficient in the grid template')
    else:
        variables = spec.sweep
        if set(policy['axes']) - set(AXES):
            raise ValueError('Sweep refinement supports U, V, t and epsilon; named variables require a grid template')
    for axis in policy['axes']:
        values = variables.get(axis)
        if (not isinstance(values, list) or len(values) < 2 or any(not number(v) for v in values)
                or len(set(values)) != len(values)):
            raise ValueError('Refined axis ' + axis + ' needs at least two distinct finite seed values')


def prepare_source(spec, policy):
    from computational_study_agent.planner import _load_base_model_spec
    validate_source(spec, policy)
    source = spec.to_dict()
    source['grid_refinement'] = {}
    source['base_model_spec'] = _load_base_model_spec(spec)
    source['base_model_input_file'] = None  # freeze the model before any execution
    return source


def build_point(source, variables):
    from computational_study_agent.planner import _model_cases
    from computational_study_agent.schema import StudySpec
    from computational_study_agent.capabilities import default_study_capabilities
    spec = StudySpec.from_dict(source)
    if spec.case_design:
        spec.case_design['variables'] = {name: [value] for name, value in variables.items()}
    else:
        spec.sweep = {name: [value] for name, value in variables.items()}
    return _model_cases(spec, default_study_capabilities())[0]


def groups(cases, axes, *, rectangular=True):
    result = {}
    for case in cases:
        fixed = {name: value for name, value in case.variables.items() if name not in axes}
        group = result.setdefault(key(fixed), {'fixed': fixed, 'cases': [], 'coordinates': {a: set() for a in axes}})
        group['cases'].append(case)
        for axis in axes:
            value = case.variables.get(axis)
            if not number(value):
                raise ValueError('Every refined case requires a finite ' + axis + ' coordinate')
            group['coordinates'][axis].add(float(value))
    for group in result.values():
        coordinates = group['coordinates']
        expected = math.prod(len(v) for v in coordinates.values())
        if ((rectangular and len(group['cases']) != expected)
                or len({key(c.variables) for c in group['cases']}) != len(group['cases'])):
            raise ValueError('Each fixed-parameter group must be a complete rectangular grid with unique coordinates')
        # The solver configuration is invariant within a refined grid.
        if len({key(c.request.get('solver')) for c in group['cases']}) != 1:
            raise ValueError('Solver options must be constant within each refined grid')
    return result


def validate_plan_contract(plan):
    policy = normalize_policy(plan.grid_refinement)
    if not policy:
        return
    from computational_study_agent.schema import StudySpec
    if not plan.grid_refinement_source:
        raise ValueError('A refined plan requires its frozen grid_refinement_source')
    source = StudySpec.from_dict(plan.grid_refinement_source)
    validate_source(source, policy)
    if plan.system_type != source.system_type:
        raise ValueError('Grid source and plan system types differ')
    groups(plan.cases, policy['axes'], rectangular=len(policy['axes']) == 1)


def metric_sample(record, metric, case, row):
    task = (record or {}).get('task_report') or {}
    results = task.get('structured_results') or {}
    if (task.get('execution_status') != 'succeeded' or row.get('publication_eligible') is False
            or any(results.get(k) is False for k in ('converged', 'scf_converged', 'ccsd_converged', 'dmet_converged'))):
        return None
    if results.get('solver') in ('mp2', 'ccsd', 'ccsd_t') and results.get('reference_converged') is False:
        return None
    name = metric['name']
    if name == 'energy_per_site':
        value = results.get(name)
        energy = results.get('final_energy', results.get('energy'))
        if value is None and number(energy):
            value = energy / len(case.model_spec['sites'])
        scope = {'unit': results.get('energy_unit', case.model_spec.get('energy_unit')),
                 'site_count': len(case.model_spec['sites'])}
    else:
        diagnostic, field = METRICS[name]
        diagnostics = (results.get('strong_correlation_diagnostics') or {}).get('diagnostics') or []
        item = next((d.get('value') for d in diagnostics if d.get('name') == diagnostic), None)
        if (not isinstance(item, dict) or item.get('valid') is False or item.get('quality_errors')
                or item.get('status') in ('failed', 'invalid')):
            return None
        value = item.get(field)
        scope = {k: item.get(k) for k in ('coverage', 'source', 'scope')}
        local = results.get('dmet_local_observables') or {}
        field = ('double_occupancy' if name == 'mean_double_occupancy' else
                 'spin_correlation' if name == 'nearest_neighbor_spin_correlation' else
                 'charge_correlation' if name == 'nearest_neighbor_charge_correlation' else 'density')
        def mask(value):
            if isinstance(value, list):
                return [mask(v) for v in value]
            return number(value)
        if local:
            scope['covered_entries'] = mask(local.get(field))

    return (float(value), key(scope)) if number(value) else None


def assess_grid(plan, report, seed_cases):
    """Return line insertions and interval evidence; no mutation or solver call."""
    policy = plan.grid_refinement
    axes = policy['axes']
    seeds = groups(seed_cases, axes)
    current = groups(plan.cases, axes)
    records = {c['case_id']: c for c in report.cases}
    rows = {r['case_id']: r for r in report.comparison_table}
    candidates, assessments = [], []
    for group_key, group in current.items():
        points = {key(c.variables): c for c in group['cases']}
        for axis in axes:
            other_axes = [a for a in axes if a != axis]
            lines = [dict(zip(other_axes, vals)) for vals in itertools.product(
                *(sorted(group['coordinates'][a]) for a in other_axes))]
            def evaluate(left, right, cause=None, depth=0):
                middle = float((Decimal(str(left)) + Decimal(str(right))) / 2)
                base = {'group': group['fixed'], 'axis': axis, 'interval': [left, right], 'midpoint': middle}
                evidence, missing, usable_endpoints = [], [], False
                has_middle = middle in group['coordinates'][axis]
                for line in lines:
                    samples = []
                    for coordinate in ([left, middle, right] if has_middle else [left, right]):
                        case = points[key({**group['fixed'], **line, axis: coordinate})]
                        samples.append((case, records.get(case.case_id), rows.get(case.case_id, {})))
                    required_ok, measured = True, 0
                    for metric in policy['metrics']:
                        values = [metric_sample(record, metric, case, row) for case, record, row in samples]
                        if any(v is None for v in values) or len({v[1] for v in values if v}) != 1:
                            missing.append({'line': line, 'metric': metric['name'], 'required': metric['required']})
                            if metric['required']:
                                required_ok = False
                            continue
                        measured += 1
                        if has_middle:
                            a, b, c = [v[0] for v in values]
                            residual = abs(b - (a + c) / 2)
                            # A fixed physical absolute tolerance plus a local magnitude scale.
                            tolerance = metric['atol'] + metric['rtol'] * max(abs(a), abs(b), abs(c))
                            ratio = residual / tolerance if tolerance else (0.0 if residual == 0 else 1e300)
                            evidence.append({'line': line, 'metric': metric['name'], 'error': residual,
                                             'tolerance': tolerance, 'ratio': ratio})
                    usable_endpoints |= required_ok and measured > 0
                max_error = max((e['ratio'] for e in evidence), default=0.0)
                unknown = any(m['required'] for m in missing)
                maximum = policy['max_interval'].get(axis)
                force = maximum is not None and (right - left) / 2 > maximum
                if has_middle and max_error <= 1 and not force:
                    assessments.append({**base, 'status': 'insufficient_evidence' if unknown else 'satisfied',
                                        'evidence': evidence, 'missing': missing})
                    return
                if not usable_endpoints:
                    assessments.append({**base, 'status': 'insufficient_evidence', 'missing': missing})
                    return
                if middle <= left or middle >= right or depth >= 48 or (right - left) / 2 < policy['min_spacing'][axis]:
                    assessments.append({**base, 'status': 'min_spacing_reached', 'evidence': evidence, 'missing': missing})
                    return
                if has_middle:
                    trigger = {'reason': 'max_interval' if force and max_error <= 1 else 'interpolation_error',
                               'evidence': evidence, 'max_error_ratio': max_error}
                    evaluate(left, middle, trigger, depth + 1)
                    evaluate(middle, right, trigger, depth + 1)
                else:
                    variables = [{**group['fixed'], **line, axis: middle} for line in lines]
                    candidates.append({**base, **(cause or {'reason': 'midpoint_probe', 'max_error_ratio': 0.0}),
                                       'variables': variables, 'point_count': len(variables)})
            ordered = sorted(seeds[group_key]['coordinates'][axis])
            for left, right in zip(ordered, ordered[1:]):
                evaluate(left, right)
    return candidates, assessments
