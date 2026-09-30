"""Compare DMET candidates and select density donors from final observables.

All classifications are numerical labels for saved solutions, not a claim of
thermodynamic phases. This module does not execute calculations.
"""

from __future__ import annotations

import copy
import math

from pyscf_agent.backend.model_hamiltonian.spec import normalize_model_spec
from pyscf_agent.providers.libdmet.dmet import validate_dmet_model_request
from pyscf_agent.providers.libdmet.density_restart import (
    state_metadata,
    normalize_density_source,
    STATE_KIND,
)
from pyscf_agent.result_quality import evaluate_quality_checks
from .grid.refinement import fingerprint, number


DEFAULT_POLICY = {
    'charge_threshold': 1e-3,
    'magnetic_threshold': 1e-3,
    'threshold_margin': 0.2,
    'state_tolerance': 1e-2,
    'energy_tolerance': 1e-6,
    'distance_scales': {'U': 1.0, 'V': 1.0},
}


def normalize_policy(raw=None):
    raw = {} if raw is None else raw
    if not isinstance(raw, dict) or set(raw) - set(DEFAULT_POLICY):
        raise ValueError('Unknown DMET branch analysis policy fields')
    result = {**copy.deepcopy(DEFAULT_POLICY), **copy.deepcopy(raw)}
    for key in (
        'charge_threshold',
        'magnetic_threshold',
        'state_tolerance',
        'energy_tolerance',
    ):
        if not number(result[key]) or result[key] <= 0:
            raise ValueError(key + ' must be finite and positive')
    if (
        not number(result['threshold_margin'])
        or not 0 <= result['threshold_margin'] < 1
    ):
        raise ValueError('threshold_margin must be in [0, 1)')
    scales = result['distance_scales']
    if (
        not isinstance(scales, dict)
        or set(scales) != {'U', 'V'}
        or any(not number(v) or v <= 0 for v in scales.values())
    ):
        raise ValueError('distance_scales requires positive U and V scales')
    return result


def is_dmet(record):
    solver = (record.get('request') or {}).get('solver')
    return (solver.get('name') if isinstance(solver, dict) else solver) == 'dmet'


def candidate_records(record):
    """Current and historical Runs, deduplicated without relabeling their requests."""
    seen = set()
    for candidate in [record, *(record.get('candidate_runs') or [])]:
        report = candidate.get('task_report') or {}
        run_id = report.get('run_id')
        if report.get('execution_status') in ('blocked', 'not_executed', 'pending'):
            continue
        if run_id and run_id not in seen:
            seen.add(run_id)
            yield candidate


def preserve_candidates(previous, current):
    if not is_dmet(previous) and not is_dmet(current):
        return
    current_run = (current.get('task_report') or {}).get('run_id')
    candidates = []
    for candidate in candidate_records(previous):
        report = candidate['task_report']
        if report['run_id'] == current_run and (current.get('task_report') or {}).get(
            'execution_status'
        ) not in ('blocked', 'not_executed', 'pending'):
            continue
        # Blocked attempts do not own numerical Runs and must never replace one.
        if report.get('execution_status') in ('blocked', 'not_executed', 'pending'):
            continue
        saved = {
            key: copy.deepcopy(candidate[key])
            for key in (
                'case_id',
                'request',
                'model_spec',
                'variables',
                'task_report',
                'attempt_count',
            )
            if key in candidate
        }
        candidates.append(saved)
    if candidates:
        current['candidate_runs'] = candidates


def case_identity(case):
    """Use the provider's normalized site basis and resolved fragment/spin policy."""
    record = case.to_dict() if hasattr(case, 'to_dict') else case
    request = record.get('request') or {}
    solver = request.get('solver') or {}
    if not is_dmet(record):
        raise ValueError('DMET continuation requires DMET cases')
    options = solver.get('options') or {} if isinstance(solver, dict) else {}
    spec = (request.get('model_hamiltonian') or {}).get('spec') or record.get(
        'model_spec'
    )
    if not isinstance(spec, dict) or not spec.get('sites'):
        raise ValueError('DMET continuation requires an inline, resolved model spec')
    spec = normalize_model_spec(spec)
    errors, configuration = validate_dmet_model_request(spec, options)
    if errors:
        raise ValueError('; '.join(errors))
    if configuration['execution_mode'] != 'finite_graph':
        raise ValueError('DMET branch continuation requires finite_graph execution')
    metadata = state_metadata(spec, configuration, converged=False)
    # Impurity options include the shared beta/smearing policy when enabled.
    compatibility = metadata['compatibility']
    coordinates = {}
    for axis, key in (('U', 'onsite_u'), ('V', 'intersite_v')):
        values = metadata['parameters'][key]
        if (
            not values
            or any(not number(v) for v in values)
            or max(values) - min(values) > 1e-10
        ):
            raise ValueError(
                'Automatic DMET continuation requires uniform site U and bond V'
            )
        coordinates[axis] = values[0]
    return {
        'family': fingerprint(compatibility),
        'point_id': fingerprint(
            {'compatibility': compatibility, 'parameters': metadata['parameters']}
        ),
        'coordinates': coordinates,
        'energy_unit': spec.get('energy_unit', 'a.u.'),
    }


def classify(results, policy):
    diagnostics = (results.get('strong_correlation_diagnostics') or {}).get(
        'diagnostics'
    ) or []
    order = next(
        (
            d.get('value')
            for d in diagnostics
            if isinstance(d, dict) and d.get('name') == 'sublattice_order_parameters'
        ),
        None,
    )
    if not isinstance(order, dict):
        order = (results.get('dmet_local_observables') or {}).get(
            'sublattice_order'
        ) or {}
    if not isinstance(order, dict):
        order = {}
    values = [order.get('charge_imbalance'), order.get('staggered_magnetization')]
    if (
        order.get('valid') is False
        or order.get('quality_errors')
        or order.get('status') in ('failed', 'invalid')
        or not all(number(v) for v in values)
    ):
        return {
            'branch': 'unknown',
            'charge_imbalance': None,
            'staggered_magnetization': None,
        }
    charge, magnetic = (abs(v) for v in values)
    thresholds = (policy['charge_threshold'], policy['magnetic_threshold'])
    margin = policy['threshold_margin']
    if any(
        t * (1 - margin) <= v <= t * (1 + margin)
        for v, t in zip((charge, magnetic), thresholds)
    ):
        branch = 'ambiguous'
    else:
        c, m = charge > thresholds[0], magnetic > thresholds[1]
        branch = (
            'mixed' if c and m else 'cdw' if c else 'afm' if m else 'near_unordered'
        )
    return {
        'branch': branch,
        'charge_imbalance': charge,
        'staggered_magnetization': magnetic,
    }


def candidates(records, policy=None):
    policy = normalize_policy(policy)
    result = []
    for record in records:
        for candidate in candidate_records(record):
            if not is_dmet(candidate):
                continue
            report = candidate['task_report']
            results = report.get('structured_results') or {}
            try:
                identity = case_identity(candidate)
            except (ValueError, TypeError, KeyError) as exc:
                identity = {'identity_error': str(exc)}
            quality = evaluate_quality_checks(results.get('quality_checks'))
            energy = results.get('energy_per_site')
            eligible = (
                report.get('execution_status') == 'succeeded'
                and results.get('converged') is True
                and results.get('solver', 'dmet') == 'dmet'
                and quality['publication_eligible'] is True
                and number(energy)
                and 'identity_error' not in identity
                and results.get('energy_unit', identity.get('energy_unit'))
                == identity.get('energy_unit')
            )
            options = (
                (candidate['request']['solver'].get('options') or {})
                if isinstance(candidate['request']['solver'], dict)
                else {}
            )
            source = copy.deepcopy(options.get('reference_density_source'))
            artifact = next(
                (
                    a
                    for a in report.get('artifacts') or []
                    if a.get('kind') == STATE_KIND
                ),
                None,
            )
            result.append(
                {
                    **identity,
                    'case_id': candidate.get('case_id', record.get('case_id')),
                    'run_id': report['run_id'],
                    'execution_status': report.get('execution_status'),
                    'qualified': eligible,
                    'quality_status': quality['quality_status'],
                    'energy_per_site': float(energy) if number(energy) else None,
                    **classify(results, policy),
                    'origin': 'continuation' if source else 'independent_seed',
                    'seed': options.get('reference_density_guess', 'pm'),
                    'density_source': source,
                    'density_artifact': copy.deepcopy(artifact),
                }
            )
    return result


def select_donor(target, available, branch, *, direction='nearest', policy=None):
    policy = normalize_policy(policy)
    if branch not in ('afm', 'cdw', 'mixed', 'near_unordered'):
        raise ValueError('Select a classified DMET branch')
    if direction not in ('nearest', 'increasing', 'decreasing'):
        raise ValueError('direction must be nearest, increasing, or decreasing')
    identity = case_identity(target)
    selected = []
    for candidate in available:
        artifact = candidate.get('density_artifact') or {}
        if (
            not candidate['qualified']
            or candidate['branch'] != branch
            or candidate.get('family') != identity['family']
            or candidate.get('point_id') == identity['point_id']
            or not artifact.get('path')
            or not artifact.get('sha256')
        ):
            continue
        try:
            normalize_density_source(
                {'path': artifact['path'], 'sha256': artifact['sha256']}
            )
        except ValueError:
            continue
        delta = {
            axis: identity['coordinates'][axis] - candidate['coordinates'][axis]
            for axis in ('U', 'V')
        }
        if direction != 'nearest' and (
            abs(delta['U']) > 1e-10
            or (
                delta['V'] <= 1e-10
                if direction == 'increasing'
                else delta['V'] >= -1e-10
            )
        ):
            continue
        distance = math.sqrt(
            sum((delta[k] / policy['distance_scales'][k]) ** 2 for k in delta)
        )
        selected.append(
            (
                distance,
                candidate['energy_per_site'],
                candidate['case_id'],
                candidate['run_id'],
                candidate,
            )
        )
    if not selected:
        return None
    distance, _, _, _, donor = min(selected, key=lambda item: item[:4])
    return {**copy.deepcopy(donor), 'distance': distance, 'direction': direction}


def analyze_branches(records, policy=None):
    policy = normalize_policy(policy)
    available = candidates(records, policy)
    points = {}
    for candidate in available:
        if 'point_id' not in candidate:
            continue
        point = points.setdefault(
            candidate['point_id'],
            {
                'point_id': candidate['point_id'],
                'family': candidate['family'],
                'coordinates': candidate['coordinates'],
                'energy_unit': candidate['energy_unit'],
                'candidates': [],
            },
        )
        point['candidates'].append(candidate)
    for point in points.values():
        valid = sorted(
            (c for c in point['candidates'] if c['qualified']),
            key=lambda c: (c['energy_per_site'], c['run_id']),
        )
        point['winner'] = valid[0] if valid else None
        point['branch_minima'] = {}
        for candidate in valid:
            if candidate['branch'] not in ('unknown', 'ambiguous'):
                point['branch_minima'].setdefault(candidate['branch'], candidate)
        known = [c for c in valid if c['branch'] not in ('unknown', 'ambiguous')]
        point['possible_hysteresis'] = any(
            max(
                abs(a[k] - b[k])
                for k in ('charge_imbalance', 'staggered_magnetization')
            )
            > policy['state_tolerance']
            for i, a in enumerate(known)
            for b in known[i + 1 :]
        )
        point['energy_tied_runs'] = [
            c['run_id']
            for c in valid
            if c['energy_per_site'] - valid[0]['energy_per_site']
            <= policy['energy_tolerance']
        ]
    # Interpolate only adjacent sampled points with BOTH branches and no lower third state.
    lines = {}
    for point in points.values():
        lines.setdefault((point['family'], point['coordinates']['U']), []).append(point)
    crossings, coexistence = [], []
    for (family, u), line in sorted(lines.items()):
        line.sort(key=lambda p: p['coordinates']['V'])
        overlap = [
            p['coordinates']['V']
            for p in line
            if {'afm', 'cdw'} <= set(p['branch_minima'])
        ]
        coexistence.append(
            {
                'family': family,
                'U': u,
                'sampled_coexistence_V': overlap,
                'endpoints_are_physical_limits': False,
            }
        )
        for left, right in zip(line, line[1:]):
            if any(
                not {'afm', 'cdw'} <= set(p['branch_minima']) for p in (left, right)
            ):
                continue
            differences = [
                p['branch_minima']['afm']['energy_per_site']
                - p['branch_minima']['cdw']['energy_per_site']
                for p in (left, right)
            ]
            a, b = differences
            tolerance = policy['energy_tolerance']
            if a * b > 0 and min(abs(a), abs(b)) > tolerance:
                continue
            bracket = [p['coordinates']['V'] for p in (left, right)]
            estimate = (
                bracket[0] - a * (bracket[1] - bracket[0]) / (b - a)
                if a * b < 0
                else None
            )
            lower_third = any(
                p['winner']['energy_per_site']
                < min(p['branch_minima'][s]['energy_per_site'] for s in ('afm', 'cdw'))
                - tolerance
                for p in (left, right)
            )
            crossings.append(
                {
                    'family': family,
                    'U': u,
                    'V_bracket': bracket,
                    'V_linear_estimate': estimate,
                    'energy_difference_afm_minus_cdw': differences,
                    'status': 'lower_competing_state'
                    if lower_third
                    else 'candidate_crossing',
                    'run_ids': [
                        [p['branch_minima'][s]['run_id'] for s in ('afm', 'cdw')]
                        for p in (left, right)
                    ],
                }
            )
    return {
        'schema': 'pyscf-agent.dmet-branch-analysis.v1',
        'policy': policy,
        'order_parameter_convention': 'absolute_sublattice_half_difference',
        'selection': 'minimum_qualified_energy_per_site',
        'transfer': 'mean_field_density_only',
        'points': sorted(
            points.values(),
            key=lambda p: (p['family'], p['coordinates']['U'], p['coordinates']['V']),
        ),
        'unclassified_identity_runs': [c for c in available if 'identity_error' in c],
        'crossings': crossings,
        'coexistence': coexistence,
    }
