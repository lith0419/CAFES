"""Local rectangular-cell refinement of a two-parameter model scan.

Centre residuals are sampling heuristics, not error bounds. Shared edge probes
already present in the Study are also checked against bilinear interpolation.
"""
from __future__ import annotations

import copy
import itertools
from decimal import Decimal

from computational_study_agent.grid.refinement import fingerprint, groups, key, metric_sample


def midpoint(a, b):
    return float((Decimal(str(a)) + Decimal(str(b))) / 2)


def make_cell(fixed, bounds, depth):
    identity = {'group': fixed, 'bounds': bounds}
    return {**copy.deepcopy(identity), 'cell_id': 'cell-' + fingerprint(identity)[:20], 'depth': depth}


def seed_cells(cases, axes):
    cells = []
    for group in groups(cases, axes).values():
        coordinates = [sorted(group['coordinates'][axis]) for axis in axes]
        intervals = [list(zip(values, values[1:])) for values in coordinates]
        for pair in itertools.product(*intervals):
            cells.append(make_cell(group['fixed'], dict(zip(axes, map(list, pair))), 0))
    return cells


def cell_points(cell, axes):
    x, y = axes
    a, b = cell['bounds'][x]
    c, d = cell['bounds'][y]
    m, n = midpoint(a, b), midpoint(c, d)
    def point(u, v):
        return {**cell['group'], x: u, y: v}
    corners = [point(a, c), point(b, c), point(a, d), point(b, d)]
    # The weights match the corner order above.
    probes = [(point(m, n), [.25] * 4),
              (point(m, c), [.5, .5, 0, 0]), (point(m, d), [0, 0, .5, .5]),
              (point(a, n), [.5, 0, .5, 0]), (point(b, n), [0, .5, 0, .5])]
    return corners, probes


def children(cell, axes):
    intervals = []
    for axis in axes:
        a, b = cell['bounds'][axis]
        m = midpoint(a, b)
        intervals.append([[a, m], [m, b]])
    return [make_cell(cell['group'], dict(zip(axes, pair)), cell['depth'] + 1)
            for pair in itertools.product(*intervals)]


def assess_cells(plan, report, cells):
    policy, axes = plan.grid_refinement, plan.grid_refinement['axes']
    groups(plan.cases, axes, rectangular=False)
    points = {key(case.variables): case for case in plan.cases}
    records = {case['case_id']: case for case in report.cases}
    rows = {row['case_id']: row for row in report.comparison_table}
    def sample(variables, metric):
        case = points.get(key(variables))
        return metric_sample(records.get(case.case_id), metric, case, rows.get(case.case_id, {})) if case else None
    candidates, assessments = [], []
    for cell in cells:
        corners, probes = cell_points(cell, axes)
        centre = probes[0][0]
        has_centre = key(centre) in points
        evidence, missing = [], []
        for metric in policy['metrics']:
            values = [sample(point, metric) for point in corners]
            if any(v is None for v in values) or len({v[1] for v in values if v}) != 1:
                missing.append({'metric': metric['name'], 'required': metric['required'], 'where': 'corners'})
                continue
            if not has_centre:
                continue
            for variables, weights in probes:
                if key(variables) not in points:
                    continue
                measured = sample(variables, metric)
                if measured is None or measured[1] != values[0][1]:
                    missing.append({'metric': metric['name'], 'required': metric['required'], 'where': variables})
                    continue
                prediction = sum(weight * value[0] for weight, value in zip(weights, values))
                error = abs(measured[0] - prediction)
                tolerance = metric['atol'] + metric['rtol'] * max(abs(measured[0]), *(abs(v[0]) for v in values))
                evidence.append({'metric': metric['name'], 'point': variables, 'error': error,
                                 'tolerance': tolerance, 'ratio': error / tolerance if tolerance else (0. if error == 0 else 1e300)})
        base = {**copy.deepcopy(cell), 'evidence': evidence, 'missing': missing}
        ratio = max((item['ratio'] for item in evidence), default=0.)
        force = any(cell['bounds'][axis][1] - cell['bounds'][axis][0] > maximum
                    for axis, maximum in policy['max_interval'].items())
        can_probe = all(a < midpoint(a, b) < b and (b - a) / 2 >= policy['min_spacing'][axis]
                        for axis, (a, b) in cell['bounds'].items())
        if any(item['required'] for item in missing):
            status = 'insufficient_evidence'
        elif has_centre and ratio <= 1 and not force:
            status = 'satisfied'
        elif not can_probe:
            status = 'min_spacing_reached'
        elif cell['depth'] >= policy['max_rounds'] or (has_centre and cell['depth'] + 1 >= policy['max_rounds']):
            status = 'max_rounds_reached'
        else:
            action = 'split_cell' if has_centre else 'probe_center'
            variables = ([p[0] for p in probes[1:]] if has_centre else [centre])
            variables = [p for p in variables if key(p) not in points]
            candidates.append({**base, 'action': action, 'axis': ','.join(axes),
                               'round': cell['depth'] + 1,
                               'reason': ('max_interval' if force and ratio <= 1 else 'interpolation_error') if has_centre else 'center_probe',
                               'max_error_ratio': ratio, 'variables': variables, 'point_count': len(variables)})
            status = 'needs_refinement' if has_centre else 'needs_probe'
        assessments.append({**base, 'status': status})
    return candidates, assessments


def choose_cells(candidates, remaining):
    """Budget whole local actions, charging shared coordinates only once."""
    chosen, reserved = [], set()
    for candidate in sorted(candidates, key=lambda c: (-c['max_error_ratio'], c['depth'], key(c['group']), key(c['bounds']))):
        needed = {key(point) for point in candidate['variables']} - reserved
        if len(needed) <= remaining:
            chosen.append(candidate)
            reserved.update(needed)
            remaining -= len(needed)
    return chosen


def commit_cells(cells, chosen, axes):
    split = {candidate['cell_id'] for candidate in chosen if candidate['action'] == 'split_cell'}
    return [child for cell in cells for child in (children(cell, axes) if cell['cell_id'] in split else [cell])]


def plot_cells(plan, cells):
    points = {key(case.variables): case.case_id for case in plan.cases}
    result = []
    for cell in cells:
        corners, probes = cell_points(cell, plan.grid_refinement['axes'])
        result.append({**copy.deepcopy(cell),
                       'corner_case_ids': [points.get(key(point)) for point in corners],
                       'center_case_id': points.get(key(probes[0][0]))})
    return result
