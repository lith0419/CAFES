from __future__ import annotations

import copy
import math
import re
from statistics import median
from typing import Any, Dict, List, Optional, Sequence

from .policies import scan_path_continuity_policy


PATH_DIAGNOSTICS_SCHEMA = 'pyscf-agent.scan-path-diagnostics.v1'

_NUMERIC_PATTERN = re.compile(r'^\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)')
_ENERGY_COLUMNS = ('final_energy', 'energy', 'total_energy')
_METHOD_COLUMNS = ('method', 'solver', 'refined_method', 'final_method')
_PATH_POLICY = scan_path_continuity_policy()
_MINIMUM_SAMPLE_COUNT = int(_PATH_POLICY['minimum_sample_count'])
_SLOPE_MISMATCH_RATIO = float(_PATH_POLICY['slope_mismatch_ratio'])
_ENERGY_RESIDUAL_HARTREE = float(_PATH_POLICY['energy_residual_hartree'])
_STRONG_SLOPE_MISMATCH_RATIO = float(_PATH_POLICY['strong_slope_mismatch_ratio'])
_MOLECULAR_METHOD_LEVELS = {
    str(method): int(level)
    for method, level in _PATH_POLICY['molecular_method_levels'].items()
}


def deferred_scan_path_diagnostics() -> Dict[str, Any]:
    """Describe a cross-case analysis that has not been explicitly requested."""
    return {
        'schema': PATH_DIAGNOSTICS_SCHEMA,
        'kind': 'scan_path_diagnostics',
        'status': 'deferred',
        'coordinate': None,
        'observable': None,
        'sample_count': 0,
        'method_transitions': [],
        'anomalies': [],
        'validated_findings': [],
        'summary': (
            'Cross-case scan-path continuity has not been analyzed. '
            'Run Analyze Results to inspect the completed tasks together.'
        ),
    }


def _numeric_value(value: Any) -> Optional[float]:
    if isinstance(value, bool) or value is None:
        return None
    if isinstance(value, (int, float)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, str):
        match = _NUMERIC_PATTERN.match(value)
        if not match:
            return None
        try:
            number = float(match.group(1))
        except (TypeError, ValueError):
            return None
        return number if math.isfinite(number) else None
    return None


def _normalized_method(row: Dict[str, Any]) -> str:
    for column in _METHOD_COLUMNS:
        value = str(row.get(column) or '').strip().lower()
        if value:
            return value
    return ''


def _case_variable_names(report: Dict[str, Any]) -> List[str]:
    names: List[str] = []
    for case in report.get('cases') or []:
        variables = case.get('variables') if isinstance(case, dict) else None
        if not isinstance(variables, dict):
            continue
        for name in variables:
            if name not in names:
                names.append(name)
    return names


def _coordinate_column(report: Dict[str, Any], rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    variable_names = _case_variable_names(report)
    candidates = []
    for name in variable_names:
        values = [_numeric_value(row.get(name)) for row in rows]
        finite = [value for value in values if value is not None]
        if len(finite) >= _MINIMUM_SAMPLE_COUNT and len({round(value, 12) for value in finite}) >= _MINIMUM_SAMPLE_COUNT:
            candidates.append(name)
    return candidates[0] if len(candidates) == 1 else None


def _energy_column(rows: Sequence[Dict[str, Any]]) -> Optional[str]:
    for column in _ENERGY_COLUMNS:
        if sum(_numeric_value(row.get(column)) is not None for row in rows) >= _MINIMUM_SAMPLE_COUNT:
            return column
    return None


def _method_level(method: str) -> int:
    return _MOLECULAR_METHOD_LEVELS.get(str(method or '').strip().lower(), 0)


def _sample_rows(
    rows: Sequence[Dict[str, Any]],
    coordinate: str,
    energy: str,
) -> List[Dict[str, Any]]:
    samples = []
    for row in rows:
        if not isinstance(row, dict):
            continue
        if str(row.get('status') or '').strip().lower() != 'succeeded':
            continue
        x_value = _numeric_value(row.get(coordinate))
        energy_value = _numeric_value(row.get(energy))
        case_id = str(row.get('case_id') or '').strip()
        if x_value is None or energy_value is None or not case_id:
            continue
        samples.append({
            'case_id': case_id,
            'label': row.get('label') or case_id,
            'x': x_value,
            'energy': energy_value,
            'method': _normalized_method(row),
            'path_restart_validation': row.get('path_restart_validation'),
            'path_restart_anomaly_id': row.get('path_restart_anomaly_id'),
            'path_restart_anomaly_kind': row.get('path_restart_anomaly_kind'),
            'path_restart_endpoint_side': row.get('path_restart_endpoint_side'),
        })
    samples.sort(key=lambda item: (item['x'], item['case_id']))
    unique = []
    for sample in samples:
        if unique and abs(sample['x'] - unique[-1]['x']) < 1e-12:
            continue
        unique.append(sample)
    return unique


def _validated_endpoint_case_ids(report: Dict[str, Any]) -> set[str]:
    """Return endpoint-window cases already resolved by bidirectional restart."""
    adaptive = report.get('adaptive') if isinstance(report.get('adaptive'), dict) else {}
    validation = (
        adaptive.get('path_window_restart_validation')
        if isinstance(adaptive.get('path_window_restart_validation'), dict)
        else report.get('path_window_restart_validation')
    )
    if not isinstance(validation, dict) or validation.get('status') != 'validated':
        return set()
    accepted = {
        str(case_id)
        for case_id in validation.get('accepted_case_ids') or []
        if str(case_id).strip()
    }
    if not accepted:
        return set()
    decisions = (
        adaptive.get('path_window_restart_decisions')
        if isinstance(adaptive.get('path_window_restart_decisions'), list)
        else []
    )
    endpoint_case_ids = set()
    for decision in decisions:
        if not isinstance(decision, dict):
            continue
        case_id = str(decision.get('case_id') or '')
        if (
            case_id in accepted
            and decision.get('path_anomaly_kind') == 'endpoint_continuity_risk'
        ):
            endpoint_case_ids.add(case_id)
    return endpoint_case_ids


def _endpoint_window_is_validated(
    target_samples: Sequence[Dict[str, Any]],
    validated_case_ids: set[str],
) -> bool:
    target_case_ids = {str(sample.get('case_id') or '') for sample in target_samples}
    if target_case_ids and target_case_ids <= validated_case_ids:
        return True
    return bool(target_samples) and all(
        sample.get('path_restart_validation') == 'bidirectional_validated'
        and sample.get('path_restart_anomaly_kind') == 'endpoint_continuity_risk'
        for sample in target_samples
    )


def _segment_slope(left: Dict[str, Any], right: Dict[str, Any]) -> Optional[float]:
    spacing = right['x'] - left['x']
    if abs(spacing) < 1e-12:
        return None
    return (right['energy'] - left['energy']) / spacing


def _transition_target_indices(samples: Sequence[Dict[str, Any]], boundary: int) -> List[int]:
    """Choose the lower-level side of a method boundary plus one buffer point."""
    left_method = samples[boundary]['method']
    right_method = samples[boundary + 1]['method']
    if _method_level(right_method) > _method_level(left_method):
        return list(range(max(0, boundary - 1), boundary + 1))
    if _method_level(left_method) > _method_level(right_method):
        return list(range(boundary + 1, min(len(samples), boundary + 3)))
    return [boundary, boundary + 1]


def _endpoint_anomaly(
    samples: Sequence[Dict[str, Any]],
    *,
    side: str,
    coordinate: str,
    energy: str,
    anomaly_index: int,
) -> Optional[Dict[str, Any]]:
    """Flag a one-sided endpoint that should be checked from both directions."""
    if len(samples) < _MINIMUM_SAMPLE_COUNT:
        return None
    if side == 'left':
        endpoint = samples[0]
        neighbor = samples[1]
        endpoint_slope = _segment_slope(samples[0], samples[1])
        comparison_slopes = (
            _segment_slope(samples[1], samples[2]),
            _segment_slope(samples[2], samples[3]),
        )
        target_samples = list(samples[:2])
    else:
        endpoint = samples[-1]
        neighbor = samples[-2]
        endpoint_slope = _segment_slope(samples[-2], samples[-1])
        comparison_slopes = (
            _segment_slope(samples[-3], samples[-2]),
            _segment_slope(samples[-4], samples[-3]),
        )
        target_samples = list(samples[-2:])
    if endpoint_slope is None or any(value is None for value in comparison_slopes):
        return None
    reference_slope = sum(float(value) for value in comparison_slopes) / len(comparison_slopes)
    mismatch = abs(endpoint_slope - reference_slope)
    local_scale = max(median(abs(float(value)) for value in comparison_slopes), 1e-3)
    mismatch_ratio = mismatch / local_scale
    spacing = abs(endpoint['x'] - neighbor['x'])
    energy_residual = mismatch * spacing
    if mismatch_ratio < _SLOPE_MISMATCH_RATIO or energy_residual < _ENERGY_RESIDUAL_HARTREE:
        return None
    transition = {
        'left_case_id': target_samples[0]['case_id'],
        'right_case_id': target_samples[-1]['case_id'],
        'left_method': target_samples[0]['method'],
        'right_method': target_samples[-1]['method'],
        'coordinate_value': endpoint['x'],
        'left_coordinate_value': target_samples[0]['x'],
        'right_coordinate_value': target_samples[-1]['x'],
        'endpoint_side': side,
        'endpoint_case_id': endpoint['case_id'],
        'endpoint_slope': endpoint_slope,
        'interior_reference_slope': reference_slope,
        'slope_mismatch_ratio': mismatch_ratio,
        'energy_residual': energy_residual,
    }
    return {
        'id': 'path-anomaly-{0:03d}'.format(anomaly_index),
        'kind': 'endpoint_continuity_risk',
        'severity': 'strong' if mismatch_ratio >= _STRONG_SLOPE_MISMATCH_RATIO else 'moderate',
        'coordinate': coordinate,
        'observable': energy,
        'endpoint_side': side,
        'transition': transition,
        'target_case_ids': [sample['case_id'] for sample in target_samples],
        'recommended_method': endpoint['method'],
        'reason': (
            'The {0} endpoint has only one-sided history and its first local slope differs '
            'from the next two segments by {1:.1f}x ({2:.3f} Ha over one step). This may be '
            'a physical edge feature or a branch-dependent solution. Recompute the local '
            'window by propagating compatible 1RDM states from both directions before '
            'changing the method.'
        ).format(side, mismatch_ratio, energy_residual),
    }


def analyze_scan_path(report: Dict[str, Any]) -> Dict[str, Any]:
    """Identify non-smooth transitions and one-sided endpoint risks in a scan.

    A large local slope mismatch alone can be a physical feature near a minimum.
    Internal anomalies therefore require a method transition. Endpoint anomalies
    are explicitly labeled as continuity risks and trigger bidirectional
    validation, not immediate method promotion. These are routing-quality signals,
    not phase classifiers.
    """
    payload = copy.deepcopy(report) if isinstance(report, dict) else {}
    rows = [row for row in payload.get('comparison_table') or [] if isinstance(row, dict)]
    coordinate = _coordinate_column(payload, rows)
    energy = _energy_column(rows)
    base = {
        'schema': PATH_DIAGNOSTICS_SCHEMA,
        'kind': 'scan_path_diagnostics',
        'status': 'not_applicable',
        'coordinate': coordinate,
        'observable': energy,
        'sample_count': 0,
        'method_transitions': [],
        'anomalies': [],
        'validated_findings': [],
        'summary': 'Scan-path continuity diagnostics require a completed one-dimensional scan with at least four energy points.',
    }
    if not coordinate or not energy:
        return base

    samples = _sample_rows(rows, coordinate, energy)
    base['sample_count'] = len(samples)
    if len(samples) < _MINIMUM_SAMPLE_COUNT:
        return base

    transitions = []
    anomalies = []
    validated_findings = []
    validated_endpoint_case_ids = _validated_endpoint_case_ids(payload)
    for boundary in range(1, len(samples) - 2):
        left = samples[boundary]
        right = samples[boundary + 1]
        if not left['method'] or not right['method'] or left['method'] == right['method']:
            continue
        previous_slope = _segment_slope(samples[boundary - 1], left)
        boundary_slope = _segment_slope(left, right)
        next_slope = _segment_slope(right, samples[boundary + 2])
        if previous_slope is None or boundary_slope is None or next_slope is None:
            continue
        reference_slope = 0.5 * (previous_slope + next_slope)
        mismatch = abs(boundary_slope - reference_slope)
        local_scale = max(median([abs(previous_slope), abs(next_slope)]), 1e-3)
        mismatch_ratio = mismatch / local_scale
        spacing = right['x'] - left['x']
        energy_residual = mismatch * abs(spacing)
        transition = {
            'left_case_id': left['case_id'],
            'right_case_id': right['case_id'],
            'left_method': left['method'],
            'right_method': right['method'],
            'coordinate_value': 0.5 * (left['x'] + right['x']),
            'left_coordinate_value': left['x'],
            'right_coordinate_value': right['x'],
            'slope_before': previous_slope,
            'slope_at_transition': boundary_slope,
            'slope_after': next_slope,
            'slope_mismatch_ratio': mismatch_ratio,
            'energy_residual': energy_residual,
        }
        transitions.append(transition)
        if mismatch_ratio < _SLOPE_MISMATCH_RATIO or energy_residual < _ENERGY_RESIDUAL_HARTREE:
            continue
        target_indices = _transition_target_indices(samples, boundary)
        target_cases = [samples[index]['case_id'] for index in target_indices]
        stronger_method = right['method'] if _method_level(right['method']) >= _method_level(left['method']) else left['method']
        anomaly = {
            'id': 'path-anomaly-{0:03d}'.format(len(anomalies) + 1),
            'kind': 'method_transition_discontinuity',
            'severity': 'strong' if mismatch_ratio >= _STRONG_SLOPE_MISMATCH_RATIO else 'moderate',
            'coordinate': coordinate,
            'observable': energy,
            'transition': transition,
            'target_case_ids': target_cases,
            'recommended_method': stronger_method,
            'reason': (
                'The {0}->{1} transition has a local slope mismatch of {2:.1f}x '
                'and an estimated discontinuity of {3:.3f} Ha. First retry the lower-level '
                'side from compatible left and right reference densities; if the branches do not '
                'establish a smooth result, recompute the local overlap window with {4} before '
                'interpreting this scan segment.'
            ).format(
                left['method'].upper(),
                right['method'].upper(),
                mismatch_ratio,
                energy_residual,
                stronger_method.upper(),
            ),
        }
        anomalies.append(anomaly)

    for side in ('left', 'right'):
        endpoint = _endpoint_anomaly(
            samples,
            side=side,
            coordinate=coordinate,
            energy=energy,
            anomaly_index=len(anomalies) + 1,
        )
        if endpoint:
            target_case_ids = {
                str(case_id)
                for case_id in endpoint.get('target_case_ids') or []
                if str(case_id).strip()
            }
            target_samples = [
                sample for sample in samples
                if sample.get('case_id') in target_case_ids
            ]
            if _endpoint_window_is_validated(target_samples, validated_endpoint_case_ids):
                validated_findings.append({
                    **endpoint,
                    'status': 'validated_physical_endpoint',
                    'reason': (
                        'The endpoint remains steep, but independent left-to-right and '
                        'right-to-left SCF-reference 1RDM continuations converged to the '
                        'same result. The one-sided feature is treated as validated rather '
                        'than queued for another restart or method promotion.'
                    ),
                })
            else:
                anomalies.append(endpoint)

    base['status'] = 'review_required' if anomalies else 'completed'
    base['method_transitions'] = transitions
    base['anomalies'] = anomalies
    base['validated_findings'] = validated_findings
    if anomalies:
        endpoint_count = sum(
            1 for item in anomalies
            if item.get('kind') == 'endpoint_continuity_risk'
        )
        transition_count = len(anomalies) - endpoint_count
        findings = []
        if transition_count:
            findings.append('{0} non-smooth method transition{1}'.format(
                transition_count,
                '' if transition_count == 1 else 's',
            ))
        if endpoint_count:
            findings.append('{0} one-sided endpoint continuity risk{1}'.format(
                endpoint_count,
                '' if endpoint_count == 1 else 's',
            ))
        base['summary'] = (
            'Detected {0}; review the affected points '
            'with a bidirectional continuation retry, then local method promotion only if needed.'
        ).format(' and '.join(findings))
    elif validated_findings:
        base['summary'] = (
            'No actionable scan-path anomaly remains. {0} endpoint continuity '
            'risk{1} passed bidirectional 1RDM validation.'
        ).format(
            len(validated_findings),
            '' if len(validated_findings) == 1 else 's',
        )
    else:
        base['summary'] = 'No method-transition discontinuity or endpoint continuity risk exceeded the scan-path review threshold.'
    return base


def annotate_path_diagnostics(report: Dict[str, Any], diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    """Attach compact path-review labels to comparison-table rows for the UI."""
    payload = copy.deepcopy(report)
    rows = payload.get('comparison_table') if isinstance(payload.get('comparison_table'), list) else []
    annotations: Dict[str, List[str]] = {}
    for anomaly in diagnostics.get('anomalies') or []:
        if not isinstance(anomaly, dict):
            continue
        for case_id in anomaly.get('target_case_ids') or []:
            annotations.setdefault(str(case_id), []).append(str(anomaly.get('reason') or 'Scan-path continuity review required.'))
    for row in rows:
        if not isinstance(row, dict):
            continue
        row.pop('path_review', None)
        row.pop('path_recommendation', None)
        messages = annotations.get(str(row.get('case_id') or ''))
        if not messages:
            continue
        row['path_review'] = 'required'
        row['path_recommendation'] = ' '.join(messages)
    payload['scan_path_diagnostics'] = copy.deepcopy(diagnostics)
    return payload


__all__ = [
    'PATH_DIAGNOSTICS_SCHEMA',
    'analyze_scan_path',
    'annotate_path_diagnostics',
    'deferred_scan_path_diagnostics',
]
