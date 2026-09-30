from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple


DMRG_STATE_TRACKING_SCHEMA = 'pyscf-agent.dmrg-state-tracking.v1'


def _solver_name(request: Dict[str, Any]) -> str:
    solver = request.get('solver') if isinstance(request, dict) else None
    if isinstance(solver, dict):
        solver = solver.get('name')
    value = str(solver or '').strip().lower().replace('-', '_')
    return 'block2_dmrg' if value in ('block2', 'dmrg') else value


def _task_report(case_record: Dict[str, Any]) -> Dict[str, Any]:
    value = case_record.get('task_report') if isinstance(case_record, dict) else None
    return value if isinstance(value, dict) else {}


def _dmrg_result(case_record: Dict[str, Any]) -> Dict[str, Any]:
    structured = _task_report(case_record).get('structured_results')
    structured = structured if isinstance(structured, dict) else {}
    direct = structured.get('dmrg_result')
    if isinstance(direct, dict):
        return direct
    cas_result = structured.get('cas_result')
    cas_result = cas_result if isinstance(cas_result, dict) else {}
    nested = cas_result.get('dmrg_result')
    return nested if isinstance(nested, dict) else {}


def _status(case_record: Dict[str, Any]) -> str:
    return str(_task_report(case_record).get('execution_status') or '').strip().lower()


def _artifact_path(case_record: Dict[str, Any], kind: str) -> Optional[Path]:
    task_report = _task_report(case_record)
    for artifact in task_report.get('artifacts') or ():
        if not isinstance(artifact, dict) or artifact.get('kind') != kind:
            continue
        raw_path = artifact.get('path')
        if not isinstance(raw_path, str) or not raw_path.strip():
            continue
        path = Path(raw_path).expanduser()
        if not path.is_absolute():
            work_dir = task_report.get('work_dir')
            if isinstance(work_dir, str) and work_dir.strip():
                path = Path(work_dir).expanduser() / path
        if path.is_file():
            return path.resolve()
    return None


def _root_rdms(case_record: Dict[str, Any]) -> Optional[List[Any]]:
    path = _artifact_path(case_record, 'block2_dmrg_arrays')
    if path is None:
        return None
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        with np.load(path, allow_pickle=False) as archive:
            if 'root_rdm1' in archive:
                return [np.asarray(value, dtype=float) for value in archive['root_rdm1']]
            if 'rdm1' in archive:
                return [np.asarray(archive['rdm1'], dtype=float)]
    except Exception:
        return None
    return None


def _root_signatures(case_record: Dict[str, Any]) -> List[Dict[str, Any]]:
    result = _dmrg_result(case_record)
    signatures = result.get('root_signatures')
    if isinstance(signatures, list) and signatures:
        return [copy.deepcopy(item) for item in signatures if isinstance(item, dict)]
    energies = result.get('state_energies')
    energies = energies if isinstance(energies, list) else []
    excitations = result.get('excitation_energies')
    excitations = excitations if isinstance(excitations, list) else []
    occupations = result.get('root_natural_occupations')
    occupations = occupations if isinstance(occupations, list) else []
    return [
        {
            'root': root,
            'energy': float(energy),
            'excitation_energy': (
                float(excitations[root])
                if root < len(excitations)
                else float(energy - energies[0])
            ),
            'natural_occupations': (
                [float(value) for value in occupations[root]]
                if root < len(occupations) and isinstance(occupations[root], list)
                else None
            ),
        }
        for root, energy in enumerate(energies)
    ]


def _sector(case_record: Dict[str, Any]) -> Dict[str, Any]:
    result = _dmrg_result(case_record)
    value = result.get('state_sector')
    return copy.deepcopy(value) if isinstance(value, dict) else {}


def _sector_compatible(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    keys = ('n_orbitals', 'n_electrons', 'total_electrons', 'spin', 'symmetry')
    return all(
        key not in left
        or key not in right
        or left.get(key) == right.get(key)
        for key in keys
    )


def _occupation_similarity(left: Any, right: Any) -> Optional[float]:
    if not isinstance(left, list) or not isinstance(right, list) or len(left) != len(right) or not left:
        return None
    rms = math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(left, right)) / len(left))
    return max(0.0, 1.0 - min(1.0, rms / 0.5))


def _rdm_similarity(left: Any, right: Any) -> Optional[float]:
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        left_array = np.asarray(left, dtype=float)
        right_array = np.asarray(right, dtype=float)
        if left_array.shape != right_array.shape or left_array.size == 0:
            return None
        denominator = float(np.linalg.norm(left_array) * np.linalg.norm(right_array))
        if denominator <= 0.0:
            return None
        return max(0.0, min(1.0, float(np.vdot(left_array, right_array).real) / denominator))
    except Exception:
        return None


def _energy_similarity(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_value = float(left.get('excitation_energy') or 0.0)
    right_value = float(right.get('excitation_energy') or 0.0)
    scale = max(abs(left_value), abs(right_value), 0.05)
    return 1.0 / (1.0 + abs(left_value - right_value) / scale)


def _pair_score(
    left: Dict[str, Any],
    right: Dict[str, Any],
    left_rdm: Any = None,
    right_rdm: Any = None,
) -> Tuple[float, Dict[str, Any]]:
    occupation = _occupation_similarity(
        left.get('natural_occupations'),
        right.get('natural_occupations'),
    )
    rdm = _rdm_similarity(left_rdm, right_rdm)
    energy = _energy_similarity(left, right)
    if rdm is not None:
        score = 0.70 * rdm + 0.20 * (occupation if occupation is not None else rdm) + 0.10 * energy
        evidence = 'one_particle_density_matrix'
    elif occupation is not None:
        score = 0.85 * occupation + 0.15 * energy
        evidence = 'natural_occupation_spectrum'
    else:
        score = 0.35 * energy
        evidence = 'energy_only'
    return score, {
        'score': score,
        'rdm1_similarity': rdm,
        'occupation_similarity': occupation,
        'excitation_energy_similarity': energy,
        'evidence': evidence,
    }


def _assign_roots(scores: Sequence[Sequence[float]]) -> Tuple[List[int], float, Optional[float]]:
    size = len(scores)
    if size == 0:
        return [], 0.0, None
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    matrix = np.asarray(scores, dtype=float)
    if matrix.shape != (size, size) or not np.all(np.isfinite(matrix)):
        raise ValueError('Root similarity scores must form a finite square matrix')
    rows, columns = linear_sum_assignment(-matrix)
    best = float(matrix[rows, columns].sum())
    # Every different assignment omits at least one edge of the optimum.
    # Excluding each chosen edge therefore finds the exact runner-up.
    second = None
    if size > 1:
        alternatives = []
        for row, column in zip(rows, columns):
            cost = -matrix.copy()
            cost[row, column] = np.inf
            other_rows, other_columns = linear_sum_assignment(cost)
            alternatives.append(float(matrix[other_rows, other_columns].sum()))
        second = max(alternatives)
    return columns.tolist(), best / size, None if second is None else max(0.0, (best - second) / size)


def _case_identifier(case_record: Dict[str, Any]) -> str:
    return str(case_record.get('case_id') or '')


def analyze_dmrg_state_tracking(report: Dict[str, Any]) -> Dict[str, Any]:
    """Track targeted DMRG roots across the ordered cases of one study."""

    cases = [
        item for item in report.get('cases') or []
        if isinstance(item, dict)
        and _status(item) == 'succeeded'
        and _solver_name(item.get('request') or {}) == 'block2_dmrg'
        and _root_signatures(item)
    ]
    payload: Dict[str, Any] = {
        'schema': DMRG_STATE_TRACKING_SCHEMA,
        'status': 'not_applicable',
        'case_ids': [_case_identifier(item) for item in cases],
        'links': [],
        'tracks': [],
        'ambiguous_case_ids': [],
        'summary': 'At least two successful block2 cases are required for cross-task state tracking.',
    }
    if len(cases) < 2:
        return payload

    track_by_case: Dict[str, Dict[int, str]] = {}
    first_signatures = _root_signatures(cases[0])
    track_by_case[_case_identifier(cases[0])] = {
        root: 'state-track-{0:03d}'.format(root)
        for root in range(len(first_signatures))
    }
    ambiguous = set()
    links = []
    for left_case, right_case in zip(cases, cases[1:]):
        left_id = _case_identifier(left_case)
        right_id = _case_identifier(right_case)
        left_signatures = _root_signatures(left_case)
        right_signatures = _root_signatures(right_case)
        left_sector = _sector(left_case)
        right_sector = _sector(right_case)
        link: Dict[str, Any] = {
            'source_case_id': left_id,
            'target_case_id': right_id,
            'source_variables': copy.deepcopy(left_case.get('variables') or {}),
            'target_variables': copy.deepcopy(right_case.get('variables') or {}),
            'sector_compatible': _sector_compatible(left_sector, right_sector),
            'mapping': [],
        }
        if not link['sector_compatible'] or len(left_signatures) != len(right_signatures):
            track_by_case[right_id] = {
                root: 'state-track-{0}-{1:03d}'.format(right_id, root)
                for root in range(len(right_signatures))
            }
            link.update({
                'status': 'incompatible',
                'confidence': 0.0,
                'reason': 'The targeted root count or quantum-number sector changed.',
            })
            ambiguous.update((left_id, right_id))
            links.append(link)
            continue

        left_rdms = _root_rdms(left_case)
        right_rdms = _root_rdms(right_case)
        matrix = []
        evidence = []
        for source_root, source_signature in enumerate(left_signatures):
            score_row = []
            evidence_row = []
            for target_root, target_signature in enumerate(right_signatures):
                score, detail = _pair_score(
                    source_signature,
                    target_signature,
                    left_rdms[source_root] if left_rdms and source_root < len(left_rdms) else None,
                    right_rdms[target_root] if right_rdms and target_root < len(right_rdms) else None,
                )
                score_row.append(score)
                evidence_row.append(detail)
            matrix.append(score_row)
            evidence.append(evidence_row)
        assignment, confidence, margin = _assign_roots(matrix)
        source_tracks = track_by_case.get(left_id, {})
        target_tracks: Dict[int, str] = {}
        mappings = []
        pair_scores = []
        for source_root, target_root in enumerate(assignment):
            detail = copy.deepcopy(evidence[source_root][target_root])
            pair_scores.append(float(detail['score']))
            track_id = source_tracks.get(
                source_root,
                'state-track-{0:03d}'.format(source_root),
            )
            target_tracks[target_root] = track_id
            mappings.append({
                'source_root': source_root,
                'target_root': target_root,
                'track_id': track_id,
                **detail,
            })
        track_by_case[right_id] = target_tracks
        ambiguous_link = (
            confidence < 0.72
            or min(pair_scores or [0.0]) < 0.65
            or (margin is not None and margin < 0.04 and len(assignment) > 1)
        )
        if ambiguous_link:
            ambiguous.update((left_id, right_id))
        link.update({
            'status': 'ambiguous' if ambiguous_link else 'tracked',
            'confidence': confidence,
            'assignment_margin': margin,
            'root_order_changed': any(source != target for source, target in enumerate(assignment)),
            'mapping': mappings,
        })
        links.append(link)

    tracks: Dict[str, List[Dict[str, Any]]] = {}
    for case_record in cases:
        case_id = _case_identifier(case_record)
        signatures = _root_signatures(case_record)
        for root, track_id in sorted(track_by_case.get(case_id, {}).items()):
            if root >= len(signatures):
                continue
            tracks.setdefault(track_id, []).append({
                'case_id': case_id,
                'root': root,
                'variables': copy.deepcopy(case_record.get('variables') or {}),
                'energy': signatures[root].get('energy'),
                'excitation_energy': signatures[root].get('excitation_energy'),
            })
    payload.update({
        'status': 'review_required' if ambiguous else 'tracked',
        'links': links,
        'tracks': [
            {'track_id': track_id, 'states': states}
            for track_id, states in sorted(tracks.items())
        ],
        'ambiguous_case_ids': sorted(ambiguous),
        'summary': (
            'Low-energy DMRG state identity is ambiguous near {0} case(s); exclude those cases as automatic MPS anchors.'.format(len(ambiguous))
            if ambiguous
            else 'Low-energy DMRG roots were tracked consistently across {0} successful cases.'.format(len(cases))
        ),
    })
    return payload


__all__ = [
    'DMRG_STATE_TRACKING_SCHEMA',
    'analyze_dmrg_state_tracking',
]
