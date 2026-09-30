from __future__ import annotations

import copy
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple


ESSENTIAL_NATURAL_OCCUPATION_FRACTIONALITY = 0.10
ESSENTIAL_T2_AMPLITUDE = 0.10
CRITICAL_NATURAL_OCCUPATION_FRACTIONALITY = 0.25
CRITICAL_T2_AMPLITUDE = 0.20
CHEMICAL_BASELINE_COVERAGE_TARGET = 0.90

_REVIEWABLE_CANDIDATE_STATUSES = frozenset(('available', 'mapping_ambiguous'))
_CANDIDATE_ROLE_ORDER = {
    'manual': 0,
    'baseline': 1,
    'diagnostic': 2,
    'expanded': 3,
    'broad': 4,
}


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _nelecas_sector(nelecas: Any, ncas: Any, spin: int) -> Optional[Tuple[int, int]]:
    try:
        orbitals = int(ncas)
    except (TypeError, ValueError):
        return None
    if isinstance(nelecas, (list, tuple)) and len(nelecas) == 2:
        nalpha, nbeta = int(nelecas[0]), int(nelecas[1])
    else:
        try:
            total = int(nelecas)
        except (TypeError, ValueError):
            return None
        if (total + int(spin)) % 2:
            return None
        nalpha = (total + int(spin)) // 2
        nbeta = total - nalpha
    if min(nalpha, nbeta) < 0 or max(nalpha, nbeta) > orbitals:
        return None
    return nalpha, nbeta


def build_active_space_selection_evidence(
    natural_summary: Dict[str, Any],
    natural_occupation_quality: Dict[str, Any],
    t2_importance: Dict[int, float],
    t2_supplement: Dict[str, Any],
) -> Dict[str, Any]:
    """Build high-confidence static-correlation evidence for candidate ranking.

    The broad occupation/T2 union remains available as a diagnostic candidate.
    Recommendation uses only clearly fractional occupations and large T2 signals,
    so weak threshold-edge evidence cannot accumulate into the default CAS.
    """

    signals: List[Dict[str, Any]] = []
    if (
        natural_summary.get('status') == 'available'
        and natural_occupation_quality.get('status') == 'physical'
    ):
        for orbital in natural_summary.get('orbitals') or []:
            occupation = _as_float(orbital.get('occupation'))
            canonical_index = orbital.get('dominant_canonical_mo')
            if occupation is None or canonical_index is None:
                continue
            fractionality = max(0.0, min(occupation, 2.0 - occupation))
            if fractionality < ESSENTIAL_NATURAL_OCCUPATION_FRACTIONALITY:
                continue
            signals.append({
                'source': 'natural_occupation',
                'orbital_index': int(canonical_index),
                'value': occupation,
                'weight': fractionality,
                'reason': 'essential natural-orbital fractionality={0:.6g}'.format(fractionality),
            })

    accepted_t2 = {
        int(index)
        for index in t2_supplement.get('accepted_indices') or []
    }
    for index, amplitude in sorted((t2_importance or {}).items()):
        value = _as_float(amplitude)
        if int(index) not in accepted_t2 or value is None or value < ESSENTIAL_T2_AMPLITUDE:
            continue
        signals.append({
            'source': 't2_amplitude',
            'orbital_index': int(index),
            'value': value,
            'weight': min(value, 1.0),
            'reason': 'essential max|t2|={0:.6g}'.format(value),
        })

    total_weight = sum(float(item['weight']) for item in signals)
    return {
        'schema': 'pyscf-agent.active-space-selection-evidence.v1',
        'natural_occupation_fractionality_threshold': ESSENTIAL_NATURAL_OCCUPATION_FRACTIONALITY,
        't2_amplitude_threshold': ESSENTIAL_T2_AMPLITUDE,
        'signals': signals,
        'essential_orbital_indices': sorted({int(item['orbital_index']) for item in signals}),
        'total_weight': total_weight,
    }


def _candidate_coverage(candidate: Dict[str, Any], evidence: Dict[str, Any]) -> Dict[str, Any]:
    indices = {int(index) for index in candidate.get('orbital_indices') or []}
    signals = evidence.get('signals') or []
    total_weight = float(evidence.get('total_weight') or 0.0)
    captured = [item for item in signals if int(item['orbital_index']) in indices]
    missing = [item for item in signals if int(item['orbital_index']) not in indices]
    captured_weight = sum(float(item.get('weight') or 0.0) for item in captured)
    coverage = captured_weight / total_weight if total_weight > 0.0 else 1.0
    by_source: Dict[str, Dict[str, float]] = {}
    for item in signals:
        source = str(item.get('source') or 'unknown')
        source_payload = by_source.setdefault(source, {'total_weight': 0.0, 'captured_weight': 0.0})
        weight = float(item.get('weight') or 0.0)
        source_payload['total_weight'] += weight
        if int(item['orbital_index']) in indices:
            source_payload['captured_weight'] += weight
    for source_payload in by_source.values():
        source_total = source_payload['total_weight']
        source_payload['coverage'] = (
            source_payload['captured_weight'] / source_total if source_total > 0.0 else 1.0
        )
    return {
        'coverage': coverage,
        'captured_weight': captured_weight,
        'total_weight': total_weight,
        'captured_signal_count': len(captured),
        'total_signal_count': len(signals),
        'missing_orbital_indices': sorted({int(item['orbital_index']) for item in missing}),
        'missing_signals': copy.deepcopy(missing),
        'by_source': by_source,
    }


def _critical_missing_signals(candidate: Dict[str, Any]) -> List[Dict[str, Any]]:
    coverage = ((candidate.get('evaluation') or {}).get('evidence_coverage') or {})
    critical = []
    for signal in coverage.get('missing_signals') or []:
        source = signal.get('source')
        weight = float(signal.get('weight') or 0.0)
        if source == 'natural_occupation' and weight >= CRITICAL_NATURAL_OCCUPATION_FRACTIONALITY:
            critical.append(copy.deepcopy(signal))
        elif source == 't2_amplitude' and weight >= CRITICAL_T2_AMPLITUDE:
            critical.append(copy.deepcopy(signal))
    return critical


def _exact_sector_dimension(candidate: Dict[str, Any], spin: int) -> Optional[int]:
    sector = _nelecas_sector(candidate.get('estimated_nelecas'), candidate.get('ncas'), spin)
    if sector is None:
        return None
    nalpha, nbeta = sector
    ncas = int(candidate['ncas'])
    return math.comb(ncas, nalpha) * math.comb(ncas, nbeta)


def _candidate_sort_key(candidate: Dict[str, Any]) -> Tuple[int, float, int, str]:
    status = str(candidate.get('status') or '')
    try:
        ncas = float(int(candidate.get('ncas')))
    except (TypeError, ValueError):
        ncas = math.inf
    return (
        0 if status in _REVIEWABLE_CANDIDATE_STATUSES else 1,
        ncas,
        _CANDIDATE_ROLE_ORDER.get(str(candidate.get('role') or ''), 99),
        str(candidate.get('method') or ''),
    )


def evaluate_active_space_candidates(
    candidates: Sequence[Dict[str, Any]],
    evidence: Dict[str, Any],
    *,
    chemical_baseline_indices: Sequence[int],
    spin: int,
    target_solver: Optional[str],
) -> List[Dict[str, Any]]:
    baseline = {int(index) for index in chemical_baseline_indices}
    evaluated = []
    for raw_candidate in candidates:
        candidate = copy.deepcopy(raw_candidate)
        indices = {int(index) for index in candidate.get('orbital_indices') or []}
        coverage = _candidate_coverage(candidate, evidence)
        sector_dimension = _exact_sector_dimension(candidate, spin)
        risk_flags = []
        if candidate.get('status') == 'mapping_ambiguous':
            risk_flags.append('orbital_mapping_ambiguous')
        if candidate.get('ncas') is not None and candidate.get('estimated_nelecas') is None:
            risk_flags.append('active_electron_count_unavailable')
        if baseline and not baseline.issubset(indices):
            risk_flags.append('chemical_valence_manifold_incomplete')
        solver = str(target_solver or '').strip().lower().replace('-', '_')
        if sector_dimension is None:
            route = 'review_required'
        elif sector_dimension <= 4096:
            route = 'small_exact_sector'
        elif solver in ('block2', 'block2_dmrg', 'dmrg'):
            route = 'dmrg'
        else:
            route = 'large_active_space'
            risk_flags.append('large_exact_sector')
        candidate['evaluation'] = {
            'evidence_coverage': coverage,
            'chemical_valence_complete': bool(not baseline or baseline.issubset(indices)),
            'size_growth_from_chemical_baseline': max(0, len(indices) - len(baseline)) if baseline else None,
            'exact_sector_dimension': sector_dimension,
            'solver_route': route,
            'risk_flags': risk_flags,
        }
        candidate['recommended'] = False
        evaluated.append(candidate)
    return evaluated


def recommend_active_space_candidate(
    candidates: Sequence[Dict[str, Any]],
    *,
    selection_method: str,
    recovery_candidate_method: Optional[str],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    evaluated = [copy.deepcopy(candidate) for candidate in candidates]
    by_method = {str(candidate.get('method')): candidate for candidate in evaluated}

    selected_method = 'unresolved'
    reason = 'No executable active-space candidate is available.'
    confidence = 'low'
    expansion_method: Optional[str] = None
    expansion_reason: Optional[str] = None
    critical_missing: List[Dict[str, Any]] = []
    if selection_method == 'manual' and by_method.get('manual', {}).get('status') == 'available':
        selected_method = 'manual'
        reason = 'The explicit user-supplied active-space contract takes precedence.'
        confidence = 'user_approved_input'
    elif selection_method == 'avas' and by_method.get('avas', {}).get('status') in ('available', 'mapping_ambiguous'):
        selected_method = 'avas'
        reason = 'The explicitly requested AVAS projection defines the review candidate.'
        confidence = 'high' if by_method['avas'].get('status') == 'available' else 'moderate'
    elif selection_method == 'uno' and by_method.get('uno', {}).get('status') == 'available':
        selected_method = 'uno'
        reason = 'The requested mean-field occupation window selects genuine UHF natural orbitals; their full orbital matrix is retained for review.'
        confidence = 'moderate'
    elif recovery_candidate_method == 'chemical_valence' and by_method.get('chemical_valence', {}).get('status') in ('available', 'mapping_ambiguous'):
        selected_method = 'chemical_valence'
        reason = 'The diagnostic probe is unusable, so the complete chemical-valence candidate is retained for review.'
        confidence = 'moderate'
    else:
        baseline = by_method.get('chemical_valence') or {}
        expanded = by_method.get('evidence_expanded') or {}
        baseline_available = baseline.get('status') in ('available', 'mapping_ambiguous')
        expanded_available = expanded.get('status') == 'available'
        baseline_coverage = float(
            ((baseline.get('evaluation') or {}).get('evidence_coverage') or {}).get('coverage') or 0.0
        )
        critical_missing = _critical_missing_signals(baseline)
        if baseline_available and critical_missing and expanded_available:
            selected_method = 'chemical_valence'
            expansion_method = 'evidence_expanded'
            expansion_coverage = float(
                ((expanded.get('evaluation') or {}).get('evidence_coverage') or {}).get('coverage') or 0.0
            )
            reason = (
                'The complete chemical-valence manifold is the smallest physically resolved '
                'candidate and is selected first. It captures {0:.1%} of weighted evidence.'
            ).format(baseline_coverage)
            expansion_reason = (
                'If the baseline calculation remains unreliable, expand to the next candidate; '
                'it adds {0} critical occupation/T2 signal(s) and captures {1:.1%} of weighted evidence.'
            ).format(len(critical_missing), expansion_coverage)
            confidence = 'moderate'
        elif baseline_available and (
            baseline_coverage >= CHEMICAL_BASELINE_COVERAGE_TARGET
            or not critical_missing
        ):
            selected_method = 'chemical_valence'
            reason = (
                'The complete chemical-valence manifold captures {0:.1%} of high-confidence '
                'occupation/T2 evidence and omits no critical signal; broader threshold-edge '
                'orbitals remain diagnostic alternatives.'
            ).format(baseline_coverage)
            confidence = 'high' if baseline.get('status') == 'available' else 'moderate'
        elif baseline_available:
            selected_method = 'chemical_valence'
            reason = (
                'The complete chemical-valence manifold is the only chemically resolved executable '
                'candidate; missing evidence is retained for manual review rather than silently truncating orbitals.'
            )
            confidence = 'moderate'
        elif by_method.get('merged', {}).get('status') == 'available':
            selected_method = 'merged'
            reason = 'No chemically resolved baseline is available; the evidence union is retained for explicit review.'
            confidence = 'low'

    selected = by_method.get(selected_method)
    if selected is not None:
        selected['recommended'] = True
        selected['recommendation_stage'] = 'initial'
        selected['recommendation_reason'] = reason
    expansion = by_method.get(expansion_method) if expansion_method else None
    if expansion is not None:
        expansion['recommendation_stage'] = 'next_expansion'
        expansion['expansion_candidate'] = True
        expansion['recommendation_reason'] = expansion_reason
    for candidate in evaluated:
        candidate.setdefault('recommendation_stage', 'alternative')
        candidate.setdefault('expansion_candidate', False)
    evaluated.sort(key=_candidate_sort_key)
    decision = {
        'schema': 'pyscf-agent.active-space-candidate-decision.v1',
        'selection_policy': 'uhf_natural_occupation_window' if selected_method == 'uno' else 'smallest_chemically_complete_first',
        'selected_candidate_method': selected_method,
        'expansion_candidate_method': expansion_method,
        'confidence': confidence,
        'reason': reason,
        'expansion_reason': expansion_reason,
        'critical_missing_orbital_indices': sorted({
            int(signal['orbital_index'])
            for signal in critical_missing
            if signal.get('orbital_index') is not None
        }),
        'chemical_baseline_coverage_target': CHEMICAL_BASELINE_COVERAGE_TARGET,
        'critical_natural_occupation_fractionality': CRITICAL_NATURAL_OCCUPATION_FRACTIONALITY,
        'critical_t2_amplitude': CRITICAL_T2_AMPLITUDE,
        'alternatives': [
            candidate.get('method')
            for candidate in evaluated
            if candidate.get('method') != selected_method
            and candidate.get('status') in _REVIEWABLE_CANDIDATE_STATUSES
        ],
    }
    return evaluated, decision


__all__ = [
    'build_active_space_selection_evidence',
    'evaluate_active_space_candidates',
    'recommend_active_space_candidate',
]
