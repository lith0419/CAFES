from __future__ import annotations

import copy
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple


STUDY_ACTIVE_SPACE_SCHEMA = 'pyscf-agent.study-active-space.v1'

_NON_EXECUTABLE_CANDIDATE_METHODS = {
    'frontier_fallback',
    'unresolved',
}

_CANDIDATE_METHOD_PRIORITY = {
    'chemical_valence': 5,
    'avas': 4,
    'manual': 3,
    'uno': 2,
    'evidence_expanded': 1,
    'merged': 1,
    'unspecified': 0,
}


def _active_electron_signature(nelecas: Any) -> Optional[Tuple[int, int]]:
    if isinstance(nelecas, (list, tuple)):
        if len(nelecas) != 2:
            return None
        try:
            nalpha, nbeta = (int(value) for value in nelecas)
        except (TypeError, ValueError):
            return None
        return nalpha + nbeta, nalpha - nbeta
    try:
        return int(nelecas), 0
    except (TypeError, ValueError):
        return None


def _candidate_method(contract: Dict[str, Any]) -> str:
    provenance = contract.get('candidate_provenance')
    provenance = provenance if isinstance(provenance, dict) else {}
    audit_summary = contract.get('audit_summary')
    audit_summary = audit_summary if isinstance(audit_summary, dict) else {}
    return str(
        provenance.get('method')
        or audit_summary.get('selected_candidate_method')
        or ('manual' if contract.get('selection_method') == 'manual' else None)
        or 'unspecified'
    ).strip().lower()


def _target_signature(contract: Dict[str, Any]) -> Tuple[str, ...]:
    provenance = contract.get('candidate_provenance')
    provenance = provenance if isinstance(provenance, dict) else {}
    target_summary = provenance.get('target_summary')
    target_summary = target_summary if isinstance(target_summary, dict) else {}
    primary_targets = target_summary.get('primary_targets')
    if isinstance(primary_targets, list) and primary_targets:
        return tuple(sorted(str(value).strip().lower() for value in primary_targets if str(value).strip()))
    resolved_targets = provenance.get('resolved_targets')
    if isinstance(resolved_targets, list) and resolved_targets:
        return tuple(sorted(str(value).strip().lower() for value in resolved_targets if str(value).strip()))
    return ()


def _expected_chemical_dimension(contract: Dict[str, Any]) -> Optional[Tuple[int, Tuple[int, int]]]:
    provenance = contract.get('candidate_provenance')
    provenance = provenance if isinstance(provenance, dict) else {}
    target_summary = provenance.get('target_summary')
    target_summary = target_summary if isinstance(target_summary, dict) else {}
    try:
        ncas = int(target_summary.get('primary_orbital_count'))
    except (TypeError, ValueError):
        return None
    electron_signature = _active_electron_signature(target_summary.get('primary_electron_count'))
    if ncas <= 0 or electron_signature is None:
        return None
    return ncas, electron_signature


def _contract_dimension(contract: Dict[str, Any]) -> Optional[Tuple[int, Tuple[int, int]]]:
    try:
        ncas = int(contract.get('ncas'))
    except (TypeError, ValueError):
        return None
    electron_signature = _active_electron_signature(contract.get('nelecas'))
    orbital_indices = contract.get('orbital_indices')
    if (
        ncas <= 0
        or electron_signature is None
        or electron_signature[0] <= 0
        or electron_signature[0] > 2 * ncas
        or not isinstance(orbital_indices, list)
        or len(orbital_indices) != ncas
    ):
        return None
    return ncas, electron_signature


def _physical_family_key(candidate: Dict[str, Any]) -> Tuple[str, Tuple[str, ...]]:
    """Group candidates by physical content, not by selection algorithm."""

    target_signature = candidate['target_signature']
    if target_signature:
        return 'chemical_target', target_signature
    return 'selection_method', (candidate['method'],)


def executable_active_space_contract(contract: Any) -> bool:
    if not isinstance(contract, dict) or _contract_dimension(contract) is None:
        return False
    return _candidate_method(contract) not in _NON_EXECUTABLE_CANDIDATE_METHODS


def _fixed_selection(contract: Any) -> bool:
    """An explicit manual or approved selection constrains study resolution.

    Automatic candidates are also materialized with selection_method=manual;
    their audit/provenance method takes precedence over that execution field.
    """
    return isinstance(contract, dict) and (
        contract.get('approved') is True or _candidate_method(contract) == 'manual'
    )


def _candidate_contracts(contract: Any) -> List[Dict[str, Any]]:
    if not isinstance(contract, dict):
        return []
    candidates = [copy.deepcopy(contract)] if executable_active_space_contract(contract) else []
    if _fixed_selection(contract):
        return candidates
    for option in contract.get('candidate_options') or []:
        if not isinstance(option, dict):
            continue
        option_contract = copy.deepcopy(contract)
        option_contract['ncas'] = option.get('ncas')
        option_contract['nelecas'] = copy.deepcopy(option.get('estimated_nelecas'))
        option_contract['orbital_indices'] = copy.deepcopy(option.get('orbital_indices'))
        option_contract['initial_mo_coeff'] = copy.deepcopy(option.get('initial_mo_coeff'))
        option_contract['candidate_provenance'] = {
            key: copy.deepcopy(value)
            for key, value in option.items()
            if key not in ('orbital_indices', 'initial_mo_coeff')
        }
        audit_summary = option_contract.get('audit_summary')
        if isinstance(audit_summary, dict):
            audit_summary['selected_candidate_method'] = option.get('method')
            audit_summary['selected_orbital_count'] = option.get('ncas')
            audit_summary['estimated_nelecas'] = copy.deepcopy(option.get('estimated_nelecas'))
        if executable_active_space_contract(option_contract):
            candidates.append(option_contract)
    return candidates


def resolve_study_active_space_contracts(
    decisions: List[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Choose one evidence-backed CAS definition for a molecular study.

    The study shares the chemical target and CAS dimensions.  Each case keeps
    its own canonical-orbital mapping and initial MO coefficients because those
    quantities vary along a geometry or parameter scan.
    """

    resolved = [copy.deepcopy(item) for item in decisions]
    candidates: List[Dict[str, Any]] = []
    for decision in resolved:
        for contract in _candidate_contracts(decision.get('active_space_contract')):
            dimension = _contract_dimension(contract)
            candidates.append({
                'case_id': str(decision.get('case_id') or ''),
                'contract': contract,
                'method': _candidate_method(contract),
                'target_signature': _target_signature(contract),
                'dimension': dimension,
                'expected_dimension': _expected_chemical_dimension(contract),
                'fixed_selection': _fixed_selection(decision.get('active_space_contract')),
            })

    if not candidates:
        policy = {
            'schema': STUDY_ACTIVE_SPACE_SCHEMA,
            'status': 'unavailable',
            'reason': 'No evidence-backed ActiveSpaceAudit candidate is available for this study.',
            'supporting_case_ids': [],
            'missing_case_ids': [str(item.get('case_id') or '') for item in resolved],
            'conflicting_case_ids': [],
        }
        return resolved, policy

    families: Dict[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]] = defaultdict(list)
    for candidate in candidates:
        families[_physical_family_key(candidate)].append(candidate)

    def family_rank(item: Tuple[Tuple[str, Tuple[str, ...]], List[Dict[str, Any]]]) -> Tuple[int, int, int, int, int]:
        (family_kind, family_signature), family_candidates = item
        return (
            sum(entry['fixed_selection'] for entry in family_candidates),
            len({entry['case_id'] for entry in family_candidates}),
            1 if family_kind == 'chemical_target' else 0,
            len(family_signature),
            max(_CANDIDATE_METHOD_PRIORITY.get(entry['method'], 0) for entry in family_candidates),
        )

    selected_family_key, selected_family = max(families.items(), key=family_rank)
    dimensions: Dict[Tuple[int, Tuple[int, int]], List[Dict[str, Any]]] = defaultdict(list)
    for candidate in selected_family:
        dimensions[candidate['dimension']].append(candidate)

    def dimension_rank(item: Tuple[Tuple[int, Tuple[int, int]], List[Dict[str, Any]]]) -> Tuple[int, int, int, int]:
        dimension, dimension_candidates = item
        chemical_matches = sum(
            1 for entry in dimension_candidates
            if entry.get('expected_dimension') == dimension
        )
        return (
            sum(entry['fixed_selection'] for entry in dimension_candidates),
            len({entry['case_id'] for entry in dimension_candidates}),
            chemical_matches,
            -dimension[0],
        )

    selected_dimension, supporting = max(dimensions.items(), key=dimension_rank)
    supporting = sorted(
        supporting,
        key=lambda entry: (entry['fixed_selection'], _CANDIDATE_METHOD_PRIORITY.get(entry['method'], 0)),
        reverse=True,
    )
    representative = supporting[0]['contract']
    selected_methods = sorted(
        {entry['method'] for entry in supporting},
        key=lambda method: _CANDIDATE_METHOD_PRIORITY.get(method, 0),
        reverse=True,
    )
    selected_method = supporting[0]['method']
    family_kind, family_signature = selected_family_key
    selected_targets = family_signature if family_kind == 'chemical_target' else ()
    supporting_case_ids = sorted({entry['case_id'] for entry in supporting if entry['case_id']})
    compatible_by_case: Dict[str, Dict[str, Any]] = {}
    for entry in candidates:
        case_id = entry['case_id']
        if (
            not case_id
            or _physical_family_key(entry) != selected_family_key
            or entry['dimension'] != selected_dimension
        ):
            continue
        current = compatible_by_case.get(case_id)
        if (
            current is None
            or _CANDIDATE_METHOD_PRIORITY.get(entry['method'], 0)
            > _CANDIDATE_METHOD_PRIORITY.get(current['method'], 0)
        ):
            compatible_by_case[case_id] = entry
    missing_case_ids: List[str] = []
    conflicting_case_ids: List[str] = []

    policy = {
        'schema': STUDY_ACTIVE_SPACE_SCHEMA,
        'status': 'available',
        'candidate_method': selected_method,
        'candidate_methods': selected_methods,
        'family_kind': family_kind,
        'target_signature': list(selected_targets),
        'ncas': selected_dimension[0],
        'nelecas': copy.deepcopy(representative.get('nelecas')),
        'active_electron_count': selected_dimension[1][0],
        'spin_difference': selected_dimension[1][1],
        'supporting_case_ids': supporting_case_ids,
        'missing_case_ids': missing_case_ids,
        'conflicting_case_ids': conflicting_case_ids,
        'case_mapping': 'case_specific_canonical_orbitals',
        'selection_rule': 'preserve explicit manual/approved choices, then select the most-supported physical target family and CAS sector',
    }

    for decision in resolved:
        case_id = str(decision.get('case_id') or '')
        candidate = compatible_by_case.get(case_id)
        compatible = bool(candidate)
        tags = list(decision.get('tags') or [])
        reasons = list(decision.get('reasons') or [])
        if compatible:
            contract = copy.deepcopy(candidate['contract'])
            contract['study_active_space'] = {
                'schema': STUDY_ACTIVE_SPACE_SCHEMA,
                'candidate_method': candidate['method'],
                'candidate_methods': selected_methods,
                'ncas': selected_dimension[0],
                'nelecas': copy.deepcopy(policy['nelecas']),
                'case_mapping': 'case_specific_canonical_orbitals',
            }
            decision['active_space_contract'] = contract
            decision['study_active_space_status'] = 'compatible'
        else:
            original = decision.get('active_space_contract')
            if isinstance(original, dict):
                decision['case_active_space_contract'] = copy.deepcopy(original)
                conflicting_case_ids.append(case_id)
                if 'study_active_space_conflict' not in tags:
                    tags.append('study_active_space_conflict')
                if _fixed_selection(original):
                    reasons.append(
                        'The explicit manual/approved choice conflicts with the shared study active space. '
                        'It is preserved in case_active_space_contract; review the choices before execution.'
                    )
                else:
                    reasons.append(
                        'This case does not provide the shared study CAS({0}, {1}) chemical manifold; '
                        'rerun its active-space probe before multireference execution.'.format(
                            policy['nelecas'], policy['ncas'],
                        )
                    )
                decision['study_active_space_status'] = 'conflict'
            else:
                missing_case_ids.append(case_id)
                decision['study_active_space_status'] = 'missing'
            decision['active_space_contract'] = None
            action = 'active_space_choice_review_required' if _fixed_selection(original) else 'active_space_probe_required'
            if action not in tags:
                tags.append(action)
        decision['tags'] = tags
        decision['reasons'] = reasons
        decision['study_active_space_policy'] = policy

    policy['missing_case_ids'] = sorted({value for value in missing_case_ids if value})
    policy['conflicting_case_ids'] = sorted({value for value in conflicting_case_ids if value})
    if policy['conflicting_case_ids']:
        policy['status'] = 'conflict'
    elif policy['missing_case_ids']:
        policy['status'] = 'incomplete'
    return resolved, policy


__all__ = [
    'STUDY_ACTIVE_SPACE_SCHEMA',
    'executable_active_space_contract',
    'resolve_study_active_space_contracts',
]
