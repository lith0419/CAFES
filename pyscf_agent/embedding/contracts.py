from __future__ import annotations

import copy
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence


EMBEDDING_REFERENCE_SCHEMA = 'pyscf-agent.embedding-reference.v1'
LOCALIZED_SUBSPACE_SCHEMA = 'pyscf-agent.localized-subspace.v1'
LOCALIZED_HAMILTONIAN_SCHEMA = 'pyscf-agent.localized-hamiltonian.v1'
CORRELATED_SUBSPACE_AUDIT_SCHEMA = 'pyscf-agent.correlated-subspace-audit.v1'


def _timestamp() -> str:
    return datetime.now(timezone.utc).isoformat()


def _unique_nonnegative_indices(values: Iterable[Any], field_name: str) -> List[int]:
    normalized: List[int] = []
    for value in values:
        try:
            index = int(value)
        except (TypeError, ValueError) as exc:
            raise ValueError('{0} must contain integer indices'.format(field_name)) from exc
        if index < 0:
            raise ValueError('{0} must contain non-negative indices'.format(field_name))
        if index not in normalized:
            normalized.append(index)
    return normalized


def _normalized_fragments(
    fragments: Optional[Sequence[Mapping[str, Any]]],
    selected_orbitals: Sequence[int],
) -> List[Dict[str, Any]]:
    selected = set(selected_orbitals)
    normalized: List[Dict[str, Any]] = []
    seen_ids = set()
    for position, raw_fragment in enumerate(fragments or []):
        if not isinstance(raw_fragment, Mapping):
            raise ValueError('embedding fragments must be mappings')
        fragment_id = str(raw_fragment.get('fragment_id') or 'fragment-{0}'.format(position + 1)).strip()
        if not fragment_id or fragment_id in seen_ids:
            raise ValueError('embedding fragment_id values must be non-empty and unique')
        seen_ids.add(fragment_id)
        orbital_indices = _unique_nonnegative_indices(
            raw_fragment.get('orbital_indices') or [],
            'fragment orbital_indices',
        )
        if selected and any(index not in selected for index in orbital_indices):
            raise ValueError('fragment orbital_indices must belong to the correlated subspace')
        normalized.append({
            'fragment_id': fragment_id,
            'label': str(raw_fragment.get('label') or fragment_id),
            'site_ids': _unique_nonnegative_indices(
                raw_fragment.get('site_ids') or [],
                'fragment site_ids',
            ),
            'orbital_indices': orbital_indices,
            'atom_indices': _unique_nonnegative_indices(
                raw_fragment.get('atom_indices') or [],
                'fragment atom_indices',
            ),
            'metadata': copy.deepcopy(raw_fragment.get('metadata') or {}),
        })
    return normalized


def build_correlated_subspace_audit(
    *,
    system_type: str,
    provider: str,
    localization_method: str,
    orbital_indices: Sequence[int],
    total_orbitals: Optional[int] = None,
    atom_indices: Optional[Sequence[int]] = None,
    fragments: Optional[Sequence[Mapping[str, Any]]] = None,
    orbital_labels: Optional[Sequence[Any]] = None,
    occupations: Optional[Sequence[Any]] = None,
    orbital_contributions: Optional[Mapping[Any, Any]] = None,
    selection_reasons: Optional[Sequence[str]] = None,
    interaction: Optional[Mapping[str, Any]] = None,
    electron_count: Any = None,
    approved: bool = False,
) -> Dict[str, Any]:
    """Build the reviewable correlated-subspace boundary shared by DMET and DMFT."""

    selected = _unique_nonnegative_indices(orbital_indices, 'orbital_indices')
    selected_atoms = _unique_nonnegative_indices(atom_indices or [], 'atom_indices')
    if total_orbitals is not None:
        total_orbitals = int(total_orbitals)
        if total_orbitals < 1:
            raise ValueError('total_orbitals must be positive')
        if any(index >= total_orbitals for index in selected):
            raise ValueError('orbital_indices exceed total_orbitals')
    if approved:
        raise ValueError(
            'Use approve_correlated_subspace_audit so approval records an actor and timestamp'
        )

    labels = list(orbital_labels or [])
    occupation_values = list(occupations or [])
    contributions = orbital_contributions or {}
    normalized_fragments = _normalized_fragments(fragments, selected)
    fragment_membership = {
        index: [
            fragment['fragment_id']
            for fragment in normalized_fragments
            if index in fragment['orbital_indices']
        ]
        for index in selected
    }
    records = []
    for index in selected:
        raw_contribution = contributions.get(index, contributions.get(str(index), {}))
        records.append({
            'orbital_index': index,
            'label': str(labels[index]) if index < len(labels) else '',
            'occupation': (
                float(occupation_values[index])
                if index < len(occupation_values) and occupation_values[index] is not None
                else None
            ),
            'contributions': copy.deepcopy(raw_contribution) if isinstance(raw_contribution, Mapping) else {},
            'fragment_ids': fragment_membership[index],
        })

    return {
        'schema': CORRELATED_SUBSPACE_AUDIT_SCHEMA,
        'system_type': str(system_type or '').strip().lower(),
        'provider': str(provider or '').strip().lower(),
        'localization_method': str(localization_method or '').strip().lower(),
        'orbital_indices': selected,
        'atom_indices': selected_atoms,
        'fragments': normalized_fragments,
        'orbital_records': records,
        'electron_count': copy.deepcopy(electron_count),
        'interaction': copy.deepcopy(dict(interaction or {})),
        'selection_reasons': [str(reason) for reason in (selection_reasons or []) if str(reason).strip()],
        'consistency': {
            'selected_orbital_count': len(selected),
            'total_orbitals': total_orbitals,
            'indices_unique': len(selected) == len(set(selected)),
            'indices_in_range': total_orbitals is None or all(index < total_orbitals for index in selected),
            'fragment_orbitals_in_subspace': True,
        },
        'approval': {
            'status': 'pending',
            'approved': False,
            'approved_by': '',
            'approved_at': '',
            'note': '',
        },
    }


def approve_correlated_subspace_audit(
    audit: Mapping[str, Any],
    *,
    approved_by: str,
    note: str = '',
    approved_at: Optional[str] = None,
) -> Dict[str, Any]:
    if str(audit.get('schema') or '') != CORRELATED_SUBSPACE_AUDIT_SCHEMA:
        raise ValueError('Unsupported correlated-subspace audit schema')
    if not list(audit.get('orbital_indices') or []):
        raise ValueError('an empty correlated subspace cannot be approved')
    actor = str(approved_by or '').strip()
    if not actor:
        raise ValueError('approved_by is required')
    approved = copy.deepcopy(dict(audit))
    approved['approval'] = {
        'status': 'approved',
        'approved': True,
        'approved_by': actor,
        'approved_at': str(approved_at or _timestamp()),
        'note': str(note or ''),
    }
    return approved
