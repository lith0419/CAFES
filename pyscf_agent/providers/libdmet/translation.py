from __future__ import annotations

import itertools
import math
from collections import Counter
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple


TRANSLATION_AUDIT_SCHEMA = 'pyscf-agent.builder-translation-audit.v1'


def resolve_primitive_cell_layout(
    sites: Sequence[Mapping[str, Any]],
) -> Tuple[Optional[Dict[str, Any]], List[str]]:
    """Resolve Builder primitive-cell membership without using site ordering."""

    tagged = [site for site in sites if site.get('cell_index') is not None]
    if not tagged:
        return None, []
    if len(tagged) != len(sites):
        return None, ['Primitive-cell metadata must be present on every model site or omitted entirely']

    cells: Dict[Tuple[int, ...], List[Tuple[int, int]]] = {}
    dimension: Optional[int] = None
    for site in sites:
        raw_cell_index = site.get('cell_index')
        if not isinstance(raw_cell_index, (list, tuple)) or not raw_cell_index:
            return None, ['Every cell_index must be a non-empty integer list']
        try:
            cell_index = tuple(int(value) for value in raw_cell_index)
            basis_index = int(site.get('basis_index'))
            site_id = int(site.get('id'))
        except (TypeError, ValueError):
            return None, ['cell_index, basis_index, and site id must be integers']
        if any(value < 0 for value in cell_index) or basis_index < 0:
            return None, ['cell_index and basis_index values must be non-negative']
        if dimension is None:
            dimension = len(cell_index)
        elif len(cell_index) != dimension:
            return None, ['All Builder cell_index values must have the same dimension']
        cells.setdefault(cell_index, []).append((basis_index, site_id))

    ordered_cells = sorted(cells)
    reference_basis = sorted(basis for basis, _ in cells[ordered_cells[0]])
    if reference_basis != list(range(len(reference_basis))):
        return None, ['basis_index values in each primitive cell must be contiguous from zero']
    for cell_index, members in cells.items():
        basis_indices = sorted(basis for basis, _ in members)
        if basis_indices != reference_basis:
            return None, [
                'Every primitive cell must contain the same basis_index set; '
                'cell {0} is incomplete'.format(list(cell_index))
            ]

    axes = [
        sorted({cell[index] for cell in ordered_cells})
        for index in range(int(dimension or 0))
    ]
    expected_cells = {tuple(coordinates) for coordinates in itertools.product(*axes)}
    if set(ordered_cells) != expected_cells:
        return None, ['Builder primitive-cell indices must form a complete rectangular cell grid']
    return {
        'dimension': int(dimension or 0),
        'lattice_shape': [len(axis) for axis in axes],
        'basis_size': len(reference_basis),
        'axes': axes,
        'cells': {
            cell_index: [site_id for _, site_id in sorted(members)]
            for cell_index, members in cells.items()
        },
    }, []


def _rounded(value: Any, tolerance: float) -> float:
    digits = max(0, min(14, int(round(-math.log10(tolerance)))))
    return round(float(value), digits)


def _cell_basis_records(
    sites: Sequence[Mapping[str, Any]],
) -> Tuple[Dict[int, Tuple[Tuple[int, ...], int]], Dict[Tuple[Tuple[int, ...], int], Mapping[str, Any]]]:
    by_site_id: Dict[int, Tuple[Tuple[int, ...], int]] = {}
    by_cell_basis: Dict[Tuple[Tuple[int, ...], int], Mapping[str, Any]] = {}
    for site in sites:
        site_id = int(site['id'])
        cell_index = tuple(int(value) for value in site['cell_index'])
        basis_index = int(site['basis_index'])
        by_site_id[site_id] = (cell_index, basis_index)
        by_cell_basis[(cell_index, basis_index)] = site
    return by_site_id, by_cell_basis


def _basis_bipartition(
    basis_size: int,
    one_body_terms: Sequence[Mapping[str, Any]],
    tolerance: float,
) -> Optional[List[List[int]]]:
    adjacency = {index: set() for index in range(basis_size)}
    for term in one_body_terms:
        if abs(float(term['t'])) <= tolerance:
            continue
        source = int(term['source_basis'])
        target = int(term['target_basis'])
        if source == target:
            return None
        adjacency[source].add(target)
        adjacency[target].add(source)
    colors: Dict[int, int] = {}
    for start in range(basis_size):
        if start in colors:
            continue
        colors[start] = 0
        pending = [start]
        while pending:
            source = pending.pop()
            for target in adjacency[source]:
                expected = 1 - colors[source]
                if target in colors and colors[target] != expected:
                    return None
                if target not in colors:
                    colors[target] = expected
                    pending.append(target)
    return [
        [index for index in range(basis_size) if colors[index] == color]
        for color in (0, 1)
    ]


def audit_builder_translation(
    spec: Mapping[str, Any],
    *,
    tolerance: float = 1.0e-10,
) -> Dict[str, Any]:
    """Verify that Builder cells carry one translation-equivalent Hamiltonian template."""

    sites = list(spec.get('sites') or [])
    bonds = list(spec.get('bonds') or [])
    layout, layout_errors = resolve_primitive_cell_layout(sites)
    issues = list(layout_errors)
    checks: Dict[str, bool] = {
        'primitive_cell_metadata': layout is not None and not layout_errors,
        'primitive_cell_contract': False,
        'translation_equivalent_onsite_terms': False,
        'translation_equivalent_bonds': False,
    }
    audit: Dict[str, Any] = {
        'schema': TRANSLATION_AUDIT_SCHEMA,
        'eligible': False,
        'checks': checks,
        'issues': issues,
        'dimension': None,
        'repetitions': [],
        'basis_size': None,
        'reference_cell': [],
        'representative_site_ids': [],
        'basis_sites': [],
        'one_body_terms': [],
        'basis_bipartition': None,
    }
    if layout is None or layout_errors:
        if not issues:
            issues.append('Builder primitive-cell metadata is unavailable')
        return audit

    dimension = int(layout['dimension'])
    repetitions = list(layout['lattice_shape'])
    basis_size = int(layout['basis_size'])
    primitive_cell = spec.get('primitive_cell')
    primitive_cell = primitive_cell if isinstance(primitive_cell, Mapping) else {}
    declared_repetitions = primitive_cell.get('repetitions')
    declared_basis_size = primitive_cell.get('basis_size')
    vectors = primitive_cell.get('vectors')
    try:
        primitive_contract = (
            isinstance(declared_repetitions, (list, tuple))
            and [int(value) for value in declared_repetitions] == repetitions
            and int(declared_basis_size or 0) == basis_size
            and isinstance(vectors, (list, tuple))
            and len(vectors) == dimension
            and all(
                isinstance(vector, (list, tuple))
                and len(vector) >= dimension
                and all(math.isfinite(float(component)) for component in vector)
                for vector in vectors
            )
        )
    except (TypeError, ValueError):
        primitive_contract = False
    checks['primitive_cell_contract'] = primitive_contract
    if not primitive_contract:
        issues.append(
            'primitive_cell vectors, repetitions, or basis_size do not match the site metadata'
        )

    by_site_id, by_cell_basis = _cell_basis_records(sites)
    reference_cell = tuple(axis[0] for axis in layout['axes'])
    representative_site_ids = list(layout['cells'][reference_cell])

    onsite_reference: Dict[int, Tuple[float, float]] = {}
    onsite_equivalent = True
    for basis_index in range(basis_size):
        reference_site = by_cell_basis[(reference_cell, basis_index)]
        signature = (
            _rounded(reference_site.get('epsilon', 0.0), tolerance),
            _rounded(reference_site.get('U', 0.0), tolerance),
        )
        onsite_reference[basis_index] = signature
        for cell_index in layout['cells']:
            site = by_cell_basis[(cell_index, basis_index)]
            candidate = (
                _rounded(site.get('epsilon', 0.0), tolerance),
                _rounded(site.get('U', 0.0), tolerance),
            )
            if candidate != signature:
                onsite_equivalent = False
                issues.append(
                    'Onsite terms for basis_index {0} differ in cell {1}'.format(
                        basis_index,
                        list(cell_index),
                    )
                )
                break
    checks['translation_equivalent_onsite_terms'] = onsite_equivalent

    profiles: Dict[Tuple[Tuple[int, ...], int], Counter] = {
        key: Counter() for key in by_cell_basis
    }
    invalid_bond = False
    for position, bond in enumerate(bonds):
        try:
            source_id = int(bond['source'])
            target_id = int(bond['target'])
            source_cell, source_basis = by_site_id[source_id]
            target_cell, target_basis = by_site_id[target_id]
        except (KeyError, TypeError, ValueError):
            invalid_bond = True
            issues.append('Bond {0} does not reference valid Builder sites'.format(position))
            continue
        hopping = _rounded(bond.get('effective_t', bond.get('t', 0.0)), tolerance)
        interaction = _rounded(bond.get('effective_V', bond.get('V', 0.0)), tolerance)
        forward_offset = tuple(
            (target - source) % length
            for source, target, length in zip(source_cell, target_cell, repetitions)
        )
        reverse_offset = tuple((-value) % length for value, length in zip(forward_offset, repetitions))
        profiles[(source_cell, source_basis)][
            (target_basis, forward_offset, hopping, interaction)
        ] += 1
        profiles[(target_cell, target_basis)][
            (source_basis, reverse_offset, hopping, interaction)
        ] += 1

    bonds_equivalent = not invalid_bond
    reference_profiles = {
        basis_index: profiles[(reference_cell, basis_index)]
        for basis_index in range(basis_size)
    }
    for cell_index in layout['cells']:
        for basis_index in range(basis_size):
            if profiles[(cell_index, basis_index)] != reference_profiles[basis_index]:
                bonds_equivalent = False
                issues.append(
                    'Bond environment for basis_index {0} differs in cell {1}'.format(
                        basis_index,
                        list(cell_index),
                    )
                )
    checks['translation_equivalent_bonds'] = bonds_equivalent

    one_body_terms: List[Dict[str, Any]] = []
    for source_basis, profile in reference_profiles.items():
        for (target_basis, cell_offset, hopping, interaction), multiplicity in sorted(profile.items()):
            one_body_terms.append({
                'source_basis': int(source_basis),
                'target_basis': int(target_basis),
                'cell_offset': list(cell_offset),
                't': float(hopping),
                'V': float(interaction),
                'multiplicity': int(multiplicity),
            })

    basis_sites = []
    origin_site = by_cell_basis[(reference_cell, 0)]
    origin = [float(origin_site.get(axis, 0.0)) for axis in ('x', 'y', 'z')[:dimension]]
    for basis_index in range(basis_size):
        site = by_cell_basis[(reference_cell, basis_index)]
        position = [
            float(site.get(axis, 0.0)) - origin[axis_index]
            for axis_index, axis in enumerate(('x', 'y', 'z')[:dimension])
        ]
        basis_sites.append({
            'basis_index': basis_index,
            'site_id': int(site['id']),
            'position': position,
            'epsilon': float(onsite_reference[basis_index][0]),
            'U': float(onsite_reference[basis_index][1]),
            'sublattice': site.get('sublattice'),
        })

    audit.update({
        'eligible': all(checks.values()),
        'dimension': dimension,
        'repetitions': repetitions,
        'basis_size': basis_size,
        'reference_cell': list(reference_cell),
        'representative_site_ids': representative_site_ids,
        'basis_sites': basis_sites,
        'one_body_terms': one_body_terms,
        'basis_bipartition': _basis_bipartition(basis_size, one_body_terms, tolerance),
    })
    return audit
