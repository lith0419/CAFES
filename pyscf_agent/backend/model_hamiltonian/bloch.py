from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Sequence, Tuple

from .spec import normalize_model_spec, validate_model_hamiltonian_spec


METAL_TOLERANCE = 1e-8
DOS_TEMPORARY_ELEMENT_BUDGET = 1_000_000


def _ordered_sites(spec: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
    sites = sorted(spec.get('sites') or [], key=lambda item: int(item['id']))
    return sites, {int(site['id']): index for index, site in enumerate(sites)}


def _reduced_offset(bond: Dict[str, Any], dimension: int) -> Tuple[int, ...]:
    return tuple(int(value) for value in bond.get('cell_offset', [0] * dimension))


def build_bloch_hamiltonian(spec: Dict[str, Any], reduced_kpoint: Sequence[float]):
    """Build h(k) using reduced reciprocal coordinates and integer cell offsets."""
    normalized = normalize_model_spec(spec)
    errors = validate_model_hamiltonian_spec(normalized, solver_name='tight_binding')
    if errors:
        raise ValueError('; '.join(errors))
    return _build_bloch_hamiltonian_prevalidated(normalized, reduced_kpoint)


def _build_bloch_hamiltonian_prevalidated(spec: Dict[str, Any], reduced_kpoint: Sequence[float]):
    import numpy as np  # pylint: disable=import-outside-toplevel

    normalized = spec
    dimension = int(normalized['dimension'])
    if len(reduced_kpoint) != dimension:
        raise ValueError('reduced_kpoint must match the periodic dimension')

    sites, site_to_orbital = _ordered_sites(normalized)
    hamiltonian = np.zeros((len(sites), len(sites)), dtype=complex)
    for index, site in enumerate(sites):
        hamiltonian[index, index] += float(site.get('epsilon', 0.0))

    kpoint = np.asarray(reduced_kpoint, dtype=float)
    for bond in normalized.get('bonds') or []:
        source = site_to_orbital[int(bond['source'])]
        target = site_to_orbital[int(bond['target'])]
        hopping = float(bond.get('effective_t', bond.get('t', 0.0)))
        offset = np.asarray(_reduced_offset(bond, dimension), dtype=float)
        matrix_element = hopping * np.exp(2j * np.pi * float(np.dot(kpoint, offset)))
        hamiltonian[source, target] += matrix_element
        if bool(bond.get('add_hermitian_conjugate', True)):
            hamiltonian[target, source] += np.conjugate(matrix_element)

    if not np.allclose(hamiltonian, hamiltonian.conj().T, atol=1e-10, rtol=1e-10):
        raise ValueError(
            'Bloch Hamiltonian is not Hermitian; add reverse hopping terms or enable add_hermitian_conjugate'
        )
    return hamiltonian


def _reciprocal_vectors(lattice_vectors: Sequence[Sequence[float]]):
    import numpy as np  # pylint: disable=import-outside-toplevel

    direct = np.asarray(lattice_vectors, dtype=float)
    return 2.0 * np.pi * np.linalg.pinv(direct).T


def _mesh_axis(size: int, scheme: str, shift: float):
    import numpy as np  # pylint: disable=import-outside-toplevel

    indices = np.arange(size, dtype=float)
    if scheme == 'monkhorst_pack':
        coordinates = (2.0 * indices + 1.0 - size) / (2.0 * size)
    else:
        coordinates = (indices - math.floor(size / 2)) / size
    coordinates = coordinates + float(shift) / size
    return (coordinates + 0.5) % 1.0 - 0.5


def build_reduced_kmesh(spec: Dict[str, Any]):
    import numpy as np  # pylint: disable=import-outside-toplevel

    reciprocal = spec['reciprocal_space']
    kmesh = [int(value) for value in reciprocal['kmesh']]
    scheme = str(reciprocal.get('scheme') or 'gamma_centered').strip().lower().replace('-', '_')
    shift = [float(value) for value in reciprocal.get('shift', [0.0] * len(kmesh))]
    axes = [_mesh_axis(size, scheme, shift[index]) for index, size in enumerate(kmesh)]
    meshes = np.meshgrid(*axes, indexing='ij')
    return np.stack([mesh.reshape(-1) for mesh in meshes], axis=1)


def _automatic_path_vertices(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    dimension = int(spec['dimension'])
    if dimension == 1:
        return [
            {'label': '-X', 'k': [-0.5]},
            {'label': 'Gamma', 'k': [0.0]},
            {'label': 'X', 'k': [0.5]},
        ]
    preset = str(spec.get('preset') or '').strip().lower()
    if preset in ('honeycomb', 'triangular', 'kagome', 'dice'):
        return [
            {'label': 'Gamma', 'k': [0.0, 0.0]},
            {'label': 'M', 'k': [0.5, 0.0]},
            {'label': 'K', 'k': [1.0 / 3.0, -1.0 / 3.0]},
            {'label': 'Gamma', 'k': [0.0, 0.0]},
        ]
    return [
        {'label': 'Gamma', 'k': [0.0, 0.0]},
        {'label': 'X', 'k': [0.5, 0.0]},
        {'label': 'M', 'k': [0.5, 0.5]},
        {'label': 'Gamma', 'k': [0.0, 0.0]},
    ]


def _path_vertices(spec: Dict[str, Any]) -> List[Dict[str, Any]]:
    path = spec['reciprocal_space']['path']
    if str(path.get('mode') or 'automatic').strip().lower() != 'custom':
        return _automatic_path_vertices(spec)
    return [
        {
            'label': str(point.get('label') or 'k{0}'.format(index + 1)),
            'k': [float(value) for value in point['k']],
        }
        for index, point in enumerate(path['points'])
    ]


def _interpolate_band_path(spec: Dict[str, Any]):
    import numpy as np  # pylint: disable=import-outside-toplevel

    vertices = _path_vertices(spec)
    points_per_segment = int(spec['reciprocal_space']['path'].get('points_per_segment', 80))
    reciprocal_vectors = _reciprocal_vectors(spec['lattice_vectors'])
    reduced_points: List[List[float]] = []
    distances: List[float] = []
    labels: List[Dict[str, Any]] = []
    segments: List[Dict[str, Any]] = []
    distance = 0.0
    previous_cartesian = None

    for segment_index, (start, end) in enumerate(zip(vertices[:-1], vertices[1:])):
        start_k = np.asarray(start['k'], dtype=float)
        end_k = np.asarray(end['k'], dtype=float)
        segment_start_index = len(reduced_points) - (1 if segment_index else 0)
        for point_index, fraction in enumerate(np.linspace(0.0, 1.0, points_per_segment)):
            if segment_index and point_index == 0:
                continue
            reduced = start_k + fraction * (end_k - start_k)
            cartesian = reduced @ reciprocal_vectors
            if previous_cartesian is not None:
                distance += float(np.linalg.norm(cartesian - previous_cartesian))
            reduced_points.append(reduced.tolist())
            distances.append(distance)
            previous_cartesian = cartesian
        segment_end_index = len(reduced_points) - 1
        segments.append({
            'start_label': start['label'],
            'end_label': end['label'],
            'start_index': max(segment_start_index, 0),
            'end_index': segment_end_index,
        })

    vertex_indices = [0]
    for segment in segments:
        vertex_indices.append(segment['end_index'])
    for vertex, index in zip(vertices, vertex_indices):
        labels.append({'label': vertex['label'], 'index': index, 'distance': distances[index]})
    return reduced_points, distances, labels, segments, vertices


def _eigenvalues(spec: Dict[str, Any], kpoints: Iterable[Sequence[float]]):
    import numpy as np  # pylint: disable=import-outside-toplevel

    return np.asarray([
        np.linalg.eigvalsh(_build_bloch_hamiltonian_prevalidated(spec, kpoint)).real
        for kpoint in kpoints
    ])


def _zero_temperature_occupations(energies, electrons_per_cell: float):
    import numpy as np  # pylint: disable=import-outside-toplevel

    nkpoints, norb = energies.shape
    occupations = np.zeros_like(energies, dtype=float)
    target = float(electrons_per_cell) * nkpoints
    flattened_energies = np.asarray(energies, dtype=float).reshape(-1)
    order = np.argsort(flattened_energies, kind='mergesort')
    ordered_energies = flattened_energies[order]
    flattened_occupations = occupations.reshape(-1)
    span = float(np.max(energies) - np.min(energies)) if ordered_energies.size else 0.0
    degeneracy_tolerance = max(1e-10, span * 1e-10)
    remaining = target
    group_index = 0
    partially_occupied_energy = None
    while group_index < ordered_energies.size and remaining > 1e-12:
        end = group_index + 1
        while end < ordered_energies.size and abs(ordered_energies[end] - ordered_energies[group_index]) <= degeneracy_tolerance:
            end += 1
        group_indices = order[group_index:end]
        electrons = min(remaining, 2.0 * len(group_indices))
        occupation = electrons / len(group_indices)
        flattened_occupations[group_indices] = occupation
        if occupation < 2.0 - 1e-10:
            partially_occupied_energy = float(ordered_energies[group_index])
        remaining -= electrons
        group_index = end

    occupied_levels = flattened_energies[flattened_occupations > 1e-10]
    available_levels = flattened_energies[flattened_occupations < 2.0 - 1e-10]
    occupied_edge = float(np.max(occupied_levels)) if occupied_levels.size else None
    available_edge = float(np.min(available_levels)) if available_levels.size else None
    chemical_potential = partially_occupied_energy
    chemical_potential_source = 'partially_occupied_level' if chemical_potential is not None else None
    if chemical_potential is None and occupied_edge is not None and available_edge is not None:
        chemical_potential = 0.5 * (occupied_edge + available_edge)
        chemical_potential_source = 'occupied_unoccupied_mesh_bracket'
    elif chemical_potential is None and occupied_edge is not None:
        chemical_potential = occupied_edge
        chemical_potential_source = 'highest_occupied_level'
    elif chemical_potential is None and available_edge is not None:
        chemical_potential = available_edge
        chemical_potential_source = 'lowest_unoccupied_level'
    return occupations, chemical_potential, chemical_potential_source


def _band_edges(energies, occupations, electrons_per_cell: float) -> Dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    _nkpoints, norb = energies.shape
    filled_band_count = electrons_per_cell / 2.0
    integer_band_filling = abs(filled_band_count - round(filled_band_count)) <= 1e-10
    partial_occupation = bool(np.any((occupations > 1e-10) & (occupations < 2.0 - 1e-10)))
    if integer_band_filling and 0 < round(filled_band_count) < norb:
        occupied_band_count = int(round(filled_band_count))
        valence = energies[:, occupied_band_count - 1]
        conduction = energies[:, occupied_band_count]
        valence_max = float(np.max(valence))
        conduction_min = float(np.min(conduction))
        indirect_gap = float(conduction_min - valence_max)
        direct_gap = float(np.min(conduction - valence))
        is_metal = partial_occupation or indirect_gap <= METAL_TOLERANCE
        return {
            'valence_band_max': valence_max,
            'conduction_band_min': conduction_min,
            'band_gap': 0.0 if is_metal else indirect_gap,
            'raw_indirect_gap': indirect_gap,
            'direct_gap': max(0.0, direct_gap),
            'is_metal': is_metal,
            'partially_filled_band': partial_occupation,
            'filled_band_count': occupied_band_count,
        }

    occupied = energies[occupations > 1e-10]
    available = energies[occupations < 2.0 - 1e-10]
    return {
        'valence_band_max': float(np.max(occupied)) if occupied.size else None,
        'conduction_band_min': float(np.min(available)) if available.size else None,
        'band_gap': None if electrons_per_cell <= 1e-12 or electrons_per_cell >= 2.0 * norb - 1e-12 else 0.0,
        'raw_indirect_gap': None,
        'direct_gap': None if electrons_per_cell <= 1e-12 or electrons_per_cell >= 2.0 * norb - 1e-12 else 0.0,
        'is_metal': 0.0 < electrons_per_cell < 2.0 * norb,
        'partially_filled_band': 0.0 < electrons_per_cell < 2.0 * norb,
        'filled_band_count': int(math.floor(filled_band_count)),
    }


def _density_of_states(energies, settings: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    energy_min = float(np.min(energies))
    energy_max = float(np.max(energies))
    bandwidth = energy_max - energy_min
    sigma_setting = settings.get('sigma')
    sigma = float(sigma_setting) if sigma_setting is not None else max(bandwidth / 100.0, 1e-3)
    point_count = int(settings.get('points', 401))
    grid = np.linspace(energy_min - 4.0 * sigma, energy_max + 4.0 * sigma, point_count)
    flattened_energies = np.asarray(energies, dtype=float).reshape(-1)
    chunk_size = max(1, min(
        flattened_energies.size,
        DOS_TEMPORARY_ELEMENT_BUDGET // max(point_count, 1),
    ))
    density_sum = np.zeros(point_count, dtype=float)
    normalization = sigma * math.sqrt(2.0 * math.pi)
    for start in range(0, flattened_energies.size, chunk_size):
        energy_chunk = flattened_energies[start:start + chunk_size]
        difference = grid[:, None] - energy_chunk[None, :]
        density_sum += np.sum(np.exp(-0.5 * (difference / sigma) ** 2) / normalization, axis=1)
    density = 2.0 * density_sum / energies.shape[0]
    return {
        'method': 'gaussian_broadening',
        'sigma': sigma,
        'points': point_count,
        'normalization': '2 states per orbital per cell',
        'energy': grid.tolist(),
        'density': density.tolist(),
    }


def _complex_matrix_payload(matrix) -> Dict[str, Any]:
    return {
        'real': matrix.real.tolist(),
        'imag': matrix.imag.tolist(),
        'shape': list(matrix.shape),
    }


def _hamiltonian_samples(spec: Dict[str, Any], vertices: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return [
        {
            'label': vertex['label'],
            'reduced_kpoint': vertex['k'],
            'hamiltonian': _complex_matrix_payload(_build_bloch_hamiltonian_prevalidated(spec, vertex['k'])),
        }
        for vertex in vertices
    ]


def _interaction_treatment(spec: Dict[str, Any]) -> Dict[str, Any]:
    nonzero_u = [float(site.get('U', 0.0)) for site in spec.get('sites') or [] if abs(float(site.get('U', 0.0))) > 1e-12]
    nonzero_v = [
        float(bond.get('effective_V', bond.get('V', 0.0)))
        for bond in spec.get('bonds') or []
        if abs(float(bond.get('effective_V', bond.get('V', 0.0)))) > 1e-12
    ]
    present = bool(nonzero_u or nonzero_v)
    return {
        'solver': 'tight_binding',
        'one_body_terms_applied': True,
        'interaction_terms_applied': False,
        'nonzero_onsite_u_count': len(nonzero_u),
        'nonzero_intersite_v_count': len(nonzero_v),
        'status': 'recorded_not_applied' if present else 'not_present',
        'reason': (
            'The Bloch tight-binding solver diagonalizes only h(k); U and V are retained in the input contract '
            'for future mean-field, DMFT, DMET, or many-body solvers.'
            if present
            else 'No nonzero U or V terms were supplied.'
        ),
    }


def run_bloch_tight_binding(spec: Dict[str, Any], outputs: Any = None) -> Dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    normalized = normalize_model_spec(spec)
    errors = validate_model_hamiltonian_spec(normalized, solver_name='tight_binding')
    if errors:
        raise ValueError('; '.join(errors))

    sites, _site_to_orbital = _ordered_sites(normalized)
    energy_unit = str(normalized.get('energy_unit') or 'a.u.').strip() or 'a.u.'
    reciprocal = normalized['reciprocal_space']
    kmesh_points = build_reduced_kmesh(normalized)
    mesh_energies = _eigenvalues(normalized, kmesh_points)
    electrons_per_cell = float(normalized['occupation']['electrons_per_cell'])
    occupations, chemical_potential, chemical_potential_source = _zero_temperature_occupations(
        mesh_energies,
        electrons_per_cell,
    )
    edges = _band_edges(mesh_energies, occupations, electrons_per_cell)
    fermi_energy = chemical_potential
    fermi_energy_source = chemical_potential_source
    if not edges['is_metal'] and edges['valence_band_max'] is not None:
        fermi_energy = edges['valence_band_max']
        fermi_energy_source = 'valence_band_max'

    path_kpoints, path_distances, path_labels, path_segments, vertices = _interpolate_band_path(normalized)
    path_energies = _eigenvalues(normalized, path_kpoints)
    bandwidth = float(np.max(mesh_energies) - np.min(mesh_energies))
    band_energy = float(np.sum(mesh_energies * occupations) / len(kmesh_points))
    nonzero_hopping_count = sum(
        abs(float(bond.get('effective_t', bond.get('t', 0.0)))) > 1e-12
        for bond in normalized.get('bonds') or []
    )
    interaction_treatment = _interaction_treatment(normalized)
    result = {
        'task_type': 'model_hamiltonian',
        'representation': 'bloch',
        'model': normalized.get('model', 'hubbard'),
        'solver': 'tight_binding',
        'converged': True,
        'energy': band_energy,
        'energy_kind': 'noninteracting_band_energy_per_cell',
        'energy_unit': energy_unit,
        'norb': len(sites),
        'site_count': len(sites),
        'bond_count': len(normalized.get('bonds') or []),
        'nonzero_hopping_count': int(nonzero_hopping_count),
        'nonzero_v_count': interaction_treatment['nonzero_intersite_v_count'],
        'onsite_u_count': interaction_treatment['nonzero_onsite_u_count'],
        'electrons_per_cell': electrons_per_cell,
        'filling': electrons_per_cell / len(sites),
        'band_filling_fraction': electrons_per_cell / (2.0 * len(sites)),
        'kmesh': [int(value) for value in reciprocal['kmesh']],
        'kpoint_count': int(len(kmesh_points)),
        'kpoint_scheme': str(reciprocal.get('scheme') or 'gamma_centered'),
        'kpoint_shift': [float(value) for value in reciprocal.get('shift') or []],
        'fermi_energy': fermi_energy,
        'fermi_energy_source': fermi_energy_source,
        'valence_band_max': edges['valence_band_max'],
        'conduction_band_min': edges['conduction_band_min'],
        'band_gap': edges['band_gap'],
        'raw_indirect_gap': edges['raw_indirect_gap'],
        'direct_gap': edges['direct_gap'],
        'bandwidth': bandwidth,
        'is_metal': edges['is_metal'],
        'partially_filled_band': edges['partially_filled_band'],
        'filled_band_count': edges['filled_band_count'],
        'interaction_treatment': interaction_treatment,
        'bloch_band_structure': {
            'path_mode': reciprocal['path'].get('mode', 'automatic'),
            'reduced_kpoints': path_kpoints,
            'distance': path_distances,
            'energies': path_energies.tolist(),
            'labels': path_labels,
            'segments': path_segments,
            'band_count': len(sites),
            'energy_unit': energy_unit,
            'distance_unit': 'reciprocal_length',
        },
        'bloch_kmesh': {
            'mesh': [int(value) for value in reciprocal['kmesh']],
            'scheme': str(reciprocal.get('scheme') or 'gamma_centered'),
            'shift_mesh_steps': [float(value) for value in reciprocal.get('shift') or []],
            'reduced_kpoints': kmesh_points.tolist(),
            'energies': mesh_energies.tolist(),
            'occupations': occupations.tolist(),
            'spin_degeneracy': 2,
            'energy_unit': energy_unit,
        },
        'bloch_dos': {
            **_density_of_states(mesh_energies, reciprocal['dos']),
            'energy_unit': energy_unit,
            'density_unit': 'states per cell per energy unit',
        },
        'bloch_hamiltonian_samples': _hamiltonian_samples(normalized, vertices),
        'raw_scf_output': '',
        'analysis_text': (
            'Bloch tight-binding calculation completed: model={model}, norb={norb}, '
            'electrons_per_cell={electrons:g}, kmesh={kmesh}, band_energy={energy:.12f} {unit}/cell, '
            'band_gap={gap}, is_metal={metal}'
        ).format(
            model=normalized.get('model', 'hubbard'),
            norb=len(sites),
            electrons=electrons_per_cell,
            kmesh=tuple(int(value) for value in reciprocal['kmesh']),
            energy=band_energy,
            unit=energy_unit,
            gap=edges['band_gap'],
            metal=edges['is_metal'],
        ),
    }
    if outputs and 'strong_correlation_diagnostics' in outputs:
        result['strong_correlation_diagnostics_unavailable_reason'] = (
            'The noninteracting Bloch tight-binding solver does not evaluate strong-correlation diagnostics.'
        )
    return result
