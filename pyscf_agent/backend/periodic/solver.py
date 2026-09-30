from __future__ import annotations

import io
import math
import re
import warnings
from collections import Counter
from itertools import product
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from ...registry.platform import (
    SUPPORTED_PERIODIC_ANALYSIS,
    SUPPORTED_PERIODIC_BAND_PATH_MODES,
    SUPPORTED_PERIODIC_BASIS_SETS,
    SUPPORTED_PERIODIC_DENSITY_FITTING_METHODS,
    SUPPORTED_PERIODIC_EXXDIV_OPTIONS,
    SUPPORTED_PERIODIC_JOBS,
    SUPPORTED_PERIODIC_KPOINT_SCHEMES,
    SUPPORTED_PERIODIC_METHODS,
    SUPPORTED_PERIODIC_PSEUDOPOTENTIALS,
    SUPPORTED_PERIODIC_SMEARING_METHODS,
    SUPPORTED_PERIODIC_XC_FUNCTIONALS,
)
from ...contracts import TaskSpec


MAX_PERIODIC_STRUCTURE_BYTES = 2 * 1024 * 1024
MAX_PERIODIC_ATOMS = 2000
MAX_PERIODIC_KPOINTS = 512
MAX_PERIODIC_FFT_GRID_POINTS = 250_000_000
MAX_PERIODIC_BAND_PATH_POINTS = 400
MIN_SEEKPATH_REFERENCE_DISTANCE = 0.005
MAX_SEEKPATH_REFERENCE_DISTANCE = 0.5
MIN_SEEKPATH_SYMPREC = 1e-8
MAX_SEEKPATH_SYMPREC = 0.1
SUPPORTED_STRUCTURE_FORMATS = ('poscar', 'cif')
BAND_PATH_LABEL_PATTERN = re.compile(r'[A-Z][a-z0-9]*')


def _normalize_structure_format(value: Any) -> str:
    normalized = str(value or 'poscar').strip().lower()
    aliases = {
        'vasp': 'poscar',
        'contcar': 'poscar',
    }
    return aliases.get(normalized, normalized)


def _parse_poscar(source_text: str) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]], Dict[str, Any]]:
    lines = [line.strip() for line in source_text.replace('\r\n', '\n').replace('\r', '\n').split('\n') if line.strip()]
    if len(lines) < 8:
        raise ValueError('POSCAR must contain a title, scale, three lattice vectors, species, counts, and coordinates.')

    try:
        raw_scale = float(lines[1].split()[0])
        raw_lattice = np.asarray([
            [float(value) for value in lines[index].split()[:3]]
            for index in range(2, 5)
        ], dtype=float)
    except (TypeError, ValueError) as exc:
        raise ValueError('POSCAR scale and lattice vectors must be numeric.') from exc
    if raw_lattice.shape != (3, 3):
        raise ValueError('POSCAR must contain exactly three three-component lattice vectors.')
    if not math.isfinite(raw_scale) or not np.all(np.isfinite(raw_lattice)):
        raise ValueError('POSCAR scale and lattice vectors must contain finite numeric values.')
    raw_volume = abs(float(np.linalg.det(raw_lattice)))
    if raw_volume <= 1e-12:
        raise ValueError('POSCAR lattice vectors are singular.')
    if raw_scale == 0:
        raise ValueError('POSCAR scale factor cannot be zero.')
    scale = raw_scale if raw_scale > 0 else (-raw_scale / raw_volume) ** (1.0 / 3.0)
    lattice = raw_lattice * scale

    species = lines[5].split()
    try:
        counts = [int(value) for value in lines[6].split()]
    except ValueError as exc:
        raise ValueError('VASP 5 style POSCAR element symbols and integer counts are required.') from exc
    if not species or len(species) != len(counts) or any(count <= 0 for count in counts):
        raise ValueError('POSCAR element symbols and atom counts must have matching positive entries.')
    if not all(re.fullmatch(r'[A-Za-z][A-Za-z]?', symbol) for symbol in species):
        raise ValueError('VASP 5 style POSCAR element symbols are required.')

    coordinate_mode_index = 7
    if lines[coordinate_mode_index].lower().startswith('s'):
        coordinate_mode_index += 1
    if coordinate_mode_index >= len(lines):
        raise ValueError('POSCAR is missing the Direct or Cartesian coordinate mode.')
    mode = lines[coordinate_mode_index].lower()
    fractional = mode.startswith('d')
    cartesian = mode.startswith('c') or mode.startswith('k')
    if not fractional and not cartesian:
        raise ValueError('POSCAR coordinate mode must be Direct or Cartesian.')

    atom_count = sum(counts)
    coordinate_lines = lines[coordinate_mode_index + 1:coordinate_mode_index + 1 + atom_count]
    if len(coordinate_lines) != atom_count:
        raise ValueError('POSCAR atom count does not match the number of coordinate lines.')
    coordinates: List[np.ndarray] = []
    selective_dynamics: List[Optional[List[bool]]] = []
    for line in coordinate_lines:
        parts = line.split()
        if len(parts) < 3:
            raise ValueError('Each POSCAR coordinate line must contain at least three numeric values.')
        try:
            coordinate = np.asarray([float(parts[0]), float(parts[1]), float(parts[2])], dtype=float)
        except ValueError as exc:
            raise ValueError('POSCAR coordinates must be numeric.') from exc
        if not np.all(np.isfinite(coordinate)):
            raise ValueError('POSCAR coordinates must contain finite numeric values.')
        coordinates.append(coordinate.dot(lattice) if fractional else coordinate * scale)
        flags = parts[3:6]
        if lines[7].lower().startswith('s') and len(flags) >= 3:
            normalized_flags = [flag.strip().lower() for flag in flags]
            if any(flag not in ('t', 'f') for flag in normalized_flags):
                raise ValueError('POSCAR selective-dynamics flags must be T or F.')
            selective_dynamics.append([flag == 't' for flag in normalized_flags])
        else:
            selective_dynamics.append(None)

    elements: List[str] = []
    for symbol, count in zip(species, counts):
        normalized_symbol = symbol[0].upper() + symbol[1:].lower()
        elements.extend([normalized_symbol] * count)
    return lattice, list(zip(elements, coordinates)), {
        'parser': 'internal_poscar',
        'source_site_count': atom_count,
        'symmetry_expanded': False,
        'coordinate_mode': 'direct' if fractional else 'cartesian',
        'selective_dynamics': selective_dynamics,
    }


def _parse_cif(source_text: str) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]], Dict[str, Any]]:
    try:
        from ase.io import read as ase_read  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ValueError('CIF parsing requires the ASE dependency.') from exc

    try:
        cif_atoms = ase_read(
            io.BytesIO(source_text.encode('utf-8')),
            format='cif',
            index=0,
            store_tags=True,
            fractional_occupancies=True,
        )
    except Exception as exc:
        raise ValueError('CIF parsing failed: {0}'.format(exc)) from exc
    if cif_atoms is None or len(cif_atoms) == 0:
        raise ValueError('CIF parsing produced no atoms.')

    raw_occupancies = cif_atoms.info.get('_atom_site_occupancy')
    if isinstance(raw_occupancies, (list, tuple, np.ndarray)):
        try:
            partial_occupancy = any(abs(float(value) - 1.0) > 1e-8 for value in raw_occupancies)
        except (TypeError, ValueError) as exc:
            raise ValueError('CIF site occupancies must be numeric.') from exc
        if partial_occupancy:
            raise ValueError(
                'CIF contains partial site occupancies; choose an explicit ordered structure before running PySCF.'
            )
    occupancy_map = cif_atoms.info.get('occupancy')
    if isinstance(occupancy_map, dict):
        for site_occupancy in occupancy_map.values():
            if not isinstance(site_occupancy, dict):
                continue
            if len(site_occupancy) != 1 or any(abs(float(value) - 1.0) > 1e-8 for value in site_occupancy.values()):
                raise ValueError(
                    'CIF contains mixed or partial site occupancies; choose an explicit ordered structure before running PySCF.'
                )

    normalized_lattice = np.asarray(cif_atoms.cell.array, dtype=float)
    normalized_atoms = [
        (str(symbol), np.asarray(coordinate, dtype=float))
        for symbol, coordinate in zip(cif_atoms.get_chemical_symbols(), cif_atoms.get_positions())
    ]
    source_sites = cif_atoms.info.get('_atom_site_fract_x')
    source_site_count = len(source_sites) if isinstance(source_sites, (list, tuple, np.ndarray)) else len(normalized_atoms)
    space_group = cif_atoms.info.get('spacegroup')
    return normalized_lattice, normalized_atoms, {
        'parser': 'ase_cif',
        'source_site_count': int(source_site_count),
        'symmetry_expanded': len(normalized_atoms) > source_site_count,
        'space_group_number': int(getattr(space_group, 'no', 0)) or None,
        'space_group_symbol': str(getattr(space_group, 'symbol', '') or '').strip() or None,
        'selective_dynamics': [None] * len(normalized_atoms),
    }


def _validate_periodic_geometry(lattice: np.ndarray, atoms: List[Tuple[str, np.ndarray]]) -> None:
    determinant = float(np.linalg.det(lattice))
    if lattice.shape != (3, 3) or not np.all(np.isfinite(lattice)) or determinant <= 1e-12:
        raise ValueError('Periodic lattice must contain three finite, non-singular right-handed lattice vectors.')
    if not atoms:
        raise ValueError('Periodic structure must contain at least one atom.')
    if len(atoms) > MAX_PERIODIC_ATOMS:
        raise ValueError('Periodic structure exceeds the {0}-atom limit.'.format(MAX_PERIODIC_ATOMS))
    inverse_lattice = np.linalg.inv(lattice)
    fractional_coordinates = []
    for _symbol, coordinate in atoms:
        normalized_coordinate = np.asarray(coordinate, dtype=float)
        if normalized_coordinate.shape != (3,) or not np.all(np.isfinite(normalized_coordinate)):
            raise ValueError('Periodic atom coordinates must contain three finite numeric values.')
        fractional_coordinates.append(normalized_coordinate.dot(inverse_lattice))
    for first_index in range(len(fractional_coordinates)):
        for second_index in range(first_index):
            fractional_delta = fractional_coordinates[first_index] - fractional_coordinates[second_index]
            fractional_delta -= np.rint(fractional_delta)
            distance = float(np.linalg.norm(fractional_delta.dot(lattice)))
            if distance < 0.1:
                raise ValueError(
                    'Periodic structure contains overlapping or unphysically close atoms at indices {0} and {1}.'.format(
                        second_index,
                        first_index,
                    )
                )


def _parse_periodic_structure_details(source_text: Any, structure_format: Any) -> Dict[str, Any]:
    if not isinstance(source_text, str) or not source_text.strip():
        raise ValueError('Periodic structure text is required.')
    if len(source_text.encode('utf-8')) > MAX_PERIODIC_STRUCTURE_BYTES:
        raise ValueError('Periodic structure input exceeds the 2 MiB limit.')
    normalized_format = _normalize_structure_format(structure_format)
    if normalized_format not in SUPPORTED_STRUCTURE_FORMATS:
        raise ValueError('Unsupported periodic structure format: {0}. Use POSCAR or CIF.'.format(structure_format))
    if normalized_format == 'poscar':
        lattice, atoms, metadata = _parse_poscar(source_text)
    else:
        lattice, atoms, metadata = _parse_cif(source_text)
    _validate_periodic_geometry(lattice, atoms)
    return {
        'source_format': normalized_format,
        'lattice': lattice,
        'atoms': atoms,
        'metadata': metadata,
    }


def parse_periodic_structure(source_text: Any, structure_format: Any) -> Tuple[np.ndarray, List[Tuple[str, np.ndarray]]]:
    details = _parse_periodic_structure_details(source_text, structure_format)
    return details['lattice'], details['atoms']


def _lattice_angles(lattice: np.ndarray) -> List[float]:
    angles = []
    for first, second in ((1, 2), (0, 2), (0, 1)):
        denominator = float(np.linalg.norm(lattice[first]) * np.linalg.norm(lattice[second]))
        cosine = float(np.dot(lattice[first], lattice[second]) / denominator)
        angles.append(math.degrees(math.acos(max(-1.0, min(1.0, cosine)))))
    return angles


def _normalize_band_path(value: Any) -> str:
    text = str(value or '').strip().replace('Γ', 'G').replace('γ', 'G')
    return re.sub(r'[\s\-\u2192\u2013\u2014]+', '', text)


def _band_path_sections(path: str) -> List[List[str]]:
    from ase.dft.kpoints import parse_path_string  # pylint: disable=import-outside-toplevel

    if not path:
        raise ValueError('A custom band path is required.')
    if len(path) > 256:
        raise ValueError('The custom band path exceeds the 256-character limit.')
    sections = parse_path_string(path)
    if not sections or any(len(section) < 2 for section in sections):
        raise ValueError('Each continuous band-path section must contain at least two special-point labels.')
    reconstructed = ','.join(''.join(section) for section in sections)
    if reconstructed != path or any(
        BAND_PATH_LABEL_PATTERN.fullmatch(label) is None
        for section in sections
        for label in section
    ):
        raise ValueError('Use ASE labels such as GXWKGL or disconnected sections such as GX,LU.')
    return sections


def _normalized_band_path_special_points(
    special_points: Dict[str, Sequence[float]],
) -> Dict[str, np.ndarray]:
    if len(special_points) > 64:
        raise ValueError('Explicit band paths support at most 64 special points.')
    normalized: Dict[str, np.ndarray] = {}
    for raw_label, raw_coordinates in special_points.items():
        label = str(raw_label).strip().replace('Γ', 'G').replace('γ', 'G')
        if BAND_PATH_LABEL_PATTERN.fullmatch(label) is None:
            raise ValueError('Invalid explicit special-point label: {0}.'.format(raw_label))
        if label in normalized:
            raise ValueError('Duplicate explicit special-point label after normalization: {0}.'.format(label))
        coordinates = np.asarray(raw_coordinates, dtype=float)
        if coordinates.shape != (3,) or not np.all(np.isfinite(coordinates)):
            raise ValueError(
                'Explicit special point {0} must contain three finite reduced reciprocal coordinates.'.format(label)
            )
        normalized[label] = coordinates
    return normalized


def _ase_band_path(
    lattice_angstrom: np.ndarray,
    *,
    mode: str,
    path: Optional[str],
    special_points: Dict[str, Sequence[float]],
    npoints: int,
) -> Any:
    from ase.cell import Cell as ASECell  # pylint: disable=import-outside-toplevel

    ase_cell = ASECell(np.asarray(lattice_angstrom, dtype=float))
    if mode == 'auto':
        return ase_cell.bandpath(npoints=npoints)

    normalized_path = _normalize_band_path(path)
    sections = _band_path_sections(normalized_path)
    requested_labels = list(dict.fromkeys(label for section in sections for label in section))
    if mode == 'custom':
        standard_path = ase_cell.bandpath(npoints=max(2, npoints))
        available_labels = set(standard_path.special_points)
        unknown_labels = [label for label in requested_labels if label not in available_labels]
        if unknown_labels:
            raise ValueError(
                'Unknown standard special-point label(s): {0}. Available labels: {1}.'.format(
                    ', '.join(unknown_labels),
                    ', '.join(sorted(available_labels)),
                )
            )
        return ase_cell.bandpath(path=normalized_path, npoints=npoints)

    if mode != 'explicit':
        raise ValueError(
            'Unsupported periodic band-path mode: {0}. Use one of: {1}.'.format(
                mode,
                ', '.join(SUPPORTED_PERIODIC_BAND_PATH_MODES),
            )
        )
    normalized_points = _normalized_band_path_special_points(special_points)
    if not normalized_points:
        raise ValueError('Explicit band-path mode requires a special-point coordinate mapping.')
    missing_labels = [label for label in requested_labels if label not in normalized_points]
    if missing_labels:
        raise ValueError(
            'Explicit band path is missing coordinates for label(s): {0}.'.format(', '.join(missing_labels))
        )
    return ase_cell.bandpath(
        path=normalized_path,
        npoints=npoints,
        special_points=normalized_points,
    )


def _band_path_preview(lattice: np.ndarray) -> Dict[str, Any]:
    from ase.cell import Cell as ASECell  # pylint: disable=import-outside-toplevel

    ase_cell = ASECell(np.asarray(lattice, dtype=float))
    try:
        bravais_lattice = ase_cell.get_bravais_lattice()
        band_path = ase_cell.bandpath(npoints=80)
    except Exception as exc:
        return {
            'status': 'unavailable',
            'reason': str(exc),
        }
    return {
        'status': 'available',
        'convention': 'ASE Bravais-lattice path',
        'bravais_lattice': str(bravais_lattice),
        'bravais_lattice_name': getattr(bravais_lattice, 'longname', type(bravais_lattice).__name__),
        'automatic_path': str(band_path.path),
        'special_points_scaled': {
            str(label): [float(value) for value in coordinates]
            for label, coordinates in sorted(band_path.special_points.items())
        },
    }


def _seekpath_standardization(
    lattice: np.ndarray,
    atoms: List[Tuple[str, np.ndarray]],
    *,
    symprec: float,
    reference_distance: Optional[float] = None,
) -> Dict[str, Any]:
    try:
        import seekpath  # pylint: disable=import-outside-toplevel
        from pyscf.data import elements  # pylint: disable=import-outside-toplevel
    except ImportError as exc:
        raise ValueError('SeeK-path mode requires the seekpath package.') from exc

    inverse_lattice = np.linalg.inv(lattice)
    fractional_positions = [
        (np.asarray(coordinate, dtype=float).dot(inverse_lattice) % 1.0).tolist()
        for _symbol, coordinate in atoms
    ]
    atomic_numbers = [int(elements.charge(symbol)) for symbol, _coordinate in atoms]
    structure = (
        np.asarray(lattice, dtype=float),
        np.asarray(fractional_positions, dtype=float),
        np.asarray(atomic_numbers, dtype=int),
    )
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter('always')
        try:
            if reference_distance is None:
                raw = seekpath.get_path(structure, symprec=float(symprec))
            else:
                raw = seekpath.get_explicit_k_path(
                    structure,
                    reference_distance=float(reference_distance),
                    symprec=float(symprec),
                )
        except Exception as exc:
            raise ValueError('SeeK-path standardization failed: {0}'.format(exc)) from exc

    primitive_lattice = np.asarray(raw['primitive_lattice'], dtype=float)
    primitive_positions = np.asarray(raw['primitive_positions'], dtype=float)
    primitive_types = np.asarray(raw['primitive_types'], dtype=int)
    primitive_atoms = [
        (
            str(elements.ELEMENTS[int(atomic_number)]),
            np.asarray(position, dtype=float).dot(primitive_lattice),
        )
        for position, atomic_number in zip(primitive_positions, primitive_types)
    ]
    path_segments = [[str(start), str(stop)] for start, stop in raw.get('path', [])]
    path_display = ' | '.join('{0}-{1}'.format(start, stop) for start, stop in path_segments)
    warning_rows = [
        {
            'category': type(item.message).__name__,
            'message': str(item.message),
        }
        for item in caught_warnings
        if not issubclass(item.category, DeprecationWarning)
    ]
    audit = {
        'status': 'available',
        'library': 'seekpath',
        'recipe': 'hpkot',
        'symprec_angstrom': float(symprec),
        'reference_distance_1_per_angstrom': (
            float(reference_distance) if reference_distance is not None else None
        ),
        'spacegroup_number': int(raw['spacegroup_number']),
        'spacegroup_international': str(raw['spacegroup_international']),
        'bravais_lattice': str(raw['bravais_lattice']),
        'bravais_lattice_extended': str(raw['bravais_lattice_extended']),
        'has_inversion_symmetry': bool(raw['has_inversion_symmetry']),
        'augmented_path': bool(raw['augmented_path']),
        'volume_original_wrt_primitive': float(raw['volume_original_wrt_prim']),
        'volume_original_wrt_conventional': float(raw['volume_original_wrt_conv']),
        'input_atom_count': int(len(atoms)),
        'primitive_atom_count': int(len(primitive_atoms)),
        'path': path_display,
        'path_segments': path_segments,
        'point_coords': {
            str(label): [float(value) for value in coordinates]
            for label, coordinates in raw.get('point_coords', {}).items()
        },
        'primitive_lattice_angstrom': primitive_lattice.tolist(),
        'primitive_positions_scaled': primitive_positions.tolist(),
        'primitive_types': primitive_types.tolist(),
        'primitive_transformation_matrix': np.asarray(
            raw['primitive_transformation_matrix'], dtype=float
        ).tolist(),
        'inverse_primitive_transformation_matrix': np.asarray(
            raw['inverse_primitive_transformation_matrix'], dtype=float
        ).tolist(),
        'rotation_matrix': np.asarray(raw['rotation_matrix'], dtype=float).tolist(),
        'warnings': warning_rows,
    }
    context: Dict[str, Any] = {
        'audit': audit,
        'primitive_lattice': primitive_lattice,
        'primitive_atoms': primitive_atoms,
        'path_definition': None,
    }
    if reference_distance is not None:
        scaled_kpoints = np.asarray(raw['explicit_kpoints_rel'], dtype=float)
        distances = np.asarray(raw['explicit_kpoints_linearcoord'], dtype=float)
        explicit_labels = [str(label) for label in raw['explicit_kpoints_labels']]
        audit['explicit_point_count'] = int(len(scaled_kpoints))
        context['path_definition'] = {
            'source': 'seekpath.get_explicit_k_path',
            'path': path_display,
            'path_segments': path_segments,
            'segment_indices': [list(map(int, segment)) for segment in raw['explicit_segments']],
            'scaled_kpoints': scaled_kpoints,
            'distances_1_per_angstrom': distances,
            'special_point_positions_1_per_angstrom': [
                float(distance)
                for distance, label in zip(distances, explicit_labels)
                if label
            ],
            'special_point_labels': [label for label in explicit_labels if label],
            'special_points_scaled': audit['point_coords'],
        }
    return context


def _structure_summary_from_components(
    lattice: np.ndarray,
    atoms: List[Tuple[str, np.ndarray]],
    metadata: Dict[str, Any],
    source_format: str,
    *,
    include_path_previews: bool,
    seekpath_symprec: float = 1e-5,
    seekpath_reference_distance: Optional[float] = None,
) -> Dict[str, Any]:
    symbols = [symbol for symbol, _coordinate in atoms]
    counts = Counter(symbols)
    ordered_symbols = list(dict.fromkeys(symbols))
    formula = ''.join(
        '{0}{1}'.format(symbol, counts[symbol] if counts[symbol] != 1 else '')
        for symbol in ordered_symbols
    )
    inverse_lattice = np.linalg.inv(lattice)
    atom_rows = []
    for index, (symbol, coordinate) in enumerate(atoms):
        fractional = np.asarray(coordinate, dtype=float).dot(inverse_lattice)
        wrapped_fractional = fractional - np.floor(fractional)
        selective_dynamics = metadata.get('selective_dynamics') or []
        atom_rows.append({
            'index': index,
            'element': symbol,
            'cartesian_angstrom': [float(value) for value in coordinate],
            'fractional': [float(value) for value in wrapped_fractional],
            'selective_dynamics': selective_dynamics[index] if index < len(selective_dynamics) else None,
        })
    summary = {
        'source_format': source_format,
        'parser': metadata.get('parser'),
        'formula': formula,
        'atom_count': len(atoms),
        'source_site_count': metadata.get('source_site_count', len(atoms)),
        'symmetry_expanded': bool(metadata.get('symmetry_expanded')),
        'space_group_number': metadata.get('space_group_number'),
        'space_group_symbol': metadata.get('space_group_symbol'),
        'composition': {symbol: counts[symbol] for symbol in ordered_symbols},
        'lattice_vectors_angstrom': [[float(value) for value in row] for row in lattice],
        'lattice_lengths_angstrom': [float(np.linalg.norm(row)) for row in lattice],
        'lattice_angles_degree': _lattice_angles(lattice),
        'cell_volume_angstrom3': abs(float(np.linalg.det(lattice))),
        'atoms': atom_rows,
    }
    if include_path_previews:
        summary['band_path_preview'] = _band_path_preview(lattice)
        try:
            summary['seekpath_preview'] = _seekpath_standardization(
                lattice,
                atoms,
                symprec=seekpath_symprec,
                reference_distance=seekpath_reference_distance,
            )['audit']
        except ValueError as exc:
            summary['seekpath_preview'] = {
                'status': 'unavailable',
                'reason': str(exc),
            }
    return summary


def _seekpath_changes_input_cell(audit: Dict[str, Any], input_atom_count: int) -> bool:
    return (
        abs(float(audit['volume_original_wrt_primitive']) - 1.0) > 1e-6
        or int(audit['primitive_atom_count']) != int(input_atom_count)
    )


def _seekpath_reduction_is_unsafe(
    task_spec: TaskSpec,
    audit: Dict[str, Any],
    input_atom_count: int,
) -> bool:
    return _seekpath_changes_input_cell(audit, input_atom_count) and (
        task_spec.system.charge != 0
        or task_spec.system.spin != 0
        or task_spec.method.restricted is False
    )


def periodic_structure_summary(
    source_text: Any,
    structure_format: Any,
    *,
    seekpath_symprec: float = 1e-5,
    seekpath_reference_distance: Optional[float] = None,
) -> Dict[str, Any]:
    try:
        normalized_symprec = float(seekpath_symprec)
    except (TypeError, ValueError) as exc:
        raise ValueError('SeeK-path symmetry tolerance must be numeric.') from exc
    if not (
        math.isfinite(normalized_symprec)
        and MIN_SEEKPATH_SYMPREC <= normalized_symprec <= MAX_SEEKPATH_SYMPREC
    ):
        raise ValueError(
            'SeeK-path symmetry tolerance must be between {0:g} and {1:g} Angstrom.'.format(
                MIN_SEEKPATH_SYMPREC,
                MAX_SEEKPATH_SYMPREC,
            )
        )
    normalized_reference_distance = None
    if seekpath_reference_distance is not None:
        try:
            normalized_reference_distance = float(seekpath_reference_distance)
        except (TypeError, ValueError) as exc:
            raise ValueError('SeeK-path reference distance must be numeric.') from exc
        if not (
            math.isfinite(normalized_reference_distance)
            and MIN_SEEKPATH_REFERENCE_DISTANCE
            <= normalized_reference_distance
            <= MAX_SEEKPATH_REFERENCE_DISTANCE
        ):
            raise ValueError(
                'SeeK-path reference distance must be between {0:g} and {1:g} 1/Angstrom.'.format(
                    MIN_SEEKPATH_REFERENCE_DISTANCE,
                    MAX_SEEKPATH_REFERENCE_DISTANCE,
                )
            )
    details = _parse_periodic_structure_details(source_text, structure_format)
    lattice = details['lattice']
    atoms = details['atoms']
    metadata = details['metadata']
    return _structure_summary_from_components(
        lattice,
        atoms,
        metadata,
        details['source_format'],
        include_path_previews=True,
        seekpath_symprec=normalized_symprec,
        seekpath_reference_distance=normalized_reference_distance,
    )


def normalized_periodic_poscar(structure_summary: Dict[str, Any]) -> str:
    atoms = structure_summary.get('atoms') if isinstance(structure_summary.get('atoms'), list) else []
    lattice = structure_summary.get('lattice_vectors_angstrom')
    if not atoms or not isinstance(lattice, list) or len(lattice) != 3:
        raise ValueError('A parsed periodic structure summary is required to write POSCAR output.')
    symbols = list(dict.fromkeys(str(atom.get('element')) for atom in atoms))
    grouped_atoms = [
        atom
        for symbol in symbols
        for atom in atoms
        if str(atom.get('element')) == symbol
    ]
    counts = [sum(1 for atom in atoms if str(atom.get('element')) == symbol) for symbol in symbols]
    lines = [
        '{0} normalized by PySCF Agent'.format(structure_summary.get('formula') or 'Periodic structure'),
        '1.0',
    ]
    lines.extend('  ' + ' '.join('{0:.16g}'.format(float(value)) for value in row) for row in lattice)
    lines.append('  ' + ' '.join(symbols))
    lines.append('  ' + ' '.join(str(value) for value in counts))
    lines.append('Direct')
    for atom in grouped_atoms:
        fractional = atom.get('fractional') or []
        lines.append('  ' + ' '.join('{0:.16g}'.format(float(value)) for value in fractional))
    return '\n'.join(lines) + '\n'


def _basis_pseudo_availability_errors(
    atoms: List[Tuple[str, np.ndarray]],
    basis_name: str,
    pseudo_name: str,
) -> List[str]:
    from pyscf.pbc.gto import basis as pbc_basis  # pylint: disable=import-outside-toplevel
    from pyscf.pbc.gto import pseudo as pbc_pseudo  # pylint: disable=import-outside-toplevel

    symbols = list(dict.fromkeys(symbol for symbol, _coordinate in atoms))
    missing_basis = []
    missing_pseudo = []
    for symbol in symbols:
        try:
            pbc_basis.load(basis_name, symbol)
        except Exception:  # PySCF raises different lookup errors across data formats.
            missing_basis.append(symbol)
        try:
            pbc_pseudo.load(pseudo_name, symbol)
        except Exception:
            missing_pseudo.append(symbol)
    errors = []
    if missing_basis:
        compatible_basis_sets = compatible_periodic_basis_sets(symbols)
        message = 'Periodic basis {0} is unavailable for element(s): {1}.'.format(
            basis_name,
            ', '.join(missing_basis),
        )
        if compatible_basis_sets:
            message += ' Verified compatible registered basis replacements for all structure elements: {0}.'.format(
                ', '.join(compatible_basis_sets)
            )
        else:
            message += ' No compatible registered basis replacement was found for all structure elements.'
        errors.append(message)
    if missing_pseudo:
        compatible_pseudopotentials = compatible_periodic_pseudopotentials(symbols)
        message = 'Periodic pseudopotential {0} is unavailable for element(s): {1}.'.format(
            pseudo_name,
            ', '.join(missing_pseudo),
        )
        if compatible_pseudopotentials:
            message += ' Verified compatible registered pseudopotential replacements for all structure elements: {0}.'.format(
                ', '.join(compatible_pseudopotentials)
            )
        else:
            message += ' No compatible registered pseudopotential replacement was found for all structure elements.'
        errors.append(message)
    return errors


def _compatible_periodic_resources(
    elements: Sequence[str],
    resource_names: Sequence[str],
    loader: Any,
) -> List[str]:
    symbols = list(dict.fromkeys(str(element).strip() for element in elements if str(element).strip()))
    compatible = []
    for resource_name in resource_names:
        try:
            for symbol in symbols:
                loader(resource_name, symbol)
        except Exception:  # PySCF lookup exceptions vary by resource data format.
            continue
        compatible.append(resource_name)
    return compatible


def compatible_periodic_basis_sets(elements: Sequence[str]) -> List[str]:
    """Return registered bases that PySCF can load for every requested element."""
    from pyscf.pbc.gto import basis as pbc_basis  # pylint: disable=import-outside-toplevel

    return _compatible_periodic_resources(elements, SUPPORTED_PERIODIC_BASIS_SETS, pbc_basis.load)


def compatible_periodic_pseudopotentials(elements: Sequence[str]) -> List[str]:
    """Return registered pseudopotentials available for every requested element."""
    from pyscf.pbc.gto import pseudo as pbc_pseudo  # pylint: disable=import-outside-toplevel

    return _compatible_periodic_resources(elements, SUPPORTED_PERIODIC_PSEUDOPOTENTIALS, pbc_pseudo.load)


def validate_periodic_task(task_spec: TaskSpec) -> List[str]:
    errors: List[str] = []
    periodic = task_spec.periodic
    solver_name = str(task_spec.solver.name or '').strip().lower().replace('-', '_')
    parsed_atoms: List[Tuple[str, np.ndarray]] = []
    parsed_lattice: Optional[np.ndarray] = None
    structure_format = _normalize_structure_format(periodic.structure_format)
    periodic.structure_format = structure_format
    if structure_format not in SUPPORTED_STRUCTURE_FORMATS:
        errors.append('Periodic structure format must be POSCAR or CIF.')
    elif not periodic.structure_text or not periodic.structure_text.strip():
        errors.append('Periodic structure text is required; upload or paste a POSCAR/CIF file.')
    else:
        try:
            parsed_lattice, parsed_atoms = parse_periodic_structure(periodic.structure_text, structure_format)
        except ValueError as exc:
            errors.append(str(exc))

    if periodic.dimension != 3:
        errors.append('Initial periodic support requires dimension=3.')
    if periodic.basis not in SUPPORTED_PERIODIC_BASIS_SETS:
        errors.append('Unsupported periodic basis: {0}.'.format(periodic.basis))
    if periodic.pseudo not in SUPPORTED_PERIODIC_PSEUDOPOTENTIALS:
        errors.append('Unsupported periodic pseudopotential: {0}.'.format(periodic.pseudo))
    if (
        parsed_atoms
        and periodic.basis in SUPPORTED_PERIODIC_BASIS_SETS
        and periodic.pseudo in SUPPORTED_PERIODIC_PSEUDOPOTENTIALS
    ):
        errors.extend(_basis_pseudo_availability_errors(parsed_atoms, periodic.basis, periodic.pseudo))
    if task_spec.method.name not in SUPPORTED_PERIODIC_METHODS:
        errors.append('Unsupported periodic method: {0}.'.format(task_spec.method.name))
    if solver_name == 'hf_dmft':
        if task_spec.method.name != 'hf':
            errors.append('HF+DMFT requires periodic method=hf for the reference calculation.')
        if not task_spec.embedding.enabled:
            errors.append('HF+DMFT requires an enabled embedding specification.')
        elif task_spec.embedding.provider != 'fcdmft':
            errors.append('HF+DMFT requires embedding.provider=fcdmft.')
        try:
            from ...providers.fcdmft import (
                normalize_hf_dmft_options,
                validate_hf_dmft_preparation_request,
            )

            task_spec.solver.options = normalize_hf_dmft_options(task_spec.solver.options)
            if not task_spec.embedding.approved:
                validate_hf_dmft_preparation_request(task_spec)
        except ValueError as exc:
            errors.append(str(exc))
        if 'excited_states' in task_spec.analysis.outputs:
            errors.append('The current HF+DMFT contract does not support low-lying excited roots.')
    elif solver_name == 'gw':
        try:
            from ...providers.fcdmft import (
                normalize_periodic_gw_options,
                validate_periodic_gw_reference,
            )

            task_spec.solver.options = normalize_periodic_gw_options(task_spec.solver.options)
            validate_periodic_gw_reference(task_spec)
        except ValueError as exc:
            errors.append(str(exc))
        if 'excited_states' in task_spec.analysis.outputs:
            errors.append(
                'Periodic GW provides quasiparticle energies, not a low-lying many-body root calculation.'
            )
    elif solver_name == 'gw_dmft':
        if task_spec.method.name != 'dft':
            errors.append('GW+DMFT requires periodic method=dft for the reference calculation.')
        if not task_spec.embedding.enabled:
            errors.append('GW+DMFT requires an enabled embedding specification.')
        elif task_spec.embedding.provider != 'fcdmft':
            errors.append('GW+DMFT requires embedding.provider=fcdmft.')
        try:
            from ...providers.fcdmft import (
                normalize_gw_dmft_options,
                validate_fcdmft_preparation_request,
                validate_periodic_gw_reference,
            )

            task_spec.solver.options = normalize_gw_dmft_options(task_spec.solver.options)
            validate_periodic_gw_reference(task_spec, for_dmft=True)
            if not task_spec.embedding.approved:
                validate_fcdmft_preparation_request(task_spec)
        except ValueError as exc:
            errors.append(str(exc))
        if 'excited_states' in task_spec.analysis.outputs:
            errors.append('The current GW+DMFT contract does not support low-lying excited roots.')
    elif task_spec.embedding.enabled and task_spec.embedding.provider == 'fcdmft':
        errors.append('embedding.provider=fcdmft requires solver=hf_dmft or gw_dmft.')
    if task_spec.job.name not in SUPPORTED_PERIODIC_JOBS:
        errors.append('Unsupported periodic job: {0}.'.format(task_spec.job.name))
    if task_spec.method.name == 'dft':
        if not task_spec.method.xc:
            errors.append('Periodic DFT requires an xc functional.')
        elif task_spec.method.xc not in SUPPORTED_PERIODIC_XC_FUNCTIONALS:
            errors.append('Unsupported periodic xc functional: {0}.'.format(task_spec.method.xc))
    else:
        task_spec.method.xc = None
    if len(periodic.kmesh) != 3 or any(value <= 0 for value in periodic.kmesh):
        errors.append('Periodic kmesh must contain three positive integers.')
    elif math.prod(periodic.kmesh) > MAX_PERIODIC_KPOINTS:
        errors.append('Periodic kmesh exceeds the initial limit of {0} k-points.'.format(MAX_PERIODIC_KPOINTS))
    if periodic.kpoint_scheme not in SUPPORTED_PERIODIC_KPOINT_SCHEMES:
        errors.append('Unsupported periodic k-point scheme: {0}.'.format(periodic.kpoint_scheme))
    if len(periodic.kpoint_shift) != 3 or any(not math.isfinite(value) for value in periodic.kpoint_shift):
        errors.append('Periodic k-point shift must contain three finite numeric values.')
    elif any(abs(value) > 0.5 for value in periodic.kpoint_shift):
        errors.append('Periodic k-point shift components must lie between -0.5 and 0.5 mesh steps.')
    if not math.isfinite(periodic.precision) or periodic.precision <= 0:
        errors.append('Periodic integral precision must be a positive finite value.')
    if periodic.ke_cutoff is not None and (
        not math.isfinite(periodic.ke_cutoff) or periodic.ke_cutoff <= 0
    ):
        errors.append('Periodic plane-wave auxiliary cutoff must be a positive finite value in Hartree.')
    if periodic.fft_mesh is not None:
        if len(periodic.fft_mesh) != 3 or any(value <= 0 for value in periodic.fft_mesh):
            errors.append('Periodic FFT mesh must contain three positive integers.')
        elif math.prod(periodic.fft_mesh) > MAX_PERIODIC_FFT_GRID_POINTS:
            errors.append(
                'Periodic FFT mesh exceeds the {0}-point safety limit.'.format(MAX_PERIODIC_FFT_GRID_POINTS)
            )
    if periodic.ke_cutoff is not None and periodic.fft_mesh is not None:
        errors.append('Set either periodic ke_cutoff or fft_mesh, not both.')
    if periodic.density_fitting_method not in SUPPORTED_PERIODIC_DENSITY_FITTING_METHODS:
        errors.append(
            'Unsupported periodic density-fitting method: {0}.'.format(periodic.density_fitting_method)
        )
    if periodic.density_fitting_auxbasis and periodic.density_fitting_method not in ('gdf', 'mdf'):
        errors.append('A periodic auxiliary Gaussian basis is only valid with GDF or MDF.')
    if periodic.exxdiv not in SUPPORTED_PERIODIC_EXXDIV_OPTIONS:
        errors.append('Unsupported periodic exact-exchange divergence treatment: {0}.'.format(periodic.exxdiv))
    if (
        periodic.exxdiv in ('vcut_sph', 'vcut_ws')
        and periodic.density_fitting_method != 'fft'
    ):
        errors.append('vcut_sph and vcut_ws require FFTDF in the current periodic contract.')
    if periodic.smearing_method not in SUPPORTED_PERIODIC_SMEARING_METHODS:
        errors.append('Unsupported periodic smearing method: {0}.'.format(periodic.smearing_method))
    if periodic.smearing_method == 'none':
        periodic.smearing_sigma = None
    elif periodic.smearing_sigma is None or (
        not math.isfinite(periodic.smearing_sigma) or periodic.smearing_sigma <= 0
    ):
        errors.append('Periodic smearing requires a positive finite sigma in Hartree.')
    outputs = list(dict.fromkeys(task_spec.analysis.outputs))
    mode_aliases = {
        'automatic': 'auto',
        'standardized': 'seekpath',
        'seek_path': 'seekpath',
        'labels': 'custom',
        'custom_labels': 'custom',
        'coordinates': 'explicit',
        'custom_coordinates': 'explicit',
    }
    periodic.band_path_mode = mode_aliases.get(periodic.band_path_mode, periodic.band_path_mode)
    if periodic.band_path_mode not in SUPPORTED_PERIODIC_BAND_PATH_MODES:
        errors.append(
            'Unsupported periodic band-path mode: {0}. Use one of: {1}.'.format(
                periodic.band_path_mode,
                ', '.join(SUPPORTED_PERIODIC_BAND_PATH_MODES),
            )
        )
    if 'band_structure' in outputs:
        if periodic.band_path_mode == 'seekpath':
            periodic.band_path = None
            periodic.band_path_special_points = {}
            seekpath_numerics_valid = True
            if not (
                math.isfinite(periodic.band_path_reference_distance)
                and MIN_SEEKPATH_REFERENCE_DISTANCE
                <= periodic.band_path_reference_distance
                <= MAX_SEEKPATH_REFERENCE_DISTANCE
            ):
                errors.append(
                    'SeeK-path reference distance must be between {0:g} and {1:g} 1/Angstrom.'.format(
                        MIN_SEEKPATH_REFERENCE_DISTANCE,
                        MAX_SEEKPATH_REFERENCE_DISTANCE,
                    )
                )
                seekpath_numerics_valid = False
            if not (
                math.isfinite(periodic.band_path_symprec)
                and MIN_SEEKPATH_SYMPREC <= periodic.band_path_symprec <= MAX_SEEKPATH_SYMPREC
            ):
                errors.append(
                    'SeeK-path symmetry tolerance must be between {0:g} and {1:g} Angstrom.'.format(
                        MIN_SEEKPATH_SYMPREC,
                        MAX_SEEKPATH_SYMPREC,
                    )
                )
                seekpath_numerics_valid = False
            if seekpath_numerics_valid and parsed_lattice is not None and parsed_atoms:
                try:
                    seekpath_context = _seekpath_standardization(
                        parsed_lattice,
                        parsed_atoms,
                        symprec=periodic.band_path_symprec,
                        reference_distance=periodic.band_path_reference_distance,
                    )
                    path_definition = seekpath_context['path_definition']
                    point_count = len(path_definition['scaled_kpoints'])
                    if point_count > MAX_PERIODIC_BAND_PATH_POINTS:
                        errors.append(
                            'SeeK-path generated {0} points, exceeding the {1}-point safety limit; '
                            'increase periodic.band_path_reference_distance.'.format(
                                point_count,
                                MAX_PERIODIC_BAND_PATH_POINTS,
                            )
                        )
                    if _seekpath_reduction_is_unsafe(
                        task_spec,
                        seekpath_context['audit'],
                        len(parsed_atoms),
                    ):
                        errors.append(
                            'SeeK-path reduced the input to a different primitive cell, but the input uses nonzero '
                            'total charge/spin or an unrestricted reference whose electron or magnetic-cell semantics '
                            'cannot be inferred safely. Supply the primitive cell explicitly or use auto/custom '
                            'band-path mode to preserve the original charged or magnetic cell.'
                        )
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append('Periodic SeeK-path is invalid: {0}'.format(exc))
        elif periodic.band_path_mode in SUPPORTED_PERIODIC_BAND_PATH_MODES:
            if not 2 <= periodic.band_path_npoints <= MAX_PERIODIC_BAND_PATH_POINTS:
                errors.append(
                    'Periodic band-path sampling requires between 2 and {0} total points.'.format(
                        MAX_PERIODIC_BAND_PATH_POINTS
                    )
                )
            elif parsed_lattice is not None:
                try:
                    if periodic.band_path_mode != 'auto':
                        periodic.band_path = _normalize_band_path(periodic.band_path)
                    if periodic.band_path_mode == 'explicit':
                        normalized_points = _normalized_band_path_special_points(
                            periodic.band_path_special_points
                        )
                        periodic.band_path_special_points = {
                            label: tuple(float(value) for value in coordinates)
                            for label, coordinates in normalized_points.items()
                        }
                    _ase_band_path(
                        parsed_lattice,
                        mode=periodic.band_path_mode,
                        path=periodic.band_path,
                        special_points=periodic.band_path_special_points,
                        npoints=periodic.band_path_npoints,
                    )
                except (KeyError, TypeError, ValueError) as exc:
                    errors.append('Periodic band path is invalid: {0}'.format(exc))
    unsupported_outputs = [output for output in outputs if output not in SUPPORTED_PERIODIC_ANALYSIS]
    if unsupported_outputs:
        errors.append('Unsupported periodic outputs: {0}.'.format(', '.join(unsupported_outputs)))
    task_spec.analysis.outputs = [output for output in outputs if output in SUPPORTED_PERIODIC_ANALYSIS]
    return errors


def build_periodic_cell(
    task_spec: TaskSpec,
    stdout: io.StringIO = None,
) -> Tuple[Any, Dict[str, Any], Dict[str, Any]]:
    from pyscf.pbc import gto as pbc_gto  # pylint: disable=import-outside-toplevel

    periodic = task_spec.periodic
    use_seekpath = (
        periodic.band_path_mode == 'seekpath'
        and 'band_structure' in task_spec.analysis.outputs
    )
    parsed_structure = _parse_periodic_structure_details(
        periodic.structure_text,
        periodic.structure_format,
    )
    lattice = parsed_structure['lattice']
    atoms = parsed_structure['atoms']
    input_summary = _structure_summary_from_components(
        lattice,
        atoms,
        parsed_structure['metadata'],
        parsed_structure['source_format'],
        include_path_previews=False,
    )
    context: Dict[str, Any] = {
        'input_structure': None,
        'seekpath_standardization': None,
        'path_definition': None,
    }
    summary = input_summary
    if use_seekpath:
        seekpath_context = _seekpath_standardization(
            lattice,
            atoms,
            symprec=periodic.band_path_symprec,
            reference_distance=periodic.band_path_reference_distance,
        )
        audit = seekpath_context['audit']
        if _seekpath_reduction_is_unsafe(task_spec, audit, len(atoms)):
            raise ValueError(
                'SeeK-path primitive-cell standardization cannot safely preserve this input-cell charge, spin, '
                'or unrestricted magnetic reference.'
            )
        lattice = seekpath_context['primitive_lattice']
        atoms = seekpath_context['primitive_atoms']
        summary = _structure_summary_from_components(
            lattice,
            atoms,
            {
                'parser': 'seekpath_hpkot',
                'source_site_count': len(atoms),
                'symmetry_expanded': False,
                'space_group_number': audit['spacegroup_number'],
                'space_group_symbol': audit['spacegroup_international'],
                'selective_dynamics': [None] * len(atoms),
            },
            'seekpath_primitive',
            include_path_previews=False,
        )
        summary['cell_role'] = 'seekpath_standardized_primitive'
        input_summary['cell_role'] = 'user_input'
        context = {
            'input_structure': input_summary,
            'seekpath_standardization': audit,
            'path_definition': seekpath_context['path_definition'],
        }
    else:
        summary['cell_role'] = 'user_input'
    cell = pbc_gto.Cell()
    if stdout is not None:
        cell.stdout = stdout
    cell.a = lattice
    cell.atom = [(symbol, tuple(float(value) for value in coordinate)) for symbol, coordinate in atoms]
    cell.unit = 'Angstrom'
    cell.basis = periodic.basis
    cell.pseudo = periodic.pseudo
    cell.dimension = periodic.dimension
    cell.charge = task_spec.system.charge
    cell.spin = task_spec.system.spin
    cell.verbose = task_spec.runtime.verbose
    cell.precision = periodic.precision
    if periodic.ke_cutoff is not None:
        cell.ke_cutoff = periodic.ke_cutoff
    if periodic.fft_mesh is not None:
        cell.mesh = np.asarray(periodic.fft_mesh, dtype=int)
    cell.build(dump_input=False, parse_arg=False)
    return cell, summary, context


def _periodic_kpoints(cell: Any, task_spec: TaskSpec) -> Tuple[Any, np.ndarray, np.ndarray, bool]:
    periodic = task_spec.periodic
    scaled_axes = []
    for count, shift in zip(periodic.kmesh, periodic.kpoint_shift):
        if periodic.kpoint_scheme == 'monkhorst_pack':
            axis = (np.arange(count, dtype=float) + 0.5) / count - 0.5
        else:
            axis = np.arange(count, dtype=float) / count
            axis[axis >= 0.5] -= 1.0
        axis += float(shift) / count
        axis[axis >= 0.5] -= 1.0
        axis[axis < -0.5] += 1.0
        scaled_axes.append(axis)
    scaled_kpoints = np.asarray(list(product(*scaled_axes)), dtype=float)
    absolute_kpoints = np.asarray(cell.get_abs_kpts(scaled_kpoints), dtype=float)
    gamma_point = len(scaled_kpoints) == 1 and bool(np.allclose(scaled_kpoints[0], 0.0, atol=1e-12))
    return (None if gamma_point else absolute_kpoints), scaled_kpoints, absolute_kpoints, gamma_point


def _configure_periodic_density_fitting(mean_field: Any, cell: Any, kpts: Any, task_spec: TaskSpec) -> Any:
    from pyscf.pbc import df as pbc_df  # pylint: disable=import-outside-toplevel

    periodic = task_spec.periodic
    constructors = {
        'fft': pbc_df.FFTDF,
        'gdf': pbc_df.GDF,
        'mdf': pbc_df.MDF,
        'aft': pbc_df.AFTDF,
    }
    with_df = constructors[periodic.density_fitting_method](cell, kpts=kpts)
    if periodic.density_fitting_auxbasis and hasattr(with_df, 'auxbasis'):
        with_df.auxbasis = periodic.density_fitting_auxbasis
    mean_field.with_df = with_df
    return mean_field


def _as_kpoint_arrays(values: Any) -> List[np.ndarray]:
    array = np.asarray(values, dtype=float)
    if array.ndim == 1:
        return [array]
    if array.ndim == 2:
        return [np.asarray(row, dtype=float) for row in array]
    raise ValueError('Periodic orbital data must have one band axis and at most one k-point axis.')


def _orbital_channels(mo_energy: Any, mo_occ: Any, restricted: bool) -> List[Dict[str, Any]]:
    if restricted:
        channel_payloads = [('restricted', mo_energy, mo_occ)]
    else:
        if len(mo_energy) != 2 or len(mo_occ) != 2:
            raise ValueError('Unrestricted periodic orbital data must contain alpha and beta channels.')
        channel_payloads = [
            ('alpha', mo_energy[0], mo_occ[0]),
            ('beta', mo_energy[1], mo_occ[1]),
        ]
    channels = []
    for spin, energy_values, occupation_values in channel_payloads:
        energies = _as_kpoint_arrays(energy_values)
        occupations = _as_kpoint_arrays(occupation_values)
        if len(energies) != len(occupations):
            raise ValueError('Periodic orbital energies and occupations have different k-point counts.')
        for energy_row, occupation_row in zip(energies, occupations):
            if energy_row.shape != occupation_row.shape:
                raise ValueError('Periodic orbital energies and occupations have different band counts.')
        channels.append({
            'spin': spin,
            'energies': energies,
            'occupations': occupations,
            'maximum_occupation': 2.0 if restricted else 1.0,
        })
    return channels


def _band_edge_record(energy: float, spin: str, kpoint_index: int, band_index: int) -> Dict[str, Any]:
    return {
        'energy': float(energy),
        'spin': spin,
        'kpoint_index': int(kpoint_index),
        'band_index': int(band_index),
    }


def _periodic_electronic_structure(
    mean_field: Any,
    *,
    restricted: bool,
    scaled_kpoints: np.ndarray,
    absolute_kpoints: np.ndarray,
) -> Dict[str, Any]:
    channels = _orbital_channels(mean_field.mo_energy, mean_field.mo_occ, restricted)
    valence_record = None
    conduction_record = None
    direct_gap_record = None
    has_fractional_occupations = False
    for channel in channels:
        spin = channel['spin']
        maximum_occupation = float(channel['maximum_occupation'])
        occupation_boundary = 0.5 * maximum_occupation
        for kpoint_index, (energies, occupations) in enumerate(zip(
            channel['energies'],
            channel['occupations'],
        )):
            occupied_indices = np.where(occupations > occupation_boundary)[0]
            virtual_indices = np.where(occupations <= occupation_boundary)[0]
            has_fractional_occupations = has_fractional_occupations or bool(np.any(
                (occupations > 1e-4) & (occupations < maximum_occupation - 1e-4)
            ))
            local_valence = None
            local_conduction = None
            if occupied_indices.size:
                band_index = int(occupied_indices[np.argmax(energies[occupied_indices])])
                local_valence = _band_edge_record(energies[band_index], spin, kpoint_index, band_index)
                if valence_record is None or local_valence['energy'] > valence_record['energy']:
                    valence_record = local_valence
            if virtual_indices.size:
                band_index = int(virtual_indices[np.argmin(energies[virtual_indices])])
                local_conduction = _band_edge_record(energies[band_index], spin, kpoint_index, band_index)
                if conduction_record is None or local_conduction['energy'] < conduction_record['energy']:
                    conduction_record = local_conduction
            if local_valence is not None and local_conduction is not None:
                candidate_gap = max(0.0, local_conduction['energy'] - local_valence['energy'])
                if direct_gap_record is None or candidate_gap < direct_gap_record['gap']:
                    direct_gap_record = {
                        'gap': float(candidate_gap),
                        'spin': spin,
                        'kpoint_index': int(kpoint_index),
                        'valence_band_index': local_valence['band_index'],
                        'conduction_band_index': local_conduction['band_index'],
                    }

    fundamental_gap = None
    if valence_record is not None and conduction_record is not None:
        fundamental_gap = max(0.0, conduction_record['energy'] - valence_record['energy'])
    is_metal = bool(
        has_fractional_occupations
        or (fundamental_gap is not None and fundamental_gap <= 1e-6)
    )

    fermi_by_spin: Dict[str, float] = {}
    fermi_source = 'vbm_fallback'
    try:
        raw_fermi = mean_field.get_fermi()
        fermi_values = np.asarray(raw_fermi, dtype=float).reshape(-1)
        if restricted and fermi_values.size:
            fermi_by_spin['restricted'] = float(fermi_values[0])
        elif not restricted:
            for spin, value in zip(('alpha', 'beta'), fermi_values):
                fermi_by_spin[spin] = float(value)
        fermi_source = (
            'pyscf_get_fermi_smearing_chemical_potential'
            if getattr(mean_field, 'smearing_method', None)
            else 'pyscf_get_fermi_vbm_convention'
        )
    except Exception:
        if valence_record is not None:
            fermi_by_spin[valence_record['spin']] = float(valence_record['energy'])
    fermi_energy = max(fermi_by_spin.values()) if fermi_by_spin else (
        float(valence_record['energy']) if valence_record is not None else None
    )

    spectrum_channels = []
    for channel in channels:
        spectrum_channels.append({
            'spin': channel['spin'],
            'maximum_occupation': channel['maximum_occupation'],
            'energies': [[float(value) for value in row] for row in channel['energies']],
            'occupations': [[float(value) for value in row] for row in channel['occupations']],
        })
    direct_gap = direct_gap_record['gap'] if direct_gap_record is not None else None
    gap_type = None
    if fundamental_gap is not None and valence_record is not None and conduction_record is not None:
        gap_type = (
            'direct'
            if valence_record['kpoint_index'] == conduction_record['kpoint_index']
            and valence_record['spin'] == conduction_record['spin']
            else 'indirect'
        )

    def attach_kpoint(record: Optional[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
        if record is None:
            return None
        enriched = dict(record)
        kpoint_index = int(enriched['kpoint_index'])
        enriched['kpoint_scaled'] = [float(value) for value in scaled_kpoints[kpoint_index]]
        enriched['kpoint_abs_1_per_bohr'] = [float(value) for value in absolute_kpoints[kpoint_index]]
        return enriched

    valence_record = attach_kpoint(valence_record)
    conduction_record = attach_kpoint(conduction_record)
    direct_gap_record = attach_kpoint(direct_gap_record)
    return {
        'valence_band_max': valence_record['energy'] if valence_record is not None else None,
        'conduction_band_min': conduction_record['energy'] if conduction_record is not None else None,
        'band_gap': fundamental_gap,
        'direct_gap': direct_gap,
        'gap_type': gap_type,
        'is_metal': is_metal,
        'fermi_energy': fermi_energy,
        'fermi_energy_by_spin': fermi_by_spin,
        'fermi_energy_source': fermi_source,
        'band_edges': {
            'energy_unit': 'Hartree',
            'vbm': valence_record,
            'cbm': conduction_record,
            'direct_gap': direct_gap_record,
        },
        'periodic_orbitals': {
            'energy_unit': 'Hartree',
            'kpoints_scaled': [[float(value) for value in row] for row in scaled_kpoints],
            'kpoints_abs_1_per_bohr': [[float(value) for value in row] for row in absolute_kpoints],
            'channels': spectrum_channels,
        },
    }


def _periodic_reference_label(method: str, restricted: bool, gamma_point: bool) -> str:
    prefix = '' if gamma_point else 'k'
    if method == 'dft':
        return '{0}{1}'.format(prefix, 'rks' if restricted else 'uks')
    return '{0}{1}'.format(prefix, 'rhf' if restricted else 'uhf')


def _effective_periodic_restricted(task_spec: TaskSpec) -> bool:
    if task_spec.method.restricted is None:
        return task_spec.system.spin == 0
    return bool(task_spec.method.restricted)


def _band_energy_channels(values: Any, restricted: bool) -> List[Tuple[str, np.ndarray]]:
    if restricted:
        raw_channels = [('restricted', values)]
    elif isinstance(values, (list, tuple)) and len(values) == 2:
        raw_channels = [('alpha', values[0]), ('beta', values[1])]
    else:
        array = np.asarray(values, dtype=float)
        if array.ndim < 2 or array.shape[0] != 2:
            raise ValueError('Unrestricted periodic band energies must contain alpha and beta channels.')
        raw_channels = [('alpha', array[0]), ('beta', array[1])]

    channels: List[Tuple[str, np.ndarray]] = []
    for spin, raw_values in raw_channels:
        array = np.asarray(raw_values, dtype=float)
        if array.ndim == 1:
            array = array.reshape(1, -1)
        if array.ndim != 2:
            raise ValueError('Periodic band energies must have k-point and band axes.')
        channels.append((spin, array))
    return channels


def _periodic_band_structure(
    mean_field: Any,
    cell: Any,
    *,
    restricted: bool,
    mode: str,
    requested_path: Optional[str],
    requested_special_points: Dict[str, Sequence[float]],
    npoints: int,
    path_definition: Optional[Dict[str, Any]],
    fermi_energy: Optional[float],
    fermi_by_spin: Dict[str, float],
) -> Dict[str, Any]:
    from pyscf.data import nist  # pylint: disable=import-outside-toplevel
    from pyscf.pbc.tools import pyscf_ase  # pylint: disable=import-outside-toplevel

    path_segments: List[List[str]] = []
    segment_indices: List[List[int]] = []
    if mode == 'seekpath':
        if not isinstance(path_definition, dict):
            raise ValueError('SeeK-path execution requires the validated explicit path definition.')
        scaled_kpoints = np.asarray(path_definition['scaled_kpoints'], dtype=float)
        distances = np.asarray(path_definition['distances_1_per_angstrom'], dtype=float)
        label_positions = path_definition['special_point_positions_1_per_angstrom']
        labels = path_definition['special_point_labels']
        special_points = path_definition['special_points_scaled']
        path_value = str(path_definition['path'])
        path_source = str(path_definition['source'])
        path_segments = [list(segment) for segment in path_definition.get('path_segments', [])]
        segment_indices = [list(map(int, segment)) for segment in path_definition.get('segment_indices', [])]
    elif mode == 'auto':
        band_path = pyscf_ase.bandpath(cell, npoints=npoints)
        path_source = 'pyscf.pbc.tools.pyscf_ase.bandpath'
        scaled_kpoints = np.asarray(band_path.kpts, dtype=float)
        distances, label_positions, labels = band_path.get_linear_kpoint_axis()
        special_points = band_path.special_points
        path_value = str(band_path.path)
    else:
        lattice_angstrom = np.asarray(cell.lattice_vectors(), dtype=float) * float(nist.BOHR)
        band_path = _ase_band_path(
            lattice_angstrom,
            mode=mode,
            path=requested_path,
            special_points=requested_special_points,
            npoints=npoints,
        )
        path_source = 'ase.cell.Cell.bandpath'
        scaled_kpoints = np.asarray(band_path.kpts, dtype=float)
        distances, label_positions, labels = band_path.get_linear_kpoint_axis()
        special_points = band_path.special_points
        path_value = str(band_path.path)
    absolute_kpoints = np.asarray(cell.get_abs_kpts(scaled_kpoints), dtype=float)
    band_energies = mean_field.get_bands(absolute_kpoints)[0]

    channels = []
    for spin, energies in _band_energy_channels(band_energies, restricted):
        reference = fermi_by_spin.get(spin, fermi_energy)
        relative_energies = None
        if reference is not None:
            relative_energies = (energies - float(reference)) * float(nist.HARTREE2EV)
        channels.append({
            'spin': spin,
            'fermi_energy_hartree': float(reference) if reference is not None else None,
            'energies_hartree': [[float(value) for value in row] for row in energies],
            'energies_relative_to_fermi_ev': (
                [[float(value) for value in row] for row in relative_energies]
                if relative_energies is not None
                else None
            ),
        })

    return {
        'status': 'completed',
        'source': '{0} + mean_field.get_bands'.format(path_source),
        'path_mode': mode,
        'requested_path': requested_path if mode in ('custom', 'explicit') else None,
        'requested_special_points_scaled': (
            {
                str(label): [float(value) for value in coordinates]
                for label, coordinates in requested_special_points.items()
            }
            if mode == 'explicit'
            else {}
        ),
        'path': path_value,
        'path_segments': path_segments,
        'segment_indices': segment_indices,
        'point_count': int(len(scaled_kpoints)),
        'energy_reference': 'mean_field_fermi_energy',
        'absolute_energy_unit': 'Hartree',
        'relative_energy_unit': 'eV',
        'distance_unit': '1/Angstrom',
        'kpoints_scaled': [[float(value) for value in row] for row in scaled_kpoints],
        'kpoints_abs_1_per_bohr': [[float(value) for value in row] for row in absolute_kpoints],
        'path_distances_1_per_angstrom': [float(value) for value in distances],
        'special_point_positions_1_per_angstrom': [float(value) for value in label_positions],
        'special_point_labels': [str(label) for label in labels],
        'special_points_scaled': {
            str(label): [float(value) for value in coordinates]
            for label, coordinates in special_points.items()
        },
        'channels': channels,
    }


def run_periodic_task(
    task_spec: TaskSpec,
    *,
    deferred_modules: Tuple[str, ...] = (),
) -> Dict[str, Any]:
    from pyscf.pbc import dft as pbc_dft  # pylint: disable=import-outside-toplevel
    from pyscf.pbc import scf as pbc_scf  # pylint: disable=import-outside-toplevel

    scf_stdout = io.StringIO()
    cell, structure_summary, periodic_context = build_periodic_cell(task_spec, stdout=scf_stdout)
    kmesh = tuple(int(value) for value in task_spec.periodic.kmesh)
    restricted = _effective_periodic_restricted(task_spec)
    kpts, scaled_kpoints, absolute_kpoints, gamma_point = _periodic_kpoints(cell, task_spec)

    if task_spec.method.name == 'hf':
        if gamma_point:
            mean_field = pbc_scf.RHF(cell) if restricted else pbc_scf.UHF(cell)
        else:
            mean_field = pbc_scf.KRHF(cell, kpts=kpts) if restricted else pbc_scf.KUHF(cell, kpts=kpts)
    else:
        if gamma_point:
            mean_field = pbc_dft.RKS(cell) if restricted else pbc_dft.UKS(cell)
        else:
            mean_field = pbc_dft.KRKS(cell, kpts=kpts) if restricted else pbc_dft.KUKS(cell, kpts=kpts)
        mean_field.xc = task_spec.method.xc

    mean_field = _configure_periodic_density_fitting(mean_field, cell, kpts, task_spec)
    mean_field.exxdiv = None if task_spec.periodic.exxdiv == 'none' else task_spec.periodic.exxdiv
    if task_spec.periodic.smearing_method != 'none':
        mean_field = mean_field.smearing(
            sigma=task_spec.periodic.smearing_sigma,
            method=task_spec.periodic.smearing_method,
            fix_spin=task_spec.periodic.smearing_fix_spin,
        )
    mean_field.stdout = scf_stdout
    mean_field.max_cycle = task_spec.runtime.max_cycle
    if task_spec.runtime.conv_tol is not None:
        mean_field.conv_tol = task_spec.runtime.conv_tol
    energy = float(mean_field.kernel())
    electronic_structure = _periodic_electronic_structure(
        mean_field,
        restricted=restricted,
        scaled_kpoints=scaled_kpoints,
        absolute_kpoints=absolute_kpoints,
    )
    kpoint_count = len(scaled_kpoints)
    kpoint_weights = [1.0 / kpoint_count] * kpoint_count
    density_fitting = {
        'method': task_spec.periodic.density_fitting_method,
        'implementation': type(mean_field.with_df).__name__ if getattr(mean_field, 'with_df', None) is not None else None,
        'auxbasis': task_spec.periodic.density_fitting_auxbasis,
    }
    smearing = {
        'method': task_spec.periodic.smearing_method,
        'sigma': task_spec.periodic.smearing_sigma,
        'fix_spin': bool(task_spec.periodic.smearing_fix_spin),
        'entropy': float(mean_field.entropy) if getattr(mean_field, 'entropy', None) is not None else None,
        'free_energy': float(mean_field.e_free) if getattr(mean_field, 'e_free', None) is not None else None,
        'zero_temperature_energy': float(mean_field.e_zero) if getattr(mean_field, 'e_zero', None) is not None else None,
    }
    periodic_numerics = {
        'precision': float(cell.precision),
        'requested_ke_cutoff_hartree': task_spec.periodic.ke_cutoff,
        'fft_mesh': [int(value) for value in np.asarray(cell.mesh, dtype=int)],
        'fft_mesh_source': 'explicit' if task_spec.periodic.fft_mesh is not None else (
            'ke_cutoff' if task_spec.periodic.ke_cutoff is not None else 'precision_default'
        ),
        'density_fitting': density_fitting,
        'exxdiv': task_spec.periodic.exxdiv,
        'kpoints': {
            'mesh': list(kmesh),
            'scheme': task_spec.periodic.kpoint_scheme,
            'shift_mesh_steps': [float(value) for value in task_spec.periodic.kpoint_shift],
            'scaled': [[float(value) for value in row] for row in scaled_kpoints],
            'absolute_1_per_bohr': [[float(value) for value in row] for row in absolute_kpoints],
            'weights': kpoint_weights,
        },
        'band_path': {
            'mode': task_spec.periodic.band_path_mode,
            'requested_path': task_spec.periodic.band_path,
            'requested_special_points_scaled': {
                str(label): [float(value) for value in coordinates]
                for label, coordinates in task_spec.periodic.band_path_special_points.items()
            },
            'requested_point_count': (
                None
                if task_spec.periodic.band_path_mode == 'seekpath'
                else task_spec.periodic.band_path_npoints
            ),
            'generated_point_count': (
                len(periodic_context['path_definition']['scaled_kpoints'])
                if isinstance(periodic_context.get('path_definition'), dict)
                else None
            ),
            'reference_distance_1_per_angstrom': task_spec.periodic.band_path_reference_distance,
            'symprec_angstrom': task_spec.periodic.band_path_symprec,
            'kmesh_cell_role': structure_summary.get('cell_role'),
            'enabled': 'band_structure' in task_spec.analysis.outputs,
        },
        'smearing': smearing,
    }
    result = {
        'task_type': 'periodic',
        'method': task_spec.method.name,
        'xc': task_spec.method.xc if task_spec.method.name == 'dft' else None,
        'reference': _periodic_reference_label(task_spec.method.name, restricted, gamma_point),
        'converged': bool(mean_field.converged),
        'energy': energy,
        'final_energy': energy,
        'final_method': task_spec.method.name,
        'energy_unit': (
            'Ha/standardized primitive cell'
            if task_spec.periodic.band_path_mode == 'seekpath'
            and 'band_structure' in task_spec.analysis.outputs
            else 'Ha/cell'
        ),
        'kmesh': list(kmesh),
        'kpoint_count': kpoint_count,
        'kpoint_scheme': task_spec.periodic.kpoint_scheme,
        'kpoint_shift': [float(value) for value in task_spec.periodic.kpoint_shift],
        'gamma_point': gamma_point,
        'nao': int(cell.nao_nr()),
        'nelectron': int(cell.nelectron),
        'basis': task_spec.periodic.basis,
        'pseudo': task_spec.periodic.pseudo,
        'dimension': int(cell.dimension),
        'periodic_structure': structure_summary,
        'periodic_input_structure': periodic_context.get('input_structure'),
        'seekpath_standardization': periodic_context.get('seekpath_standardization'),
        'periodic_standardization': (
            {
                key: periodic_context['seekpath_standardization'].get(key)
                for key in (
                    'status',
                    'library',
                    'recipe',
                    'symprec_angstrom',
                    'reference_distance_1_per_angstrom',
                    'spacegroup_number',
                    'spacegroup_international',
                    'bravais_lattice',
                    'bravais_lattice_extended',
                    'input_atom_count',
                    'primitive_atom_count',
                    'volume_original_wrt_primitive',
                    'path',
                    'path_segments',
                    'explicit_point_count',
                    'warnings',
                )
            }
            if isinstance(periodic_context.get('seekpath_standardization'), dict)
            else None
        ),
        'periodic_numerics': periodic_numerics,
        'density_fitting': density_fitting,
        'smearing': smearing,
        'raw_scf_output': scf_stdout.getvalue(),
        'analysis_text': '',
    }
    result.update(electronic_structure)
    result['gap'] = electronic_structure['band_gap']
    result['homo'] = electronic_structure['valence_band_max']
    result['lumo'] = electronic_structure['conduction_band_min']
    deferred = set(deferred_modules)
    dmft_deferred = bool({
        'embedding.fcdmft.periodic_gw',
        'embedding.fcdmft.prepare_subspace',
        'embedding.fcdmft.gw_double_counting',
        'embedding.fcdmft.hf_dmft',
        'embedding.fcdmft.gw_dmft',
    }.intersection(deferred))
    if dmft_deferred:
        result['_transient_module_context'] = {
            'mean_field': mean_field,
            'cell': cell,
            'restricted': restricted,
            'fermi_energy': electronic_structure['fermi_energy'],
            'fermi_energy_by_spin': electronic_structure['fermi_energy_by_spin'],
            'electron_count': int(cell.nelectron),
        }
    if 'band_structure' in task_spec.analysis.outputs:
        band_analysis_deferred = 'periodic.band_analysis' in deferred
        if band_analysis_deferred:
            result.setdefault('_transient_module_context', {}).update({
                'mean_field': mean_field,
                'cell': cell,
                'restricted': restricted,
                'band_path_mode': task_spec.periodic.band_path_mode,
                'band_path': task_spec.periodic.band_path,
                'band_path_special_points': task_spec.periodic.band_path_special_points,
                'band_path_npoints': task_spec.periodic.band_path_npoints,
                'path_definition': periodic_context.get('path_definition'),
                'fermi_energy': electronic_structure['fermi_energy'],
                'fermi_energy_by_spin': electronic_structure['fermi_energy_by_spin'],
            })
        elif mean_field.converged:
            band_structure = _periodic_band_structure(
                mean_field,
                cell,
                restricted=restricted,
                mode=task_spec.periodic.band_path_mode,
                requested_path=task_spec.periodic.band_path,
                requested_special_points=task_spec.periodic.band_path_special_points,
                npoints=task_spec.periodic.band_path_npoints,
                path_definition=periodic_context.get('path_definition'),
                fermi_energy=electronic_structure['fermi_energy'],
                fermi_by_spin=electronic_structure['fermi_energy_by_spin'],
            )
            result['periodic_band_structure'] = band_structure
            result['band_path_mode'] = band_structure['path_mode']
            result['band_path'] = band_structure['path']
            result['band_path_point_count'] = band_structure['point_count']
            result['band_path_labels'] = band_structure['special_point_labels']
            result['band_structure_status'] = 'completed'
        else:
            result['band_structure_status'] = 'skipped_unconverged_scf'
    return result
