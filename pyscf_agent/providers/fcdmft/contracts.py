from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Tuple

from ...input_validation import boolean, finite_float

from ...embedding.artifacts import read_embedding_hdf5
from ...embedding.contracts import (
    CORRELATED_SUBSPACE_AUDIT_SCHEMA,
    EMBEDDING_REFERENCE_SCHEMA,
    LOCALIZED_HAMILTONIAN_SCHEMA,
)


HF_DMFT_RESULT_SCHEMA = 'pyscf-agent.hf-dmft-result.v1'
HF_DMFT_ARRAYS_SCHEMA = 'pyscf-agent.hf-dmft-arrays.v1'
PERIODIC_GW_RESULT_SCHEMA = 'pyscf-agent.periodic-gw-result.v1'
PERIODIC_GW_ARRAYS_SCHEMA = 'pyscf-agent.periodic-gw-arrays.v1'
GW_DMFT_RESULT_SCHEMA = 'pyscf-agent.gw-dmft-result.v1'
GW_DMFT_ARRAYS_SCHEMA = 'pyscf-agent.gw-dmft-arrays.v1'

DEFAULT_HF_DMFT_OPTIONS: Dict[str, Any] = {
    'impurity_solver': 'cc',
    'ncore': 0,
    'nval': None,
    'nbath': 4,
    'nb_per_e': None,
    'bath_discretization': 'opt',
    'max_iterations': 10,
    'convergence_tolerance': 1e-3,
    'damping': 0.7,
    'gmres_tolerance': 1e-3,
    'chemical_potential': None,
    'optimize_chemical_potential': False,
    'target_occupancy': None,
    'bath_window': [-0.4, 0.4],
    'broadening': 0.1,
    'diagonal_bath_fit': False,
    'max_memory_mb': None,
    'n_threads': None,
}

_ALLOWED_OPTIONS = frozenset(DEFAULT_HF_DMFT_OPTIONS)
_IMPURITY_SOLVERS = frozenset(('cc', 'ucc', 'fci'))
_BATH_DISCRETIZATIONS = frozenset(('opt', 'direct', 'linear', 'gauss', 'log'))

DEFAULT_PERIODIC_GW_OPTIONS: Dict[str, Any] = {
    'gw_analytic_continuation': 'pade',
    'gw_broadening': 0.1 / 27.211386,
    'gw_frequency_window': [0.0, 18.0 / 27.211386],
    'gw_real_frequency_points': 181,
    'gw_imaginary_frequency_points': 100,
    'gw_full_self_energy': True,
    'gw_finite_size_correction': True,
    'gw_quasiparticle_energies': True,
    'gw_orbital_indices': [],
    'gw_kpoint_indices': [],
    'gw_artifacts': {},
    'max_memory_mb': None,
    'n_threads': None,
}

_GW_ALLOWED_OPTIONS = frozenset(DEFAULT_PERIODIC_GW_OPTIONS)
_GW_DMFT_ALLOWED_OPTIONS = frozenset(DEFAULT_HF_DMFT_OPTIONS).union(_GW_ALLOWED_OPTIONS).union({
    'gw_dc_imaginary_time_points',
})


def _positive_int(value: Any, default: Optional[int], field: str) -> Optional[int]:
    if value is None and default is None:
        return None
    raw = default if value is None else value
    if isinstance(raw, bool):
        raise ValueError('{0} must be a positive integer'.format(field))
    try:
        normalized = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError('{0} must be a positive integer'.format(field)) from exc
    if str(raw).strip() not in (str(normalized), '{0}.0'.format(normalized)) or normalized <= 0:
        raise ValueError('{0} must be a positive integer'.format(field))
    return normalized


def _nonnegative_int(value: Any, default: int, field: str) -> int:
    raw = default if value is None else value
    if isinstance(raw, bool):
        raise ValueError('{0} must be a non-negative integer'.format(field))
    try:
        normalized = int(raw)
    except (TypeError, ValueError) as exc:
        raise ValueError('{0} must be a non-negative integer'.format(field)) from exc
    if str(raw).strip() not in (str(normalized), '{0}.0'.format(normalized)) or normalized < 0:
        raise ValueError('{0} must be a non-negative integer'.format(field))
    return normalized


def _positive_float(value: Any, default: float, field: str) -> float:
    raw = default if value is None else value
    try:
        normalized = finite_float(raw, field)
    except (TypeError, ValueError) as exc:
        raise ValueError('{0} must be a positive finite number'.format(field)) from exc
    if not math.isfinite(normalized) or normalized <= 0:
        raise ValueError('{0} must be a positive finite number'.format(field))
    return normalized


def _optional_finite_float(value: Any, field: str) -> Optional[float]:
    if value in (None, ''):
        return None
    try:
        normalized = finite_float(value, field)
    except (TypeError, ValueError) as exc:
        raise ValueError('{0} must be a finite number'.format(field)) from exc
    if not math.isfinite(normalized):
        raise ValueError('{0} must be a finite number'.format(field))
    return normalized


def _integer_indices(value: Any, field: str) -> list:
    if value in (None, ''):
        return []
    if not isinstance(value, (list, tuple)):
        raise ValueError('{0} must be a list of non-negative integers'.format(field))
    normalized = []
    for item in value:
        index = _nonnegative_int(item, 0, field)
        normalized.append(index)
    if len(set(normalized)) != len(normalized):
        raise ValueError('{0} must not contain duplicates'.format(field))
    return normalized


def _normalize_gw_fields(raw: Mapping[str, Any]) -> Dict[str, Any]:
    normalized = copy.deepcopy(DEFAULT_PERIODIC_GW_OPTIONS)
    normalized.update({key: copy.deepcopy(value) for key, value in raw.items() if key in _GW_ALLOWED_OPTIONS})
    normalized['gw_analytic_continuation'] = str(
        normalized.get('gw_analytic_continuation') or ''
    ).strip().lower()
    if normalized['gw_analytic_continuation'] != 'pade':
        raise ValueError('The executable periodic GW contract currently requires gw_analytic_continuation=pade')
    normalized['gw_broadening'] = _positive_float(
        normalized.get('gw_broadening'), DEFAULT_PERIODIC_GW_OPTIONS['gw_broadening'], 'gw_broadening'
    )
    window = normalized.get('gw_frequency_window')
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError('gw_frequency_window must contain [lower, upper] energies in Hartree')
    lower = _optional_finite_float(window[0], 'gw_frequency_window[0]')
    upper = _optional_finite_float(window[1], 'gw_frequency_window[1]')
    if lower is None or upper is None or lower >= upper:
        raise ValueError('gw_frequency_window lower bound must be smaller than its upper bound')
    normalized['gw_frequency_window'] = [lower, upper]
    normalized['gw_real_frequency_points'] = _positive_int(
        normalized.get('gw_real_frequency_points'), 181, 'gw_real_frequency_points'
    )
    if normalized['gw_real_frequency_points'] < 2:
        raise ValueError('gw_real_frequency_points must be at least 2')
    normalized['gw_imaginary_frequency_points'] = _positive_int(
        normalized.get('gw_imaginary_frequency_points'), 100, 'gw_imaginary_frequency_points'
    )
    normalized['gw_full_self_energy'] = boolean(normalized.get('gw_full_self_energy', False), 'gw_full_self_energy')
    normalized['gw_finite_size_correction'] = boolean(normalized.get('gw_finite_size_correction', False), 'gw_finite_size_correction')
    normalized['gw_quasiparticle_energies'] = boolean(normalized.get('gw_quasiparticle_energies', False), 'gw_quasiparticle_energies')
    normalized['gw_orbital_indices'] = _integer_indices(
        normalized.get('gw_orbital_indices'), 'gw_orbital_indices'
    )
    normalized['gw_kpoint_indices'] = _integer_indices(
        normalized.get('gw_kpoint_indices'), 'gw_kpoint_indices'
    )
    artifacts = normalized.get('gw_artifacts')
    if artifacts in (None, ''):
        artifacts = {}
    if not isinstance(artifacts, Mapping):
        raise ValueError('gw_artifacts must be a mapping of registered artifact references')
    normalized['gw_artifacts'] = copy.deepcopy(dict(artifacts))
    normalized['max_memory_mb'] = _positive_int(
        normalized.get('max_memory_mb'), None, 'max_memory_mb'
    )
    normalized['n_threads'] = _positive_int(
        normalized.get('n_threads'), None, 'n_threads'
    )
    return normalized


def normalize_periodic_gw_options(options: Any) -> Dict[str, Any]:
    if options is not None and not isinstance(options, Mapping):
        raise ValueError('Solver options must be an object')
    raw = copy.deepcopy(dict(options or {}))
    unknown = sorted(set(raw).difference(_GW_ALLOWED_OPTIONS))
    if unknown:
        raise ValueError('Unsupported periodic GW solver option(s): {0}'.format(', '.join(unknown)))
    return _normalize_gw_fields(raw)


def normalize_gw_dmft_options(options: Any) -> Dict[str, Any]:
    if options is not None and not isinstance(options, Mapping):
        raise ValueError('Solver options must be an object')
    raw = copy.deepcopy(dict(options or {}))
    unknown = sorted(set(raw).difference(_GW_DMFT_ALLOWED_OPTIONS))
    if unknown:
        raise ValueError('Unsupported GW+DMFT solver option(s): {0}'.format(', '.join(unknown)))
    dmft_raw = {key: value for key, value in raw.items() if key in _ALLOWED_OPTIONS}
    normalized = normalize_hf_dmft_options(dmft_raw)
    normalized.update(_normalize_gw_fields(raw))
    normalized['gw_dc_imaginary_time_points'] = _positive_int(
        raw.get('gw_dc_imaginary_time_points'), 2000, 'gw_dc_imaginary_time_points'
    )
    if not normalized['gw_full_self_energy']:
        raise ValueError('GW+DMFT requires gw_full_self_energy=true for the local matrix self-energy')
    return normalized


def normalize_hf_dmft_options(options: Any) -> Dict[str, Any]:
    if options is not None and not isinstance(options, Mapping):
        raise ValueError('Solver options must be an object')
    raw = copy.deepcopy(dict(options or {}))
    unknown = sorted(set(raw).difference(_ALLOWED_OPTIONS))
    if unknown:
        raise ValueError('Unsupported HF+DMFT solver option(s): {0}'.format(', '.join(unknown)))
    normalized = copy.deepcopy(DEFAULT_HF_DMFT_OPTIONS)
    normalized.update(raw)
    normalized['impurity_solver'] = str(normalized['impurity_solver'] or '').strip().lower()
    if normalized['impurity_solver'] not in _IMPURITY_SOLVERS:
        raise ValueError('HF+DMFT impurity_solver must be cc, ucc, or fci')
    normalized['ncore'] = _nonnegative_int(normalized.get('ncore'), 0, 'ncore')
    normalized['nval'] = _positive_int(normalized.get('nval'), None, 'nval')
    normalized['nbath'] = _positive_int(normalized.get('nbath'), 4, 'nbath')
    normalized['nb_per_e'] = _positive_int(normalized.get('nb_per_e'), None, 'nb_per_e')
    normalized['max_iterations'] = _positive_int(
        normalized.get('max_iterations'), 10, 'max_iterations'
    )
    normalized['convergence_tolerance'] = _positive_float(
        normalized.get('convergence_tolerance'), 1e-3, 'convergence_tolerance'
    )
    normalized['damping'] = _positive_float(normalized.get('damping'), 0.7, 'damping')
    if normalized['damping'] > 1.0:
        raise ValueError('damping must not exceed 1')
    normalized['gmres_tolerance'] = _positive_float(
        normalized.get('gmres_tolerance'), 1e-3, 'gmres_tolerance'
    )
    normalized['chemical_potential'] = _optional_finite_float(
        normalized.get('chemical_potential'), 'chemical_potential'
    )
    normalized['target_occupancy'] = _optional_finite_float(
        normalized.get('target_occupancy'), 'target_occupancy'
    )
    if normalized['target_occupancy'] is not None and normalized['target_occupancy'] <= 0:
        raise ValueError('target_occupancy must be a positive finite number')
    normalized['broadening'] = _positive_float(
        normalized.get('broadening'), 0.1, 'broadening'
    )
    normalized['bath_discretization'] = str(
        normalized.get('bath_discretization') or ''
    ).strip().lower()
    if normalized['bath_discretization'] not in _BATH_DISCRETIZATIONS:
        raise ValueError(
            'bath_discretization must be opt, direct, linear, gauss, or log'
        )
    if (
        normalized['bath_discretization'] in ('opt', 'direct', 'log')
        and normalized['nbath'] < 2
    ):
        raise ValueError(
            'fcDMFT bath_discretization={0} requires nbath >= 2 because its '
            'direct integration grid needs at least two bath energies'.format(
                normalized['bath_discretization']
            )
        )
    window = normalized.get('bath_window')
    if not isinstance(window, (list, tuple)) or len(window) != 2:
        raise ValueError('bath_window must contain [lower, upper] energies in Hartree')
    lower = _optional_finite_float(window[0], 'bath_window[0]')
    upper = _optional_finite_float(window[1], 'bath_window[1]')
    if lower is None or upper is None or lower >= upper:
        raise ValueError('bath_window lower bound must be smaller than its upper bound')
    normalized['bath_window'] = [lower, upper]
    normalized['optimize_chemical_potential'] = boolean(normalized.get('optimize_chemical_potential', False), 'optimize_chemical_potential')
    normalized['diagonal_bath_fit'] = boolean(normalized.get('diagonal_bath_fit', False), 'diagonal_bath_fit')
    normalized['max_memory_mb'] = _positive_int(
        normalized.get('max_memory_mb'), None, 'max_memory_mb'
    )
    normalized['n_threads'] = _positive_int(
        normalized.get('n_threads'), None, 'n_threads'
    )
    if normalized['optimize_chemical_potential'] and normalized['target_occupancy'] is None:
        raise ValueError('optimize_chemical_potential requires target_occupancy')
    return normalized


def _artifact_path(reference: Any, kind: str) -> Path:
    if not isinstance(reference, Mapping) or str(reference.get('kind') or '') != kind:
        raise ValueError('HF+DMFT requires a registered {0} artifact'.format(kind))
    path = Path(str(reference.get('path') or '')).expanduser()
    if not path.is_file():
        raise ValueError('HF+DMFT artifact is unavailable: {0}'.format(path))
    return path


def load_hf_dmft_inputs(embedding: Any) -> Dict[str, Any]:
    """Load the approved interchange artifacts and enforce the fcDMFT layout."""

    import numpy as np

    reference_path = _artifact_path(embedding.reference_artifact, 'embedding_reference')
    hamiltonian_path = _artifact_path(
        embedding.localized_hamiltonian_artifact, 'localized_hamiltonian'
    )
    reference = read_embedding_hdf5(reference_path)
    hamiltonian = read_embedding_hdf5(hamiltonian_path)
    if reference['schema'] != EMBEDDING_REFERENCE_SCHEMA:
        raise ValueError('HF+DMFT embedding reference schema is incompatible')
    if hamiltonian['schema'] != LOCALIZED_HAMILTONIAN_SCHEMA:
        raise ValueError('HF+DMFT localized Hamiltonian schema is incompatible')
    datasets = hamiltonian['datasets']
    missing = [
        name for name in ('one_body_local', 'fock_local', 'density_local', 'two_body_local')
        if name not in datasets
    ]
    if missing:
        raise ValueError(
            'HF+DMFT localized Hamiltonian is missing dataset(s): {0}'.format(', '.join(missing))
        )
    hcore = np.asarray(datasets['one_body_local'])
    fock = np.asarray(datasets['fock_local'])
    density = np.asarray(datasets['density_local'])
    eri = np.asarray(datasets['two_body_local'])
    if hcore.ndim not in (3, 4):
        raise ValueError('HF+DMFT one-body arrays must use k-point or spin/k-point layout')
    if fock.shape != hcore.shape or density.shape != hcore.shape:
        raise ValueError('HF+DMFT hcore, fock, and density arrays must have identical shapes')
    if hcore.ndim == 3:
        hcore = hcore[np.newaxis, ...]
        fock = fock[np.newaxis, ...]
        density = density[np.newaxis, ...]
    spin = hcore.shape[0]
    nlo = hcore.shape[-1]
    if hcore.shape[-2] != nlo:
        raise ValueError('HF+DMFT one-body arrays must be square')
    expected_eri_shape = (spin * (spin + 1) // 2, nlo, nlo, nlo, nlo)
    if eri.ndim == 4:
        eri = eri[np.newaxis, ...]
    if eri.shape != expected_eri_shape:
        raise ValueError(
            'HF+DMFT localized ERI shape must be {0}; received {1}'.format(
                expected_eri_shape, eri.shape
            )
        )
    metadata = copy.deepcopy(hamiltonian.get('metadata') or {})
    expected_pair_order = 'restricted' if spin == 1 else 'aa_bb_ab'
    pair_order = str(metadata.get('two_body_spin_pair_order') or '').strip().lower()
    if pair_order != expected_pair_order:
        raise ValueError(
            'HF+DMFT localized ERIs require two_body_spin_pair_order={0}'.format(
                expected_pair_order
            )
        )
    hf_jk = None
    if 'hf_effective_potential_local' in datasets:
        hf_jk = np.asarray(datasets['hf_effective_potential_local'])
        if hf_jk.ndim == 3:
            hf_jk = hf_jk[np.newaxis, ...]
        if hf_jk.shape != hcore.shape:
            raise ValueError('Localized HF effective potential must match the one-body matrix shape')
    return {
        'hcore_k': hcore,
        'jk_k': fock - hcore,
        'hf_jk_k': hf_jk,
        'density_k': density,
        'eri': eri,
        'spin_channels': spin,
        'nkpts': hcore.shape[-3],
        'nlo': nlo,
        'metadata': metadata,
    }


def _initial_hybridization_norm(
    inputs: Mapping[str, Any],
    *,
    ncore: int,
    nval: int,
) -> float:
    """Estimate whether the approved lattice can generate a DMFT bath."""

    import numpy as np

    fock_k = np.asarray(inputs['hcore_k']) + np.asarray(inputs['jk_k'])
    spin_channels, nkpts, orbital_count, _ = fock_k.shape
    correlated = slice(ncore, nval)
    broadening = 0.1
    norms = []
    for spin in range(spin_channels):
        local_fock = np.mean(fock_k[spin], axis=0)
        identity = np.eye(orbital_count, dtype=np.complex128)
        correlated_identity = np.eye(nval - ncore, dtype=np.complex128)
        for frequency in (-0.25, 0.0, 0.25):
            z_value = complex(frequency, broadening)
            impurity_green = np.linalg.inv(
                z_value * correlated_identity - local_fock[correlated, correlated]
            )
            lattice_green = np.zeros(
                (orbital_count, orbital_count),
                dtype=np.complex128,
            )
            for kpoint in range(nkpts):
                lattice_green += np.linalg.inv(
                    z_value * identity - fock_k[spin, kpoint]
                ) / float(nkpts)
            projected_green = lattice_green[correlated, correlated]
            hybridization = (
                np.linalg.inv(impurity_green)
                - np.linalg.inv(projected_green)
            )
            norms.append(float(np.linalg.norm(hybridization)))
    return max(norms, default=0.0)


def _reference_is_restricted(task_spec: Any) -> bool:
    restricted = task_spec.method.restricted
    if restricted is None:
        return int(task_spec.system.spin or 0) == 0
    return bool(restricted)


def validate_periodic_gw_reference(task_spec: Any, *, for_dmft: bool = False) -> Dict[str, Any]:
    label = 'GW+DMFT' if for_dmft else 'periodic GW'
    if str(task_spec.task_type or '').strip().lower() != 'periodic':
        raise ValueError('{0} requires a periodic TaskSpec'.format(label))
    if str(task_spec.method.name or '').strip().lower() not in ('hf', 'dft'):
        raise ValueError('{0} requires a periodic HF or DFT reference'.format(label))
    if not _reference_is_restricted(task_spec):
        raise ValueError('{0} currently requires a restricted periodic reference'.format(label))
    if str(task_spec.periodic.density_fitting_method or '').strip().lower() != 'gdf':
        raise ValueError('{0} requires periodic density_fitting_method=gdf'.format(label))
    if str(task_spec.periodic.smearing_method or 'none').strip().lower() != 'none':
        raise ValueError('{0} currently requires integer occupations without smearing'.format(label))
    if str(task_spec.periodic.kpoint_scheme or '').strip().lower() != 'gamma_centered':
        raise ValueError('{0} currently requires a gamma-centered k mesh'.format(label))
    if any(abs(float(value)) > 1e-12 for value in task_spec.periodic.kpoint_shift):
        raise ValueError('{0} currently requires zero k-point shift'.format(label))
    kpoint_count = math.prod(int(value) for value in task_spec.periodic.kmesh)
    if kpoint_count <= 1:
        raise ValueError(
            '{0} requires a k-resolved periodic reference with more than one k point'.format(label)
        )
    options = (
        normalize_gw_dmft_options(task_spec.solver.options)
        if for_dmft
        else normalize_periodic_gw_options(task_spec.solver.options)
    )
    return options


def load_registered_gw_artifacts(options: Mapping[str, Any], *, require_local: bool = False) -> Dict[str, Path]:
    references = options.get('gw_artifacts') if isinstance(options, Mapping) else None
    if not isinstance(references, Mapping):
        raise ValueError('GW artifacts are unavailable; run the periodic GW stage first')
    required = {
        'lattice_result': 'periodic_gw_result',
        'lattice_ac': 'periodic_gw_analytic_continuation',
        'lattice_vxc': 'periodic_gw_mean_field_potential',
        'lattice_sigma_imag': 'periodic_gw_imaginary_self_energy',
    }
    if require_local:
        required.update({
            'local_ac': 'gw_dmft_local_analytic_continuation',
            'local_sigma_imag': 'gw_dmft_local_imaginary_self_energy',
            'local_transform': 'gw_dmft_local_orbital_transform',
        })
    paths = {}
    for key, kind in required.items():
        paths[key] = _artifact_path(references.get(key), kind)
    return paths


def validate_hf_dmft_request(task_spec: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if str(task_spec.task_type or '').strip().lower() != 'periodic':
        raise ValueError('HF+DMFT currently requires a periodic TaskSpec')
    if str(task_spec.method.name or '').strip().lower() != 'hf':
        raise ValueError('HF+DMFT currently requires method=hf as its reference calculation')
    if not task_spec.embedding.enabled or not task_spec.embedding.approved:
        raise ValueError('HF+DMFT requires an approved correlated-subspace audit')
    audit_path = _artifact_path(
        task_spec.embedding.audit_artifact,
        'correlated_subspace_audit',
    )
    try:
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('HF+DMFT correlated-subspace audit could not be read') from exc
    if str(audit.get('schema') or '') != CORRELATED_SUBSPACE_AUDIT_SCHEMA:
        raise ValueError('HF+DMFT correlated-subspace audit schema is incompatible')
    approval = audit.get('approval') if isinstance(audit.get('approval'), Mapping) else {}
    if not bool(approval.get('approved')):
        raise ValueError('HF+DMFT correlated-subspace audit does not contain an approval record')
    options = normalize_hf_dmft_options(task_spec.solver.options)
    inputs = load_hf_dmft_inputs(task_spec.embedding)
    nval = options['nval'] or inputs['nlo']
    ncore = options['ncore']
    if ncore >= nval or nval > inputs['nlo']:
        raise ValueError('HF+DMFT requires 0 <= ncore < nval <= the localized orbital count')
    selected = list(task_spec.embedding.correlated_orbital_indices)
    expected = list(range(ncore, nval))
    if selected != expected:
        raise ValueError(
            'HF+DMFT correlated_orbital_indices must equal the contiguous fcDMFT window '
            '[ncore, nval); expected {0}'.format(expected)
        )
    options['nval'] = nval
    options['nb_per_e'] = options['nb_per_e'] or (nval - ncore)
    if options['nb_per_e'] > nval - ncore:
        raise ValueError('nb_per_e must not exceed nval - ncore')
    restricted = _reference_is_restricted(task_spec)
    expected_spin_channels = 1 if restricted else 2
    if inputs['spin_channels'] != expected_spin_channels:
        raise ValueError(
            'HF+DMFT reference and localized artifacts disagree: expected {0} spin '
            'channel(s), received {1}'.format(
                expected_spin_channels,
                inputs['spin_channels'],
            )
        )
    if restricted and options['impurity_solver'] == 'ucc':
        raise ValueError('The ucc impurity solver requires an unrestricted HF reference')
    if not restricted and options['impurity_solver'] != 'ucc':
        raise ValueError('Unrestricted HF+DMFT currently requires impurity_solver=ucc')
    hybridization_norm = _initial_hybridization_norm(
        inputs,
        ncore=ncore,
        nval=nval,
    )
    if not math.isfinite(hybridization_norm) or hybridization_norm <= 1e-10:
        raise ValueError(
            'The approved HF+DMFT subspace has no resolvable lattice hybridization. '
            'A full-cell Gamma-only subspace is an isolated finite problem and cannot '
            'define a nontrivial DMFT bath; use a k-resolved periodic reference or leave '
            'explicit environment orbitals outside the correlated window.'
        )
    inputs['initial_hybridization_norm'] = hybridization_norm
    return options, inputs


def validate_gw_dmft_request(task_spec: Any) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Path]]:
    if str(task_spec.method.name or '').strip().lower() != 'dft':
        raise ValueError('GW+DMFT currently requires method=dft as its periodic reference')
    options = validate_periodic_gw_reference(task_spec, for_dmft=True)
    if not task_spec.embedding.enabled or not task_spec.embedding.approved:
        raise ValueError('GW+DMFT requires an approved correlated-subspace audit')
    audit_path = _artifact_path(
        task_spec.embedding.audit_artifact,
        'correlated_subspace_audit',
    )
    try:
        audit = json.loads(audit_path.read_text(encoding='utf-8'))
    except (OSError, ValueError) as exc:
        raise ValueError('GW+DMFT correlated-subspace audit could not be read') from exc
    if str(audit.get('schema') or '') != CORRELATED_SUBSPACE_AUDIT_SCHEMA:
        raise ValueError('GW+DMFT correlated-subspace audit schema is incompatible')
    approval = audit.get('approval') if isinstance(audit.get('approval'), Mapping) else {}
    if not bool(approval.get('approved')):
        raise ValueError('GW+DMFT correlated-subspace audit does not contain an approval record')
    inputs = load_hf_dmft_inputs(task_spec.embedding)
    if inputs.get('hf_jk_k') is None:
        raise ValueError(
            'GW+DMFT localized Hamiltonian is missing hf_effective_potential_local; '
            'rebuild the correlated subspace from the DFT reference'
        )
    inputs['jk_k'] = inputs['hf_jk_k']
    nval = options['nval'] or inputs['nlo']
    ncore = options['ncore']
    if ncore >= nval or nval > inputs['nlo']:
        raise ValueError('GW+DMFT requires 0 <= ncore < nval <= the localized orbital count')
    selected = list(task_spec.embedding.correlated_orbital_indices)
    expected = list(range(ncore, nval))
    if selected != expected:
        raise ValueError(
            'GW+DMFT correlated_orbital_indices must equal the contiguous fcDMFT window '
            '[ncore, nval); expected {0}'.format(expected)
        )
    options['nval'] = nval
    options['nb_per_e'] = options['nb_per_e'] or (nval - ncore)
    if options['nb_per_e'] > nval - ncore:
        raise ValueError('nb_per_e must not exceed nval - ncore')
    if inputs['spin_channels'] != 1:
        raise ValueError('GW+DMFT currently requires one restricted spin channel')
    if options['impurity_solver'] == 'ucc':
        raise ValueError('GW+DMFT with a restricted reference does not support impurity_solver=ucc')
    hybridization_norm = _initial_hybridization_norm(inputs, ncore=ncore, nval=nval)
    if not math.isfinite(hybridization_norm) or hybridization_norm <= 1e-10:
        raise ValueError(
            'The approved GW+DMFT subspace has no resolvable lattice hybridization; '
            'use a k-resolved reference and retain the periodic environment'
        )
    inputs['initial_hybridization_norm'] = hybridization_norm
    artifact_paths = load_registered_gw_artifacts(options, require_local=True)
    return options, inputs, artifact_paths


__all__ = [
    'DEFAULT_HF_DMFT_OPTIONS',
    'HF_DMFT_ARRAYS_SCHEMA',
    'HF_DMFT_RESULT_SCHEMA',
    'GW_DMFT_ARRAYS_SCHEMA',
    'GW_DMFT_RESULT_SCHEMA',
    'PERIODIC_GW_ARRAYS_SCHEMA',
    'PERIODIC_GW_RESULT_SCHEMA',
    'DEFAULT_PERIODIC_GW_OPTIONS',
    'load_hf_dmft_inputs',
    'load_registered_gw_artifacts',
    'normalize_gw_dmft_options',
    'normalize_hf_dmft_options',
    'normalize_periodic_gw_options',
    'validate_gw_dmft_request',
    'validate_hf_dmft_request',
    'validate_periodic_gw_reference',
]
