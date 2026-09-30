from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Sequence, Tuple

from ...embedding.preparation import prepare_embedding_artifacts
from ...providers.libdmet.transforms import transform_density_to_local
from .contracts import normalize_gw_dmft_options, normalize_hf_dmft_options


_AUTOMATIC_LOCALIZATION_METHODS = frozenset(('iao', 'iao_pao'))


class _KPointMeanFieldView:
    """Expose Gamma-point PySCF references with libDMET k-point shapes."""

    def __init__(self, reference: Any, mo_coeff: Any, mo_occ: Any, overlap: Any):
        self._reference = reference
        self.mo_coeff = mo_coeff
        self.mo_occ = mo_occ
        self._overlap = overlap

    def get_ovlp(self) -> Any:
        return self._overlap

    def __getattr__(self, name: str) -> Any:
        return getattr(self._reference, name)


def validate_hf_dmft_preparation_request(task_spec: Any) -> Dict[str, Any]:
    """Validate the HF reference that will generate a reviewable DMFT subspace."""

    if str(task_spec.task_type or '').strip().lower() != 'periodic':
        raise ValueError('HF+DMFT correlated-subspace preparation requires a periodic TaskSpec')
    if str(task_spec.method.name or '').strip().lower() != 'hf':
        raise ValueError('HF+DMFT correlated-subspace preparation requires method=hf')
    if not task_spec.embedding.enabled:
        raise ValueError('HF+DMFT requires embedding.enabled=true')
    if str(task_spec.embedding.provider or '').strip().lower() != 'fcdmft':
        raise ValueError('HF+DMFT requires embedding.provider=fcdmft')
    method = str(task_spec.embedding.localization_method or '').strip().lower()
    if method not in _AUTOMATIC_LOCALIZATION_METHODS:
        raise ValueError(
            'Automatic HF+DMFT subspace preparation currently supports '
            'localization_method=iao or iao_pao'
        )
    if str(task_spec.periodic.smearing_method or 'none').strip().lower() != 'none':
        raise ValueError(
            'Automatic IAO preparation currently requires integer occupations; '
            'disable periodic smearing or provide approved embedding artifacts manually'
        )
    if str(task_spec.periodic.kpoint_scheme or '').strip().lower() != 'gamma_centered':
        raise ValueError(
            'Automatic HF+DMFT subspace preparation currently requires a gamma-centered k mesh'
        )
    if any(abs(float(value)) > 1e-12 for value in task_spec.periodic.kpoint_shift):
        raise ValueError(
            'Automatic HF+DMFT subspace preparation currently requires zero k-point shift'
        )
    return normalize_hf_dmft_options(task_spec.solver.options)


def validate_fcdmft_preparation_request(task_spec: Any) -> Dict[str, Any]:
    solver_name = str(task_spec.solver.name or '').strip().lower().replace('-', '_')
    if solver_name == 'hf_dmft':
        return validate_hf_dmft_preparation_request(task_spec)
    if solver_name != 'gw_dmft':
        raise ValueError('fcDMFT correlated-subspace preparation requires hf_dmft or gw_dmft')
    if str(task_spec.task_type or '').strip().lower() != 'periodic':
        raise ValueError('GW+DMFT correlated-subspace preparation requires a periodic TaskSpec')
    if str(task_spec.method.name or '').strip().lower() != 'dft':
        raise ValueError('GW+DMFT correlated-subspace preparation requires method=dft')
    if task_spec.method.restricted is False or int(task_spec.system.spin or 0) != 0:
        raise ValueError('GW+DMFT correlated-subspace preparation requires a restricted closed-shell reference')
    if not task_spec.embedding.enabled:
        raise ValueError('GW+DMFT requires embedding.enabled=true')
    if str(task_spec.embedding.provider or '').strip().lower() != 'fcdmft':
        raise ValueError('GW+DMFT requires embedding.provider=fcdmft')
    method = str(task_spec.embedding.localization_method or '').strip().lower()
    if method not in _AUTOMATIC_LOCALIZATION_METHODS:
        raise ValueError(
            'Automatic GW+DMFT subspace preparation currently supports '
            'localization_method=iao or iao_pao'
        )
    if str(task_spec.periodic.smearing_method or 'none').strip().lower() != 'none':
        raise ValueError('Automatic GW+DMFT IAO preparation requires integer occupations')
    if str(task_spec.periodic.kpoint_scheme or '').strip().lower() != 'gamma_centered':
        raise ValueError('Automatic GW+DMFT IAO preparation requires a gamma-centered k mesh')
    if any(abs(float(value)) > 1e-12 for value in task_spec.periodic.kpoint_shift):
        raise ValueError('Automatic GW+DMFT IAO preparation requires zero k-point shift')
    return normalize_gw_dmft_options(task_spec.solver.options)


def _periodic_lattice(cell: Any, kmesh: Sequence[int], mean_field: Any) -> Any:
    import numpy as np
    from libdmet.system import lattice

    periodic_lattice = lattice.Lattice(cell, [int(value) for value in kmesh])
    reference_kpoints = getattr(mean_field, 'kpts', None)
    if reference_kpoints is not None:
        reference_kpoints = np.asarray(reference_kpoints)
        lattice_kpoints = np.asarray(periodic_lattice.kpts)
        if reference_kpoints.ndim == 1:
            reference_kpoints = reference_kpoints[np.newaxis, ...]
        if reference_kpoints.shape != lattice_kpoints.shape or not np.allclose(
            reference_kpoints,
            lattice_kpoints,
            atol=1e-9,
            rtol=0.0,
        ):
            raise ValueError(
                'The libDMET lattice k points do not match the periodic HF reference; '
                'provide approved localized artifacts for this k-point convention'
            )
    return periodic_lattice


def _libdmet_kpoint_mean_field(mean_field: Any, *, restricted: bool) -> Any:
    """Normalize PySCF Gamma references for libDMET's periodic IAO API."""

    import numpy as np

    coefficients = np.asarray(mean_field.mo_coeff)
    occupations = np.asarray(mean_field.mo_occ)
    overlap = np.asarray(mean_field.get_ovlp())
    original_shapes = (coefficients.shape, occupations.shape, overlap.shape)

    if restricted:
        if coefficients.ndim == 2:
            coefficients = coefficients[np.newaxis, ...]
        if occupations.ndim == 1:
            occupations = occupations[np.newaxis, ...]
        if coefficients.ndim != 3 or occupations.ndim != 2:
            raise ValueError(
                'Restricted periodic HF orbitals must use AO/MO or k-point/AO/MO layout'
            )
    else:
        if coefficients.ndim == 3 and coefficients.shape[0] == 2:
            coefficients = coefficients[:, np.newaxis, ...]
        if occupations.ndim == 2 and occupations.shape[0] == 2:
            occupations = occupations[:, np.newaxis, ...]
        if coefficients.ndim != 4 or coefficients.shape[0] != 2:
            raise ValueError(
                'Unrestricted periodic HF orbitals must use spin/AO/MO or '
                'spin/k-point/AO/MO layout'
            )
        if occupations.ndim != 3 or occupations.shape[0] != 2:
            raise ValueError(
                'Unrestricted periodic HF occupations must use spin/MO or '
                'spin/k-point/MO layout'
            )

    if overlap.ndim == 2:
        overlap = overlap[np.newaxis, ...]
    if overlap.ndim != 3:
        raise ValueError('Periodic HF overlap must use AO/AO or k-point/AO/AO layout')
    if coefficients.shape[-3] != overlap.shape[0]:
        raise ValueError('Periodic HF orbital and overlap k-point dimensions do not match')

    normalized_shapes = (coefficients.shape, occupations.shape, overlap.shape)
    if normalized_shapes == original_shapes:
        return mean_field
    return _KPointMeanFieldView(mean_field, coefficients, occupations, overlap)


def _iao_coefficients(
    periodic_lattice: Any,
    mean_field: Any,
    *,
    minimal_basis: str,
    include_pao: bool,
    restricted: bool,
) -> Tuple[Any, Tuple[str, ...]]:
    import numpy as np
    from libdmet.basis_transform import make_basis

    libdmet_mean_field = _libdmet_kpoint_mean_field(
        mean_field,
        restricted=restricted,
    )
    result = make_basis.get_C_ao_lo_iao(
        periodic_lattice,
        libdmet_mean_field,
        minao=str(minimal_basis or 'minao'),
        orth_virt=True,
        full_virt=False,
        full_return=True,
        return_labels=True,
        allow_smearing=False,
    )
    coefficients = np.asarray(result[0] if include_pao else result[1])
    labels = tuple(str(label) for label in (result[-1] or ()))
    if labels:
        labels = labels[:coefficients.shape[-1]]
    if len(labels) < coefficients.shape[-1]:
        labels = labels + tuple(
            'LO {0}'.format(index)
            for index in range(len(labels), coefficients.shape[-1])
        )
    return coefficients, labels


def _canonical_coefficients(coefficients: Any, *, restricted: bool) -> Any:
    import numpy as np

    values = np.asarray(coefficients)
    if restricted:
        if values.ndim == 2:
            values = values[np.newaxis, ...]
        if values.ndim != 3:
            raise ValueError('Restricted periodic localized orbitals must use k-point layout')
        return values
    if values.ndim == 3 and values.shape[0] == 2:
        values = values[:, np.newaxis, ...]
    if values.ndim != 4 or values.shape[0] != 2:
        raise ValueError('Unrestricted periodic localized orbitals must use spin/k-point layout')
    return values


def _canonical_reference_matrices(mean_field: Any, *, restricted: bool) -> Tuple[Any, Any, Any, Any]:
    import numpy as np

    hcore = np.asarray(mean_field.get_hcore())
    fock = np.asarray(mean_field.get_fock())
    density = np.asarray(mean_field.make_rdm1())
    overlap = np.asarray(mean_field.get_ovlp())
    if overlap.ndim == 2:
        overlap = overlap[np.newaxis, ...]

    if restricted:
        if hcore.ndim == 2:
            hcore = hcore[np.newaxis, ...]
        if fock.ndim == 2:
            fock = fock[np.newaxis, ...]
        if density.ndim == 2:
            density = density[np.newaxis, ...]
        if hcore.ndim != 3 or fock.ndim != 3 or density.ndim != 3:
            raise ValueError('Restricted periodic HF matrices must use k-point layout')
        return hcore, fock, density, overlap

    if hcore.ndim == 2:
        hcore = hcore[np.newaxis, ...]
    if hcore.ndim == 3:
        hcore = np.broadcast_to(hcore, (2,) + hcore.shape).copy()
    if fock.ndim == 3 and fock.shape[0] == 2:
        fock = fock[:, np.newaxis, ...]
    if density.ndim == 3 and density.shape[0] == 2:
        density = density[:, np.newaxis, ...]
    if hcore.ndim != 4 or fock.ndim != 4 or density.ndim != 4:
        raise ValueError('Unrestricted periodic HF matrices must use spin/k-point layout')
    return hcore, fock, density, overlap


def _selected_source_orbitals(task_spec: Any, orbital_count: int, options: Mapping[str, Any]) -> Tuple[int, ...]:
    requested = tuple(int(index) for index in task_spec.embedding.correlated_orbital_indices)
    if requested:
        selected = requested
    elif options.get('nval') is not None:
        selected = tuple(range(int(options.get('ncore') or 0), int(options['nval'])))
    else:
        selected = tuple(range(int(orbital_count)))
    if not selected:
        raise ValueError('HF+DMFT requires at least one correlated localized orbital')
    if len(set(selected)) != len(selected) or any(
        index < 0 or index >= orbital_count for index in selected
    ):
        raise ValueError('HF+DMFT correlated orbital indices are duplicated or out of range')
    return selected


def _slice_coefficients(coefficients: Any, selected: Sequence[int]) -> Any:
    import numpy as np

    # libDMET forwards this array to PySCF's low-level AO-to-MO kernels, which
    # require a contiguous coefficient buffer. Advanced indexing alone does
    # not guarantee either C- or Fortran-contiguous storage.
    return np.ascontiguousarray(np.asarray(coefficients)[..., list(selected)])


def _localized_occupations(
    density: Any,
    coefficients: Any,
    overlap: Any,
    *,
    restricted: bool,
) -> Tuple[float, ...]:
    import numpy as np

    local_density = np.asarray(transform_density_to_local(
        density,
        coefficients,
        overlap,
        layout='kpoint' if restricted else 'spin_kpoint',
    ))
    if restricted:
        averaged = np.mean(local_density, axis=0)
    else:
        averaged = np.mean(np.sum(local_density, axis=0), axis=0)
    return tuple(float(np.real(value)) for value in np.diag(averaged))


def _localized_unit_eri(
    cell: Any,
    mean_field: Any,
    coefficients: Any,
    *,
    max_memory_mb: Any,
) -> Any:
    import numpy as np
    from libdmet.basis_transform import eri_transform

    density_fitting = getattr(mean_field, 'with_df', None)
    if density_fitting is None:
        raise ValueError('HF+DMFT localized ERIs require a periodic density-fitting provider')
    eri = np.asarray(eri_transform.get_unit_eri(
        cell,
        density_fitting,
        C_ao_lo=coefficients,
        symmetry=1,
        max_memory=max_memory_mb,
        incore=True,
    ))
    if eri.ndim == 4:
        eri = eri[np.newaxis, ...]
    return eri


def _restricted_hf_effective_potential(
    cell: Any,
    mean_field: Any,
    *,
    finite_size_correction: bool,
) -> Any:
    """Evaluate the periodic HF potential on the converged restricted density."""

    import numpy as np
    from pyscf.pbc import scf

    density = np.asarray(mean_field.make_rdm1())
    kpoints = getattr(mean_field, 'kpts', None)
    if kpoints is None:
        raise ValueError('GW+DMFT requires a k-point mean-field reference')
    hf_reference = scf.KRHF(cell, kpts=kpoints, exxdiv=None)
    hf_reference.with_df = mean_field.with_df
    hf_reference.max_memory = getattr(mean_field, 'max_memory', getattr(cell, 'max_memory', 4000))
    effective_potential = np.asarray(hf_reference.get_veff(dm_kpts=density))
    if not finite_size_correction:
        return effective_potential

    coefficients = np.asarray(mean_field.mo_coeff)
    occupations = np.asarray(mean_field.mo_occ)
    overlap = np.asarray(mean_field.get_ovlp())
    if coefficients.ndim == 2:
        coefficients = coefficients[np.newaxis, ...]
    if occupations.ndim == 1:
        occupations = occupations[np.newaxis, ...]
    if overlap.ndim == 2:
        overlap = overlap[np.newaxis, ...]
    nkpts = coefficients.shape[0]
    correction = -2.0 / np.pi * (6.0 * np.pi ** 2 / float(cell.vol) / nkpts) ** (1.0 / 3.0)
    for kpoint in range(nkpts):
        occupied = coefficients[kpoint][:, occupations[kpoint] > 1.0]
        if occupied.size:
            projector = overlap[kpoint] @ occupied @ occupied.conj().T @ overlap[kpoint]
            effective_potential[kpoint] += correction * projector
    return effective_potential


def prepare_periodic_hf_subspace(
    state: Dict[str, Any],
    task_spec: Any,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    """Build reviewable localized periodic artifacts from a converged HF reference."""

    options = validate_hf_dmft_preparation_request(task_spec)
    mean_field = context.get('mean_field')
    cell = context.get('cell')
    if mean_field is None or cell is None or not bool(getattr(mean_field, 'converged', False)):
        raise ValueError('HF+DMFT subspace preparation requires a converged periodic HF reference')
    restricted = bool(context.get('restricted'))
    periodic_lattice = _periodic_lattice(cell, task_spec.periodic.kmesh, mean_field)
    localization_method = str(task_spec.embedding.localization_method).strip().lower()
    include_pao = localization_method == 'iao_pao'
    coefficients, labels = _iao_coefficients(
        periodic_lattice,
        mean_field,
        minimal_basis=task_spec.embedding.minimal_basis,
        include_pao=include_pao,
        restricted=restricted,
    )
    coefficients = _canonical_coefficients(coefficients, restricted=restricted)
    hcore, fock, density, overlap = _canonical_reference_matrices(
        mean_field,
        restricted=restricted,
    )
    selected_source = _selected_source_orbitals(
        task_spec,
        coefficients.shape[-1],
        options,
    )
    coefficients = _slice_coefficients(coefficients, selected_source)
    selected_labels = tuple(labels[index] for index in selected_source)
    occupations = _localized_occupations(
        density,
        coefficients,
        overlap,
        restricted=restricted,
    )
    eri = _localized_unit_eri(
        cell,
        mean_field,
        coefficients,
        max_memory_mb=options.get('max_memory_mb'),
    )
    spin_blocks = 1 if restricted else 3
    expected_eri_shape = (
        spin_blocks,
        coefficients.shape[-1],
        coefficients.shape[-1],
        coefficients.shape[-1],
        coefficients.shape[-1],
    )
    if eri.shape != expected_eri_shape:
        raise ValueError(
            'libDMET returned localized ERIs with shape {0}; expected {1}'.format(
                eri.shape,
                expected_eri_shape,
            )
        )

    remapped_indices = tuple(range(coefficients.shape[-1]))
    prepared = prepare_embedding_artifacts(
        state,
        system_type='periodic',
        one_body=hcore,
        fock=fock,
        density=density,
        overlap=overlap,
        coefficients=coefficients,
        two_body=eri,
        two_body_spin_pair_order='restricted' if restricted else 'aa_bb_ab',
        correlated_orbital_indices=remapped_indices,
        provider='fcdmft',
        localization_method=localization_method,
        matrix_layout='kpoint' if restricted else 'spin_kpoint',
        orbital_labels=selected_labels,
        occupations=occupations,
        orbital_contributions={
            index: {'source_localized_orbital_index': int(source_index)}
            for index, source_index in enumerate(selected_source)
        },
        interaction={
            'source': 'libdmet_periodic_unit_eri',
            'representation': 'localized_unit_cell_eri',
            'spin_pair_order': 'restricted' if restricted else 'aa_bb_ab',
        },
        selection_reasons=(
            'Selected from the periodic HF reference using {0}.'.format(localization_method),
            'The approved fcDMFT basis will contain only these localized orbitals.',
        ),
        electron_count={
            'periodic_cell': int(context.get('electron_count') or 0),
            'correlated_subspace': float(sum(occupations)),
        },
    )
    patch = copy.deepcopy(prepared['task_spec_patch'])
    patch['solver'] = {
        'name': 'hf_dmft',
        'options': {
            **copy.deepcopy(options),
            'ncore': 0,
            'nval': len(remapped_indices),
            'nb_per_e': options.get('nb_per_e') or len(remapped_indices),
        },
    }
    prepared['task_spec_patch'] = patch
    prepared['source_orbital_indices'] = list(selected_source)
    prepared['localized_orbital_count'] = len(remapped_indices)
    prepared['estimated_eri_memory_mb'] = float(eri.nbytes) / (1024.0 * 1024.0)
    return prepared


def prepare_periodic_fcdmft_subspace(
    state: Dict[str, Any],
    task_spec: Any,
    context: Mapping[str, Any],
) -> Dict[str, Any]:
    solver_name = str(task_spec.solver.name or '').strip().lower().replace('-', '_')
    if solver_name == 'hf_dmft':
        return prepare_periodic_hf_subspace(state, task_spec, context)

    options = validate_fcdmft_preparation_request(task_spec)
    mean_field = context.get('mean_field')
    cell = context.get('cell')
    if mean_field is None or cell is None or not bool(getattr(mean_field, 'converged', False)):
        raise ValueError('GW+DMFT subspace preparation requires a converged periodic DFT reference')
    restricted = bool(context.get('restricted'))
    if not restricted:
        raise ValueError('GW+DMFT subspace preparation currently requires a restricted reference')
    periodic_lattice = _periodic_lattice(cell, task_spec.periodic.kmesh, mean_field)
    localization_method = str(task_spec.embedding.localization_method).strip().lower()
    coefficients, labels = _iao_coefficients(
        periodic_lattice,
        mean_field,
        minimal_basis=task_spec.embedding.minimal_basis,
        include_pao=localization_method == 'iao_pao',
        restricted=True,
    )
    coefficients = _canonical_coefficients(coefficients, restricted=True)
    hcore, fock, density, overlap = _canonical_reference_matrices(
        mean_field,
        restricted=True,
    )
    selected = _selected_source_orbitals(task_spec, coefficients.shape[-1], options)
    if tuple(selected) != tuple(range(selected[0], selected[-1] + 1)):
        raise ValueError('GW+DMFT correlated orbitals must form one contiguous localized window')
    occupations = _localized_occupations(
        density,
        coefficients,
        overlap,
        restricted=True,
    )
    eri = _localized_unit_eri(
        cell,
        mean_field,
        coefficients,
        max_memory_mb=options.get('max_memory_mb'),
    )
    hf_effective_potential = _restricted_hf_effective_potential(
        cell,
        mean_field,
        finite_size_correction=bool(options['gw_finite_size_correction']),
    )
    prepared = prepare_embedding_artifacts(
        state,
        system_type='periodic',
        one_body=hcore,
        fock=fock,
        density=density,
        overlap=overlap,
        coefficients=coefficients,
        two_body=eri,
        two_body_spin_pair_order='restricted',
        correlated_orbital_indices=selected,
        provider='fcdmft',
        localization_method=localization_method,
        matrix_layout='kpoint',
        orbital_labels=labels,
        occupations=occupations,
        orbital_contributions={
            index: {'source_localized_orbital_index': int(index)}
            for index in range(coefficients.shape[-1])
        },
        interaction={
            'source': 'libdmet_periodic_unit_eri',
            'representation': 'localized_unit_cell_eri',
            'spin_pair_order': 'restricted',
        },
        selection_reasons=(
            'Localized the periodic DFT reference using {0}.'.format(localization_method),
            'Retained the full localized basis for lattice GW and selected one contiguous DMFT window.',
        ),
        electron_count={
            'periodic_cell': int(context.get('electron_count') or 0),
            'correlated_subspace': float(sum(occupations[index] for index in selected)),
        },
        additional_one_body_operators={
            'hf_effective_potential_local': hf_effective_potential,
        },
    )
    patch = copy.deepcopy(prepared['task_spec_patch'])
    patch['solver'] = {
        'name': 'gw_dmft',
        'options': {
            **copy.deepcopy(options),
            'ncore': int(selected[0]),
            'nval': int(selected[-1]) + 1,
            'nb_per_e': options.get('nb_per_e') or len(selected),
        },
    }
    prepared['task_spec_patch'] = patch
    prepared['source_orbital_indices'] = list(selected)
    prepared['localized_orbital_count'] = int(coefficients.shape[-1])
    prepared['estimated_eri_memory_mb'] = float(eri.nbytes) / (1024.0 * 1024.0)
    return prepared


__all__ = [
    'prepare_periodic_hf_subspace',
    'prepare_periodic_fcdmft_subspace',
    'validate_fcdmft_preparation_request',
    'validate_hf_dmft_preparation_request',
]
