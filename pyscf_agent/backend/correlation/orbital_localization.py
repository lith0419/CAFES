"""Localize a fixed orbital subspace, optionally partitioned by occupations."""
from __future__ import annotations

from typing import Any, Dict, Tuple

from ...contracts import normalize_orbital_processing


def localize_orbital_block(
    mol: Any, coefficients: Any, *, method: str,
    occupation_thresholds: Any = (), reference_rdm1: Any = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Use the spin-summed density in this orbital basis to define subspaces.

    Empty thresholds preserve whole-block localization. Otherwise diagonalize
    the reference density first, then localize each occupation interval without
    mixing intervals. This works for UNO or rotated/optimized input orbitals;
    orbital positions are never used as proxies for occupations.
    """
    import numpy as np
    from pyscf import lo

    options = normalize_orbital_processing({
        'localization_method': method, 'localization_scope': 'active_space',
        'localization_occupation_thresholds': list(occupation_thresholds),
    })
    method = options['localization_method']
    if method not in ('boys', 'pipek_mezey'):
        raise ValueError('An orbital localization method is required')
    thresholds = options['localization_occupation_thresholds']
    original = np.asarray(coefficients, dtype=float)
    overlap = mol.intor_symmetric('int1e_ovlp')
    if original.ndim != 2 or original.shape[0] != overlap.shape[0] or original.shape[1] == 0:
        raise ValueError('Localization requires a nonempty AO orbital matrix')
    count = original.shape[1]
    if not np.all(np.isfinite(original)) or np.max(np.abs(original.T @ overlap @ original - np.eye(count))) > 1e-7:
        raise ValueError('Localization input orbitals must be finite and AO-metric orthonormal')
    orbitals = original.copy()
    occupations = None
    groups = [np.arange(count)]
    if thresholds:
        density = np.asarray(reference_rdm1, dtype=float)
        if density.shape != (count, count) or not np.all(np.isfinite(density)):
            raise ValueError('Split localization requires a finite spin-summed reference 1-RDM in the orbital basis')
        if np.max(np.abs(density - density.T)) > 1e-7:
            raise ValueError('Split-localization reference 1-RDM must be Hermitian')
        occupations, rotation = np.linalg.eigh((density + density.T) * .5)
        occupations, rotation = occupations[::-1], rotation[:, ::-1]
        if occupations.min() < -1e-7 or occupations.max() > 2 + 1e-7:
            raise ValueError('Split-localization reference occupations must lie between 0 and 2')
        orbitals = original @ rotation
        # Exactly on a threshold belongs to the lower-occupation interval,
        # matching the split_low/split_high convention in block2's UNO example.
        bins = np.searchsorted(thresholds, occupations, side='left')
        groups = [np.flatnonzero(bins == index) for index in range(len(thresholds) + 1)]
    localized = orbitals.copy()
    group_records = []
    for index, columns in enumerate(groups):
        record = {'interval': index, 'orbital_indices': columns.tolist(), 'count': len(columns)}
        if len(columns) > 1:
            localizer = (lo.PM if method == 'pipek_mezey' else lo.Boys)(mol, orbitals[:, columns])
            trace = {}
            def observe(state):
                trace.update(converged=bool(state['conv']), iterations=int(state['imacro']) + 1,
                             gradient_norm=float(state['norm_gorb']))
            localized[:, columns] = localizer.kernel(callback=observe)
            record.update(trace)
        else:
            record['status'] = 'empty' if len(columns) == 0 else 'single_orbital'
        group_records.append(record)
    rotation = original.T @ overlap @ localized
    orthogonality_error = float(np.max(np.abs(localized.T @ overlap @ localized - np.eye(count))))
    subspace_error = float(np.max(np.abs(rotation.T @ rotation - np.eye(count))))
    if not np.all(np.isfinite(localized)) or max(orthogonality_error, subspace_error) > 1e-7:
        raise ValueError('Localization did not preserve the orthonormal input subspace')
    return localized, {
        'localization_occupation_thresholds': thresholds,
        'localization_partition': 'occupation_intervals' if thresholds else 'whole_active_space',
        'reference_natural_occupations': occupations.tolist() if occupations is not None else None,
        'localization_groups': group_records,
        'orthonormality_max_abs_error': orthogonality_error,
        'subspace_max_abs_error': subspace_error,
    }
