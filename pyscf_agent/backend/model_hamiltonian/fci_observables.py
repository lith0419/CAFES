from __future__ import annotations

from typing import Any, Dict, List


FULL_DIAGONALIZATION_MAX_DIMENSION = 5_000
FULL_DIAGONALIZATION_WORKING_SET_MULTIPLIER = 5


def fci_determinant_space_dimension(norb: int, nelec: Any) -> int:
    """Return the spin-resolved FCI determinant-space dimension."""
    from pyscf.fci import cistring  # pylint: disable=import-outside-toplevel

    electrons = tuple(nelec)
    return int(
        cistring.num_strings(int(norb), int(electrons[0]))
        * cistring.num_strings(int(norb), int(electrons[1]))
    )


def full_diagonalization_resource_estimate(dimension: int) -> Dict[str, Any]:
    """Conservative resources for the dense matrix construction plus eigensolve."""
    dimension = int(dimension)
    matrix_bytes = dimension * dimension * 8
    return {
        'dimension': dimension,
        'max_dimension': FULL_DIAGONALIZATION_MAX_DIMENSION,
        'supported': dimension <= FULL_DIAGONALIZATION_MAX_DIMENSION,
        'matrix_memory_mb': round(matrix_bytes / (1024 ** 2), 2),
        'working_memory_mb': round(
            matrix_bytes * FULL_DIAGONALIZATION_WORKING_SET_MULTIPLIER / (1024 ** 2),
            2,
        ),
        'work_units': dimension ** 3,
    }


def require_supported_full_diagonalization(norb: int, nelec: Any) -> Dict[str, Any]:
    dimension = fci_determinant_space_dimension(norb, nelec)
    estimate = full_diagonalization_resource_estimate(dimension)
    if not estimate['supported']:
        raise ValueError(
            'Full FCI diagonalization for strong-correlation diagnostics requires '
            'a determinant space no larger than {limit:,}; this request has {dimension:,}. '
            'Reduce the system or request FCI energy without full-spectrum diagnostics.'.format(
                limit=FULL_DIAGONALIZATION_MAX_DIMENSION,
                dimension=dimension,
            )
        )
    return estimate


def _compute_model_hamiltonian_energy_levels(h1e, eri, metadata: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel
    from pyscf import fci  # pylint: disable=import-outside-toplevel
    from pyscf.fci import cistring  # pylint: disable=import-outside-toplevel

    norb = metadata['norb']
    nelec = tuple(metadata['nelec'])
    na = cistring.num_strings(norb, nelec[0])
    nb = cistring.num_strings(norb, nelec[1])
    dimension = int(na * nb)
    resource_estimate = require_supported_full_diagonalization(norb, nelec)
    h2e = fci.direct_spin1.absorb_h1e(h1e, eri, norb, nelec, fac=0.5)
    columns = []
    for column_index in range(dimension):
        ci_vector = np.zeros((na, nb), dtype=float)
        ci_vector.reshape(-1)[column_index] = 1.0
        columns.append(fci.direct_spin1.contract_2e(h2e, ci_vector, norb, nelec).reshape(-1))
    hamiltonian = np.column_stack(columns)
    hamiltonian = 0.5 * (hamiltonian + hamiltonian.T)
    levels = np.linalg.eigvalsh(hamiltonian)
    return {
        'method': 'fci',
        'basis_dimension': dimension,
        'levels': [float(value) for value in levels],
        'diagonalization': {
            'kind': 'full_dense',
            **resource_estimate,
        },
    }


def _fci_occupation_probabilities(ci_vector, metadata: Dict[str, Any]) -> Dict[str, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel
    from pyscf.fci import cistring  # pylint: disable=import-outside-toplevel

    norb = metadata['norb']
    nelec = tuple(metadata['nelec'])
    alpha_strings = cistring.gen_strings4orblist(range(norb), nelec[0])
    beta_strings = cistring.gen_strings4orblist(range(norb), nelec[1])
    coefficients = np.asarray(ci_vector).reshape(len(alpha_strings), len(beta_strings))
    probabilities = np.abs(coefficients) ** 2
    norm = float(probabilities.sum())
    return {
        'norb': norb,
        'alpha_strings': alpha_strings,
        'beta_strings': beta_strings,
        'probabilities': probabilities,
        'norm': norm,
    }


def _fci_site_occupations(occupation: Dict[str, Any]):
    import numpy as np  # pylint: disable=import-outside-toplevel

    site_occupations = []
    for orb in range(occupation['norb']):
        mask = 1 << orb
        alpha_occupation = np.array([1.0 if int(string) & mask else 0.0 for string in occupation['alpha_strings']])
        beta_occupation = np.array([1.0 if int(string) & mask else 0.0 for string in occupation['beta_strings']])
        site_occupations.append(alpha_occupation[:, None] + beta_occupation[None, :])
    return site_occupations


def _compute_fci_site_density(ci_vector, metadata: Dict[str, Any]) -> List[float]:
    occupation = _fci_occupation_probabilities(ci_vector, metadata)
    probabilities = occupation['probabilities']
    norm = occupation['norm']
    if norm <= 0:
        return [0.0 for _index in range(occupation['norb'])]

    density = []
    for site_occupation in _fci_site_occupations(occupation):
        density.append(float((probabilities * site_occupation).sum() / norm))
    return density


def _compute_fci_double_occupancy(ci_vector, metadata: Dict[str, Any]) -> List[float]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    occupation = _fci_occupation_probabilities(ci_vector, metadata)
    norb = occupation['norb']
    probabilities = occupation['probabilities']
    norm = occupation['norm']
    if norm <= 0:
        return [0.0 for _index in range(norb)]

    values = []
    for orb in range(norb):
        mask = 1 << orb
        alpha_occupation = np.array([1.0 if int(string) & mask else 0.0 for string in occupation['alpha_strings']])
        beta_occupation = np.array([1.0 if int(string) & mask else 0.0 for string in occupation['beta_strings']])
        value = float((probabilities * alpha_occupation[:, None] * beta_occupation[None, :]).sum() / norm)
        values.append(value)
    return values


def _compute_fci_spin_correlation(ci_vector, metadata: Dict[str, Any]) -> List[List[float]]:
    import numpy as np  # pylint: disable=import-outside-toplevel
    from pyscf.fci import direct_spin1  # pylint: disable=import-outside-toplevel

    norb = metadata['norb']
    nelec = tuple(metadata['nelec'])
    (dm1a, dm1b), (dm2aa, dm2ab, dm2bb) = direct_spin1.make_rdm12s(ci_vector, norb, nelec)
    matrix = np.zeros((norb, norb), dtype=float)
    for i in range(norb):
        for j in range(norb):
            if i == j:
                density = dm1a[i, i] + dm1b[i, i]
                double_occupancy = dm2ab[i, i, i, i]
                matrix[i, j] = 0.75 * (density - 2.0 * double_occupancy)
            else:
                same_spin = dm2aa[i, i, j, j] + dm2bb[i, i, j, j]
                opposite_spin = dm2ab[i, i, j, j] + dm2ab[j, j, i, i]
                spin_z = 0.25 * (same_spin - opposite_spin)
                spin_flip = -0.5 * (dm2ab[i, j, j, i] + dm2ab[j, i, i, j])
                matrix[i, j] = spin_z + spin_flip
    return matrix.tolist()


def _compute_fci_charge_correlation(ci_vector, metadata: Dict[str, Any]) -> List[List[float]]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    occupation = _fci_occupation_probabilities(ci_vector, metadata)
    norb = occupation['norb']
    probabilities = occupation['probabilities']
    norm = occupation['norm']
    if norm <= 0:
        return np.zeros((norb, norb), dtype=float).tolist()

    site_occupations = _fci_site_occupations(occupation)
    densities = [float((probabilities * site_occupation).sum() / norm) for site_occupation in site_occupations]
    matrix = np.zeros((norb, norb), dtype=float)
    for i in range(norb):
        for j in range(norb):
            density_product = float((probabilities * site_occupations[i] * site_occupations[j]).sum() / norm)
            matrix[i, j] = density_product - densities[i] * densities[j]
    return matrix.tolist()


def _compute_site_density_from_spin_rdms(dm1_alpha: Any, dm1_beta: Any) -> List[float]:
    """Return <n_i> from spin-resolved one-particle density matrices."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    density = np.diag(np.asarray(dm1_alpha) + np.asarray(dm1_beta)).real
    return density.astype(float).tolist()


def _compute_double_occupancy_from_spin_rdms(dm2_alpha_beta: Any) -> List[float]:
    """Return <n_i,alpha n_i,beta> using PySCF's spin-RDM convention."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    dm2ab = np.asarray(dm2_alpha_beta)
    return [float(dm2ab[index, index, index, index].real) for index in range(dm2ab.shape[0])]


def _compute_spin_correlation_from_spin_rdms(
    dm1_alpha: Any,
    dm1_beta: Any,
    dm2_alpha_alpha: Any,
    dm2_alpha_beta: Any,
    dm2_beta_beta: Any,
) -> List[List[float]]:
    """Return <S_i dot S_j> from spin-resolved one- and two-particle RDMs."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    dm1a = np.asarray(dm1_alpha)
    dm1b = np.asarray(dm1_beta)
    dm2aa = np.asarray(dm2_alpha_alpha)
    dm2ab = np.asarray(dm2_alpha_beta)
    dm2bb = np.asarray(dm2_beta_beta)
    norb = dm1a.shape[0]
    matrix = np.zeros((norb, norb), dtype=float)
    for i in range(norb):
        for j in range(norb):
            if i == j:
                density = dm1a[i, i] + dm1b[i, i]
                double_occupancy = dm2ab[i, i, i, i]
                matrix[i, j] = float((0.75 * (density - 2.0 * double_occupancy)).real)
            else:
                same_spin = dm2aa[i, i, j, j] + dm2bb[i, i, j, j]
                opposite_spin = dm2ab[i, i, j, j] + dm2ab[j, j, i, i]
                spin_z = 0.25 * (same_spin - opposite_spin)
                spin_flip = -0.5 * (dm2ab[i, j, j, i] + dm2ab[j, i, i, j])
                matrix[i, j] = float((spin_z + spin_flip).real)
    return matrix.tolist()


def _compute_charge_correlation_from_spin_rdms(
    dm1_alpha: Any,
    dm1_beta: Any,
    dm2_alpha_alpha: Any,
    dm2_alpha_beta: Any,
    dm2_beta_beta: Any,
) -> List[List[float]]:
    """Return connected <(n_i-<n_i>)(n_j-<n_j>)> from spin RDMs."""
    import numpy as np  # pylint: disable=import-outside-toplevel

    dm1a = np.asarray(dm1_alpha)
    dm1b = np.asarray(dm1_beta)
    dm2aa = np.asarray(dm2_alpha_alpha)
    dm2ab = np.asarray(dm2_alpha_beta)
    dm2bb = np.asarray(dm2_beta_beta)
    density = np.diag(dm1a + dm1b).real
    norb = dm1a.shape[0]
    matrix = np.zeros((norb, norb), dtype=float)
    for i in range(norb):
        for j in range(norb):
            if i == j:
                density_product = density[i] + 2.0 * dm2ab[i, i, i, i].real
            else:
                density_product = (
                    dm2aa[i, i, j, j]
                    + dm2bb[i, i, j, j]
                    + dm2ab[i, i, j, j]
                    + dm2ab[j, j, i, i]
                ).real
            matrix[i, j] = float(density_product - density[i] * density[j])
    return matrix.tolist()


def _site_ids_from_spec(spec: Dict[str, Any]) -> List[int]:
    return [int(site['id']) for site in sorted(spec['sites'], key=lambda item: int(item['id']))]


def _site_vector_payload(values: List[float], site_ids: List[int], *, operator: str) -> Dict[str, Any]:
    return {
        'kind': 'site_vector',
        'site_ids': list(site_ids),
        'shape': [len(site_ids)],
        'values': values,
        'operator': operator,
    }


def _pair_matrix_payload(values: List[List[float]], site_ids: List[int], *, operator: str) -> Dict[str, Any]:
    return {
        'kind': 'pair_matrix',
        'site_ids': list(site_ids),
        'shape': [len(site_ids), len(site_ids)],
        'values': values,
        'operator': operator,
    }
