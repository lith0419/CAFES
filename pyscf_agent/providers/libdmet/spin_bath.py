"""Explicit max-dimension completion for finite-graph eigenvalue baths."""

from __future__ import annotations


def max_spin_eigen_bath(lattice, density, *, tolerance=1e-9):
    """Retain selected eigenvectors and complete smaller channels orthonormally.

    Only all-valence finite graphs are supported. Added orbitals are the omitted
    environment eigenvectors closest to fractional occupation, with eigenvalue
    order breaking ties. No density or electron-count rescaling is performed.
    """
    import numpy as np
    from scipy.linalg import eigh

    impurity = list(lattice.imp_idx)
    if lattice.ncells != 1 or set(lattice.val_idx) != set(impurity):
        raise ValueError(
            'Max spin bath requires a single-cell all-valence finite graph'
        )
    nsite = int(lattice.nscsites)
    environment = [i for i in range(nsite) if i not in impurity]
    rho = np.asarray(density)
    if rho.ndim == 3:
        rho = rho[None]
    full = np.asarray(lattice.expand(rho))
    if not np.all(np.isfinite(full)):
        raise ValueError('Nonfinite density in max spin bath')
    spectra, orbitals, selected = [], [], []
    for block in full[:, environment][:, :, environment]:
        if not np.allclose(block, block.conj().T, atol=1e-10, rtol=0):
            raise ValueError('Non-Hermitian environment density in max spin bath')
        values, vectors = eigh(block)
        if np.any(values < -1e-7) or np.any(values > 1 + 1e-7):
            raise ValueError(
                'Environment density eigenvalues outside physical [0,1] range'
            )
        keep = np.flatnonzero(
            (np.abs(values) > tolerance) & (np.abs(1 - values) > tolerance)
        )
        spectra.append(values)
        orbitals.append(vectors)
        selected.append(keep.tolist())
    counts = [len(indices) for indices in selected]
    target = max(counts)
    basis = np.zeros(
        (len(counts), nsite, len(impurity) + target),
        dtype=np.result_type(full.dtype, float),
    )
    additions = []
    for spin, (values, vectors, keep) in enumerate(zip(spectra, orbitals, selected)):
        omitted = [i for i in range(len(values)) if i not in keep]
        omitted.sort(key=lambda i: (-min(abs(values[i]), abs(1 - values[i])), i))
        added = omitted[: target - len(keep)]
        chosen = keep + added
        basis[spin, impurity, : len(impurity)] = np.eye(len(impurity))
        basis[spin, environment, len(impurity) :] = vectors[:, chosen]
        additions.append({'indices': added, 'eigenvalues': values[added].tolist()})
    overlap = basis.conj().transpose(0, 2, 1) @ basis
    error = float(np.max(np.abs(overlap - np.eye(basis.shape[-1]))))
    if error > 1e-10:
        raise ValueError('Max spin bath failed full embedding orthogonality check')
    diagnostics = {
        'policy': 'max',
        'original_bath_counts': counts,
        'common_bath_count': target,
        'added_environment_orbitals': additions,
        'orthogonality_max_abs': error,
        'tol_bath': tolerance,
        'electron_sector_policy': 'existing_fractional_valence_sector',
    }
    return basis[:, None], diagnostics
