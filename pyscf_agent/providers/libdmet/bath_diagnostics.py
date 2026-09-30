"""Read-only snapshots of failed bath construction, without changing selection."""

from pathlib import Path
import json


def save_bath_failure(directory, lattice, density, mean_field, potential, *, iteration):
    import numpy as np
    from scipy import linalg

    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    full = np.asarray(lattice.expand(density))
    impurity = list(lattice.imp_idx)
    bath_impurity = list(lattice.val_idx)
    environment = [i for i in range(full.shape[-1]) if i not in bath_impurity]
    spectra = np.asarray([
        linalg.eigh(dm[np.ix_(environment, environment)].real)[0] for dm in full
    ])
    mask = (np.abs(spectra) > 1e-9) & (np.abs(1 - spectra) > 1e-9)
    arrays = {
        'density': np.asarray(density), 'full_density': full,
        'environment_eigenvalues': spectra,
        'impurity_indices': np.asarray(impurity),
        'bath_impurity_indices': np.asarray(bath_impurity),
        'environment_indices': np.asarray(environment),
        'correlation_potential': np.asarray(potential.get()),
    }
    for key in ('mo_occ', 'mo_energy', 'mo_coeff', 'rho_k'):
        if mean_field.get(key) is not None:
            arrays[key] = np.asarray(mean_field[key])
    summary = {
        'iteration': iteration,
        'impurity_indices': impurity,
        'bath_counts': mask.sum(axis=1).tolist(),
        'tol_bath': 1e-9,
        'environment_eigenvalues': spectra.tolist(),
        'selected_environment_eigenvalues': [e[m].tolist() for e, m in zip(spectra, mask)],
        'full_density_eigenvalues': np.linalg.eigvalsh(full).tolist(),
        'electron_counts': np.trace(full, axis1=-2, axis2=-1).real.tolist(),
        'idempotency_errors': [float(np.linalg.norm(d @ d - d)) for d in full],
        'mo_occ': arrays['mo_occ'].tolist() if 'mo_occ' in arrays else None,
        'mo_energy': arrays['mo_energy'].tolist() if 'mo_energy' in arrays else None,
    }
    np.savez_compressed(directory / 'bath-failure.npz', **arrays)
    (directory / 'bath-failure.json').write_text(json.dumps(summary, indent=2) + '\n')
