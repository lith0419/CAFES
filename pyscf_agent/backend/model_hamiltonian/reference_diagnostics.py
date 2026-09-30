from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple


FRONTIER_DEGENERACY_TOLERANCE = 0.02
FRONTIER_OCCUPATION_EPS = 1.0e-8
SPIN_SYMMETRY_BREAKING_TOLERANCE = 0.2


def _spin_orbital_channels(mo_energy: Any, mo_occ: Any) -> List[Tuple[str, Any, Any]]:
    energy_ndim = getattr(mo_energy, 'ndim', None)
    occ_ndim = getattr(mo_occ, 'ndim', None)
    if energy_ndim == 2 or occ_ndim == 2:
        return [('alpha', mo_energy[0], mo_occ[0]), ('beta', mo_energy[1], mo_occ[1])]
    try:
        energy_items = list(mo_energy)
        occ_items = list(mo_occ)
    except TypeError:
        return [('restricted', mo_energy, mo_occ)]
    looks_like_spin_pair = (
        len(energy_items) == 2
        and len(occ_items) == 2
        and any(hasattr(item, '__len__') for item in energy_items + occ_items)
    )
    if looks_like_spin_pair:
        return [('alpha', energy_items[0], occ_items[0]), ('beta', energy_items[1], occ_items[1])]
    return [('restricted', mo_energy, mo_occ)]


def _clip_unit(value: float) -> float:
    return max(0.0, min(1.0, float(value)))


def _frontier_degeneracy_from_spin_channels(
    spin_channels: Dict[str, Dict[str, Any]],
    *,
    tolerance: float = FRONTIER_DEGENERACY_TOLERANCE,
    evaluate_alpha_beta: bool = False,
) -> Dict[str, Any]:
    if not spin_channels:
        return {
            'status': 'unavailable',
            'reason': 'mean-field spin-channel frontiers are unavailable',
            'tolerance': tolerance,
        }

    same_spin_clusters = []
    for label, summary in spin_channels.items():
        for frontier in ('homo', 'lumo'):
            count = int(summary.get('{0}_degeneracy'.format(frontier)) or 0)
            if count > 1:
                same_spin_clusters.append({
                    'channel': label,
                    'frontier': frontier,
                    'count': count,
                    'indices': summary.get('{0}_indices'.format(frontier), []),
                })
    max_same_spin_degeneracy = max(
        [
            int(summary.get('homo_degeneracy') or 0)
            for summary in spin_channels.values()
        ] + [
            int(summary.get('lumo_degeneracy') or 0)
            for summary in spin_channels.values()
        ] + [0]
    )
    same_spin_score = _clip_unit((max_same_spin_degeneracy - 1.0) / 2.0)

    alpha_beta_pairs = []
    alpha_beta_score = 0.0
    if evaluate_alpha_beta and 'alpha' in spin_channels and 'beta' in spin_channels:
        frontier_values = {
            'alpha_homo': spin_channels['alpha'].get('homo'),
            'alpha_lumo': spin_channels['alpha'].get('lumo'),
            'beta_homo': spin_channels['beta'].get('homo'),
            'beta_lumo': spin_channels['beta'].get('lumo'),
        }
        for left, right in (
            ('alpha_homo', 'beta_homo'),
            ('alpha_lumo', 'beta_lumo'),
            ('alpha_homo', 'beta_lumo'),
            ('alpha_lumo', 'beta_homo'),
        ):
            left_value = frontier_values.get(left)
            right_value = frontier_values.get(right)
            if left_value is None or right_value is None:
                continue
            spacing = abs(float(left_value) - float(right_value))
            pair_score = _clip_unit(1.0 - spacing / max(float(tolerance), 1.0e-12))
            alpha_beta_score = max(alpha_beta_score, pair_score)
            alpha_beta_pairs.append({
                'pair': '{0}-{1}'.format(left, right),
                'energy_difference': spacing,
                'degenerate': spacing <= tolerance,
            })

    score = max(same_spin_score, alpha_beta_score)
    return {
        'status': 'available',
        'tolerance': tolerance,
        'score': round(float(score), 6),
        'max_same_spin_degeneracy': max_same_spin_degeneracy,
        'same_spin_degenerate_clusters': same_spin_clusters,
        'alpha_beta': {
            'evaluated': bool(evaluate_alpha_beta and 'alpha' in spin_channels and 'beta' in spin_channels),
            'pairs': alpha_beta_pairs,
            'degenerate_pairs': [
                item for item in alpha_beta_pairs
                if item.get('degenerate')
            ],
            'score': round(float(alpha_beta_score), 6),
        },
    }


def _alpha_beta_frontier_is_relevant(mf: Any) -> bool:
    mol = getattr(mf, 'mol', None)
    try:
        spin_value = int(getattr(mol, 'spin', 0) or 0)
    except (TypeError, ValueError):
        spin_value = 0
    if spin_value:
        return True
    try:
        spin_square = mf.spin_square()
        value = spin_square[0] if isinstance(spin_square, tuple) else spin_square
        return abs(float(value)) > SPIN_SYMMETRY_BREAKING_TOLERANCE
    except Exception:
        return False


def _mean_field_homo_lumo_summary(mf: Any) -> Optional[Dict[str, Any]]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    mo_energy = getattr(mf, 'mo_energy', None)
    mo_occ = getattr(mf, 'mo_occ', None)
    if mo_energy is None or mo_occ is None:
        return None

    occupied = []
    virtual = []
    spin_channels: Dict[str, Dict[str, Optional[float]]] = {}
    for label, energy_values, occ_values in _spin_orbital_channels(mo_energy, mo_occ):
        try:
            energies = np.asarray(energy_values, dtype=float).reshape(-1)
            occupations = np.asarray(occ_values, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            continue
        if energies.size != occupations.size or energies.size == 0:
            continue
        indexed_rows = [
            {'index': int(index), 'energy': float(energy), 'occupation': float(occ)}
            for index, (energy, occ) in enumerate(zip(energies, occupations))
        ]
        occupied_rows = sorted(
            (row for row in indexed_rows if row['occupation'] > FRONTIER_OCCUPATION_EPS),
            key=lambda row: row['energy'],
        )
        virtual_rows = sorted(
            (row for row in indexed_rows if row['occupation'] <= FRONTIER_OCCUPATION_EPS),
            key=lambda row: row['energy'],
        )
        channel_occupied = [row['energy'] for row in occupied_rows]
        channel_virtual = [row['energy'] for row in virtual_rows]
        homo = channel_occupied[-1] if channel_occupied else None
        lumo = channel_virtual[0] if channel_virtual else None
        homo_cluster = [
            row for row in occupied_rows
            if homo is not None and abs(row['energy'] - float(homo)) <= FRONTIER_DEGENERACY_TOLERANCE
        ]
        lumo_cluster = [
            row for row in virtual_rows
            if lumo is not None and abs(row['energy'] - float(lumo)) <= FRONTIER_DEGENERACY_TOLERANCE
        ]
        spin_channels[label] = {
            'homo': homo,
            'lumo': lumo,
            'gap': None if homo is None or lumo is None else float(lumo - homo),
            'homo_degeneracy': len(homo_cluster) if homo is not None else 0,
            'lumo_degeneracy': len(lumo_cluster) if lumo is not None else 0,
            'homo_indices': [row['index'] for row in homo_cluster],
            'lumo_indices': [row['index'] for row in lumo_cluster],
        }
        occupied.extend(channel_occupied)
        virtual.extend(channel_virtual)
    if not occupied or not virtual:
        return None
    homo = max(occupied)
    lumo = min(virtual)
    channel_gaps = [
        value['gap']
        for value in spin_channels.values()
        if value.get('gap') is not None
    ]
    return {
        'homo': float(homo),
        'lumo': float(lumo),
        'gap': float(lumo - homo),
        'min_spin_channel_gap': None if not channel_gaps else float(min(channel_gaps)),
        'spin_channels': spin_channels,
        'frontier_degeneracy': _frontier_degeneracy_from_spin_channels(
            spin_channels,
            evaluate_alpha_beta=_alpha_beta_frontier_is_relevant(mf),
        ),
        'reference': mf.__class__.__name__.lower(),
    }


def _spin_summed_density_matrix(dm1: Any):
    import numpy as np  # pylint: disable=import-outside-toplevel

    try:
        density = np.asarray(dm1)
    except (TypeError, ValueError):
        return None
    if density.ndim == 3 and density.shape[0] == 2:
        return density[0] + density[1]
    return density


def _natural_occupation_summary_from_dm1(
    dm1: Any, *, nelec: Any = None, source: str = 'correlated_1rdm',
    scope: str = 'full_system',
) -> Optional[Dict[str, Any]]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    matrix = _spin_summed_density_matrix(dm1)
    if matrix is None or matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.size == 0:
        return None
    if not np.all(np.isfinite(matrix)):
        return {'status': 'failed', 'reason': 'Density matrix contains nonfinite values'}
    hermitian = 0.5 * (matrix + matrix.conj().T)
    occupations = sorted((float(value) for value in np.linalg.eigvalsh(hermitian)), reverse=True)
    if any(value < -1e-6 or value > 2.0 + 1e-6 for value in occupations):
        return {
            'status': 'out_of_bounds', 'occupations': occupations,
            'reason': 'Natural occupations are outside [0, 2] beyond numerical tolerance',
            'average_fractionality': None, 'max_fractionality': None,
        }
    clipped = [max(0.0, min(2.0, value)) for value in occupations]
    fractionality_values = [1.0 - abs(value - 1.0) for value in clipped]
    fractional_orbitals = [value for value in clipped if 0.02 < value < 1.98]
    average_fractionality = float(np.mean(fractionality_values)) if fractionality_values else 0.0
    max_fractionality = max(fractionality_values, default=0.0)
    summary = {
        'status': 'available',
        'source': source,
        'scope': scope,
        'occupations': occupations,
        'frontier_occupations': [value for value in clipped if 0.02 < value < 1.98],
        'average_fractionality': float(max(0.0, min(1.0, average_fractionality))),
        'max_fractionality': float(max(0.0, min(1.0, max_fractionality))),
        'fractional_orbital_count': len(fractional_orbitals),
    }
    density = np.asarray(dm1)
    spin_summary = {
        'status': 'unavailable', 'score': None,
        'reason': 'Spin-resolved 1RDM is required; spatial SOMOs are not correlation evidence.',
        'source': source, 'scope': scope,
    }
    summary['spin_resolved'] = spin_summary
    if density.ndim != 3 or density.shape[0] != 2:
        return summary
    channels = {}
    fractionality = []
    for index, label in enumerate(('alpha', 'beta')):
        channel = density[index]
        if not np.all(np.isfinite(channel)):
            spin_summary.update(status='failed', reason='Spin density contains nonfinite values')
            return summary
        if not np.allclose(channel, channel.conj().T, atol=1e-6, rtol=0):
            spin_summary.update(status='failed', reason='Spin density is not Hermitian')
            return summary
        eigenvalues = np.linalg.eigvalsh(0.5 * (channel + channel.conj().T))[::-1]
        channels[label] = {
            'occupations': eigenvalues.tolist(),
            'electron_count': float(eigenvalues.sum()),
            'site_populations': np.diag(channel).real.astype(float).tolist(),
        }
        spin_summary['channels'] = channels
        if np.any(eigenvalues < -1e-6) or np.any(eigenvalues > 1.0 + 1e-6):
            spin_summary.update(status='out_of_bounds', reason='Spin natural occupations are outside [0, 1]')
            return summary
        if nelec is not None and abs(float(eigenvalues.sum()) - float(nelec[index])) > 1e-5:
            spin_summary.update(status='failed', reason='Spin density trace does not match the requested electron sector')
            return summary
        # A collinear Slater determinant is idempotent in each spin channel,
        # including open shells and broken-symmetry antiferromagnetic UHF.
        values = np.clip(eigenvalues, 0.0, 1.0)
        fractionality.extend((4.0 * values * (1.0 - values)).tolist())
    spin_summary.update(
        status='available', reason=None,
        score=float(np.mean(fractionality)),
        max_fractionality=float(max(fractionality)),
        definition='mean_4n_times_1_minus_n_over_spin_orbitals',
        baseline='spin_channel_idempotent_determinant',
    )
    return summary


def _natural_occupation_summary_from_solver(solver: Any) -> Optional[Dict[str, Any]]:
    make_rdm1 = getattr(solver, 'make_rdm1', None)
    if not callable(make_rdm1):
        return None
    try:
        dm1 = make_rdm1(ao_repr=True)
    except TypeError:
        try:
            dm1 = make_rdm1()
        except Exception:  # pragma: no cover - PySCF solver-specific fallback
            return None
    except Exception:  # pragma: no cover - PySCF solver-specific fallback
        return None
    return _natural_occupation_summary_from_dm1(
        dm1, nelec=getattr(getattr(solver, 'mol', None), 'nelec', None),
        source='correlated_solver_1rdm',
    )


def _fci_natural_occupation_summary(ci_vector: Any, metadata: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    from pyscf.fci import direct_spin1  # pylint: disable=import-outside-toplevel

    norb = metadata['norb']
    nelec = tuple(metadata['nelec'])
    try:
        dm1 = direct_spin1.make_rdm1s(ci_vector, norb, nelec)
    except Exception:  # pragma: no cover - PySCF solver-specific fallback
        return None
    return _natural_occupation_summary_from_dm1(dm1, nelec=nelec, source='fci_ground_state_1rdm')


def _flatten_numeric_amplitudes(value: Any):
    import numpy as np  # pylint: disable=import-outside-toplevel

    arrays = []

    def visit(item: Any) -> None:
        if item is None:
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
            return
        try:
            array = np.asarray(item, dtype=float).reshape(-1)
        except (TypeError, ValueError):
            return
        if array.size:
            arrays.append(array)

    visit(value)
    if not arrays:
        return None
    return np.concatenate(arrays)


def _double_excitation_amplitude_summary(solver: Any) -> Optional[Dict[str, Any]]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    amplitudes = _flatten_numeric_amplitudes(getattr(solver, 't2', None))
    if amplitudes is None or amplitudes.size == 0:
        return None
    absolute = np.abs(amplitudes)
    return {
        'max_abs_t2': float(absolute.max()),
        'rms_t2': float(np.sqrt(np.mean(absolute ** 2))),
        'large_amplitude_count_0_10': int(np.count_nonzero(absolute >= 0.10)),
        'large_amplitude_count_0_20': int(np.count_nonzero(absolute >= 0.20)),
    }
