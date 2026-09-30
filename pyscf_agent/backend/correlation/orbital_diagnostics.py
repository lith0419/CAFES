from __future__ import annotations

from typing import Any, Dict, Optional, Sequence, Tuple


def _as_float(value: Any) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _overlap_matrix(mf: Any) -> Any:
    try:
        return mf.get_ovlp()
    except Exception:
        mol = getattr(mf, 'mol', None)
        try:
            return mol.intor_symmetric('int1e_ovlp')
        except Exception:
            return None


def _spin_channels(value: Any) -> Optional[Tuple[Any, Any]]:
    if isinstance(value, tuple) and len(value) >= 2:
        return value[0], value[1]
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        array = np.asarray(value)
        if array.ndim == 3 and array.shape[0] >= 2:
            return array[0], array[1]
    except Exception:
        return None
    return None


def _canonical_projection_weights(mf: Any, natural_coeff: Any, overlap: Any) -> Any:
    import numpy as np  # pylint: disable=import-outside-toplevel

    coefficients = getattr(mf, 'mo_coeff', None)
    channels = _spin_channels(coefficients)
    if channels is not None:
        channel_weights = []
        for channel in channels:
            matrix = np.asarray(channel, dtype=float)
            if matrix.ndim == 2:
                channel_weights.append(np.abs(matrix.T @ overlap @ natural_coeff) ** 2)
        if channel_weights:
            return sum(channel_weights) / len(channel_weights)
    matrix = np.asarray(coefficients, dtype=float)
    if matrix.ndim == 2:
        return np.abs(matrix.T @ overlap @ natural_coeff) ** 2
    return None


def _ao_natural_orbitals(matrix: Any, mf: Any) -> Tuple[Any, Any, Any]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    overlap = _overlap_matrix(mf)
    if overlap is None:
        raise ValueError('AO overlap matrix is unavailable')
    overlap = np.asarray(overlap, dtype=float)
    density = np.asarray(matrix, dtype=float)
    if overlap.shape != density.shape:
        raise ValueError('AO density and overlap matrices have incompatible shapes')
    overlap = 0.5 * (overlap + overlap.T)
    overlap_values, overlap_vectors = np.linalg.eigh(overlap)
    if np.any(overlap_values <= 1.0e-10):
        raise ValueError('AO overlap matrix is singular')
    overlap_half = overlap_vectors @ np.diag(np.sqrt(overlap_values)) @ overlap_vectors.T
    overlap_inv_half = overlap_vectors @ np.diag(1.0 / np.sqrt(overlap_values)) @ overlap_vectors.T
    orthogonal_density = overlap_half @ density @ overlap_half
    occupations, vectors = np.linalg.eigh(0.5 * (orthogonal_density + orthogonal_density.T))
    order = np.argsort(occupations)[::-1]
    occupations = occupations[order]
    natural_coeff = overlap_inv_half @ vectors[:, order]
    return occupations, natural_coeff, overlap


def natural_orbital_summary_from_occupations(
    occupations: Sequence[Any],
    *,
    source: str,
) -> Dict[str, Any]:
    """Normalize natural occupations already produced by a correlated solver."""

    orbitals = []
    for index, raw_value in enumerate(occupations or []):
        value = _as_float(raw_value)
        if value is None:
            continue
        orbitals.append({
            'natural_orbital_index': int(index),
            'occupation': value,
            'dominant_canonical_mo': None,
            'dominant_canonical_weight': None,
        })
    return {
        'status': 'available' if orbitals else 'unavailable',
        'source': source,
        'representation': 'correlated_solver_natural_orbitals',
        'occupation_sum': sum(item['occupation'] for item in orbitals),
        'orbitals': orbitals,
    }


def natural_orbital_summary(post_hf: Any, mf: Any = None) -> Dict[str, Any]:
    """Diagonalize a correlated spin-summed 1-RDM in a common orbital basis.

    Restricted post-HF densities are often returned in the canonical MO basis.
    Unrestricted alpha and beta densities belong to different MO bases, so they
    are transformed to the shared AO basis before they are spin-summed.
    """

    summary: Dict[str, Any] = {
        'status': 'unavailable',
        'source': None,
        'orbitals': [],
    }
    if post_hf is None or not hasattr(post_hf, 'make_rdm1'):
        summary['reason'] = 'post-HF one-particle density matrix is unavailable'
        return summary
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        ao_requested = False
        try:
            dm1 = post_hf.make_rdm1(ao_repr=True)
            ao_requested = True
        except TypeError:
            dm1 = post_hf.make_rdm1()
        channels = _spin_channels(dm1)
        if channels is not None:
            alpha = np.asarray(channels[0], dtype=float)
            beta = np.asarray(channels[1], dtype=float)
            if not ao_requested and mf is not None:
                coefficient_channels = _spin_channels(getattr(mf, 'mo_coeff', None))
                if coefficient_channels is not None:
                    alpha_coeff = np.asarray(coefficient_channels[0], dtype=float)
                    beta_coeff = np.asarray(coefficient_channels[1], dtype=float)
                    alpha = alpha_coeff @ alpha @ alpha_coeff.T
                    beta = beta_coeff @ beta @ beta_coeff.T
                    ao_requested = True
            matrix = alpha + beta
            summary['source'] = '{0}.make_rdm1 spin-summed'.format(type(post_hf).__name__)
        else:
            matrix = np.asarray(dm1, dtype=float)
            summary['source'] = '{0}.make_rdm1'.format(type(post_hf).__name__)
        if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
            summary['reason'] = 'post-HF density matrix is not square'
            return summary
        matrix = 0.5 * (matrix + matrix.T)
        if ao_requested and mf is not None:
            occupations, natural_coeff, overlap = _ao_natural_orbitals(matrix, mf)
            projection_weights = _canonical_projection_weights(mf, natural_coeff, overlap)
            summary['representation'] = 'spin_summed_ao'
        else:
            occupations, transform = np.linalg.eigh(matrix)
            order = np.argsort(occupations)[::-1]
            occupations = occupations[order]
            transform = transform[:, order]
            projection_weights = np.abs(transform) ** 2
            summary['representation'] = 'shared_mo'
        orbitals = []
        for no_index, occupation in enumerate(occupations):
            weights = (
                projection_weights[:, no_index]
                if projection_weights is not None and projection_weights.shape[1] > no_index
                else np.asarray([])
            )
            dominant_index = int(np.argmax(weights)) if weights.size else None
            orbitals.append({
                'natural_orbital_index': int(no_index),
                'occupation': float(occupation),
                'dominant_canonical_mo': dominant_index,
                'dominant_canonical_weight': (
                    float(weights[dominant_index]) if dominant_index is not None else None
                ),
            })
        summary['status'] = 'available'
        summary['occupation_sum'] = float(np.sum(occupations))
        expected_electrons = _as_float(getattr(getattr(mf, 'mol', None), 'nelectron', None))
        if expected_electrons is not None:
            summary['expected_electron_count'] = expected_electrons
            summary['electron_count_error'] = float(np.sum(occupations) - expected_electrons)
        summary['orbitals'] = orbitals
        return summary
    except Exception as exc:  # pragma: no cover - depends on PySCF post-HF object details
        summary['status'] = 'failed'
        summary['reason'] = str(exc)
        return summary


def occupation_fractionality(occupation: Optional[float]) -> float:
    if occupation is None:
        return 0.0
    return max(0.0, 1.0 - abs(float(occupation) - 1.0))


def t2_orbital_importance(post_hf: Any, mf: Any) -> Dict[str, Any]:
    """Report the largest absolute T2 amplitude touching each spatial MO and evidence status."""

    t2 = getattr(post_hf, 't2', None)
    if t2 is None:
        return {'status': 'unavailable', 'importance': {}, 'reason': 'T2 amplitudes are unavailable'}
    try:
        import numpy as np  # pylint: disable=import-outside-toplevel

        mo_occ = getattr(mf, 'mo_occ', None)
        if hasattr(mo_occ, 'tolist'):
            mo_occ = mo_occ.tolist()
        if isinstance(mo_occ, (list, tuple)) and mo_occ and isinstance(mo_occ[0], (list, tuple)):
            alpha_occ = list(mo_occ[0])
            beta_occ = list(mo_occ[1]) if len(mo_occ) > 1 and isinstance(mo_occ[1], (list, tuple)) else []
            occ_alpha = [i for i, occ in enumerate(alpha_occ) if _as_float(occ) and float(occ) > 1.0e-8]
            vir_alpha = [i for i, occ in enumerate(alpha_occ) if not (_as_float(occ) and float(occ) > 1.0e-8)]
            occ_beta = [i for i, occ in enumerate(beta_occ) if _as_float(occ) and float(occ) > 1.0e-8]
            vir_beta = [i for i, occ in enumerate(beta_occ) if not (_as_float(occ) and float(occ) > 1.0e-8)]
            components = t2 if isinstance(t2, tuple) else (t2,)
            importance: Dict[int, float] = {}

            def update(index: int, value: Any) -> None:
                current = importance.get(index, 0.0)
                converted = abs(float(value))
                if converted > current:
                    importance[index] = converted

            if len(components) >= 1:
                arr = np.abs(np.asarray(components[0], dtype=float))
                if arr.ndim != 4 or not np.all(np.isfinite(arr)):
                    raise ValueError('T2 amplitudes must be finite rank-four tensors')
                for i, mo_i in enumerate(occ_alpha[:arr.shape[0]]):
                    update(mo_i, arr[i, :, :, :].max() if arr.size else 0.0)
                for j, mo_j in enumerate(occ_alpha[:arr.shape[1]]):
                    update(mo_j, arr[:, j, :, :].max() if arr.size else 0.0)
                for a, mo_a in enumerate(vir_alpha[:arr.shape[2]]):
                    update(mo_a, arr[:, :, a, :].max() if arr.size else 0.0)
                for b, mo_b in enumerate(vir_alpha[:arr.shape[3]]):
                    update(mo_b, arr[:, :, :, b].max() if arr.size else 0.0)
            if len(components) >= 2:
                arr = np.abs(np.asarray(components[1], dtype=float))
                if arr.ndim != 4 or not np.all(np.isfinite(arr)):
                    raise ValueError('T2 amplitudes must be finite rank-four tensors')
                for i, mo_i in enumerate(occ_alpha[:arr.shape[0]]):
                    update(mo_i, arr[i, :, :, :].max() if arr.size else 0.0)
                for j, mo_j in enumerate(occ_beta[:arr.shape[1]]):
                    update(mo_j, arr[:, j, :, :].max() if arr.size else 0.0)
                for a, mo_a in enumerate(vir_alpha[:arr.shape[2]]):
                    update(mo_a, arr[:, :, a, :].max() if arr.size else 0.0)
                for b, mo_b in enumerate(vir_beta[:arr.shape[3]]):
                    update(mo_b, arr[:, :, :, b].max() if arr.size else 0.0)
            if len(components) >= 3:
                arr = np.abs(np.asarray(components[2], dtype=float))
                if arr.ndim != 4 or not np.all(np.isfinite(arr)):
                    raise ValueError('T2 amplitudes must be finite rank-four tensors')
                for i, mo_i in enumerate(occ_beta[:arr.shape[0]]):
                    update(mo_i, arr[i, :, :, :].max() if arr.size else 0.0)
                for j, mo_j in enumerate(occ_beta[:arr.shape[1]]):
                    update(mo_j, arr[:, j, :, :].max() if arr.size else 0.0)
                for a, mo_a in enumerate(vir_beta[:arr.shape[2]]):
                    update(mo_a, arr[:, :, a, :].max() if arr.size else 0.0)
                for b, mo_b in enumerate(vir_beta[:arr.shape[3]]):
                    update(mo_b, arr[:, :, :, b].max() if arr.size else 0.0)
            return {'status': 'available', 'importance': importance}

        occupations = list(mo_occ) if isinstance(mo_occ, (list, tuple)) else []
        occ_indices = [i for i, occ in enumerate(occupations) if _as_float(occ) and float(occ) > 1.0e-8]
        vir_indices = [i for i, occ in enumerate(occupations) if not (_as_float(occ) and float(occ) > 1.0e-8)]
        arr = np.abs(np.asarray(t2, dtype=float))
        if arr.ndim != 4 or not np.all(np.isfinite(arr)):
            raise ValueError('T2 amplitudes must be a finite rank-four tensor')
        importance = {}
        for i, mo_i in enumerate(occ_indices[:arr.shape[0]]):
            importance[mo_i] = max(importance.get(mo_i, 0.0), float(arr[i, :, :, :].max()) if arr.size else 0.0)
        for j, mo_j in enumerate(occ_indices[:arr.shape[1]]):
            importance[mo_j] = max(importance.get(mo_j, 0.0), float(arr[:, j, :, :].max()) if arr.size else 0.0)
        for a, mo_a in enumerate(vir_indices[:arr.shape[2]]):
            importance[mo_a] = max(importance.get(mo_a, 0.0), float(arr[:, :, a, :].max()) if arr.size else 0.0)
        for b, mo_b in enumerate(vir_indices[:arr.shape[3]]):
            importance[mo_b] = max(importance.get(mo_b, 0.0), float(arr[:, :, :, b].max()) if arr.size else 0.0)
        return {'status': 'available', 'importance': importance}
    except Exception as exc:  # Optional evidence must not invalidate a completed energy.
        return {'status': 'failed', 'importance': {}, 'reason': str(exc)}


__all__ = [
    'natural_orbital_summary',
    'natural_orbital_summary_from_occupations',
    'occupation_fractionality',
    't2_orbital_importance',
]
