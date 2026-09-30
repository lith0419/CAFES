from __future__ import annotations

from typing import Any


def _numpy() -> Any:
    try:
        import numpy as np
    except ImportError as exc:  # pragma: no cover - required package dependency
        raise RuntimeError('Embedding transformations require NumPy') from exc
    return np


def _libdmet_make_basis() -> Any:
    try:
        from libdmet.basis_transform import make_basis
    except ImportError as exc:  # pragma: no cover - optional provider
        raise RuntimeError('The libDMET embedding provider is not installed') from exc
    return make_basis


def identity_localized_orbitals(
    norb: int,
    *,
    layout: str = 'auto',
    leading_shape: Any = None,
) -> Any:
    np = _numpy()
    norb = int(norb)
    if norb < 1:
        raise ValueError('norb must be positive')
    identity = np.eye(norb)
    normalized_layout = str(layout or 'auto').strip().lower()
    if leading_shape:
        return np.broadcast_to(identity, tuple(int(item) for item in leading_shape) + identity.shape).copy()
    if normalized_layout in ('kpoint', 'spin_kpoint'):
        return identity[np.newaxis]
    return identity


def _as_libdmet_layout(matrix: Any, coefficients: Any, layout: str) -> tuple[Any, Any, bool]:
    np = _numpy()
    matrix_array = np.asarray(matrix)
    coefficient_array = np.asarray(coefficients)
    normalized_layout = str(layout or 'auto').strip().lower()
    if matrix_array.shape[-1] != matrix_array.shape[-2]:
        raise ValueError('operator matrices must be square')
    if coefficient_array.shape[-2] != matrix_array.shape[-1]:
        raise ValueError('localized-orbital coefficients are incompatible with the AO dimension')
    if matrix_array.ndim == 2 and coefficient_array.ndim == 2:
        return matrix_array[np.newaxis], coefficient_array[np.newaxis], True
    if normalized_layout == 'spin' and matrix_array.ndim == 3 and coefficient_array.ndim in (2, 3):
        if coefficient_array.ndim == 2:
            coefficient_array = np.broadcast_to(
                coefficient_array,
                (matrix_array.shape[0],) + coefficient_array.shape,
            )
        return matrix_array[:, np.newaxis], coefficient_array[:, np.newaxis], True
    if matrix_array.ndim == 3 and coefficient_array.ndim == 3:
        if matrix_array.shape[0] != coefficient_array.shape[0]:
            raise ValueError('operator and coefficient k-point dimensions do not match')
        return matrix_array, coefficient_array, False
    if matrix_array.ndim == 4 and coefficient_array.ndim == 4:
        if matrix_array.shape[:2] != coefficient_array.shape[:2]:
            raise ValueError('operator and coefficient spin/k-point dimensions do not match')
        return matrix_array, coefficient_array, False
    raise ValueError(
        'Unsupported matrix layout; use 2D molecular, 3D k-point, 3D spin with layout="spin", or 4D spin/k-point arrays'
    )


def transform_one_body_to_local(operator: Any, coefficients: Any, *, layout: str = 'auto') -> Any:
    make_basis = _libdmet_make_basis()
    operator_array, coefficient_array, squeeze_kpoint = _as_libdmet_layout(
        operator,
        coefficients,
        layout,
    )
    transformed = make_basis.transform_h1_to_lo(operator_array, coefficient_array)
    if squeeze_kpoint:
        return transformed[..., 0, :, :]
    return transformed


def transform_density_to_local(
    density: Any,
    coefficients: Any,
    overlap: Any,
    *,
    layout: str = 'auto',
) -> Any:
    np = _numpy()
    make_basis = _libdmet_make_basis()
    density_array, coefficient_array, squeeze_kpoint = _as_libdmet_layout(
        density,
        coefficients,
        layout,
    )
    overlap_array = np.asarray(overlap)
    if overlap_array.ndim == 2:
        overlap_array = overlap_array[np.newaxis]
    if overlap_array.ndim != 3:
        raise ValueError('overlap must be a 2D molecular or 3D k-point array')
    transformed = make_basis.transform_rdm1_to_lo(
        density_array,
        coefficient_array,
        overlap_array,
    )
    if squeeze_kpoint:
        return transformed[..., 0, :, :]
    return transformed


def build_molecular_iao_coefficients(
    mean_field: Any,
    *,
    minimal_basis: str = 'minao',
    include_pao: bool = True,
    orthogonalize_virtual: bool = True,
) -> Any:
    make_basis = _libdmet_make_basis()
    result = make_basis.get_C_ao_lo_iao_mol(
        mean_field,
        minao=minimal_basis,
        orth_virt=orthogonalize_virtual,
        full_virt=False,
        full_return=True,
    )
    coefficients, valence_iao = result[0], result[1]
    return coefficients if include_pao else valence_iao


def build_periodic_iao_coefficients(
    lattice: Any,
    mean_field: Any,
    *,
    minimal_basis: str = 'minao',
    include_pao: bool = True,
    orthogonalize_virtual: bool = True,
) -> Any:
    make_basis = _libdmet_make_basis()
    result = make_basis.get_C_ao_lo_iao(
        lattice,
        mean_field,
        minao=minimal_basis,
        orth_virt=orthogonalize_virtual,
        full_virt=False,
        full_return=True,
    )
    coefficients, valence_iao = result[0], result[1]
    return coefficients if include_pao else valence_iao


def build_lowdin_coefficients(mean_field_or_lattice: Any, *, method: str = 'meta_lowdin', overlap: Any = None) -> Any:
    make_basis = _libdmet_make_basis()
    return make_basis.get_C_ao_lo_lowdin(mean_field_or_lattice, method=method, s=overlap)
