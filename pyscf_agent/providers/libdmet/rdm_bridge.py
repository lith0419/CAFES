from __future__ import annotations

from typing import Any, Optional


def _solver_mo_coefficients(solver: Any) -> Any:
    scf_solver = getattr(solver, 'scfsolver', None)
    mean_field = getattr(scf_solver, 'mf', None)
    coefficients = getattr(mean_field, 'mo_coeff', None)
    if coefficients is not None:
        return coefficients
    coupled_cluster = getattr(solver, 'cisolver', None)
    return getattr(coupled_cluster, 'mo_coeff', None)


def _transform_rdm2_block(rdm2_mo: Any, left_coefficients: Any, right_coefficients: Any, np: Any) -> Any:
    return np.einsum(
        'pa,qb,abcd,rc,sd->pqrs',
        left_coefficients,
        left_coefficients,
        rdm2_mo,
        right_coefficients.conj(),
        right_coefficients.conj(),
        optimize=True,
    )


def impurity_spin_rdm2(solver: Any, impurity_orbital_count: int) -> Optional[Any]:
    """Return the impurity-site 2RDM in ``aa, bb, ab`` spin order.

    libDMET impurity solvers do not expose one common representation: block2
    keeps its 2RDM in the embedding basis, while the native FCI/CCSD adapters
    may retain it in the solver MO basis.  This bridge normalizes only that
    representation boundary; model-observable interpretation stays in the
    model-Hamiltonian backend.
    """

    import numpy as np  # pylint: disable=import-outside-toplevel

    count = int(impurity_orbital_count)
    if count < 1:
        return None

    direct = getattr(solver, 'twopdm', None)
    if callable(direct):
        try:
            direct = direct()
        except TypeError:
            direct = None
    rdm2 = None if direct is None else np.asarray(direct)

    if rdm2 is None or rdm2.ndim != 5:
        rdm2_mo = getattr(solver, 'twopdm_mo', None)
        coefficients = _solver_mo_coefficients(solver)
        if rdm2_mo is None or coefficients is None:
            return None
        rdm2_mo = np.asarray(rdm2_mo)
        coefficients = np.asarray(coefficients)
        if rdm2_mo.ndim != 5:
            return None
        if rdm2_mo.shape[0] == 1:
            coefficient_block = coefficients[0] if coefficients.ndim == 3 else coefficients
            rdm2 = _transform_rdm2_block(
                rdm2_mo[0], coefficient_block, coefficient_block, np,
            )[None, ...]
        elif rdm2_mo.shape[0] == 3 and coefficients.ndim == 3 and coefficients.shape[0] == 2:
            # libDMET's solver-facing convention is aa, bb, ab.
            rdm2 = np.asarray([
                _transform_rdm2_block(rdm2_mo[0], coefficients[0], coefficients[0], np),
                _transform_rdm2_block(rdm2_mo[1], coefficients[1], coefficients[1], np),
                _transform_rdm2_block(rdm2_mo[2], coefficients[0], coefficients[1], np),
            ])
        else:
            return None

    if rdm2.shape[-1] < count:
        return None
    impurity = tuple(slice(0, count) for _axis in range(4))
    return np.asarray(rdm2[(slice(None),) + impurity])


__all__ = ['impurity_spin_rdm2']
