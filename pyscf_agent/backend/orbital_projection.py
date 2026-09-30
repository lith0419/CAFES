"""Solver-independent transfer of real molecular orbital subspaces.

Callers supply ordered blocks of MO columns. This module knows no occupations,
active-space sizes, frozen-core choices, checkpoints, or scan paths. Block order
sets priority during orthogonalization; it is a physical choice of the caller.
"""
from __future__ import annotations

from numbers import Integral

import numpy as np


def molecular_basis_signature(mol):
    """Describe AO layout, excluding coordinates, for continuation admission."""
    if mol is None:
        return {}
    atom_count = int(getattr(mol, 'natm', 0) or 0)
    labels = [str(value) for value in mol.ao_labels()] if hasattr(mol, 'ao_labels') else []
    return {
        'atom_symbols': [str(mol.atom_symbol(index)) for index in range(atom_count)],
        'basis': repr(getattr(mol, 'basis', None)),
        'ecp': repr(getattr(mol, 'ecp', None)),
        'cartesian_gaussians': bool(getattr(mol, 'cart', False)),
        'ao_labels': labels,
        'nao': int(mol.nao_nr()) if hasattr(mol, 'nao_nr') else len(labels),
    }


def geometry_matches(source, target):
    return (
        source.natm == target.natm
        and [source.atom_symbol(i) for i in range(source.natm)]
        == [target.atom_symbol(i) for i in range(target.natm)]
        and source.charge == target.charge and source.spin == target.spin
        and np.allclose(source.atom_coords(), target.atom_coords(), atol=1e-12, rtol=0)
    )


def _validate_blocks(blocks, nmo):
    groups = []
    names = set()
    columns = []
    for name, indices in blocks:
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError('Orbital blocks require unique nonempty names.')
        indices = list(indices)
        if any(isinstance(i, (bool, np.bool_)) or not isinstance(i, Integral)
               or i < 0 or i >= nmo for i in indices):
            raise ValueError('Orbital block columns must be integer MO indices in range.')
        indices = [int(i) for i in indices]
        names.add(name)
        columns.extend(indices)
        groups.append((name, indices))
    if sorted(columns) != list(range(nmo)):
        raise ValueError('Orbital blocks must cover every supplied MO column exactly once.')
    return groups


def project_orbital_blocks(source, target, coefficients, *, blocks,
                           rank_tolerance=1e-10, orthonormality_tolerance=1e-8):
    """Project and orthonormalize explicitly partitioned spatial orbital guesses.

    ``blocks`` is an ordered sequence of ``(name, column_indices)`` pairs covering
    all supplied columns exactly once; empty blocks are allowed. Later blocks
    are orthogonalized against earlier blocks, then symmetrically within each
    block. No columns are relabelled, discarded, or supplied from a target SCF.

    Rectangular real coefficient matrices and different scalar molecular AO
    bases are supported when all supplied columns remain independent. A partial
    set stays partial: this does not complete missing virtual orbitals. The
    caller owns geometry correspondence, occupations, method compatibility,
    and acceptance of subspace-overlap diagnostics. This is not a periodic,
    spinor, unrestricted-pair, MPS, or electronic-state transfer interface.
    """
    from pyscf import gto, scf

    matrix = np.asarray(coefficients)
    if (matrix.ndim != 2 or matrix.shape[0] != source.nao_nr()
            or matrix.shape[1] == 0 or np.iscomplexobj(matrix)
            or not np.issubdtype(matrix.dtype, np.number)
            or not np.isfinite(matrix).all()):
        raise ValueError('Orbital projection requires a finite real AO-by-MO matrix.')
    matrix = np.asarray(matrix, dtype=float)
    nmo = matrix.shape[1]
    if nmo > min(source.nao_nr(), target.nao_nr()):
        raise ValueError('Supplied orbital rank exceeds the source or target AO dimension.')
    for name, value in [('rank', rank_tolerance), ('orthonormality', orthonormality_tolerance)]:
        if not np.isfinite(value) or value <= 0:
            raise ValueError('{0} tolerance must be finite and positive.'.format(name.capitalize()))
    groups = _validate_blocks(blocks, nmo)
    s_source = source.intor_symmetric('int1e_ovlp')
    s_target = target.intor_symmetric('int1e_ovlp')
    source_error = float(np.max(np.abs(matrix.T @ s_source @ matrix - np.eye(nmo))))
    if not np.isfinite(source_error) or source_error > orthonormality_tolerance:
        raise ValueError('Source orbitals are not orthonormal in their own AO metric.')
    projected = scf.addons.project_mo_nr2nr(source, matrix, target)
    cross = gto.intor_cross('int1e_ovlp', source, target)
    result = np.empty_like(projected)
    previous = np.empty((target.nao_nr(), 0))
    diagnostics = []
    for name, indices in groups:
        if not indices:
            diagnostics.append({'partition': name, 'columns': [], 'dimension': 0})
            continue
        block = projected[:, indices].copy()
        # Two passes control roundoff when diffuse orbitals overlap strongly.
        for _ in range(2):
            block -= previous @ (previous.T @ s_target @ block)
        metric = block.T @ s_target @ block
        values, vectors = np.linalg.eigh((metric + metric.T) * .5)
        if values.min() <= rank_tolerance:
            raise ValueError('Projected {0} partition loses rank; review the orbital transfer.'.format(name))
        block = block @ ((vectors * values**-.5) @ vectors.T)
        result[:, indices] = block
        previous = np.column_stack((previous, block))
        singular_values = np.linalg.svd(matrix[:, indices].T @ cross @ block, compute_uv=False)
        diagnostics.append({
            'partition': name, 'columns': indices, 'dimension': len(indices),
            'residual_metric_min_eigenvalue': float(values.min()),
            'cross_geometry_overlap_singular_values': singular_values.tolist(),
            'minimum_subspace_overlap': float(singular_values.min()),
        })
    error = float(np.max(np.abs(result.T @ s_target @ result - np.eye(nmo))))
    if not np.isfinite(error) or error > orthonormality_tolerance:
        raise ValueError('Projected orbitals failed target-metric orthonormality.')
    return result, {
        'schema': 'pyscf-agent.orbital-projection.v1',
        'method': 'pyscf.scf.addons.project_mo_nr2nr',
        'orthonormalization': 'partition_ordered_metric_gram_schmidt_lowdin',
        'rank_tolerance': float(rank_tolerance),
        'orthonormality_tolerance': float(orthonormality_tolerance),
        'same_geometry': geometry_matches(source, target),
        'source_orthonormality_max_abs_error': source_error,
        'orthonormality_max_abs_error': error,
        'source_nao': int(source.nao_nr()), 'target_nao': int(target.nao_nr()),
        'nmo': int(nmo), 'partitions': diagnostics,
    }
