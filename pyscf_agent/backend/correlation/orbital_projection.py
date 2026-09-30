"""CASSCF admission and partition policy over the shared AO projection kernel."""
from __future__ import annotations

import numpy as np

from ..orbital_projection import geometry_matches, molecular_basis_signature, project_orbital_blocks


def project_casscf_active_guess(source, casscf, coefficients, *, frozen_indices=(),
                               rank_tolerance=1e-10):
    """Keep target SCF frozen MOs, transfer the active guess, complete virtuals.

    PySCF's public SVD-based project_init_guess owns the orthogonal completion.
    Only occupied/active columns enter the projection priority: externals are an
    unconstrained complement, not a physical subspace to transport. Frozen
    indices name target SCF columns and matching output CASSCF columns.
    """
    from pyscf import gto, mcscf, scf

    target = casscf.mol
    if molecular_basis_signature(source) != molecular_basis_signature(target):
        raise ValueError('Orbital continuation requires matching atom ordering and AO layout.')
    if (source.charge, source.spin) != (target.charge, target.spin):
        raise ValueError('Orbital continuation requires matching charge and spin.')
    matrix = np.asarray(coefficients)
    reference = np.asarray(casscf._scf.mo_coeff)
    ncore, ncas = casscf.ncore, casscf.ncas
    nao = target.nao_nr()
    if (not isinstance(ncore, (int, np.integer)) or not 0 <= ncore < ncore + ncas <= nao
            or matrix.shape != (nao, nao) or reference.shape != (nao, nao)
            or np.iscomplexobj(matrix) or np.iscomplexobj(reference)
            or not np.isfinite(matrix).all() or not np.isfinite(reference).all()):
        raise ValueError('Target-SCF continuation requires finite real full-space spatial orbitals.')
    frozen = list(frozen_indices)
    if len(set(frozen)) != len(frozen) or any(
            isinstance(i, (bool, np.bool_)) or not isinstance(i, (int, np.integer))
            or i < 0 or i >= ncore for i in frozen):
        raise ValueError('Target-SCF frozen indices must be distinct inactive SCF columns.')
    frozen = sorted(int(i) for i in frozen)
    active = list(range(ncore, ncore + ncas))
    inactive = [i for i in range(ncore) if i not in frozen]
    overlap = target.intor_symmetric('int1e_ovlp')
    source_overlap = source.intor_symmetric('int1e_ovlp')
    for mo, metric in [(matrix, source_overlap), (reference, overlap)]:
        if np.max(np.abs(mo.T @ metric @ mo - np.eye(nao))) > 1e-8:
            raise ValueError('Continuation input orbitals must be orthonormal in their own AO metric.')
    # PySCF's explicit priority masks span nmo even for incomplete guesses.
    # Supply a full-width array but omit external columns from every mask;
    # project_init_guess replaces them with its target-basis complement.
    initial = reference.copy()
    initial[:, :ncore + ncas] = scf.addons.project_mo_nr2nr(source, matrix[:, :ncore + ncas], target)
    initial[:, frozen] = reference[:, frozen]
    groups = [(name, columns) for name, columns in [
        ('frozen_core', frozen), ('active', active), ('inactive', inactive),
    ] if columns]
    # Reject rank loss before PySCF's SVD could complete a missing guessed direction.
    previous = np.empty((nao, 0))
    ranks = {}
    for name, columns in groups:
        block = initial[:, columns].copy()
        for _ in range(2):
            block -= previous @ (previous.T @ overlap @ block)
        metric = block.T @ overlap @ block
        values, vectors = np.linalg.eigh((metric + metric.T) * .5)
        if values.min() <= rank_tolerance:
            raise ValueError('Projected {0} guess loses rank after target-SCF core selection.'.format(name))
        ranks[name] = float(values.min())
        previous = np.column_stack((previous, block @ ((vectors * values**-.5) @ vectors.T)))
    result = mcscf.addons.project_init_guess(
        casscf, initial, priority=[columns for _, columns in groups], use_hf_core=False,
    )
    error = float(np.max(np.abs(result.T @ overlap @ result - np.eye(nao))))
    frozen_error = float(np.max(np.abs(result[:, frozen] - reference[:, frozen]))) if frozen else 0.
    if not np.isfinite(error) or error > 1e-8 or frozen_error > 1e-8:
        raise ValueError('Target-SCF continuation failed orthonormality or frozen-column preservation.')
    cross = gto.intor_cross('int1e_ovlp', source, target)
    diagnostics = []
    for name, columns in groups:
        sv = np.linalg.svd(matrix[:, columns].T @ cross @ result[:, columns], compute_uv=False)
        diagnostics.append({'partition': name, 'dimension': len(columns), 'columns': columns,
                            'residual_metric_min_eigenvalue': ranks[name],
                            'minimum_subspace_overlap': float(sv.min()),
                            'cross_geometry_overlap_singular_values': sv.tolist()})
    return result, {
        'schema': 'pyscf-agent.orbital-projection.v1',
        'method': 'target_scf_frozen_core_and_pyscf.mcscf.addons.project_init_guess',
        'continuation_policy': 'target_scf_core', 'same_geometry': geometry_matches(source, target),
        'ncore': int(ncore), 'ncas': int(ncas), 'nmo': nao,
        'frozen_orbital_indices': frozen, 'frozen_reference': 'target_scf',
        'target_scf_frozen_max_abs_error': frozen_error,
        'orthonormality_max_abs_error': error, 'partitions': diagnostics,
        'external_dimension': nao - ncore - ncas,
        'external_source': 'orthogonal_complement_of_target_scf_basis',
        'external_truncated': False, 'mps_transferred': False,
    }


def project_partitioned_orbitals(source, target, coefficients, *, ncore, ncas,
                                frozen_indices=(), rank_tolerance=1e-10,
                                active_priority=False):
    """Transfer a full CASSCF orbital guess with unchanged partition definitions.

    Frozen inactive, other inactive, active, and external blocks are derived
    from the calculation's counts and indices, with that priority. This adapter
    deliberately retains the same-layout, full-space, real-orbital contract;
    broader AO projections do not imply compatible CASSCF or MPS continuation.
    """
    if molecular_basis_signature(source) != molecular_basis_signature(target):
        raise ValueError('Orbital projection requires the same atom ordering and AO basis layout.')
    if (source.charge, source.spin) != (target.charge, target.spin):
        raise ValueError('Orbital projection requires the same charge and spin.')
    matrix = np.asarray(coefficients)
    if (matrix.ndim != 2 or matrix.shape[0] != source.nao_nr()
            or matrix.shape[1] != source.nao_nr() or np.iscomplexobj(matrix)
            or not np.issubdtype(matrix.dtype, np.number) or not np.isfinite(matrix).all()):
        raise ValueError('Orbital projection requires a finite real full-space MO matrix.')
    if (isinstance(ncore, bool) or isinstance(ncas, bool)
            or int(ncore) != ncore or int(ncas) != ncas
            or ncore < 0 or ncas <= 0 or ncore + ncas > matrix.shape[1]):
        raise ValueError('Invalid core/active orbital partition.')
    frozen = list(frozen_indices)
    if (len(frozen) != len(set(frozen)) or any(
            isinstance(i, bool) or int(i) != i or i < 0 or i >= ncore for i in frozen)):
        raise ValueError('Cross-geometry projection supports frozen inactive core orbitals only.')
    ncore, ncas = int(ncore), int(ncas)
    frozen = sorted(int(i) for i in frozen)
    groups = [
        ('frozen_core', frozen),
        ('inactive', [i for i in range(ncore) if i not in frozen]),
        ('active', list(range(ncore, ncore + ncas))),
        ('external', list(range(ncore + ncas, matrix.shape[1]))),
    ]
    if active_priority:
        groups[1], groups[2] = groups[2], groups[1]
    result, diagnostics = project_orbital_blocks(
        source, target, matrix, blocks=groups, rank_tolerance=rank_tolerance,
    )
    return result, {
        **diagnostics, 'ncore': ncore, 'ncas': ncas,
        'frozen_orbital_indices': frozen, 'mps_transferred': False,
    }


def transport_mps_orbitals(source, target, coefficients, *, ncore, ncas,
                           frozen_indices=(), minimum_active_overlap=0.9):
    """Carry orbital labels continuously for an approximate MPS initial guess.

    The old occupation coefficients are interpreted in the transported basis,
    not claimed to represent the identical real-space wavefunction. Align each
    partition to the source by orthogonal Procrustes; do not relocalize, reorder,
    or mix inactive/active/external columns afterwards. The caller must preserve
    MPS site order and validate electron, spin, root and weight compatibility.
    """
    from pyscf import gto

    if (isinstance(minimum_active_overlap, (bool, np.bool_))
            or not np.isfinite(minimum_active_overlap)
            or not 0 < minimum_active_overlap <= 1):
        raise ValueError('MPS minimum active overlap must be in (0, 1].')
    result, info = project_partitioned_orbitals(
        source, target, coefficients, ncore=ncore, ncas=ncas,
        frozen_indices=frozen_indices, active_priority=True,
    )
    cross = gto.intor_cross('int1e_ovlp', source, target)
    source_mo = np.asarray(coefficients)
    for partition in info['partitions']:
        columns = partition['columns']
        if not columns:
            continue
        overlap = source_mo[:, columns].T @ cross @ result[:, columns]
        left, values, right = np.linalg.svd(overlap)
        if partition['partition'] == 'active' and values.min() < minimum_active_overlap:
            raise ValueError(
                'MPS active-space overlap {0:.6g} is below {1:.6g}; '
                'use orbital-only continuation with a fresh MPS.'.format(
                    values.min(), minimum_active_overlap))
        # O = U s V^T; rotating target by V U^T gives a positive overlap.
        result[:, columns] = result[:, columns] @ (right.T @ left.T)
        aligned = source_mo[:, columns].T @ cross @ result[:, columns]
        partition['aligned_orbital_overlaps'] = np.diag(aligned).tolist()
    error = float(np.max(np.abs(
        result.T @ target.intor_symmetric('int1e_ovlp') @ result - np.eye(result.shape[1]))))
    if not np.isfinite(error) or error > 1e-8:
        raise ValueError('Transported MPS orbitals failed target-metric orthonormality.')
    info.update(
        orthonormality_max_abs_error=error,
        orbital_alignment='partitionwise_orthogonal_procrustes',
        minimum_active_overlap_required=float(minimum_active_overlap),
        mps_transfer_mode='occupation_coefficients_in_transported_orbitals',
        exact_wavefunction_transform=False,
    )
    # Actual MPS loading is reported by the solver, after compatibility checks.
    return result, info
