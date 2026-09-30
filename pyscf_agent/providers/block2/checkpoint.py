from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from ...backend.orbital_projection import geometry_matches, molecular_basis_signature


JOINT_CHECKPOINT_SCHEMA = 'pyscf-agent.block2-joint-checkpoint.v1'
OPTIMIZED_ORBITAL_FILENAME = 'optimized-mo-coeff.npy'


def read_checkpoint_manifest(value: Any) -> Optional[Dict[str, Any]]:
    if value in (None, ''):
        return None
    if isinstance(value, dict):
        return dict(value)
    path = Path(str(value)).expanduser().resolve()
    with path.open('r', encoding='utf-8') as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError('block2 restart manifest must contain a JSON object.')
    return payload


def pin_restart_orbital_ordering(
    manifest: Dict[str, Any],
    permutation: Optional[list[int]] = None,
) -> list[int]:
    """Keep an MPS restart in the orbital ordering used to create it."""

    source = manifest.get('orbital_ordering')
    source = source if isinstance(source, dict) else {}
    fixed_order = [
        int(value)
        for value in (
            permutation
            if permutation is not None
            else source.get('permutation') or []
        )
    ]
    if not fixed_order:
        return []
    source.update({
        'continuation_source_requested_method': source.get('requested_method'),
        'continuation_source_requested_order': list(source.get('requested_order') or []),
        'requested_method': 'manual',
        'requested_order': list(fixed_order),
        'permutation': list(fixed_order),
    })
    manifest['orbital_ordering'] = source
    return fixed_order


def _spin_resolved_nelecas(value: Any, spin: int = 0) -> list[int]:
    if isinstance(value, (list, tuple)):
        if len(value) != 2:
            raise ValueError('Spin-resolved nelecas must contain alpha and beta counts.')
        return [int(value[0]), int(value[1])]
    total = int(value)
    if abs(int(spin)) > total or (total + int(spin)) % 2:
        raise ValueError('Scalar nelecas is incompatible with the target spin.')
    return [(total + int(spin)) // 2, (total - int(spin)) // 2]


def attach_optimized_orbitals(
    manifest: Dict[str, Any],
    mo_coeff: Any,
    *,
    ncore: int,
    ncas: int,
    nelecas: Any,
    reference: str,
    orbital_provenance: Optional[Dict[str, Any]] = None,
    molecular_signature: Optional[Dict[str, Any]] = None,
    molecule: Any = None,
    frozen_orbital_indices: Optional[list[int]] = None,
) -> Dict[str, Any]:
    """Add the optimized full-space orbitals to a block2 MPS manifest."""

    import numpy as np  # pylint: disable=import-outside-toplevel

    # Match the driver's logical shared path rather than this node's symlink target.
    scratch = Path(str(manifest.get('scratch_directory') or '')).expanduser().absolute()
    if not str(manifest.get('scratch_directory') or '').strip():
        raise ValueError('A joint block2 checkpoint requires a scratch_directory.')
    scratch.mkdir(parents=True, exist_ok=True)
    matrix = np.asarray(mo_coeff, dtype=float)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError('Optimized CASSCF orbitals must be a nonempty two-dimensional matrix.')
    if int(ncore) < 0 or int(ncore) + int(ncas) > matrix.shape[1]:
        raise ValueError('Optimized CASSCF orbitals are incompatible with ncore/ncas.')

    path = scratch / OPTIMIZED_ORBITAL_FILENAME
    np.save(path, matrix, allow_pickle=False)
    file_entry = {
        'path': str(path),
        'relative_path': OPTIMIZED_ORBITAL_FILENAME,
        'size_bytes': int(path.stat().st_size),
        'role': 'casscf_optimized_mo_coeff',
    }
    files = [
        dict(item)
        for item in manifest.get('files') or ()
        if isinstance(item, dict)
        and str(item.get('relative_path') or '') != OPTIMIZED_ORBITAL_FILENAME
    ]
    files.append(file_entry)
    manifest['files'] = files
    context = {
        'schema': JOINT_CHECKPOINT_SCHEMA,
        'kind': 'casscf_optimized_orbitals_and_mps',
        'external_restart_supported': True,
        'orbital_file': dict(file_entry),
        'orbital_basis': 'ao_mo_coefficients',
        'matrix_shape': [int(value) for value in matrix.shape],
        'ncore': int(ncore),
        'ncas': int(ncas),
        'nelecas': (
            [int(value) for value in nelecas]
            if isinstance(nelecas, (list, tuple))
            else int(nelecas)
        ),
        'active_column_range': [int(ncore), int(ncore + ncas - 1)],
        'reference': str(reference or 'restricted'),
        'orbital_provenance': dict(orbital_provenance or {}),
        'molecular_basis_signature': dict(molecular_signature or {}),
        'molecule': molecule.dumps() if molecule is not None else None,
        'frozen_orbital_indices': list(frozen_orbital_indices or []),
    }
    manifest['orbital_context'] = context
    manifest['checkpoint_components'] = ['mps', 'optimized_orbitals']
    return context


def _orthonormalize_orbitals(matrix: Any, overlap: Any) -> Tuple[Any, float]:
    import numpy as np  # pylint: disable=import-outside-toplevel

    metric = matrix.T.conj().dot(overlap).dot(matrix)
    metric = 0.5 * (metric + metric.T.conj())
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    if eigenvalues.size == 0 or float(np.min(eigenvalues)) <= 1.0e-10:
        raise ValueError('Checkpoint orbitals are linearly dependent in the target AO metric.')
    inverse_sqrt = eigenvectors.dot(np.diag(eigenvalues ** -0.5)).dot(eigenvectors.T.conj())
    transformed = matrix.dot(inverse_sqrt)
    error = float(np.max(np.abs(
        transformed.T.conj().dot(overlap).dot(transformed) - np.eye(transformed.shape[1])
    )))
    return transformed.real, error


def load_optimized_orbitals(
    manifest_value: Any,
    mf: Any,
    *,
    ncas: int,
    nelecas: Any,
    require_same_active_space: bool = True,
    allow_geometry_projection: bool = False,
    restart_geometry_policy: str = 'same_geometry',
    restart_min_active_overlap: float = 0.9,
    frozen_orbital_indices: Optional[list[int]] = None,
    continuation_policy: str = 'project_all',
    casscf: Any = None,
) -> Tuple[Any, Dict[str, Any]]:
    """Load unchanged MOs or explicitly transport an orbital/MPS initial guess.

    By default joint restarts preserve the saved orbital basis exactly.
    Explicit transport interprets saved occupation coefficients in continuously
    aligned target orbitals as an approximate initial guess, not an exact
    wavefunction transformation. Source geometry metadata is required.
    """

    import numpy as np  # pylint: disable=import-outside-toplevel

    manifest = read_checkpoint_manifest(manifest_value)
    if restart_geometry_policy not in ('same_geometry', 'transport'):
        raise ValueError('Unsupported MPS restart_geometry_policy.')
    transport = restart_geometry_policy == 'transport'
    if transport and (allow_geometry_projection or not require_same_active_space
                      or continuation_policy != 'project_all'):
        raise ValueError('MPS transport requires a joint checkpoint and project_all policy.')
    if not isinstance(manifest, dict):
        raise ValueError('A DMRG-CASSCF orbital restart requires a block2 checkpoint manifest.')
    context = manifest.get('orbital_context')
    context = context if isinstance(context, dict) else {}
    if context.get('external_restart_supported') is not True:
        raise ValueError(
            'The block2 checkpoint does not contain externally reusable optimized CASSCF orbitals.'
        )
    source_ncas = int(context.get('ncas') or -1)
    source_nelecas = context.get('nelecas')
    mol = getattr(mf, 'mol', None)
    spin = int(getattr(mol, 'spin', 0) or 0)
    normalized_source_nelecas = _spin_resolved_nelecas(source_nelecas, spin)
    normalized_nelecas = _spin_resolved_nelecas(nelecas, spin)
    if require_same_active_space and (
        source_ncas != int(ncas)
        or normalized_source_nelecas != normalized_nelecas
    ):
        raise ValueError('Joint orbital/MPS restart requires the same ncas and nelecas as its source.')
    source_signature = context.get('molecular_basis_signature')
    source_signature = source_signature if isinstance(source_signature, dict) else {}
    target_signature = molecular_basis_signature(mol)
    if source_signature and source_signature != target_signature:
        raise ValueError(
            'Checkpoint optimized orbitals use a different atom ordering, basis, or AO layout.'
        )

    orbital_file = context.get('orbital_file')
    orbital_file = orbital_file if isinstance(orbital_file, dict) else {}
    relative_path = str(orbital_file.get('relative_path') or '').strip()
    source_directory = Path(str(manifest.get('scratch_directory') or '')).expanduser().resolve()
    path = source_directory / relative_path
    if not relative_path or not path.is_file():
        raise ValueError('The optimized-orbital file referenced by the block2 checkpoint is missing.')
    matrix = np.load(path, allow_pickle=False)
    if matrix.ndim != 2:
        raise ValueError('The optimized-orbital checkpoint is not a two-dimensional matrix.')
    expected_nao = int(mol.nao_nr()) if mol is not None and hasattr(mol, 'nao_nr') else matrix.shape[0]
    if matrix.shape[0] != expected_nao:
        raise ValueError('Checkpoint orbitals are incompatible with the target AO dimension.')
    from pyscf import gto
    from ...backend.correlation.orbital_projection import project_partitioned_orbitals

    source_dump = context.get('molecule')
    if not source_dump:
        raise ValueError('Checkpoint lacks source geometry; verify and migrate its orbital metadata before restart.')
    source_mol = gto.loads(source_dump)
    if molecular_basis_signature(source_mol) != target_signature:
        raise ValueError('Checkpoint source molecule differs from the target AO layout.')
    same_geometry = geometry_matches(source_mol, mol)
    if transport:
        if (source_mol.charge, source_mol.spin) != (mol.charge, mol.spin):
            raise ValueError('MPS transport requires the same charge and spin.')
        if casscf is not None and int(context['ncore']) != casscf.ncore:
            raise ValueError('MPS transport requires the same inactive partition.')
    projection = None
    if continuation_policy not in ('project_all', 'target_scf_core'):
        raise ValueError('Unsupported orbital continuation policy.')
    if continuation_policy == 'target_scf_core':
        if not allow_geometry_projection or casscf is None:
            raise ValueError('Target-SCF core continuation requires an orbital-only CASSCF restart with a fresh MPS.')
        if source_ncas != int(ncas) or normalized_source_nelecas != normalized_nelecas:
            raise ValueError('Target-SCF core continuation requires the same active space.')
        if int(context['ncore']) != casscf.ncore:
            raise ValueError('Target-SCF core continuation requires the same inactive partition.')
        from ...backend.correlation.orbital_projection import project_casscf_active_guess
        transformed, projection = project_casscf_active_guess(
            source_mol, casscf, matrix,
            frozen_indices=list(frozen_orbital_indices or []),
        )
        error = projection['orthonormality_max_abs_error']
    elif not same_geometry:
        if not allow_geometry_projection and not transport:
            raise ValueError('MPS continuation requires the same geometry and orbital basis; use an orbital-only projection with a fresh MPS.')
        source_frozen = list(context.get('frozen_orbital_indices') or [])
        if frozen_orbital_indices is not None and sorted(source_frozen) != sorted(frozen_orbital_indices):
            raise ValueError('Cross-geometry projection requires the same frozen-core partition.')
        if not (source_ncas == int(ncas) and normalized_source_nelecas == normalized_nelecas):
            raise ValueError('Cross-geometry projection requires the same active-space partition.')
        if transport:
            from ...backend.correlation.orbital_projection import transport_mps_orbitals
            transformed, projection = transport_mps_orbitals(
                source_mol, mol, matrix, ncore=int(context['ncore']), ncas=int(ncas),
                frozen_indices=source_frozen, minimum_active_overlap=restart_min_active_overlap,
            )
        else:
            transformed, projection = project_partitioned_orbitals(
                source_mol, mol, matrix, ncore=int(context['ncore']), ncas=int(ncas),
                frozen_indices=source_frozen,
            )
        error = projection['orthonormality_max_abs_error']
    else:
        overlap = mf.get_ovlp() if hasattr(mf, 'get_ovlp') else mol.intor_symmetric('int1e_ovlp')
        error = float(np.max(np.abs(matrix.T.conj() @ overlap @ matrix - np.eye(matrix.shape[1]))))
        if not np.isfinite(error) or error > 1e-8:
            raise ValueError('Same-basis checkpoint orbitals are not orthonormal; cannot rotate orbitals while reusing an MPS.')
        transformed = matrix.copy()
    return transformed, {
        'schema': JOINT_CHECKPOINT_SCHEMA,
        'source_scratch_directory': str(source_directory),
        'source_orbital_file': dict(orbital_file),
        'source_ncas': source_ncas,
        'source_nelecas': normalized_source_nelecas,
        'target_ncas': int(ncas),
        'target_nelecas': normalized_nelecas,
        'same_active_space': (
            source_ncas == int(ncas)
            and normalized_source_nelecas == normalized_nelecas
        ),
        'molecular_basis_signature_match': (
            source_signature == target_signature if source_signature else None
        ),
        'same_geometry': same_geometry,
        'restart_geometry_policy': restart_geometry_policy,
        'mps_initial_guess_transport': bool(transport and not same_geometry),
        'continuation_policy': continuation_policy,
        'metric_orthonormalization': 'partitioned' if projection else 'none_basis_preserved',
        'orbital_projection': projection,
        'orthonormality_max_abs_error': error,
    }


__all__ = [
    'JOINT_CHECKPOINT_SCHEMA',
    'OPTIMIZED_ORBITAL_FILENAME',
    'attach_optimized_orbitals',
    'load_optimized_orbitals',
    'molecular_basis_signature',
    'pin_restart_orbital_ordering',
    'read_checkpoint_manifest',
]
