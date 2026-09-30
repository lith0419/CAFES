from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Sequence

from ..backend.artifacts import write_json_artifact
from ..providers.libdmet.transforms import (
    identity_localized_orbitals,
    transform_density_to_local,
    transform_one_body_to_local,
)
from .artifacts import write_embedding_hdf5_artifact
from .contracts import build_correlated_subspace_audit


def prepare_embedding_artifacts(
    state: Dict[str, Any],
    *,
    system_type: str,
    one_body: Any,
    density: Any,
    overlap: Any,
    correlated_orbital_indices: Sequence[int],
    coefficients: Any = None,
    two_body: Any = None,
    two_body_spin_pair_order: str = '',
    fock: Any = None,
    provider: str = 'libdmet',
    localization_method: str = 'manual',
    matrix_layout: str = 'auto',
    orbital_labels: Optional[Sequence[Any]] = None,
    occupations: Optional[Sequence[Any]] = None,
    atom_indices: Optional[Sequence[int]] = None,
    fragments: Optional[Sequence[Mapping[str, Any]]] = None,
    orbital_contributions: Optional[Mapping[Any, Any]] = None,
    interaction: Optional[Mapping[str, Any]] = None,
    selection_reasons: Optional[Sequence[str]] = None,
    electron_count: Any = None,
    additional_one_body_operators: Optional[Mapping[str, Any]] = None,
) -> Dict[str, Any]:
    """Create the common numerical boundary consumed by DMET and DMFT modules.

    ``one_body``, ``fock``, and ``density`` are transformed with the supplied
    localized-orbital coefficients.  ``two_body`` must already use that same
    localized basis; this shared helper does not transform four-index ERIs.
    """

    import numpy as np

    one_body_array = np.asarray(one_body)
    density_array = np.asarray(density)
    overlap_array = np.asarray(overlap)
    if coefficients is None:
        normalized_layout = str(matrix_layout or 'auto').strip().lower()
        leading_shape = (
            one_body_array.shape[:-2]
            if normalized_layout in ('kpoint', 'spin_kpoint')
            or (normalized_layout == 'auto' and one_body_array.ndim == 3)
            else None
        )
        coefficients = identity_localized_orbitals(
            one_body_array.shape[-1],
            layout=matrix_layout,
            leading_shape=leading_shape,
        )
    coefficient_array = np.asarray(coefficients)
    local_one_body = transform_one_body_to_local(
        one_body_array,
        coefficient_array,
        layout=matrix_layout,
    )
    local_density = transform_density_to_local(
        density_array,
        coefficient_array,
        overlap_array,
        layout=matrix_layout,
    )
    local_fock = (
        transform_one_body_to_local(
            np.asarray(fock),
            coefficient_array,
            layout=matrix_layout,
        )
        if fock is not None
        else None
    )
    common_metadata = {
        'system_type': str(system_type or '').strip().lower(),
        'provider': str(provider or '').strip().lower(),
        'localization_method': str(localization_method or '').strip().lower(),
        'matrix_layout': str(matrix_layout or 'auto').strip().lower(),
        'correlated_orbital_indices': [int(index) for index in correlated_orbital_indices],
    }
    if two_body is not None:
        pair_order = str(two_body_spin_pair_order or '').strip().lower()
        two_body_array = np.asarray(two_body)
        spin_blocks = 1 if two_body_array.ndim == 4 else int(two_body_array.shape[0])
        if spin_blocks == 1:
            pair_order = 'restricted'
        elif spin_blocks == 3 and pair_order != 'aa_bb_ab':
            raise ValueError(
                'Unrestricted embedding ERIs require two_body_spin_pair_order=aa_bb_ab'
            )
        common_metadata['two_body_spin_pair_order'] = pair_order
    reference_datasets = {
        'one_body': one_body_array,
        'density': density_array,
        'overlap': overlap_array,
    }
    if fock is not None:
        reference_datasets['fock'] = np.asarray(fock)
    reference_ref = write_embedding_hdf5_artifact(
        state,
        'embedding_reference',
        'embedding-reference.h5',
        datasets=reference_datasets,
        metadata=common_metadata,
        description='Mean-field numerical reference for embedding preparation.',
    )
    subspace_ref = write_embedding_hdf5_artifact(
        state,
        'localized_subspace',
        'localized-subspace.h5',
        datasets={'coefficients_ao_lo': coefficient_array},
        metadata={
            **common_metadata,
            'orbital_labels': [str(label) for label in (orbital_labels or [])],
        },
        description='Localized-orbital definition and correlated-subspace mapping.',
    )
    local_datasets = {
        'one_body_local': local_one_body,
        'density_local': local_density,
    }
    if local_fock is not None:
        local_datasets['fock_local'] = local_fock
    if two_body is not None:
        local_datasets['two_body_local'] = np.asarray(two_body)
    for raw_name, operator in dict(additional_one_body_operators or {}).items():
        name = str(raw_name or '').strip()
        if not name:
            raise ValueError('Additional one-body operator names must be non-empty')
        if name in local_datasets:
            raise ValueError("Additional one-body operator '{0}' duplicates a standard dataset".format(name))
        local_datasets[name] = transform_one_body_to_local(
            np.asarray(operator),
            coefficient_array,
            layout=matrix_layout,
        )
    hamiltonian_ref = write_embedding_hdf5_artifact(
        state,
        'localized_hamiltonian',
        'localized-hamiltonian.h5',
        datasets=local_datasets,
        metadata={
            **common_metadata,
            'interaction': copy.deepcopy(dict(interaction or {})),
            'two_body_basis': 'localized' if two_body is not None else 'not_provided',
        },
        description='One-particle operators and optional local interaction tensor in the localized basis.',
    )
    audit = build_correlated_subspace_audit(
        system_type=system_type,
        provider=provider,
        localization_method=localization_method,
        orbital_indices=correlated_orbital_indices,
        total_orbitals=coefficient_array.shape[-1],
        atom_indices=atom_indices,
        fragments=fragments,
        orbital_labels=orbital_labels,
        occupations=occupations,
        orbital_contributions=orbital_contributions,
        selection_reasons=selection_reasons,
        interaction=interaction,
        electron_count=electron_count,
    )
    audit['artifacts'] = {
        'embedding_reference': copy.deepcopy(reference_ref),
        'localized_subspace': copy.deepcopy(subspace_ref),
        'localized_hamiltonian': copy.deepcopy(hamiltonian_ref),
    }
    audit_ref = write_json_artifact(
        state,
        'correlated_subspace_audit',
        'correlated-subspace-audit.json',
        audit,
        description='Review and approval record for the correlated orbital subspace.',
    )
    embedding_task_patch = {
        'enabled': True,
        'provider': common_metadata['provider'],
        'localization_method': common_metadata['localization_method'],
        'correlated_orbital_indices': list(common_metadata['correlated_orbital_indices']),
        'correlated_atom_indices': [int(index) for index in (atom_indices or [])],
        'fragments': copy.deepcopy(list(fragments or [])),
        'interaction': copy.deepcopy(dict(interaction or {})),
        'approved': bool(audit['approval']['approved']),
        'audit_artifact': copy.deepcopy(audit_ref),
        'reference_artifact': copy.deepcopy(reference_ref),
        'localized_subspace_artifact': copy.deepcopy(subspace_ref),
        'localized_hamiltonian_artifact': copy.deepcopy(hamiltonian_ref),
    }
    return {
        'embedding_reference': reference_ref,
        'localized_subspace': subspace_ref,
        'localized_hamiltonian': hamiltonian_ref,
        'correlated_subspace_audit': audit_ref,
        'audit': audit,
        'task_spec_patch': {'embedding': embedding_task_patch},
    }
