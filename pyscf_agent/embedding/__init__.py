"""Reusable embedding preparation contracts and numerical artifacts."""

from .artifacts import (
    EMBEDDING_HDF5_SCHEMAS,
    read_embedding_hdf5,
    serialize_embedding_hdf5,
    write_embedding_hdf5_artifact,
)
from .contracts import (
    CORRELATED_SUBSPACE_AUDIT_SCHEMA,
    EMBEDDING_REFERENCE_SCHEMA,
    LOCALIZED_HAMILTONIAN_SCHEMA,
    LOCALIZED_SUBSPACE_SCHEMA,
    approve_correlated_subspace_audit,
    build_correlated_subspace_audit,
)
from .preparation import prepare_embedding_artifacts
from .reference_density import (
    DEFAULT_REFERENCE_DENSITY_BIAS,
    DEFAULT_REFERENCE_DENSITY_GUESS,
    REFERENCE_DENSITY_GUESSES,
    build_reference_density_seed,
    normalize_reference_density_bias,
    normalize_reference_density_guess,
    validate_reference_density_guess,
)

__all__ = [
    'CORRELATED_SUBSPACE_AUDIT_SCHEMA',
    'EMBEDDING_HDF5_SCHEMAS',
    'EMBEDDING_REFERENCE_SCHEMA',
    'LOCALIZED_HAMILTONIAN_SCHEMA',
    'LOCALIZED_SUBSPACE_SCHEMA',
    'DEFAULT_REFERENCE_DENSITY_BIAS',
    'DEFAULT_REFERENCE_DENSITY_GUESS',
    'REFERENCE_DENSITY_GUESSES',
    'approve_correlated_subspace_audit',
    'build_correlated_subspace_audit',
    'build_reference_density_seed',
    'normalize_reference_density_bias',
    'normalize_reference_density_guess',
    'prepare_embedding_artifacts',
    'read_embedding_hdf5',
    'serialize_embedding_hdf5',
    'write_embedding_hdf5_artifact',
    'validate_reference_density_guess',
]
