"""Optional libDMET provider for embedding preparation and Hubbard DMET."""

from .dmet import (
    DMET_ITERATION_HISTORY_SCHEMA,
    DMET_RESULT_SCHEMA,
    normalize_dmet_options,
    run_hubbard_dmet,
    validate_dmet_model_request,
)
from .availability import libdmet_availability, libdmet_is_available
from .block2_impurity import Block2DmetImpuritySolver, DMET_BLOCK2_IMPURITY_SCHEMA
from .translation import (
    TRANSLATION_AUDIT_SCHEMA,
    audit_builder_translation,
    resolve_primitive_cell_layout,
)
from .transforms import (
    build_lowdin_coefficients,
    build_molecular_iao_coefficients,
    build_periodic_iao_coefficients,
    identity_localized_orbitals,
    transform_density_to_local,
    transform_one_body_to_local,
)

__all__ = [
    'DMET_ITERATION_HISTORY_SCHEMA',
    'DMET_BLOCK2_IMPURITY_SCHEMA',
    'DMET_RESULT_SCHEMA',
    'TRANSLATION_AUDIT_SCHEMA',
    'audit_builder_translation',
    'build_lowdin_coefficients',
    'build_molecular_iao_coefficients',
    'build_periodic_iao_coefficients',
    'Block2DmetImpuritySolver',
    'identity_localized_orbitals',
    'libdmet_availability',
    'libdmet_is_available',
    'normalize_dmet_options',
    'run_hubbard_dmet',
    'resolve_primitive_cell_layout',
    'transform_density_to_local',
    'transform_one_body_to_local',
    'validate_dmet_model_request',
]
