from __future__ import annotations

from .active_space import (
    CAS_METHODS as CAS_METHODS,
    active_space_audit_table as active_space_audit_table,
    active_space_summary_table as active_space_summary_table,
    configure_fci_solver as configure_fci_solver,
    configure_single_root_fci_solver as configure_single_root_fci_solver,
    orbital_summary_table as orbital_summary_table,
    propose_active_space as propose_active_space,
    run_cas_method as run_cas_method,
    state_target_configuration as state_target_configuration,
)
from .molecular import (
    FRONTIER_DEGENERACY_TOLERANCE as FRONTIER_DEGENERACY_TOLERANCE,
    FRONTIER_OCCUPATION_EPS as FRONTIER_OCCUPATION_EPS,
    _as_float as _as_float,
    _clip_unit as _clip_unit,
    _flatten_restricted_values as _flatten_restricted_values,
    _frontier_orbital_degeneracy_summary as _frontier_orbital_degeneracy_summary,
    _mean_score as _mean_score,
    _molecular_method_recommendation as _molecular_method_recommendation,
    _natural_occupation_metrics as _natural_occupation_metrics,
    _orbital_table as _orbital_table,
    _risk_level as _risk_level,
    build_correlation_diagnostics as build_correlation_diagnostics,
    build_molecular_correlation_risk as build_molecular_correlation_risk,
    build_orbital_processing_summary as build_orbital_processing_summary,
    build_scf_stability_summary as build_scf_stability_summary,
)

# Historical private names remain available to existing backend callers.
from .orbital_diagnostics import natural_orbital_summary as _natural_orbital_summary  # noqa: F401 - compatibility export
from .orbital_diagnostics import occupation_fractionality as _occupation_fractionality  # noqa: F401 - compatibility export
from .orbital_diagnostics import t2_orbital_importance as _t2_orbital_importance  # noqa: F401 - compatibility export
