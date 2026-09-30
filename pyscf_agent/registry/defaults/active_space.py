from __future__ import annotations

from typing import Any, Callable, Dict, List

BLOCK2_PRESETS = {
    'screening': {
        'bond_dimensions': [100] * 4 + [200] * 4,
        'noises': [1e-4] * 4 + [1e-5] * 2 + [0.0] * 2,
        'davidson_thresholds': [1e-7] * 4 + [1e-8] * 4,
    },
    'balanced': {
        'bond_dimensions': [250] * 4 + [500] * 6,
        'noises': [1e-4] * 4 + [1e-5] * 3 + [0.0] * 3,
        'davidson_thresholds': [1e-8] * 4 + [1e-9] * 6,
    },
    'high_accuracy': {
        'bond_dimensions': [500] * 4 + [1000] * 8,
        'noises': [1e-4] * 4 + [1e-5] * 4 + [0.0] * 4,
        'davidson_thresholds': [1e-9] * 4 + [1e-10] * 8,
    },
}


def build_probe_strategy_capabilities(
    item_factory: Callable[..., Any],
    *,
    namespace: str,
) -> List[Any]:
    """Share probe execution and refinement policy between Assistant and Study."""
    return [
        item_factory(
            'auto',
            'Auto',
            namespace,
            metadata={
                'execution_method': 'hf',
                'refinement_method': 'mp2',
                'refinement_triggers': [
                    'active_space_candidate_not_review_ready',
                    'scf_instability',
                    'frontier_orbital_degeneracy',
                    'near_correlation_decision_boundary',
                ],
                'default': True,
                'supported_system_types': ['molecular'],
            },
        ),
        item_factory(
            'mp2',
            'MP2',
            namespace,
            metadata={
                'execution_method': 'mp2',
                'supported_system_types': ['molecular'],
            },
        ),
        item_factory(
            'fci',
            'FCI',
            namespace,
            metadata={
                'execution_method': 'fci',
                'supported_system_types': ['molecular'],
            },
            limitations=['Use only for small exact-diagnostic probes after resource review.'],
        ),
    ]


def build_active_space_families(
    item_factory: Callable[..., Any],
    *,
    capability_design_only: str,
    capability_planned: str,
) -> Dict[str, List[Any]]:
    _capability = item_factory
    CAPABILITY_DESIGN_ONLY = capability_design_only
    CAPABILITY_PLANNED = capability_planned

    return {
        'active_space_probe_strategies': build_probe_strategy_capabilities(
            item_factory, namespace='molecular.active_space_probe_strategy',
        ),
        'active_space_selection_methods': [
            _capability('manual', 'Manual active space', 'molecular.active_space'),
            _capability('occupation_window', 'Occupation-window active space proposal', 'molecular.active_space'),
            _capability('energy_window', 'Energy-window active space proposal', 'molecular.active_space'),
            _capability(
                'uno',
                'UNO active-space candidate',
                'molecular.active_space',
                CAPABILITY_DESIGN_ONLY,
                planner_allowed=False,
                backend_allowed=False,
                ui_visibility='hidden',
                limitations=['Generated inside ActiveSpaceAudit as an internal candidate; not a direct request selection mode yet.'],
            ),
            _capability(
                'avas',
                'AVAS active space',
                'molecular.active_space',
                limitations=[
                    'Provide AO labels or atom:/fragment: selectors, then run an MP2/HF-style active-space audit before CAS approval.',
                    'Unrestricted probe references are projected through a spin-summed UNO spatial basis before AVAS; the approved CAS execution remains restricted or ROHF spin-adapted.',
                ],
            ),
            _capability(
                'chemical_valence',
                'Chemical-valence active-space candidate',
                'molecular.active_space',
                CAPABILITY_DESIGN_ONLY,
                planner_allowed=False,
                backend_allowed=False,
                ui_visibility='hidden',
                limitations=[
                    'Generated inside ActiveSpaceAudit from complete element valence AO manifolds and PySCF AVAS projection.',
                    'It is a review candidate, not a molecule-specific guarantee of the optimal active space.',
                ],
            ),
            _capability(
                'evidence_expanded',
                'Evidence-expanded active-space candidate',
                'molecular.active_space',
                CAPABILITY_DESIGN_ONLY,
                planner_allowed=False,
                backend_allowed=False,
                ui_visibility='hidden',
                limitations=[
                    'Generated inside ActiveSpaceAudit by adding only critical natural-occupation or T2 evidence to a complete chemical-valence baseline.',
                    'Near-degenerate partners are retained together; a fixed orbital-count truncation is never applied.',
                ],
            ),
            _capability(
                'merged',
                'Merged active-space candidate',
                'molecular.active_space',
                CAPABILITY_DESIGN_ONLY,
                planner_allowed=False,
                backend_allowed=False,
                ui_visibility='hidden',
                limitations=[
                    'Generated inside ActiveSpaceAudit as the broad union of available evidence.',
                    'Retained as a diagnostic alternative and not preferred over a smaller chemically complete candidate.',
                ],
            ),
        ],
        'active_space_solvers': [
            _capability(
                'fci',
                'PySCF Full CI active-space solver',
                'molecular.active_space_solver',
                required_provider_ids=('provider.pyscf',),
                metadata={
                    'compatible_methods': ['casci', 'casscf'],
                },
            ),
            _capability(
                'block2_dmrg',
                'block2 DMRG active-space solver',
                'molecular.active_space_solver',
                required_provider_ids=('provider.block2',),
                metadata={
                    'aliases': ['block2', 'block2-dmrg', 'dmrg'],
                    'compatible_methods': ['casci', 'casscf'],
                    'optional_dependency': 'block2',
                    'presets': BLOCK2_PRESETS,
                    'automatic_recovery': {
                        'max_bond_dimension': 1024,
                        'max_adaptive_stages': 4,
                        'max_adaptive_sweeps': 24,
                        'minimum_adaptive_sweeps': 8,
                        'adaptive_sweep_increment': 4,
                    },
                    'artifact_kinds': [
                        'block2_dmrg_result',
                        'block2_bond_dimension_plan',
                        'block2_dmrg_casscf',
                        'block2_dmrg_arrays',
                        'block2_mps_manifest',
                        'entanglement_active_space_recommendation',
                    ],
                    'active_orbital_localization': ['boys', 'pipek_mezey'],
                    'orbital_ordering': ['canonical', 'fiedler', 'manual'],
                    'mpo_algorithms': ['fast_bipartite', 'conventional'],
                    'default_mpo_algorithm': 'conventional',
                    'orbital_provenance_result': 'cas_result.orbital_provenance',
                    'adaptive_convergence': {
                        'controls': [
                            'max_adaptive_stages',
                            'max_bond_dimension',
                            'bond_dimension_growth_factor',
                            'adaptive_sweeps',
                            'adaptive_noise',
                        ],
                        'result_fields': ['adaptive_schedule', 'energy_error_estimate'],
                    },
                    'bond_dimension_planning': {
                        'method': 'fixed_particle_number_schmidt_rank',
                        'result_field': 'bond_dimension_plan',
                    },
                    'casscf_state_averaging': {
                        'mode': 'same_spin_and_symmetry_sector',
                        'options': ['nroots', 'state_average_weights'],
                        'result_fields': [
                            'state_energies',
                            'excitation_energies',
                            'state_average_energy',
                            'state_average_weights',
                        ],
                    },
                    'cross_task_casscf_restart': 'joint_optimized_orbitals_and_mps',
                    'cross_geometry_mps_initial_guess': {
                        'option': 'restart_geometry_policy',
                        'choices': ['same_geometry', 'transport'],
                        'default': 'same_geometry',
                        'minimum_active_overlap_option': 'restart_min_active_overlap',
                        'minimum_active_overlap_default': 0.9,
                        'exact_wavefunction_transform': False,
                        'requires': ['joint_checkpoint', 'same_active_space', 'same_root_weights',
                                     'same_atom_basis_layout', 'preserved_mps_site_order'],
                    },
                    'active_space_expansion': 'entanglement_active_space_audit_approval',
                    'result_contract_schema': 'pyscf-agent.block2-result-contract.v1',
                    'recovery_schema': 'pyscf-agent.block2-recovery-recommendation.v1',
                },
                limitations=[
                    'The integration supports fixed-orbital DMRG-CASCI and orbital-optimized DMRG-CASSCF with restricted or ROHF references, including optional Boys/Pipek-Mezey initial active-orbital localization and canonical/Fiedler/manual DMRG ordering.',
                    'DMRG-CASSCF supports single-state optimization and state-averaged optimization of multiple roots in one spin and symmetry sector. Different-spin state averaging is not implemented yet.',
                    'Compatible cross-task continuation requires a joint optimized-orbital/MPS checkpoint; an expanded active space reuses only the optimized orbitals and initializes a fresh MPS.',
                    'Unrestricted active spaces and SC-NEVPT2 coupling are not implemented yet.',
                ],
            ),
        ],
        'orbital_processing_tools': [
            _capability(
                'orbital_continuation',
                'CASSCF orbital continuation policy',
                'molecular.orbital_processing',
                metadata={
                    'task_spec_field': 'orbital_processing.continuation_policy',
                    'values': ['project_all', 'target_scf_core'],
                    'default': 'project_all',
                    'meaning': 'target_scf_core keeps explicitly selected target SCF core columns, projects the active guess first, then the remaining inactive guess, and completes all external orbitals from the target basis. It requires an orbital-only CASSCF restart with a fresh MPS.',
                },
            ),
            _capability(
                'frozen_orbitals',
                'Frozen CASSCF orbitals',
                'molecular.orbital_processing',
                metadata={
                    'task_spec_field': 'orbital_processing.frozen_orbital_indices',
                    'methods': ['casscf'],
                    'solvers': ['fci', 'block2_dmrg'],
                    'reference': 'restricted_or_rohf',
                    'default': [],
                    'index_basis': 'zero_based_full_casscf_mo_columns_after_active_space_preparation',
                    'provenance_result': 'cas_result.orbital_optimization',
                    'meaning': 'Keep explicitly listed MOs fixed during orbital optimization using PySCF mc.frozen. This does not remove AO basis functions or change the CAS electron count.',
                },
            ),
            _capability(
                'orbital_table',
                'Orbital table',
                'molecular.orbital_processing',
                metadata={
                    'task_spec_field': 'orbital_processing',
                    'result_field': 'orbital_processing',
                    'summary_fields': ['enabled', 'use_natural_orbitals', 'notes'],
                    'preview_tables': {'orbital_table': {'limit': 8}},
                    'artifact_kinds': ['orbital_processing', 'orbital_summary_table'],
                    'generated_output_rule': 'Orbital processing results and artifacts are generated outputs when this result field is present.',
                },
            ),
            _capability(
                'boys',
                'Boys localization',
                'molecular.orbital_processing',
                metadata={
                    'localization_method': True,
                    'task_spec_field': 'orbital_processing',
                    'result_field': 'orbital_processing',
                    'summary_fields': ['enabled', 'localization_method', 'localization_scope', 'localization_status', 'localized_orbital_shape', 'orbital_ordering', 'notes'],
                    'preview_tables': {'orbital_table': {'limit': 8}},
                    'artifact_kinds': ['orbital_processing', 'orbital_summary_table'],
                    'supported_scopes': ['analysis', 'active_space'],
                    'localization_occupation_thresholds': {
                        'default': [], 'example': [1.0],
                        'meaning': 'Optional increasing boundaries in (0,2); diagonalize the projected spin-summed SCF density and localize each occupation interval independently.',
                    },
                    'active_space_execution': {
                        'methods': ['casci', 'casscf'],
                        'solvers': ['block2_dmrg'],
                        'provenance_result': 'cas_result.orbital_provenance',
                    },
                    'generated_output_rule': 'A completed Boys localization is an orbital-processing output; do not report that no output was generated when localization_status is present.',
                },
            ),
            _capability(
                'pipek_mezey',
                'Pipek-Mezey localization',
                'molecular.orbital_processing',
                metadata={
                    'localization_method': True,
                    'task_spec_field': 'orbital_processing',
                    'result_field': 'orbital_processing',
                    'summary_fields': ['enabled', 'localization_method', 'localization_scope', 'localization_status', 'localized_orbital_shape', 'orbital_ordering', 'notes'],
                    'preview_tables': {'orbital_table': {'limit': 8}},
                    'artifact_kinds': ['orbital_processing', 'orbital_summary_table'],
                    'supported_scopes': ['analysis', 'active_space'],
                    'localization_occupation_thresholds': {
                        'default': [], 'example': [1.0],
                        'meaning': 'Optional increasing boundaries in (0,2); diagonalize the projected spin-summed SCF density and localize each occupation interval independently.',
                    },
                    'active_space_execution': {
                        'methods': ['casci', 'casscf'],
                        'solvers': ['block2_dmrg'],
                        'provenance_result': 'cas_result.orbital_provenance',
                    },
                    'generated_output_rule': 'A completed Pipek-Mezey localization is an orbital-processing output; do not report that no output was generated when localization_status is present.',
                },
            ),
            _capability(
                'correlated_natural_orbitals',
                'Correlated natural orbitals',
                'molecular.orbital_processing',
                CAPABILITY_PLANNED,
                planner_allowed=False,
                backend_allowed=False,
                limitations=['Current natural-orbital summary uses available mean-field occupations unless a correlated density is provided.'],
            ),
        ],
    }


__all__ = ['build_active_space_families', 'build_probe_strategy_capabilities']
