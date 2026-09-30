from __future__ import annotations

from pyscf_agent.schema_contracts import BATCH_EXECUTION_RESULT_SCHEMA, BATCH_HANDLE_SCHEMA, BATCH_TASK_SCHEMA, JOB_HANDLE_SCHEMA, JOB_STATUS_SCHEMA, SLURM_BATCH_MANIFEST_SCHEMA, STUDY_EXECUTION_RECEIPT_SCHEMA, STUDY_EXECUTION_STATUS_SCHEMA

from typing import Any, Callable, Dict, List

from .active_space import build_probe_strategy_capabilities


def build_study_families(
    item_factory: Callable[..., Any],
    *,
    capability_design_only: str,
    capability_planned: str,
) -> Dict[str, List[Any]]:
    _capability = item_factory
    CAPABILITY_DESIGN_ONLY = capability_design_only
    CAPABILITY_PLANNED = capability_planned

    return {
        'study_modes': [
            _capability(
                'static',
                'Static Scan',
                'study.mode',
                metadata={
                    'aliases': ['static_scan'],
                    'case_grid': 'full_requested_grid',
                    'optional_grid_refinement': 'axis_midpoint',
                    'automatic_method_routing': False,
                    'supported_system_types': ['molecular', 'model_hamiltonian'],
                },
            ),
            _capability(
                'adaptive',
                'Adaptive Scan',
                'study.mode',
                metadata={
                    'aliases': ['adaptive_scan'],
                    'case_grid': 'full_requested_grid',
                    'automatic_method_routing': True,
                    'supported_system_types': ['molecular'],
                },
            ),
        ],
        'adaptive_initial_scan_strategies': build_probe_strategy_capabilities(
            item_factory, namespace='study.adaptive_initial_scan',
        ),
        'study_policies': [
            _capability(
                'grid_refinement', 'Automatic parameter-grid refinement', 'study.policy',
                metadata={
                    'supported_system_types': ['model_hamiltonian'],
                    'study_modes': ['static'], 'configuration_field': 'grid_refinement',
                    'dimensions': [1, 2], 'algorithm': 'alternating_axis_midpoint',
                    'continuous_parameters': ['U', 'V', 't', 'epsilon'],
                    'supports_template_variable_names': True,
                    'automatic_method_routing': False, 'budget_unit': 'new_calculation_points',
                    'default_max_new_points': 64, 'default_max_rounds_per_axis': 4,
                    'result_field': 'grid_refinement',
                    'supports_convergence_barrier': True,
                    'convergence_retry_fields': ['correlation_potential_mixing', 'max_iterations', 'diis_enabled'],
                    'metrics': ['energy_per_site', 'mean_double_occupancy',
                                'nearest_neighbor_spin_correlation', 'nearest_neighbor_charge_correlation',
                                'staggered_magnetization', 'sublattice_charge_imbalance'],
                },
            ),
            _capability(
                'adaptive_routing',
                'Adaptive method and resource routing policy',
                'study.policy',
                planner_allowed=False,
                ui_visibility='hidden',
                metadata={
                    'model_method_policy': {
                        'weak': 'mp2',
                        'moderate': 'ccsd',
                        'strong_small': 'fci',
                        'strong_large': 'ccsd',
                    },
                    'molecular_method_policy': {
                        'weak': 'ccsd_t',
                        'moderate': 'ccsd',
                        'strong_small': 'casscf',
                        'strong_large': 'casscf',
                    },
                    'resource_limits': {
                        'max_fci_sites': 8,
                        'max_molecular_fci_orbitals': 6,
                        'max_molecular_fci_electrons': 6,
                        'max_cas_orbitals': 14,
                    },
                    'recovery_solver_order': ['fci', 'ccsd', 'mp2'],
                    'molecular_single_reference_methods': ['hf', 'dft', 'mp2', 'ccsd_t'],
                },
            ),
            _capability(
                'scan_path_continuity',
                'Scan-path continuity diagnostic policy',
                'study.policy',
                planner_allowed=False,
                ui_visibility='hidden',
                metadata={
                    'minimum_sample_count': 4,
                    'slope_mismatch_ratio': 3.0,
                    'energy_residual_hartree': 0.005,
                    'strong_slope_mismatch_ratio': 6.0,
                    'molecular_method_levels': {
                        'hf': 0,
                        'dft': 0,
                        'mp2': 1,
                        'ccsd': 2,
                        'ccsd_t': 3,
                        'ccsd(t)': 3,
                        'casci': 4,
                        'casscf': 4,
                        'fci': 5,
                    },
                },
                limitations=[
                    'Thresholds diagnose continuity risks in finite sampled paths; they are not physical phase boundaries.',
                ],
            ),
        ],
        'study_workflow_options': [
            _capability(
                'block2_mps_continuation',
                'block2 MPS continuation',
                'study.workflow_option',
                planner_allowed=False,
                metadata={
                    'input_schema': 'pyscf-agent.block2-mps-manifest.v1',
                    'compatible_task_types': ['molecular', 'model_hamiltonian'],
                    'compatibility_checks': [
                        'n_orbitals',
                        'n_electrons',
                        'spin',
                        'symmetry',
                        'nroots',
                        'orbital_ordering_request',
                        'orbital_ordering_permutation',
                    ],
                    'hamiltonian_policy': 'allow_changed_integrals_with_matching_quantum_sector',
                },
                limitations=[
                    'Study-level continuation is proposed only during explicit result analysis and requires user approval.',
                    'The source and target must use the same executor target and compatible orbital, electron, spin, symmetry, root, and orbital-ordering sectors.',
                ],
            ),
            _capability(
                'block2_state_tracking',
                'block2 low-energy state tracking',
                'study.workflow_option',
                planner_allowed=False,
                metadata={
                    'result_schema': 'pyscf-agent.dmrg-state-tracking.v1',
                    'artifact_kinds': ['adaptive-dmrg-state-tracking'],
                    'compatible_task_types': ['molecular', 'model_hamiltonian'],
                    'execution_trigger': 'explicit_result_analysis',
                    'evidence_priority': [
                        'root_one_particle_density_matrix',
                        'root_natural_occupation_spectrum',
                        'excitation_energy',
                    ],
                    'ambiguity_policy': 'exclude_ambiguous_cases_from_automatic_mps_anchors',
                },
                limitations=[
                    'Root assignment is an evidence-based identity track, not a transition-density or wavefunction-overlap calculation.',
                ],
            ),
            _capability(
                'task_job_lifecycle',
                'Submitted task job lifecycle',
                'study.workflow_option',
                metadata={
                    'handle_schema': JOB_HANDLE_SCHEMA,
                    'status_schema': JOB_STATUS_SCHEMA,
                    'states': ['queued', 'running', 'completed', 'failed', 'cancelled'],
                    'operations': ['submit_task', 'status', 'cancel', 'fetch', 'logs', 'artifacts'],
                    'status_boundary': {
                        'job_state': 'scheduler_lifecycle',
                        'task_report_execution_status': 'scientific_task_outcome',
                    },
                    'local_submission_mode': 'immediate',
                    'remote_submission_mode': 'deferred',
                },
            ),
            _capability(
                'slurm_independent_task_batches',
                'Slurm independent-task batches',
                'study.workflow_option',
                metadata={
                    'executor_id': 'slurm',
                    'batch_transport': 'job_array',
                    'task_schema': BATCH_TASK_SCHEMA,
                    'batch_handle_schema': BATCH_HANDLE_SCHEMA,
                    'batch_result_schema': BATCH_EXECUTION_RESULT_SCHEMA,
                    'manifest_schema': SLURM_BATCH_MANIFEST_SCHEMA,
                    'initial_scan_execution': 'independent_batch',
                    'refinement_execution': 'single_task_queue',
                    'dependency_policy': 'projected_1rdm_tasks_are_not_batched',
                    'verification_command': 'configure.py verify-remote --submit-batch-smoke',
                    'verification_scope': 'two independent H2/HF tasks in one scheduler batch with TaskReport and artifact checks',
                },
            ),
            _capability(
                'recoverable_study_execution',
                'Recoverable submitted-study execution',
                'study.workflow_option',
                metadata={
                    'receipt_schema': STUDY_EXECUTION_RECEIPT_SCHEMA,
                    'status_schema': STUDY_EXECUTION_STATUS_SCHEMA,
                    'artifact_kinds': ['execution-receipt'],
                    'operations': ['inspect', 'collect'],
                    'resubmission_policy': 'reuse_matching_submitted_jobs',
                },
            ),
            _capability(
                'resource_cost_review',
                'Resource cost review',
                'study.workflow_option',
                metadata={
                    'schema': 'pyscf-agent.cost-estimate.v1',
                    'artifact_kinds': ['cost-estimate'],
                },
            ),
            _capability(
                'checkpoint_resume',
                'Checkpoint and resume',
                'study.workflow_option',
                metadata={
                    'schema': 'pyscf-agent.study-state.v1',
                    'artifact_kinds': ['study-state', 'adaptive-request-manifest'],
                    'retryable_statuses': ['failed', 'unconverged'],
                },
            ),
            _capability(
                'case_local_recovery',
                'Case-local adaptive recovery',
                'study.workflow_option',
                metadata={
                    'workflow_schema': 'pyscf-agent.adaptive-workflow.v1',
                    'decision_log_schema': 'pyscf-agent.adaptive-decision-log.v1',
                    'execution_status_policy': {
                        'failed': 'strong_solver_stress',
                        'unconverged': 'strong_solver_stress',
                        'blocked': 'not_correlation_evidence',
                    },
                },
            ),
            _capability(
                'continuation_restart',
                'Bidirectional continuation-window retry',
                'study.workflow_option',
                metadata={
                    'workflow_schema': 'pyscf-agent.path-window-restart.v1',
                    'activation': 'explicit_analyze_results',
                    'adaptive_execution_policy': 'independent_cases_no_cross_case_1rdm_reuse',
                    'initial_state_mode': 'projected_1rdm',
                    'scope': 'molecular_ao_bidirectional_adjacent_task_continuation',
                    'propagation_policy': {
                        'left_branch': 'ascending_coordinate_previous_case_1rdm',
                        'right_branch': 'descending_coordinate_previous_case_1rdm',
                        'endpoint_seed': 'original_endpoint_one_particle_state',
                        'target_method': 'preserved',
                    },
                    'source_policy': {
                        'same_method_and_reference': 'automatic_after_result_analysis',
                        'cross_method_or_reference': 'explicit_path_restart_approval',
                        'one_sided_endpoint_risk': 'explicit_path_restart_approval',
                    },
                    'approval_schema': 'pyscf-agent.path-restart-approval.v1',
                    'approval_evidence': [
                        'branch_direction',
                        'source_case_and_method',
                        'source_mode',
                        'target_case_and_method',
                        'one_particle_state_artifact',
                        'registered_artifact_reference',
                        'deferred_adjacent_case_dependency',
                    ],
                    'acceptance_checks': [
                        'both_succeeded',
                        'final_energy_agreement',
                        'reference_energy_agreement',
                        'spin_square_agreement',
                        'scf_stability_agreement',
                    ],
                    'fallback': 'path_window_active_space_review',
                },
            ),
            _capability(
                'merged_report_postprocessing',
                'Merged-report postprocessing',
                'study.workflow_option',
                metadata={
                    'requires_full_report_merge': True,
                    'plot_data_artifacts': True,
                },
            ),
        ],
    }


__all__ = ['build_study_families']
