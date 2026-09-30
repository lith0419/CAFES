from __future__ import annotations

from typing import Any, Callable, Dict, List


def build_postprocessing_families(
    item_factory: Callable[..., Any],
    *,
    capability_design_only: str,
    capability_planned: str,
) -> Dict[str, List[Any]]:
    _capability = item_factory
    CAPABILITY_DESIGN_ONLY = capability_design_only
    CAPABILITY_PLANNED = capability_planned

    return {
        'postprocessing_metrics': [
            _capability(
                'mean_double_occupancy',
                'Mean double occupancy',
                'postprocessing.metric',
                planner_allowed=False,
                description='Site-averaged local double occupancy derived from an executed model-Hamiltonian strong-correlation diagnostic.',
                metadata={
                    'system_types': ['model_hamiltonian'],
                    'study_request_output': 'strong_correlation_diagnostics',
                    'result_field': 'strong_correlation_diagnostics',
                    'comparison_summary': 'mean_double_occupancy',
                    'diagnostic_name': 'double_occupancy_suppression',
                    'diagnostic_key': 'mean_double_occupancy',
                    'derived_from': ['double_occupancy'],
                    'result_shape': 'scalar',
                    'unit': 'dimensionless',
                    'plot_priority': 2,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                    'generated_output_rule': (
                        'Emit only when executed strong-correlation diagnostics provide double occupancy; '
                        'keep the per-site vector in the diagnostic payload rather than the comparison table.'
                    ),
                },
            ),
            _capability(
                'nearest_neighbor_spin_correlation',
                'Nearest-neighbor spin correlation',
                'postprocessing.metric',
                planner_allowed=False,
                description='Mean nearest-neighbor S_i dot S_j correlation derived from an executed model-Hamiltonian strong-correlation diagnostic.',
                metadata={
                    'system_types': ['model_hamiltonian'],
                    'result_field': 'strong_correlation_diagnostics',
                    'comparison_summary': 'nearest_neighbor_spin_correlation',
                    'diagnostic_name': 'nearest_neighbor_spin_correlation',
                    'diagnostic_key': 'mean',
                    'derived_from': ['spin_correlation'],
                    'result_shape': 'scalar',
                    'unit': 'dimensionless',
                    'plot_priority': 3,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                    'generated_output_rule': (
                        'Emit only when executed strong-correlation diagnostics provide spin correlations; '
                        'keep the pair matrix in the diagnostic payload rather than the comparison table.'
                    ),
                },
            ),
            _capability(
                'nearest_neighbor_charge_correlation',
                'Nearest-neighbor charge correlation',
                'postprocessing.metric',
                planner_allowed=False,
                description='Mean connected nearest-neighbor charge correlation derived from an executed model-Hamiltonian strong-correlation diagnostic.',
                metadata={
                    'system_types': ['model_hamiltonian'],
                    'result_field': 'strong_correlation_diagnostics',
                    'comparison_summary': 'nearest_neighbor_charge_correlation',
                    'diagnostic_name': 'nearest_neighbor_charge_correlation',
                    'diagnostic_key': 'mean',
                    'derived_from': ['charge_correlation'],
                    'result_shape': 'scalar',
                    'unit': 'dimensionless',
                    'plot_priority': 4,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                    'generated_output_rule': (
                        'Emit only when executed strong-correlation diagnostics provide connected charge correlations; '
                        'keep the pair matrix in the diagnostic payload rather than the comparison table.'
                    ),
                },
            ),
            _capability(
                'sublattice_charge_imbalance',
                'Sublattice charge imbalance',
                'postprocessing.metric',
                planner_allowed=False,
                description='Difference in mean site density between labeled model sublattices.',
                metadata={
                    'system_types': ['model_hamiltonian'],
                    'result_field': 'strong_correlation_diagnostics',
                    'comparison_summary': 'sublattice_charge_imbalance',
                    'diagnostic_name': 'sublattice_order_parameters',
                    'diagnostic_key': 'charge_imbalance',
                    'derived_from': ['density', 'site_sublattice_labels'],
                    'result_shape': 'scalar',
                    'unit': 'dimensionless',
                    'plot_priority': 5,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                },
            ),
            _capability(
                'staggered_magnetization',
                'Staggered magnetization',
                'postprocessing.metric',
                planner_allowed=False,
                description='Difference in mean local spin polarization between labeled model sublattices.',
                metadata={
                    'system_types': ['model_hamiltonian'],
                    'result_field': 'strong_correlation_diagnostics',
                    'comparison_summary': 'staggered_magnetization',
                    'diagnostic_name': 'sublattice_order_parameters',
                    'diagnostic_key': 'staggered_magnetization',
                    'derived_from': ['local_magnetization', 'site_sublattice_labels'],
                    'result_shape': 'scalar',
                    'unit': 'dimensionless',
                    'plot_priority': 6,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                },
            ),
            _capability(
                'mean_nearest_neighbor_one_body_coherence',
                'Nearest-neighbor one-body coherence',
                'postprocessing.metric',
                planner_allowed=False,
                description='Mean absolute nearest-neighbor off-diagonal element of the assembled one-particle density matrix.',
                metadata={
                    'system_types': ['model_hamiltonian'],
                    'result_field': 'strong_correlation_diagnostics',
                    'comparison_summary': 'mean_nearest_neighbor_one_body_coherence',
                    'diagnostic_name': 'nearest_neighbor_one_body_coherence',
                    'diagnostic_key': 'mean',
                    'derived_from': ['one_particle_density_matrix'],
                    'result_shape': 'scalar',
                    'unit': 'dimensionless',
                    'plot_priority': 7,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                },
            ),
            _capability(
                'energy_over_abs_t',
                'Energy / |t|',
                'postprocessing.metric',
                planner_allowed=False,
                metadata={
                    'system_types': ['model_hamiltonian'],
                    'result_field': 'energy_over_abs_t',
                    'comparison_summary': 'energy_over_abs_t',
                    'derived_from': ['energy', 'mean_absolute_hopping'],
                    'result_shape': 'scalar',
                    'unit': 'dimensionless',
                    'plot_priority': 8,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                },
            ),
            _capability(
                'energy_per_site_over_abs_t',
                'Energy per site / |t|',
                'postprocessing.metric',
                planner_allowed=False,
                metadata={
                    'system_types': ['model_hamiltonian'],
                    'result_field': 'energy_per_site_over_abs_t',
                    'comparison_summary': 'energy_per_site_over_abs_t',
                    'derived_from': ['energy_per_site', 'mean_absolute_hopping'],
                    'result_shape': 'scalar',
                    'unit': 'dimensionless',
                    'plot_priority': 9,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                },
            ),
            _capability(
                'first_excitation_energy',
                'First DMRG excitation energy',
                'postprocessing.metric',
                planner_allowed=False,
                metadata={
                    'system_types': ['molecular', 'model_hamiltonian'],
                    'result_field': 'dmrg_result',
                    'comparison_summary': 'first_excitation_energy',
                    'derived_from': ['excitation_energies'],
                    'result_shape': 'scalar',
                    'unit': 'energy',
                    'plot_priority': 12,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                },
            ),
            _capability(
                'max_single_orbital_entropy',
                'Maximum single-orbital entropy',
                'postprocessing.metric',
                planner_allowed=False,
                metadata={
                    'system_types': ['molecular', 'model_hamiltonian'],
                    'result_field': 'entanglement_diagnostics',
                    'comparison_summary': 'max_single_orbital_entropy',
                    'derived_from': ['single_orbital_entropy'],
                    'result_shape': 'scalar',
                    'unit': 'dimensionless',
                    'plot_priority': 20,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                },
            ),
            _capability(
                'mean_single_orbital_entropy',
                'Mean single-orbital entropy',
                'postprocessing.metric',
                planner_allowed=False,
                metadata={
                    'system_types': ['molecular', 'model_hamiltonian'],
                    'result_field': 'entanglement_diagnostics',
                    'comparison_summary': 'mean_single_orbital_entropy',
                    'derived_from': ['single_orbital_entropy'],
                    'result_shape': 'scalar',
                    'unit': 'dimensionless',
                    'plot_priority': 21,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                },
            ),
            _capability(
                'max_bipartite_entanglement',
                'Maximum bipartite entanglement',
                'postprocessing.metric',
                planner_allowed=False,
                metadata={
                    'system_types': ['molecular', 'model_hamiltonian'],
                    'result_field': 'entanglement_diagnostics',
                    'comparison_summary': 'max_bipartite_entanglement',
                    'derived_from': ['bipartite_entanglement'],
                    'result_shape': 'scalar',
                    'unit': 'dimensionless',
                    'plot_priority': 23,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                },
            ),
            _capability(
                'dmrg_spin_square',
                'DMRG spin-square expectation',
                'postprocessing.metric',
                planner_allowed=False,
                metadata={
                    'system_types': ['molecular', 'model_hamiltonian'],
                    'result_field': 'symmetry_analysis',
                    'comparison_summary': 'dmrg_spin_square',
                    'derived_from': ['spin_square'],
                    'result_shape': 'scalar',
                    'unit': 'dimensionless',
                    'plot_priority': 24,
                    'supports_plot': ['line_plot', 'scatter_plot', 'heatmap'],
                },
            ),
        ],
        'postprocessing_tools': [
            _capability(
                tool_id,
                label,
                'postprocessing.tool',
                metadata={
                    'artifact_kinds': [
                        'postprocess-plot',
                        'postprocess-plot-spec',
                        'postprocess-plot-data-tsv',
                        'postprocess-plot-data-json',
                    ],
                },
            )
            for tool_id, label in (
                ('line_plot', 'Line plot'),
                ('scatter_plot', 'Scatter plot'),
                ('bar_plot', 'Bar plot'),
                ('heatmap', 'Heatmap'),
            )
        ],
        'postprocessing_actions': [
            _capability(
                'select_energy_stratified_frames',
                'Select frames by energy quantiles',
                'postprocessing.action',
                description=(
                    'Select a requested number of geometries per molecule from an accepted '
                    'trajectory index: apply a time window, sort by potential energy, '
                    'divide into equal-population bins and take each bin median. '
                    'Exports coordinates and provenance without new calculations or matrix loading.'
                ),
                metadata={
                    'system_types': ['molecular'],
                    'requires_result_field': 'dataset_manifest',
                    'parameters': {
                        'count_per_molecule': {'type': 'integer', 'minimum': 1, 'required': True},
                        'time_start_fs': {'type': 'number', 'minimum': 0, 'default': 0},
                        'time_end_fs': {'type': ['number', 'null'], 'minimum': 0, 'default': None},
                    },
                    'artifact_kinds': ['trajectory_frame_selection', 'selected_trajectory_frames',
                                       'selected_trajectory_xyz'],
                },
            ),
            _capability(
                'generate_hamiltonian_dataset',
                'Generate Hamiltonian dataset',
                'postprocessing.action',
                description=(
                    'Materialize and validate a complete portable dataset on the '
                    'execution target without downloading trajectory arrays.'
                ),
                metadata={
                    'system_types': ['molecular'],
                    'requires_result_field': 'dataset_manifest',
                    'artifact_kinds': [
                        'hamiltonian_dataset_generation_receipt',
                    ],
                },
            ),
            _capability(
                'collect_hamiltonian_dataset',
                'Collect Hamiltonian dataset',
                'postprocessing.action',
                description=(
                    'Download a previously generated portable dataset from its '
                    'execution target into the local study directory.'
                ),
                metadata={
                    'system_types': ['molecular'],
                    'requires_result_field': 'dataset_manifest',
                    'requires_action': 'generate_hamiltonian_dataset',
                    'artifact_kinds': [
                        'hamiltonian_collected_dataset_manifest',
                        'hamiltonian_collected_sample_index',
                        'hamiltonian_collected_rejection_index',
                        'hamiltonian_collected_trajectory_arrays',
                    ],
                },
            ),
        ],
    }


__all__ = ['build_postprocessing_families']
