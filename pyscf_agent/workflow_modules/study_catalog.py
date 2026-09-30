from __future__ import annotations

from pyscf_agent.schema_contracts import STUDY_PLAN_SCHEMA, STUDY_REPORT_SCHEMA, STUDY_SPEC_SCHEMA

from typing import List

from .contracts import ActivationRule, DataPortContract, ModuleContract


def _port(name: str, schema: str = '', *, optional: bool = False) -> DataPortContract:
    return DataPortContract(name=name, schema=schema, optional=optional)


def build_default_study_module_contracts() -> List[ModuleContract]:
    """Describe reusable study stages; quality gates are compiled separately."""

    return [
        ModuleContract(
            module_id='core.study_plan',
            version='1.0',
            permitted_stages=('study.prepare',),
            default_stage='study.prepare',
            required_inputs=(_port('study.spec', STUDY_SPEC_SCHEMA),),
            provided_outputs=(_port('study.plan', STUDY_PLAN_SCHEMA),),
            capability_ids=(
                'study.mode.static',
                'study.mode.adaptive',
                'study.workflow_option.resource_cost_review',
                'model_hamiltonian.operation.set_global_parameter',
                'model_hamiltonian.operation.set_site_parameter',
                'model_hamiltonian.operation.shift_site_parameter',
                'model_hamiltonian.operation.scale_site_parameter',
                'model_hamiltonian.operation.add_site_defect',
                'model_hamiltonian.operation.set_bond_parameter',
                'model_hamiltonian.operation.shift_bond_parameter',
                'model_hamiltonian.operation.scale_bond_parameter',
                'model_hamiltonian.operation.add_bond_defect',
                'model_hamiltonian.operation.change_nelec',
                'model_hamiltonian.operation.change_boundary',
                'model_hamiltonian.operation.change_solver',
            ),
            always_select=True,
            description='Expand a StudySpec into validated, reviewable calculation cases.',
        ),
        ModuleContract(
            module_id='study.static_execution',
            version='1.0',
            permitted_stages=('study.execute',),
            default_stage='study.execute',
            required_inputs=(_port('study.plan'),),
            provided_outputs=(_port('study.final_case_reports'),),
            capability_ids=(
                'study.policy.grid_refinement',
                'study.workflow_option.task_job_lifecycle',
                'study.workflow_option.slurm_independent_task_batches',
            ),
            activation_rules=(ActivationRule('study_mode', 'equals', 'static'),),
            description='Execute fixed-method cases, optionally adding model-parameter grid points within a bounded sampling policy.',
        ),
        ModuleContract(
            module_id='study.initial_scan',
            version='1.0',
            permitted_stages=('study.probe',),
            default_stage='study.probe',
            required_inputs=(_port('study.plan'),),
            provided_outputs=(_port('study.initial_results'),),
            capability_ids=(
                'study.adaptive_initial_scan.auto',
                'study.adaptive_initial_scan.mp2',
                'study.adaptive_initial_scan.fci',
            ),
            activation_rules=(ActivationRule('study_mode', 'equals', 'adaptive'),),
            description='Run low-cost diagnostic calculations for every planned case.',
        ),
        ModuleContract(
            module_id='study.correlation_routing',
            version='1.0',
            permitted_stages=('study.diagnose',),
            default_stage='study.diagnose',
            required_inputs=(_port('study.initial_results'),),
            provided_outputs=(_port('study.correlation_decisions'),),
            capability_ids=('study.policy.adaptive_routing',),
            activation_rules=(ActivationRule('study_mode', 'equals', 'adaptive'),),
            description='Classify correlation regimes and propose a method for each case.',
        ),
        ModuleContract(
            module_id='study.method_refinement',
            version='1.0',
            permitted_stages=('study.configure',),
            default_stage='study.configure',
            required_inputs=(
                _port('study.plan'),
                _port('study.correlation_decisions'),
            ),
            provided_outputs=(_port('study.refined_plan'),),
            activation_rules=(ActivationRule('study_mode', 'equals', 'adaptive'),),
            description='Build the refined per-case method and active-space configuration.',
        ),
        ModuleContract(
            module_id='study.adaptive_execution',
            version='1.0',
            permitted_stages=('study.execute',),
            default_stage='study.execute',
            required_inputs=(_port('study.refined_plan'),),
            provided_outputs=(_port('study.refined_case_reports'),),
            activation_rules=(ActivationRule('study_mode', 'equals', 'adaptive'),),
            description='Execute the method-routed refined cases.',
        ),
        ModuleContract(
            module_id='study.recovery',
            version='1.0',
            permitted_stages=('study.review',),
            default_stage='study.review',
            required_inputs=(
                _port('study.refined_plan'),
                _port('study.refined_case_reports'),
            ),
            provided_outputs=(_port('study.final_case_reports'),),
            capability_ids=(
                'study.workflow_option.recoverable_study_execution',
                'study.workflow_option.checkpoint_resume',
                'study.workflow_option.case_local_recovery',
            ),
            activation_rules=(ActivationRule('study_mode', 'equals', 'adaptive'),),
            description='Apply bounded case-specific recovery and preserve review gates.',
        ),
        ModuleContract(
            module_id='study.path_continuity',
            version='1.0',
            permitted_stages=('study.review',),
            default_stage='study.review',
            required_inputs=(_port('study.final_case_reports'),),
            provided_outputs=(_port('study.path_review'),),
            capability_ids=(
                'study.workflow_option.continuation_restart',
                'study.policy.scan_path_continuity',
            ),
            description='Detect cross-case discontinuities when result analysis is requested.',
        ),
        ModuleContract(
            module_id='analysis.block2.state_tracking',
            version='1.0',
            permitted_stages=('study.review',),
            default_stage='study.review',
            required_inputs=(_port('study.final_case_reports'),),
            provided_outputs=(
                _port('study.dmrg_state_tracking', 'pyscf-agent.dmrg-state-tracking.v1'),
            ),
            optional_configuration={
                'type': 'object',
                'properties': {
                    'enabled': {'type': 'boolean'},
                },
                'additionalProperties': False,
            },
            default_configuration={'enabled': True},
            capability_ids=('study.workflow_option.block2_state_tracking',),
            description='Track low-energy DMRG root identities across related study cases.',
        ),
        ModuleContract(
            module_id='study.result_integration',
            version='1.0',
            permitted_stages=('study.finalize',),
            default_stage='study.finalize',
            required_inputs=(_port('study.final_case_reports'),),
            provided_outputs=(_port('study.integrated_results'),),
            capability_ids=('study.workflow_option.merged_report_postprocessing',),
            always_select=True,
            description='Merge TaskReports into one ordered study result table.',
        ),
        ModuleContract(
            module_id='study.postprocessing',
            version='1.0',
            permitted_stages=('study.finalize',),
            default_stage='study.finalize',
            required_inputs=(_port('study.integrated_results'),),
            provided_outputs=(_port('study.postprocessing_artifacts'),),
            capability_ids=(
                'postprocessing.tool.line_plot',
                'postprocessing.tool.scatter_plot',
                'postprocessing.tool.bar_plot',
                'postprocessing.tool.heatmap',
                'postprocessing.action.generate_hamiltonian_dataset',
                'postprocessing.action.collect_hamiltonian_dataset',
                'postprocessing.action.select_energy_stratified_frames',
            ),
            always_select=True,
            description='Generate registered tables, plots, datasets, and trajectory frame selections.',
        ),
        ModuleContract(
            module_id='study.hamiltonian_dataset_assembly',
            version='1.0',
            permitted_stages=('study.finalize',),
            default_stage='study.finalize',
            required_inputs=(_port('study.final_case_reports'),),
            provided_outputs=(
                _port(
                    'study.hamiltonian_dataset_manifest',
                    'pyscf-agent.hamiltonian-dataset-manifest.v1',
                ),
            ),
            activation_rules=(
                ActivationRule(
                    'comparison.mode',
                    'equals',
                    'hamiltonian_dataset_assembly',
                ),
            ),
            before=('core.study_report',),
            description=(
                'Assemble self-describing MD frame metadata into accepted and '
                'rejected Hamiltonian dataset indexes.'
            ),
        ),
        ModuleContract(
            module_id='core.study_report',
            version='1.0',
            permitted_stages=('study.finalize',),
            default_stage='study.finalize',
            required_inputs=(
                _port('study.integrated_results'),
                _port('study.postprocessing_artifacts'),
                _port('study.hamiltonian_dataset_manifest', optional=True),
            ),
            provided_outputs=(_port('report.study', STUDY_REPORT_SCHEMA),),
            always_select=True,
            after=('study.result_integration', 'study.postprocessing'),
            description='Assemble the final StudyReport and complete provenance.',
        ),
    ]
