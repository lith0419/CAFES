from __future__ import annotations

from typing import Iterable, List

from .contracts import ProviderBinding, ProviderContract, WorkflowTemplateContract


OPTION_SET_IDS = {
    'molecular_basis_sets': 'options.molecular.basis',
    'molecular_auxbasis_sets': 'options.molecular.auxbasis',
    'molecular_xc_functionals': 'options.molecular.xc',
    'periodic_basis_sets': 'options.periodic.basis',
    'periodic_pseudopotentials': 'options.periodic.pseudopotential',
    'periodic_xc_functionals': 'options.periodic.xc',
    'periodic_density_fitting_methods': 'options.periodic.density_fitting',
    'periodic_kpoint_schemes': 'options.periodic.kpoint_scheme',
    'periodic_band_path_modes': 'options.periodic.band_path_mode',
    'periodic_smearing_methods': 'options.periodic.smearing',
    'periodic_exxdiv_options': 'options.periodic.exxdiv',
    'model_hamiltonian_builder_templates': 'options.model_hamiltonian.builder_template',
}


def task_module_entry_id(module_id: str) -> str:
    return 'task.module.' + module_id


def study_module_entry_id(module_id: str) -> str:
    return 'study.module.' + module_id


def task_gate_entry_id(gate_id: str) -> str:
    return 'task.gate.' + gate_id


def study_gate_entry_id(gate_id: str) -> str:
    return 'study.gate.' + gate_id


def build_default_provider_contracts() -> List[ProviderContract]:
    return [
        ProviderContract(
            provider_id='provider.pyscf_agent',
            label='PySCF Agent runtime',
            description='Owns orchestration, validation, diagnostics, reporting, and integrated study stages.',
            aliases=('pyscf_agent', 'internal'),
        ),
        ProviderContract(
            provider_id='provider.pyscf',
            label='PySCF',
            optional_dependency='pyscf',
            description='Provides molecular, periodic, and finite-Hamiltonian numerical kernels.',
            aliases=('pyscf',),
        ),
        ProviderContract(
            provider_id='provider.block2',
            label='block2',
            optional_dependency='block2',
            description='Provides DMRG solvers, MPS continuation, and entanglement analysis.',
            aliases=('block2',),
        ),
        ProviderContract(
            provider_id='provider.libdmet',
            label='libDMET',
            optional_dependency='libdmet',
            description='Provides embedding transforms and the registered Hubbard DMET self-consistency runtime.',
            aliases=('libdmet', 'libdmet_embedding'),
        ),
        ProviderContract(
            provider_id='provider.fcdmft',
            label='fcDMFT',
            optional_dependency='fcdmft',
            description=(
                'Provides periodic GW, local GW double counting, and shared '
                'HF+DMFT and GW+DMFT impurity self-energy iteration.'
            ),
            aliases=('fcdmft', 'fc_dmft'),
        ),
    ]


TASK_PROVIDER_BINDINGS = {
    'core.input_generation': ('provider.pyscf_agent', 'pyscf_agent.backend.execution.input_generator'),
    'core.execution': ('provider.pyscf', 'pyscf_agent.backend.execution.runner'),
    'embedding.libdmet.reference_density.pm': (
        'provider.libdmet', 'pyscf_agent.providers.libdmet.reference_density'
    ),
    'embedding.libdmet.reference_density.af': (
        'provider.libdmet', 'pyscf_agent.providers.libdmet.reference_density'
    ),
    'embedding.libdmet.reference_density.fm': (
        'provider.libdmet', 'pyscf_agent.providers.libdmet.reference_density'
    ),
    'embedding.libdmet.reference_density.cdw': (
        'provider.libdmet', 'pyscf_agent.providers.libdmet.reference_density'
    ),
    'embedding.libdmet.dmet': ('provider.libdmet', 'pyscf_agent.providers.libdmet.dmet'),
    'embedding.fcdmft.prepare': (
        'provider.fcdmft', 'pyscf_agent.providers.fcdmft.prepare_hf_dmft'
    ),
    'embedding.fcdmft.prepare_subspace': (
        'provider.fcdmft', 'pyscf_agent.providers.fcdmft.prepare_hf_dmft_subspace'
    ),
    'embedding.fcdmft.hf_dmft': (
        'provider.fcdmft', 'pyscf_agent.providers.fcdmft.execute_hf_dmft'
    ),
    'embedding.fcdmft.periodic_gw': (
        'provider.fcdmft', 'pyscf_agent.providers.fcdmft.execute_periodic_gw'
    ),
    'embedding.fcdmft.gw_double_counting': (
        'provider.fcdmft', 'pyscf_agent.providers.fcdmft.execute_local_gw_double_counting'
    ),
    'embedding.fcdmft.gw_dmft': (
        'provider.fcdmft', 'pyscf_agent.providers.fcdmft.execute_gw_dmft'
    ),
    'solver.block2.dmrg': ('provider.block2', 'pyscf_agent.providers.block2.prepare'),
    'solver.block2.bond_dimension_planning': (
        'provider.block2', 'pyscf_agent.providers.block2.bond_dimension_planning'
    ),
    'solver.block2.entanglement_diagnostics': (
        'provider.block2', 'pyscf_agent.providers.block2.entanglement'
    ),
    'solver.block2.excited_states': (
        'provider.block2', 'pyscf_agent.providers.block2.excited_states'
    ),
    'solver.block2.mps_continuation': (
        'provider.block2', 'pyscf_agent.providers.block2.mps_continuation'
    ),
    'solver.block2.symmetry_analysis': (
        'provider.block2', 'pyscf_agent.providers.block2.symmetry_analysis'
    ),
    'core.repair_retry': ('provider.pyscf_agent', 'pyscf_agent.backend.execution.repair_or_retry'),
    'core.result_extraction': ('provider.pyscf_agent', 'pyscf_agent.backend.execution.result_extractor'),
    'core.result_analysis': ('provider.pyscf_agent', 'pyscf_agent.backend.execution.result_analyst'),
    'molecular.correlation_diagnostics': (
        'provider.pyscf_agent', 'pyscf_agent.backend.correlation_diagnostics'
    ),
    'molecular.active_space_probe': (
        'provider.pyscf_agent', 'pyscf_agent.backend.active_space.probe'
    ),
    'molecular.active_space_probe_refinement': (
        'provider.pyscf_agent', 'pyscf_agent.backend.active_space.probe_refinement'
    ),
    'molecular.orbital_processing': (
        'provider.pyscf_agent', 'pyscf_agent.backend.orbital_processing'
    ),
    'molecular.active_space_audit': (
        'provider.pyscf_agent', 'pyscf_agent.backend.active_space.audit'
    ),
    'model.correlation_diagnostics': (
        'provider.pyscf_agent', 'pyscf_agent.backend.model_hamiltonian.diagnostics'
    ),
    'periodic.band_analysis': (
        'provider.pyscf_agent', 'pyscf_agent.backend.periodic.band_analysis'
    ),
    'core.task_report': ('provider.pyscf_agent', 'pyscf_agent.backend.execution.task_reporter'),
}

STUDY_PROVIDER_BINDINGS = {
    'core.study_plan': ('provider.pyscf_agent', 'computational_study_agent.planner.build_study_plan'),
    'study.static_execution': ('provider.pyscf_agent', 'computational_study_agent.executor.run_study'),
    'study.initial_scan': ('provider.pyscf_agent', 'computational_study_agent.adaptive.initial_scan'),
    'study.correlation_routing': (
        'provider.pyscf_agent', 'computational_study_agent.adaptive.correlation_routing'
    ),
    'study.method_refinement': (
        'provider.pyscf_agent', 'computational_study_agent.adaptive.method_refinement'
    ),
    'study.adaptive_execution': (
        'provider.pyscf_agent', 'computational_study_agent.adaptive.refined_execution'
    ),
    'study.recovery': ('provider.pyscf_agent', 'computational_study_agent.adaptive.recovery'),
    'study.path_continuity': (
        'provider.pyscf_agent', 'computational_study_agent.adaptive.path_continuity'
    ),
    'analysis.block2.state_tracking': (
        'provider.pyscf_agent', 'computational_study_agent.adaptive.state_tracking'
    ),
    'study.result_integration': (
        'provider.pyscf_agent', 'computational_study_agent.results.integrate'
    ),
    'study.postprocessing': (
        'provider.pyscf_agent', 'computational_study_agent.postprocessing.run_postprocessing'
    ),
    'study.hamiltonian_dataset_assembly': (
        'provider.pyscf_agent',
        'computational_study_agent.datasets.hamiltonian.finalize.finalize_hamiltonian_dataset',
    ),
    'core.study_report': ('provider.pyscf_agent', 'computational_study_agent.results.study_report'),
}


def build_default_provider_bindings(
    task_modules: Iterable[object],
    study_modules: Iterable[object],
) -> List[ProviderBinding]:
    bindings: List[ProviderBinding] = []
    for scope, modules, entry_id, specifications in (
        ('task', task_modules, task_module_entry_id, TASK_PROVIDER_BINDINGS),
        ('study', study_modules, study_module_entry_id, STUDY_PROVIDER_BINDINGS),
    ):
        registered_ids = {str(getattr(module, 'module_id')) for module in modules}
        missing = sorted(registered_ids.difference(specifications))
        unknown = sorted(set(specifications).difference(registered_ids))
        if missing or unknown:
            raise ValueError(
                '{0} provider bindings do not match module contracts; missing={1}, unknown={2}.'.format(
                    scope, missing, unknown
                )
            )
        for module in modules:
            module_id = str(getattr(module, 'module_id'))
            module_entry_id = entry_id(module_id)
            provider_id, runtime_id = specifications[module_id]
            bindings.append(ProviderBinding(
                binding_id='{0}.binding.{1}'.format(scope, module_id),
                module_id=module_entry_id,
                provider_id=provider_id,
                runtime_id=runtime_id,
                description='Runtime binding for {0}.'.format(module_id),
            ))
    return bindings


def build_default_workflow_templates() -> List[WorkflowTemplateContract]:
    return [
        WorkflowTemplateContract(
            template_id='task.template.default',
            label='Default calculation task',
            scope='task',
            required_module_ids=tuple(task_module_entry_id(module_id) for module_id in (
                'core.input_generation',
                'core.execution',
                'core.result_extraction',
                'core.result_analysis',
                'core.task_report',
            )),
            optional_module_ids=tuple(task_module_entry_id(module_id) for module_id in (
                'core.repair_retry',
                'embedding.libdmet.reference_density.pm',
                'embedding.libdmet.reference_density.af',
                'embedding.libdmet.reference_density.fm',
                'embedding.libdmet.reference_density.cdw',
                'molecular.correlation_diagnostics',
                'molecular.active_space_probe',
                'molecular.active_space_probe_refinement',
                'molecular.orbital_processing',
                'molecular.active_space_audit',
                'model.correlation_diagnostics',
                'periodic.band_analysis',
                'embedding.libdmet.dmet',
                'embedding.fcdmft.prepare',
                'embedding.fcdmft.periodic_gw',
                'embedding.fcdmft.prepare_subspace',
                'embedding.fcdmft.gw_double_counting',
                'embedding.fcdmft.hf_dmft',
                'embedding.fcdmft.gw_dmft',
                'solver.block2.dmrg',
                'solver.block2.bond_dimension_planning',
                'solver.block2.entanglement_diagnostics',
                'solver.block2.excited_states',
                'solver.block2.mps_continuation',
                'solver.block2.symmetry_analysis',
            )),
            entry_capability_ids=(
                'task_type.molecular',
                'task_type.periodic',
                'task_type.model_hamiltonian',
            ),
            description='Compile, execute, diagnose, and report one validated TaskSpec.',
        ),
        WorkflowTemplateContract(
            template_id='molecular.template.active_space',
            label='Molecular active-space workflow',
            scope='task',
            required_module_ids=tuple(task_module_entry_id(module_id) for module_id in (
                'molecular.active_space_probe',
                'molecular.active_space_probe_refinement',
                'molecular.orbital_processing',
                'molecular.active_space_audit',
            )),
            description='Probe, process, audit, and approve a molecular active space before execution.',
        ),
        WorkflowTemplateContract(
            template_id='solver.block2.template.dmrg',
            label='block2 DMRG workflow',
            scope='task',
            required_module_ids=(task_module_entry_id('solver.block2.dmrg'),),
            optional_module_ids=tuple(task_module_entry_id(module_id) for module_id in (
                'solver.block2.bond_dimension_planning',
                'solver.block2.entanglement_diagnostics',
                'solver.block2.excited_states',
                'solver.block2.mps_continuation',
                'solver.block2.symmetry_analysis',
            )),
            description='Configure block2 and attach optional DMRG analysis and continuation modules.',
        ),
        WorkflowTemplateContract(
            template_id='embedding.template.dmet',
            label='libDMET Hubbard workflow',
            scope='task',
            required_module_ids=(task_module_entry_id('embedding.libdmet.dmet'),),
            optional_module_ids=(task_module_entry_id('model.correlation_diagnostics'),),
            entry_capability_ids=(
                'embedding.method.dmet',
                'model_hamiltonian.solver.dmet',
            ),
            description=(
                'Run the bounded libDMET mean-field, bath, impurity, '
                'correlation-potential, and self-consistency workflow.'
            ),
        ),
        WorkflowTemplateContract(
            template_id='embedding.template.hf_dmft',
            label='fcDMFT HF+DMFT workflow',
            scope='task',
            required_module_ids=tuple(task_module_entry_id(module_id) for module_id in (
                'embedding.fcdmft.prepare',
                'embedding.fcdmft.prepare_subspace',
                'embedding.fcdmft.hf_dmft',
            )),
            entry_capability_ids=('embedding.method.hf_dmft',),
            description=(
                'Run periodic HF, prepare reviewable localized artifacts when needed, '
                'and execute fcDMFT only after correlated-subspace approval.'
            ),
        ),
        WorkflowTemplateContract(
            template_id='embedding.template.gw_dmft',
            label='fcDMFT GW+DMFT workflow',
            scope='task',
            required_module_ids=tuple(task_module_entry_id(module_id) for module_id in (
                'embedding.fcdmft.prepare',
                'embedding.fcdmft.periodic_gw',
                'embedding.fcdmft.prepare_subspace',
                'embedding.fcdmft.gw_double_counting',
                'embedding.fcdmft.gw_dmft',
            )),
            entry_capability_ids=('embedding.method.gw_dmft',),
            description=(
                'Run periodic DFT and GW, approve a localized correlated subspace, '
                'build local GW double counting, and execute GW+DMFT.'
            ),
        ),
        WorkflowTemplateContract(
            template_id='study.template.static',
            label='Static computational study',
            scope='study',
            required_module_ids=tuple(study_module_entry_id(module_id) for module_id in (
                'core.study_plan',
                'study.static_execution',
                'study.result_integration',
                'study.postprocessing',
                'core.study_report',
            )),
            optional_module_ids=(
                study_module_entry_id('study.hamiltonian_dataset_assembly'),
            ),
            description='Execute and integrate a fixed-method collection of related cases.',
        ),
        WorkflowTemplateContract(
            template_id='study.template.adaptive',
            label='Adaptive computational study',
            scope='study',
            required_module_ids=tuple(study_module_entry_id(module_id) for module_id in (
                'core.study_plan',
                'study.initial_scan',
                'study.correlation_routing',
                'study.method_refinement',
                'study.adaptive_execution',
                'study.recovery',
                'study.path_continuity',
                'study.result_integration',
                'study.postprocessing',
                'core.study_report',
            )),
            optional_module_ids=(study_module_entry_id('analysis.block2.state_tracking'),),
            description='Probe, route, refine, recover, and review related calculations as one study.',
        ),
    ]


__all__ = [
    'OPTION_SET_IDS',
    'STUDY_PROVIDER_BINDINGS',
    'TASK_PROVIDER_BINDINGS',
    'build_default_provider_bindings',
    'build_default_provider_contracts',
    'build_default_workflow_templates',
    'study_gate_entry_id',
    'study_module_entry_id',
    'task_gate_entry_id',
    'task_module_entry_id',
]
