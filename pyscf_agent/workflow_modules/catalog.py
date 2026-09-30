from __future__ import annotations

from pyscf_agent.schema_contracts import TASK_REPORT_SCHEMA, TASK_SPEC_SCHEMA

from typing import List

from .contracts import ActivationRule, DataPortContract, ModuleCompatibility, ModuleContract


def _port(name: str, schema: str = '', *, merge_policy: str = 'single', optional: bool = False) -> DataPortContract:
    return DataPortContract(name=name, schema=schema, merge_policy=merge_policy, optional=optional)


def _dmet_reference_density_modules() -> List[ModuleContract]:
    labels = {
        'pm': 'paramagnetic',
        'af': 'antiferromagnetic',
        'fm': 'ferromagnetic',
        'cdw': 'charge-density-wave',
    }
    return [
        ModuleContract(
            module_id='embedding.libdmet.reference_density.{0}'.format(strategy),
            version='1.0',
            permitted_stages=('task.prepare',),
            default_stage='task.prepare',
            required_inputs=(_port('task.spec', TASK_SPEC_SCHEMA),),
            provided_outputs=(_port(
                'embedding.reference_density.ready',
                'pyscf-agent.reference-density-initialization.v1',
            ),),
            compatibility=ModuleCompatibility(
                task_types=('model_hamiltonian',),
                methods=('dmet',),
            ),
            capability_ids=(
                'embedding.reference_density.{0}'.format(strategy),
            ),
            exclusive_group='embedding.libdmet.reference_density',
            before=('embedding.libdmet.dmet', 'core.input_generation'),
            activation_rules=(
                ActivationRule('solver.name', 'equals', 'dmet'),
                ActivationRule(
                    'solver.options.reference_density_guess',
                    'missing_or_equals' if strategy == 'pm' else 'equals',
                    strategy,
                ),
            ),
            description=(
                'Prepare a {0} reference-density seed while leaving the auxiliary '
                'DMET correlation potential at zero.'
            ).format(labels[strategy]),
        )
        for strategy in ('pm', 'af', 'fm', 'cdw')
    ]


def build_default_module_contracts() -> List[ModuleContract]:
    correlated_methods = ('mp2', 'ccsd', 'ccsd_t', 'casci', 'casscf', 'fci')
    molecular_methods = ('hf', 'dft', 'mp2', 'ccsd', 'ccsd_t', 'fci', 'casci', 'casscf')
    return [
        ModuleContract(
            module_id='core.input_generation',
            version='1.0',
            permitted_stages=('task.prepare',),
            default_stage='task.prepare',
            required_inputs=(_port('task.spec', TASK_SPEC_SCHEMA),),
            provided_outputs=(_port('task.generated_input'),),
            capability_ids=(
                'task_type.molecular',
                'task_type.model_hamiltonian',
                'task_type.periodic',
                'model_hamiltonian.model.hubbard',
            ),
            always_select=True,
            description='Generate a reproducible input program from the validated TaskSpec.',
        ),
        ModuleContract(
            module_id='core.execution',
            version='1.0',
            optional_configuration={
                'type': 'object',
                'properties': {'label_provenance': {'type': 'object'}},
                'additionalProperties': False,
            },
            permitted_stages=('task.execute',),
            default_stage='task.execute',
            required_inputs=(
                _port('task.generated_input'),
                _port('solver.provider.ready', optional=True),
            ),
            provided_outputs=(_port('execution.raw'),),
            capability_ids=(
                'molecular.job.single_point',
                'molecular.job.molecular_dynamics',
                'molecular.method.hf',
                'molecular.method.dft',
                'molecular.method.mp2',
                'molecular.method.ccsd',
                'molecular.method.ccsd_t',
                'molecular.method.fci',
                'molecular.method.casci',
                'molecular.method.casscf',
                'molecular.active_space_solver.fci',
                'molecular.workflow_option.density_fitting',
                'molecular.workflow_option.sc_nevpt2',
                'periodic.job.single_point',
                'periodic.method.hf',
                'periodic.method.dft',
                'model_hamiltonian.solver.mp2',
                'model_hamiltonian.solver.ccsd',
                'model_hamiltonian.solver.ccsd_t',
                'model_hamiltonian.solver.fci',
                'model_hamiltonian.solver.tight_binding',
            ),
            always_select=True,
            description='Execute one molecular, periodic, or model-Hamiltonian task.',
        ),
        *_dmet_reference_density_modules(),
        ModuleContract(
            module_id='embedding.libdmet.dmet',
            version='1.0',
            permitted_stages=('task.prepare',),
            default_stage='task.prepare',
            required_inputs=(
                _port('task.spec', TASK_SPEC_SCHEMA),
                _port(
                    'embedding.reference_density.ready',
                    'pyscf-agent.reference-density-initialization.v1',
                ),
            ),
            provided_outputs=(_port('solver.provider.ready', 'pyscf-agent.solver-provider.v1'),),
            compatibility=ModuleCompatibility(
                task_types=('model_hamiltonian',),
                methods=('dmet',),
            ),
            capability_ids=(
                'embedding.backend.libdmet',
                'embedding.localization.manual',
                'embedding.localization.identity_sites',
                'embedding.localization.iao',
                'embedding.localization.iao_pao',
                'embedding.localization.lowdin',
                'embedding.operation.define_correlated_subspace',
                'embedding.operation.transform_one_particle_operators',
                'embedding.operation.serialize_embedding_artifacts',
                'embedding.operation.audit_builder_translation',
                'embedding.operation.build_dmet_bath',
                'embedding.operation.solve_dmet_impurity',
                'embedding.operation.fit_dmet_correlation_potential',
                'embedding.operation.iterate_dmet_self_consistency',
                'embedding.method.dmet',
                'embedding.impurity_solver.fci',
                'embedding.impurity_solver.ccsd',
                'embedding.impurity_solver.block2_dmrg',
                'model_hamiltonian.solver.dmet',
            ),
            before=('core.input_generation',),
            activation_rules=(ActivationRule('solver.name', 'equals', 'dmet'),),
            description=(
                'Validate solver.options and configure the composite libDMET Hubbard workflow: '
                'mean field, Schmidt bath, impurity solution, correlation-potential '
                'fit, and self-consistency.'
            ),
        ),
        ModuleContract(
            module_id='embedding.fcdmft.prepare',
            version='1.0',
            permitted_stages=('task.prepare',),
            default_stage='task.prepare',
            required_inputs=(_port('task.spec', TASK_SPEC_SCHEMA),),
            provided_outputs=(_port(
                'solver.provider.ready',
                'pyscf-agent.solver-provider.v1',
            ),),
            compatibility=ModuleCompatibility(
                task_types=('periodic',),
                methods=('hf', 'dft'),
            ),
            optional_configuration={
                'type': 'object',
                'properties': {
                    'impurity_solver': {'enum': ['cc', 'ucc', 'fci']},
                    'ncore': {'type': 'integer', 'minimum': 0},
                    'nval': {'type': 'integer', 'minimum': 1},
                    'nbath': {'type': 'integer', 'minimum': 1},
                    'nb_per_e': {'type': 'integer', 'minimum': 1},
                    'bath_discretization': {
                        'enum': ['opt', 'direct', 'linear', 'gauss', 'log'],
                    },
                    'max_iterations': {'type': 'integer', 'minimum': 1},
                    'convergence_tolerance': {
                        'type': 'number',
                        'exclusiveMinimum': 0,
                    },
                    'chemical_potential': {'type': 'number'},
                    'target_occupancy': {'type': 'number'},
                },
                'additionalProperties': True,
            },
            capability_ids=(
                'embedding.backend.fcdmft',
            ),
            before=('core.input_generation', 'core.execution'),
            activation_rules=(ActivationRule('solver.name', 'in', ('hf_dmft', 'gw', 'gw_dmft')),),
            description=(
                'Validate the fcDMFT installation and configure periodic GW, '
                'HF+DMFT, or GW+DMFT provider options.'
            ),
        ),
        ModuleContract(
            module_id='embedding.fcdmft.periodic_gw',
            version='1.0',
            permitted_stages=('task.execute',),
            default_stage='task.execute',
            required_inputs=(
                _port('execution.raw'),
                _port('solver.provider.ready', 'pyscf-agent.solver-provider.v1'),
            ),
            provided_outputs=(
                _port('execution.periodic_gw', 'pyscf-agent.periodic-gw-result.v1'),
                _port('artifacts.periodic_gw', 'pyscf-agent.artifact-reference-map.v1'),
            ),
            compatibility=ModuleCompatibility(
                task_types=('periodic',),
                methods=('dft',),
            ),
            capability_ids=('embedding.method.gw',),
            requires_modules=('embedding.fcdmft.prepare',),
            after=('core.execution',),
            before=(
                'embedding.fcdmft.prepare_subspace',
                'embedding.fcdmft.gw_double_counting',
            ),
            activation_rules=(ActivationRule('solver.name', 'in', ('gw', 'gw_dmft')),),
            description=(
                'Run or reuse the k-resolved fcDMFT periodic GW stage and register '
                'its quasiparticle and self-energy artifacts.'
            ),
        ),
        ModuleContract(
            module_id='embedding.fcdmft.prepare_subspace',
            version='1.0',
            permitted_stages=('task.execute',),
            default_stage='task.execute',
            required_inputs=(
                _port('execution.raw'),
                _port('solver.provider.ready', 'pyscf-agent.solver-provider.v1'),
            ),
            provided_outputs=(_port(
                'embedding.correlated_subspace.review',
                'pyscf-agent.correlated-subspace-audit.v1',
            ),),
            compatibility=ModuleCompatibility(
                task_types=('periodic',),
                methods=('hf', 'dft'),
            ),
            capability_ids=(
                'embedding.operation.build_interaction_tensor',
            ),
            requires_modules=('embedding.fcdmft.prepare',),
            after=('core.execution',),
            before=('embedding.fcdmft.hf_dmft', 'embedding.fcdmft.gw_double_counting'),
            activation_rules=(ActivationRule('solver.name', 'in', ('hf_dmft', 'gw_dmft')),),
            description=(
                'Generate IAO-localized periodic one- and two-particle artifacts '
                'and a correlated-subspace audit when approval is not yet present.'
            ),
        ),
        ModuleContract(
            module_id='embedding.fcdmft.gw_double_counting',
            version='1.0',
            permitted_stages=('task.execute',),
            default_stage='task.execute',
            required_inputs=(
                _port('execution.raw'),
                _port('execution.periodic_gw', 'pyscf-agent.periodic-gw-result.v1'),
                _port('embedding.correlated_subspace.review', 'pyscf-agent.correlated-subspace-audit.v1'),
            ),
            provided_outputs=(
                _port('execution.gw_double_counting', 'pyscf-agent.gw-double-counting-result.v1'),
                _port('artifacts.gw_double_counting', 'pyscf-agent.artifact-reference-map.v1'),
            ),
            compatibility=ModuleCompatibility(task_types=('periodic',), methods=('dft',)),
            capability_ids=('embedding.operation.gw_double_counting',),
            requires_modules=(
                'embedding.fcdmft.prepare',
                'embedding.fcdmft.periodic_gw',
                'embedding.fcdmft.prepare_subspace',
            ),
            after=('embedding.fcdmft.periodic_gw', 'embedding.fcdmft.prepare_subspace'),
            before=('embedding.fcdmft.gw_dmft',),
            activation_rules=(ActivationRule('solver.name', 'equals', 'gw_dmft'),),
            description=(
                'Build or reuse the local GW self-energy and orbital-transform '
                'artifacts required for explicit GW+DMFT double counting.'
            ),
        ),
        ModuleContract(
            module_id='embedding.fcdmft.hf_dmft',
            version='1.0',
            permitted_stages=('task.execute',),
            default_stage='task.execute',
            required_inputs=(
                _port('execution.raw'),
                _port('solver.provider.ready', 'pyscf-agent.solver-provider.v1'),
            ),
            provided_outputs=(_port(
                'execution.dmft',
                'pyscf-agent.hf-dmft-result.v1',
            ),),
            compatibility=ModuleCompatibility(
                task_types=('periodic',),
                methods=('hf',),
            ),
            capability_ids=(
                'embedding.operation.build_dmft_bath',
                'embedding.method.hf_dmft',
                'embedding.dmft_impurity_solver.cc',
                'embedding.dmft_impurity_solver.ucc',
                'embedding.dmft_impurity_solver.fci',
            ),
            requires_modules=(
                'embedding.fcdmft.prepare',
                'embedding.fcdmft.prepare_subspace',
            ),
            after=('embedding.fcdmft.prepare_subspace',),
            activation_rules=(ActivationRule('solver.name', 'equals', 'hf_dmft'),),
            description=(
                'Run fcDMFT HF+DMFT after periodic HF and finalize the task from '
                'DMFT convergence rather than the intermediate reference status.'
            ),
        ),
        ModuleContract(
            module_id='embedding.fcdmft.gw_dmft',
            version='1.0',
            permitted_stages=('task.execute',),
            default_stage='task.execute',
            required_inputs=(
                _port('execution.raw'),
                _port('execution.periodic_gw', 'pyscf-agent.periodic-gw-result.v1'),
                _port('execution.gw_double_counting', 'pyscf-agent.gw-double-counting-result.v1'),
            ),
            provided_outputs=(_port('execution.dmft', 'pyscf-agent.gw-dmft-result.v1'),),
            compatibility=ModuleCompatibility(task_types=('periodic',), methods=('dft',)),
            capability_ids=(
                'embedding.method.gw_dmft',
            ),
            requires_modules=(
                'embedding.fcdmft.prepare',
                'embedding.fcdmft.periodic_gw',
                'embedding.fcdmft.prepare_subspace',
                'embedding.fcdmft.gw_double_counting',
            ),
            after=('embedding.fcdmft.gw_double_counting',),
            activation_rules=(ActivationRule('solver.name', 'equals', 'gw_dmft'),),
            description=(
                'Run the terminal fcDMFT GW+DMFT loop from approved localized '
                'artifacts, lattice GW data, and explicit local GW double counting.'
            ),
        ),
        ModuleContract(
            module_id='solver.block2.dmrg',
            version='1.0',
            permitted_stages=('task.prepare',),
            default_stage='task.prepare',
            required_inputs=(_port('task.spec', TASK_SPEC_SCHEMA),),
            provided_outputs=(_port('solver.provider.ready', 'pyscf-agent.solver-provider.v1'),),
            compatibility=ModuleCompatibility(
                task_types=('molecular', 'model_hamiltonian'),
                methods=('casci', 'casscf', 'block2_dmrg', 'block2', 'block2-dmrg', 'dmrg'),
            ),
            optional_configuration={
                'type': 'object',
                'properties': {
                    'preset': {'enum': ['screening', 'balanced', 'high_accuracy']},
                    'symmetry': {'enum': ['auto', 'su2', 'sz']},
                    'mpo_algorithm': {'enum': ['fast_bipartite', 'conventional']},
                    'integral_cutoff': {'type': 'number', 'minimum': 0},
                    'orbital_ordering': {'enum': ['canonical', 'fiedler', 'manual']},
                    'orbital_order': {
                        'type': 'array',
                        'items': {'type': 'integer', 'minimum': 0},
                    },
                    'adaptive_noise': {'type': 'number', 'minimum': 0},
                },
                'additionalProperties': True,
            },
            capability_ids=(
                'molecular.active_space_solver.block2_dmrg',
                'model_hamiltonian.solver.block2_dmrg',
            ),
            before=('core.input_generation',),
            activation_rules=(ActivationRule('solver.name', 'in', ('block2_dmrg', 'block2', 'block2-dmrg', 'dmrg')),),
            description='Validate block2 DMRG, active-space localization provenance, and canonical/Fiedler/manual orbital ordering before execution.',
        ),
        ModuleContract(
            module_id='solver.block2.bond_dimension_planning',
            version='1.0',
            permitted_stages=('task.prepare',),
            default_stage='task.prepare',
            required_inputs=(
                _port('task.spec', TASK_SPEC_SCHEMA),
                _port('solver.provider.ready', 'pyscf-agent.solver-provider.v1'),
            ),
            provided_outputs=(
                _port('solver.block2.bond_dimension_planning.ready', 'pyscf-agent.block2-feature.v1'),
            ),
            compatibility=ModuleCompatibility(
                task_types=('molecular', 'model_hamiltonian'),
                methods=('casci', 'casscf', 'block2_dmrg', 'block2', 'block2-dmrg', 'dmrg'),
            ),
            optional_configuration={
                'type': 'object',
                'properties': {'bond_dimension_planning': {'type': 'boolean'}},
                'additionalProperties': False,
            },
            default_configuration={},
            requires_modules=('solver.block2.dmrg',),
            before=('core.input_generation',),
            activation_rules=(
                ActivationRule('solver.name', 'in', ('block2_dmrg', 'block2', 'block2-dmrg', 'dmrg')),
            ),
            description=(
                'Bound block2 schedules by the exact fixed-particle-number '
                'Schmidt rank before numerical execution.'
            ),
        ),
        ModuleContract(
            module_id='solver.block2.entanglement_diagnostics',
            version='1.0',
            permitted_stages=('task.prepare',),
            default_stage='task.prepare',
            required_inputs=(
                _port('task.spec', TASK_SPEC_SCHEMA),
                _port('solver.provider.ready', 'pyscf-agent.solver-provider.v1'),
            ),
            provided_outputs=(_port('solver.block2.entanglement.ready', 'pyscf-agent.block2-feature.v1'),),
            compatibility=ModuleCompatibility(
                task_types=('molecular', 'model_hamiltonian'),
                methods=('casci', 'casscf', 'block2_dmrg', 'block2', 'block2-dmrg', 'dmrg'),
            ),
            optional_configuration={
                'type': 'object',
                'properties': {
                    'compute_entanglement': {'type': 'boolean'},
                    'compute_mutual_information': {'type': 'boolean'},
                    'compute_bipartite_entanglement': {'type': 'boolean'},
                },
                'additionalProperties': False,
            },
            default_configuration={
                'compute_entanglement': True,
                'compute_mutual_information': True,
                'compute_bipartite_entanglement': True,
            },
            requires_modules=('solver.block2.dmrg',),
            before=('core.input_generation',),
            activation_rules=(
                ActivationRule('solver.name', 'in', ('block2_dmrg', 'block2', 'block2-dmrg', 'dmrg')),
                ActivationRule('analysis.outputs', 'contains_any', ('entanglement_diagnostics',)),
            ),
            description='Configure single-orbital entropy, mutual information, and bipartite entropy outputs.',
        ),
        ModuleContract(
            module_id='solver.block2.excited_states',
            version='1.0',
            permitted_stages=('task.prepare',),
            default_stage='task.prepare',
            required_inputs=(
                _port('task.spec', TASK_SPEC_SCHEMA),
                _port('solver.provider.ready', 'pyscf-agent.solver-provider.v1'),
            ),
            provided_outputs=(_port('solver.block2.excited_states.ready', 'pyscf-agent.block2-feature.v1'),),
            compatibility=ModuleCompatibility(
                task_types=('molecular', 'model_hamiltonian'),
                methods=('casci', 'casscf', 'block2_dmrg', 'block2', 'block2-dmrg', 'dmrg'),
            ),
            optional_configuration={
                'type': 'object',
                'properties': {
                    'nroots': {'type': 'integer', 'minimum': 2},
                    'excited_state_mode': {'enum': ['state_averaged']},
                    'state_average_weights': {
                        'type': 'array',
                        'items': {'type': 'number', 'minimum': 0},
                        'minItems': 2,
                    },
                },
                'additionalProperties': False,
            },
            default_configuration={},
            requires_modules=('solver.block2.dmrg',),
            before=('core.input_generation',),
            activation_rules=(
                ActivationRule('solver.name', 'in', ('block2_dmrg', 'block2', 'block2-dmrg', 'dmrg')),
                ActivationRule('analysis.outputs', 'contains_any', ('excited_states',)),
            ),
            description='Configure state-averaged multi-root DMRG and excitation-energy output.',
        ),
        ModuleContract(
            module_id='solver.block2.mps_continuation',
            version='1.0',
            permitted_stages=('task.prepare',),
            default_stage='task.prepare',
            required_inputs=(
                _port('task.spec', TASK_SPEC_SCHEMA),
                _port('solver.provider.ready', 'pyscf-agent.solver-provider.v1'),
            ),
            provided_outputs=(_port('solver.block2.continuation.ready', 'pyscf-agent.block2-feature.v1'),),
            compatibility=ModuleCompatibility(
                task_types=('molecular', 'model_hamiltonian'),
                methods=('casci', 'casscf', 'block2_dmrg', 'block2', 'block2-dmrg', 'dmrg'),
            ),
            optional_configuration={
                'type': 'object',
                'properties': {
                    'restart_manifest': {
                        'anyOf': [
                            {'type': 'string'},
                            {'type': 'object'},
                        ],
                    },
                    'orbital_restart_manifest': {
                        'anyOf': [
                            {'type': 'string'},
                            {'type': 'object'},
                        ],
                    },
                    'restart_tag': {'type': 'string'},
                    'restart_required': {'type': 'boolean'},
                    'restart_geometry_policy': {'type': 'string', 'enum': ['same_geometry', 'transport']},
                    'restart_min_active_overlap': {'type': 'number', 'exclusiveMinimum': 0, 'maximum': 1},
                },
                'additionalProperties': False,
            },
            default_configuration={'restart_tag': 'GS', 'restart_required': True},
            capability_ids=('study.workflow_option.block2_mps_continuation',),
            requires_modules=('solver.block2.dmrg',),
            before=('core.input_generation',),
            activation_rules=(
                ActivationRule('solver.name', 'in', ('block2_dmrg', 'block2', 'block2-dmrg', 'dmrg')),
                ActivationRule(
                    'solver.options',
                    'contains_any',
                    ('restart_manifest', 'orbital_restart_manifest'),
                ),
            ),
            description=(
                'Validate and stage a compatible block2 checkpoint. Fixed-orbital tasks '
                'reuse an MPS; DMRG-CASSCF continuation restores optimized orbitals and '
                'a compatible MPS together.'
            ),
        ),
        ModuleContract(
            module_id='solver.block2.symmetry_analysis',
            version='1.0',
            permitted_stages=('task.prepare',),
            default_stage='task.prepare',
            required_inputs=(
                _port('task.spec', TASK_SPEC_SCHEMA),
                _port('solver.provider.ready', 'pyscf-agent.solver-provider.v1'),
            ),
            provided_outputs=(_port('solver.block2.symmetry.ready', 'pyscf-agent.block2-feature.v1'),),
            compatibility=ModuleCompatibility(
                task_types=('molecular', 'model_hamiltonian'),
                methods=('casci', 'casscf', 'block2_dmrg', 'block2', 'block2-dmrg', 'dmrg'),
            ),
            optional_configuration={
                'type': 'object',
                'properties': {'compute_symmetry_analysis': {'type': 'boolean'}},
                'additionalProperties': False,
            },
            default_configuration={'compute_symmetry_analysis': True},
            requires_modules=('solver.block2.dmrg',),
            before=('core.input_generation',),
            activation_rules=(
                ActivationRule('solver.name', 'in', ('block2_dmrg', 'block2', 'block2-dmrg', 'dmrg')),
                ActivationRule('analysis.outputs', 'contains_any', ('symmetry_analysis',)),
            ),
            description='Configure particle-number, spin-sector, spin-square, and orbital-symmetry reporting.',
        ),
        ModuleContract(
            module_id='core.repair_retry',
            version='1.0',
            permitted_stages=('task.recover',),
            default_stage='task.recover',
            required_inputs=(_port('execution.raw'),),
            provided_outputs=(_port('execution.final'),),
            always_select=True,
            description='Apply the bounded single-task recovery policy.',
        ),
        ModuleContract(
            module_id='core.result_extraction',
            version='1.0',
            permitted_stages=('task.extract',),
            default_stage='task.extract',
            required_inputs=(_port('execution.final'),),
            provided_outputs=(_port('result.structured', 'pyscf-agent.structured-results.v1'),),
            always_select=True,
            description='Convert solver output into structured numerical results.',
        ),
        ModuleContract(
            module_id='core.result_analysis',
            version='1.0',
            permitted_stages=('task.diagnose',),
            default_stage='task.diagnose',
            required_inputs=(_port('result.structured'),),
            provided_outputs=(_port('result.analysis'),),
            always_select=True,
            after=(
                'molecular.correlation_diagnostics',
                'molecular.orbital_processing',
                'molecular.active_space_audit',
                'model.correlation_diagnostics',
                'periodic.band_analysis',
            ),
            description='Apply deterministic result interpretation before reporting.',
        ),
        ModuleContract(
            module_id='molecular.correlation_diagnostics',
            version='1.0',
            permitted_stages=('task.diagnose',),
            default_stage='task.diagnose',
            required_inputs=(_port('result.structured'),),
            provided_outputs=(_port('diagnostics.correlation'),),
            optional_configuration={
                'type': 'object',
                'properties': {'enabled': {'type': 'boolean'}, 'scf_stability': {'type': 'boolean'}},
                'additionalProperties': False,
            },
            default_configuration={'enabled': True, 'scf_stability': False},
            capability_ids=(
                'molecular.workflow_option.correlation_diagnostics',
                'molecular.workflow_option.scf_stability',
            ),
            compatibility=ModuleCompatibility(
                task_types=('molecular',),
                methods=molecular_methods,
                jobs=('single_point',),
            ),
            activation_rules=(
                ActivationRule('task_type', 'equals', 'molecular'),
                ActivationRule('method.name', 'in', molecular_methods),
                ActivationRule('job.name', 'equals', 'single_point'),
            ),
            description='Evaluate molecular static-correlation indicators and solver stress.',
        ),
        ModuleContract(
            module_id='molecular.active_space_probe',
            version='1.0',
            permitted_stages=('task.prepare',),
            default_stage='task.prepare',
            required_inputs=(_port('task.spec', TASK_SPEC_SCHEMA),),
            provided_outputs=(
                _port('active_space.probe.ready', 'pyscf-agent.active-space-probe.v1'),
            ),
            capability_ids=(
                'molecular.active_space_probe_strategy.auto',
                'molecular.active_space_probe_strategy.mp2',
                'molecular.active_space_probe_strategy.fci',
            ),
            compatibility=ModuleCompatibility(
                task_types=('molecular',),
                methods=('hf', 'mp2', 'fci'),
            ),
            optional_configuration={
                'type': 'object',
                'properties': {
                    'requested_strategy': {'type': 'string', 'enum': ['auto', 'mp2', 'fci']},
                    'probe_method': {'type': 'string', 'enum': ['hf', 'mp2', 'fci']},
                    'refinement_method': {'type': 'string', 'enum': ['mp2']},
                    'occupation_window': {
                        'type': 'array',
                        'items': {'type': 'number'},
                    },
                },
                'additionalProperties': False,
            },
            conflicts_with=('solver.block2.dmrg',),
            before=('core.input_generation',),
            activation_rules=(
                ActivationRule('task_type', 'equals', 'molecular'),
                ActivationRule('method.name', 'in', ('hf', 'mp2', 'fci')),
                ActivationRule('active_space.enabled'),
                ActivationRule('active_space.target_method', 'in', ('casci', 'casscf')),
            ),
            description=(
                'Prepare a Registry-selected active-space probe while retaining the requested CAS '
                'method and solver only as target metadata.'
            ),
        ),
        ModuleContract(
            module_id='molecular.active_space_probe_refinement',
            version='1.0',
            permitted_stages=('task.diagnose',),
            default_stage='task.diagnose',
            required_inputs=(
                _port('result.structured'),
                _port('active_space.probe.ready', 'pyscf-agent.active-space-probe.v1'),
            ),
            provided_outputs=(
                _port(
                    'active_space.probe.refinement',
                    'pyscf-agent.active-space-probe-refinement.v1',
                ),
            ),
            compatibility=ModuleCompatibility(task_types=('molecular',), methods=('hf',)),
            optional_configuration={
                'type': 'object',
                'properties': {
                    'initial_method': {'type': 'string', 'enum': ['hf']},
                    'refinement_method': {'type': 'string', 'enum': ['mp2']},
                },
                'additionalProperties': False,
            },
            before=('molecular.correlation_diagnostics', 'molecular.active_space_audit'),
            activation_rules=(
                ActivationRule('task_type', 'equals', 'molecular'),
                ActivationRule('method.name', 'equals', 'hf'),
                ActivationRule('active_space.enabled'),
                ActivationRule('active_space.target_method', 'in', ('casci', 'casscf')),
            ),
            description=(
                'Accept a consistent chemical-valence AVAS/UNO proposal from SCF, or add MP2 '
                'natural-occupation and amplitude evidence when the proposal is unresolved or '
                'SCF diagnostics indicate instability, frontier degeneracy, or a routing boundary.'
            ),
        ),
        ModuleContract(
            module_id='molecular.orbital_processing',
            version='1.0',
            permitted_stages=('task.diagnose',),
            default_stage='task.diagnose',
            required_inputs=(_port('result.structured'),),
            provided_outputs=(_port('orbitals.processed'),),
            capability_ids=(
                'molecular.orbital_processing.orbital_table',
                'molecular.orbital_processing.frozen_orbitals',
                'molecular.orbital_processing.orbital_continuation',
                'molecular.orbital_processing.boys',
                'molecular.orbital_processing.pipek_mezey',
                'molecular.workflow_option.orbital_processing',
            ),
            compatibility=ModuleCompatibility(task_types=('molecular',)),
            activation_rules=(ActivationRule('orbital_processing.enabled'),),
            description='Produce analysis-only orbital summaries, or record initial localization, DMRG ordering, and orbital optimization for block2 CASCI/CASSCF.',
        ),
        ModuleContract(
            module_id='molecular.active_space_audit',
            version='1.0',
            permitted_stages=('task.diagnose',),
            default_stage='task.diagnose',
            required_inputs=(_port('result.structured'),),
            provided_outputs=(_port('active_space.audit', 'pyscf-agent.active-space-audit.v1'),),
            compatibility=ModuleCompatibility(task_types=('molecular',)),
            optional_configuration={
                'type': 'object',
                'properties': {
                    'candidate_limit': {'type': 'integer', 'minimum': 1},
                    'require_approval': {'type': 'boolean'},
                },
                'additionalProperties': False,
            },
            default_configuration={'candidate_limit': 7, 'require_approval': True},
            capability_ids=(
                'molecular.active_space.manual',
                'molecular.active_space.occupation_window',
                'molecular.active_space.energy_window',
                'molecular.active_space.uno',
                'molecular.active_space.avas',
                'molecular.active_space.chemical_valence',
                'molecular.active_space.evidence_expanded',
                'molecular.active_space.merged',
                'molecular.workflow_option.active_space',
            ),
            after=('molecular.correlation_diagnostics', 'molecular.orbital_processing'),
            activation_rules=(ActivationRule('active_space.enabled'),),
            description='Produce a reviewable and provenance-complete active-space proposal.',
        ),
        ModuleContract(
            module_id='model.correlation_diagnostics',
            version='1.0',
            permitted_stages=('task.diagnose',),
            default_stage='task.diagnose',
            required_inputs=(_port('result.structured'),),
            provided_outputs=(_port('diagnostics.correlation'),),
            compatibility=ModuleCompatibility(
                task_types=('model_hamiltonian',),
                methods=('mp2', 'ccsd', 'ccsd_t', 'fci', 'block2_dmrg', 'dmet'),
            ),
            activation_rules=(
                ActivationRule('task_type', 'equals', 'model_hamiltonian'),
                ActivationRule(
                    'solver.name',
                    'in',
                    ('mp2', 'ccsd', 'ccsd_t', 'fci', 'block2_dmrg', 'dmet'),
                ),
                ActivationRule('analysis.outputs', 'contains_any', ('strong_correlation_diagnostics',)),
            ),
            description='Evaluate finite-cluster Hubbard-style correlation diagnostics.',
        ),
        ModuleContract(
            module_id='periodic.band_analysis',
            version='1.0',
            permitted_stages=('task.diagnose',),
            default_stage='task.diagnose',
            required_inputs=(_port('result.structured'),),
            provided_outputs=(_port('periodic.band_summary'),),
            compatibility=ModuleCompatibility(task_types=('periodic',)),
            activation_rules=(
                ActivationRule('task_type', 'equals', 'periodic'),
                ActivationRule('analysis.outputs', 'contains_any', ('band_structure',)),
            ),
            description='Summarize periodic band paths, gaps, and Fermi-level references.',
        ),
        ModuleContract(
            module_id='core.task_report',
            version='1.0',
            permitted_stages=('task.finalize',),
            default_stage='task.finalize',
            required_inputs=(_port('result.structured'), _port('result.analysis')),
            provided_outputs=(_port('report.task', TASK_REPORT_SCHEMA),),
            always_select=True,
            description='Assemble the final TaskReport and artifact references.',
        ),
    ]
