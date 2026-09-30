from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from computational_study_agent.planner import build_study_plan
from computational_study_agent.application.service import StudyApplicationService
from computational_study_agent.module_runtime import get_study_module_runtime_registry
from computational_study_agent.schema import StudySpec

from pyscf_agent.application.calculation_service import CalculationApplicationService
from pyscf_agent.backend.parsing import task_spec_from_partial
from pyscf_agent.contracts import task_spec_from_dict, task_spec_to_dict
from pyscf_agent.backend.artifacts import write_text_artifact
from pyscf_agent.backend.execution import generate_input_script, task_reporter
from pyscf_agent.backend.module_compilation import module_compiler
from pyscf_agent.backend.module_runtime import (
    get_default_runtime_dispatcher,
    get_default_runtime_registry,
)
from pyscf_agent.backend.probe_modules import active_space_probe_refinement_decision
from pyscf_agent.backend.state import default_state
from pyscf_agent.registry import default_registry
from pyscf_agent.workflow_modules import (
    DataPortContract,
    ModuleCompatibility,
    ResolvedModuleContract,
    ModuleRuntimeAdapter,
    ModuleRuntimeRegistry,
    WorkflowCompilationError,
    WorkflowRuntimeDispatcher,
    WorkflowRuntimeError,
    compile_study_workflow,
    compile_task_workflow,
    validate_runtime_configuration,
)
from pyscf_agent.workflow_modules.runtime import (
    ModuleInvocation,
    RuntimeIssue,
    record_module_observation,
)


def _port(name):
    return DataPortContract(name=name)


def _module(
    module_id,
    *,
    required=(),
    provided=(),
    always=True,
    stage='task.diagnose',
    **kwargs
):
    return ResolvedModuleContract(
        module_id=module_id,
        version='1.0',
        permitted_stages=(stage,),
        default_stage=stage,
        required_inputs=tuple(_port(item) for item in required),
        provided_outputs=tuple(_port(item) for item in provided),
        always_select=always,
        provider_id='provider.tests',
        runtime_id='tests.' + module_id,
        **kwargs
    )


class WorkflowModuleCompilerTests(unittest.TestCase):
    def setUp(self):
        self.registry = default_registry()

    @staticmethod
    def _spec(payload=None):
        return task_spec_to_dict(task_spec_from_dict(payload or {
            'task_type': 'molecular',
            'method': {'name': 'mp2'},
        }))

    def test_default_registry_exposes_strongly_typed_module_contracts(self):
        contract = self.registry.module('core.execution')
        self.assertIsNotNone(contract)
        self.assertEqual(contract.default_stage, 'task.execute')
        self.assertIn('task.generated_input', [item.name for item in contract.required_inputs])
        payload = self.registry.as_dict()
        entries = payload['registry']['entries']
        task_modules = [
            item['contract']
            for item in entries
            if item['kind'] == 'module' and item['namespace'] == 'task.module'
        ]
        study_modules = [
            item['contract']
            for item in entries
            if item['kind'] == 'module' and item['namespace'] == 'study.module'
        ]
        self.assertIn('core.execution', [item['module_id'] for item in task_modules])
        self.assertIn('molecular.active_space_probe', [
            item['module_id'] for item in task_modules
        ])
        self.assertIn('core.study_plan', [item['module_id'] for item in study_modules])
        self.assertIn('analysis.block2.state_tracking', [
            item['module_id'] for item in study_modules
        ])

    def test_runtime_observations_share_one_uniform_contract(self):
        state = {'execution_status': 'running'}
        invocation = ModuleInvocation(
            workflow_id='workflow-test',
            module_id='tests.module',
            runtime_id='tests.runtime',
            stage='task.analyze',
            configuration={'threshold': 0.1},
        )

        observation = record_module_observation(
            state,
            invocation,
            'completed',
            provided_result_paths=('diagnostics',),
            provider='tests',
        )

        self.assertEqual(observation['execution_status'], 'running')
        self.assertEqual(observation['configuration'], {'threshold': 0.1})
        self.assertEqual(observation['provided_result_paths'], ['diagnostics'])
        self.assertEqual(observation['provider'], 'tests')
        self.assertEqual(state['module_runtime_observations'], [observation])

    def test_default_molecular_workflow_is_compiled_in_dependency_order(self):
        result = compile_task_workflow(self._spec(), self.registry.modules_for(scope='task'))
        self.assertEqual(result.execution_order[0], 'core.input_generation')
        self.assertEqual(result.execution_order[-1], 'core.task_report')
        self.assertIn('molecular.correlation_diagnostics', result.execution_order)
        self.assertEqual(result.provenance['status'], 'compiled')
        self.assertTrue(result.workflow_id.startswith('workflow-'))

    def test_every_default_module_resolves_to_a_runtime_adapter(self):
        configuration = compile_task_workflow(self._spec(), self.registry.modules_for(scope='task')).to_dict()
        runtime_registry = get_default_runtime_registry()
        self.assertEqual(validate_runtime_configuration(configuration, runtime_registry), ())
        self.assertTrue(all(
            runtime_registry.resolve(node['runtime_id']) is not None
            for node in configuration['nodes']
        ))

    def test_numerical_modules_use_independent_runtime_handlers(self):
        runtime_registry = get_default_runtime_registry()
        for runtime_id in (
            'pyscf_agent.backend.correlation_diagnostics',
            'pyscf_agent.backend.orbital_processing',
            'pyscf_agent.backend.active_space.audit',
            'pyscf_agent.backend.model_hamiltonian.diagnostics',
            'pyscf_agent.backend.periodic.band_analysis',
        ):
            with self.subTest(runtime_id=runtime_id):
                self.assertEqual(runtime_registry.resolve(runtime_id).mode, 'handler')

    def test_model_and_periodic_modules_activate_only_for_requested_outputs(self):
        model_base = {
            'task_type': 'model_hamiltonian',
            'solver': {'name': 'fci'},
            'analysis': {'outputs': ['energy']},
        }
        energy_only = compile_task_workflow(model_base, self.registry.modules_for(scope='task'))
        self.assertNotIn('model.correlation_diagnostics', energy_only.execution_order)
        model_base['analysis']['outputs'].append('strong_correlation_diagnostics')
        diagnostics = compile_task_workflow(model_base, self.registry.modules_for(scope='task'))
        self.assertIn('model.correlation_diagnostics', diagnostics.execution_order)

        periodic = {
            'task_type': 'periodic',
            'method': {'name': 'hf'},
            'analysis': {'outputs': ['energy', 'band_structure']},
        }
        band_workflow = compile_task_workflow(periodic, self.registry.modules_for(scope='task'))
        self.assertIn('periodic.band_analysis', band_workflow.execution_order)

    def test_dmet_reference_density_strategy_is_an_exclusive_prepare_module(self):
        for strategy in ('pm', 'af', 'fm', 'cdw'):
            with self.subTest(strategy=strategy):
                spec = {
                    'task_type': 'model_hamiltonian',
                    'solver': {
                        'name': 'dmet',
                        'options': {'reference_density_guess': strategy},
                    },
                }
                workflow = compile_task_workflow(
                    spec,
                    self.registry.modules_for(scope='task'),
                )
                selected = [
                    module_id for module_id in workflow.execution_order
                    if module_id.startswith('embedding.libdmet.reference_density.')
                ]

                self.assertEqual(
                    selected,
                    ['embedding.libdmet.reference_density.{0}'.format(strategy)],
                )
                self.assertLess(
                    workflow.execution_order.index(selected[0]),
                    workflow.execution_order.index('embedding.libdmet.dmet'),
                )

    def test_molecular_numerical_modules_run_before_result_analysis(self):
        spec = self._spec({
            'task_type': 'molecular',
            'method': {'name': 'hf'},
            'orbital_processing': {'enabled': True},
            'active_space': {'enabled': True},
        })
        order = list(compile_task_workflow(spec, self.registry.modules_for(scope='task')).execution_order)
        analysis_index = order.index('core.result_analysis')
        for module_id in (
            'molecular.correlation_diagnostics',
            'molecular.orbital_processing',
            'molecular.active_space_audit',
        ):
            self.assertLess(order.index(module_id), analysis_index)

    def test_active_space_probe_is_independent_from_target_block2_solver(self):
        spec = self._spec({
            'task_type': 'molecular',
            'method': {'name': 'mp2'},
            'active_space': {
                'enabled': True,
                'selection_method': 'occupation_window',
                'target_method': 'casscf',
                'target_solver': 'block2_dmrg',
                'target_solver_options': {'nroots': 2},
            },
            'workflow': {'modules': ['molecular.active_space_probe']},
        })

        workflow = compile_task_workflow(spec, self.registry.modules_for(scope='task'))

        self.assertIn('molecular.active_space_probe', workflow.execution_order)
        self.assertNotIn('solver.block2.dmrg', workflow.execution_order)
        self.assertLess(
            workflow.execution_order.index('molecular.active_space_probe'),
            workflow.execution_order.index('core.input_generation'),
        )
        self.assertEqual(
            validate_runtime_configuration(
                workflow.to_dict(),
                get_default_runtime_registry(),
            ),
            (),
        )

    def test_auto_active_space_probe_compiles_scf_first_refinement(self):
        prepared = CalculationApplicationService().prepare_active_space_probe({
            'task_type': 'molecular',
            'atom': 'H 0 0 0; H 0 0 0.74',
            'basis': 'sto-3g',
            'method': 'casscf',
        })
        spec = task_spec_to_dict(task_spec_from_partial(prepared['probe_request']))
        workflow = compile_task_workflow(spec, self.registry.modules_for(scope='task'))

        self.assertEqual(spec['method']['name'], 'hf')
        self.assertIn('molecular.active_space_probe_refinement', workflow.execution_order)
        self.assertLess(
            workflow.execution_order.index('core.result_extraction'),
            workflow.execution_order.index('molecular.active_space_probe_refinement'),
        )
        self.assertLess(
            workflow.execution_order.index('molecular.active_space_probe_refinement'),
            workflow.execution_order.index('molecular.active_space_audit'),
        )
        script = generate_input_script(task_spec_from_dict(spec))
        self.assertIn('run_workflow_sequential', script)
        self.assertNotIn('_run_pyscf_task', script)

    def test_active_space_probe_accepts_review_ready_chemical_valence_candidate(self):
        decision = active_space_probe_refinement_decision({
            'ncas': 6,
            'nelecas': 6,
            'audit': {
                'selected_candidate_method': 'chemical_valence',
                'candidate_active_spaces': [{
                    'method': 'chemical_valence',
                    'status': 'available',
                    'evaluation': {'risk_flags': []},
                }],
                'ncas_nelecas_consistency': {'consistent': True},
            },
            'audit_summary': {'selection_confidence': 'high'},
        })

        self.assertFalse(decision['refinement_required'])
        self.assertEqual(decision['status'], 'review_ready')

    def test_active_space_probe_refines_review_ready_candidate_for_risky_scf_diagnostics(self):
        active_space = {
            'ncas': 6,
            'nelecas': 6,
            'audit': {
                'selected_candidate_method': 'chemical_valence',
                'candidate_active_spaces': [{
                    'method': 'chemical_valence',
                    'status': 'available',
                    'evaluation': {'risk_flags': []},
                }],
                'ncas_nelecas_consistency': {'consistent': True},
            },
            'audit_summary': {'selection_confidence': 'high'},
        }
        decision = active_space_probe_refinement_decision(
            active_space,
            diagnostics={
                'scf_stability': {'status': 'completed', 'stable': False},
                'frontier_orbital_degeneracy': {
                    'status': 'available',
                    'score': 0.5,
                },
                'molecular_correlation_risk': {'overall_score': 0.36},
            },
        )

        self.assertTrue(decision['refinement_required'])
        self.assertIn('scf_instability', decision['reasons'])
        self.assertIn('frontier_orbital_degeneracy', decision['reasons'])
        self.assertIn('near_correlation_decision_boundary', decision['reasons'])

    def test_active_space_probe_refines_ambiguous_scf_candidate(self):
        decision = active_space_probe_refinement_decision({
            'ncas': 6,
            'nelecas': 6,
            'audit': {
                'selected_candidate_method': 'chemical_valence',
                'candidate_active_spaces': [{
                    'method': 'chemical_valence',
                    'status': 'available',
                    'evaluation': {'risk_flags': ['orbital_mapping_ambiguous']},
                }],
                'ncas_nelecas_consistency': {'consistent': True},
            },
        })

        self.assertTrue(decision['refinement_required'])
        self.assertIn('orbital_mapping_ambiguous', decision['reasons'])

    def test_active_space_probe_runtime_records_target_without_loading_solver(self):
        spec = self._spec({
            'task_type': 'molecular',
            'system': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
            },
            'method': {'name': 'mp2', 'restricted': False},
            'active_space': {
                'enabled': True,
                'selection_method': 'occupation_window',
                'target_method': 'casscf',
                'target_solver': 'block2_dmrg',
                'target_solver_options': {'nroots': 2},
            },
            'workflow': {'modules': ['molecular.active_space_probe']},
        })
        workflow = compile_task_workflow(spec, self.registry.modules_for(scope='task'))
        with tempfile.TemporaryDirectory() as work_dir:
            state = default_state('{}', work_dir=work_dir, run_id='probe-runtime')
            state['task_spec'] = spec
            state['workflow_configuration'] = workflow.to_dict()

            result = get_default_runtime_dispatcher().run_stage(state, 'task.prepare')
            reported = task_reporter(result)

        self.assertNotIn('solver_provider', result)
        self.assertEqual(result['calculation_role'], 'active_space_probe')
        self.assertEqual(result['active_space_probe']['probe_method'], 'mp2')
        self.assertEqual(result['active_space_probe']['target_method'], 'casscf')
        self.assertEqual(result['active_space_probe']['target_solver'], 'block2_dmrg')
        self.assertEqual(reported['task_report']['calculation_role'], 'active_space_probe')
        self.assertEqual(reported['task_report']['task_spec']['method']['name'], 'mp2')
        observation = next(
            item for item in result['module_runtime_observations']
            if item.get('module_id') == 'molecular.active_space_probe'
        )
        self.assertEqual(observation['role'], 'active_space_probe')
        self.assertEqual(observation['status'], 'completed')

    def test_fci_active_space_probe_runtime_preserves_requested_strategy(self):
        spec = self._spec({
            'task_type': 'molecular',
            'system': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
            },
            'method': {'name': 'fci', 'restricted': False},
            'active_space': {
                'enabled': True,
                'selection_method': 'occupation_window',
                'target_method': 'casscf',
                'target_solver': 'fci',
            },
            'workflow': {
                'modules': ['molecular.active_space_probe'],
                'module_config': {
                    'molecular.active_space_probe': {
                        'requested_strategy': 'fci',
                        'probe_method': 'fci',
                        'occupation_window': [0.02, 1.98],
                    },
                },
            },
        })
        workflow = compile_task_workflow(spec, self.registry.modules_for(scope='task'))
        with tempfile.TemporaryDirectory() as work_dir:
            state = default_state('{}', work_dir=work_dir, run_id='fci-probe-runtime')
            state['task_spec'] = spec
            state['workflow_configuration'] = workflow.to_dict()
            result = get_default_runtime_dispatcher().run_stage(state, 'task.prepare')

        self.assertEqual(result['active_space_probe']['requested_strategy'], 'fci')
        self.assertEqual(result['active_space_probe']['probe_method'], 'fci')

    def test_rewriting_artifact_path_replaces_stale_provenance(self):
        with tempfile.TemporaryDirectory() as work_dir:
            state = default_state('{}', work_dir=work_dir, run_id='artifact-refresh')
            write_text_artifact(state, 'structured_results', 'result.json', 'first')
            write_text_artifact(state, 'structured_results', 'result.json', 'second')
            matching = [item for item in state['artifacts'] if Path(item['path']).name == 'result.json']
            self.assertEqual(len(matching), 1)
            self.assertEqual(matching[0]['size_bytes'], len('second'))

    def test_runtime_validation_rejects_unavailable_adapter(self):
        configuration = compile_task_workflow(self._spec(), self.registry.modules_for(scope='task')).to_dict()
        configuration['nodes'][0]['runtime_id'] = 'missing.runtime'
        issues = validate_runtime_configuration(configuration, get_default_runtime_registry())
        self.assertIn('unknown_runtime_adapter', [issue.code for issue in issues])

    def test_dispatcher_invokes_compiled_hook_order_and_records_trace(self):
        calls = []

        def handler(state, invocation):
            state = dict(state)
            calls.append(invocation.module_id)
            state.setdefault('values', []).append(invocation.configuration['value'])
            return state

        runtime_registry = ModuleRuntimeRegistry([
            ModuleRuntimeAdapter('tests.runtime', handler, ('task.diagnose',)),
        ])
        configuration = {
            'workflow_id': 'workflow-test',
            'nodes': [
                {
                    'module_id': 'module.a',
                    'runtime_id': 'tests.runtime',
                    'stage': 'task.diagnose',
                    'configuration': {'value': 'a'},
                    'required_inputs': [],
                    'provided_outputs': [{'name': 'data.a'}],
                },
                {
                    'module_id': 'module.b',
                    'runtime_id': 'tests.runtime',
                    'stage': 'task.diagnose',
                    'configuration': {'value': 'b'},
                    'required_inputs': [{'name': 'data.a'}],
                    'provided_outputs': [{'name': 'data.b'}],
                },
            ],
            'hooks': {'task.diagnose': ['module.a', 'module.b']},
            'execution_order': ['module.a', 'module.b'],
        }
        result = WorkflowRuntimeDispatcher(runtime_registry).run_stage(
            {'workflow_configuration': configuration, 'module_execution_trace': []},
            'task.diagnose',
        )
        self.assertEqual(calls, ['module.a', 'module.b'])
        self.assertEqual(result['values'], ['a', 'b'])
        self.assertEqual(
            [entry['module_id'] for entry in result['module_execution_trace']],
            ['module.a', 'module.b'],
        )
        self.assertTrue(all(
            entry['status'] == 'completed'
            for entry in result['module_execution_trace']
        ))

    def test_dispatcher_rejects_hook_order_that_differs_from_compiler_order(self):
        runtime_registry = ModuleRuntimeRegistry([
            ModuleRuntimeAdapter(
                'tests.runtime',
                lambda state, _invocation: state,
                ('task.diagnose',),
            ),
        ])
        configuration = {
            'workflow_id': 'workflow-test',
            'nodes': [
                {'module_id': 'module.a', 'runtime_id': 'tests.runtime', 'stage': 'task.diagnose'},
                {'module_id': 'module.b', 'runtime_id': 'tests.runtime', 'stage': 'task.diagnose'},
            ],
            'hooks': {'task.diagnose': ['module.b', 'module.a']},
            'execution_order': ['module.a', 'module.b'],
        }
        with self.assertRaises(WorkflowRuntimeError):
            WorkflowRuntimeDispatcher(runtime_registry).run_stage(
                {'workflow_configuration': configuration},
                'task.diagnose',
            )

    def test_dispatcher_preserves_failed_module_trace_and_identity(self):
        def fail(_state, _invocation):
            raise ValueError('invalid provider option')

        runtime_registry = ModuleRuntimeRegistry([
            ModuleRuntimeAdapter('tests.failure', fail, ('task.prepare',)),
        ])
        configuration = {
            'workflow_id': 'workflow-test',
            'nodes': [{
                'module_id': 'module.failure',
                'runtime_id': 'tests.failure',
                'stage': 'task.prepare',
                'configuration': {'nroots': 4},
            }],
            'hooks': {'task.prepare': ['module.failure']},
            'execution_order': ['module.failure'],
        }

        with self.assertRaises(WorkflowRuntimeError) as caught:
            WorkflowRuntimeDispatcher(runtime_registry).run_stage(
                {'workflow_configuration': configuration, 'module_execution_trace': []},
                'task.prepare',
            )

        self.assertEqual(caught.exception.issues[0].module_id, 'module.failure')
        self.assertEqual(caught.exception.state['module_execution_trace'][0]['status'], 'failed')
        self.assertIn('invalid provider option', caught.exception.state['module_execution_trace'][0]['error'])

    def test_dispatcher_preserves_state_attached_to_structured_runtime_error(self):
        def fail(state, _invocation):
            state = dict(state)
            state['provider_failure_artifact'] = '/tmp/provider.log'
            raise WorkflowRuntimeError((RuntimeIssue(
                'provider_failed',
                'provider retained diagnostic output',
            ),), state=state)

        runtime_registry = ModuleRuntimeRegistry([
            ModuleRuntimeAdapter('tests.structured_failure', fail, ('task.execute',)),
        ])
        configuration = {
            'workflow_id': 'workflow-test',
            'nodes': [{
                'module_id': 'module.failure',
                'runtime_id': 'tests.structured_failure',
                'stage': 'task.execute',
                'configuration': {},
            }],
            'hooks': {'task.execute': ['module.failure']},
            'execution_order': ['module.failure'],
        }

        with self.assertRaises(WorkflowRuntimeError) as caught:
            WorkflowRuntimeDispatcher(runtime_registry).run_stage(
                {'workflow_configuration': configuration, 'module_execution_trace': []},
                'task.execute',
            )

        self.assertEqual(caught.exception.state['provider_failure_artifact'], '/tmp/provider.log')
        self.assertEqual(caught.exception.issues[0].module_id, 'module.failure')
        self.assertEqual(caught.exception.state['module_execution_trace'][0]['status'], 'failed')

    def test_explicit_module_configuration_is_merged_and_validated(self):
        spec = self._spec({
            'task_type': 'molecular',
            'method': {'name': 'casscf'},
            'active_space': {'enabled': True},
            'workflow': {
                'modules': ['molecular.active_space_audit'],
                'module_config': {
                    'molecular.active_space_audit': {'candidate_limit': 7},
                },
            },
        })
        result = compile_task_workflow(spec, self.registry.modules_for(scope='task'))
        node = next(item for item in result.nodes if item['module_id'] == 'molecular.active_space_audit')
        self.assertEqual(node['configuration']['candidate_limit'], 7)
        self.assertTrue(node['configuration']['require_approval'])

    def test_unique_data_provider_is_inserted_in_auto_mode(self):
        contracts = [
            _module('provider', provided=('data.x',), always=False),
            _module('consumer', required=('data.x',), provided=('data.y',), always=False),
        ]
        spec = self._spec({'workflow': {'modules': ['consumer'], 'dependency_policy': 'auto'}})
        result = compile_task_workflow(spec, contracts)
        self.assertEqual(result.execution_order, ('provider', 'consumer'))
        provider = next(item for item in result.provenance['selected_modules'] if item['module_id'] == 'provider')
        self.assertIn('data_dependency:consumer:data.x', provider['selection_reasons'])

    def test_strict_mode_rejects_missing_input(self):
        contracts = [_module('consumer', required=('data.missing',), always=False)]
        spec = self._spec({'workflow': {'modules': ['consumer'], 'dependency_policy': 'strict'}})
        with self.assertRaises(WorkflowCompilationError) as caught:
            compile_task_workflow(spec, contracts)
        self.assertIn('missing_input', [item.code for item in caught.exception.issues])

    def test_incompatible_method_is_rejected(self):
        contracts = [_module(
            'casscf.only',
            compatibility=ModuleCompatibility(task_types=('molecular',), methods=('casscf',)),
        )]
        with self.assertRaises(WorkflowCompilationError) as caught:
            compile_task_workflow(self._spec(), contracts)
        self.assertIn('incompatible_method', [item.code for item in caught.exception.issues])

    def test_casscf_accepts_state_averaged_excited_state_module(self):
        module_id = 'solver.block2.excited_states'
        spec = self._spec({
            'task_type': 'molecular',
            'method': {'name': 'casscf'},
            'solver': {'name': 'block2_dmrg'},
            'active_space': {'enabled': True},
            'workflow': {
                'modules': [module_id],
                'module_config': {
                    module_id: {
                        'nroots': 3,
                        'excited_state_mode': 'state_averaged',
                        'state_average_weights': [0.5, 0.3, 0.2],
                    },
                },
            },
        })

        workflow = compile_task_workflow(spec, self.registry.modules_for(scope='task'))
        node = next(item for item in workflow.nodes if item['module_id'] == module_id)
        self.assertEqual(node['configuration']['nroots'], 3)
        self.assertEqual(node['configuration']['excited_state_mode'], 'state_averaged')
        self.assertEqual(node['configuration']['state_average_weights'], [0.5, 0.3, 0.2])

    def test_casscf_accepts_joint_block2_continuation_module(self):
        spec = self._spec({
            'task_type': 'molecular',
            'method': {'name': 'casscf'},
            'solver': {
                'name': 'block2_dmrg',
                'options': {
                    'restart_manifest': {
                        'checkpoint_components': ['mps', 'optimized_orbitals'],
                    },
                },
            },
            'active_space': {'enabled': True},
            'workflow': {'modules': ['solver.block2.mps_continuation']},
        })

        workflow = compile_task_workflow(spec, self.registry.modules_for(scope='task'))

        self.assertIn('solver.block2.mps_continuation', workflow.execution_order)

    def test_conflicting_modules_are_rejected(self):
        contracts = [
            _module('choice.a', conflicts_with=('choice.b',)),
            _module('choice.b'),
        ]
        with self.assertRaises(WorkflowCompilationError) as caught:
            compile_task_workflow(self._spec(), contracts)
        self.assertIn('module_conflict', [item.code for item in caught.exception.issues])

    def test_dependency_cycle_is_rejected(self):
        contracts = [
            _module('cycle.a', required=('data.b',), provided=('data.a',)),
            _module('cycle.b', required=('data.a',), provided=('data.b',)),
        ]
        with self.assertRaises(WorkflowCompilationError) as caught:
            compile_task_workflow(self._spec(), contracts)
        self.assertIn('module_dependency_cycle', [item.code for item in caught.exception.issues])

    def test_invalid_hook_is_rejected(self):
        contract = ResolvedModuleContract(
            module_id='bad.hook',
            version='1.0',
            permitted_stages=('task.execute',),
            default_stage='task.diagnose',
            always_select=True,
            provider_id='provider.tests',
            runtime_id='tests.bad.hook',
        )
        with self.assertRaises(WorkflowCompilationError) as caught:
            compile_task_workflow(self._spec(), [contract])
        self.assertIn('invalid_module_stage', [item.code for item in caught.exception.issues])

    def test_duplicate_module_ids_are_rejected(self):
        contracts = [_module('duplicate'), _module('duplicate')]
        with self.assertRaises(WorkflowCompilationError) as caught:
            compile_task_workflow(self._spec(), contracts)
        self.assertIn('duplicate_module_id', [item.code for item in caught.exception.issues])

    def test_compilation_order_is_deterministic_without_content_hashes(self):
        first = compile_task_workflow(self._spec(), self.registry.modules_for(scope='task'))
        second = compile_task_workflow(self._spec(), list(reversed(self.registry.modules_for(scope='task'))))
        self.assertNotEqual(first.workflow_id, second.workflow_id)
        self.assertEqual(first.execution_order, second.execution_order)
        self.assertNotIn('registry_sha256', first.to_dict())
        self.assertNotIn('task_spec_sha256', first.to_dict())

    def test_task_spec_round_trip_preserves_workflow_request(self):
        task_spec = task_spec_from_dict({
            'workflow': {
                'modules': ['molecular.active_space_audit'],
                'module_config': {'molecular.active_space_audit': {'candidate_limit': 3}},
                'dependency_policy': 'strict',
            },
        })
        payload = task_spec_to_dict(task_spec)
        self.assertEqual(payload['workflow']['modules'], ['molecular.active_space_audit'])
        self.assertEqual(payload['workflow']['dependency_policy'], 'strict')

    def test_backend_node_writes_configuration_and_provenance_artifacts(self):
        with tempfile.TemporaryDirectory() as work_dir:
            state = default_state('{}', work_dir=work_dir)
            state['task_spec'] = self._spec()
            compiled = module_compiler(state)
            self.assertTrue(compiled['workflow_configuration'])
            self.assertEqual(compiled['workflow_provenance']['status'], 'compiled')
            kinds = [item['kind'] for item in compiled['artifacts']]
            self.assertIn('workflow-configuration', kinds)
            self.assertIn('workflow-provenance', kinds)

    def test_application_service_can_compile_without_execution(self):
        service = CalculationApplicationService()
        result = service.compile_task_spec(self._spec())
        self.assertIn('execution_order', result)
        self.assertIn('core.execution', result['execution_order'])

    def test_study_workflow_request_is_propagated_to_every_case(self):
        spec = StudySpec.from_dict({
            'name': 'workflow-study',
            'objective': 'test module propagation',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'mp2',
            },
            'sweep': {'charge': [0, 1]},
            'workflow': {
                'modules': ['molecular.correlation_diagnostics'],
                'dependency_policy': 'auto',
            },
        })
        plan = build_study_plan(spec)
        self.assertEqual(len(plan.cases), 2)
        self.assertTrue(all(
            case.request['workflow']['modules'] == ['molecular.correlation_diagnostics']
            for case in plan.cases
        ))

    def test_static_study_workflow_is_compiled_into_plan(self):
        spec = StudySpec.from_dict({
            'name': 'static-workflow',
            'objective': 'compare two charges',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'hf',
            },
            'sweep': {'charge': [0, 1]},
        })
        plan = build_study_plan(spec)
        order = plan.workflow_configuration['execution_order']
        self.assertEqual(plan.workflow_configuration['schema'], 'pyscf-agent.study-workflow-configuration.v1')
        self.assertIn('study.static_execution', order)
        self.assertNotIn('study.initial_scan', order)
        self.assertEqual(plan.workflow_provenance['status'], 'compiled')

    def test_every_study_module_has_an_explicit_orchestrator_owner(self):
        configuration = compile_study_workflow(
            {
                'name': 'runtime-study',
                'objective': 'validate study adapters',
                'system_type': 'molecular',
                'study_mode': 'adaptive',
                'base_task': {'method': 'mp2'},
                'study_workflow': {'modules': ['study.path_continuity']},
            },
            self.registry.modules_for(scope='study'),
        ).to_dict()
        runtime_registry = get_study_module_runtime_registry()
        self.assertEqual(validate_runtime_configuration(configuration, runtime_registry), ())
        self.assertTrue(all(
            runtime_registry.resolve(node['runtime_id']).owner.endswith('StudyApplicationService')
            for node in configuration['nodes']
        ))
        self.assertTrue(all(
            hasattr(
                StudyApplicationService,
                runtime_registry.resolve(node['runtime_id']).operation,
            )
            for node in configuration['nodes']
        ))

    def test_adaptive_study_workflow_compiles_routing_and_recovery(self):
        spec = {
            'name': 'adaptive-workflow',
            'objective': 'adaptive bond scan',
            'system_type': 'molecular',
            'study_mode': 'adaptive',
            'base_task': {'method': 'mp2'},
            'study_workflow': {'modules': ['study.path_continuity']},
        }
        configuration = compile_study_workflow(
            spec,
            self.registry.modules_for(scope='study'),
        )
        order = list(configuration.execution_order)
        for module_id in (
            'study.initial_scan',
            'study.correlation_routing',
            'study.method_refinement',
            'study.adaptive_execution',
            'study.recovery',
            'study.path_continuity',
        ):
            self.assertIn(module_id, order)
        self.assertLess(order.index('study.initial_scan'), order.index('study.correlation_routing'))
        self.assertLess(order.index('study.method_refinement'), order.index('study.adaptive_execution'))
        self.assertLess(order.index('study.adaptive_execution'), order.index('study.recovery'))

    def test_study_application_service_compiles_without_execution(self):
        configuration = StudyApplicationService().compile_study_spec({
            'name': 'compile-only',
            'objective': 'compile a static study',
            'system_type': 'molecular',
        })
        self.assertIn('study.static_execution', configuration['execution_order'])
        self.assertTrue(configuration['workflow_id'].startswith('study-workflow-'))


if __name__ == '__main__':
    unittest.main()
