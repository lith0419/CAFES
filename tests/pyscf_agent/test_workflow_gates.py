from __future__ import annotations

import unittest

from computational_study_agent.gates.presentation import build_report_workflow
from computational_study_agent.gates.runtime import get_study_gate_runtime_registry

from pyscf_agent.backend.gate_runtime import (
    get_default_gate_runtime_dispatcher,
    get_default_gate_runtime_registry,
)
from pyscf_agent.registry import default_registry
from pyscf_agent.workflow_gates import (
    GateCompilationError,
    GateContract,
    GateRuntimeAdapter,
    GateRuntimeRegistry,
    compile_study_gates,
    compile_task_gates,
    gate_artifact_documents,
    validate_gate_runtime_configuration,
)


def _gate(gate_id, *, after=(), config=None):
    return GateContract(
        gate_id=gate_id,
        version='1.0',
        scope='task',
        permitted_hooks=('task.before_execute',),
        default_hook='task.before_execute',
        evaluator_id='tests.' + gate_id,
        mandatory=True,
        after=tuple(after),
        optional_configuration=config or {},
    )


class WorkflowGateCompilerTests(unittest.TestCase):
    def setUp(self):
        self.registry = default_registry()

    def test_registry_exposes_task_and_study_gate_contracts(self):
        payload = self.registry.as_dict()
        entries = payload['registry']['entries']
        task_ids = [
            item['contract']['gate_id']
            for item in entries
            if item['kind'] == 'gate' and item['namespace'] == 'task.gate'
        ]
        study_ids = [
            item['contract']['gate_id']
            for item in entries
            if item['kind'] == 'gate' and item['namespace'] == 'study.gate'
        ]
        self.assertIn('core.task_compilation', task_ids)
        self.assertIn('study.path_consistency', study_ids)
        self.assertIsNotNone(self.registry.gate('study.active_space_approval'))

    def test_default_task_gates_compile_in_hook_order(self):
        configuration = compile_task_gates(
            {'task_type': 'molecular', 'method': {'name': 'mp2'}},
            self.registry.gates_for(scope='task'),
        )
        self.assertEqual(configuration.evaluation_order, (
            'core.task_compilation',
            'task.input_readiness',
            'task.execution_quality',
            'task.artifact_integrity',
        ))
        self.assertEqual(configuration.provenance['status'], 'compiled')

    def test_default_study_gates_resolve_to_runtime_adapters(self):
        configuration = compile_study_gates(
            {
                'name': 'gate-study',
                'objective': 'test gates',
                'system_type': 'molecular',
                'base_task': {'method': 'mp2'},
            },
            self.registry.gates_for(scope='study'),
        ).to_dict()
        self.assertEqual(
            validate_gate_runtime_configuration(
                configuration,
                get_study_gate_runtime_registry(),
            ),
            (),
        )
        self.assertIn('study.path_consistency', configuration['hooks']['study.after_review'])

    def test_default_task_gates_resolve_to_runtime_adapters(self):
        configuration = compile_task_gates(
            {'task_type': 'molecular', 'method': {'name': 'hf'}},
            self.registry.gates_for(scope='task'),
        ).to_dict()
        self.assertEqual(
            validate_gate_runtime_configuration(
                configuration,
                get_default_gate_runtime_registry(),
            ),
            (),
        )

    def test_task_execution_gate_reviews_failed_convergence_checks(self):
        configuration = compile_task_gates(
            {'task_type': 'model_hamiltonian', 'solver': {'name': 'dmet'}},
            self.registry.gates_for(scope='task'),
        ).to_dict()
        result = get_default_gate_runtime_dispatcher().run_hook(
            {
                'execution_status': 'succeeded',
                'retry_count': 0,
                'max_retries': 1,
                'errors': [],
                'structured_results': {
                    'quality_checks': [{
                        'id': 'dmet_energy_change',
                        'category': 'convergence',
                        'observed': 1.0e-4,
                        'operator': 'absolute_less_than',
                        'limit': 1.0e-6,
                        'required': True,
                        'source': 'iteration_history.records[-1].energy_change',
                    }],
                },
            },
            configuration,
            'task.after_recovery',
        )
        decision = next(
            item for item in result['gate_decisions']
            if item['gate_id'] == 'task.execution_quality'
        )

        self.assertEqual(decision['status'], 'review_required')
        self.assertEqual(decision['details']['quality_status'], 'review_required')
        self.assertFalse(decision['details']['publication_eligible'])

    def test_task_execution_gate_blocks_success_with_missing_equal_evidence(self):
        configuration = compile_task_gates(
            {'task_type': 'model_hamiltonian', 'solver': {'name': 'dmet'}},
            self.registry.gates_for(scope='task'),
        ).to_dict()
        result = get_default_gate_runtime_dispatcher().run_hook(
            {
                'execution_status': 'succeeded', 'retry_count': 0,
                'max_retries': 1, 'errors': [],
                'structured_results': {'quality_checks': [{
                    'id': 'dmet_impurity_sector', 'category': 'constraint',
                    'operator': 'equal', 'required': True,
                    'source': 'fragments[*].impurity_spin_sector', 'status': 'passed',
                }]},
            },
            configuration,
            'task.after_recovery',
        )
        decision = next(
            item for item in result['gate_decisions']
            if item['gate_id'] == 'task.execution_quality'
        )
        self.assertEqual(decision['status'], 'blocked')
        self.assertFalse(decision['details']['publication_eligible'])
        check = next(item for item in decision['checks'] if item['id'] == 'dmet_impurity_sector')
        self.assertEqual(check['status'], 'failed')
        self.assertIn('Invalid quality evidence', check['message'])

    def test_task_execution_gate_keeps_legacy_success_compatible(self):
        configuration = compile_task_gates(
            {'task_type': 'molecular', 'method': {'name': 'hf'}},
            self.registry.gates_for(scope='task'),
        ).to_dict()
        result = get_default_gate_runtime_dispatcher().run_hook(
            {
                'execution_status': 'succeeded',
                'retry_count': 0,
                'max_retries': 1,
                'errors': [],
                'structured_results': {'energy': -1.0},
            },
            configuration,
            'task.after_recovery',
        )
        decision = next(
            item for item in result['gate_decisions']
            if item['gate_id'] == 'task.execution_quality'
        )

        self.assertEqual(decision['status'], 'passed')
        self.assertEqual(decision['details']['quality_status'], 'legacy')
        self.assertIsNone(decision['details']['publication_eligible'])


    def test_unknown_requested_gate_is_rejected(self):
        with self.assertRaises(GateCompilationError) as caught:
            compile_task_gates(
                {'quality_gates': {'gates': ['missing.gate']}},
                self.registry.gates_for(scope='task'),
            )
        self.assertIn('unknown_gate', [item.code for item in caught.exception.issues])

    def test_gate_dependency_cycle_is_rejected(self):
        contracts = [
            _gate('gate.a', after=('gate.b',)),
            _gate('gate.b', after=('gate.a',)),
        ]
        with self.assertRaises(GateCompilationError) as caught:
            compile_task_gates({}, contracts)
        self.assertIn('gate_dependency_cycle', [item.code for item in caught.exception.issues])

    def test_gate_configuration_schema_is_enforced(self):
        contract = _gate('configured', config={
            'type': 'object',
            'properties': {'limit': {'type': 'integer', 'minimum': 1}},
            'additionalProperties': False,
        })
        with self.assertRaises(GateCompilationError) as caught:
            compile_task_gates({
                'quality_gates': {
                    'gate_config': {'configured': {'limit': 0, 'unknown': True}},
                },
            }, [contract])
        codes = [item.code for item in caught.exception.issues]
        self.assertIn('invalid_gate_configuration', codes)
        self.assertIn('unknown_gate_configuration', codes)

    def test_runtime_validation_rejects_unknown_evaluator(self):
        configuration = compile_task_gates({}, [_gate('runtime')]).to_dict()
        issues = validate_gate_runtime_configuration(configuration, GateRuntimeRegistry())
        self.assertIn('unknown_gate_evaluator', [item.code for item in issues])
        runtime = GateRuntimeRegistry([
            GateRuntimeAdapter(
                'tests.runtime',
                lambda _context, _invocation: None,
                ('task.before_execute',),
            ),
        ])
        self.assertEqual(validate_gate_runtime_configuration(configuration, runtime), ())

    def test_gate_artifacts_include_configuration_provenance_and_trace(self):
        configuration = compile_task_gates({}, [_gate('artifact')]).to_dict()
        documents = gate_artifact_documents({
            'gate_configuration': configuration,
            'gate_decisions': [{'gate_id': 'artifact', 'status': 'passed'}],
            'gate_execution_trace': [{'gate_id': 'artifact', 'status': 'completed'}],
        })
        self.assertEqual(
            [item['kind'] for item in documents],
            ['gate-configuration', 'gate-provenance', 'gate-execution-trace'],
        )
        self.assertEqual(documents[-1]['payload']['decisions'][0]['gate_id'], 'artifact')

    def test_path_consistency_gate_is_deferred_until_analysis(self):
        report = {
            'name': 'path-study',
            'objective': 'scan a coordinate',
            'system_type': 'molecular',
            'status': 'succeeded',
            'comparison_table': [
                {'case_id': 'case-0001', 'status': 'succeeded'},
                {'case_id': 'case-0002', 'status': 'succeeded'},
            ],
            'scan_path_diagnostics': {
                'status': 'review_required',
                'summary': 'A local discontinuity was detected.',
                'anomalies': [{
                    'case_id': 'case-0002',
                    'left_case_id': 'case-0001',
                    'right_case_id': 'case-0002',
                }],
            },
            'adaptive': {},
        }
        deferred = build_report_workflow(report)
        self.assertEqual(deferred['stage'], 'completed')
        reviewed = build_report_workflow(report, analysis_requested=True)
        self.assertEqual(reviewed['stage'], 'path_consistency_review_required')
        self.assertEqual(reviewed['allowed_actions'][0]['id'], 'prepare_path_restart')
        self.assertEqual(reviewed['gate_decision']['gate_id'], 'study.path_consistency')
        self.assertTrue(report['gate_decisions'])
        self.assertNotIn('gate_decisions', report['adaptive'])

    def test_static_unconverged_dmet_requires_targeted_rerun(self):
        report = {
            'study_id': 'static-dmet-study',
            'name': 'static-dmet-study',
            'objective': 'scan an interaction parameter',
            'system_type': 'model_hamiltonian',
            'status': 'completed_with_issues',
            'comparison_table': [
                {'case_id': 'case-0001', 'status': 'succeeded', 'solver': 'dmet'},
                {'case_id': 'case-0002', 'status': 'unconverged', 'solver': 'dmet'},
            ],
        }

        workflow = build_report_workflow(report)

        self.assertEqual(workflow['stage'], 'recovery_review_required')
        self.assertEqual(workflow['active_plan_kind'], 'static')
        self.assertEqual(workflow['pending_case_ids'], ['case-0002'])
        self.assertEqual(workflow['current_case_ids'], ['case-0002'])
        self.assertEqual(
            [item['id'] for item in workflow['allowed_actions']],
            ['increase_dmet_iterations', 'acknowledge'],
        )
        self.assertIn('must be rerun', workflow['message'])

    def test_successful_case_with_failed_quality_is_not_complete(self):
        report = {
            'study_id': 'quality-study',
            'name': 'quality-study',
            'objective': 'verify result quality propagation',
            'system_type': 'model_hamiltonian',
            'status': 'completed_with_issues',
            'comparison_table': [{
                'case_id': 'case-0001',
                'status': 'succeeded',
                'solver': 'dmet',
                'quality_status': 'review_required',
                'publication_eligible': False,
            }],
        }

        workflow = build_report_workflow(report)

        self.assertEqual(workflow['stage'], 'recovery_review_required')
        self.assertEqual(workflow['pending_case_ids'], ['case-0001'])
        self.assertEqual(
            [item['id'] for item in workflow['allowed_actions']],
            ['show_case_guidance', 'acknowledge'],
        )

    def test_active_space_gate_exposes_cross_case_contract_changes(self):
        report = {
            'name': 'cas-path-study',
            'objective': 'scan a molecular coordinate',
            'system_type': 'molecular',
            'status': 'pending_review',
            'comparison_table': [],
            'adaptive': {
                'refined_plan': {
                    'cases': [
                        {
                            'case_id': 'case-0001',
                            'label': 'R=1.6',
                            'request': {
                                'method': 'casscf',
                                'active_space': {
                                    'enabled': True,
                                    'approved': False,
                                    'ncas': 13,
                                    'nelecas': 14,
                                    'orbital_indices': list(range(13)),
                                },
                            },
                        },
                        {
                            'case_id': 'case-0002',
                            'label': 'R=1.8',
                            'request': {
                                'method': 'casscf',
                                'active_space': {
                                    'enabled': True,
                                    'approved': False,
                                    'ncas': 12,
                                    'nelecas': 12,
                                    'orbital_indices': list(range(12)),
                                },
                            },
                        },
                    ],
                },
            },
        }

        workflow = build_report_workflow(report)
        continuity = workflow['gate_decision']['details']['active_space_continuity']

        self.assertEqual(workflow['stage'], 'active_space_review_required')
        self.assertFalse(continuity['consistent'])
        self.assertEqual(continuity['variant_count'], 2)
        checks = {
            item['id']: item
            for item in workflow['gate_decision']['checks']
        }
        self.assertEqual(checks['active_space_continuity']['status'], 'review_required')


if __name__ == '__main__':
    unittest.main()
