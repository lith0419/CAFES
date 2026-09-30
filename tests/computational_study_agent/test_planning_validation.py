from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path

from computational_study_agent.executor import run_study
from computational_study_agent.capabilities import default_study_capabilities
from computational_study_agent.planner import build_study_plan
from computational_study_agent.postprocessing import suggest_plot_specs
from tests.computational_study_agent.support import (
    StudyAgentTestCase,
    disconnected_bond_spec,
    hubbard_dimer_spec,
)
from computational_study_agent.validation import (
    has_errors,
    validate_study_plan,
    validate_study_spec,
    validate_study_workflow,
)
from pyscf_agent.backend.model_hamiltonian.solver import generate_model_hamiltonian_input_script


class StudyPlanningAndValidationTests(StudyAgentTestCase):
    def test_model_studies_reject_adaptive_mode(self):
        spec = {
            'name': 'adaptive-model-grid',
            'objective': 'Scan a model grid',
            'system_type': 'model_hamiltonian',
            'study_mode': 'adaptive',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'sweep': {'U': [2, 4]},
            'observables': ['energy'],
        }

        issues = validate_study_spec(spec)

        self.assertTrue(has_errors(issues))
        self.assertIn('unsupported_study_mode', [issue.code for issue in issues])
        with self.assertRaisesRegex(ValueError, 'static scans only'):
            build_study_plan(spec)

    def test_model_static_scan_keeps_explicit_solver(self):
        spec = {
            'name': 'static-model-grid',
            'objective': 'Scan a model grid',
            'system_type': 'model_hamiltonian',
            'study_mode': 'static',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'sweep': {'U': [2, 4]},
            'observables': ['energy'],
        }

        self.assertFalse(has_errors(validate_study_spec(spec)))
        plan = build_study_plan(spec)

        self.assertEqual(len(plan.cases), 2)
        self.assertTrue(all(case.request['solver'] == 'fci' for case in plan.cases))

    def test_model_dmet_solver_options_are_preserved_in_every_case(self):
        solver = {
            'name': 'dmet',
            'options': {
                'impurity_solver': 'block2_dmrg',
                'impurity_solver_options': {'preset': 'balanced'},
                'execution_mode': 'finite_graph',
                'fragment_definition': 'site_count',
                'impurity_size': 1,
                'reference': 'unrestricted',
            },
        }
        spec = {
            'name': 'dmet-grid',
            'objective': 'Compare self-consistent embedding across U',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': solver},
            'sweep': {'U': [2, 4]},
            'observables': ['energy'],
        }

        self.assertFalse(has_errors(validate_study_workflow(spec)))
        plan = build_study_plan(spec)

        self.assertEqual(len(plan.cases), 2)
        self.assertTrue(all(case.request['solver'] == solver for case in plan.cases))
        self.assertIsNot(plan.cases[0].request['solver'], plan.cases[1].request['solver'])

    def test_model_dmet_provider_contract_is_checked_before_execution(self):
        model_spec = hubbard_dimer_spec()
        model_spec['bonds'][0]['V'] = 0.5
        model_spec['bonds'][0]['effective_V'] = 0.5
        invalid_solver = {
            'name': 'dmet',
            'options': {
                'execution_mode': 'translational',
                'fragment_definition': 'primitive_cell',
            },
        }
        invalid_spec = {
            'name': 'invalid-translated-dmet',
            'objective': 'Reject an incompatible translated DMET plan',
            'system_type': 'model_hamiltonian',
            'base_model_spec': model_spec,
            'base_task': {'solver': invalid_solver},
            'sweep': {'U': [2, 4]},
            'observables': ['energy'],
        }

        issues = validate_study_workflow(invalid_spec)

        self.assertTrue(has_errors(issues))
        dmet_issues = [issue for issue in issues if issue.code == 'invalid_dmet_case_request']
        self.assertEqual(len(dmet_issues), 1)
        self.assertIn('no_intersite_interaction', dmet_issues[0].message)
        self.assertIn('Affects all 2 planned cases', dmet_issues[0].message)

        valid_spec = dict(invalid_spec)
        valid_spec['base_task'] = {
            'solver': {
                'name': 'dmet',
                'options': {
                    'execution_mode': 'finite_graph',
                    'fragment_definition': 'site_count',
                    'impurity_size': 1,
                },
            },
        }
        self.assertFalse(has_errors(validate_study_workflow(valid_spec)))

    def test_builds_molecular_method_comparison_plan(self):
        plan = build_study_plan({
            'name': 'h2-methods',
            'objective': 'compare_methods',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'outputs': ['energy'],
            },
            'sweep': {
                'method': ['hf', 'mp2'],
            },
            'observables': ['energy'],
        })

        self.assertEqual(plan.system_type, 'molecular')
        self.assertEqual(len(plan.cases), 2)
        self.assertEqual(plan.cases[0].request['method'], 'hf')
        self.assertEqual(plan.cases[1].request['method'], 'mp2')
        self.assertIn('molecular_methods', plan.capability_snapshot)
        self.assertIn('platform_catalog', plan.capability_snapshot)
        model_capabilities = {
            item['local_id']: item['contract']
            for item in plan.capability_snapshot['platform_catalog']['registry']['entries']
            if item['namespace'] == 'model_hamiltonian.model'
        }
        self.assertEqual(model_capabilities['holstein_hubbard']['status'], 'design_only')
        self.assertNotIn('holstein_hubbard', plan.capability_snapshot['model_hamiltonian_models'])

    def test_rejects_unsupported_model_parameter(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported model Hamiltonian sweep parameter'):
            build_study_plan({
                'name': 'bad-model-sweep',
                'objective': 'sweep',
                'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(),
                'sweep': {'J': [0.1, 0.2]},
            })

    def test_model_case_design_can_place_site_defects(self):
        plan = build_study_plan({
            'name': 'dimer-site-defects',
            'objective': 'Compare local site defects',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'case_design': {
                'mode': 'grid',
                'variables': {
                    'defect_site': [0, 1],
                    'defect_strength': [-0.5, 0.5],
                },
                'template': {
                    'operations': [
                        {
                            'op': 'add_site_defect',
                            'site': '$defect_site',
                            'epsilon_shift': '$defect_strength',
                        }
                    ]
                },
            },
            'observables': ['energy'],
        })

        self.assertEqual(len(plan.cases), 4)
        first_case = plan.cases[0]
        self.assertEqual(first_case.variables, {'defect_site': 0, 'defect_strength': -0.5})
        self.assertEqual(first_case.operations[0]['site'], 0)
        self.assertEqual(first_case.operations[0]['epsilon_shift'], -0.5)
        self.assertEqual(first_case.model_spec['sites'][0]['epsilon'], -0.5)
        self.assertEqual(first_case.model_spec['sites'][1]['epsilon'], 0)
        last_case = plan.cases[-1]
        self.assertEqual(last_case.model_spec['sites'][0]['epsilon'], 0)
        self.assertEqual(last_case.model_spec['sites'][1]['epsilon'], 0.5)
        self.assertIn('model_hamiltonian_operations', plan.capability_snapshot)

    def test_model_case_design_supports_selectors_and_request_updates(self):
        plan = build_study_plan({
            'name': 'dimer-boundary-defect',
            'objective': 'Apply a defect to all boundary sites and compare solvers',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'case_design': {
                'mode': 'cases',
                'template': {
                    'operations': [
                        {
                            'op': 'set_site_parameter',
                            'selector': {'kind': 'boundary_sites'},
                            'parameter': 'U',
                            'value': '$u_value',
                        },
                        {
                            'op': 'change_nelec',
                            'nelec': '$nelec',
                        },
                    ],
                    'request_updates': {
                        'solver': '$solver',
                    },
                },
                'cases': [
                    {'variables': {'u_value': 2.0, 'nelec': [1, 0], 'solver': 'fci'}},
                    {'variables': {'u_value': 3.0, 'nelec': [1, 1], 'solver': 'mp2'}},
                ],
            },
            'observables': ['energy'],
        })

        self.assertEqual(len(plan.cases), 2)
        self.assertEqual([site['U'] for site in plan.cases[0].model_spec['sites']], [2.0, 2.0])
        self.assertEqual(plan.cases[0].model_spec['nelec'], [1, 0])
        self.assertEqual(plan.cases[0].request['solver'], 'fci')
        self.assertEqual([site['U'] for site in plan.cases[1].model_spec['sites']], [3.0, 3.0])
        self.assertEqual(plan.cases[1].request['solver'], 'mp2')

    def test_model_case_design_with_input_file_inherits_unswept_model_defaults(self):
        script = generate_model_hamiltonian_input_script(hubbard_dimer_spec())
        with tempfile.TemporaryDirectory() as tmpdir:
            input_path = Path(tmpdir) / 'pyscf_model_hamiltonian_input.py'
            input_path.write_text(script, encoding='utf-8')
            plan = build_study_plan({
                'name': 'dimer-u-t-grid',
                'objective': 'Sweep U and t while inheriting the builder model',
                'system_type': 'model_hamiltonian',
                'base_model_input_file': str(input_path),
                'base_model_spec': {'model': 'hubbard'},
                'base_task': {'solver': 'mp2'},
                'case_design': {
                    'mode': 'grid',
                    'variables': {
                        'U_value': [0, 4],
                        't_value': [-2, -1],
                    },
                    'template': {
                        'operations': [
                            {
                                'op': 'set_site_parameter',
                                'selector': {'kind': 'all'},
                                'parameter': 'U',
                                'value': '$U_value',
                            },
                            {
                                'op': 'set_bond_parameter',
                                'selector': {'kind': 'all'},
                                'parameter': 't',
                                'value': '$t_value',
                            },
                        ],
                    },
                },
                'observables': ['energy'],
            })

        self.assertEqual(len(plan.cases), 4)
        first_case = plan.cases[0]
        self.assertEqual(first_case.model_spec['nelec'], [1, 1])
        self.assertEqual(len(first_case.model_spec['sites']), 2)
        self.assertEqual(len(first_case.model_spec['bonds']), 1)
        self.assertEqual([site['U'] for site in first_case.model_spec['sites']], [0, 0])
        self.assertEqual(first_case.model_spec['bonds'][0]['t'], -2)
        self.assertEqual(first_case.request['solver'], 'mp2')

    def test_model_total_electron_sweep_normalizes_to_spin_resolved_nelec(self):
        plan = build_study_plan({
            'name': 'dimer-electron-scan',
            'objective': 'Scan total electron count',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'sweep': {'nelec': [1, 2, 3, 4]},
            'observables': ['energy'],
        })

        self.assertEqual([case.model_spec['nelec'] for case in plan.cases], [
            [1, 0],
            [1, 1],
            [2, 1],
            [2, 2],
        ])

    def test_model_cases_pass_requested_observables_to_backend(self):
        plan = build_study_plan({
            'name': 'dimer-observable-routing',
            'objective': 'Route requested model observables',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'sweep': {'U': [4]},
            'observables': ['energy', 'strong_correlation_diagnostics'],
        })

        self.assertEqual(plan.cases[0].request['analysis']['outputs'], ['energy', 'strong_correlation_diagnostics'])

    def test_mean_double_occupancy_request_routes_to_diagnostics(self):
        for observables in (
            ['energy', 'mean_double_occupancy'],
            ['energy', 'mean_double_occupancy', 'strong_correlation_diagnostics', 'mean_double_occupancy'],
        ):
            with self.subTest(observables=observables):
                spec = {
                    'name': 'double-occupancy-request',
                    'objective': 'Request the derived double-occupancy metric',
                    'system_type': 'model_hamiltonian',
                    'base_model_spec': hubbard_dimer_spec(),
                    'base_task': {'solver': 'fci'},
                    'observables': observables,
                }
                issues = validate_study_workflow(spec)
                self.assertFalse(has_errors(issues), issues)
                plan = build_study_plan(spec)
                expected = ['energy', 'strong_correlation_diagnostics']
                self.assertEqual(plan.observables, expected)
                self.assertEqual(plan.cases[0].request['analysis']['outputs'], expected)
                self.assertEqual(spec['observables'], observables)
        capabilities = default_study_capabilities()
        self.assertIn('mean_double_occupancy', capabilities.model_hamiltonian_observables)
        self.assertFalse(capabilities.supports_model_observable('mean_double_occupancy_typo'))
        self.assertFalse(capabilities.supports_model_observable('double_occupancy'))
        self.assertFalse(capabilities.supports_molecular_observable('mean_double_occupancy'))

    def test_mean_double_occupancy_request_returns_analytic_dimer_value(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_study({
                'name': 'double-occupancy-dimer',
                'objective': 'Verify requested mean double occupancy against the exact dimer',
                'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(),
                'base_task': {'solver': 'fci'},
                'observables': ['energy', 'mean_double_occupancy'],
            }, work_dir=tmpdir, locale='en')
            self.assertEqual(report.status, 'succeeded')
            row = report.comparison_table[0]
            # Half-filled Hubbard dimer, U=4 and |t|=1: (dE/dU)/2.
            expected = (1.0 - 4.0 / math.sqrt(4.0 ** 2 + 16.0)) / 4.0
            self.assertAlmostEqual(row['mean_double_occupancy'], expected, places=8)
            self.assertIn('strong_correlation_diagnostics', row)

    def test_model_bond_operation_accepts_site_pair_selector(self):
        plan = build_study_plan({
            'name': 'site-pair-bond-selector',
            'objective': 'Set V only on the bond connecting site 2 and site 4',
            'system_type': 'model_hamiltonian',
            'base_model_spec': disconnected_bond_spec(),
            'base_task': {'solver': 'fci'},
            'case_design': {
                'mode': 'grid',
                'variables': {'v_value': [1, 3]},
                'template': {
                    'operations': [{
                        'op': 'set_bond_parameter',
                        'selector': {'kind': 'site_pair', 'sites': [2, 4]},
                        'parameter': 'V',
                        'value': '$v_value',
                    }],
                },
            },
            'observables': ['energy'],
        })

        self.assertEqual(len(plan.cases), 2)
        self.assertEqual(plan.cases[0].model_spec['bonds'][0]['V'], 0)
        self.assertEqual(plan.cases[0].model_spec['bonds'][1]['V'], 1)
        self.assertEqual(plan.cases[1].model_spec['bonds'][1]['effective_V'], 3)

    def test_model_bond_operation_accepts_source_target_pair(self):
        plan = build_study_plan({
            'name': 'source-target-bond-selector',
            'objective': 'Set V only on the bond connecting site 2 and site 4',
            'system_type': 'model_hamiltonian',
            'base_model_spec': disconnected_bond_spec(),
            'base_task': {'solver': 'fci'},
            'case_design': {
                'mode': 'cases',
                'cases': [{
                    'operations': [{
                        'op': 'set_bond_parameter',
                        'source': 2,
                        'target': 4,
                        'parameter': 'V',
                        'value': 2,
                    }],
                }],
            },
            'observables': ['energy'],
        })

        self.assertEqual(plan.cases[0].model_spec['bonds'][0]['V'], 0)
        self.assertEqual(plan.cases[0].model_spec['bonds'][1]['V'], 2)

    def test_change_nelec_operation_accepts_total_electron_scalar(self):
        plan = build_study_plan({
            'name': 'dimer-electron-cases',
            'objective': 'Scan total electron count',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'case_design': {
                'mode': 'grid',
                'variables': {'N': [1, 2, 3]},
                'template': {
                    'operations': [
                        {'op': 'change_nelec', 'nelec': '$N'},
                    ],
                },
            },
            'observables': ['energy'],
        })

        self.assertEqual([case.model_spec['nelec'] for case in plan.cases], [
            [1, 0],
            [1, 1],
            [2, 1],
        ])
        self.assertEqual([case.model_spec['spin_multiplicity'] for case in plan.cases], [
            2,
            1,
            2,
        ])

    def test_model_case_design_explicit_cases_can_define_operations(self):
        plan = build_study_plan({
            'name': 'explicit-model-cases',
            'objective': 'Use explicit operation lists for non-grid model cases',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'case_design': {
                'mode': 'cases',
                'template': {
                    'operations': [
                        {
                            'op': 'set_global_parameter',
                            'parameter': 'U',
                            'value': '$u_value',
                        }
                    ],
                },
                'cases': [
                    {
                        'label': 'left attractive defect',
                        'variables': {'u_value': 2.0},
                        'operations': [
                            {
                                'op': 'add_site_defect',
                                'site': 0,
                                'epsilon_shift': -1.0,
                            }
                        ],
                    },
                    {
                        'label': 'weakened bond',
                        'variables': {'u_value': 4.0},
                        'operations': [
                            {
                                'op': 'set_bond_parameter',
                                'bond': [0, 1],
                                'parameter': 't',
                                'value': -0.5,
                            }
                        ],
                    },
                ],
            },
            'observables': ['energy'],
        })

        self.assertEqual(len(plan.cases), 2)
        self.assertEqual(plan.cases[0].label, 'left attractive defect')
        self.assertEqual(plan.cases[0].model_spec['sites'][0]['U'], 2.0)
        self.assertEqual(plan.cases[0].model_spec['sites'][0]['epsilon'], -1.0)
        self.assertEqual(plan.cases[1].label, 'weakened bond')
        self.assertEqual(plan.cases[1].model_spec['sites'][0]['U'], 4.0)
        self.assertEqual(plan.cases[1].model_spec['bonds'][0]['t'], -0.5)
        self.assertEqual(plan.cases[1].model_spec['bonds'][0]['effective_t'], -0.5)

    def test_validation_accepts_valid_case_design_workflow(self):
        issues = validate_study_workflow({
            'name': 'valid-defect-workflow',
            'objective': 'Validate a local defect workflow',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'case_design': {
                'mode': 'grid',
                'variables': {'site_id': [0, 1]},
                'template': {
                    'operations': [
                        {
                            'op': 'add_site_defect',
                            'site': '$site_id',
                            'epsilon_shift': -0.5,
                        }
                    ],
                },
            },
            'observables': ['energy'],
        })

        self.assertFalse(has_errors(issues))

    def test_planner_canonicalizes_method_aliases_before_costing(self):
        plan = build_study_plan({
            'name': 'h2-ccsd-t-alias',
            'objective': 'normalize method aliases',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'CCSD(T)',
            },
            'observables': ['energy'],
        })

        self.assertEqual(plan.cases[0].request['method'], 'ccsd_t')
        self.assertEqual(plan.cost_estimate['cases'][0]['method'], 'ccsd_t')

        model_plan = build_study_plan({
            'name': 'dimer-ccsd-t-alias',
            'objective': 'normalize solver aliases',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'CCSD(T)'},
            'observables': ['energy'],
        })
        self.assertEqual(model_plan.cases[0].request['solver'], 'ccsd_t')
        self.assertEqual(model_plan.cost_estimate['cases'][0]['method'], 'ccsd_t')

    def test_static_cas_contract_is_preflight_validated(self):
        base_spec = {
            'name': 'h2-casscf-contract',
            'objective': 'validate CAS request contract',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'casscf',
            },
            'observables': ['energy'],
        }
        issues = validate_study_workflow(base_spec)
        codes = {issue.code for issue in issues}
        self.assertTrue({
            'missing_active_space_ncas',
            'missing_active_space_nelecas',
            'active_space_not_approved',
        }.issubset(codes))

        valid_spec = dict(base_spec)
        valid_spec['base_task'] = dict(base_spec['base_task'])
        valid_spec['base_task']['active_space'] = {
            'ncas': 2,
            'nelecas': 2,
            'approved': True,
        }
        self.assertFalse(has_errors(validate_study_workflow(valid_spec)))

    def test_sc_nevpt2_preflight_rejects_unrestricted_cas(self):
        issues = validate_study_workflow({
            'name': 'h2-unrestricted-nevpt2',
            'objective': 'validate post-CAS reference contract',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'casscf',
                'restricted': False,
                'active_space': {'ncas': 2, 'nelecas': 2, 'approved': True},
                'post_cas': {'sc_nevpt2': {'enabled': True, 'root': 0}},
            },
            'observables': ['energy'],
        })

        self.assertIn('sc_nevpt2_reference_not_supported', {issue.code for issue in issues})

    def test_validation_reports_invalid_case_design_before_planning(self):
        issues = validate_study_spec({
            'name': 'bad-operation',
            'objective': 'Reject an unsupported operation',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'case_design': {
                'mode': 'grid',
                'variables': {'site_id': [0]},
                'template': {
                    'operations': [
                        {
                            'op': 'invent_new_physics',
                            'site': '$site_id',
                        }
                    ],
                },
            },
        })

        self.assertTrue(has_errors(issues))
        self.assertTrue(any(issue.code == 'unsupported_model_operation' for issue in issues))

    def test_validation_uses_registry_for_model_observables(self):
        valid_issues = validate_study_spec({
            'name': 'valid-model-observables',
            'objective': 'Request executable model observables',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'observables': ['energy', 'strong_correlation_diagnostics'],
        })
        invalid_issues = validate_study_spec({
            'name': 'planned-model-observable',
            'objective': 'Reject planned observables until backend support exists',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'observables': ['energy', 'structure_factor'],
        })

        self.assertFalse(has_errors(valid_issues))
        self.assertTrue(has_errors(invalid_issues))
        self.assertTrue(any(issue.code == 'unsupported_model_observable' for issue in invalid_issues))

        internal_issues = validate_study_spec({
            'name': 'internal-model-observable',
            'objective': 'Reject diagnostics internals as standalone observables',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'observables': ['energy', 'density'],
        })
        self.assertTrue(has_errors(internal_issues))
        self.assertTrue(any(issue.code == 'unsupported_model_observable' for issue in internal_issues))

    def test_validation_reports_missing_site_selector_for_defect_operation(self):
        issues = validate_study_spec({
            'name': 'bad-defect-selector',
            'objective': 'Reject a defect operation without a target site',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'case_design': {
                'mode': 'grid',
                'variables': {'site_id': [0]},
                'template': {
                    'operations': [
                        {
                            'op': 'add_site_defect',
                            'epsilon_shift': -0.5,
                        }
                    ],
                },
            },
        })

        self.assertTrue(has_errors(issues))
        self.assertTrue(any(issue.code == 'missing_site_selector' for issue in issues))

    def test_validation_reports_invalid_expanded_plan(self):
        plan = build_study_plan({
            'name': 'bad-expanded-plan',
            'objective': 'Generate a plan and then corrupt it',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'sweep': {'U': [2]},
        })
        plan.cases[0].request['solver'] = 'made-up-solver'

        issues = validate_study_plan(plan)

        self.assertTrue(has_errors(issues))
        self.assertTrue(any(issue.code == 'unsupported_model_solver' for issue in issues))

    def test_molecular_case_design_uses_request_updates(self):
        plan = build_study_plan({
            'name': 'h2-case-design-methods',
            'objective': 'Compare selected molecular methods',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'outputs': ['energy'],
            },
            'case_design': {
                'mode': 'grid',
                'variables': {'method_name': ['hf', 'mp2']},
                'template': {
                    'request_updates': {'method': '$method_name'},
                },
            },
            'observables': ['energy'],
        })

        self.assertEqual(len(plan.cases), 2)
        self.assertEqual(plan.cases[0].request['method'], 'hf')
        self.assertEqual(plan.cases[1].request['method'], 'mp2')

    def test_case_design_override_changes_only_selected_grid_case(self):
        plan = build_study_plan({
            'name': 'h2-case-method-override',
            'objective': 'Change one point without replacing the scan',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'mp2',
            },
            'case_design': {
                'mode': 'grid',
                'variables': {'bond_factor': [1.0, 1.6, 2.0]},
                'template': {
                    'request_updates': {
                        'atom': 'H 0 0 0; H 0 0 $(bond_factor * 0.74)',
                    },
                },
                'overrides': [{
                    'selector': {'bond_factor': 1.6},
                    'request_updates': {'method': 'ccsd', 'xc': None},
                }],
            },
            'observables': ['energy'],
        })

        self.assertEqual([case.request['method'] for case in plan.cases], ['mp2', 'ccsd', 'mp2'])
        self.assertEqual([case.variables['bond_factor'] for case in plan.cases], [1.0, 1.6, 2.0])

    def test_molecular_case_design_interpolates_atom_coordinates(self):
        plan = build_study_plan({
            'name': 'h2-bond-interpolation',
            'objective': 'Scan H2 bond length',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'hf',
                'outputs': ['energy'],
            },
            'case_design': {
                'mode': 'grid',
                'variables': {'bond_length': [0.74, 1.20]},
                'template': {
                    'request_updates': {'atom': 'H 0 0 0; H 0 0 $bond_length'},
                },
            },
            'observables': ['energy'],
        })

        self.assertEqual(len(plan.cases), 2)
        self.assertEqual(plan.cases[0].request['atom'], 'H 0 0 0; H 0 0 0.74')
        self.assertEqual(plan.cases[1].request['atom'], 'H 0 0 0; H 0 0 1.2')
        self.assertNotIn('$bond_length', plan.cases[0].request['atom'])

    def test_molecular_study_outputs_final_energy_for_dissociation_plots(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_study({
                'name': 'h2-dissociation-energy',
                'objective': 'plot H2 dissociation energy',
                'system_type': 'molecular',
                'base_task': {
                    'atom': 'H 0 0 0; H 0 0 0.74',
                    'basis': 'sto-3g',
                    'method': 'hf',
                },
                'case_design': {
                    'mode': 'cases',
                    'cases': [
                        {
                            'label': 'R=0.74',
                            'variables': {'bond': 0.74},
                            'request_updates': {'atom': 'H 0 0 0; H 0 0 0.74'},
                        },
                        {
                            'label': 'R=1.20',
                            'variables': {'bond': 1.20},
                            'request_updates': {'atom': 'H 0 0 0; H 0 0 1.20'},
                        },
                    ],
                },
                'observables': ['energy'],
            }, work_dir=tmpdir, locale='en')

        self.assertEqual(report.status, 'succeeded')
        self.assertTrue(all('final_energy' in row for row in report.comparison_table))
        self.assertTrue(all(str(row['final_energy']).endswith(' Ha') for row in report.comparison_table))
        specs = suggest_plot_specs(report)
        self.assertTrue(specs)
        self.assertEqual(specs[0]['x'], 'bond')
        self.assertEqual(specs[0]['y'], 'final_energy')
        self.assertEqual(specs[0]['xlabel'], 'Bond length')
        self.assertEqual(specs[0]['units']['bond'], 'Angstrom')

    def test_runs_model_hamiltonian_u_sweep(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_study({
                'name': 'dimer-u-sweep',
                'objective': 'sweep_U',
                'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(),
                'base_task': {'solver': 'fci'},
                'sweep': {'U': [2, 4]},
                'observables': ['energy', 'energy_level_count'],
            }, work_dir=tmpdir, locale='en')

            self.assertEqual(report.status, 'succeeded')
            self.assertEqual(len(report.comparison_table), 2)
            self.assertEqual(report.comparison_table[0]['U'], '2 a.u.')
            self.assertEqual(report.comparison_table[1]['U'], '4 a.u.')
            self.assertTrue(all(row['status'] == 'succeeded' for row in report.comparison_table))
            self.assertNotIn('run_id', report.comparison_table[0])
            self.assertNotIn('energy_unit', report.comparison_table[0])
            self.assertTrue(str(report.comparison_table[0]['run_dir']).startswith('cases/case-0001'))
            self.assertTrue(str(report.comparison_table[0]['energy']).endswith(' a.u.'))
            for metric in (
                'mean_double_occupancy',
                'nearest_neighbor_spin_correlation',
                'nearest_neighbor_charge_correlation',
            ):
                self.assertTrue(all(metric in row for row in report.comparison_table))
            table_path = Path(report.work_dir) / 'comparison-table.tsv'
            self.assertTrue(table_path.exists())
            table_text = table_path.read_text(encoding='utf-8')
            self.assertIn('energy', table_text)
            self.assertIn('2 a.u.', table_text)
            self.assertIn(' a.u.', table_text)
            self.assertNotIn('energy_unit', table_text)
            self.assertNotIn('run_id', table_text)
            self.assertTrue((Path(report.work_dir) / 'study-report.json').exists())
            self.assertEqual(report.lifecycle['stage'], 'completed')
            self.assertTrue(report.lifecycle['terminal'])
            lifecycle_events = [item['event'] for item in report.lifecycle['history']]
            self.assertEqual(lifecycle_events[0], 'created')
            self.assertIn('execution_started', lifecycle_events)
            self.assertEqual(lifecycle_events[-1], 'execution_completed')
            for artifact in report.artifacts:
                self.assertTrue(
                    {'kind', 'path', 'size_bytes', 'mime_type', 'description'}
                    <= set(artifact)
                )
                self.assertNotIn('sha256', artifact)
                self.assertTrue(artifact['description'])
            for filename, schema in (
                ('study-plan.json', 'pyscf-agent.study-plan.v1'),
                ('cost-estimate.json', 'pyscf-agent.cost-estimate.v1'),
                ('study-state.json', 'pyscf-agent.study-state.v1'),
                ('study-report.json', 'pyscf-agent.study-report.v1'),
            ):
                payload = json.loads(
                    (Path(report.work_dir) / filename).read_text(encoding='utf-8')
                )
                self.assertEqual(payload['schema'], schema)
                if filename in {'study-state.json', 'study-report.json'}:
                    self.assertEqual(payload['lifecycle']['stage'], 'completed')

    def test_static_model_study_adds_default_strong_correlation_diagnostics(self):
        plan = build_study_plan({
            'name': 'static-model-default-diagnostics',
            'objective': 'Compare Hubbard interaction trends',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'fci'},
            'sweep': {'U': [2, 4]},
            'observables': ['energy'],
        })

        self.assertEqual(plan.observables, ['energy', 'strong_correlation_diagnostics'])
        self.assertTrue(all(
            case.request['analysis']['outputs'] == ['energy', 'strong_correlation_diagnostics']
            for case in plan.cases
        ))

    def test_runs_model_hamiltonian_strong_correlation_diagnostics(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_study({
                'name': 'dimer-observables',
                'objective': 'Collect strong-correlation diagnostics',
                'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(),
                'base_task': {'solver': 'fci'},
                'sweep': {'U': [4]},
                'observables': ['energy', 'strong_correlation_diagnostics'],
            }, work_dir=tmpdir, locale='en')

            self.assertEqual(report.status, 'succeeded')
            self.assertEqual(len(report.comparison_table), 1)
            row = report.comparison_table[0]
            self.assertTrue(str(row['energy']).endswith(' a.u.'))
            diagnostics = row['strong_correlation_diagnostics']
            self.assertEqual(diagnostics['kind'], 'strong_correlation_diagnostics')
            self.assertIn(diagnostics['level'], ('weak', 'moderate', 'strong'))
            for key in ('gap', 'density', 'double_occupancy', 'spin_correlation', 'charge_correlation', 'natural_occupations'):
                self.assertTrue(diagnostics['internal_quantities'][key])
            self.assertFalse(diagnostics['internal_quantities']['mean_field_gap'])
            self.assertFalse(diagnostics['internal_quantities']['max_double_excitation_amplitude'])
            diagnostic_names = {item['name'] for item in diagnostics['diagnostics']}
            self.assertIn('natural_orbital_occupations', diagnostic_names)
            self.assertIn('many_body_gap', diagnostic_names)
            self.assertIn('double_occupancy_suppression', diagnostic_names)
            self.assertIn('nearest_neighbor_spin_correlation', diagnostic_names)
            self.assertIn('nearest_neighbor_charge_correlation', diagnostic_names)
            self.assertIn('gap', row)
            self.assertTrue(str(row['gap']).endswith(' a.u.'))
            self.assertNotIn('correlation_energy', row)
            self.assertIn('natural_occupation_fractionality', row)
            self.assertIn('mean_double_occupancy', row)
            self.assertIn('nearest_neighbor_spin_correlation', row)
            self.assertIn('nearest_neighbor_charge_correlation', row)
            self.assertIsInstance(row['mean_double_occupancy'], (int, float))
            self.assertIsInstance(row['nearest_neighbor_spin_correlation'], (int, float))
            self.assertIsInstance(row['nearest_neighbor_charge_correlation'], (int, float))
            self.assertNotIn('density', row)
            self.assertNotIn('double_occupancy', row)
            self.assertNotIn('spin_correlation', row)
            self.assertNotIn('charge_correlation', row)
            artifact_kinds = {item['kind'] for item in report.artifacts}
            self.assertNotIn('observable-density', artifact_kinds)
            self.assertNotIn('observable-double_occupancy', artifact_kinds)
            self.assertNotIn('observable-spin_correlation', artifact_kinds)
            self.assertNotIn('observable-charge_correlation', artifact_kinds)
