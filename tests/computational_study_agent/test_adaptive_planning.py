from __future__ import annotations

import copy

from computational_study_agent.planner import build_study_plan
from computational_study_agent.adaptive.executor import (
    _active_space_contract_from_case_report,
    analyze_initial_scan_report,
    build_adaptive_initial_scan_plan,
    build_refined_plan_from_decisions,
)
from computational_study_agent.adaptive.refinement import (
    build_casscf_active_space_probe_plan,
    build_casscf_active_space_review_plan,
    build_casscf_active_space_review_from_probe,
)
from computational_study_agent.adaptive.study_active_space import (
    resolve_study_active_space_contracts,
)
from tests.computational_study_agent.support import StudyAgentTestCase, hubbard_dimer_spec
from computational_study_agent.transformations import resolve_templates


class AdaptivePlanningTests(StudyAgentTestCase):
    def test_study_resolution_preserves_manual_and_approved_choices(self):
        alternative = {
            'method': 'chemical_valence', 'status': 'available',
            'ncas': 2, 'estimated_nelecas': 2, 'orbital_indices': [2, 3],
            'target_summary': {'primary_targets': ['H 1s'], 'primary_orbital_count': 2, 'primary_electron_count': 2},
        }
        for origin in ('manual', 'plain_manual', 'approved_uno'):
            with self.subTest(origin=origin):
                contract = {
                    'selection_method': 'manual', 'ncas': 2, 'nelecas': 2,
                    'orbital_indices': [0, 1], 'initial_mo_coeff': [[1., 0.], [0., 1.]],
                    'candidate_options': [copy.deepcopy(alternative)],
                }
                if origin != 'plain_manual':
                    contract['candidate_provenance'] = {'method': 'uno' if origin == 'approved_uno' else 'manual'}
                if origin == 'approved_uno':
                    contract['approved'] = True
                original = copy.deepcopy(contract)
                resolved, policy = resolve_study_active_space_contracts([
                    {'case_id': 'one', 'active_space_contract': contract},
                ])
                chosen = resolved[0]['active_space_contract']
                self.assertEqual(chosen['orbital_indices'], [0, 1])
                self.assertEqual(chosen['initial_mo_coeff'], original['initial_mo_coeff'])
                self.assertEqual(policy['candidate_method'], 'uno' if origin == 'approved_uno' else 'manual')
                self.assertEqual(contract, original)

    def test_manual_constraints_outweigh_automatic_cross_case_majority(self):
        target = {'primary_targets': ['H 1s']}
        manual = {
            'selection_method': 'manual', 'ncas': 2, 'nelecas': 2, 'orbital_indices': [0, 1],
            'candidate_provenance': {'method': 'manual', 'target_summary': target},
        }
        automatic = {
            'selection_method': 'manual', 'ncas': 3, 'nelecas': 2, 'orbital_indices': [0, 1, 2],
            'candidate_provenance': {'method': 'chemical_valence', 'target_summary': target},
            'candidate_options': [{
                'method': 'avas', 'ncas': 2, 'estimated_nelecas': 2,
                'orbital_indices': [1, 2], 'target_summary': target,
            }],
        }
        resolved, policy = resolve_study_active_space_contracts([
            {'case_id': 'manual', 'active_space_contract': manual},
            *[{'case_id': f'auto-{index}', 'active_space_contract': automatic} for index in range(3)],
        ])
        self.assertEqual(policy['status'], 'available')
        self.assertEqual(policy['ncas'], 2)
        self.assertEqual(policy['candidate_method'], 'manual')
        self.assertEqual(resolved[0]['active_space_contract']['orbital_indices'], [0, 1])
        self.assertTrue(all(item['active_space_contract']['orbital_indices'] == [1, 2] for item in resolved[1:]))

    def test_conflicting_manual_spaces_require_choice_review(self):
        contracts = [{
            'selection_method': 'manual', 'ncas': ncas, 'nelecas': 2,
            'orbital_indices': list(range(ncas)),
            'candidate_options': [{
                'method': 'chemical_valence', 'ncas': 2, 'estimated_nelecas': 2,
                'orbital_indices': [3, 4],
            }],
        } for ncas in (2, 3)]
        decisions = [{'case_id': f'case-{index}', 'active_space_contract': contract} for index, contract in enumerate(contracts)]
        resolved, policy = resolve_study_active_space_contracts(decisions)
        self.assertEqual(policy['status'], 'conflict')
        conflict = next(item for item in resolved if item['study_active_space_status'] == 'conflict')
        self.assertEqual(conflict['case_active_space_contract'], contracts[1])
        self.assertIsNone(conflict['active_space_contract'])
        self.assertIn('active_space_choice_review_required', conflict['tags'])
        self.assertNotIn('active_space_probe_required', conflict['tags'])

    def test_manual_audit_selection_survives_report_extraction_and_study_resolution(self):
        report = {'task_report': {'structured_results': {'active_space': {
            'ncas': 2, 'nelecas': 2, 'orbital_indices': [0, 1],
            'audit': {
                'selected_candidate_method': 'manual',
                'candidate_active_spaces': [
                    {'method': method, 'status': 'available', 'ncas': 2,
                     'estimated_nelecas': 2, 'orbital_indices': indices}
                    for method, indices in [('manual', [0, 1]), ('chemical_valence', [2, 3])]
                ],
            },
        }}}}
        contract = _active_space_contract_from_case_report(report)
        resolved, _ = resolve_study_active_space_contracts([{'case_id': 'one', 'active_space_contract': contract}])
        self.assertEqual(resolved[0]['active_space_contract']['orbital_indices'], [0, 1])

    def test_adaptive_options_normalization_preserves_saved_request_identity(self):
        from computational_study_agent.adaptive.initial_scan import normalize_adaptive_options
        for value in (None, {}, {'active_space_orbital_processing': {}},
                      {'active_space_orbital_processing': {'localization_method': 'boys'}}):
            once = normalize_adaptive_options(value)
            self.assertEqual(normalize_adaptive_options(once), once)

    def test_study_active_space_uses_one_chemical_dimension_with_local_mappings(self):
        target_summary = {
            'primary_targets': ['Cr 3d', 'Cr 4s'],
            'primary_orbital_count': 12,
            'primary_electron_count': 12,
        }

        def decision(case_id, ncas, nelecas, indices):
            return {
                'case_id': case_id,
                'recommended_method': 'casscf',
                'active_space_contract': {
                    'enabled': True,
                    'selection_method': 'manual',
                    'ncas': ncas,
                    'nelecas': nelecas,
                    'orbital_indices': indices,
                    'candidate_provenance': {
                        'method': 'chemical_valence',
                        'target_summary': target_summary,
                    },
                },
            }

        decisions, policy = resolve_study_active_space_contracts([
            decision('case-0001', 12, 12, list(range(10, 22))),
            decision('case-0002', 12, 12, list(range(11, 23))),
            decision('case-0003', 13, 14, list(range(9, 22))),
            {'case_id': 'case-0004', 'recommended_method': 'casscf'},
        ])

        self.assertEqual(policy['candidate_method'], 'chemical_valence')
        self.assertEqual(policy['ncas'], 12)
        self.assertEqual(policy['nelecas'], 12)
        self.assertEqual(policy['status'], 'conflict')
        self.assertEqual(policy['conflicting_case_ids'], ['case-0003'])
        self.assertEqual(policy['missing_case_ids'], ['case-0004'])
        self.assertEqual(decisions[0]['active_space_contract']['orbital_indices'], list(range(10, 22)))
        self.assertEqual(decisions[1]['active_space_contract']['orbital_indices'], list(range(11, 23)))
        self.assertIsNone(decisions[2]['active_space_contract'])
        self.assertIn('active_space_probe_required', decisions[2]['tags'])

    def test_study_active_space_prefers_chemical_candidate_over_broad_selected_space(self):
        decisions, policy = resolve_study_active_space_contracts([{
            'case_id': 'case-0001',
            'recommended_method': 'casscf',
            'active_space_contract': {
                'enabled': True,
                'selection_method': 'manual',
                'ncas': 14,
                'nelecas': 14,
                'orbital_indices': list(range(14)),
                'audit_summary': {'selected_candidate_method': 'merged'},
                'candidate_provenance': {'method': 'merged'},
                'candidate_options': [{
                    'method': 'chemical_valence',
                    'status': 'available',
                    'ncas': 12,
                    'estimated_nelecas': 12,
                    'orbital_indices': list(range(1, 13)),
                    'target_summary': {
                        'primary_targets': ['Cr 3d', 'Cr 4s'],
                        'primary_orbital_count': 12,
                        'primary_electron_count': 12,
                    },
                }],
            },
        }])

        self.assertEqual(policy['candidate_method'], 'chemical_valence')
        self.assertEqual(policy['ncas'], 12)
        self.assertEqual(policy['nelecas'], 12)
        self.assertEqual(decisions[0]['active_space_contract']['orbital_indices'], list(range(1, 13)))

    def test_study_active_space_merges_selection_methods_for_one_physical_target(self):
        target_summary = {
            'primary_targets': ['metal 3d', 'ligand 2p'],
            'primary_orbital_count': 8,
            'primary_electron_count': 10,
        }

        def decision(case_id, method, indices):
            return {
                'case_id': case_id,
                'recommended_method': 'casscf',
                'active_space_contract': {
                    'enabled': True,
                    'selection_method': 'manual',
                    'ncas': 8,
                    'nelecas': 10,
                    'orbital_indices': indices,
                    'candidate_provenance': {
                        'method': method,
                        'target_summary': target_summary,
                    },
                },
            }

        decisions, policy = resolve_study_active_space_contracts([
            decision('case-0001', 'avas', list(range(4, 12))),
            decision('case-0002', 'chemical_valence', list(range(5, 13))),
        ])

        self.assertEqual(policy['status'], 'available')
        self.assertEqual(policy['family_kind'], 'chemical_target')
        self.assertEqual(policy['candidate_methods'], ['chemical_valence', 'avas'])
        self.assertEqual(policy['supporting_case_ids'], ['case-0001', 'case-0002'])
        self.assertTrue(all(item['study_active_space_status'] == 'compatible' for item in decisions))

    def test_study_active_space_prefers_cross_case_support_over_method_priority(self):
        def decision(case_id, method, target, offset):
            return {
                'case_id': case_id,
                'recommended_method': 'casscf',
                'active_space_contract': {
                    'enabled': True,
                    'selection_method': 'manual',
                    'ncas': 6,
                    'nelecas': 6,
                    'orbital_indices': list(range(offset, offset + 6)),
                    'candidate_provenance': {
                        'method': method,
                        'target_summary': {
                            'primary_targets': [target],
                            'primary_orbital_count': 6,
                            'primary_electron_count': 6,
                        },
                    },
                },
            }

        decisions, policy = resolve_study_active_space_contracts([
            decision('case-0001', 'avas', 'frontier manifold', 2),
            decision('case-0002', 'avas', 'frontier manifold', 3),
            decision('case-0003', 'chemical_valence', 'isolated local shell', 4),
        ])

        self.assertEqual(policy['candidate_method'], 'avas')
        self.assertEqual(policy['target_signature'], ['frontier manifold'])
        self.assertEqual(policy['supporting_case_ids'], ['case-0001', 'case-0002'])
        self.assertEqual(policy['conflicting_case_ids'], ['case-0003'])
        self.assertIsNone(decisions[2]['active_space_contract'])

    def test_model_adaptive_refinement_attaches_configured_dmet_options(self):
        spec = {
            'name': 'dmet-adaptive',
            'objective': 'Use DMET in the strong-correlation region',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'dmet'},
            'sweep': {'U': [4]},
            'observables': ['energy'],
        }
        decisions = [{
            'case_id': 'case-0001',
            'initial_scan_status': 'succeeded',
            'level': 'strong',
            'recommended_solver': 'dmet',
            'tags': ['strong'],
        }]
        options = {
            'method_policy': {'strong_small': 'dmet', 'strong_large': 'dmet'},
            'model_solver_options': {
                'dmet': {
                    'impurity_solver': 'fci',
                    'execution_mode': 'translational',
                    'fragment_definition': 'primitive_cell',
                },
            },
        }

        payload = build_refined_plan_from_decisions(spec, decisions, options)
        request = payload['refined_plan']['cases'][0]['request']

        self.assertEqual(request['solver']['name'], 'dmet')
        self.assertEqual(request['solver']['options'], options['model_solver_options']['dmet'])

    def test_direct_casscf_with_explicit_candidate_goes_straight_to_review(self):
        review = build_casscf_active_space_review_plan({
            'name': 'explicit-cas',
            'objective': 'run the user-selected active space',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'casscf',
                'active_space': {
                    'enabled': True,
                    'ncas': 2,
                    'nelecas': 2,
                    'orbital_indices': [0, 1],
                    'approved': False,
                },
            },
            'observables': ['energy'],
        })

        self.assertIsNotNone(review)
        case = review['active_space_review_plan']['cases'][0]
        self.assertEqual(case['request']['active_space']['ncas'], 2)
        self.assertFalse(case['request']['active_space']['approved'])

    def test_direct_casscf_scan_probes_each_case_before_active_space_review(self):
        probe = build_casscf_active_space_probe_plan({
            'name': 'h2-dmrg-casscf-scan',
            'objective': 'scan H2 with DMRG-CASSCF',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'casscf',
                'solver': {'name': 'block2_dmrg', 'options': {'preset': 'balanced'}},
            },
            'case_design': {
                'mode': 'grid',
                'variables': {'bond_length': [0.74, 1.20]},
                'template': {
                    'request_updates': {'atom': 'H 0 0 0; H 0 0 $bond_length'},
                },
            },
            'observables': ['energy'],
        }, {'initial_scan_strategy': 'auto'})

        plan = probe['active_space_probe_plan']
        self.assertEqual(len(plan['cases']), 2)
        self.assertTrue(all(case['request']['method'] == 'hf' for case in plan['cases']))
        self.assertTrue(all('solver' not in case['request'] for case in plan['cases']))
        self.assertTrue(all(
            case['request']['active_space']['target_solver'] == 'block2_dmrg'
            for case in plan['cases']
        ))
        self.assertTrue(all(
            case['request']['active_space']['ncas'] is None
            for case in plan['cases']
        ))

        report_cases = []
        for case, ncas in zip(plan['cases'], (2, 2)):
            report_cases.append({
                'case_id': case['case_id'],
                'task_report': {
                    'structured_results': {
                        'active_space': {
                            'ncas': ncas,
                            'nelecas': ncas,
                            'orbital_indices': list(range(ncas)),
                            'audit_summary': {'ncas_nelecas_consistent': True},
                        },
                    },
                },
            })
        review = build_casscf_active_space_review_from_probe(
            plan,
            {'cases': report_cases},
        )

        requests = [case['request'] for case in review['active_space_review_plan']['cases']]
        self.assertEqual([request['active_space']['ncas'] for request in requests], [2, 2])
        self.assertEqual(review['study_active_space_policy']['ncas'], 2)
        self.assertTrue(all(request['method'] == 'casscf' for request in requests))
        self.assertTrue(all(request['solver']['name'] == 'block2_dmrg' for request in requests))
        self.assertTrue(all(not request['active_space']['approved'] for request in requests))

    def test_cr2_probe_results_share_chemical_valence_space_across_scan(self):
        bond_lengths = [round(1.2 + 0.2 * index, 1) for index in range(11)]
        probe = build_casscf_active_space_probe_plan({
            'name': 'cr2-dmrg-casscf-scan',
            'objective': 'scan Cr2 with DMRG-CASSCF',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'Cr 0 0 0; Cr $bond_length 0 0',
                'basis': 'def2-tzvp',
                'method': 'casscf',
                'solver': {'name': 'block2_dmrg', 'options': {'preset': 'balanced'}},
            },
            'case_design': {
                'mode': 'grid',
                'variables': {'bond_length': bond_lengths},
            },
            'observables': ['energy'],
        })
        plan = probe['active_space_probe_plan']
        target_summary = {
            'primary_targets': ['Cr 3d', 'Cr 4s'],
            'primary_orbital_count': 12,
            'primary_electron_count': 12,
        }
        report_cases = []
        for offset, case in enumerate(plan['cases']):
            chemical_indices = list(range(20 + offset, 32 + offset))
            broad_indices = list(range(19 + offset, 33 + offset))
            report_cases.append({
                'case_id': case['case_id'],
                'task_report': {
                    'structured_results': {
                        'active_space': {
                            'ncas': 14,
                            'nelecas': 13,
                            'orbital_indices': broad_indices,
                            'audit_summary': {'selected_candidate_method': 'merged'},
                            'audit': {
                                'selected_candidate_method': 'merged',
                                'candidate_active_spaces': [
                                    {
                                        'method': 'chemical_valence',
                                        'status': 'available',
                                        'ncas': 12,
                                        'estimated_nelecas': 12,
                                        'orbital_indices': chemical_indices,
                                        'target_summary': target_summary,
                                    },
                                    {
                                        'method': 'merged',
                                        'status': 'available',
                                        'ncas': 14,
                                        'estimated_nelecas': 13,
                                        'orbital_indices': broad_indices,
                                    },
                                ],
                            },
                        },
                    },
                },
            })

        review = build_casscf_active_space_review_from_probe(
            plan,
            {'cases': report_cases},
        )

        policy = review['study_active_space_policy']
        self.assertEqual(policy['candidate_method'], 'chemical_valence')
        self.assertEqual(policy['ncas'], 12)
        self.assertEqual(policy['nelecas'], 12)
        review_cases = review['active_space_review_plan']['cases']
        self.assertEqual(len(review_cases), len(bond_lengths))
        for offset, case in enumerate(review_cases):
            request = case['request']
            self.assertEqual(request['method'], 'casscf')
            self.assertEqual(request['solver']['name'], 'block2_dmrg')
            self.assertEqual(request['active_space']['ncas'], 12)
            self.assertEqual(request['active_space']['nelecas'], 12)
            self.assertEqual(
                request['active_space']['orbital_indices'],
                list(range(20 + offset, 32 + offset)),
            )
            self.assertFalse(request['active_space']['approved'])

    def test_direct_casscf_mixed_candidates_probe_only_missing_cases_then_merge_review(self):
        spec = {
            'name': 'mixed-active-spaces',
            'objective': 'preserve explicit evidence and probe missing evidence',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'casscf',
            },
            'case_design': {
                'mode': 'cases',
                'cases': [
                    {
                        'label': 'explicit',
                        'variables': {'bond_length': 0.74},
                        'request_updates': {
                            'active_space': {
                                'enabled': True,
                                'ncas': 2,
                                'nelecas': 2,
                                'orbital_indices': [0, 1],
                                'approved': False,
                            },
                        },
                    },
                    {
                        'label': 'probe',
                        'variables': {'bond_length': 1.20},
                        'request_updates': {'atom': 'H 0 0 0; H 0 0 1.20'},
                    },
                ],
            },
            'observables': ['energy'],
        }

        self.assertIsNone(build_casscf_active_space_review_plan(spec))
        probe = build_casscf_active_space_probe_plan(spec)
        plan = probe['active_space_probe_plan']
        self.assertEqual(len(plan['cases']), 1)
        self.assertEqual(plan['cases'][0]['label'], 'probe')
        report = {
            'cases': [{
                'case_id': plan['cases'][0]['case_id'],
                'task_report': {
                    'structured_results': {
                        'active_space': {
                            'ncas': 4,
                            'nelecas': 4,
                            'orbital_indices': [0, 1, 2, 3],
                        },
                    },
                },
            }],
        }
        with self.assertRaisesRegex(ValueError, 'Review the preserved choices'):
            build_casscf_active_space_review_from_probe(plan, report)

    def test_direct_casscf_probe_does_not_publish_partial_review(self):
        probe = build_casscf_active_space_probe_plan({
            'name': 'incomplete-probe',
            'objective': 'require evidence for every point',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'casscf',
            },
            'sweep': {'spin': [0, 2]},
            'observables': ['energy'],
        })
        plan = probe['active_space_probe_plan']
        successful_case = plan['cases'][0]
        report = {
            'cases': [{
                'case_id': successful_case['case_id'],
                'task_report': {
                    'structured_results': {
                        'active_space': {
                            'ncas': 2,
                            'nelecas': 2,
                            'orbital_indices': [0, 1],
                        },
                    },
                },
            }],
        }

        self.assertIsNone(build_casscf_active_space_review_from_probe(plan, report))

    def test_internal_adaptive_initial_scan_plan_keeps_full_model_grid(self):
        payload = build_adaptive_initial_scan_plan({
            'name': 'adaptive-u-sweep',
            'objective': 'initial scan U before refined method choice',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'ccsd'},
            'sweep': {'U': [0, 2, 4, 6, 8]},
            'observables': ['energy'],
        }, {'initial_scan_strategy': 'auto'})

        plan = payload['initial_scan_plan']
        self.assertEqual(plan['study_id'], 'initial-scan')
        self.assertEqual(len(plan['cases']), 5)
        self.assertEqual([case['variables']['U'] for case in plan['cases']], [0, 2, 4, 6, 8])
        self.assertEqual(payload['initial_scan_method'], 'mp2')
        self.assertTrue(all(case['request']['solver'] == 'mp2' for case in plan['cases']))
        self.assertEqual(plan['observables'], ['energy', 'strong_correlation_diagnostics'])

    def test_internal_model_initial_scan_helper_can_select_fci(self):
        payload = build_adaptive_initial_scan_plan({
            'name': 'adaptive-u-fci-initial-scan',
            'objective': 'exact initial scan U before refined method choice',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'mp2'},
            'sweep': {'U': [0, 4, 8]},
            'observables': ['energy'],
        }, {'initial_scan_strategy': 'fci'})

        self.assertEqual(payload['initial_scan_method'], 'fci')
        self.assertTrue(all(
            case['request']['solver'] == 'fci'
            for case in payload['initial_scan_plan']['cases']
        ))

    def test_removed_composite_initial_scan_strategy_is_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported initial_scan_strategy'):
            build_adaptive_initial_scan_plan({
                'name': 'adaptive-u-invalid-composite',
                'objective': 'reject an obsolete composite initial scan',
                'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(),
                'base_task': {'solver': 'mp2'},
                'sweep': {'U': [0, 4, 8]},
                'observables': ['energy'],
            }, {'initial_scan_strategy': 'mp2_ccsd'})

    def test_obsolete_screening_options_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'Unknown adaptive options'):
            build_adaptive_initial_scan_plan({
                'name': 'adaptive-u-sweep',
                'objective': 'initial scan U before refined method choice',
                'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(),
                'base_task': {'solver': 'fci'},
                'sweep': {'U': [0, 8]},
                'observables': ['energy'],
            }, {'screening_solver': 'fci'})

    def test_molecular_auto_initial_scan_uses_shared_hf_then_mp2_probe(self):
        payload = build_adaptive_initial_scan_plan({
            'name': 'adaptive-h2-initial-scan',
            'objective': 'initial scan molecular correlation risk',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'hf',
                'outputs': ['energy'],
            },
            'observables': ['energy'],
        }, {})

        plan = payload['initial_scan_plan']
        self.assertEqual(plan['system_type'], 'molecular')
        self.assertEqual(plan['observables'], ['energy', 'homo_lumo', 'dipole'])
        request = plan['cases'][0]['request']
        self.assertEqual(payload['initial_scan_strategy'], 'auto')
        self.assertEqual(payload['initial_scan_method'], 'hf')
        self.assertEqual(request['method'], 'hf')
        self.assertIsNone(request['xc'])
        self.assertFalse(request['restricted'])
        self.assertEqual(request['analysis']['outputs'], ['energy', 'homo_lumo', 'dipole'])
        self.assertEqual(request['active_space']['selection_method'], 'occupation_window')
        self.assertEqual(request['active_space']['occupation_window'], [0.02, 1.98])
        self.assertEqual(request['active_space']['target_method'], 'casscf')
        self.assertEqual(request['active_space']['target_solver'], 'fci')
        self.assertFalse(request['active_space']['approved'])
        self.assertFalse(request['post_cas']['sc_nevpt2']['enabled'])
        self.assertEqual(
            request['workflow']['module_config']['molecular.active_space_probe'],
            {
                'requested_strategy': 'auto',
                'probe_method': 'hf',
                'occupation_window': [0.02, 1.98],
                'refinement_method': 'mp2',
            },
        )
        self.assertEqual(
            request['workflow']['module_config']['molecular.active_space_probe_refinement'],
            {'initial_method': 'hf', 'refinement_method': 'mp2'},
        )

    def test_molecular_adaptive_initial_scan_can_use_fci(self):
        payload = build_adaptive_initial_scan_plan({
            'name': 'adaptive-h2-fci-initial-scan',
            'objective': 'exact molecular initial scan',
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
        }, {'initial_scan_strategy': 'fci'})

        cases = payload['initial_scan_plan']['cases']
        self.assertEqual(payload['adaptive_options']['initial_scan_strategy'], 'fci')
        self.assertEqual([case['request']['method'] for case in cases], ['fci', 'fci'])
        self.assertTrue(all(
            case['request']['workflow']['module_config']['molecular.active_space_probe']['requested_strategy'] == 'fci'
            for case in cases
        ))
        self.assertTrue(all(case['request']['xc'] is None for case in cases))
        self.assertEqual([case['request']['atom'] for case in cases], [
            'H 0 0 0; H 0 0 0.74',
            'H 0 0 0; H 0 0 1.2',
        ])

    def test_molecular_initial_scan_preserves_explicit_block2_target_contract(self):
        payload = build_adaptive_initial_scan_plan({
            'name': 'adaptive-block2-target',
            'objective': 'probe before DMRG-CASSCF',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 1.2',
                'basis': 'sto-3g',
                'method': 'casscf',
                'solver': {'name': 'block2_dmrg', 'options': {'preset': 'balanced'}},
                'orbital_processing': {
                    'enabled': True,
                    'localization_method': 'pipek_mezey',
                    'localization_scope': 'active_space',
                    'orbital_ordering': 'fiedler',
                },
            },
            'observables': ['energy'],
        }, {
            'initial_scan_strategy': 'auto',
            'active_space_solver': 'block2_dmrg',
            'block2_dmrg_options': {'preset': 'balanced'},
            'active_space_orbital_processing': {
                'enabled': True,
                'localization_method': 'pipek_mezey',
                'orbital_ordering': 'fiedler',
            },
        })

        request = payload['initial_scan_plan']['cases'][0]['request']
        self.assertEqual(request['method'], 'hf')
        self.assertNotIn('solver', request)
        self.assertEqual(request['active_space']['target_solver'], 'block2_dmrg')
        self.assertEqual(
            request['active_space']['target_solver_options'],
            {'preset': 'balanced'},
        )
        self.assertEqual(request['orbital_processing']['orbital_ordering'], 'fiedler')

    def test_molecular_adaptive_initial_scan_interpolates_atom_templates(self):
        payload = build_adaptive_initial_scan_plan({
            'name': 'adaptive-h2-bond-template',
            'objective': 'initial scan H2 bond stretch with templated coordinates',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'hf',
                'outputs': ['energy'],
            },
            'case_design': {
                'mode': 'grid',
                'variables': {'bond_length': [0.20, 0.74, 1.40, 2.20, 3.00]},
                'template': {
                    'request_updates': {'atom': 'H 0 0 0; H 0 0 $bond_length'},
                },
            },
            'observables': ['energy'],
        }, {})

        cases = payload['initial_scan_plan']['cases']
        self.assertEqual([case['variables']['bond_length'] for case in cases], [0.20, 0.74, 1.40, 2.20, 3.00])
        self.assertEqual([case['request']['atom'] for case in cases], [
            'H 0 0 0; H 0 0 0.2',
            'H 0 0 0; H 0 0 0.74',
            'H 0 0 0; H 0 0 1.4',
            'H 0 0 0; H 0 0 2.2',
            'H 0 0 0; H 0 0 3.0',
        ])
        self.assertTrue(all(case['request']['method'] == 'hf' for case in cases))
        self.assertTrue(all('$bond_length' not in case['request']['atom'] for case in cases))

    def test_case_templates_support_safe_numeric_expressions(self):
        resolved = resolve_templates({
            'bond_length': '$(bond_factor * 1.09768)',
            'atom': 'N 0 0 0; N 0 0 $(bond_factor * 1.09768)',
            'offset': '$(bond_factor - 0.2)',
        }, {'bond_factor': 2.0})

        self.assertAlmostEqual(resolved['bond_length'], 2.19536)
        self.assertEqual(resolved['atom'], 'N 0 0 0; N 0 0 2.19536')
        self.assertAlmostEqual(resolved['offset'], 1.8)

    def test_molecular_case_design_resolves_derived_bond_coordinates(self):
        plan = build_study_plan({
            'name': 'n2-scaled-bond-scan',
            'objective': 'scan N2 at multiples of its reference bond length',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'N 0 0 0; N 0 0 1.09768',
                'basis': 'sto-3g',
                'method': 'mp2',
            },
            'case_design': {
                'mode': 'grid',
                'variables': {'bond_factor': [0.2, 2.0]},
                'template': {
                    'request_updates': {
                        'atom': 'N 0 0 0; N 0 0 $(bond_factor * 1.09768)',
                    },
                },
            },
            'observables': ['energy'],
        })

        self.assertEqual([case.request['atom'] for case in plan.cases], [
            'N 0 0 0; N 0 0 0.219536',
            'N 0 0 0; N 0 0 2.19536',
        ])

    def test_regression_case_template_accepts_braced_numeric_expression(self):
        resolved = resolve_templates({
            'atom': 'N 0 0 0; N 0 0 ${bond_factor * 1.09768}',
            'bond_length': '${bond_factor * 1.09768}',
            'label': 'factor=${bond_factor}',
        }, {'bond_factor': 2.0})

        self.assertEqual(resolved['atom'], 'N 0 0 0; N 0 0 2.19536')
        self.assertEqual(resolved['bond_length'], 2.19536)
        self.assertEqual(resolved['label'], 'factor=2.0')

    def test_regression_molecular_geometry_coordinate_aliases_override_base_task(self):
        plan = build_study_plan({
            'name': 'n2-geometry-alias-scan',
            'objective': 'scan N2 from geometry alias',
            'system_type': 'molecular',
            'base_task': {
                'geometry': 'N 0 0 0; N 0 0 1.09768',
                'basis': 'sto-3g',
                'method': 'mp2',
            },
            'case_design': {
                'mode': 'grid',
                'variables': {'bond_factor': [0.8, 4.0]},
                'template': {
                    'request_updates': {
                        'coordinates': 'N 0 0 0; N 0 0 $(bond_factor * 1.09768)',
                    },
                },
            },
            'observables': ['energy'],
        })

        self.assertEqual([case.request['atom'] for case in plan.cases], [
            'N 0 0 0; N 0 0 0.878144',
            'N 0 0 0; N 0 0 4.39072',
        ])
        self.assertTrue(all('geometry' not in case.request for case in plan.cases))
        self.assertTrue(all('coordinates' not in case.request for case in plan.cases))

    def test_case_template_expressions_reject_code_and_unknown_variables(self):
        with self.assertRaisesRegex(ValueError, 'Unsupported case expression'):
            resolve_templates('$(abs(bond_factor))', {'bond_factor': 2.0})
        with self.assertRaisesRegex(ValueError, 'Unknown case variable in expression'):
            resolve_templates('$(unknown_factor * 1.09768)', {})

    def test_molecular_initial_scan_decision_uses_correlation_risk_and_active_space_audit(self):
        decisions = analyze_initial_scan_report({
            'cases': [
                {
                    'case_id': 'case-0001',
                    'label': 'stretched H2',
                    'variables': {'bond': 2.5},
                    'request': {'task_type': 'molecular', 'method': 'mp2'},
                    'task_report': {
                        'execution_status': 'succeeded',
                        'structured_results': {
                            'task_type': 'molecular',
                            'method': 'mp2',
                            'active_space': {
                                'enabled': True,
                                'selection_method': 'occupation_window',
                                'ncas': 4,
                                'nelecas': 4,
                                'orbital_indices': [1, 2, 3, 4],
                                'approved': False,
                            },
                            'correlation_diagnostics': {
                                'kind': 'molecular_correlation_diagnostics',
                                'level': 'strong',
                                'score': 0.82,
                                'confidence': 'high',
                                'physics_score': 0.9,
                                'solver_stress_score': 0.67,
                                'molecular_correlation_risk': {
                                    'kind': 'molecular_correlation_risk',
                                    'level': 'strong',
                                    'overall_score': 0.82,
                                    'physics_score': 0.9,
                                    'solver_stress_score': 0.67,
                                    'physics_components': [
                                        {
                                            'name': 'correlated_natural_occupations',
                                            'score': 0.95,
                                            'interpretation': 'Correlated natural occupations away from 2/0 identify multireference frontier orbitals.',
                                        },
                                    ],
                                    'solver_stress_components': [
                                        {
                                            'name': 'max_double_excitation_amplitude',
                                            'score': 0.7,
                                            'interpretation': 'Large double-excitation amplitudes indicate stress in a low-order single-reference expansion.',
                                        },
                                    ],
                                    'method_recommendation': {
                                        'preferred': ['casscf', 'casci'],
                                        'next_step': 'Build or approve an ActiveSpaceAudit candidate before multireference treatment.',
                                    },
                                },
                            },
                        },
                    },
                },
            ],
            'comparison_table': [
                {
                    'case_id': 'case-0001',
                    'label': 'stretched H2',
                    'status': 'succeeded',
                    'method': 'mp2',
                    'energy': -1.0,
                },
            ],
        })

        self.assertEqual(len(decisions), 1)
        decision = decisions[0]
        self.assertEqual(decision['recommended_method'], 'casscf')
        self.assertEqual(decision['recommended_solver'], 'casscf')
        self.assertEqual(decision['diagnostic_level'], 'strong')
        self.assertIn('requires_active_space_approval', decision['tags'])
        self.assertEqual(decision['active_space_contract']['ncas'], 4)
        self.assertEqual(decision['active_space_contract']['nelecas'], 4)
        self.assertTrue(any('ActiveSpaceAudit proposed CAS' in reason for reason in decision['reasons']))
        self.assertTrue(any('MolecularCorrelationRisk=strong' in reason for reason in decision['reasons']))

    def test_solver_stress_only_initial_scan_routes_through_ccsd_without_cas_approval(self):
        decisions = analyze_initial_scan_report({
            'cases': [{
                'case_id': 'case-0001',
                'label': 'stressed MP2 density',
                'variables': {'bond_length': 2.4},
                'request': {'task_type': 'molecular', 'method': 'hf'},
                'task_report': {
                    'execution_status': 'succeeded',
                    'structured_results': {
                        'task_type': 'molecular',
                        'method': 'hf',
                        'active_space': {
                            'enabled': True,
                            'ncas': 6,
                            'nelecas': 6,
                            'orbital_indices': [4, 5, 6, 7, 8, 9],
                            'approved': False,
                        },
                        'correlation_diagnostics': {
                            'kind': 'molecular_correlation_diagnostics',
                            'level': 'strong',
                            'routing_level': 'moderate',
                            'score': 0.28,
                            'confidence': 'high',
                            'molecular_correlation_risk': {
                                'kind': 'molecular_correlation_risk',
                                'level': 'strong',
                                'level_reason': 'nonphysical_correlated_density',
                                'routing_level': 'moderate',
                                'routing_reason': 'validate_single_reference_with_ccsd_before_multireference_promotion',
                                'overall_score': 0.28,
                                'physics_level': 'weak',
                                'solver_stress_level': 'strong',
                                'physics_components': [],
                                'solver_stress_components': [],
                                'method_recommendation': {
                                    'preferred': ['ccsd'],
                                    'next_step': 'Run CCSD diagnostics before multireference promotion.',
                                },
                            },
                        },
                    },
                },
            }],
            'comparison_table': [{
                'case_id': 'case-0001',
                'label': 'stressed MP2 density',
                'status': 'succeeded',
                'method': 'hf',
            }],
        })

        decision = decisions[0]
        self.assertEqual(decision['diagnostic_level'], 'strong')
        self.assertEqual(decision['physics_level'], 'weak')
        self.assertEqual(decision['solver_stress_level'], 'strong')
        self.assertEqual(decision['level'], 'moderate')
        self.assertEqual(decision['recommended_method'], 'ccsd')
        self.assertNotIn('requires_active_space_approval', decision['tags'])
        self.assertTrue(any('route through CCSD' in reason for reason in decision['reasons']))

    def test_molecular_initial_scan_tags_inconsistent_active_space_for_review(self):
        decisions = analyze_initial_scan_report({
            'cases': [
                {
                    'case_id': 'case-0001',
                    'label': 'suspicious active space',
                    'variables': {'bond': 2.2},
                    'request': {'task_type': 'molecular', 'method': 'mp2'},
                    'task_report': {
                        'execution_status': 'succeeded',
                        'structured_results': {
                            'task_type': 'molecular',
                            'method': 'mp2',
                            'active_space': {
                                'enabled': True,
                                'selection_method': 'occupation_window',
                                'ncas': 6,
                                'nelecas': 2,
                                'orbital_indices': [4, 5, 6, 7, 8, 9],
                                'approved': False,
                                'audit_summary': {
                                    'ncas_nelecas_consistent': False,
                                },
                            },
                            'correlation_diagnostics': {
                                'kind': 'molecular_correlation_diagnostics',
                                'level': 'moderate',
                                'score': 0.55,
                                'confidence': 'medium',
                                'molecular_correlation_risk': {
                                    'kind': 'molecular_correlation_risk',
                                    'level': 'moderate',
                                    'overall_score': 0.55,
                                },
                            },
                        },
                    },
                },
            ],
            'comparison_table': [
                {
                    'case_id': 'case-0001',
                    'label': 'suspicious active space',
                    'status': 'succeeded',
                    'method': 'mp2',
                },
            ],
        })

        decision = decisions[0]
        self.assertIn('active_space_inconsistent', decision['tags'])
        self.assertIn('requires_active_space_approval', decision['tags'])
        self.assertFalse(decision['active_space_contract']['audit_summary']['ncas_nelecas_consistent'])

    def test_unconverged_molecular_probe_preserves_consistent_active_space_for_recovery(self):
        decisions = analyze_initial_scan_report({
            'cases': [{
                'case_id': 'case-0001',
                'label': 'stretched dimer',
                'variables': {'bond_length': 3.2},
                'request': {'task_type': 'molecular', 'method': 'hf'},
                'task_report': {
                    'execution_status': 'unconverged',
                    'analysis_summary': 'The mean-field reference did not converge.',
                    'structured_results': {
                        'task_type': 'molecular',
                        'method': 'hf',
                        'active_space': {
                            'enabled': True,
                            'selection_method': 'chemical_valence',
                            'ncas': 12,
                            'nelecas': 12,
                            'orbital_indices': list(range(18, 30)),
                            'initial_mo_coeff': [[1.0]],
                            'approved': False,
                            'audit_summary': {'ncas_nelecas_consistent': True},
                        },
                    },
                },
            }],
            'comparison_table': [{
                'case_id': 'case-0001',
                'label': 'stretched dimer',
                'status': 'unconverged',
                'method': 'hf',
            }],
        })

        decision = decisions[0]
        self.assertEqual(decision['active_space_contract']['ncas'], 12)
        self.assertEqual(decision['active_space_contract']['nelecas'], 12)
        self.assertIn('review_active_space', decision['tags'])
        self.assertTrue(any('preserve it' in reason for reason in decision['reasons']))

    def test_molecular_refined_plan_carries_casscf_active_space_without_auto_approval(self):
        refined = build_refined_plan_from_decisions({
            'name': 'adaptive-h2-refined',
            'objective': 'refine molecular method',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 2.5',
                'basis': 'sto-3g',
                'method': 'hf',
                'restricted': False,
            },
            'observables': ['energy'],
        }, [
            {
                'case_id': 'case-0001',
                'label': 'base',
                'variables': {},
                'initial_scan_status': 'succeeded',
                'initial_scan_method': 'mp2',
                'level': 'strong',
                'score': 0.82,
                'recommended_method': 'casscf',
                'recommended_solver': 'casscf',
                'active_space_contract': {
                    'enabled': True,
                    'selection_method': 'manual',
                    'ncas': 4,
                    'nelecas': 4,
                    'orbital_indices': [1, 2, 3, 4],
                    'approved': False,
                },
                'tags': ['strong', 'requires_active_space_approval'],
                'reasons': ['ActiveSpaceAudit proposed CAS(4, 4); user approval is required before CASSCF.'],
            },
        ])

        plan = refined['refined_plan']
        self.assertEqual(plan['system_type'], 'molecular')
        self.assertTrue(plan['resource_policy'])
        self.assertTrue(plan['cost_estimate'])
        request = plan['cases'][0]['request']
        self.assertEqual(request['method'], 'casscf')
        self.assertTrue(request['restricted'])
        self.assertIsNone(request['xc'])
        self.assertEqual(request['analysis']['outputs'], ['energy', 'homo_lumo', 'dipole'])
        self.assertEqual(request['active_space']['selection_method'], 'manual')
        self.assertEqual(request['active_space']['ncas'], 4)
        self.assertEqual(request['active_space']['nelecas'], 4)
        self.assertEqual(request['active_space']['orbital_indices'], [1, 2, 3, 4])
        self.assertFalse(request['active_space']['approved'])
        self.assertFalse(request['post_cas']['sc_nevpt2']['enabled'])
        self.assertEqual(refined['decision_log'][0]['recommended_method'], 'casscf')
        self.assertEqual(refined['decision_log'][0]['active_space_contract']['ncas'], 4)
        self.assertEqual(refined['decision_log'][0]['reference_transition']['source_reference'], 'unrestricted')
        self.assertEqual(refined['decision_log'][0]['reference_transition']['target_reference'], 'rhf_spin_adapted')

    def test_single_reference_refinement_drops_cas_solver_from_target_request(self):
        refined = build_refined_plan_from_decisions({
            'name': 'adaptive-cc-refined',
            'objective': 'route a weak-correlation point to CCSD(T)',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 0.74',
                'basis': 'sto-3g',
                'method': 'casscf',
                'solver': {'name': 'block2_dmrg', 'options': {'preset': 'balanced'}},
                'restricted': True,
            },
            'observables': ['energy'],
        }, [{
            'case_id': 'case-0001',
            'label': 'equilibrium',
            'variables': {},
            'initial_scan_status': 'succeeded',
            'level': 'weak',
            'recommended_method': 'ccsd_t',
            'tags': ['weak'],
            'reasons': [],
        }])

        request = refined['refined_plan']['cases'][0]['request']
        self.assertEqual(request['method'], 'ccsd_t')
        self.assertNotIn('solver', request)
        self.assertFalse(request['active_space']['enabled'])

    def test_large_active_space_routes_to_block2_dmrg_casscf_with_review(self):
        refined = build_refined_plan_from_decisions({
            'name': 'adaptive-large-cas',
            'objective': 'route a large active space',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'H 0 0 0; H 0 0 1.5',
                'basis': 'sto-3g',
                'method': 'mp2',
                'restricted': False,
            },
            'observables': ['energy'],
        }, [{
            'case_id': 'case-0001',
            'variables': {},
            'level': 'strong',
            'recommended_method': 'casscf',
            'active_space_contract': {
                'enabled': True,
                'selection_method': 'manual',
                'ncas': 16,
                'nelecas': 2,
                'orbital_indices': list(range(16)),
                'approved': False,
                'method_recommendation': {
                    'recommended_next_step': 'block2_dmrg_casci',
                },
            },
            'tags': ['strong', 'requires_active_space_approval'],
        }], {
            'active_space_solver': 'block2_dmrg',
            'active_space_orbital_processing': {
                'enabled': True,
                'localization_method': 'boys',
                'localization_scope': 'active_space',
                'orbital_ordering': 'fiedler',
            },
        })

        request = refined['refined_plan']['cases'][0]['request']
        self.assertEqual(request['method'], 'casscf')
        self.assertEqual(request['solver']['name'], 'block2_dmrg')
        self.assertEqual(request['solver']['options']['preset'], 'balanced')
        self.assertTrue(request['solver']['options']['save_mps'])
        self.assertEqual(request['orbital_processing'], {
            'enabled': True,
            'localization_method': 'boys',
            'localization_scope': 'active_space',
            'localization_occupation_thresholds': [],
            'frozen_orbital_indices': [],
            'continuation_policy': 'project_all',
            'use_natural_orbitals': False,
            'orbital_ordering': 'fiedler',
            'orbital_order': [],
        })
        self.assertIn('entanglement_diagnostics', request['analysis']['outputs'])
        self.assertIn('symmetry_analysis', request['analysis']['outputs'])
        self.assertFalse(request['active_space']['approved'])
        decision = refined['decision_log'][0]
        self.assertEqual(decision['recommended_method'], 'casscf')
        self.assertEqual(decision['recommended_solver'], 'block2_dmrg')
        estimate = refined['refined_plan']['cost_estimate']['cases'][0]
        self.assertEqual(estimate['estimation_model'], 'block2_dmrg_sweep_proxy')
        self.assertEqual(estimate['bond_dimension'], 500)
        self.assertEqual(estimate['sweeps'], 10)

    def test_molecular_refined_open_shell_casscf_uses_rohf_and_preserves_requested_nevpt2(self):
        refined = build_refined_plan_from_decisions({
            'name': 'adaptive-oh-refined',
            'objective': 'route open-shell strong correlation',
            'system_type': 'molecular',
            'base_task': {
                'atom': 'O 0 0 0; H 0 0 1.0',
                'basis': 'sto-3g',
                'method': 'mp2',
                'spin': 1,
                'restricted': False,
                'post_cas': {'sc_nevpt2': {'enabled': True, 'root': 0, 'density_fit': True}},
            },
            'observables': ['energy'],
        }, [{
            'case_id': 'case-0001',
            'variables': {},
            'level': 'strong',
            'recommended_method': 'casscf',
            'active_space_contract': {
                'enabled': True,
                'selection_method': 'manual',
                'ncas': 3,
                'nelecas': [2, 1],
                'orbital_indices': [2, 3, 4],
                'approved': False,
            },
        }])

        request = refined['refined_plan']['cases'][0]['request']
        transition = refined['decision_log'][0]['reference_transition']
        self.assertTrue(request['restricted'])
        self.assertTrue(request['post_cas']['sc_nevpt2']['enabled'])
        self.assertEqual(transition['source_reference'], 'unrestricted')
        self.assertEqual(transition['target_reference'], 'rohf_spin_adapted')

    def test_active_space_contract_carries_bounded_expansion_evidence(self):
        contract = _active_space_contract_from_case_report({
            'task_report': {
                'structured_results': {
                    'active_space': {
                        'ncas': 2,
                        'nelecas': 2,
                        'orbital_indices': [1, 2],
                        'initial_mo_coeff': [[1.0, 0.0], [0.0, 1.0]],
                        'audit': {
                            'selected_candidate_method': 'chemical_valence',
                            'candidate_active_spaces': [{
                                'method': 'chemical_valence',
                                'status': 'available',
                                'ncas': 2,
                                'estimated_nelecas': 2,
                                'target_summary': {
                                    'primary_targets': ['H 1s'],
                                    'primary_orbital_count': 2,
                                    'primary_electron_count': 2,
                                },
                            }],
                            'ncas_nelecas_consistency': {'consistent': True},
                            'orbitals': [
                                {'index': 0, 'occupation': 2.0, 'occupation_alpha': 1.0, 'occupation_beta': 1.0},
                                {'index': 1, 'occupation': 1.0, 'occupation_alpha': 1.0, 'occupation_beta': 0.0},
                                {'index': 2, 'occupation': 1.0, 'occupation_alpha': 1.0, 'occupation_beta': 0.0},
                                {'index': 3, 'occupation': 0.0, 'occupation_alpha': 0.0, 'occupation_beta': 0.0},
                            ],
                        },
                    },
                },
            },
        })

        self.assertEqual(contract['expansion_evidence']['orbital_count'], 4)
        self.assertEqual(contract['expansion_evidence']['orbitals'][0]['occupation_alpha'], 1.0)
        self.assertEqual(contract['expansion_evidence']['orbital_mapping'], 'unrestricted_index_proxy')
        self.assertEqual(contract['initial_mo_coeff'], [[1.0, 0.0], [0.0, 1.0]])
        self.assertEqual(contract['candidate_provenance']['method'], 'chemical_valence')
        self.assertEqual(
            contract['candidate_provenance']['target_summary']['primary_orbital_count'],
            2,
        )

    def test_refined_plan_estimates_cost_with_the_original_resource_policy(self):
        refined = build_refined_plan_from_decisions({
            'name': 'adaptive-cost-refined',
            'objective': 'review refined FCI cost',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'mp2'},
            'sweep': {'U': [4]},
            'observables': ['energy'],
            'resource_policy': {'total_work_review_threshold': 1, 'review_work_estimates': True},
        }, [
            {
                'case_id': 'case-0001',
                'variables': {'U_value': 4},
                'recommended_solver': 'fci',
                'recommended_method': 'fci',
                'tags': [],
                'reasons': [],
            },
        ])

        plan = refined['refined_plan']
        self.assertEqual(plan['resource_policy']['total_work_review_threshold'], 1)
        self.assertTrue(plan['cost_estimate']['approval_required'])
        self.assertFalse(plan['cost_estimate']['approved'])
