from __future__ import annotations

import json
import tempfile
from pathlib import Path

from computational_study_agent import StudyCase, StudyReport
from computational_study_agent.adaptive.executor import (
    analyze_initial_scan_report,
    run_adaptive_study,
)
from computational_study_agent.postprocessing import suggest_plot_specs
from computational_study_agent.executor import _case_summary
from tests.computational_study_agent.support import StudyAgentTestCase, hubbard_dimer_spec


class AdaptiveExecutionTests(StudyAgentTestCase):
    def test_dmet_rows_use_energy_per_site_without_reference_difference(self):
        case = StudyCase(
            case_id='case-0001',
            label='U=4',
            request={'solver': {'name': 'dmet'}},
            variables={'U': 4.0},
            model_spec={'energy_unit': 'a.u.'},
        )
        task_report = {
            'execution_status': 'succeeded',
            'structured_results': {
                'task_type': 'model_hamiltonian',
                'solver': 'dmet',
                'energy_unit': 'a.u.',
                'energy': -25.6,
                'energy_per_site': -0.8,
                'reference_energy': -18.0,
                'correlation_energy': -7.6,
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            row = _case_summary(case, task_report, ['energy'], Path(tmpdir))

        self.assertEqual(row['final_energy'], '-25.6 a.u.')
        self.assertEqual(row['energy_per_site'], '-0.8 a.u./site')
        self.assertNotIn('reference_energy', row)
        self.assertNotIn('correlation_energy', row)
        self.assertEqual(row['quality_status'], 'legacy')
        self.assertIsNone(row['publication_eligible'])


    def test_case_summary_propagates_task_quality_decision(self):
        case = StudyCase(
            case_id='case-0001',
            label='U=0',
            request={'solver': {'name': 'dmet'}},
            variables={'U': 0.0},
            model_spec={'energy_unit': 'a.u.'},
        )
        task_report = {
            'execution_status': 'succeeded',
            'structured_results': {'energy': -4.0, 'energy_unit': 'a.u.'},
            'gate_decisions': [{
                'gate_id': 'task.execution_quality',
                'status': 'review_required',
                'details': {
                    'quality_status': 'review_required',
                    'publication_eligible': False,
                },
            }],
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            row = _case_summary(case, task_report, ['energy'], Path(tmpdir))

        self.assertEqual(row['quality_status'], 'review_required')
        self.assertFalse(row['publication_eligible'])

    def test_block2_outputs_enter_comparison_table_and_postprocessing(self):
        case = StudyCase(
            case_id='case-0001',
            label='U=4',
            request={'solver': {'name': 'block2_dmrg'}},
            variables={'U': 4.0},
            model_spec={'energy_unit': 'a.u.'},
        )
        task_report = {
            'execution_status': 'succeeded',
            'structured_results': {
                'task_type': 'model_hamiltonian',
                'solver': 'block2_dmrg',
                'energy_unit': 'a.u.',
                'energy': -2.1,
                'dmrg_result': {
                    'state_energies': [-2.1, -1.8],
                    'excitation_energies': [0.0, 0.3],
                    'entanglement_diagnostics': {
                        'max_single_orbital_entropy': 1.1,
                        'mean_single_orbital_entropy': 0.8,
                        'max_mutual_information': 0.4,
                        'max_bipartite_entanglement': 0.7,
                    },
                    'symmetry_analysis': {'spin_square': 0.0},
                    'convergence': {
                        'final_discarded_weight': 1e-9,
                        'final_energy_change': 2e-10,
                    },
                    'energy_error_estimate': {
                        'status': 'available',
                        'method': 'linear_energy_vs_discarded_weight_extrapolation',
                        'estimated_absolute_error': 3e-8,
                        'extrapolated_energy': -2.10000003,
                    },
                    'adaptive_schedule': {
                        'adaptive_stages_used': 1,
                        'final_bond_dimension': 400,
                        'budget_exhausted': False,
                    },
                    'restart': {'requested': True, 'applied': True, 'source_case_id': 'case-0000'},
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            row = _case_summary(case, task_report, ['energy'], Path(tmpdir))

        self.assertEqual(row['dmrg_state_count'], 2)
        self.assertEqual(row['first_excitation_energy'], '0.3 a.u.')
        self.assertEqual(row['max_single_orbital_entropy'], 1.1)
        self.assertEqual(row['max_mutual_information'], 0.4)
        self.assertEqual(row['dmrg_spin_square'], 0.0)
        self.assertEqual(row['dmrg_estimated_energy_error'], '3e-08 a.u.')
        self.assertEqual(row['dmrg_extrapolated_energy'], '-2.10000003 a.u.')
        self.assertEqual(
            row['dmrg_error_estimate_method'],
            'linear_energy_vs_discarded_weight_extrapolation',
        )
        self.assertEqual(row['dmrg_adaptive_stages'], 1)
        self.assertEqual(row['dmrg_final_bond_dimension'], 400)
        self.assertFalse(row['dmrg_adaptive_budget_exhausted'])
        self.assertEqual(row['mps_restart_source_case'], 'case-0000')
        report = {
            'study_id': 'block2-output-study',
            'name': 'block2 output study',
            'objective': 'verify registered block2 postprocessing fields',
            'system_type': 'model_hamiltonian',
            'work_dir': tempfile.gettempdir(),
            'cases': [case.to_dict(), {**case.to_dict(), 'case_id': 'case-0002', 'variables': {'U': 6.0}}],
            'comparison_table': [row, {**row, 'case_id': 'case-0002', 'U': '6.0 a.u.', 'first_excitation_energy': '0.5 a.u.'}],
            'artifacts': [],
        }
        specs = suggest_plot_specs(report, max_specs=20)
        self.assertTrue(any(spec.get('y') == 'first_excitation_energy' for spec in specs))
        self.assertTrue(any(spec.get('y') == 'dmrg_estimated_energy_error' for spec in specs))

    def test_adaptive_study_runs_initial_scan_then_refined_solver_choices(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_adaptive_study({
                'name': 'adaptive-dimer-u',
                'objective': 'adaptively choose solvers across U',
                'system_type': 'model_hamiltonian',
                'base_model_spec': hubbard_dimer_spec(),
                'base_task': {'solver': 'ccsd'},
                'sweep': {'U': [0, 4, 8]},
                'observables': ['energy'],
            }, work_dir=tmpdir, locale='en', options={})
            adaptive_artifact_payloads = {
                artifact['kind']: json.loads(Path(artifact['path']).read_text(encoding='utf-8'))
                for artifact in report['artifacts']
                if artifact['kind'] in {'adaptive-decision-log', 'adaptive-study-report'}
            }

        self.assertEqual(report['status'], 'succeeded')
        self.assertEqual(report['adaptive']['mode'], 'adaptive_scan')
        # Both the service result and persisted adaptive report can cross the
        # typed boundary without losing stage evidence or changing schema.
        for payload in (report, adaptive_artifact_payloads['adaptive-study-report']):
            restored = StudyReport.from_dict(payload).to_dict()
            for key, value in payload.items():
                self.assertEqual(restored[key], value, key)
        self.assertEqual(report['adaptive']['workflow']['stage'], 'completed')
        self.assertEqual(report['adaptive']['workflow']['scope'], 'full')
        self.assertEqual(len(report['adaptive']['initial_scan_decisions']), 3)
        self.assertEqual(len(report['adaptive']['decision_log']), 3)
        self.assertEqual(len(report['comparison_table']), 3)
        self.assertTrue(all(item.get('initial_scan_method') == 'mp2' for item in report['adaptive']['initial_scan_decisions']))
        self.assertTrue(all('initial_scan_energy' not in item for item in report['adaptive']['initial_scan_decisions']))
        self.assertTrue(all(item.get('initial_scan_method') == 'mp2' for item in report['adaptive']['decision_log']))
        initial_scan_diagnostic_names = {
            diagnostic.get('name')
            for decision in report['adaptive']['initial_scan_decisions']
            for diagnostic in decision.get('diagnostics', {}).get('diagnostics', [])
        }
        self.assertIn('perturbative_correlation_stress', initial_scan_diagnostic_names)
        self.assertIn('mean_field_homo_lumo_gap', initial_scan_diagnostic_names)
        self.assertIn('natural_orbital_occupations', initial_scan_diagnostic_names)
        self.assertIn('max_double_excitation_amplitude', initial_scan_diagnostic_names)
        solvers = {row['solver'] for row in report['comparison_table']}
        self.assertTrue(solvers.issubset({'mp2', 'ccsd', 'fci'}))
        comparison_columns = {
            column
            for row in report['comparison_table']
            for column in row
        }
        self.assertIn('final_energy', comparison_columns)
        self.assertNotIn('correlation_energy', comparison_columns)
        self.assertIn('filling', comparison_columns)
        self.assertTrue({
            'gap',
            'mean_field_gap',
            'natural_occupation_fractionality',
            'max_double_excitation_amplitude',
        } & comparison_columns)
        plot_specs = suggest_plot_specs(report)
        self.assertTrue(plot_specs)
        plot_columns = {
            value
            for spec in plot_specs
            for value in (spec.get('x'), spec.get('y'), spec.get('color'))
            if value
        }
        self.assertTrue(plot_columns.issubset(comparison_columns))
        # U/t alone no longer forces escalation in this internal model helper.
        # The noninteracting determinant still takes the weak-evidence route.
        self.assertEqual(report['adaptive']['decision_log'][0]['recommended_solver'], 'mp2')
        artifact_kinds = {item['kind'] for item in report['artifacts']}
        self.assertIn('adaptive-initial-scan-plan', artifact_kinds)
        self.assertIn('adaptive-refined-plan', artifact_kinds)
        self.assertIn('adaptive-decision-log', artifact_kinds)
        self.assertIn('adaptive-study-report', artifact_kinds)
        for artifact in report['artifacts']:
            self.assertTrue(
                {'kind', 'path', 'size_bytes', 'mime_type', 'description'}
                <= set(artifact)
            )
            self.assertNotIn('sha256', artifact)
            self.assertTrue(artifact['description'])
        for kind, schema in (
            ('adaptive-decision-log', 'pyscf-agent.adaptive-decision-log.v1'),
            ('adaptive-study-report', 'pyscf-agent.adaptive-study-report.v1'),
        ):
            self.assertEqual(adaptive_artifact_payloads[kind]['schema'], schema)

    def test_adaptive_study_resume_reuses_initial_and_refined_case_checkpoints(self):
        study_spec = {
            'name': 'adaptive-resume-dimer',
            'objective': 'reuse completed adaptive dimer cases',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'ccsd'},
            'sweep': {'U': [4]},
            'observables': ['energy'],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            first_report = run_adaptive_study(study_spec, work_dir=tmpdir, locale='en')
            resumed_report = run_adaptive_study(
                study_spec,
                work_dir=tmpdir,
                locale='en',
                resume_study_id=first_report['study_id'],
            )

        self.assertEqual(resumed_report['study_id'], first_report['study_id'])
        self.assertEqual(resumed_report['adaptive']['initial_scan_report']['execution']['counts']['resumed'], 1)
        self.assertEqual(resumed_report['adaptive']['refined_report']['execution']['counts']['resumed'], 1)

    def test_adaptive_study_rejects_resume_directory_for_a_different_request(self):
        original_spec = {
            'name': 'adaptive-resume-dimer',
            'objective': 'reuse only matching adaptive dimer cases',
            'system_type': 'model_hamiltonian',
            'base_model_spec': hubbard_dimer_spec(),
            'base_task': {'solver': 'ccsd'},
            'sweep': {'U': [0]},
            'observables': ['energy'],
        }
        changed_spec = {
            **original_spec,
            'sweep': {'U': [8]},
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            first_report = run_adaptive_study(original_spec, work_dir=tmpdir, locale='en')
            changed_report = run_adaptive_study(
                changed_spec,
                work_dir=tmpdir,
                locale='en',
                resume_study_id=first_report['study_id'],
            )

        self.assertNotEqual(changed_report['study_id'], first_report['study_id'])

    def test_adaptive_study_stops_when_initial_scan_is_blocked(self):
        model_spec = hubbard_dimer_spec()
        model_spec['model'] = 'holstein_hubbard'
        model_spec['phonons'] = {
            'enabled': True,
            'cutoff': 4,
            'local_modes': [
                {'site': 0, 'omega': 1, 'g': 0.5},
                {'site': 1, 'omega': 1, 'g': 0.5},
            ],
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            report = run_adaptive_study({
                'name': 'adaptive-blocked-initial-scan',
                'objective': 'show blocked initial-scan reason',
                'system_type': 'model_hamiltonian',
                'base_model_spec': model_spec,
                'base_task': {'solver': 'mp2'},
                'sweep': {'U': [4]},
                'observables': ['energy'],
            }, work_dir=tmpdir, locale='en', options={})

        self.assertEqual(report['status'], 'completed_with_issues')
        self.assertEqual(report['adaptive']['workflow']['stage'], 'initial_scan_blocked')
        self.assertEqual(report['adaptive']['workflow']['pending_case_ids'], ['case-0001'])
        self.assertEqual(
            [action['id'] for action in report['adaptive']['workflow']['allowed_actions']],
            ['show_case_guidance', 'acknowledge'],
        )
        self.assertEqual(report['adaptive']['refined_plan'], None)
        self.assertEqual(report['adaptive']['decision_log'], [])
        self.assertEqual(len(report['adaptive']['initial_scan_issues']), 1)
        issue = report['adaptive']['initial_scan_issues'][0]
        self.assertEqual(issue['initial_scan_status'], 'blocked')
        self.assertTrue(any('Holstein-Hubbard' in reason for reason in issue['reasons']))
        self.assertTrue(any('initial_scan_blocked' in decision.get('tags', []) for decision in report['adaptive']['initial_scan_decisions']))
        self.assertEqual(report['comparison_table'][0]['status'], 'blocked')
        artifact_kinds = {item['kind'] for item in report['artifacts']}
        self.assertIn('adaptive-initial-scan-plan', artifact_kinds)
        self.assertIn('adaptive-decision-log', artifact_kinds)
        self.assertNotIn('adaptive-refined-plan', artifact_kinds)

    def test_adaptive_initial_scan_unconverged_recommends_recovery_solver(self):
        decisions = analyze_initial_scan_report({
            'cases': [
                {
                    'case_id': 'case-0001',
                    'label': 'U=8',
                    'variables': {'U': 8},
                    'request': {
                        'solver': 'mp2',
                        'model_hamiltonian': {'spec': hubbard_dimer_spec()},
                    },
                    'task_report': {
                        'execution_status': 'unconverged',
                        'raw_stderr': 'SCF did not converge.',
                        'analysis_summary': 'The task finished but SCF did not converge.',
                        'structured_results': {
                            'site_count': 2,
                            'strong_correlation_diagnostics': {
                                'level': 'moderate',
                                'score': 0.5,
                                'confidence': 'medium',
                                'summary': 'Moderate initial-scan warning.',
                                'parameter_summary': {'site_count': 2},
                                'diagnostics': [],
                            },
                        },
                    },
                },
            ],
            'comparison_table': [
                {
                    'case_id': 'case-0001',
                    'label': 'U=8',
                    'status': 'unconverged',
                    'solver': 'mp2',
                    'energy': '-1.0 a.u.',
                    'U': '8 a.u.',
                },
            ],
        }, {'max_fci_sites': 8})

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]['recommended_solver'], 'fci')
        self.assertIn('initial_scan_unconverged', decisions[0]['tags'])
        self.assertIn('needs_method_recovery', decisions[0]['tags'])
        self.assertTrue(any('did not converge' in reason for reason in decisions[0]['reasons']))

    def test_strong_model_diagnostics_recommend_fci_for_small_case_spec(self):
        decisions = analyze_initial_scan_report({
            'cases': [
                {
                    'case_id': 'case-0001',
                    'label': 'U=12',
                    'variables': {'U': 12},
                    'request': {
                        'solver': 'mp2',
                        'model_hamiltonian': {'spec': hubbard_dimer_spec()},
                    },
                    'task_report': {
                        'execution_status': 'succeeded',
                        'structured_results': {
                            'strong_correlation_diagnostics': {
                                'level': 'strong',
                                'score': 0.82,
                                'confidence': 'high',
                                'summary': 'Strong finite-size Hubbard strong-correlation tendency.',
                                'diagnostics': [],
                            },
                        },
                    },
                },
            ],
            'comparison_table': [
                {
                    'case_id': 'case-0001',
                    'label': 'U=12',
                    'status': 'succeeded',
                    'solver': 'mp2',
                    'energy': '-1.0 a.u.',
                    'U': '12 a.u.',
                },
            ],
        }, {'max_fci_sites': 8})

        self.assertEqual(len(decisions), 1)
        self.assertEqual(decisions[0]['recommended_solver'], 'fci')
        self.assertEqual(decisions[0]['recommended_method'], 'fci')
        self.assertIn('needs_refined_method', decisions[0]['tags'])
