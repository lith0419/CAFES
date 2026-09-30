from __future__ import annotations

import os
import tempfile
import unittest
from unittest import mock

from pyscf_agent.benchmarks import (
    BENCHMARK_MANIFEST_SCHEMA,
    BENCHMARK_RESULT_SCHEMA,
    RUN_ENVIRONMENT_VARIABLE,
    benchmark_manifest,
    build_fcdmft_si_g0w0_request,
    evaluate_fcdmft_si_g0w0_report,
    list_benchmarks,
    run_benchmark,
    run_benchmark_suite,
)
from pyscf_agent.schema_contracts import validate_public_payload


class ScientificBenchmarkTests(unittest.TestCase):
    def test_manifest_declares_three_supported_domains(self):
        manifest = benchmark_manifest()

        self.assertEqual(manifest['schema'], BENCHMARK_MANIFEST_SCHEMA)
        self.assertEqual(validate_public_payload(manifest), BENCHMARK_MANIFEST_SCHEMA)
        self.assertEqual(
            {item['domain'] for item in list_benchmarks()},
            {'molecular', 'model_hamiltonian', 'periodic'},
        )
        scaling = {
            item['id']
            for item in manifest['benchmarks']
            if item.get('tier') == 'scaling'
        }
        self.assertEqual(scaling, {
            'block2-hydrogen-chain-scaling',
            'block2-hubbard-ring-scaling',
        })

    def test_n2_reference_benchmark_checks_accuracy_and_path_continuity(self):
        result = run_benchmark('n2-dissociation-sto3g')

        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['metrics']['row_count'], 37)
        self.assertEqual(result['metrics']['path_anomaly_count'], 1)
        self.assertEqual(result['metrics']['path_review_status'], 'review_required')

    def test_hubbard_dimer_matches_analytic_energy_and_double_occupancy(self):
        result = run_benchmark('hubbard-dimer-ed')

        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['metrics']['point_count'], 4)
        self.assertLess(result['metrics']['max_energy_abs_error_Ha'], 1e-12)
        self.assertLess(result['metrics']['max_double_occupancy_abs_error'], 1e-12)

    def test_periodic_gamma_baseline_executes_through_public_backend(self):
        with tempfile.TemporaryDirectory() as directory:
            result = run_benchmark('periodic-he-gamma-rhf', work_dir=directory)

        self.assertEqual(result['status'], 'passed')
        self.assertEqual(result['metrics']['execution_status'], 'succeeded')
        self.assertTrue(result['metrics']['converged'])
        self.assertGreater(result['metrics']['artifact_count'], 0)

    def test_fcdmft_si_g0w0_is_registered_but_explicitly_opt_in(self):
        definition = next(
            item for item in list_benchmarks()
            if item['id'] == 'fcdmft-si-g0w0'
        )
        self.assertEqual(definition['domain'], 'periodic')
        self.assertEqual(definition['tier'], 'optional_provider')
        self.assertEqual(definition['source']['parameters']['kmesh'], [4, 4, 4])

        environment = dict(os.environ)
        environment.pop(RUN_ENVIRONMENT_VARIABLE, None)
        with mock.patch.dict(os.environ, environment, clear=True):
            result = run_benchmark('fcdmft-si-g0w0')

        self.assertEqual(result['status'], 'skipped')
        self.assertIn(RUN_ENVIRONMENT_VARIABLE, result['skip_reason'])

        request = build_fcdmft_si_g0w0_request()
        self.assertEqual(request['periodic']['kmesh'], [4, 4, 4])
        self.assertEqual(request['solver']['name'], 'gw')

    def test_fcdmft_si_g0w0_report_checks_physics_and_artifact_contract(self):
        required_artifacts = [
            'periodic_gw_result',
            'periodic_gw_arrays',
            'periodic_gw_analytic_continuation',
            'periodic_gw_mean_field_potential',
            'periodic_gw_imaginary_self_energy',
        ]
        report = {
            'execution_status': 'succeeded',
            'task_spec': {
                'task_type': 'periodic',
                'method': {'name': 'dft', 'xc': 'pbe', 'restricted': True},
                'periodic': {
                    'basis': 'gth-dzvp',
                    'pseudo': 'gth-pbe',
                    'kmesh': [4, 4, 4],
                    'kpoint_scheme': 'gamma_centered',
                    'kpoint_shift': [0.0, 0.0, 0.0],
                    'density_fitting_method': 'gdf',
                    'precision': 1e-12,
                    'smearing_method': 'none',
                },
                'solver': {'name': 'gw'},
            },
            'structured_results': {
                'reference_converged': True,
                'final_energy': None,
                'energy_kind': 'gw_total_energy_unavailable',
                'gw_result': {
                    'status': 'completed',
                    'reference_method': 'dft',
                    'kpoint_count': 64,
                    'analytic_continuation': 'pade',
                    'full_self_energy': True,
                    'finite_size_correction': True,
                    'real_frequency_points': 181,
                    'imaginary_frequency_points': 100,
                    'quasiparticle_homo': -0.2,
                    'quasiparticle_lumo': -0.2 + 1.2 / 27.211386,
                    'quasiparticle_gap': 1.2 / 27.211386,
                    'fermi_energy': -0.18,
                    'energy_available': False,
                },
            },
            'artifacts': [
                {'kind': kind, 'path': '/tmp/{0}'.format(kind)}
                for kind in required_artifacts
            ],
        }

        outcome = evaluate_fcdmft_si_g0w0_report(report)

        self.assertTrue(all(check['passed'] for check in outcome['checks']))
        self.assertAlmostEqual(outcome['metrics']['quasiparticle_gap_eV'], 1.2)
        self.assertFalse(outcome['metrics']['energy_available'])

        report['structured_results']['gw_result']['quasiparticle_gap'] = 0.1 / 27.211386
        report['artifacts'].pop()
        outcome = evaluate_fcdmft_si_g0w0_report(report)
        checks = {item['name']: item for item in outcome['checks']}
        self.assertFalse(checks['silicon_quasiparticle_gap_sanity']['passed'])
        self.assertFalse(checks['registered_gw_artifacts']['passed'])

    def test_block2_h4_matches_pyscf_fci_when_provider_is_available(self):
        result = run_benchmark('block2-h4-casci')

        self.assertIn(result['status'], ('passed', 'skipped'))
        if result['status'] == 'passed':
            self.assertLess(result['metrics']['energy_abs_error_Ha'], 1e-8)

    def test_block2_h2_casscf_matches_pyscf_fci_casscf_when_provider_is_available(self):
        result = run_benchmark('block2-h2-casscf')

        self.assertIn(result['status'], ('passed', 'skipped'))
        if result['status'] == 'passed':
            self.assertTrue(result['metrics']['orbital_converged'])
            self.assertTrue(result['metrics']['dmrg_converged'])
            self.assertGreater(result['metrics']['active_space_solver_calls'], 0)
            self.assertLess(result['metrics']['energy_abs_error_Ha'], 1e-8)

    def test_block2_ring_checks_roots_observables_and_restart(self):
        result = run_benchmark('block2-hubbard-ring')

        self.assertIn(result['status'], ('passed', 'skipped'))
        if result['status'] == 'passed':
            self.assertLess(result['metrics']['energy_abs_error_Ha'], 1e-8)
            self.assertLess(result['metrics']['first_excitation_abs_error_Ha'], 1e-7)
            self.assertEqual(result['metrics']['restart_source_case'], 'U=4')
            self.assertFalse(result['metrics']['restart_same_hamiltonian'])

    def test_block2_adaptive_workflows_are_numerically_and_structurally_checked(self):
        result = run_benchmark('block2-adaptive-workflows')

        self.assertIn(result['status'], ('passed', 'skipped'))
        if result['status'] == 'passed':
            metrics = result['metrics']
            self.assertTrue(metrics['joint_checkpoint_restart_applied'])
            self.assertLess(metrics['joint_restart_energy_abs_error_Ha'], 1e-8)
            self.assertEqual(metrics['active_space_review_case_count'], 1)
            self.assertFalse(metrics['recommended_active_space']['approved'])
            self.assertTrue(metrics['review_reuses_optimized_orbitals'])
            self.assertFalse(metrics['review_reuses_mps'])
            self.assertGreater(metrics['adaptive_stages_used'], 0)
            self.assertGreater(metrics['adaptive_final_bond_dimension'], 16)
            self.assertLess(metrics['adaptive_ring_energy_abs_error_Ha'], 1e-7)

    def test_suite_returns_one_versioned_report(self):
        report = run_benchmark_suite((
            'n2-dissociation-sto3g',
            'hubbard-dimer-ed',
        ))

        self.assertEqual(report['schema'], BENCHMARK_RESULT_SCHEMA)
        self.assertEqual(report['status'], 'passed')
        self.assertEqual(report['benchmark_count'], 2)
        self.assertEqual(report['passed_count'], 2)
        self.assertEqual(validate_public_payload(report), BENCHMARK_RESULT_SCHEMA)

    def test_default_suite_excludes_opt_in_scaling_benchmarks(self):
        default_ids = {
            item['id']
            for item in list_benchmarks()
            if item.get('tier') != 'scaling'
        }
        scaling_ids = {
            item['id']
            for item in list_benchmarks()
            if item.get('tier') == 'scaling'
        }

        self.assertTrue(default_ids)
        self.assertEqual(scaling_ids, {
            'block2-hydrogen-chain-scaling',
            'block2-hubbard-ring-scaling',
        })


if __name__ == '__main__':
    unittest.main()
