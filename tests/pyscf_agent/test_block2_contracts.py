from __future__ import annotations

import unittest

from computational_study_agent.adaptive.state_tracking import analyze_dmrg_state_tracking

from pyscf_agent.backend.electronic_hamiltonian import ElectronicHamiltonian
from pyscf_agent.providers.block2.config import block2_options_for_outputs, normalize_block2_options
from pyscf_agent.providers.block2.recovery import (
    block2_convergence_recovery,
    block2_scheduler_recovery,
    classify_block2_exception,
    escalated_block2_options,
)
from pyscf_agent.providers.block2.result_contract import finalize_block2_result_contract


def _tracking_case(case_id, signatures):
    return {
        'case_id': case_id,
        'variables': {'U': float(case_id.rsplit('-', 1)[-1])},
        'request': {'solver': {'name': 'block2_dmrg'}},
        'task_report': {
            'execution_status': 'succeeded',
            'structured_results': {
                'dmrg_result': {
                    'state_sector': {
                        'n_orbitals': 2,
                        'n_electrons': [1, 1],
                        'spin': 0,
                        'symmetry': 'su2',
                    },
                    'root_signatures': signatures,
                },
            },
        },
    }


class Block2ContractTests(unittest.TestCase):
    def test_scheduler_interruption_does_not_require_solver_convergence_evidence(self):
        for state in ('BOOT_FAIL', 'NODE_FAIL', 'PREEMPTED', 'REVOKED'):
            with self.subTest(state=state):
                recovery = block2_scheduler_recovery(state)
                self.assertEqual(recovery['status'], 'automatic_retry_available')
                self.assertTrue(recovery['automatic_retry_safe'])
                self.assertEqual(recovery['recommended_action'], 'retry_same_profile')

    def test_discarded_weight_failure_at_automatic_limit_requires_review(self):
        recovery = block2_convergence_recovery({
            'converged': False,
            'convergence': {'energy_converged': True, 'discarded_weight_converged': False},
            'adaptive_schedule': {'maximum_bond_dimension': 2000},
        })
        self.assertEqual(recovery['status'], 'review_required')
        self.assertFalse(recovery['automatic_retry_safe'])
        self.assertEqual(recovery['recommended_action'], 'review_resource_profile')

    def test_excited_state_outputs_keep_rdm_diagnostics_on_ground_state(self):
        options = block2_options_for_outputs({}, ['excited_states'])
        config = normalize_block2_options(options)

        self.assertEqual(config.nroots, 2)
        self.assertEqual(config.rdm1_root_scope, 'ground_state')
        self.assertEqual(config.rdm2_root_scope, 'ground_state')

    def test_entanglement_outputs_stop_at_ground_state_2rdm(self):
        options = block2_options_for_outputs({}, ['entanglement_diagnostics'])
        config = normalize_block2_options(options)

        self.assertTrue(config.compute_1rdm)
        self.assertTrue(config.compute_2rdm)
        self.assertTrue(config.compute_entanglement)
        self.assertFalse(config.compute_mutual_information)
        self.assertEqual(config.rdm1_root_scope, 'ground_state')
        self.assertEqual(config.rdm2_root_scope, 'ground_state')

    def test_common_result_contract_exposes_targeted_roots_and_convergence(self):
        hamiltonian = ElectronicHamiltonian(
            n_orbitals=2,
            n_electrons=(1, 1),
            spin=0,
            core_energy=0.0,
            h1e=[[0.0, 0.0], [0.0, 0.0]],
            g2e=[[[[0.0] * 2 for _ in range(2)] for _ in range(2)] for _ in range(2)],
        )
        config = normalize_block2_options({
            'nroots': 2,
            'compute_2rdm': True,
            'rdm1_root_scope': 'all_states',
            'rdm2_root_scope': 'ground_state',
        })
        result = {
            'energy': -1.0,
            'state_energies': [-1.0, -0.8],
            'excitation_energies': [0.0, 0.2],
            'natural_occupations': [1.9, 0.1],
            'convergence': {
                'final_discarded_weight': 1e-8,
                'final_energy_change': 1e-9,
                'sweeps_completed': 8,
            },
            'adaptive_schedule': {
                'final_bond_dimension': 100,
                'maximum_bond_dimension': 200,
            },
            'checkpoint_manifest': {'files': [{'relative_path': 'GS-MPS_INFO'}]},
        }

        finalize_block2_result_contract(result, hamiltonian, config, 'su2')

        self.assertEqual(result['computed_root_count'], 2)
        self.assertTrue(result['targeted_roots_complete'])
        self.assertEqual(len(result['root_signatures']), 2)
        self.assertEqual(result['root_signatures'][0]['natural_occupations'], [1.9, 0.1])
        self.assertIsNone(result['root_signatures'][1]['natural_occupations'])
        self.assertEqual(result['result_scope']['rdm1'], 'ground_state')
        self.assertEqual(result['result_scope']['rdm2'], 'ground_state')
        self.assertTrue(result['checkpoint_available'])
        checks = {item['id']: item for item in result['quality_checks']}
        self.assertTrue(all('status' not in item for item in result['quality_checks']))
        self.assertEqual(checks['block2_energy_change']['observed'], 1e-9)
        self.assertEqual(checks['block2_discarded_weight']['observed'], 1e-8)
        self.assertEqual(checks['block2_targeted_roots_complete']['expected'], 2)

    def test_state_tracking_follows_physical_signatures_across_root_reordering(self):
        report = {
            'cases': [
                _tracking_case('case-1', [
                    {'root': 0, 'energy': -1.0, 'excitation_energy': 0.0, 'natural_occupations': [1.9, 0.1]},
                    {'root': 1, 'energy': -0.9, 'excitation_energy': 0.1, 'natural_occupations': [1.1, 0.9]},
                ]),
                _tracking_case('case-2', [
                    {'root': 0, 'energy': -1.1, 'excitation_energy': 0.0, 'natural_occupations': [1.1, 0.9]},
                    {'root': 1, 'energy': -1.0, 'excitation_energy': 0.1, 'natural_occupations': [1.9, 0.1]},
                ]),
            ],
        }

        tracking = analyze_dmrg_state_tracking(report)

        self.assertEqual(tracking['status'], 'tracked')
        self.assertTrue(tracking['links'][0]['root_order_changed'])
        self.assertEqual(
            [(item['source_root'], item['target_root']) for item in tracking['links'][0]['mapping']],
            [(0, 1), (1, 0)],
        )

    def test_state_tracking_marks_indistinguishable_assignments_ambiguous(self):
        signature = [
            {'root': 0, 'energy': -1.0, 'excitation_energy': 0.0, 'natural_occupations': [1.0, 1.0]},
            {'root': 1, 'energy': -1.0, 'excitation_energy': 0.0, 'natural_occupations': [1.0, 1.0]},
        ]
        tracking = analyze_dmrg_state_tracking({
            'cases': [
                _tracking_case('case-1', signature),
                _tracking_case('case-2', signature),
            ],
        })

        self.assertEqual(tracking['status'], 'review_required')
        self.assertEqual(tracking['ambiguous_case_ids'], ['case-1', 'case-2'])

    def test_recovery_distinguishes_memory_and_dmrg_convergence(self):
        memory = classify_block2_exception(RuntimeError('std::bad_alloc: out of memory'))
        self.assertEqual(memory['recommended_action'], 'increase_memory_profile')
        self.assertFalse(memory['automatic_retry_safe'])

        convergence = block2_convergence_recovery({
            'converged': False,
            'convergence': {
                'energy_converged': True,
                'discarded_weight_converged': False,
                'final_discarded_weight': 1e-4,
            },
            'adaptive_schedule': {
                'final_bond_dimension': 200,
                'maximum_bond_dimension': 200,
            },
        })
        escalated = escalated_block2_options(
            {'bond_dimensions': [100, 200], 'max_bond_dimension': 200, 'adaptive_sweeps': 8},
            convergence,
        )
        self.assertEqual(convergence['recommended_action'], 'increase_bond_dimension')
        self.assertEqual(escalated['max_bond_dimension'], 400)
        self.assertEqual(escalated['adaptive_sweeps'], 12)

        timeout = block2_scheduler_recovery('TIMEOUT')
        self.assertEqual(timeout['failure_class'], 'walltime_exhausted')
        self.assertEqual(timeout['recommended_action'], 'increase_walltime_profile')


if __name__ == '__main__':
    unittest.main()
