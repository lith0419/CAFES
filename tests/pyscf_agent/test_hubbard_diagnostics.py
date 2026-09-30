from __future__ import annotations

import contextlib
import io
import math
import unittest

import numpy as np

from pyscf_agent.backend.model_hamiltonian.dmet_observables import assemble_dmet_local_observables
from pyscf_agent.backend.model_hamiltonian.observables import _compute_strong_correlation_diagnostics
from pyscf_agent.backend.model_hamiltonian.reference_diagnostics import _natural_occupation_summary_from_dm1
from pyscf_agent.backend.model_hamiltonian.solver import run_model_hamiltonian_solver
from tests.pyscf_agent.test_pyscf_backend_model_hamiltonian import hubbard_chain_spec, hubbard_dimer_spec


def _item(diagnostics, name):
    return next(item for item in diagnostics['diagnostics'] if item['name'] == name)


class HubbardDiagnosticTests(unittest.TestCase):
    def test_spin_idempotency_preserves_open_shell_and_broken_symmetry_determinants(self):
        alpha = np.diag([1., 0.])
        for beta in (np.zeros((2, 2)), np.diag([0., 1.]), np.full((2, 2), 0.5)):
            for container in (tuple, list, np.asarray):
                with self.subTest(beta=beta.tolist(), container=container):
                    summary = _natural_occupation_summary_from_dm1(container([alpha, beta]))
                    self.assertGreater(summary['average_fractionality'], 0)
                    self.assertEqual(summary['spin_resolved']['status'], 'available')
                    self.assertAlmostEqual(summary['spin_resolved']['score'], 0)

    def test_spatial_density_is_not_mistaken_for_two_spin_channels(self):
        summary = _natural_occupation_summary_from_dm1([[1., 0.], [0., 1.]])
        self.assertEqual(summary['occupations'], [1., 1.])
        self.assertEqual(summary['spin_resolved']['status'], 'unavailable')
        diagnostics = _compute_strong_correlation_diagnostics(
            hubbard_dimer_spec(), {'solver': 'fci', 'natural_occupation_summary': summary},
        )
        self.assertIsNone(diagnostics['score'])
        self.assertEqual(diagnostics['level'], 'unknown')

    def test_invalid_spin_density_is_solver_stress_even_if_spin_sum_is_valid(self):
        for alpha, beta, nelec in (
            (np.diag([1.1, -0.1]), np.diag([-0.1, 1.1]), None),
            (np.diag([1., 0.]), np.diag([0., 1.]), [2, 0]),
            (np.array([[1., 0.1], [0., 0.]]), np.diag([0., 1.]), None),
        ):
            with self.subTest(alpha=alpha.tolist(), nelec=nelec):
                summary = _natural_occupation_summary_from_dm1([alpha, beta], nelec=nelec)
                diagnostics = _compute_strong_correlation_diagnostics(
                    hubbard_dimer_spec(), {'solver': 'ccsd', 'converged': True, 'natural_occupation_summary': summary},
                )
                self.assertEqual(diagnostics['physics_level'], 'unknown')
                self.assertEqual(diagnostics['solver_stress_level'], 'strong')
                self.assertEqual(diagnostics['level_reason'], 'solver_stress')

    def test_context_alone_does_not_establish_correlation(self):
        for converged in (None, True, False):
            result = {
                'solver': 'ccsd',
                'mean_field_homo_lumo': {'gap': 0., 'frontier_degeneracy': {'status': 'available', 'score': 1.}},
                'spectrum': {'levels': [0., 0.]},
                'spin_correlation': [[0., -0.5], [-0.5, 0.]],
                'charge_correlation': [[0., -0.5], [-0.5, 0.]],
            }
            if converged is not None:
                result['converged'] = converged
            diagnostics = _compute_strong_correlation_diagnostics(hubbard_dimer_spec(), result)
            self.assertEqual(diagnostics['physics_level'], 'unknown')
            self.assertIsNone(diagnostics['score'])
            self.assertEqual(diagnostics['confidence'], 'none')
            self.assertEqual(diagnostics['level'], 'strong' if converged is False else 'unknown')
            for item in diagnostics['diagnostics']:
                if item['category'] == 'context':
                    self.assertIsNone(item['score'])

    def test_nonfinite_solver_evidence_and_inconsistent_local_density_are_stress(self):
        for result in (
            {'double_excitation_amplitude_summary': {'max_abs_t2': float('nan')}},
            {'correlation_energy': float('inf'), 'reference_energy': 0.},
            {'dmet_local_observables': {'double_occupancy': [0.1], 'double_occupancy_reference': [0.]}},
        ):
            with self.subTest(result=result):
                diagnostics = _compute_strong_correlation_diagnostics(
                    hubbard_dimer_spec(), {'solver': 'ccsd', 'converged': True, **result},
                )
                self.assertEqual(diagnostics['physics_level'], 'unknown')
                self.assertEqual(diagnostics['solver_stress_level'], 'strong')

    def test_nonconvergence_and_amplitudes_do_not_raise_physics_score(self):
        summary = _natural_occupation_summary_from_dm1([np.diag([1., 0.]), np.diag([0., 1.])])
        for stress in ({'converged': False}, {'double_excitation_amplitude_summary': {'max_abs_t2': 2.}}):
            diagnostics = _compute_strong_correlation_diagnostics(
                hubbard_dimer_spec(), {'solver': 'ccsd', 'natural_occupation_summary': summary, **stress},
            )
            self.assertEqual(diagnostics['physics_level'], 'weak')
            self.assertEqual(diagnostics['physics_score'], 0.)
            self.assertEqual(diagnostics['solver_stress_level'], 'strong')
            self.assertIn('correlation evidence: weak', diagnostics['summary'])

    def test_local_double_occupancy_uses_polarized_populations(self):
        # Each spin density is an idempotent projector with opposite local magnetization.
        alpha = np.array([[0.9, 0.3], [0.3, 0.1]])
        beta = np.array([[0.1, 0.3], [0.3, 0.9]])
        summary = _natural_occupation_summary_from_dm1([alpha, beta])
        diagnostics = _compute_strong_correlation_diagnostics(
            hubbard_dimer_spec(), {'solver': 'fci', 'natural_occupation_summary': summary, 'double_occupancy': [0.09, 0.09]},
        )
        double = _item(diagnostics, 'double_occupancy_suppression')
        self.assertAlmostEqual(double['value']['reference_uncorrelated_estimate'], 0.09)
        self.assertAlmostEqual(double['score'], 0.)
        self.assertEqual(diagnostics['physics_level'], 'weak')

    def test_zero_baselines_and_missing_sites_are_not_full_suppression(self):
        summary = _natural_occupation_summary_from_dm1([np.eye(2), np.zeros((2, 2))])
        diagnostics = _compute_strong_correlation_diagnostics(
            hubbard_dimer_spec(), {'solver': 'fci', 'natural_occupation_summary': summary, 'double_occupancy': [0., 0.]},
        )
        double = _item(diagnostics, 'double_occupancy_suppression')
        self.assertIsNone(double['score'])
        self.assertEqual(double['value']['zero_reference_site_count'], 2)
        self.assertEqual(double['value']['mean_double_occupancy'], 0.)
        diagnostics = _compute_strong_correlation_diagnostics(hubbard_dimer_spec(), {
            'solver': 'dmet', 'converged': True,
            'dmet_local_observables': {
                'double_occupancy': [0.1, None], 'mean_double_occupancy': 0.1,
                'double_occupancy_reference': [0.2, None],
                'double_occupancy_reference_source': 'same_fragment_spin_1rdm',
                'coverage': {'double_occupancy_fraction': 0.5},
            },
        })
        double = _item(diagnostics, 'double_occupancy_suppression')
        self.assertEqual(double['score'], 0.5)
        self.assertEqual(double['value']['evaluated_site_count'], 1)
        self.assertEqual(double['value']['coverage']['double_occupancy_fraction'], 0.5)
        self.assertEqual(diagnostics['confidence'], 'low')

    def test_dmet_uses_fragment_density_for_fragment_double_occupancy(self):
        # Deliberately different global populations must not set the local baseline.
        arrays = {
            'global_embedding_density_matrix_full': np.array([np.eye(2) * 0.5] * 2),
            'fragment_density_matrix': np.array([[[0.9]], [[0.1]]]),
            'fragment_spin_rdm2': np.array([[[[[0.]]]], [[[[0.]]]], [[[[0.09]]]]]),
        }
        local = assemble_dmet_local_observables(hubbard_dimer_spec(), {
            'configuration': {'reference': 'unrestricted'}, 'fragments': [{'site_ids': [0]}],
        }, arrays)
        self.assertAlmostEqual(local['double_occupancy_reference'][0], 0.09)
        self.assertIsNone(local['double_occupancy_reference'][1])
        diagnostics = _compute_strong_correlation_diagnostics(hubbard_dimer_spec(), {
            'solver': 'dmet', 'dmet_local_observables': local,
            'dmet_result': {'configuration': {'impurity_solver_options': {'beta': 1000.}}},
        })
        self.assertAlmostEqual(_item(diagnostics, 'double_occupancy_suppression')['score'], 0.)
        beta = _item(diagnostics, 'impurity_scf_smearing')
        self.assertEqual(beta['category'], 'context')
        self.assertIsNone(beta['score'])

    def test_dimer_fci_matches_analytic_correlation_trend(self):
        scores = []
        for u in (0., 2., 4., 8., 16.):
            with self.subTest(u=u), contextlib.redirect_stdout(io.StringIO()):
                spec = hubbard_dimer_spec()
                for site in spec['sites']:
                    site['U'] = u
                result = run_model_hamiltonian_solver(spec, 'fci', ['energy', 'strong_correlation_diagnostics'])
                diagnostics = result['strong_correlation_diagnostics']
                radical = math.sqrt(u * u + 16.)
                self.assertAlmostEqual(result['energy'], (u - radical) / 2., places=8)
                self.assertAlmostEqual(_item(diagnostics, 'double_occupancy_suppression')['score'], u / radical, places=6)
                self.assertAlmostEqual(result['natural_occupation_summary']['spin_resolved']['score'], u * u / (u * u + 16.), places=8)
                scores.append(diagnostics['physics_score'])
        self.assertEqual(scores, sorted(scores))
        self.assertAlmostEqual(scores[0], 0.)
        self.assertGreater(scores[-1], 0.9)

    def test_assembled_dmet_density_is_informational_not_a_pure_state_score(self):
        summary = _natural_occupation_summary_from_dm1(
            [np.eye(2) * 0.5] * 2, scope='assembled_dmet_lattice',
        )
        diagnostics = _compute_strong_correlation_diagnostics(hubbard_dimer_spec(), {
            'solver': 'dmet', 'converged': True, 'natural_occupation_summary': summary,
        })
        self.assertEqual(summary['spin_resolved']['score'], 1.)
        self.assertEqual(diagnostics['physics_level'], 'unknown')
        self.assertIsNone(_item(diagnostics, 'natural_orbital_occupations')['score'])

    def test_attractive_dimer_enhancement_is_not_scored_as_weak_suppression(self):
        spec = hubbard_dimer_spec()
        for site in spec['sites']:
            site['U'] = -8.
        with contextlib.redirect_stdout(io.StringIO()):
            result = run_model_hamiltonian_solver(spec, 'fci', ['energy', 'strong_correlation_diagnostics'])
        diagnostics = result['strong_correlation_diagnostics']
        self.assertEqual(diagnostics['physics_level'], 'strong')
        double = _item(diagnostics, 'double_occupancy_suppression')
        self.assertGreater(double['value']['connected_opposite_spin_density'], 0.)
        self.assertIsNone(double['score'])

    def test_noninteracting_open_shell_doped_and_polarized_chains(self):
        for nalpha, nbeta, u in ((2, 2, 0), (2, 1, 0), (3, 0, 0), (4, 0, 16)):
            with self.subTest(nelec=(nalpha, nbeta), u=u), contextlib.redirect_stdout(io.StringIO()):
                spec = hubbard_chain_spec(site_count=4, nalpha=nalpha, nbeta=nbeta, t=-1, u=u)
                result = run_model_hamiltonian_solver(spec, 'fci', ['energy', 'strong_correlation_diagnostics'])
                diagnostics = result['strong_correlation_diagnostics']
                self.assertEqual(diagnostics['physics_level'], 'weak')
                self.assertAlmostEqual(diagnostics['physics_score'], 0., places=6)


if __name__ == '__main__':
    unittest.main()
