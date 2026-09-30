#!/usr/bin/env python

from __future__ import annotations

import unittest

try:
    import numpy as np
except ImportError:  # pragma: no cover - numpy is present with PySCF in normal test envs
    np = None

from pyscf_agent.contracts import ActiveSpaceSpec, SCNEVPT2Spec
from pyscf_agent.backend.correlation.chemical_valence import chemical_valence_targets
from pyscf_agent.backend.correlation.molecular import build_molecular_correlation_risk
from pyscf_agent.backend.correlation.active_space_selection import (
    evaluate_active_space_candidates,
    recommend_active_space_candidate,
)
from pyscf_agent.backend.correlation.active_space import (
    _compact_chemical_avas_projection,
    _unique_projection_mapping,
)
from pyscf_agent.backend.correlation.strong import (
    _natural_orbital_summary,
    build_correlation_diagnostics,
    build_scf_stability_summary,
    propose_active_space,
    run_cas_method,
)


class FakeMol:
    def ao_labels(self, fmt=False):  # pylint: disable=unused-argument
        return [
            (0, 'H', '1s', ''),
            (1, 'H', '1s', ''),
            (2, 'H', '1s', ''),
            (3, 'H', '1s', ''),
        ]


class FakeMeanField:
    def __init__(self):
        self.mol = FakeMol()
        self.mo_energy = [-1.0, -0.2, 0.25, 1.0]
        self.mo_occ = [2.0, 2.0, 0.0, 0.0]
        self.mo_coeff = np.eye(4)
        self.converged = True

    def spin_square(self):
        return 0.0


class FakeMP2:
    def __init__(self):
        self.t2 = np.zeros((2, 2, 2, 2))
        self.t2[1, 1, 0, 0] = 0.12

    def make_rdm1(self):
        return np.diag([1.99, 1.45, 0.55, 0.01])


class FakeCCSD:
    def __init__(self, amplitude=0.12):
        self.t1 = np.array([[amplitude, 0.0], [0.0, 0.0]])
        self.t2 = np.zeros((2, 2, 2, 2))
        self.converged = True

    def make_rdm1(self):
        return np.diag([1.98, 1.42, 0.58, 0.02])


@unittest.skipIf(np is None, 'numpy is required for active-space audit tests')
class ActiveSpaceAuditTests(unittest.TestCase):
    def test_chemical_valence_targets_preserve_transition_metal_d_and_s_manifolds(self):
        class ChromiumMol:
            natm = 2
            charge = 0

            @staticmethod
            def atom_symbol(index):
                return ('Cr', 'Cr')[index]

            @staticmethod
            def ao_labels(fmt=False):  # pylint: disable=unused-argument
                return [
                    (0, 'Cr', '3d', 'xy'),
                    (0, 'Cr', '4s', ''),
                    (0, 'Cr', '4p', 'x'),
                    (1, 'Cr', '3d', 'xy'),
                    (1, 'Cr', '4s', ''),
                ]

        targets = chemical_valence_targets(ChromiumMol())

        self.assertEqual(targets['primary_targets'], ['Cr 3d', 'Cr 4s'])
        self.assertEqual(targets['elements']['Cr']['primary_shells'], ['3d', '4s'])
        self.assertEqual(targets['primary_orbital_count'], 12)
        self.assertEqual(targets['primary_electron_count'], 12)

    def test_chemical_avas_projection_compacts_weak_split_partners(self):
        avas_mo = np.eye(6)
        projection = np.zeros((6, 3))
        projection[1, 0] = np.sqrt(0.10)
        projection[2, 1] = np.sqrt(0.90)
        projection[3, 2] = np.sqrt(0.80)

        compact_mo, ncas, nelecas, active_start, metadata = _compact_chemical_avas_projection(
            avas_mo,
            projection,
            ncas=3,
            nelecas=4,
            active_column_start=1,
            target_summary={
                'primary_orbital_count': 2,
                'primary_electron_count': 2,
            },
            spin=0,
        )

        self.assertEqual(ncas, 2)
        self.assertEqual(nelecas, 2)
        self.assertEqual(active_start, 2)
        self.assertEqual(metadata['status'], 'applied')
        self.assertEqual(metadata['removed_occupied_active_indices'], [0])
        self.assertTrue(np.allclose(compact_mo, np.eye(6)))

    def test_projection_mapping_assigns_distinct_canonical_orbitals(self):
        projection = np.asarray([
            [0.9, 0.8],
            [0.7, 0.1],
            [0.0, 0.0],
        ])

        mapping = _unique_projection_mapping(projection)

        self.assertEqual(len(mapping), 2)
        self.assertEqual(len({item[1] for item in mapping}), 2)

    def test_mp2_natural_occupations_and_t2_build_active_space_audit(self):
        active_space = ActiveSpaceSpec(
            enabled=True,
            selection_method='occupation_window',
            occupation_window=(0.02, 1.98),
        )

        proposal = propose_active_space(
            FakeMeanField(),
            active_space,
            post_hf=FakeMP2(),
            localization_method='boys',
        )

        self.assertEqual(proposal['orbital_indices'], [1, 2])
        self.assertEqual(proposal['ncas'], 2)
        self.assertEqual(proposal['nelecas'], 2)
        self.assertFalse(proposal['approved'])
        audit = proposal['audit']
        self.assertEqual(audit['orbital_index_basis'], 'canonical_mo')
        self.assertEqual(audit['natural_orbital_status'], 'available')
        self.assertEqual(audit['localization_method'], 'boys')
        self.assertEqual(audit['selected_candidate_method'], 'merged')
        candidate_methods = [item['method'] for item in audit['candidate_active_spaces']]
        self.assertEqual(set(candidate_methods), {
            'manual',
            'uno',
            'chemical_valence',
            'evidence_expanded',
            'frontier_fallback',
            'avas',
            'merged',
        })
        reviewable_sizes = [
            item['ncas']
            for item in audit['candidate_active_spaces']
            if item.get('status') in ('available', 'mapping_ambiguous')
        ]
        self.assertEqual(reviewable_sizes, sorted(reviewable_sizes))
        candidates = {item['method']: item for item in audit['candidate_active_spaces']}
        self.assertEqual(candidates['avas']['status'], 'not_requested')
        self.assertEqual(candidates['merged']['orbital_indices'], [1, 2])
        self.assertTrue(audit['ncas_nelecas_consistency']['consistent'])
        selected = {row['index']: row for row in audit['selected_orbitals']}
        self.assertIn('mp2_natural_occupation_window', ';'.join(selected[1]['selection_reasons']))
        self.assertIn('t2_supplement', ';'.join(selected[1]['selection_reasons']))
        self.assertAlmostEqual(selected[1]['t2_importance'], 0.12)
        self.assertTrue(selected[1]['atom_contributions'])
        self.assertEqual(audit['manual_approval']['status'], 'requires_user_review')

    def test_nonphysical_mp2_density_does_not_create_an_executable_frontier_fallback(self):
        class N2LikeMeanField(FakeMeanField):
            def __init__(self):
                self.mol = FakeMol()
                self.mo_energy = [-1.0, -0.8, -0.4, 0.1, 0.2, 0.3]
                self.mo_occ = [2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
                self.mo_coeff = np.eye(6)
                self.converged = True

        class PathologicalMP2:
            def __init__(self):
                self.t2 = np.full((3, 3, 3, 3), 0.08)

            def make_rdm1(self):
                return np.diag([2.17, 2.17, 1.36, 0.64, -0.16, -0.16])

        active_space = ActiveSpaceSpec(
            enabled=True,
            selection_method='occupation_window',
            occupation_window=(0.02, 1.98),
        )

        proposal = propose_active_space(
            N2LikeMeanField(),
            active_space,
            post_hf=PathologicalMP2(),
        )

        self.assertEqual(proposal['orbital_indices'], [])
        self.assertIsNone(proposal['ncas'])
        self.assertIsNone(proposal['nelecas'])
        self.assertIsNone(proposal['audit_summary']['estimated_nelecas'])
        self.assertTrue(proposal['audit_summary']['ncas_nelecas_consistent'])
        self.assertEqual(proposal['audit']['selected_candidate_method'], 'unresolved')
        self.assertEqual(proposal['audit']['natural_occupation_quality']['status'], 'out_of_bounds')
        self.assertEqual(proposal['audit']['t2_supplement']['status'], 'skipped_unphysical_natural_occupations')
        candidates = {item['method']: item for item in proposal['audit']['candidate_active_spaces']}
        self.assertEqual(candidates['frontier_fallback']['status'], 'diagnostic_only')

    def test_unrestricted_post_hf_density_is_combined_in_the_ao_basis(self):
        class UHFMol:
            nelectron = 2

        class UHFMeanField:
            mol = UHFMol()
            mo_coeff = np.asarray([
                np.eye(2),
                np.asarray([[0.0, 1.0], [1.0, 0.0]]),
            ])

            @staticmethod
            def get_ovlp():
                return np.eye(2)

        class UMP2Density:
            @staticmethod
            def make_rdm1():
                return np.diag([1.0, 0.0]), np.diag([1.0, 0.0])

        summary = _natural_orbital_summary(UMP2Density(), mf=UHFMeanField())

        self.assertEqual(summary['representation'], 'spin_summed_ao')
        self.assertAlmostEqual(summary['occupation_sum'], 2.0)
        self.assertEqual(
            [round(item['occupation'], 8) for item in summary['orbitals']],
            [1.0, 1.0],
        )

    def test_final_cas_natural_occupations_promote_molecular_risk(self):
        diagnostics = build_correlation_diagnostics(
            FakeMeanField(),
            FakeMol(),
            correlated_natural_occupations=[1.05, 1.01, 0.99, 0.95],
        )

        self.assertEqual(diagnostics['level'], 'strong')
        self.assertEqual(diagnostics['level_reason'], 'correlated_natural_occupations')
        self.assertIn('correlated_natural_occupations', diagnostics['warnings'])
        summary = diagnostics['correlated_natural_occupation_summary']
        self.assertEqual(summary['source'], 'cas_result.natural_occupations')

    def test_t2_supplement_rejects_remote_virtual_orbitals(self):
        class MeanFieldWithRemoteVirtual(FakeMeanField):
            def __init__(self):
                self.mol = FakeMol()
                self.mo_energy = [-1.2, -0.8, -0.4, 0.1, 0.3, 1.8]
                self.mo_occ = [2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
                self.mo_coeff = np.eye(6)
                self.converged = True

        class MP2WithRemoteAmplitude:
            def __init__(self):
                self.t2 = np.zeros((3, 3, 3, 3))
                self.t2[1, 1, 2, 2] = 0.2

            def make_rdm1(self):
                return np.diag([1.99, 1.45, 2.0, 0.55, 0.01, 0.0])

        proposal = propose_active_space(
            MeanFieldWithRemoteVirtual(),
            ActiveSpaceSpec(enabled=True, selection_method='occupation_window', occupation_window=(0.02, 1.98)),
            post_hf=MP2WithRemoteAmplitude(),
        )

        self.assertEqual(proposal['orbital_indices'], [1, 3])
        excluded = proposal['audit']['t2_supplement']['excluded']
        self.assertTrue(any(item['index'] == 5 for item in excluded))
        self.assertEqual(proposal['audit']['t2_supplement']['status'], 'available')

    def test_small_block2_active_space_defaults_to_screening_preset(self):
        proposal = propose_active_space(
            FakeMeanField(),
            ActiveSpaceSpec(
                enabled=True,
                selection_method='occupation_window',
                occupation_window=(0.02, 1.98),
                target_method='casscf',
                target_solver='block2_dmrg',
                target_solver_options={'nroots': 1},
            ),
            post_hf=FakeMP2(),
        )

        self.assertEqual(proposal['ncas'], 2)
        self.assertEqual(proposal['nelecas'], 2)
        self.assertEqual(proposal['target_solver_options']['preset'], 'screening')
        self.assertEqual(proposal['target_solver_options']['nroots'], 1)
        self.assertEqual(
            proposal['audit']['method_recommendation']['target_solver_options']['preset'],
            'screening',
        )

    def test_explicit_block2_active_space_preset_is_preserved(self):
        proposal = propose_active_space(
            FakeMeanField(),
            ActiveSpaceSpec(
                enabled=True,
                selection_method='occupation_window',
                occupation_window=(0.02, 1.98),
                target_method='casscf',
                target_solver='block2_dmrg',
                target_solver_options={'preset': 'high_accuracy'},
            ),
            post_hf=FakeMP2(),
        )

        self.assertEqual(proposal['target_solver_options']['preset'], 'high_accuracy')

    def test_run_cas_method_rejects_sc_nevpt2_excited_state_root_directly(self):
        from pyscf import gto, scf  # pylint: disable=import-outside-toplevel

        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        active_space = ActiveSpaceSpec(
            enabled=True,
            selection_method='manual',
            ncas=2,
            nelecas=2,
            orbital_indices=[0, 1],
            approved=True,
        )

        with self.assertRaisesRegex(ValueError, 'supports root=0 only'):
            run_cas_method(
                mf,
                mol,
                'casscf',
                active_space,
                max_cycle=4,
                sc_nevpt2=SCNEVPT2Spec(enabled=True, root=1, density_fit=True),
            )

    def test_manual_active_space_records_inconsistent_requested_contract(self):
        active_space = ActiveSpaceSpec(
            enabled=True,
            selection_method='manual',
            ncas=4,
            nelecas=2,
            orbital_indices=[0, 1],
            approved=True,
        )

        proposal = propose_active_space(FakeMeanField(), active_space)

        self.assertEqual(proposal['orbital_indices'], [0, 1])
        self.assertFalse(proposal['audit']['ncas_nelecas_consistency']['consistent'])
        self.assertIn('ncas=4', proposal['audit']['ncas_nelecas_consistency']['messages'][0])
        self.assertEqual(proposal['audit']['manual_approval']['status'], 'approved')
        self.assertEqual(proposal['audit']['selected_candidate_method'], 'manual')
        candidates = {item['method']: item for item in proposal['audit']['candidate_active_spaces']}
        self.assertEqual(candidates['manual']['orbital_indices'], [0, 1])
        self.assertEqual(candidates['merged']['orbital_indices'], [0, 1])

    def test_chemical_baseline_is_preferred_over_a_large_threshold_union(self):
        evidence = {
            'signals': [
                {'source': 'natural_occupation', 'orbital_index': 4, 'weight': 0.20},
                {'source': 'natural_occupation', 'orbital_index': 5, 'weight': 0.20},
                {'source': 't2_amplitude', 'orbital_index': 20, 'weight': 0.12},
            ],
            'total_weight': 0.52,
        }
        candidates = [
            {
                'method': 'chemical_valence',
                'role': 'baseline',
                'status': 'available',
                'orbital_indices': list(range(12)),
                'ncas': 12,
                'estimated_nelecas': 12,
            },
            {
                'method': 'evidence_expanded',
                'role': 'expanded',
                'status': 'available',
                'orbital_indices': list(range(12)) + [20],
                'ncas': 13,
                'estimated_nelecas': 12,
            },
            {
                'method': 'merged',
                'role': 'broad',
                'status': 'available',
                'orbital_indices': list(range(34)),
                'ncas': 34,
                'estimated_nelecas': 18,
            },
        ]

        evaluated = evaluate_active_space_candidates(
            candidates,
            evidence,
            chemical_baseline_indices=list(range(12)),
            spin=0,
            target_solver='block2_dmrg',
        )
        recommended, decision = recommend_active_space_candidate(
            evaluated,
            selection_method='occupation_window',
            recovery_candidate_method=None,
        )

        self.assertEqual(decision['selected_candidate_method'], 'chemical_valence')
        self.assertTrue(next(item for item in recommended if item['method'] == 'chemical_valence')['recommended'])
        self.assertFalse(next(item for item in recommended if item['method'] == 'merged')['recommended'])

    def test_automatic_selection_does_not_truncate_to_a_stale_requested_ncas(self):
        proposal = propose_active_space(
            FakeMeanField(),
            ActiveSpaceSpec(
                enabled=True,
                selection_method='occupation_window',
                occupation_window=(0.02, 1.98),
                ncas=1,
            ),
            post_hf=FakeMP2(),
        )

        self.assertEqual(proposal['orbital_indices'], [1, 2])
        self.assertEqual(proposal['ncas'], 2)
        self.assertEqual(proposal['audit']['selection_parameters']['requested_ncas'], 1)
        reasons = [
            reason
            for orbital in proposal['audit']['orbitals']
            for reason in orbital.get('selection_reasons') or []
        ]
        self.assertFalse(any('truncated_by_requested_ncas' in reason for reason in reasons))

    def test_critical_missing_evidence_is_retained_as_the_next_larger_candidate(self):
        evidence = {
            'signals': [
                {'source': 'natural_occupation', 'orbital_index': 4, 'weight': 0.20},
                {'source': 'natural_occupation', 'orbital_index': 14, 'weight': 0.40},
            ],
            'total_weight': 0.60,
        }
        candidates = [
            {
                'method': 'chemical_valence',
                'role': 'baseline',
                'status': 'available',
                'orbital_indices': list(range(12)),
                'ncas': 12,
                'estimated_nelecas': 12,
            },
            {
                'method': 'evidence_expanded',
                'role': 'expanded',
                'status': 'available',
                'orbital_indices': list(range(21)),
                'ncas': 21,
                'estimated_nelecas': 12,
            },
            {
                'method': 'merged',
                'role': 'broad',
                'status': 'available',
                'orbital_indices': list(range(20)),
                'ncas': 20,
                'estimated_nelecas': 14,
            },
        ]

        evaluated = evaluate_active_space_candidates(
            candidates,
            evidence,
            chemical_baseline_indices=list(range(12)),
            spin=0,
            target_solver='block2_dmrg',
        )
        recommended, decision = recommend_active_space_candidate(
            evaluated,
            selection_method='occupation_window',
            recovery_candidate_method=None,
        )

        self.assertEqual(decision['selection_policy'], 'smallest_chemically_complete_first')
        self.assertEqual(decision['selected_candidate_method'], 'chemical_valence')
        self.assertEqual(decision['expansion_candidate_method'], 'evidence_expanded')
        self.assertEqual(decision['critical_missing_orbital_indices'], [14])
        selected = next(item for item in recommended if item['recommended'])
        self.assertEqual(selected['orbital_indices'], list(range(12)))
        expanded = next(item for item in recommended if item['method'] == 'evidence_expanded')
        self.assertEqual(expanded['recommendation_stage'], 'next_expansion')
        self.assertEqual(expanded['orbital_indices'], list(range(21)))
        reviewable_sizes = [
            item['ncas']
            for item in recommended
            if item.get('status') in ('available', 'mapping_ambiguous')
        ]
        self.assertEqual(reviewable_sizes, sorted(reviewable_sizes))

    def test_smallest_complete_policy_is_independent_of_candidate_scale(self):
        for baseline_ncas, expanded_ncas, broad_ncas in ((4, 6, 9), (8, 11, 16)):
            with self.subTest(
                baseline_ncas=baseline_ncas,
                expanded_ncas=expanded_ncas,
                broad_ncas=broad_ncas,
            ):
                evidence = {
                    'signals': [
                        {
                            'source': 'natural_occupation',
                            'orbital_index': baseline_ncas,
                            'weight': 0.40,
                        },
                    ],
                    'total_weight': 0.40,
                }
                candidates = [
                    {
                        'method': 'merged',
                        'role': 'broad',
                        'status': 'available',
                        'orbital_indices': list(range(broad_ncas)),
                        'ncas': broad_ncas,
                        'estimated_nelecas': baseline_ncas,
                    },
                    {
                        'method': 'evidence_expanded',
                        'role': 'expanded',
                        'status': 'available',
                        'orbital_indices': list(range(expanded_ncas)),
                        'ncas': expanded_ncas,
                        'estimated_nelecas': baseline_ncas,
                    },
                    {
                        'method': 'chemical_valence',
                        'role': 'baseline',
                        'status': 'available',
                        'orbital_indices': list(range(baseline_ncas)),
                        'ncas': baseline_ncas,
                        'estimated_nelecas': baseline_ncas,
                    },
                ]

                evaluated = evaluate_active_space_candidates(
                    candidates,
                    evidence,
                    chemical_baseline_indices=list(range(baseline_ncas)),
                    spin=0,
                    target_solver='block2_dmrg',
                )
                recommended, decision = recommend_active_space_candidate(
                    evaluated,
                    selection_method='occupation_window',
                    recovery_candidate_method=None,
                )

                self.assertEqual(decision['selected_candidate_method'], 'chemical_valence')
                self.assertEqual(decision['expansion_candidate_method'], 'evidence_expanded')
                self.assertEqual(
                    [item['ncas'] for item in recommended],
                    [baseline_ncas, expanded_ncas, broad_ncas],
                )

    def test_molecular_correlation_risk_splits_physics_and_solver_stress(self):
        scf_stability = {
            'status': 'completed',
            'stable': False,
            'internal_stable': False,
            'external_stable': True,
        }

        diagnostics = build_correlation_diagnostics(
            FakeMeanField(),
            FakeMol(),
            post_hf=FakeMP2(),
            scf_stability=scf_stability,
            reference_energy=-1.0,
            correlation_energy=-0.2,
        )

        self.assertEqual(diagnostics['kind'], 'molecular_correlation_diagnostics')
        risk = diagnostics['molecular_correlation_risk']
        self.assertEqual(risk['kind'], 'molecular_correlation_risk')
        self.assertIn('physics_score', risk)
        self.assertIn('physics_level', risk)
        self.assertIn('solver_stress_score', risk)
        self.assertIn('solver_stress_level', risk)
        self.assertGreaterEqual(risk['physics_score'], 0.0)
        self.assertGreater(risk['solver_stress_score'], 0.0)
        self.assertEqual(diagnostics['scf_stability']['stable'], False)
        self.assertIn('method_recommendation', risk)

    def test_molecular_nonconvergence_promotes_solver_stress_without_overwriting_physics(self):
        mean_field = FakeMeanField()
        mean_field.converged = False

        diagnostics = build_correlation_diagnostics(mean_field, FakeMol())
        risk = diagnostics['molecular_correlation_risk']

        self.assertEqual(risk['level'], 'strong')
        self.assertEqual(risk['level_reason'], 'solver_nonconvergence')
        self.assertEqual(risk['solver_stress_level'], 'strong')
        self.assertEqual(diagnostics['physics_level'], risk['physics_level'])
        self.assertEqual(diagnostics['solver_stress_level'], 'strong')

    def test_normal_somos_do_not_force_open_shell_multireference_routing(self):
        for spin in (1, 2, -1):
            for unrestricted in (False, True):
                with self.subTest(spin=spin, unrestricted=unrestricted):
                    mf = FakeMeanField()
                    mf.mol.spin = spin
                    mf.mol.nelectron = 2 + abs(spin)
                    occupations = np.array([2.] + [1.] * abs(spin) + [0.])
                    mf.mo_coeff = np.eye(len(occupations))
                    mf.mo_occ = occupations
                    mf.mo_energy = np.linspace(-1., 1., len(occupations))
                    mf.spin_square = lambda: (abs(spin) / 2.) * (abs(spin) / 2. + 1.)
                    if unrestricted:
                        alpha = (occupations > 0).astype(float)
                        beta = (occupations > 1).astype(float)
                        if spin < 0:
                            alpha, beta = beta, alpha
                        mf.mo_occ = np.array([alpha, beta])
                        mf.mo_coeff = np.array([mf.mo_coeff, mf.mo_coeff])
                        mf.mo_energy = np.array([mf.mo_energy, mf.mo_energy])
                        mf.get_ovlp = lambda: np.eye(len(occupations))
                        mf.make_rdm1 = lambda: np.array([np.diag(alpha), np.diag(beta)])
                    correlated = occupations.copy()
                    correlated[0] -= .01
                    correlated[-1] += .01
                    diagnostics = build_correlation_diagnostics(
                        mf, mf.mol, correlated_natural_occupations=correlated.tolist(), current_method='mp2',
                    )
                    risk = diagnostics['molecular_correlation_risk']
                    components = {item['name']: item for item in risk['physics_components']}
                    self.assertIsNone(components['open_shell']['score'])
                    self.assertAlmostEqual(components['mean_field_fractional_occupations']['score'], 0.)
                    metrics = risk['correlated_natural_occupation_metrics']
                    self.assertAlmostEqual(metrics['max_fractionality'], 1.)
                    self.assertAlmostEqual(metrics['routing']['score'], .01)
                    self.assertEqual(metrics['routing']['reference_occupations'], occupations.tolist())
                    self.assertNotEqual(risk['routing_level'], 'strong')
                    self.assertNotIn('casscf', diagnostics['method_recommendation']['preferred'])
                    self.assertNotIn('correlated_natural_occupations', diagnostics['warnings'])
                    self.assertNotIn('fractional_orbital_occupations', diagnostics['warnings'])
                    # The raw SOMOs remain available to active-space selection.
                    self.assertEqual(len(diagnostics['fractional_occupations']), abs(spin))

    def test_real_open_shell_densities_and_stretched_singlet_keep_distinct_routes(self):
        from pyscf import fci, gto, mp, scf

        cases = [
            ('CH3', 'C 0 0 0; H 1.079 0 0; H -.5395 .93444 0; H -.5395 -.93444 0', 1),
            ('triplet O2', 'O 0 0 0; O 0 0 1.21', 2),
        ]
        for label, atom, spin in cases:
            with self.subTest(molecule=label):
                mol = gto.M(atom=atom, basis='sto-3g', spin=spin, verbose=0)
                mf = scf.UHF(mol).run(conv_tol=1.e-10)
                self.assertTrue(mf.converged)
                post_hf = mp.UMP2(mf).run()
                diagnostics = build_correlation_diagnostics(mf, mol, post_hf=post_hf, current_method='mp2')
                metrics = diagnostics['molecular_correlation_risk']['correlated_natural_occupation_metrics']
                self.assertEqual(metrics['status'], 'available')
                self.assertGreater(metrics['max_fractionality'], .95)
                self.assertLess(metrics['routing']['score'], .1)
                self.assertNotEqual(diagnostics['routing_level'], 'strong')
                self.assertNotIn('casscf', diagnostics['method_recommendation']['preferred'])

        mol = gto.M(atom='H 0 0 0; H 0 0 3', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        solver = fci.FCI(mf).run()
        occupations = np.linalg.eigvalsh(solver.make_rdm1(solver.ci, mol.nao_nr(), mol.nelec)).tolist()
        diagnostics = build_correlation_diagnostics(mf, mol, correlated_natural_occupations=occupations)
        self.assertEqual(diagnostics['physics_level'], 'strong')
        self.assertEqual(diagnostics['routing_level'], 'strong')

    def test_open_shell_additional_fractionality_still_promotes_multireference(self):
        for spin in (1, 2):
            with self.subTest(spin=spin):
                mf = FakeMeanField()
                mf.mol.spin = spin
                occupations = [1.] * (abs(spin) + 2)
                for spectrum in (occupations, [2.] + occupations + [0.]):
                    risk = build_molecular_correlation_risk(
                        mf, mf.mol, correlated_natural_occupations=spectrum, spin_square=spin / 2. * (spin / 2. + 1.),
                    )
                    self.assertEqual(risk['physics_level'], 'strong')
                    self.assertEqual(risk['routing_level'], 'strong')
                    self.assertEqual(risk['level_reason'], 'correlated_natural_occupations')

    def test_missing_uno_is_unavailable_evidence_for_broken_symmetry(self):
        mf = FakeMeanField()
        mf.mol.spin = 0
        mf.mo_occ = np.array([[1., 0.], [1., 0.]])
        mf.mo_energy = np.array([[-1., 1.], [-1., 1.]])
        mf.mo_coeff = np.array([np.eye(2), np.eye(2)])
        mf.spin_square = lambda: 1.
        mf.make_rdm1 = lambda: np.array([np.diag([1., 0.]), np.diag([0., 1.])])
        mf.get_ovlp = lambda: np.zeros((2, 2))

        for risk in (
            build_molecular_correlation_risk(mf, mf.mol, fractional_occupations=None),
            build_correlation_diagnostics(mf, mf.mol)['molecular_correlation_risk'],
        ):
            component = next(item for item in risk['physics_components'] if item['name'] == 'mean_field_fractional_occupations')
            self.assertEqual(component['status'], 'unavailable')
            self.assertIsNone(component['score'])
            self.assertNotEqual(risk['level_reason'], 'uhf_broken_symmetry_uno')
            self.assertTrue(risk['reference_spin_symmetry']['broken_symmetry'])

    def test_nonphysical_correlated_density_routes_through_ccsd_before_cas(self):
        diagnostics = build_correlation_diagnostics(
            FakeMeanField(),
            FakeMol(),
            correlated_natural_occupations=[3.1, 1.0, 0.0],
            current_method='mp2',
        )
        risk = diagnostics['molecular_correlation_risk']

        self.assertEqual(risk['physics_level'], 'weak')
        self.assertEqual(risk['solver_stress_level'], 'strong')
        self.assertEqual(risk['level'], 'strong')
        self.assertEqual(risk['level_reason'], 'nonphysical_correlated_density')
        self.assertEqual(risk['routing_level'], 'moderate')
        self.assertIn('nonphysical_correlated_density', diagnostics['warnings'])
        self.assertEqual(diagnostics['method_recommendation']['preferred'], ['ccsd'])
        self.assertNotIn('review_active_space', diagnostics['recommended_next_steps'])
        self.assertIn('run_ccsd_diagnostics', diagnostics['recommended_next_steps'])

    def test_intermediate_natural_occupation_fractionality_is_moderate(self):
        diagnostics = build_correlation_diagnostics(
            FakeMeanField(),
            FakeMol(),
            correlated_natural_occupations=[1.4, 1.4, 0.6, 0.6],
            current_method='mp2',
        )
        risk = diagnostics['molecular_correlation_risk']

        self.assertEqual(risk['physics_level'], 'moderate')
        self.assertEqual(risk['level'], 'moderate')
        self.assertEqual(risk['routing_level'], 'moderate')
        self.assertEqual(
            risk['level_reason'],
            'correlated_natural_occupations_moderate',
        )
        self.assertEqual(risk['method_recommendation']['preferred'][0], 'ccsd')

    def test_unstable_scf_has_at_least_moderate_solver_stress(self):
        diagnostics = build_correlation_diagnostics(
            FakeMeanField(),
            FakeMol(),
            scf_stability={
                'status': 'completed',
                'stable': False,
                'internal_stable': False,
                'external_stable': False,
            },
        )

        self.assertEqual(diagnostics['solver_stress_level'], 'moderate')
        self.assertIn('scf_instability', diagnostics['warnings'])

    def test_completed_casscf_diagnostics_do_not_request_active_space_approval_again(self):
        diagnostics = build_correlation_diagnostics(
            FakeMeanField(),
            FakeMol(),
            correlated_natural_occupations=[1.8, 1.0, 1.0, 0.2],
            current_method='casscf',
        )

        self.assertNotIn('review_active_space', diagnostics['recommended_next_steps'])
        self.assertNotIn('consider_casci_or_casscf', diagnostics['recommended_next_steps'])
        self.assertIn('validate_active_space_completeness', diagnostics['recommended_next_steps'])
        self.assertIn('consider_post_cas_dynamic_correlation', diagnostics['recommended_next_steps'])
        self.assertEqual(diagnostics['method_recommendation']['preferred'], ['casscf'])
        self.assertIn(
            'current multireference treatment completed',
            diagnostics['method_recommendation']['next_step'],
        )

    def test_ccsd_t1_d1_promotes_multireference_routing(self):
        diagnostics = build_correlation_diagnostics(
            FakeMeanField(),
            FakeMol(),
            post_hf=FakeCCSD(amplitude=0.12),
            reference_energy=-1.0,
            correlation_energy=-0.2,
        )

        t1_d1 = diagnostics['ccsd_t1_d1']
        self.assertEqual(t1_d1['status'], 'available')
        self.assertAlmostEqual(t1_d1['t1'], 0.06)
        self.assertAlmostEqual(t1_d1['d1'], 0.12)
        self.assertTrue(t1_d1['promote_to_multireference'])
        self.assertEqual(diagnostics['level'], 'strong')
        self.assertEqual(
            diagnostics['molecular_correlation_risk']['level_reason'],
            'ccsd_t1_d1_promotion',
        )
        self.assertIn('ccsd_t1_d1', {
            item['name'] for item in diagnostics['molecular_correlation_risk']['solver_stress_components']
        })
        self.assertIn('casscf', diagnostics['method_recommendation']['preferred'])
        self.assertIn('ccsd_t1_d1_multireference_promotion', diagnostics['warnings'])

    def test_frontier_orbital_degeneracy_flags_closed_shell_clusters(self):
        class DegenerateMeanField(FakeMeanField):
            def __init__(self):
                super().__init__()
                self.mo_energy = [-1.0, -0.210, -0.205, 0.100, 0.108, 0.9]
                self.mo_occ = [2.0, 2.0, 2.0, 0.0, 0.0, 0.0]
                self.mo_coeff = np.eye(6)

        diagnostics = build_correlation_diagnostics(DegenerateMeanField(), FakeMol())

        frontier = diagnostics['frontier_orbital_degeneracy']
        self.assertEqual(frontier['status'], 'available')
        self.assertEqual(frontier['max_same_spin_degeneracy'], 2)
        self.assertGreaterEqual(frontier['score'], 0.5)
        self.assertIn('frontier_orbital_degeneracy', diagnostics['warnings'])
        component_names = {
            item['name']
            for item in diagnostics['molecular_correlation_risk']['physics_components']
        }
        self.assertIn('frontier_orbital_degeneracy', component_names)

    def test_frontier_orbital_degeneracy_checks_alpha_beta_pairs(self):
        class OpenShellMol(FakeMol):
            spin = 1

        class OpenShellMeanField(FakeMeanField):
            def __init__(self):
                super().__init__()
                self.mol = OpenShellMol()
                self.mo_energy = np.array([
                    [-1.0, -0.300, 0.100],
                    [-1.0, -0.305, 0.120],
                ])
                self.mo_occ = np.array([
                    [1.0, 1.0, 0.0],
                    [1.0, 0.0, 0.0],
                ])
                self.mo_coeff = np.array([np.eye(3), np.eye(3)])

            def spin_square(self):
                return 0.75

        diagnostics = build_correlation_diagnostics(OpenShellMeanField(), OpenShellMol())

        frontier = diagnostics['frontier_orbital_degeneracy']
        degenerate_pairs = {
            item['pair']
            for item in frontier['alpha_beta']['degenerate_pairs']
        }
        self.assertIn('alpha_homo-beta_lumo', degenerate_pairs)
        self.assertIn('alpha_beta_frontier_degeneracy', diagnostics['warnings'])

    def test_closed_shell_uhf_does_not_treat_spin_pair_as_fractional_or_degenerate(self):
        class ClosedShellMeanField(FakeMeanField):
            def __init__(self):
                super().__init__()
                self.mo_energy = np.array([
                    [-0.8, 0.3],
                    [-0.8, 0.3],
                ])
                self.mo_occ = np.array([
                    [1.0, 0.0],
                    [1.0, 0.0],
                ])
                self.mo_coeff = np.array([np.eye(2), np.eye(2)])

        diagnostics = build_correlation_diagnostics(ClosedShellMeanField(), FakeMol())

        frontier = diagnostics['frontier_orbital_degeneracy']
        self.assertFalse(frontier['alpha_beta']['evaluated'])
        self.assertFalse(frontier['alpha_beta']['degenerate_pairs'])
        self.assertNotIn('alpha_beta_frontier_degeneracy', diagnostics['warnings'])
        self.assertNotIn('fractional_orbital_occupations', diagnostics['warnings'])
        self.assertFalse(diagnostics['fractional_occupations'])

    def test_broken_symmetry_singlet_uhf_keeps_dissociation_warning_and_uno_occupations(self):
        class SingletMol(FakeMol):
            spin = 0
            nelectron = 2

        class BrokenSymmetryMeanField(FakeMeanField):
            def __init__(self):
                super().__init__()
                self.mol = SingletMol()
                self.mo_energy = np.array([
                    [-0.4, 0.2],
                    [-0.4, 0.2],
                ])
                self.mo_occ = np.array([
                    [1.0, 0.0],
                    [1.0, 0.0],
                ])
                self.mo_coeff = np.array([np.eye(2), np.eye(2)])

            def get_ovlp(self):
                return np.eye(2)

            def make_rdm1(self):
                return np.array([
                    [[1.0, 0.0], [0.0, 0.0]],
                    [[0.0, 0.0], [0.0, 1.0]],
                ])

            def spin_square(self):
                return 1.0

        diagnostics = build_correlation_diagnostics(BrokenSymmetryMeanField(), SingletMol())

        frontier = diagnostics['frontier_orbital_degeneracy']
        self.assertTrue(frontier['alpha_beta']['evaluated'])
        self.assertTrue(frontier['alpha_beta']['degenerate_pairs'])
        self.assertEqual(
            diagnostics['mean_field_occupation_summary']['source'],
            'uhf_spin_summed_ao_density_uno',
        )
        self.assertEqual([item['occupation'] for item in diagnostics['fractional_occupations']], [1.0, 1.0])
        self.assertIn('unrestricted_spin_symmetry_breaking', diagnostics['warnings'])
        self.assertIn('alpha_beta_frontier_degeneracy', diagnostics['warnings'])
        self.assertEqual(diagnostics['molecular_correlation_risk']['level'], 'strong')

    def test_scf_stability_summary_parses_return_status(self):
        class StableMeanField(FakeMeanField):
            def stability(self, **kwargs):
                self.stability_kwargs = kwargs
                return 'internal_orbitals', 'external_orbitals', True, False

        mean_field = StableMeanField()
        summary = build_scf_stability_summary(mean_field)

        self.assertEqual(summary['status'], 'completed')
        self.assertFalse(summary['stable'])
        self.assertTrue(summary['internal_stable'])
        self.assertFalse(summary['external_stable'])
        self.assertTrue(summary['orbital_update_available'])
        self.assertEqual(mean_field.stability_kwargs['nroots'], 1)

    def test_avas_projection_produces_a_reusable_cas_initial_guess(self):
        from pyscf import gto, scf  # pylint: disable=import-outside-toplevel

        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        proposal = propose_active_space(
            mf,
            ActiveSpaceSpec(
                enabled=True,
                selection_method='avas',
                avas_targets=['atom:0,1'],
            ),
            mol=mol,
        )

        self.assertEqual(proposal['selection_method'], 'avas')
        self.assertEqual(proposal['ncas'], 2)
        self.assertEqual(proposal['nelecas'], 2)
        self.assertTrue(proposal['initial_mo_coeff'])
        avas_candidate = {
            item['method']: item
            for item in proposal['audit']['candidate_active_spaces']
        }['avas']
        self.assertEqual(avas_candidate['status'], 'available')
        self.assertEqual(avas_candidate['resolved_targets'], ['0 H 1s', '1 H 1s'])

        result = run_cas_method(
            mf,
            mol,
            'casscf',
            ActiveSpaceSpec(
                enabled=True,
                selection_method='avas',
                ncas=proposal['ncas'],
                nelecas=proposal['nelecas'],
                orbital_indices=proposal['orbital_indices'],
                initial_mo_coeff=proposal['initial_mo_coeff'],
                avas_targets=['atom:0,1'],
                approved=True,
            ),
            max_cycle=40,
        )
        self.assertTrue(result['converged'])
        self.assertEqual(result['initial_orbital_guess'], 'active_space.initial_mo_coeff')

    def test_molecular_low_energy_roots_flag_numerical_degeneracy(self):
        diagnostics = build_correlation_diagnostics(
            FakeMeanField(),
            FakeMeanField().mol,
            state_energies=[-1.0, -1.0 + 1.0e-10, -0.8],
            state_sector={
                'electron_count': 4,
                'spin_2s': 0,
                'scope': 'fixed_particle_spin_projection_sector',
            },
        )

        manifold = diagnostics['low_energy_manifold']
        self.assertEqual(manifold['classification'], 'numerically_degenerate')
        self.assertEqual(manifold['ground_state_count_within_numerical_tolerance'], 2)
        self.assertIn('many_body_ground_state_degeneracy', diagnostics['warnings'])
        self.assertEqual(diagnostics['molecular_correlation_risk']['level'], 'strong')

    def test_pyscf_cas_methods_support_multiple_targeted_roots(self):
        from pyscf import gto, scf  # pylint: disable=import-outside-toplevel

        mol = gto.M(atom='H 0 0 0; H 0 0 0.74', basis='sto-3g', verbose=0)
        mf = scf.RHF(mol).run()
        active_space = ActiveSpaceSpec(
            enabled=True,
            selection_method='manual',
            ncas=2,
            nelecas=2,
            orbital_indices=[0, 1],
            approved=True,
        )

        casci = run_cas_method(
            mf,
            mol,
            'casci',
            active_space,
            max_cycle=30,
            solver_options={'nroots': 2},
        )
        casscf = run_cas_method(
            mf,
            mol,
            'casscf',
            active_space,
            max_cycle=30,
            solver_options={'nroots': 2, 'state_average_weights': [0.5, 0.5]},
        )

        self.assertEqual(len(casci['state_energies']), 2)
        self.assertEqual(casci['orbital_optimization_mode'], 'state_specific')
        self.assertEqual(len(casscf['state_energies']), 2)
        self.assertEqual(casscf['orbital_optimization_mode'], 'state_averaged')
        self.assertEqual(casscf['state_average_weights'], [0.5, 0.5])
        self.assertEqual(len(casscf['natural_occupations']), 2)

        single_root_casscf = run_cas_method(
            mf,
            mol,
            'casscf',
            active_space,
            max_cycle=30,
            solver_options={'nroots': 1, 'state_average_weights': [1.0]},
        )
        self.assertEqual(single_root_casscf['orbital_optimization_mode'], 'state_specific')
        self.assertNotIn('state_average_energy', single_root_casscf)
        self.assertNotIn('state_average_weights', single_root_casscf)


if __name__ == '__main__':
    unittest.main()
