from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from pyscf_agent.contracts import task_spec_from_dict, task_spec_to_dict
from pyscf_agent.backend.validation import _validate_embedding_spec
from pyscf_agent.registry import default_registry
from pyscf_agent.registry.platform import (
    artifact_kind_is_registered,
    artifact_payload_schema,
)
from pyscf_agent.embedding import (
    CORRELATED_SUBSPACE_AUDIT_SCHEMA,
    EMBEDDING_REFERENCE_SCHEMA,
    approve_correlated_subspace_audit,
    build_correlated_subspace_audit,
    prepare_embedding_artifacts,
    read_embedding_hdf5,
    serialize_embedding_hdf5,
)
from pyscf_agent.providers.libdmet import (
    build_molecular_iao_coefficients,
    libdmet_availability,
    transform_density_to_local,
    transform_one_body_to_local,
)


class EmbeddingFoundationTests(unittest.TestCase):
    def test_task_spec_round_trips_embedding_preparation(self):
        payload = task_spec_to_dict(task_spec_from_dict({
            'embedding': {
                'enabled': 'true',
                'provider': 'libDMET',
                'localization_method': 'IAO_PAO',
                'correlated_orbital_indices': '1, 3, 3',
                'correlated_atom_indices': [0, 2],
                'fragments': [{'fragment_id': 'frag-a', 'orbital_indices': [1, 3]}],
                'interaction': {'kind': 'hubbard_u', 'U': 4.0},
                'approved': False,
                'audit_artifact': {
                    'kind': 'correlated_subspace_audit',
                    'path': '/tmp/correlated-subspace-audit.json',
                },
            },
        }))

        self.assertTrue(payload['embedding']['enabled'])
        self.assertEqual(payload['embedding']['provider'], 'libdmet')
        self.assertEqual(payload['embedding']['localization_method'], 'iao_pao')
        self.assertEqual(payload['embedding']['correlated_orbital_indices'], [1, 3])
        self.assertEqual(
            payload['embedding']['audit_artifact']['path'],
            '/tmp/correlated-subspace-audit.json',
        )
        self.assertEqual(task_spec_to_dict(task_spec_from_dict(payload)), payload)

    def test_correlated_subspace_audit_is_reviewable_and_approvable(self):
        audit = build_correlated_subspace_audit(
            system_type='periodic',
            provider='libdmet',
            localization_method='iao_pao',
            orbital_indices=[1, 2, 2],
            total_orbitals=4,
            atom_indices=[0],
            fragments=[{
                'fragment_id': 'metal-d',
                'site_ids': [3],
                'orbital_indices': [1, 2],
                'atom_indices': [0],
            }],
            orbital_labels=['s', 'dxy', 'dxz', 'p'],
            occupations=[2.0, 1.2, 0.8, 0.0],
            orbital_contributions={1: {'atom:0': 0.92}},
            selection_reasons=['partially occupied d shell'],
            interaction={'kind': 'kanamori', 'U': 4.0, 'J': 0.8},
            electron_count=2,
        )

        self.assertEqual(audit['schema'], CORRELATED_SUBSPACE_AUDIT_SCHEMA)
        self.assertEqual(audit['orbital_indices'], [1, 2])
        self.assertEqual(audit['orbital_records'][0]['label'], 'dxy')
        self.assertEqual(audit['orbital_records'][0]['contributions']['atom:0'], 0.92)
        self.assertEqual(audit['fragments'][0]['site_ids'], [3])
        self.assertEqual(audit['approval']['status'], 'pending')
        approved = approve_correlated_subspace_audit(
            audit,
            approved_by='test-user',
            approved_at='2026-08-17T00:00:00+00:00',
        )
        self.assertTrue(approved['approval']['approved'])
        self.assertEqual(approved['approval']['approved_by'], 'test-user')
        self.assertEqual(audit['approval']['status'], 'pending')
        with self.assertRaisesRegex(ValueError, 'approve_correlated_subspace_audit'):
            build_correlated_subspace_audit(
                system_type='molecular',
                provider='libdmet',
                localization_method='manual',
                orbital_indices=[0],
                approved=True,
            )

    def test_hdf5_numeric_artifact_round_trip_preserves_complex_arrays(self):
        import numpy as np

        matrix = np.asarray([[1.0, 0.25j], [-0.25j, 2.0]], dtype=complex)
        content = serialize_embedding_hdf5(
            schema=EMBEDDING_REFERENCE_SCHEMA,
            datasets={'operators/fock': matrix, 'kpoints': np.zeros((1, 3))},
            metadata={'system_type': 'periodic', 'kmesh': [1, 1, 1]},
        )
        restored = read_embedding_hdf5(content)

        self.assertEqual(restored['schema'], EMBEDDING_REFERENCE_SCHEMA)
        self.assertEqual(restored['metadata']['kmesh'], [1, 1, 1])
        np.testing.assert_allclose(restored['datasets']['operators/fock'], matrix)
        self.assertEqual(
            {item['name'] for item in restored['dataset_manifest']},
            {'operators/fock', 'kpoints'},
        )

    def test_libdmet_transform_wrappers_match_matrix_definition(self):
        availability = libdmet_availability()
        if not availability['available']:
            self.skipTest(availability['reason'])
        import numpy as np

        theta = 0.3
        coefficients = np.asarray([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        one_body = np.diag([1.0, 3.0])
        density = np.diag([2.0, 0.0])
        overlap = np.eye(2)

        expected_h1 = coefficients.T @ one_body @ coefficients
        expected_density = coefficients.T @ density @ coefficients
        np.testing.assert_allclose(
            transform_one_body_to_local(one_body, coefficients),
            expected_h1,
        )
        np.testing.assert_allclose(
            transform_density_to_local(density, coefficients, overlap),
            expected_density,
        )

    def test_libdmet_iao_wrapper_distinguishes_valence_and_pao_space(self):
        availability = libdmet_availability()
        if not availability['available']:
            self.skipTest(availability['reason'])
        from pyscf import gto, scf

        molecule = gto.M(
            atom='H 0 0 0; H 0 0 0.74',
            basis='6-31g',
            verbose=0,
        )
        mean_field = scf.RHF(molecule).run()
        valence = build_molecular_iao_coefficients(mean_field, include_pao=False)
        complete = build_molecular_iao_coefficients(mean_field, include_pao=True)

        self.assertLess(valence.shape[-1], complete.shape[-1])
        self.assertEqual(complete.shape[-1], molecule.nao_nr())

    def test_preparation_writes_registered_interchange_artifacts(self):
        availability = libdmet_availability()
        if not availability['available']:
            self.skipTest(availability['reason'])
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            state = {'run_id': 'embedding-test', 'work_dir': directory, 'artifacts': []}
            prepared = prepare_embedding_artifacts(
                state,
                system_type='model_hamiltonian',
                one_body=np.diag([-1.0, 1.0]),
                density=np.diag([1.0, 1.0]),
                overlap=np.eye(2),
                fock=np.diag([-0.5, 1.5]),
                correlated_orbital_indices=[0, 1],
                localization_method='identity_sites',
                selection_reasons=['all model sites are correlated'],
                electron_count=2,
            )

            self.assertEqual(len(state['artifacts']), 4)
            for key in ('embedding_reference', 'localized_subspace', 'localized_hamiltonian'):
                ref = prepared[key]
                self.assertTrue(Path(ref['path']).is_file())
                self.assertGreater(ref['size_bytes'], 0)
                self.assertNotIn('sha256', ref)
                self.assertTrue(ref['datasets'])
            self.assertEqual(prepared['audit']['approval']['status'], 'pending')
            localized = read_embedding_hdf5(prepared['localized_hamiltonian']['path'])
            self.assertIn('fock_local', localized['datasets'])
            self.assertEqual(
                prepared['task_spec_patch']['embedding']['audit_artifact']['kind'],
                'correlated_subspace_audit',
            )

    def test_registry_separates_preparation_from_solver_loops(self):
        registry = default_registry()

        self.assertEqual(
            registry.capability('libdmet', namespace='embedding.backend').status,
            'executable',
        )
        self.assertTrue(registry.capability_is_allowed(
            'transform_one_particle_operators', namespace='embedding.operation'
        ))
        self.assertEqual(
            registry.capability('dmet', namespace='embedding.method').status,
            'executable',
        )
        self.assertEqual(
            registry.capability('gw_dmft', namespace='embedding.method').status,
            'executable',
        )
        for kind in (
            'embedding_reference',
            'localized_subspace',
            'localized_hamiltonian',
            'correlated_subspace_audit',
        ):
            self.assertTrue(artifact_kind_is_registered(kind, registry))
        self.assertEqual(
            artifact_payload_schema('correlated_subspace_audit', registry),
            CORRELATED_SUBSPACE_AUDIT_SCHEMA,
        )

    def test_approved_embedding_requires_registered_artifact_evidence(self):
        task_spec = task_spec_from_dict({
            'embedding': {
                'enabled': True,
                'provider': 'libdmet',
                'localization_method': 'manual',
                'approved': True,
                'audit_artifact': {'kind': 'correlated_subspace_audit'},
            },
        })
        errors = _validate_embedding_spec(task_spec)

        self.assertTrue(errors)
        self.assertTrue(all(error['code'] == 'invalid_embedding_approval_evidence' for error in errors))


if __name__ == '__main__':
    unittest.main()
