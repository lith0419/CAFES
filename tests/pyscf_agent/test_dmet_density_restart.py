import copy
import hashlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pyscf_agent.providers.libdmet import (
    libdmet_availability,
    validate_dmet_model_request,
)
from pyscf_agent.providers.libdmet.density_restart import (
    load_density_seed,
    normalize_density_source,
    serialize_density_state,
    state_metadata,
)
from pyscf_agent.executors.ssh_slurm import _request_attachments
from pyscf_agent.remote.rpc_cli import _materialize_attachments
from tests.pyscf_agent.test_libdmet_dmet import _ring_spec


def configuration(spec, **updates):
    errors, config = validate_dmet_model_request(
        spec,
        {
            'execution_mode': 'finite_graph',
            'reference': 'unrestricted',
            'impurity_size': 2,
            'interacting_bath': False,
            'bath_spin_dimension_policy': 'max',
            **updates,
        },
    )
    if errors:
        raise ValueError(errors)
    return config


class DensityRestartTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.path = Path(temporary.name) / 'state.npz'
        self.spec = _ring_spec(4, onsite_u=2)
        self.spec['globals'] = {'U': 2, 'V': 0, 't': -1}
        self.config = configuration(self.spec)
        self.metadata = state_metadata(self.spec, self.config, converged=True)
        self.metadata['quality_passed'] = True
        self.rho = np.array([np.eye(4) * 0.5] * 2)
        self.rho[:, 0, 1] = self.rho[:, 1, 0] = 0.1

    def save(self, *, raw=False):
        if raw:
            metadata = dict(self.metadata, source_run_id='donor-run')
            stream = io.BytesIO()
            np.savez_compressed(
                stream,
                metadata_json=json.dumps(metadata),
                mean_field_density_matrix=self.rho,
            )
            content = stream.getvalue()
        else:
            content = serialize_density_state(
                self.metadata, self.rho, run_id='donor-run'
            )
        self.path.write_bytes(content)
        self.config['reference_density_source'] = {
            'path': str(self.path),
            'sha256': hashlib.sha256(content).hexdigest(),
            'source_case_id': 'donor',
            'source_run_id': 'donor-run',
        }

    def test_full_density_and_target_parameter_changes_are_preserved(self):
        self.save()
        target = copy.deepcopy(self.spec)
        target['sites'][0]['U'] += 0.2
        target['globals'].update(U=2.2, V=0.1)
        target['bonds'][0]['V'] = target['bonds'][0]['effective_V'] = 0.1
        rho, info = load_density_seed(target, self.config, np=np)
        np.testing.assert_array_equal(rho, self.rho)
        self.assertAlmostEqual(info['max_abs_parameter_change']['onsite_u'], 0.2)
        self.assertAlmostEqual(info['max_abs_parameter_change']['intersite_v'], 0.1)
        self.assertEqual(info['source_run_id'], 'donor-run')
        self.assertEqual(info['transfer'], 'mean_field_density_only')

    def test_checksum_and_run_identity_are_enforced(self):
        self.save()
        self.path.write_bytes(self.path.read_bytes() + b'changed')
        with self.assertRaisesRegex(ValueError, 'checksum'):
            load_density_seed(self.spec, self.config, np=np)
        self.save()
        self.config['reference_density_source']['source_run_id'] = 'other-run'
        with self.assertRaisesRegex(ValueError, 'Run ID'):
            load_density_seed(self.spec, self.config, np=np)

    def test_source_must_have_converged_and_passed_required_quality_checks(self):
        for field in ('converged', 'quality_passed'):
            with self.subTest(field=field):
                self.metadata[field] = False
                self.save()
                with self.assertRaisesRegex(ValueError, 'converged source'):
                    load_density_seed(self.spec, self.config, np=np)
                self.metadata[field] = True

    def test_incompatible_graph_fragment_sector_and_solver_are_rejected(self):
        self.save()
        for field, value in (
            ('reference', 'restricted'),
            ('nalpha', 1),
            ('interacting_bath', True),
            ('impurity_solver', 'ccsd'),
            ('impurity_solver_options', {'beta': 10}),
            ('fragments', [{'site_ids': [0, 2]}, {'site_ids': [1, 3]}]),
        ):
            target_config = dict(self.config, **{field: value})
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, 'Incompatible'),
            ):
                load_density_seed(self.spec, target_config, np=np)
        for field in ('id', 'x', 'epsilon'):
            target = copy.deepcopy(self.spec)
            target['sites'][0][field] += 1
            with (
                self.subTest(field=field),
                self.assertRaisesRegex(ValueError, 'Incompatible'),
            ):
                load_density_seed(target, self.config, np=np)

    def test_invalid_density_and_legacy_archive_cannot_silently_fall_back(self):
        good = self.rho.copy()
        for failure in ('shape', 'nan', 'hermitian', 'trace', 'occupation', 'complex'):
            self.rho = good.astype(complex)
            if failure == 'shape':
                self.rho = self.rho[:, :2, :2]
            elif failure == 'nan':
                self.rho[0, 0, 0] = np.nan
            elif failure == 'hermitian':
                self.rho[0, 0, 1] = 0.3
            elif failure == 'trace':
                self.rho[0, 0, 0] = 0.6
            elif failure == 'occupation':
                self.rho[0, 0, 0], self.rho[0, 1, 1] = 1.5, -0.5
            elif failure == 'complex':
                self.rho[0, 0, 1] += 0.1j
            self.save(raw=True)
            with self.subTest(failure=failure), self.assertRaises(ValueError):
                load_density_seed(self.spec, self.config, np=np)
        self.rho = good
        self.save()
        stream = io.BytesIO()
        np.savez(stream, reference_density_seed=good)
        self.path.write_bytes(stream.getvalue())
        self.config['reference_density_source']['sha256'] = hashlib.sha256(
            stream.getvalue()
        ).hexdigest()
        with self.assertRaisesRegex(ValueError, 'old density seeds'):
            load_density_seed(self.spec, self.config, np=np)

    def test_normalization_and_execution_scope(self):
        self.assertEqual(
            normalize_density_source({'source_case_id': 'neighbor'}),
            {'source_case_id': 'neighbor'},
        )
        for bad in (
            {},
            [],
            'path',
            {'path': '/tmp/x'},
            {'source_case_id': 'x', 'unknown': 1},
            {'path': 'x', 'sha256': '0' * 64},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                normalize_density_source(bad)
        errors, _ = validate_dmet_model_request(
            self.spec,
            {
                'execution_mode': 'translational',
                'impurity_size': 2,
                'reference_density_source': {'source_case_id': 'neighbor'},
            },
        )
        self.assertTrue(any('finite_graph' in error for error in errors))

    def test_remote_attachment_round_trip_keeps_content_hash(self):
        self.save()
        request = {'solver': {'name': 'dmet', 'options': self.config}}
        attachments = _request_attachments(request)
        self.assertEqual(len(attachments), 1)
        translated = _materialize_attachments(
            request, attachments, self.path.parent / 'remote'
        )
        source = translated['solver']['options']['reference_density_source']
        self.assertNotEqual(source['path'], str(self.path))
        rho, _ = load_density_seed(self.spec, translated['solver']['options'], np=np)
        np.testing.assert_array_equal(rho, self.rho)


@unittest.skipUnless(libdmet_availability()['available'], 'libDMET unavailable')
class DensityRestartNativeTests(unittest.TestCase):
    def test_neighbor_density_reaches_target_baseline_and_saved_run(self):
        from pyscf_agent.backend.model_hamiltonian.solver import (
            run_model_hamiltonian_solver,
        )
        from pyscf_agent.backend.result_artifacts import write_result_artifacts
        from pyscf_agent.contracts import task_spec_from_dict
        from libdmet.routine import slater
        from pyscf_agent.providers.libdmet.dmet import _finite_graph_lattice

        for reference in ('restricted', 'unrestricted'):
            with (
                self.subTest(reference=reference),
                tempfile.TemporaryDirectory() as temp,
            ):
                spec = _ring_spec(6, onsite_u=0)
                options = {
                    'execution_mode': 'finite_graph',
                    'impurity_size': 2,
                    'reference': reference,
                    'interacting_bath': False,
                    'bath_spin_dimension_policy': 'max',
                    'max_iterations': 4,
                    'correlation_potential_mixing': 0.2,
                    'diis_enabled': False,
                    'solver_max_memory_mb': 1000,
                }
                source = run_model_hamiltonian_solver(
                    spec,
                    solver_name='dmet',
                    outputs=['energy'],
                    solver_options=options,
                    scratch_directory=temp + '/donor',
                )
                self.assertTrue(source['converged'])
                task = task_spec_from_dict(
                    {
                        'task_type': 'model_hamiltonian',
                        'model_hamiltonian': {'spec': spec},
                        'solver': {'name': 'dmet', 'options': options},
                    }
                )
                state = {'run_id': 'donor-run', 'work_dir': temp, 'artifacts': []}
                write_result_artifacts(state, task, source)
                artifact = next(
                    a
                    for a in state['artifacts']
                    if a['kind'] == 'dmet_mean_field_state'
                )
                with np.load(artifact['path'], allow_pickle=False) as saved:
                    expected = saved['mean_field_density_matrix']
                # The final full density must not be replaced by the diagonal analytic seed.
                self.assertGreater(
                    np.max(
                        np.abs(
                            expected
                            - source['_transient_dmet_arrays']['reference_density_seed']
                        )
                    ),
                    0.1,
                )
                target_spec = _ring_spec(6, onsite_u=0.1)
                target_options = dict(
                    options,
                    max_iterations=2,
                    correlation_potential_mixing=0.1,
                    reference_density_source={
                        k: artifact[k] for k in ('path', 'sha256')
                    },
                )
                target = run_model_hamiltonian_solver(
                    target_spec,
                    solver_name='dmet',
                    outputs=['energy'],
                    solver_options=target_options,
                    scratch_directory=temp + '/target',
                )
                arrays = target['_transient_dmet_arrays']
                np.testing.assert_array_equal(
                    arrays['reference_density_seed'], expected
                )
                _, ham = _finite_graph_lattice(target_spec, np)
                baseline = slater.get_veff(
                    expected.sum(axis=0) if reference == 'restricted' else expected,
                    ham.H2,
                )
                if reference == 'restricted':
                    baseline = np.repeat(
                        np.asarray(baseline).reshape(1, 6, 6), 2, axis=0
                    )
                np.testing.assert_allclose(
                    arrays['mean_field_baseline_potential'], baseline
                )
                np.testing.assert_array_equal(
                    arrays['initial_correlation_potential'], 0
                )
                self.assertEqual(
                    target['dmet_result']['reference_density_initialization'][
                        'source_run_id'
                    ],
                    'donor-run',
                )
                self.assertTrue(np.isfinite(target['energy']))
