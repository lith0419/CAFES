from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from pyscf_agent.backend.artifacts import compact_structured_results
from pyscf_agent.backend.execution import generate_input_script
from pyscf_agent.backend.execution import (
    repair_or_retry,
    result_analyst,
    result_extractor,
    task_reporter,
)
from pyscf_agent.backend.model_hamiltonian.spec import normalize_solver_name
from pyscf_agent.backend.runtime_context import clear_runtime_contexts, store_runtime_context
from pyscf_agent.contracts import task_spec_from_dict, task_spec_to_dict
from pyscf_agent.backend.parsing import task_spec_from_partial
from pyscf_agent.embedding import prepare_embedding_artifacts, read_embedding_hdf5
from pyscf_agent.embedding.contracts import approve_correlated_subspace_audit
from pyscf_agent.lifecycle import new_lifecycle, transition_lifecycle
from pyscf_agent.providers.fcdmft import (
    GW_DMFT_RESULT_SCHEMA,
    HF_DMFT_RESULT_SCHEMA,
    PERIODIC_GW_RESULT_SCHEMA,
    fcdmft_availability,
    normalize_gw_dmft_options,
    normalize_hf_dmft_options,
    run_gw_dmft,
    run_hf_dmft,
    run_periodic_gw,
    validate_periodic_gw_reference,
    validate_gw_dmft_request,
    validate_hf_dmft_request,
    validate_hf_dmft_preparation_request,
)
from pyscf_agent.providers.fcdmft.adapter import (
    FCDMFTExecutionError,
    _legacy_fcdmft_numpy_aliases,
)
from pyscf_agent.providers.fcdmft.module import execute_hf_dmft, prepare_hf_dmft_subspace
from pyscf_agent.providers.fcdmft.preparation import (
    _libdmet_kpoint_mean_field,
    _slice_coefficients,
)
from pyscf_agent.providers.libdmet import libdmet_availability
from pyscf_agent.registry import default_registry
from pyscf_agent.workflow_modules.compiler import compile_task_workflow
from pyscf_agent.workflow_modules.runtime import ModuleInvocation, WorkflowRuntimeError


class _FakeDMFT:
    def __init__(
        self,
        hcore_k,
        jk_k,
        density_k,
        eris,
        nval,
        ncore,
        nbath,
        nb_per_e,
        *,
        disc_type,
        solver_type,
    ):
        self.inputs = (hcore_k, jk_k, density_k, eris)
        self.window = (ncore, nval)
        self.bath = (nbath, nb_per_e, disc_type)
        self.solver_type = solver_type
        self.converged = False
        self.mu = None
        self.hyb = None
        self.sigma = None
        self.freqs = None
        self.wts = None
        self.chkfile = None

    def kernel(self, *, mu0, wl, wh, occupancy, delta, conv_tol, opt_mu, dump_chk):
        import h5py
        import numpy as np

        self.kernel_options = {
            'mu0': mu0,
            'window': (wl, wh),
            'occupancy': occupancy,
            'delta': delta,
            'conv_tol': conv_tol,
            'opt_mu': opt_mu,
            'dump_chk': dump_chk,
        }
        self.converged = True
        self.mu = float(mu0) + 0.01
        self.freqs = np.asarray([-0.2, 0.0, 0.2])
        self.wts = np.ones(3)
        self.hyb = np.zeros((1, 2, 2, 3), dtype=complex)
        self.sigma = np.ones((1, 2, 2, 3), dtype=complex) * 0.05j
        with h5py.File(self.chkfile, 'w') as handle:
            handle['dmft/mu'] = self.mu


class _FakePeriodicGW:
    def __init__(self, mean_field):
        self._scf = mean_field
        self.mo_energy = None

    def kernel(self, *, omega, orbs, kptlist, writefile, nw):
        import h5py
        import numpy as np

        self.kernel_options = {
            'omega': np.asarray(omega),
            'orbs': tuple(orbs),
            'kptlist': tuple(kptlist),
            'writefile': writefile,
            'nw': nw,
        }
        self.mo_energy = np.asarray(self._scf.mo_energy) + 0.05
        self.ef = 0.1
        for filename in ('ac_coeff.h5', 'vxc.h5', 'sigma_imag.h5'):
            with h5py.File(filename, 'w') as handle:
                handle['test'] = np.asarray([1.0])
                if filename == 'ac_coeff.h5':
                    handle['fermi'] = np.asarray([0.1])
        return None


def _approved_periodic_task(directory: str):
    if not libdmet_availability()['available']:
        raise unittest.SkipTest('embedding fixture requires the optional libDMET provider')
    import numpy as np

    state = {'run_id': 'fcdmft-input', 'work_dir': directory, 'artifacts': []}
    hcore = np.asarray([
        [[-1.0, 0.0], [0.0, 1.0]],
        [[-0.8, 0.0], [0.0, 1.2]],
    ])
    fock = np.asarray([
        [[-0.8, 0.0], [0.0, 1.2]],
        [[-0.5, 0.0], [0.0, 1.5]],
    ])
    density = np.asarray([
        [[2.0, 0.0], [0.0, 0.0]],
        [[2.0, 0.0], [0.0, 0.0]],
    ])
    prepared = prepare_embedding_artifacts(
        state,
        system_type='periodic',
        one_body=hcore,
        fock=fock,
        density=density,
        overlap=np.stack((np.eye(2), np.eye(2))),
        two_body=np.zeros((1, 2, 2, 2, 2)),
        correlated_orbital_indices=[0, 1],
        provider='fcdmft',
        localization_method='manual',
        matrix_layout='kpoint',
        electron_count=2,
    )
    embedding = prepared['task_spec_patch']['embedding']
    embedding['approved'] = True
    audit_path = Path(embedding['audit_artifact']['path'])
    audit = json.loads(audit_path.read_text(encoding='utf-8'))
    audit = approve_correlated_subspace_audit(audit, approved_by='test')
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    return task_spec_from_dict({
        'task_type': 'periodic',
        'method': {'name': 'hf', 'restricted': True},
        'solver': {
            'name': 'hf_dmft',
            'options': {
                'impurity_solver': 'cc',
                'nval': 2,
                'nbath': 3,
                'chemical_potential': 0.0,
            },
        },
        'embedding': embedding,
        'periodic': {
            'kmesh': [2, 1, 1],
            'kpoint_scheme': 'gamma_centered',
            'kpoint_shift': [0, 0, 0],
            'density_fitting_method': 'gdf',
            'smearing_method': 'none',
        },
    })


def _approved_gw_dmft_task(directory: str):
    if not libdmet_availability()['available']:
        raise unittest.SkipTest('embedding fixture requires the optional libDMET provider')
    import h5py
    import numpy as np

    state = {'run_id': 'gw-dmft-input', 'work_dir': directory, 'artifacts': []}
    hcore = np.asarray([
        [[-1.0, 0.0], [0.0, 1.0]],
        [[-0.8, 0.0], [0.0, 1.2]],
    ])
    fock = hcore + 0.2 * np.eye(2)[None, :, :]
    density = np.asarray([
        [[2.0, 0.0], [0.0, 0.0]],
        [[2.0, 0.0], [0.0, 0.0]],
    ])
    prepared = prepare_embedding_artifacts(
        state,
        system_type='periodic',
        one_body=hcore,
        fock=fock,
        density=density,
        overlap=np.stack((np.eye(2), np.eye(2))),
        coefficients=np.stack((np.eye(2), np.eye(2))),
        two_body=np.zeros((1, 2, 2, 2, 2)),
        correlated_orbital_indices=[0, 1],
        provider='fcdmft',
        localization_method='manual',
        matrix_layout='kpoint',
        electron_count=2,
        additional_one_body_operators={
            'hf_effective_potential_local': np.stack((0.3 * np.eye(2), 0.3 * np.eye(2))),
        },
    )
    embedding = prepared['task_spec_patch']['embedding']
    embedding['approved'] = True
    audit_path = Path(embedding['audit_artifact']['path'])
    audit = approve_correlated_subspace_audit(
        json.loads(audit_path.read_text(encoding='utf-8')),
        approved_by='test',
    )
    audit_path.write_text(json.dumps(audit, indent=2), encoding='utf-8')
    gw_directory = Path(directory) / 'gw-artifacts'
    gw_directory.mkdir()
    kinds = {
        'lattice_result': ('periodic_gw_result', 'periodic-gw-result.json'),
        'lattice_ac': ('periodic_gw_analytic_continuation', 'ac_coeff.h5'),
        'lattice_vxc': ('periodic_gw_mean_field_potential', 'vxc.h5'),
        'lattice_sigma_imag': ('periodic_gw_imaginary_self_energy', 'sigma_imag.h5'),
        'local_ac': ('gw_dmft_local_analytic_continuation', 'imp_ac_coeff.h5'),
        'local_sigma_imag': ('gw_dmft_local_imaginary_self_energy', 'gw_dc_sigmaI.h5'),
        'local_transform': ('gw_dmft_local_orbital_transform', 'C_mo_lo.h5'),
    }
    references = {}
    for key, (kind, filename) in kinds.items():
        path = gw_directory / filename
        if filename.endswith('.json'):
            path.write_text(json.dumps({'schema': PERIODIC_GW_RESULT_SCHEMA}), encoding='utf-8')
        else:
            with h5py.File(path, 'w') as handle:
                handle['test'] = np.asarray([1.0])
        references[key] = {'kind': kind, 'path': str(path)}
    return task_spec_from_dict({
        'task_type': 'periodic',
        'method': {'name': 'dft', 'restricted': True, 'xc': 'pbe'},
        'solver': {
            'name': 'gw_dmft',
            'options': {
                'impurity_solver': 'cc',
                'ncore': 0,
                'nval': 2,
                'nbath': 3,
                'chemical_potential': 0.0,
                'gw_artifacts': references,
            },
        },
        'embedding': embedding,
        'periodic': {
            'kmesh': [2, 1, 1],
            'kpoint_scheme': 'gamma_centered',
            'kpoint_shift': [0, 0, 0],
            'density_fitting_method': 'gdf',
            'smearing_method': 'none',
        },
    })


class FCDMFTProviderTests(unittest.TestCase):
    def test_availability_advertises_the_registered_gw_dmft_stages(self):
        stages = fcdmft_availability()['supported_stages']

        self.assertIn('periodic_gw', stages)
        self.assertIn('gw_double_counting', stages)
        self.assertIn('dmft_bath', stages)

    def test_unconverged_analysis_separates_hf_reference_from_dmft_result(self):
        state = result_analyst({
            'execution_status': 'unconverged',
            'locale': 'en',
            'structured_results': {
                'task_type': 'periodic',
                'method': 'hf_dmft',
                'solver': 'hf_dmft',
                'reference_converged': True,
                'reference_energy': -7.25,
                'energy_unit': 'Ha/cell',
                'dmft_result': {
                    'method': 'hf_dmft',
                    'converged': False,
                    'impurity_solver': 'cc',
                    'ncore': 1,
                    'nval': 3,
                    'nbath': 4,
                    'bath_discretization': 'opt',
                    'max_iterations': 6,
                    'convergence_tolerance': 1e-5,
                },
            },
            'messages': [],
            'logs': [],
            'errors': [],
        })

        summary = state['analysis_summary']
        self.assertIn('did not converge within max_iterations=6', summary)
        self.assertIn('HF mean-field reference energy=-7.25 Ha/cell', summary)
        self.assertIn('Correlated window=[1, 3)', summary)
        self.assertIn('No HF+DMFT total energy is available', summary)
        self.assertNotIn('Current energy', summary)

    def test_success_analysis_does_not_report_reference_as_dmft_energy(self):
        state = result_analyst({
            'execution_status': 'succeeded',
            'locale': 'en',
            'structured_results': {
                'task_type': 'periodic',
                'method': 'hf_dmft',
                'solver': 'hf_dmft',
                'reference_converged': True,
                'reference_energy': -7.25,
                'energy_unit': 'Ha/cell',
                'dmft_result': {
                    'method': 'hf_dmft',
                    'converged': True,
                    'impurity_solver': 'cc',
                    'ncore': 1,
                    'nval': 3,
                    'nbath': 4,
                    'bath_discretization': 'opt',
                },
            },
            'messages': [],
            'logs': [],
        })

        summary = state['analysis_summary']
        self.assertIn('Periodic HF+DMFT completed', summary)
        self.assertIn('HF_mean_field_energy=-7.25 Ha/cell', summary)
        self.assertIn('DMFT_converged=yes', summary)
        self.assertIn('DMFT_total_energy=unavailable', summary)
        self.assertNotIn('; energy=-7.25', summary)

    def test_hf_dmft_recovery_increases_dmft_iterations_and_clears_current_error(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            task_spec = _approved_periodic_task(tmpdir)
            task_spec.solver.options['max_iterations'] = 3
            state = {
                'task_spec': task_spec_to_dict(task_spec),
                'execution_status': 'unconverged',
                'retry_count': 0,
                'max_retries': 1,
                'raw_stderr': 'HF+DMFT did not converge.',
                'structured_results': {'converged': False},
                'compact_results': {'converged': False},
                'raw_stdout': 'attempt one',
                'raw_scf_output': '',
                'analysis_text': '',
                'errors': [
                    {
                        'stage': 'execution',
                        'code': 'solver_unconverged',
                        'message': 'HF+DMFT did not converge.',
                    },
                    {
                        'stage': 'planning',
                        'code': 'retained_context',
                        'message': 'Keep this non-execution context.',
                    },
                ],
                'logs': [],
                'messages': [],
            }

            recovered = repair_or_retry(state)

        self.assertEqual(recovered['execution_status'], 'pending')
        self.assertEqual(recovered['retry_count'], 1)
        self.assertEqual(recovered['task_spec']['solver']['options']['max_iterations'], 6)
        self.assertEqual(recovered['task_spec']['runtime']['max_cycle'], 50)
        self.assertEqual([error['code'] for error in recovered['errors']], ['retained_context'])
        retry_log = next(
            entry for entry in recovered['logs']
            if entry.get('event') == 'workflow.retry_scheduled'
        )
        self.assertEqual(retry_log['details']['solver_options']['max_iterations'], 6)

    def test_fcdmft_numpy_compatibility_aliases_are_scoped(self):
        import numpy as np

        had_complex = 'complex' in np.__dict__
        had_int = 'int' in np.__dict__
        with _legacy_fcdmft_numpy_aliases(np):
            self.assertIs(np.complex, complex)
            self.assertIs(np.int, int)
        self.assertEqual('complex' in np.__dict__, had_complex)
        self.assertEqual('int' in np.__dict__, had_int)

    def test_gamma_rhf_reference_is_exposed_with_kpoint_axes(self):
        import numpy as np

        reference = mock.Mock()
        reference.mo_coeff = np.eye(2)
        reference.mo_occ = np.asarray([2.0, 0.0])
        reference.get_ovlp.return_value = np.eye(2)

        view = _libdmet_kpoint_mean_field(reference, restricted=True)

        self.assertEqual(view.mo_coeff.shape, (1, 2, 2))
        self.assertEqual(view.mo_occ.shape, (1, 2))
        self.assertEqual(view.get_ovlp().shape, (1, 2, 2))

    def test_gamma_uhf_reference_is_exposed_with_spin_and_kpoint_axes(self):
        import numpy as np

        reference = mock.Mock()
        reference.mo_coeff = np.stack((np.eye(2), np.eye(2)))
        reference.mo_occ = np.asarray([[1.0, 0.0], [1.0, 0.0]])
        reference.get_ovlp.return_value = np.eye(2)

        view = _libdmet_kpoint_mean_field(reference, restricted=False)

        self.assertEqual(view.mo_coeff.shape, (2, 1, 2, 2))
        self.assertEqual(view.mo_occ.shape, (2, 1, 2))
        self.assertEqual(view.get_ovlp().shape, (1, 2, 2))

    def test_selected_localized_coefficients_are_contiguous_for_pyscf_ao2mo(self):
        import numpy as np

        coefficients = np.arange(48.0).reshape(2, 4, 6).transpose(0, 2, 1)

        selected = _slice_coefficients(coefficients, (0, 2))

        self.assertEqual(selected.shape, (2, 6, 2))
        self.assertTrue(selected.flags.c_contiguous)

    def test_dmft_solver_ids_use_registry_canonical_names(self):
        self.assertEqual(normalize_solver_name('hf_dmft'), 'hf_dmft')
        self.assertEqual(normalize_solver_name('hf-dmft'), 'hf_dmft')
        self.assertEqual(normalize_solver_name('gw_dmft'), 'gw_dmft')

    def tearDown(self):
        clear_runtime_contexts()

    def test_periodic_task_round_trip_preserves_hf_dmft_solver(self):
        task = task_spec_from_dict({
            'task_type': 'periodic',
            'method': {'name': 'hf', 'restricted': True},
            'solver': {'name': 'hf_dmft', 'options': {'nval': 4}},
        })
        payload = task_spec_to_dict(task)

        self.assertEqual(payload['solver']['name'], 'hf_dmft')
        self.assertEqual(payload['solver']['options']['nval'], 4)

    def test_periodic_task_round_trip_preserves_gw_dmft_solver(self):
        task = task_spec_from_dict({
            'task_type': 'periodic',
            'method': {'name': 'dft', 'restricted': True, 'xc': 'pbe'},
            'solver': {
                'name': 'gw_dmft',
                'options': {'nval': 4, 'gw_full_self_energy': True},
            },
        })
        payload = task_spec_to_dict(task)

        self.assertEqual(payload['solver']['name'], 'gw_dmft')
        self.assertEqual(payload['solver']['options']['nval'], 4)
        self.assertTrue(payload['solver']['options']['gw_full_self_energy'])

    def test_serialized_periodic_task_is_not_reclassified_by_empty_model_defaults(self):
        task = task_spec_from_dict({
            'task_type': 'periodic',
            'method': {'name': 'hf', 'restricted': True},
            'solver': {'name': 'hf_dmft'},
        })

        reparsed = task_spec_from_partial(task_spec_to_dict(task))

        self.assertEqual(reparsed.task_type, 'periodic')
        self.assertEqual(reparsed.method.name, 'hf')
        self.assertEqual(reparsed.job.name, 'single_point')
        self.assertEqual(reparsed.solver.name, 'hf_dmft')

    def test_options_reject_unknown_fields(self):
        with self.assertRaisesRegex(ValueError, r'Unsupported HF\+DMFT solver option'):
            normalize_hf_dmft_options({'legacy_guess': True})

    def test_periodic_gw_options_and_reference_contract_are_explicit(self):
        task = task_spec_from_dict({
            'task_type': 'periodic',
            'method': {'name': 'dft', 'restricted': True, 'xc': 'pbe'},
            'solver': {
                'name': 'gw',
                'options': {
                    'gw_analytic_continuation': 'pade',
                    'gw_frequency_window': [0.0, 0.5],
                    'gw_real_frequency_points': 51,
                },
            },
            'periodic': {
                'kmesh': [2, 1, 1],
                'kpoint_scheme': 'gamma_centered',
                'kpoint_shift': [0, 0, 0],
                'density_fitting_method': 'gdf',
                'smearing_method': 'none',
            },
        })

        options = validate_periodic_gw_reference(task)

        self.assertEqual(options['gw_analytic_continuation'], 'pade')
        self.assertEqual(options['gw_real_frequency_points'], 51)
        self.assertTrue(options['gw_finite_size_correction'])

    def test_gw_dmft_options_require_full_matrix_self_energy(self):
        with self.assertRaisesRegex(ValueError, 'requires gw_full_self_energy=true'):
            normalize_gw_dmft_options({'gw_full_self_energy': False})

    def test_options_reject_nonpositive_target_occupancy(self):
        with self.assertRaisesRegex(ValueError, 'target_occupancy must be a positive'):
            normalize_hf_dmft_options({'target_occupancy': 0})

    def test_direct_style_bath_discretization_requires_two_energies(self):
        for method in ('opt', 'direct', 'log'):
            with self.subTest(method=method):
                with self.assertRaisesRegex(ValueError, 'requires nbath >= 2'):
                    normalize_hf_dmft_options({
                        'bath_discretization': method,
                        'nbath': 1,
                    })

        options = normalize_hf_dmft_options({
            'bath_discretization': 'linear',
            'nbath': 1,
        })
        self.assertEqual(options['nbath'], 1)

    def test_unapproved_hf_dmft_request_accepts_iao_preparation(self):
        task = task_spec_from_dict({
            'task_type': 'periodic',
            'method': {'name': 'hf', 'restricted': True},
            'solver': {'name': 'hf_dmft'},
            'embedding': {
                'enabled': True,
                'provider': 'fcdmft',
                'localization_method': 'iao',
                'approved': False,
            },
            'periodic': {
                'kmesh': [1, 1, 1],
                'kpoint_scheme': 'gamma_centered',
                'kpoint_shift': [0, 0, 0],
                'smearing_method': 'none',
            },
        })

        options = validate_hf_dmft_preparation_request(task)

        self.assertIsNone(options['nval'])

    def test_unapproved_hf_dmft_request_rejects_shifted_k_mesh(self):
        task = task_spec_from_dict({
            'task_type': 'periodic',
            'method': {'name': 'hf', 'restricted': True},
            'solver': {'name': 'hf_dmft'},
            'embedding': {
                'enabled': True,
                'provider': 'fcdmft',
                'localization_method': 'iao',
            },
            'periodic': {
                'kmesh': [2, 1, 1],
                'kpoint_scheme': 'gamma_centered',
                'kpoint_shift': [0.5, 0, 0],
            },
        })

        with self.assertRaisesRegex(ValueError, 'zero k-point shift'):
            validate_hf_dmft_preparation_request(task)

    def test_approved_artifacts_supply_local_hf_dmft_matrices(self):
        with tempfile.TemporaryDirectory() as directory:
            task = _approved_periodic_task(directory)
            options, inputs = validate_hf_dmft_request(task)
            artifact = read_embedding_hdf5(task.embedding.localized_hamiltonian_artifact['path'])

            self.assertEqual(options['nval'], 2)
            self.assertEqual(options['nb_per_e'], 2)
            self.assertIn('fock_local', artifact['datasets'])
            self.assertEqual(inputs['hcore_k'].shape, (1, 2, 2, 2))
            self.assertEqual(inputs['density_k'].shape, (1, 2, 2, 2))
            self.assertEqual(inputs['eri'].shape, (1, 2, 2, 2, 2))
            self.assertGreater(inputs['initial_hybridization_norm'], 0.0)

    def test_full_cell_gamma_subspace_is_rejected_before_bath_fitting(self):
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            task = _approved_periodic_task(directory)
            artifact_path = Path(task.embedding.localized_hamiltonian_artifact['path'])
            import h5py
            with h5py.File(artifact_path, 'r+') as handle:
                for name in ('one_body_local', 'fock_local', 'density_local'):
                    values = np.asarray(handle[name])
                    del handle[name]
                    handle[name] = values[:1]

            with self.assertRaisesRegex(ValueError, 'no resolvable lattice hybridization'):
                validate_hf_dmft_request(task)

    def test_auto_reference_uses_spin_to_infer_restricted_hf(self):
        with tempfile.TemporaryDirectory() as directory:
            task = _approved_periodic_task(directory)
            task.method.restricted = None

            options, inputs = validate_hf_dmft_request(task)

            self.assertEqual(options['impurity_solver'], 'cc')
            self.assertEqual(inputs['spin_channels'], 1)

    def test_reference_and_artifact_spin_layout_must_agree(self):
        with tempfile.TemporaryDirectory() as directory:
            task = _approved_periodic_task(directory)
            task.method.restricted = False
            task.solver.options['impurity_solver'] = 'ucc'

            with self.assertRaisesRegex(ValueError, 'localized artifacts disagree'):
                validate_hf_dmft_request(task)

    def test_generated_hf_dmft_script_runs_compiled_workflow(self):
        with tempfile.TemporaryDirectory() as directory:
            task = _approved_periodic_task(directory)

            script = generate_input_script(task)

            self.assertIn('run_workflow_sequential', script)
            self.assertIn("channel='generated-script'", script)
            self.assertNotIn('backend.execution import _run_pyscf_task', script)

    def test_adapter_runs_hf_dmft_without_claiming_total_energy(self):
        with tempfile.TemporaryDirectory() as directory:
            task = _approved_periodic_task(directory)
            instances = []

            def factory(*args, **kwargs):
                instance = _FakeDMFT(*args, **kwargs)
                instances.append(instance)
                return instance

            result, arrays, provider_log = run_hf_dmft(
                task,
                scratch_directory=Path(directory) / 'solver-fcdmft',
                reference_chemical_potential=0.0,
                reference_occupancy=2,
                dmft_factory=factory,
            )

            self.assertEqual(result['schema'], HF_DMFT_RESULT_SCHEMA)
            self.assertTrue(result['converged'])
            self.assertFalse(result['energy_available'])
            self.assertEqual(result['target_occupancy'], 2.0)
            self.assertEqual(set(arrays), {
                'frequencies',
                'frequency_weights',
                'hybridization',
                'self_energy',
            })
            self.assertEqual(provider_log, '')
            self.assertEqual(instances[0].kernel_options['occupancy'], 2.0)
            self.assertTrue((Path(directory) / 'solver-fcdmft' / 'dmft-checkpoint.h5').is_file())

    def test_periodic_gw_provider_registers_native_outputs_without_total_energy(self):
        class _Cell:
            nelectron = 2
            vol = 10.0

        class _MeanField:
            converged = True
            with_df = object()
            cell = _Cell()
            mo_energy = [[-1.0, 1.0], [-0.8, 1.2]]
            mo_occ = [[2.0, 0.0], [2.0, 0.0]]

        task = task_spec_from_dict({
            'task_type': 'periodic',
            'method': {'name': 'dft', 'restricted': True, 'xc': 'pbe'},
            'solver': {'name': 'gw'},
            'periodic': {
                'kmesh': [2, 1, 1],
                'kpoint_scheme': 'gamma_centered',
                'kpoint_shift': [0, 0, 0],
                'density_fitting_method': 'gdf',
                'smearing_method': 'none',
            },
        })

        with tempfile.TemporaryDirectory() as directory:
            result, arrays, provider_log, files = run_periodic_gw(
                _MeanField(),
                task,
                scratch_directory=Path(directory) / 'solver-gw',
                gw_factory=_FakePeriodicGW,
            )
            self.assertTrue(all(path.is_file() for path in files.values()))

        self.assertEqual(result['schema'], PERIODIC_GW_RESULT_SCHEMA)
        self.assertEqual(result['status'], 'completed')
        self.assertFalse(result['energy_available'])
        self.assertNotIn('converged', result)
        self.assertAlmostEqual(result['quasiparticle_gap'], 1.8)
        self.assertAlmostEqual(result['fermi_energy'], 0.1)
        self.assertEqual(arrays['quasiparticle_orbital_energies'].shape, (2, 2))
        self.assertTrue(arrays['quasiparticle_energy_available'].all())
        self.assertEqual(provider_log, '')

    def test_periodic_gw_masks_uncomputed_quasiparticle_energies(self):
        import numpy as np

        class _Cell:
            nelectron = 2
            vol = 10.0

        class _MeanField:
            converged = True
            with_df = object()
            cell = _Cell()
            mo_energy = [[-1.0, 1.0], [-0.8, 1.2]]
            mo_occ = [[2.0, 0.0], [2.0, 0.0]]

        task = task_spec_from_dict({
            'task_type': 'periodic',
            'method': {'name': 'dft', 'restricted': True, 'xc': 'pbe'},
            'solver': {
                'name': 'gw',
                'options': {
                    'gw_orbital_indices': [1],
                    'gw_kpoint_indices': [0],
                },
            },
            'periodic': {
                'kmesh': [2, 1, 1],
                'kpoint_scheme': 'gamma_centered',
                'kpoint_shift': [0, 0, 0],
                'density_fitting_method': 'gdf',
                'smearing_method': 'none',
            },
        })

        with tempfile.TemporaryDirectory() as directory:
            result, arrays, _, _ = run_periodic_gw(
                _MeanField(),
                task,
                scratch_directory=Path(directory) / 'solver-gw-window',
                gw_factory=_FakePeriodicGW,
            )

        availability = arrays['quasiparticle_energy_available']
        energies = arrays['quasiparticle_orbital_energies']
        self.assertEqual(int(availability.sum()), 1)
        self.assertTrue(availability[0, 1])
        self.assertAlmostEqual(energies[0, 1], 1.05)
        self.assertTrue(np.isnan(energies[0, 0]))
        self.assertTrue(np.isnan(energies[1, 1]))
        self.assertIsNone(result['quasiparticle_gap'])

    def test_gw_dmft_adapter_consumes_registered_gw_artifacts(self):
        with tempfile.TemporaryDirectory() as directory:
            task = _approved_gw_dmft_task(directory)
            options, inputs, paths = validate_gw_dmft_request(task)
            instances = []

            def factory(*args, **kwargs):
                instance = _FakeDMFT(*args, **kwargs)
                instances.append(instance)
                return instance

            scratch = Path(directory) / 'solver-gw-dmft'
            result, arrays, provider_log = run_gw_dmft(
                task,
                scratch_directory=scratch,
                reference_chemical_potential=0.0,
                reference_occupancy=2,
                dmft_factory=factory,
            )

            self.assertEqual(options['nval'], 2)
            self.assertEqual(inputs['jk_k'].shape, (1, 2, 2, 2))
            self.assertEqual(set(paths), {
                'lattice_result',
                'lattice_ac',
                'lattice_vxc',
                'lattice_sigma_imag',
                'local_ac',
                'local_sigma_imag',
                'local_transform',
            })
            self.assertTrue((scratch / 'ac_coeff.h5').is_file())
            self.assertTrue((scratch / 'imp_ac_coeff.h5').is_file())
            self.assertTrue((scratch / 'C_mo_lo.h5').is_file())

        self.assertEqual(result['schema'], GW_DMFT_RESULT_SCHEMA)
        self.assertEqual(result['method'], 'gw_dmft')
        self.assertTrue(result['converged'])
        self.assertFalse(result['energy_available'])
        self.assertEqual(set(arrays), {
            'frequencies',
            'frequency_weights',
            'hybridization',
            'self_energy',
        })
        self.assertTrue(instances[0].gw_dmft)
        self.assertEqual(provider_log, '')

    def test_adapter_preserves_provider_output_and_traceback_on_failure(self):
        class _FailingDMFT(_FakeDMFT):
            def kernel(self, **_kwargs):
                print('bath optimization started')
                raise ValueError('x0 is infeasible')

        with tempfile.TemporaryDirectory() as directory:
            task = _approved_periodic_task(directory)
            with self.assertRaises(FCDMFTExecutionError) as caught:
                run_hf_dmft(
                    task,
                    scratch_directory=Path(directory) / 'solver-fcdmft',
                    reference_chemical_potential=0.0,
                    reference_occupancy=2,
                    dmft_factory=_FailingDMFT,
                )

            self.assertEqual(caught.exception.stage, 'self_consistency')
            self.assertEqual(caught.exception.exception_type, 'ValueError')
            self.assertIn('bath optimization started', caught.exception.provider_log)
            self.assertIn('ValueError: x0 is infeasible', caught.exception.provider_log)

    def test_adapter_rejects_resources_above_slurm_allocation(self):
        with tempfile.TemporaryDirectory() as directory:
            task = _approved_periodic_task(directory)
            task.solver.options['n_threads'] = 17
            with mock.patch.dict(
                'os.environ',
                {'SLURM_CPUS_PER_TASK': '16', 'SLURM_MEM_PER_NODE': '32000'},
                clear=False,
            ):
                with self.assertRaisesRegex(ValueError, 'exceeds the Slurm allocation'):
                    run_hf_dmft(
                        task,
                        scratch_directory=Path(directory) / 'solver-fcdmft',
                        reference_chemical_potential=0.0,
                        reference_occupancy=2,
                        dmft_factory=_FakeDMFT,
                    )

    def test_adapter_inherits_slurm_threads_and_conservative_memory(self):
        with tempfile.TemporaryDirectory() as directory:
            task = _approved_periodic_task(directory)
            with mock.patch.dict(
                'os.environ',
                {'SLURM_CPUS_PER_TASK': '16', 'SLURM_MEM_PER_NODE': '32000'},
                clear=False,
            ):
                result, _arrays, _provider_log = run_hf_dmft(
                    task,
                    scratch_directory=Path(directory) / 'solver-fcdmft',
                    reference_chemical_potential=0.0,
                    reference_occupancy=2,
                    dmft_factory=_FakeDMFT,
                )

        resources = result['runtime_resources']
        self.assertEqual(resources['n_threads'], 16)
        self.assertEqual(resources['n_threads_source'], 'SLURM_CPUS_PER_TASK')
        self.assertEqual(resources['max_memory_mb'], 24000)
        self.assertEqual(resources['max_memory_source'], 'slurm_allocation_75_percent')
        self.assertEqual(resources['slurm_memory_allocation_mb'], 32000)

    def test_terminal_dmft_module_finalizes_reference_execution(self):
        class _MeanField:
            converged = True

        with tempfile.TemporaryDirectory() as directory:
            task = _approved_periodic_task(directory)
            lifecycle = new_lifecycle('task', 'hf-dmft-test')
            for event in (
                'validation_passed',
                'request_prepared',
                'execution_submitted',
                'execution_started',
            ):
                lifecycle = transition_lifecycle(lifecycle, event)
            context_ref = store_runtime_context(
                'hf-dmft-test',
                0,
                {
                    'mean_field': _MeanField(),
                    'fermi_energy': 0.0,
                    'electron_count': 2,
                },
            )
            state = {
                'run_id': 'hf-dmft-test',
                'work_dir': directory,
                'task_spec': task_spec_to_dict(task),
                'lifecycle': lifecycle,
                'execution_status': 'running',
                'retry_count': 1,
                'module_context_ref': context_ref,
                '_module_results': {
                    'task_type': 'periodic',
                    'method': 'hf',
                    'solver': 'hf_dmft',
                    'converged': True,
                    'final_energy': -2.5,
                    'analysis_text': '',
                    'raw_stdout': '',
                },
                'artifacts': [],
                'logs': [],
                'attempt_history': [],
                'errors': [],
            }
            invocation = ModuleInvocation(
                workflow_id='workflow-test',
                module_id='embedding.fcdmft.hf_dmft',
                runtime_id='provider.fcdmft.execute_hf_dmft',
                stage='task.execute',
                configuration={},
            )
            fake_result = {
                'schema': HF_DMFT_RESULT_SCHEMA,
                'converged': True,
                'energy_available': False,
            }
            with mock.patch(
                'pyscf_agent.providers.fcdmft.module.run_hf_dmft',
                return_value=(fake_result, {}, ''),
            ):
                updated = execute_hf_dmft(state, invocation)

            self.assertEqual(updated['execution_status'], 'succeeded')
            self.assertEqual(updated['lifecycle']['stage'], 'succeeded')
            self.assertIsNone(updated['structured_results']['final_energy'])
            self.assertEqual(
                updated['structured_results']['energy_kind'],
                'dmft_total_energy_unavailable',
            )
            self.assertEqual(updated['structured_results']['reference_energy'], -2.5)
            self.assertTrue(updated['structured_results']['reference_converged'])
            self.assertEqual(updated['structured_results']['dmft_result'], fake_result)
            self.assertEqual(
                compact_structured_results(updated['structured_results'])['dmft_result'],
                fake_result,
            )
            artifact_kinds = {
                artifact.get('kind')
                for artifact in updated.get('artifacts', [])
                if isinstance(artifact, dict)
            }
            self.assertIn('hf_dmft_output_log', artifact_kinds)
            output_log = next(
                artifact for artifact in updated['artifacts']
                if artifact.get('kind') == 'hf_dmft_output_log'
            )
            self.assertTrue(output_log['path'].endswith('log-fcdmft-output-retry-1.log'))

    def test_terminal_dmft_module_preserves_failure_artifacts_in_runtime_error(self):
        class _MeanField:
            converged = True

        with tempfile.TemporaryDirectory() as directory:
            task = _approved_periodic_task(directory)
            lifecycle = new_lifecycle('task', 'hf-dmft-failure')
            for event in (
                'validation_passed',
                'request_prepared',
                'execution_submitted',
                'execution_started',
            ):
                lifecycle = transition_lifecycle(lifecycle, event)
            context_ref = store_runtime_context(
                'hf-dmft-failure',
                0,
                {
                    'mean_field': _MeanField(),
                    'fermi_energy': 0.0,
                    'electron_count': 2,
                },
            )
            state = {
                'run_id': 'hf-dmft-failure',
                'work_dir': directory,
                'task_spec': task_spec_to_dict(task),
                'lifecycle': lifecycle,
                'execution_status': 'running',
                'module_context_ref': context_ref,
                '_module_results': {
                    'task_type': 'periodic',
                    'method': 'hf',
                    'solver': 'hf_dmft',
                    'converged': True,
                    'final_energy': -2.5,
                    'analysis_text': '',
                    'raw_stdout': '',
                },
                'artifacts': [],
                'logs': [],
                'attempt_history': [],
                'errors': [],
            }
            invocation = ModuleInvocation(
                workflow_id='workflow-test',
                module_id='embedding.fcdmft.hf_dmft',
                runtime_id='provider.fcdmft.execute_hf_dmft',
                stage='task.execute',
                configuration={},
            )
            provider_error = FCDMFTExecutionError(
                'fcDMFT HF+DMFT failed during self_consistency: x0 is infeasible',
                stage='self_consistency',
                provider_log='bath optimization started\nValueError: x0 is infeasible\n',
                scratch_directory=Path(directory) / 'solver-fcdmft',
                exception_type='ValueError',
            )
            with mock.patch(
                'pyscf_agent.providers.fcdmft.module.run_hf_dmft',
                side_effect=provider_error,
            ):
                with self.assertRaises(WorkflowRuntimeError) as caught:
                    execute_hf_dmft(state, invocation)

            failure_state = caught.exception.state
            artifact_kinds = {
                artifact.get('kind')
                for artifact in failure_state.get('artifacts', [])
                if isinstance(artifact, dict)
            }
            self.assertIn('hf_dmft_result', artifact_kinds)
            self.assertIn('hf_dmft_output_log', artifact_kinds)
            self.assertEqual(
                failure_state['_module_results']['dmft_result']['failure_stage'],
                'self_consistency',
            )
            self.assertIsNone(failure_state['_module_results']['final_energy'])
            self.assertIsNone(failure_state['_module_results']['energy'])
            self.assertEqual(failure_state['_module_results']['reference_energy'], -2.5)
            self.assertTrue(failure_state['_module_results']['reference_converged'])
            self.assertEqual(
                failure_state['_module_results']['energy_kind'],
                'dmft_total_energy_unavailable',
            )

    def test_subspace_module_finishes_hf_and_requests_review(self):
        class _MeanField:
            converged = True

        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        task = task_spec_from_dict({
            'task_type': 'periodic',
            'method': {'name': 'hf', 'restricted': True},
            'solver': {'name': 'hf_dmft'},
            'embedding': {
                'enabled': True,
                'provider': 'fcdmft',
                'localization_method': 'iao',
                'approved': False,
            },
        })
        lifecycle = new_lifecycle('task', 'hf-dmft-proposal')
        for event in ('validation_passed', 'request_prepared', 'execution_submitted', 'execution_started'):
            lifecycle = transition_lifecycle(lifecycle, event)
        context_ref = store_runtime_context(
            'hf-dmft-proposal',
            0,
            {'mean_field': _MeanField(), 'cell': object(), 'restricted': True},
        )
        state = {
            'run_id': 'hf-dmft-proposal',
            'work_dir': directory.name,
            'task_spec': task_spec_to_dict(task),
            'lifecycle': lifecycle,
            'execution_status': 'running',
            'module_context_ref': context_ref,
            '_module_results': {
                'task_type': 'periodic',
                'method': 'hf',
                'converged': True,
                'final_energy': -1.0,
                'analysis_text': '',
            },
            'artifacts': [],
            'logs': [],
            'errors': [],
            'attempts': [],
        }
        embedding_patch = {
            'enabled': True,
            'provider': 'fcdmft',
            'localization_method': 'iao',
            'correlated_orbital_indices': [0, 1],
            'approved': False,
            'audit_artifact': {'kind': 'correlated_subspace_audit', 'path': '/tmp/audit.json'},
            'reference_artifact': {'kind': 'embedding_reference', 'path': '/tmp/reference.h5'},
            'localized_subspace_artifact': {'kind': 'localized_subspace', 'path': '/tmp/subspace.h5'},
            'localized_hamiltonian_artifact': {'kind': 'localized_hamiltonian', 'path': '/tmp/hamiltonian.h5'},
        }
        prepared = {
            'localized_orbital_count': 2,
            'source_orbital_indices': [3, 4],
            'estimated_eri_memory_mb': 0.001,
            'audit': {
                'schema': 'pyscf-agent.correlated-subspace-audit.v1',
                'orbital_indices': [0, 1],
                'orbital_records': [],
                'approval': {'approved': False, 'status': 'pending'},
            },
            'task_spec_patch': {
                'embedding': embedding_patch,
                'solver': {'name': 'hf_dmft', 'options': {'ncore': 0, 'nval': 2}},
            },
        }
        invocation = ModuleInvocation(
            workflow_id='workflow-proposal',
            module_id='embedding.fcdmft.prepare_subspace',
            runtime_id='pyscf_agent.providers.fcdmft.prepare_hf_dmft_subspace',
            stage='task.execute',
            configuration={},
        )
        with mock.patch(
            'pyscf_agent.providers.fcdmft.module.prepare_periodic_fcdmft_subspace',
            return_value=prepared,
        ):
            updated = prepare_hf_dmft_subspace(state, invocation)
        updated = result_extractor(updated)
        updated = task_reporter(updated)

        self.assertEqual(updated['execution_status'], 'succeeded')
        self.assertEqual(updated['lifecycle']['stage'], 'review_required')
        self.assertEqual(updated['task_report']['approval']['type'], 'correlated_subspace')
        self.assertEqual(
            updated['task_report']['approval']['task_spec_patch']['solver']['options']['nval'],
            2,
        )

    def test_registry_compiles_hf_then_dmft_as_one_task(self):
        registry = default_registry()
        task = {
            'task_type': 'periodic',
            'method': {'name': 'hf', 'restricted': True},
            'solver': {'name': 'hf_dmft'},
            'embedding': {'enabled': True, 'provider': 'fcdmft', 'approved': True},
        }
        workflow = compile_task_workflow(task, registry.modules_for(scope='task'))

        self.assertEqual(registry.registry_issues(), ())
        self.assertLess(
            workflow.execution_order.index('core.execution'),
            workflow.execution_order.index('embedding.fcdmft.prepare_subspace'),
        )
        self.assertLess(
            workflow.execution_order.index('embedding.fcdmft.prepare_subspace'),
            workflow.execution_order.index('embedding.fcdmft.hf_dmft'),
        )
        self.assertEqual(
            workflow.hooks['task.execute'],
            (
                'core.execution',
                'embedding.fcdmft.prepare_subspace',
                'embedding.fcdmft.hf_dmft',
            ),
        )

    def test_registry_compiles_dft_gw_local_dc_then_dmft_as_one_task(self):
        registry = default_registry()
        task = {
            'task_type': 'periodic',
            'method': {'name': 'dft', 'restricted': True, 'xc': 'pbe'},
            'solver': {'name': 'gw_dmft'},
            'embedding': {'enabled': True, 'provider': 'fcdmft', 'approved': True},
        }
        workflow = compile_task_workflow(task, registry.modules_for(scope='task'))

        self.assertEqual(
            workflow.hooks['task.execute'],
            (
                'core.execution',
                'embedding.fcdmft.periodic_gw',
                'embedding.fcdmft.prepare_subspace',
                'embedding.fcdmft.gw_double_counting',
                'embedding.fcdmft.gw_dmft',
            ),
        )
        self.assertEqual(registry.registry_issues(), ())


if __name__ == '__main__':
    unittest.main()
