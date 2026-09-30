from __future__ import annotations

import unittest
import tempfile
from types import SimpleNamespace
from unittest import mock

from pyscf_agent.providers.libdmet.dmet import normalize_dmet_options, _solver
from pyscf_agent.providers.libdmet.recovery import dmet_scf_recovery, dmet_task_recovery


DIIS_TRACEBACK = '''Traceback (most recent call last):
  File "/env/libdmet/solver/fci.py", line 122, in run
  File "/env/libdmet/solver/scf.py", line 1003, in HF
  File "/env/pyscf/lib/diis.py", line 254, in extrapolate
numpy.linalg.LinAlgError: Internal Error.
'''


def failed_dmet_task(options=None):
    return {
        'execution_status': 'failed',
        'task_spec': {'solver': {'name': 'dmet', 'options': options or {}}},
        'errors': [{
            'stage': 'execution', 'code': 'execution_exception',
            'exception_type': 'LinAlgError',
            'details': {'traceback': DIIS_TRACEBACK},
        }],
    }


class DmetRecoveryTests(unittest.TestCase):
    def test_runner_records_numerical_recovery_without_a_memory_error_code(self):
        from pyscf_agent.backend.execution import runner
        from pyscf_agent.backend.state import default_state
        from pyscf_agent.contracts import TaskSpec, SolverSpec, task_spec_to_dict

        error_type = type('LinAlgError', (Exception,), {})
        with tempfile.TemporaryDirectory() as work_dir:
            state = default_state('DMET', work_dir=work_dir)
            state['task_spec'] = task_spec_to_dict(TaskSpec(
                task_type='model_hamiltonian', solver=SolverSpec(name='dmet'),
            ))
            with mock.patch('pyscf_agent.backend.execution._run_pyscf_task', side_effect=error_type('Internal Error.')):
                with mock.patch('pyscf_agent.backend.execution.traceback.format_exc', return_value=DIIS_TRACEBACK):
                    report = runner(state)
        self.assertEqual(report['execution_status'], 'failed')
        error = report['errors'][-1]
        self.assertEqual(error['code'], 'execution_exception')
        self.assertEqual(error['details']['recovery_recommendation']['recommended_action'],
            'retry_dmet_without_impurity_diis')

    def test_option_is_registered_validated_and_scoped_to_native_impurity_solvers(self):
        self.assertIs(normalize_dmet_options({})['impurity_scf_diis'], True)
        self.assertIs(normalize_dmet_options({'impurity_scf_diis': 'false'})['impurity_scf_diis'], False)
        for invalid in (None, [], 'maybe', 0):
            with self.subTest(value=invalid), self.assertRaisesRegex(ValueError, 'must be a boolean'):
                normalize_dmet_options({'impurity_scf_diis': invalid})
        with self.assertRaisesRegex(ValueError, 'does not use a preliminary impurity SCF'):
            normalize_dmet_options({'impurity_solver': 'block2', 'impurity_scf_diis': False})

    def test_option_reaches_every_impurity_scf_call_without_changing_other_instances(self):
        for name in ('fci',):
            with self.subTest(solver=name):
                factory = lambda **_kwargs: SimpleNamespace(scfsolver=SimpleNamespace(HF=mock.Mock()))
                provider = SimpleNamespace(impurity_solver=SimpleNamespace(FCI=factory, CCSD=factory))
                config = dict(normalize_dmet_options({'impurity_solver': name, 'impurity_scf_diis': False}), libdmet_sz=0)
                with mock.patch('pyscf_agent.providers.libdmet.dmet._adapt_unrestricted_solver_for_pyscf', side_effect=lambda solver, _: solver):
                    changed = _solver(provider, config, None)
                    default = _solver(provider, dict(config, impurity_scf_diis=True), None)
                for _ in range(2):
                    changed.scfsolver.HF(MaxIter=200, InitGuess='folded_density')
                self.assertEqual(changed.scfsolver.HF.func.call_args_list, [
                    mock.call(do_diis=False, MaxIter=200, InitGuess='folded_density'),
                ] * 2)
                default.scfsolver.HF(MaxIter=200)
                default.scfsolver.HF.assert_called_once_with(MaxIter=200)

    def test_ccsd_configures_its_own_scf_policy_and_honors_diis_option(self):
        factory = mock.Mock(side_effect=lambda **kwargs: SimpleNamespace())
        provider = SimpleNamespace(impurity_solver=SimpleNamespace(CCSD=factory))
        for enabled in (True, False):
            config = dict(normalize_dmet_options({
                'impurity_solver': 'ccsd', 'impurity_scf_diis': enabled,
                'impurity_solver_options': {'beta': 1000},
            }), libdmet_sz=0)
            with mock.patch('pyscf_agent.providers.libdmet.dmet.configure_ccsd_impurity_scf',
                            side_effect=lambda solver, **kwargs: solver) as configure:
                with mock.patch('pyscf_agent.providers.libdmet.dmet._adapt_unrestricted_solver_for_pyscf',
                                side_effect=lambda solver, _: solver):
                    solver = _solver(provider, config, None)
            configure.assert_called_once_with(solver, use_diis=enabled)
            self.assertFalse(factory.call_args.kwargs['scf_newton'])
            self.assertEqual(factory.call_args.kwargs['beta'], 1000.0)

    def test_old_failed_report_produces_concrete_proposal(self):
        proposal = dmet_task_recovery(failed_dmet_task())
        self.assertEqual(proposal['solver_options_patch'], {'impurity_scf_diis': False})
        self.assertEqual(proposal['recommended_action'], 'retry_dmet_without_impurity_diis')
        self.assertIn('previously true', proposal['summary'])
        self.assertFalse(proposal['automatic_retry_safe'])

    def test_unrelated_failure_and_already_applied_proposal_are_not_repeated(self):
        variants = (
            ('ValueError', DIIS_TRACEBACK),
            ('LinAlgError', DIIS_TRACEBACK.replace('in extrapolate', 'in diagonalize')),
            ('LinAlgError', DIIS_TRACEBACK.replace('libdmet/solver/scf.py', 'lattice.py')),
            ('LinAlgError', DIIS_TRACEBACK.replace('libdmet/solver/fci.py', 'some_solver.py')),
        )
        for kind, trace in variants:
            with self.subTest(kind=kind, trace=trace):
                self.assertIsNone(dmet_scf_recovery({}, exception_type=kind, traceback_text=trace))
        task = failed_dmet_task({'impurity_scf_diis': False})
        self.assertIsNone(dmet_task_recovery(task))
        task = failed_dmet_task()
        task['execution_status'] = 'succeeded'
        self.assertIsNone(dmet_task_recovery(task))
        self.assertIsNone(dmet_task_recovery(failed_dmet_task(), {'solver': 'fci'}))

    def test_ccsd_uses_its_own_failure_evidence(self):
        proposal = dmet_scf_recovery(
            {'impurity_solver': 'ccsd'}, exception_type='LinAlgError',
            traceback_text=DIIS_TRACEBACK.replace('solver/fci.py', 'solver/cc.py'),
        )
        self.assertEqual(proposal['solver_options_patch'], {'impurity_scf_diis': False})


if __name__ == '__main__':
    unittest.main()
