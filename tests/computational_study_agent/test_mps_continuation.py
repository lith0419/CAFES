from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest import mock

from computational_study_agent.adaptive.mps_continuation import (
    build_mps_continuation_plan,
    merge_mps_continuation_report,
)
from computational_study_agent.application import StudyApplicationService
from computational_study_agent.schema import StudyCase, StudyPlan, StudyReport


def _manifest(*, spin=0, n_electrons=(1, 1), scratch='/remote/shared/case-0001'):
    return {
        'schema': 'pyscf-agent.block2-mps-manifest.v1',
        'scratch_directory': scratch,
        'n_orbitals': 2,
        'n_electrons': list(n_electrons),
        'spin': spin,
        'symmetry': 'su2',
        'nroots': 1,
        'mps_tag': 'GS',
        'files': [{'relative_path': 'GS-MPS_INFO', 'size': 128}],
    }


def _request(*, nelec=(1, 1)):
    return {
        'task_type': 'model_hamiltonian',
        'solver': {
            'name': 'block2_dmrg',
            'options': {'preset': 'screening', 'symmetry': 'su2', 'nroots': 1},
        },
        'model_hamiltonian': {
            'spec': {
                'sites': [{'id': 0}, {'id': 1}],
                'nelec': list(nelec),
            },
        },
    }


def _case_record(case_id, u_value, status, *, manifest=None, nelec=(1, 1)):
    structured = {'solver': 'block2_dmrg'}
    artifacts = []
    if manifest is not None:
        structured['cas_result'] = {
            'solver': 'block2_dmrg',
            'dmrg_result': {'checkpoint_manifest': manifest},
        }
        artifacts.append({
            'kind': 'block2_mps_manifest',
            'path': '{0}/mps-manifest.json'.format(manifest['scratch_directory']),
        })
    return {
        'case_id': case_id,
        'label': 'U={0}'.format(u_value),
        'variables': {'U': u_value},
        'request': _request(nelec=nelec),
        'task_report': {
            'execution_status': status,
            'structured_results': structured,
            'artifacts': artifacts,
        },
    }


def _plan():
    return StudyPlan(
        study_id='mps-study',
        name='mps-study',
        objective='scan U',
        system_type='model_hamiltonian',
        cases=[
            StudyCase('case-0001', 'U=1', _request(), variables={'U': 1}),
            StudyCase('case-0002', 'U=2', _request(), variables={'U': 2}),
            StudyCase('case-0003', 'U=5', _request(), variables={'U': 5}),
        ],
        observables=['energy'],
    )


def _report(work_dir):
    cases = [
        _case_record('case-0001', 1, 'succeeded', manifest=_manifest()),
        _case_record('case-0002', 2, 'unconverged'),
        _case_record(
            'case-0003',
            5,
            'succeeded',
            manifest=_manifest(scratch='/remote/shared/case-0003'),
        ),
    ]
    return {
        'study_id': 'mps-study',
        'name': 'mps-study',
        'objective': 'scan U',
        'system_type': 'model_hamiltonian',
        'status': 'completed_with_issues',
        'work_dir': str(work_dir),
        'cases': cases,
        'comparison_table': [
            {'case_id': 'case-0001', 'U': 1, 'status': 'succeeded'},
            {'case_id': 'case-0002', 'U': 2, 'status': 'unconverged'},
            {'case_id': 'case-0003', 'U': 5, 'status': 'succeeded'},
        ],
        'artifacts': [],
        'summary': 'two succeeded, one unresolved',
    }


class MpsContinuationTests(unittest.TestCase):
    def test_builder_selects_nearest_compatible_anchor(self):
        payload = build_mps_continuation_plan(_plan(), _report('/tmp/mps-study'))

        self.assertIsNotNone(payload)
        decision = payload['mps_continuation_decisions'][0]
        retry = payload['mps_continuation_plan']['cases'][0]
        options = retry['request']['solver']['options']
        self.assertEqual(decision['case_id'], 'case-0002')
        self.assertEqual(decision['source_case_id'], 'case-0001')
        self.assertTrue(decision['compatibility']['compatible'])
        self.assertEqual(options['restart_manifest']['scratch_directory'], '/remote/shared/case-0001')
        self.assertTrue(options['restart_required'])
        self.assertEqual(retry['request']['solver']['name'], 'block2_dmrg')

    def test_builder_rejects_incompatible_quantum_sector(self):
        report = _report('/tmp/mps-study')
        report['cases'][0] = _case_record(
            'case-0001',
            1,
            'succeeded',
            manifest=_manifest(spin=2, n_electrons=(2, 0)),
            nelec=(2, 0),
        )
        report['cases'][2] = _case_record(
            'case-0003',
            5,
            'succeeded',
            manifest=_manifest(spin=2, n_electrons=(2, 0), scratch='/remote/shared/case-0003'),
            nelec=(2, 0),
        )

        self.assertIsNone(build_mps_continuation_plan(_plan(), report))

    def test_builder_does_not_use_state_ambiguous_cases_as_anchors(self):
        report = _report('/tmp/mps-study')
        report['adaptive'] = {
            'dmrg_state_tracking': {
                'schema': 'pyscf-agent.dmrg-state-tracking.v1',
                'status': 'review_required',
                'ambiguous_case_ids': ['case-0001', 'case-0003'],
            },
        }

        self.assertIsNone(build_mps_continuation_plan(_plan(), report))

    def test_molecular_casscf_builder_requires_and_uses_joint_checkpoint(self):
        def request(distance):
            return {
                'task_type': 'molecular',
                'system': {
                    'atom': 'H 0 0 0; H 0 0 {0}'.format(distance),
                    'basis': '6-31g',
                    'spin': 0,
                },
                'method': 'casscf',
                'solver': {
                    'name': 'block2_dmrg',
                    'options': {'symmetry': 'su2', 'nroots': 1},
                },
                'active_space': {
                    'enabled': True,
                    'ncas': 2,
                    'nelecas': 2,
                    'approved': True,
                },
            }

        manifest = _manifest()
        manifest.update({
            'checkpoint_components': ['mps', 'optimized_orbitals'],
            'orbital_context': {
                'kind': 'casscf_optimized_orbitals_and_mps',
                'external_restart_supported': True,
                'ncas': 2,
                'nelecas': [1, 1],
            },
        })
        plan = StudyPlan(
            study_id='molecular-mps-study',
            name='molecular-mps-study',
            objective='scan H2',
            system_type='molecular',
            cases=[
                StudyCase('case-0001', 'R=1.0', request(1.0), variables={'R': 1.0}),
                StudyCase('case-0002', 'R=1.1', request(1.1), variables={'R': 1.1}),
            ],
            observables=['energy'],
        )
        source = {
            'case_id': 'case-0001',
            'label': 'R=1.0',
            'variables': {'R': 1.0},
            'request': request(1.0),
            'task_report': {
                'execution_status': 'succeeded',
                'structured_results': {
                    'cas_result': {
                        'dmrg_result': {'checkpoint_manifest': manifest},
                    },
                },
                'artifacts': [],
            },
        }
        target = {
            'case_id': 'case-0002',
            'label': 'R=1.1',
            'variables': {'R': 1.1},
            'request': request(1.1),
            'task_report': {'execution_status': 'unconverged'},
        }

        payload = build_mps_continuation_plan(
            plan,
            {'system_type': 'molecular', 'cases': [source, target]},
        )

        self.assertIsNotNone(payload)
        decision = payload['mps_continuation_decisions'][0]
        retry = payload['mps_continuation_plan']['cases'][0]
        self.assertEqual(
            decision['checkpoint_components'],
            ['mps', 'optimized_orbitals'],
        )
        self.assertIn('restart_manifest', retry['request']['solver']['options'])

    def test_service_requires_explicit_approval_and_can_skip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            runner = mock.Mock()
            service = StudyApplicationService(study_runner=runner)
            pending = service.run_result_mps_continuation(_report(tmpdir), _plan().to_dict())

            self.assertEqual(pending['status'], 'approval_required')
            self.assertEqual(runner.call_count, 0)
            skipped = service.run_result_mps_continuation(
                _report(tmpdir),
                _plan().to_dict(),
                approval={
                    'decision': 'skip',
                    'approval_token': pending['approval']['approval_token'],
                },
            )
            self.assertEqual(skipped['status'], 'skipped')
            self.assertEqual(runner.call_count, 0)

    def test_service_rejects_a_different_execution_target(self):
        report = _report('/tmp/mps-study')
        report['execution'] = {
            'executor': {
                'executor_id': 'slurm',
                'execution_mode': 'remote_scheduler',
                'location': 'remote_slurm_cluster',
                'transport': 'ssh',
                'remote_profile': 'amarel',
            },
        }
        local_executor = mock.Mock()
        local_executor.describe.return_value = {
            'executor_id': 'local',
            'execution_mode': 'synchronous',
            'location': 'current_process',
        }
        service = StudyApplicationService(
            study_runner=mock.Mock(),
            task_executor=local_executor,
        )

        result = service.run_result_mps_continuation(report, _plan().to_dict())

        self.assertEqual(result['status'], 'unavailable')
        self.assertIn('same execution target', result['summary'])

    def test_approved_retry_runs_subset_and_merges_into_full_report(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            continuation_report = StudyReport(
                study_id='mps-study-mps-continuation',
                name='mps-study-mps-continuation',
                objective='retry one case',
                system_type='model_hamiltonian',
                status='succeeded',
                work_dir=str(Path(tmpdir) / 'mps-continuation-restarts'),
                cases=[_case_record('case-0002', 2, 'succeeded')],
                comparison_table=[
                    {'case_id': 'case-0002', 'U': 2, 'status': 'succeeded', 'final_energy': -1.2},
                ],
                artifacts=[{'kind': 'study_report', 'path': 'continuation.json'}],
            )
            runner = mock.Mock(return_value=continuation_report)
            service = StudyApplicationService(study_runner=runner)
            pending = service.run_result_mps_continuation(_report(tmpdir), _plan().to_dict())
            completed = service.run_result_mps_continuation(
                _report(tmpdir),
                _plan().to_dict(),
                approval={
                    'decision': 'approve',
                    'approval_token': pending['approval']['approval_token'],
                },
            )

            self.assertEqual(completed['status'], 'validated')
            self.assertEqual(runner.call_count, 1)
            retry_plan = runner.call_args.args[0]
            self.assertEqual([case.case_id for case in retry_plan.cases], ['case-0002'])
            self.assertFalse(runner.call_args.kwargs['batch_independent'])
            rows = {row['case_id']: row for row in completed['report']['comparison_table']}
            self.assertEqual(set(rows), {'case-0001', 'case-0002', 'case-0003'})
            self.assertEqual(rows['case-0002']['status'], 'succeeded')
            self.assertEqual(rows['case-0002']['mps_restart_source_case'], 'case-0001')
            self.assertEqual(completed['report']['status'], 'succeeded')

    def test_merger_preserves_original_successful_cases(self):
        report = _report('/tmp/mps-study')
        continuation = {
            'cases': [_case_record('case-0002', 2, 'succeeded')],
            'comparison_table': [{'case_id': 'case-0002', 'status': 'succeeded'}],
            'artifacts': [],
        }
        merged = merge_mps_continuation_report(
            report,
            continuation,
            [{'case_id': 'case-0002', 'source_case_id': 'case-0001'}],
        )

        self.assertEqual([case['case_id'] for case in merged['cases']], [
            'case-0001', 'case-0002', 'case-0003',
        ])
        self.assertEqual(merged['status'], 'succeeded')


if __name__ == '__main__':
    unittest.main()
