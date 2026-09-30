import copy
import json
import tempfile
import unittest
from pathlib import Path

from computational_study_agent.application import StudyApplicationService
from computational_study_agent.dmet_branches import (
    analyze_branches,
    candidates,
    case_identity,
    classify,
    normalize_policy,
    preserve_candidates,
    select_donor,
)
from computational_study_agent.dmet_continuation import resolve_dmet_density
from computational_study_agent.schema import StudyCase
from computational_study_agent.study_state import checkpoint_payload, load_checkpoint
from tests.computational_study_agent.test_retry_collection import (
    Remote,
    plan_for_service,
)
from tests.computational_study_agent.test_study_background import inline_launcher
from tests.pyscf_agent.test_libdmet_dmet import _ring_spec


def record(case_id, v, branch='afm', energy=-1, *, run_id=None, seed='af', u=2):
    spec = _ring_spec(6, u)
    for bond in spec['bonds']:
        bond['V'] = bond['effective_V'] = v
    request = {
        'task_type': 'model_hamiltonian',
        'model_hamiltonian': {'spec': spec},
        'solver': {
            'name': 'dmet',
            'options': {
                'execution_mode': 'finite_graph',
                'impurity_size': 2,
                'interacting_bath': False,
                'bath_spin_dimension_policy': 'max',
                'reference': 'unrestricted',
                'reference_density_guess': seed,
            },
        },
    }
    charge, magnetic = {
        'afm': (0, 0.4),
        'cdw': (0.5, 0),
        'mixed': (0.3, 0.3),
        'near_unordered': (0, 0),
    }[branch]
    report = {
        'run_id': run_id or case_id,
        'execution_status': 'succeeded',
        'structured_results': {
            'converged': True,
            'solver': 'dmet',
            'energy_per_site': energy,
            'energy': energy * 6,
            'energy_unit': 'a.u.',
            'quality_checks': [
                {
                    'id': 'convergence',
                    'category': 'convergence',
                    'required': True,
                    'operator': 'equal',
                    'source': 'dmet',
                    'observed': True,
                    'expected': True,
                }
            ],
            'strong_correlation_diagnostics': {
                'diagnostics': [
                    {
                        'name': 'sublattice_order_parameters',
                        'value': {
                            'charge_imbalance': charge,
                            'staggered_magnetization': magnetic,
                        },
                    }
                ]
            },
        },
        'artifacts': [
            {
                'kind': 'dmet_mean_field_state',
                'path': '/shared/' + (run_id or case_id) + '.npz',
                'sha256': 'a' * 64,
            }
        ],
    }
    return {
        'case_id': case_id,
        'request': request,
        'variables': {'U': u, 'V': v},
        'attempt_count': 1,
        'task_report': report,
    }


class BranchRemote(Remote):
    def __init__(self):
        super().__init__(fail_first=False)
        self.run_branches = {}

    def _report(self, task_id):
        task = next(t for t in self.submissions[-1] if t.task_id == task_id)
        request = json.loads(task.request)
        v = request['model_hamiltonian']['spec']['bonds'][0]['V']
        source = request['solver']['options'].get('reference_density_source')
        branch = (
            self.run_branches[source['source_run_id']]
            if source
            else 'afm'
            if v == 0
            else 'cdw'
        )
        self.run_branches[task.run_id] = branch
        energy = -1 + 0.4 * v if branch == 'afm' else -0.2 - 0.4 * v
        return record(task_id, v, branch, energy, run_id=task.run_id)['task_report']


class DmetBranchTests(unittest.TestCase):
    def test_classify_final_order_not_seed_and_ambiguous_not_a_branch(self):
        sample = record('a', 1, 'cdw', seed='af')
        self.assertEqual(candidates([sample])[0]['branch'], 'cdw')
        results = sample['task_report']['structured_results']
        order = results['strong_correlation_diagnostics']['diagnostics'][0]['value']
        order['charge_imbalance'] = 1e-3
        self.assertEqual(classify(results, normalize_policy())['branch'], 'ambiguous')
        del order['charge_imbalance']
        self.assertEqual(classify(results, normalize_policy())['branch'], 'unknown')
        for policy in (
            {'charge_threshold': float('nan')},
            {'distance_scales': {'U': 0, 'V': 1}},
            {'threshold_margin': 1},
        ):
            with self.assertRaises(ValueError):
                normalize_policy(policy)

    def test_nearest_donor_respects_branch_quality_direction_and_compatibility(self):
        target = StudyCase.from_dict(record('target', 2))
        available = [
            record('af', 0),
            record('cdw', 1.99, 'cdw', seed='af'),
            record('failed', 1.8),
            record('wrong-u', 1.9, u=4),
            record('ahead', 3),
        ]
        available[2]['task_report']['structured_results']['quality_checks'][0][
            'observed'
        ] = False
        donor = select_donor(
            target, candidates(available), 'afm', direction='increasing'
        )
        self.assertEqual(donor['case_id'], 'af')
        self.assertEqual(
            select_donor(target, candidates(available), 'afm', direction='nearest')[
                'case_id'
            ],
            'ahead',
        )
        changed = copy.deepcopy(target)
        changed.request['solver']['options']['reference'] = 'restricted'
        changed.request['solver']['options']['reference_density_guess'] = 'pm'
        self.assertIsNone(select_donor(changed, candidates(available), 'afm'))
        available[0]['task_report']['artifacts'] = []
        self.assertIsNone(
            select_donor(target, candidates(available), 'afm', direction='increasing')
        )

    def test_history_survives_failure_disk_round_trip_and_pinned_donor_resolution(self):
        first = record('case-a', 0, energy=-2)
        current = record('case-a', 0, energy=-10, run_id='retry')
        current['task_report']['execution_status'] = 'unconverged'
        preserve_candidates(first, current)
        with tempfile.TemporaryDirectory() as root:
            path = Path(root) / 'study-state.json'
            payload = checkpoint_payload(
                path,
                {
                    'schema': 'pyscf-agent.study-state.v1',
                    'study_id': 'study',
                    'cases': {'case-a': current},
                },
            )
            self.assertNotIn(
                'task_report', payload['cases']['case-a']['candidate_runs'][0]
            )
            path.write_text(json.dumps(payload))
            state = load_checkpoint(path, 'study')
        analysis = analyze_branches(state['cases'].values())
        self.assertEqual(analysis['points'][0]['winner']['run_id'], 'case-a')
        self.assertEqual(len(analysis['points'][0]['candidates']), 2)
        target = StudyCase.from_dict(record('target', 1))
        target.request['solver']['options']['reference_density_source'] = {
            'source_case_id': 'case-a',
            'source_run_id': 'case-a',
        }
        self.assertIsNone(resolve_dmet_density(target, state))
        self.assertEqual(
            target.request['solver']['options']['reference_density_source']['path'],
            '/shared/case-a.npz',
        )

    def test_coexistence_sign_symmetry_crossings_gaps_and_lower_third_state(self):
        rows = [
            record('af0', 0, 'afm', -1),
            record('cdw0', 0, 'cdw', 0),
            record('af1', 1, 'afm', 0),
            record('cdw1', 1, 'cdw', -1),
        ]
        analysis = analyze_branches(rows)
        self.assertEqual(len(analysis['points']), 2)
        self.assertTrue(all(p['possible_hysteresis'] for p in analysis['points']))
        self.assertAlmostEqual(analysis['crossings'][0]['V_linear_estimate'], 0.5)
        self.assertEqual(analysis['crossings'][0]['status'], 'candidate_crossing')
        rows.append(record('mixed0', 0, 'mixed', -2))
        self.assertEqual(
            analyze_branches(rows)['crossings'][0]['status'], 'lower_competing_state'
        )
        rows.append(record('middle', 0.5, 'afm', -0.5))
        self.assertEqual(analyze_branches(rows)['crossings'], [])
        a, b = record('a', 0), record('b', 0)
        b['task_report']['structured_results']['strong_correlation_diagnostics'][
            'diagnostics'
        ][0]['value']['staggered_magnetization'] = -0.4
        self.assertFalse(analyze_branches([a, b])['points'][0]['possible_hysteresis'])

    def test_public_bidirectional_sweep_reselects_new_and_historical_runs(self):
        remote = BranchRemote()
        service = StudyApplicationService(
            task_executor=remote, execution_config={'execution_target': 'remote'}
        )
        service._study_launcher = inline_launcher(service)
        plan = plan_for_service(3)
        plan.cases = [
            StudyCase.from_dict(record(f'case-{i + 1:04d}', i)) for i in range(3)
        ]
        with tempfile.TemporaryDirectory() as root:
            before = service.run_study(plan, work_dir=root)
            preview = service.prepare_dmet_continuation(plan.study_id, work_dir=root)
            self.assertEqual(preview['max_runs'], 6)
            service.start_dmet_continuation(
                plan.study_id, preview['action_id'], work_dir=root
            )
            invocation = json.loads(
                (Path(root) / plan.study_id / 'study-invocation.json').read_text()
            )
            self.assertIsNone(invocation['error'], invocation['error'])
            action = json.loads(
                (
                    Path(root)
                    / plan.study_id
                    / 'dmet-continuations'
                    / (preview['action_id'] + '.json')
                ).read_text()
            )
            self.assertEqual(action['status'], 'completed')
            self.assertEqual(
                [s['status'] for s in action['steps']],
                [
                    'skipped',
                    'completed',
                    'completed',
                    'skipped',
                    'completed',
                    'completed',
                ],
            )
            self.assertEqual(len(remote.submissions), 5)
            # The second AF continuation uses the first new AF Run, not the old CDW seed result.
            self.assertEqual(
                action['steps'][2]['donor']['run_id'], action['steps'][1]['run_id']
            )
            # CDW comes from the historical independent high-V result, now superseded by AF in the task view.
            self.assertEqual(action['steps'][4]['donor']['run_id'], 'case-0003')
            after = service.load_report(plan.study_id, work_dir=root)
            analysis = after['dmet_phase_analysis']
            self.assertEqual(analysis['points'][0]['winner']['branch'], 'afm')
            self.assertEqual(analysis['points'][2]['winner']['branch'], 'cdw')
            self.assertEqual(len(analysis['points'][1]['energy_tied_runs']), 3)
            self.assertTrue(all(p['possible_hysteresis'] for p in analysis['points']))
            self.assertEqual(sum(len(p['candidates']) for p in analysis['points']), 7)
            self.assertEqual(len(before.cases), len(after['cases']))
            self.assertFalse(
                service.start_dmet_continuation(
                    plan.study_id, preview['action_id'], work_dir=root
                )['started']
            )
            for batch in remote.submissions[1:]:
                options = json.loads(batch[0].request)['solver']['options']
                self.assertEqual(options['initial_correlation_potential'], 'zero')
                self.assertEqual(options['correlation_potential_mixing'], 0.2)
                self.assertFalse(options['diis_enabled'])
            # An already-prepared review is invalidated by a real Study revision change.
            pending = service.prepare_dmet_continuation(plan.study_id, work_dir=root)
            state_path = Path(root) / plan.study_id / 'study-state.json'
            saved = json.loads(state_path.read_text())
            saved['revision_test'] = True
            state_path.write_text(json.dumps(saved))
            with self.assertRaisesRegex(ValueError, 'changed'):
                service.start_dmet_continuation(
                    plan.study_id, pending['action_id'], work_dir=root
                )

    def test_grouping_uses_hamiltonian_not_seed_or_mixing(self):
        a = record('a', 1, seed='af')
        b = record('b', 1, seed='cdw')
        b['request']['solver']['options']['correlation_potential_mixing'] = 0.1
        self.assertEqual(case_identity(a)['point_id'], case_identity(b)['point_id'])
        b['request']['model_hamiltonian']['spec']['sites'][0]['epsilon'] = 0.1
        self.assertNotEqual(case_identity(a)['point_id'], case_identity(b)['point_id'])

    def test_required_quality_excludes_candidate_but_density_fit_diagnostic_does_not(
        self,
    ):
        item = record('a', 0)
        checks = item['task_report']['structured_results']['quality_checks']
        checks.append(
            {
                'id': 'density_fit',
                'category': 'convergence',
                'source': 'dmet',
                'required': False,
                'operator': 'less_than',
                'observed': 0.3,
                'limit': 1e-4,
            }
        )
        self.assertTrue(candidates([item])[0]['qualified'])
        checks[-1]['required'] = True
        self.assertFalse(candidates([item])[0]['qualified'])
        checks.clear()
        self.assertFalse(candidates([item])[0]['qualified'])

    def test_interrupted_sweep_keeps_receipt_and_collects_without_resubmission(self):
        remote = BranchRemote()
        service = StudyApplicationService(
            task_executor=remote, execution_config={'execution_target': 'remote'}
        )
        service._study_launcher = inline_launcher(service)
        plan = plan_for_service(2)
        plan.cases = [
            StudyCase.from_dict(record(f'case-{i + 1:04d}', i)) for i in range(2)
        ]
        with tempfile.TemporaryDirectory() as root:
            service.run_study(plan, work_dir=root)
            action = service.prepare_dmet_continuation(plan.study_id, work_dir=root)
            remote.omit_report = True
            service.start_dmet_continuation(
                plan.study_id, action['action_id'], work_dir=root
            )
            ledger = (
                Path(root)
                / plan.study_id
                / 'dmet-continuations'
                / (action['action_id'] + '.json')
            )
            self.assertEqual(json.loads(ledger.read_text())['status'], 'interrupted')
            self.assertEqual(len(remote.submissions), 2)
            with self.assertRaisesRegex(ValueError, 'pending'):
                service.prepare_dmet_continuation(plan.study_id, work_dir=root)
            remote.omit_report = False
            report = service.collect_saved_study(plan.study_id, work_dir=root)
            self.assertEqual(len(remote.submissions), 2)
            self.assertEqual(len(report['cases'][1]['candidate_runs']), 1)
            self.assertFalse(
                service.start_dmet_continuation(
                    plan.study_id, action['action_id'], work_dir=root
                )['started']
            )

    def test_whole_sweep_cost_is_reviewed_before_any_submission(self):
        from computational_study_agent.costing import CostApprovalRequired

        remote = BranchRemote()
        service = StudyApplicationService(
            task_executor=remote, execution_config={'execution_target': 'remote'}
        )
        service._study_launcher = inline_launcher(service)
        plan = plan_for_service(2)
        plan.cases = [
            StudyCase.from_dict(record(f'case-{i + 1:04d}', i)) for i in range(2)
        ]
        for case in plan.cases:
            case.request['solver']['options'].update(
                impurity_solver='block2_dmrg',
                impurity_solver_options={'bond_dimensions': [4], 'sweeps': 2},
            )
        with tempfile.TemporaryDirectory() as root:
            service.run_study(plan, work_dir=root)
            path = Path(root) / plan.study_id / 'study-plan.json'
            saved = json.loads(path.read_text())
            saved['resource_policy'].update(
                review_work_estimates=True,
                total_work_review_threshold=1,
                approved=False,
            )
            path.write_text(json.dumps(saved))
            action = service.prepare_dmet_continuation(plan.study_id, work_dir=root)
            self.assertFalse(action['cost_estimate']['can_execute'])
            with self.assertRaises(CostApprovalRequired):
                service.start_dmet_continuation(
                    plan.study_id, action['action_id'], work_dir=root
                )
            self.assertEqual(len(remote.submissions), 1)
            service.start_dmet_continuation(
                plan.study_id, action['action_id'], work_dir=root, approve_cost=True
            )
            self.assertEqual(len(remote.submissions), 3)
