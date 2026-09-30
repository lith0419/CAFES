from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from computational_study_agent.application import StudyApplicationService
from computational_study_agent.execution_receipts import StudyExecutionInterrupted
from pyscf_agent.schema_contracts import ADAPTIVE_STUDY_REPORT_SCHEMA
from tests.computational_study_agent.test_retry_collection import Remote, plan_for_service


class ResultExecutor(Remote):
    def collect_independent_tasks(self, batches):
        result = super().collect_independent_tasks(batches)
        for report in result.reports.values():
            report['structured_results']['final_energy'] = -float(self.submit_count)
        return result


class StudyTaskRunTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.plan = plan_for_service()
        self.executor = ResultExecutor(fail_first=False)
        self.service = StudyApplicationService(task_executor=self.executor)
        self.study_dir = self.root / self.plan.study_id

    def initial(self):
        return self.service.run_study(self.plan, work_dir=str(self.root))

    def subset(self, index=0, *, kind='static'):
        plan = self.plan.to_dict()
        plan['cases'] = [plan['cases'][index]]
        plan['_review_kind'] = kind
        return plan

    def retry(self, index=0, **kwargs):
        return self.service.run_study(self.subset(index), work_dir=str(self.root), resume=False, **kwargs)

    def collect(self):
        return StudyApplicationService(task_executor=self.executor).collect_study(
            self.plan.study_id, work_dir=str(self.root),
        )

    def persisted(self, name='study-report.json'):
        return json.loads((self.study_dir / name).read_text())

    def test_subset_retry_keeps_study_identity_and_only_replaces_selected_run(self):
        first = self.initial()
        old = first.cases[0]['task_report']
        old_path = Path(old['work_dir']) / old['run_id'] / 'task-report.json'
        old_bytes = old_path.read_bytes()
        second = self.retry()
        self.assertEqual(first.study_id, second.study_id)
        self.assertEqual([p.name for p in self.root.iterdir()], [self.plan.study_id])
        self.assertEqual([task.task_id for task in self.executor.submissions[-1]], ['case-0001'])
        self.assertNotEqual(first.cases[0]['execution']['run_id'], second.cases[0]['execution']['run_id'])
        self.assertEqual(first.cases[1], second.cases[1])
        self.assertEqual([case['attempt_count'] for case in second.cases], [2, 1])
        self.assertEqual(old_bytes, old_path.read_bytes())
        state = self.persisted('study-state.json')
        self.assertNotIn('parent_study', state)
        self.assertTrue(all('task_reference' not in case for case in second.cases))
        self.assertEqual(len(self.persisted('study-plan.json')['cases']), 2)
        self.assertEqual(len(second.comparison_table), 2)

    def test_two_subsets_share_task_counters_and_keep_each_others_results(self):
        self.initial()
        first = self.retry(0)
        second = self.retry(1)
        self.assertEqual(second.cases[0], first.cases[0])
        self.assertEqual([case['attempt_count'] for case in second.cases], [2, 2])
        self.assertEqual(second.status, 'succeeded')
        self.assertEqual(self.executor.submit_count, 3)

    def test_review_action_uses_study_identity_for_repeated_preparations(self):
        report = self.initial().to_dict()
        for action in ('increase_recovery_max_cycle', 'try_fci_recovery'):
            result = self.service.apply_review_action(
                action, study_state=report, plan=self.plan.to_dict(),
                case_ids=['case-0001'], plan_kind='static',
            )
            self.assertEqual(result['plan']['study_id'], self.plan.study_id)
            self.assertEqual([case['case_id'] for case in result['plan']['cases']], ['case-0001'])
            self.assertNotIn('_adaptive_parent_study_id', result['plan'])

    def test_changed_review_request_is_saved_and_collect_recovers_by_study_id(self):
        self.initial()
        subset = self.subset()
        subset['cases'][0]['request']['runtime'] = {'max_cycle': 400}
        self.executor.wait_count = 0
        with self.assertRaises(StudyExecutionInterrupted):
            self.service.run_study(subset, work_dir=str(self.root), resume=False)
        saved = self.persisted('study-plan.json')
        self.assertEqual(len(saved['cases']), 2)
        self.assertEqual(saved['cases'][0]['request']['runtime']['max_cycle'], 400)
        report = self.collect()
        self.assertEqual(report.status, 'succeeded')
        self.assertEqual([case['attempt_count'] for case in report.cases], [2, 1])
        self.assertEqual(self.executor.submit_count, 2)

    def test_pending_retry_never_leaves_a_successful_study_report(self):
        first = self.initial()
        self.executor.wait_count = 0
        with self.assertRaises(StudyExecutionInterrupted):
            self.retry()
        self.assertEqual(self.persisted()['status'], 'completed_with_issues')
        pending = self.persisted('study-state.json')['cases']['case-0001']['execution']
        self.assertTrue(pending['pending'])
        current = self.collect()
        self.assertNotEqual(current.cases[0]['task_report']['run_id'], first.cases[0]['task_report']['run_id'])
        self.assertEqual(current.status, 'succeeded')
        self.assertEqual(self.executor.submit_count, 2)

    def test_run_recovers_other_pending_tasks_before_starting_another_retry(self):
        self.initial()
        self.executor.wait_count = 0
        with self.assertRaises(StudyExecutionInterrupted):
            self.retry(1)
        report = self.retry(0)
        self.assertEqual(report.status, 'succeeded')
        self.assertEqual([case['attempt_count'] for case in report.cases], [1, 2])
        self.assertEqual(self.executor.submit_count, 2)
        report = self.retry(0)
        self.assertEqual([case['attempt_count'] for case in report.cases], [2, 2])
        self.assertEqual(self.executor.submit_count, 3)

    def test_collect_rebuilds_cached_rows_from_current_checkpoint_results(self):
        self.initial()
        saved = self.persisted()
        saved['cases'][0]['task_report']['run_id'] = 'obsolete'
        saved['cases'][0]['task_report']['structured_results']['final_energy'] = -999
        saved['comparison_table'][1]['final_energy'] = '-999 Ha'
        saved['cases'].append({'case_id': 'cached-extra', 'task_report': {'execution_status': 'succeeded'}})
        saved['comparison_table'].append({'case_id': 'cached-extra', 'status': 'succeeded'})
        saved['status'] = 'failed'
        (self.study_dir / 'study-report.json').write_text(json.dumps(saved))
        report = self.collect()
        self.assertEqual(len(report.cases), 2)
        self.assertEqual(report.status, 'succeeded')
        self.assertEqual([row['final_energy'] for row in report.comparison_table], ['-1.0 Ha', '-1.0 Ha'])
        self.assertEqual(self.executor.submit_count, 1)

    def test_failed_task_and_missing_rows_are_reflected_in_full_report(self):
        self.initial()
        state = self.persisted('study-state.json')
        report_path = Path(state['cases']['case-0001']['task_report_ref']['path'])
        task = json.loads(report_path.read_text())
        task['execution_status'] = 'failed'
        report_path.write_text(json.dumps(task))
        state['cases']['case-0001']['task_summary']['execution_status'] = 'failed'
        (self.study_dir / 'study-state.json').write_text(json.dumps(state))
        saved = self.persisted()
        saved['comparison_table'] = []
        (self.study_dir / 'study-report.json').write_text(json.dumps(saved))
        report = self.collect()
        self.assertEqual(report.status, 'completed_with_issues')
        self.assertEqual([row['status'] for row in report.comparison_table], ['failed', 'succeeded'])
        self.assertEqual(self.executor.submit_count, 1)

    def test_unselected_comparison_annotations_and_report_extensions_survive(self):
        self.plan.observables = ['energy', 'energy_per_site']
        self.initial()
        saved = self.persisted()
        saved['comparison_table'][1]['path_branch'] = 'forward'
        saved['adaptive'] = {'decision_log': [{'stage': 'initial'}]}
        saved['postprocessing'] = {'status': 'skipped', 'reason': 'preserved evidence'}
        saved['custom_analysis'] = {'note': 'keep'}
        (self.study_dir / 'study-report.json').write_text(json.dumps(saved))
        subset = self.subset()
        subset['observables'] = ['energy']
        after = self.service.run_study(subset, work_dir=str(self.root), resume=False).to_dict()
        self.assertEqual(after['comparison_table'][1], saved['comparison_table'][1])
        for key in ('adaptive', 'postprocessing', 'custom_analysis'):
            self.assertEqual(after[key], saved[key])
        self.assertIn('energy_per_site', self.persisted('study-plan.json')['observables'])
        self.assertEqual(len((self.study_dir / 'comparison-table.tsv').read_text().splitlines()), 3)

    def test_adaptive_review_updates_the_same_study_and_keeps_scientific_stage_evidence(self):
        first = self.initial().to_dict()
        first['schema'] = ADAPTIVE_STUDY_REPORT_SCHEMA
        first['adaptive'] = {'mode': 'adaptive_scan', 'decision_log': [{'stage': 'initial'}]}
        path = self.study_dir / 'adaptive-study-report.json'
        path.write_text(json.dumps(first))
        report = self.service.run_study(
            self.subset(kind='recovery'), work_dir=str(self.root), resume=False,
        ).to_dict()
        self.assertEqual(report['study_id'], self.plan.study_id)
        self.assertEqual(report['adaptive']['recovery_report']['study_id'], self.plan.study_id)
        self.assertEqual(report['adaptive']['decision_log'], first['adaptive']['decision_log'])
        self.assertEqual(json.loads(path.read_text())['cases'], report['cases'])
        self.assertEqual(len(report['cases']), 2)

    def test_direct_review_scaffold_starts_tasks_without_a_child_study(self):
        scaffold = {
            'study_id': self.plan.study_id, 'system_type': self.plan.system_type,
            'status': 'pending_review', 'cases': [], 'comparison_table': [],
            'adaptive': {'mode': 'direct_casscf_review'},
        }
        payload = {**self.plan.to_dict(), '_review_kind': 'direct_casscf_review'}
        report = self.service.run_study(payload, work_dir=str(self.root), study_report=scaffold)
        self.assertEqual(report.study_id, self.plan.study_id)
        self.assertEqual(report.status, 'succeeded')
        self.assertEqual(len(report.cases), 2)
        self.assertEqual([path.name for path in self.root.iterdir()], [self.plan.study_id])

    def test_unknown_review_case_and_wrong_study_are_rejected_before_submission(self):
        first = self.initial().to_dict()
        subset = self.subset()
        subset['cases'][0]['case_id'] = 'outside'
        with self.assertRaisesRegex(ValueError, 'outside the Study'):
            self.service.run_study(subset, work_dir=str(self.root))
        subset = self.subset()
        subset['study_id'] = 'other-study'
        with self.assertRaisesRegex(ValueError, 'same Study'):
            self.service.run_study(subset, work_dir=str(self.root), study_report=first)
        self.assertEqual(self.executor.submit_count, 1)
        self.assertEqual(self.persisted()['status'], 'succeeded')

    def test_stale_browser_report_cannot_restore_old_results(self):
        stale = self.initial().to_dict()
        self.retry(0)
        result = self.retry(1, study_report=stale)
        self.assertEqual(result.cases[0]['task_report']['structured_results']['final_energy'], -2.0)
        self.assertEqual([case['attempt_count'] for case in result.cases], [2, 2])

    def test_explicit_subset_does_not_restore_an_old_unselected_request(self):
        self.initial()
        subset = self.subset()
        subset['cases'][0]['request']['runtime'] = {'max_cycle': 400}
        changed = self.service.run_study(subset, work_dir=str(self.root), resume=False)
        report = self.service.run_study(self.plan, work_dir=str(self.root), rerun_case_ids=['case-0002'])
        saved = self.persisted('study-plan.json')
        self.assertEqual(saved['cases'][0]['request']['runtime']['max_cycle'], 400)
        self.assertEqual(report.cases[0]['task_report'], changed.cases[0]['task_report'])
        self.assertEqual(self.collect().status, 'succeeded')

    def test_study_report_is_written_after_all_current_task_reports(self):
        self.initial()
        report = self.retry()
        report_time = (self.study_dir / 'study-report.json').stat().st_mtime_ns
        for case in report.cases:
            task = case['task_report']
            path = Path(task['work_dir']) / task['run_id'] / 'task-report.json'
            self.assertGreaterEqual(report_time, path.stat().st_mtime_ns)

    def test_web_collect_needs_only_study_id_after_interruption(self):
        from computational_study_agent.web_api import handle_study_execution_collect_request
        self.initial()
        self.executor.wait_count = 0
        with self.assertRaises(StudyExecutionInterrupted):
            self.retry()
        with patch('computational_study_agent.web_api.get_study_application_service', return_value=self.service):
            status, _, body = handle_study_execution_collect_request(json.dumps({
                'study_id': self.plan.study_id, 'work_dir': str(self.root),
            }).encode())
        self.assertEqual(status.value, 200, body)
        report = json.loads(body)['report']
        self.assertEqual(report['study_id'], self.plan.study_id)
        self.assertEqual(len(report['cases']), 2)
        self.assertEqual(self.executor.submit_count, 2)

    def test_web_run_returns_full_report_without_a_second_merge(self):
        from computational_study_agent.web_api import handle_study_run_request
        self.initial()
        with patch('computational_study_agent.web_api.get_study_application_service', return_value=self.service):
            status, _, body = handle_study_run_request(json.dumps({
                'plan': self.subset(), 'work_dir': str(self.root), 'resume': False,
            }).encode())
        self.assertEqual(status.value, 200, body)
        report = json.loads(body)['report']
        self.assertEqual(report['study_id'], self.plan.study_id)
        self.assertEqual([case['attempt_count'] for case in report['cases']], [2, 1])

    def test_completed_legacy_child_results_are_imported_without_resubmission(self):
        initial = self.initial().to_dict()
        legacy = copy.deepcopy(self.plan)
        legacy.study_id = 'old-review-execution'
        legacy.cases[0].request['runtime'] = {'max_cycle': 400}
        old = self.service.run_study(legacy, work_dir=str(self.root))
        old_path = Path(old.execution['checkpoint'])
        old_state = json.loads(old_path.read_text())
        old_state['parent_study'] = {'study_id': self.plan.study_id, 'report_path': str(self.study_dir / 'study-report.json')}
        old_path.write_text(json.dumps(old_state))
        initial['cases'][0]['task_reference'] = {'study_id': legacy.study_id, 'checkpoint': str(old_path)}
        (self.study_dir / 'study-report.json').write_text(json.dumps(initial))
        report = self.collect()
        self.assertEqual(self.persisted('study-plan.json')['cases'][0]['request']['runtime']['max_cycle'], 400)
        self.assertEqual(report.cases[0]['task_report']['structured_results']['final_energy'], -2.0)
        self.assertNotIn('task_reference', report.cases[0])
        self.assertEqual(self.executor.submit_count, 2)
        retry = self.retry()
        self.assertEqual(retry.cases[0]['attempt_count'], 2)
        self.assertEqual(retry.cases[0]['task_report']['work_dir'], str(self.study_dir / 'cases'))
        self.assertTrue(old_path.exists())

    def test_adaptive_report_can_start_same_study_tasks_from_its_completed_stage(self):
        completed = self.initial().to_dict()
        study_id = 'adaptive-scan'
        study_dir = self.root / study_id
        study_dir.mkdir()
        completed.update(study_id=study_id, work_dir=str(study_dir), schema=ADAPTIVE_STUDY_REPORT_SCHEMA,
                         adaptive={'mode': 'adaptive_scan', 'decision_log': [{'stage': 'initial'}]})
        (study_dir / 'adaptive-study-report.json').write_text(json.dumps(completed))
        subset = self.subset(kind='recovery')
        subset['study_id'] = study_id
        report = self.service.run_study(subset, work_dir=str(self.root), resume=False)
        self.assertEqual(report.study_id, study_id)
        self.assertEqual([case['attempt_count'] for case in report.cases], [2, 1])
        self.assertEqual(report.cases[1]['task_report'], completed['cases'][1]['task_report'])
        collected = self.service.collect_study(study_id, work_dir=str(self.root))
        self.assertEqual(collected.status, 'succeeded')
        self.assertEqual(self.executor.submit_count, 2)

    def test_partial_sequential_collection_preserves_unstarted_tasks(self):
        self.executor.wait_count = 0
        with self.assertRaises(StudyExecutionInterrupted):
            self.service.run_study(self.plan, work_dir=str(self.root), batch_independent=False)
        report = self.collect()
        self.assertEqual([row['status'] for row in report.comparison_table], ['succeeded', 'not_executed'])
        self.assertEqual(self.executor.submit_count, 1)
        self.assertEqual(self.initial().status, 'succeeded')
        self.assertEqual(self.executor.submit_count, 2)


if __name__ == '__main__':
    unittest.main()
