from __future__ import annotations

import unittest

from pyscf_agent.lifecycle import (
    LifecycleTransitionError,
    begin_study_execution,
    complete_study_execution,
    new_lifecycle,
    prepare_task_for_execution,
    task_lifecycle_for_preparation,
    task_lifecycle_from_job_status,
    synchronize_study_report_lifecycle,
    transition_lifecycle,
)


class LifecycleTests(unittest.TestCase):
    def test_task_happy_path_records_ordered_events(self):
        lifecycle = task_lifecycle_for_preparation('ready', entity_id='task-1')
        lifecycle = prepare_task_for_execution(lifecycle)
        lifecycle = transition_lifecycle(lifecycle, 'execution_succeeded')

        self.assertEqual(lifecycle['stage'], 'succeeded')
        self.assertTrue(lifecycle['terminal'])
        self.assertEqual(
            [item['event'] for item in lifecycle['history']],
            [
                'created',
                'validation_passed',
                'request_prepared',
                'execution_submitted',
                'execution_started',
                'execution_succeeded',
            ],
        )

    def test_configuration_change_creates_new_revision_without_erasing_history(self):
        lifecycle = task_lifecycle_for_preparation('ready', entity_id='task-1')
        lifecycle = transition_lifecycle(
            lifecycle,
            'configuration_changed',
            details={'changed_fields': ['method']},
        )

        self.assertEqual(lifecycle['stage'], 'draft')
        self.assertEqual(lifecycle['revision'], 2)
        self.assertEqual(lifecycle['history'][-1]['details']['changed_fields'], ['method'])
        self.assertEqual(len(lifecycle['history']), 4)

    def test_unconverged_task_can_enter_review_and_retry(self):
        lifecycle = prepare_task_for_execution(new_lifecycle('task', 'task-1'))
        lifecycle = transition_lifecycle(lifecycle, 'execution_unconverged')
        lifecycle = transition_lifecycle(lifecycle, 'review_requested')
        lifecycle = transition_lifecycle(lifecycle, 'approval_granted')
        lifecycle = prepare_task_for_execution(lifecycle)

        self.assertEqual(lifecycle['stage'], 'running')
        self.assertEqual(lifecycle['current_event'], 'execution_started')

    def test_repeated_job_status_polling_is_idempotent(self):
        lifecycle = task_lifecycle_for_preparation('ready', entity_id='task-1')
        lifecycle = task_lifecycle_from_job_status(lifecycle, 'queued')
        lifecycle = task_lifecycle_from_job_status(lifecycle, 'running')
        repeated = task_lifecycle_from_job_status(lifecycle, 'running')
        completed = task_lifecycle_from_job_status(
            repeated,
            'completed',
            task_status='succeeded',
        )

        self.assertEqual(repeated, lifecycle)
        self.assertEqual(completed['stage'], 'succeeded')

    def test_prepared_task_can_be_blocked_by_revalidation(self):
        lifecycle = task_lifecycle_for_preparation('ready', entity_id='task-1')
        lifecycle = task_lifecycle_for_preparation('blocked', lifecycle=lifecycle)

        self.assertEqual(lifecycle['stage'], 'blocked')
        self.assertEqual(lifecycle['history'][-1]['event'], 'validation_failed')

    def test_blocked_job_status_requires_review(self):
        lifecycle = task_lifecycle_from_job_status(
            task_lifecycle_for_preparation('ready', entity_id='task-1'),
            'completed',
            task_status='blocked',
        )

        self.assertEqual(lifecycle['stage'], 'review_required')
        self.assertEqual(
            [item['event'] for item in lifecycle['history']][-2:],
            ['execution_failed', 'review_requested'],
        )

    def test_study_review_refinement_and_analysis_path(self):
        lifecycle = new_lifecycle('study', 'study-1')
        lifecycle = transition_lifecycle(lifecycle, 'execution_started')
        lifecycle = transition_lifecycle(lifecycle, 'execution_completed')
        lifecycle = transition_lifecycle(lifecycle, 'review_requested')
        lifecycle = transition_lifecycle(lifecycle, 'approval_granted')
        lifecycle = transition_lifecycle(lifecycle, 'refinement_started')
        lifecycle = transition_lifecycle(lifecycle, 'execution_completed')
        lifecycle = transition_lifecycle(lifecycle, 'analysis_started')
        lifecycle = transition_lifecycle(lifecycle, 'analysis_completed')

        self.assertEqual(lifecycle['stage'], 'completed')
        self.assertTrue(lifecycle['terminal'])

    def test_study_helpers_preserve_one_history_across_review_and_rerun(self):
        lifecycle = begin_study_execution(None, entity_id='study-1')
        lifecycle = complete_study_execution(lifecycle)
        lifecycle = synchronize_study_report_lifecycle(
            lifecycle,
            decisions=[{'status': 'review_required'}],
        )
        lifecycle = transition_lifecycle(lifecycle, 'approval_granted')
        lifecycle = transition_lifecycle(lifecycle, 'refinement_started')
        lifecycle = complete_study_execution(lifecycle)

        self.assertEqual(lifecycle['stage'], 'completed')
        self.assertEqual(
            [item['event'] for item in lifecycle['history']],
            [
                'created',
                'execution_started',
                'execution_completed',
                'review_requested',
                'approval_granted',
                'refinement_started',
                'execution_completed',
            ],
        )

    def test_illegal_transition_is_rejected(self):
        with self.assertRaises(LifecycleTransitionError):
            transition_lifecycle(new_lifecycle('task', 'task-1'), 'execution_started')


if __name__ == '__main__':
    unittest.main()
