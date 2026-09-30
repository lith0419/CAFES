from __future__ import annotations

import unittest
import copy
import json

from computational_study_agent import StudyPlan, StudyReport, StudySpec
from pyscf_agent import (
    TaskReport,
    TaskSpec,
)
from pyscf_agent.contracts import task_spec_from_dict, task_spec_to_dict
from pyscf_agent.schema_contracts import (
    ADAPTIVE_STUDY_REPORT_SCHEMA,
    PUBLIC_CONTRACT_VERSION,
    TASK_REPORT_SCHEMA,
    TASK_SPEC_SCHEMA,
    STUDY_REPORT_SCHEMA,
    public_schema_ids,
    public_schema_manifest,
    validate_public_payload,
)
from pyscf_agent.executors import JOB_HANDLE_SCHEMA, JobHandle


class PublicContractTests(unittest.TestCase):
    def test_task_spec_is_versioned_and_round_trips(self):
        original = TaskSpec()
        payload = task_spec_to_dict(original)

        self.assertEqual(payload['schema'], TASK_SPEC_SCHEMA)
        self.assertNotIn('solver', payload)
        self.assertEqual(task_spec_to_dict(task_spec_from_dict(payload)), payload)
        self.assertEqual(validate_public_payload(payload), TASK_SPEC_SCHEMA)

    def test_task_spec_preserves_unspecified_dmet_options(self):
        payload = task_spec_to_dict(task_spec_from_dict({
            'task_type': 'model_hamiltonian',
            'solver': {'name': 'dmet'},
        }))

        self.assertEqual(payload['solver'], {'name': 'dmet', 'options': {}})

    def test_task_and_study_reports_round_trip_additive_fields(self):
        task_payload = TaskReport().to_dict()
        task_payload['future_additive_field'] = {'preserved_by_transport': True}
        restored_task = TaskReport.from_dict(task_payload)

        study_payload = StudyReport(
            study_id='study-1',
            name='contract-test',
            objective='validate contracts',
            system_type='molecular',
            status='succeeded',
            work_dir='/tmp/study',
            cases=[],
            comparison_table=[],
        ).to_dict()
        study_payload['future_additive_field'] = {'evidence': [1, {'value': None}]}
        restored_study = StudyReport.from_dict(study_payload)

        self.assertEqual(restored_task.to_dict()['schema'], TASK_REPORT_SCHEMA)
        self.assertEqual(restored_study.to_dict(), study_payload)

    @staticmethod
    def _extended_study_payload():
        return StudyReport(
            study_id='study-extensions', name='extensions', objective='preserve evidence',
            system_type='molecular', status='succeeded', work_dir='/tmp/study',
            cases=[{'case_id': 'case-0001', 'attempt_count': 2,
                    'execution': {'run_id': 'retry-2', 'pending': False}}],
            comparison_table=[{'case_id': 'case-0001', 'status': 'succeeded'}],
            adaptive={'decision_log': [{'case_id': 'case-0001', 'method': 'ccsd'}],
                      'recovery_plan': None, 'workflow': {'stage': 'completed'}},
            postprocessing={'status': 'succeeded', 'artifacts': [{'path': 'energy.png'}]},
        ).to_dict()

    def test_study_report_preserves_extensions_for_both_existing_schemas(self):
        for schema in (STUDY_REPORT_SCHEMA, ADAPTIVE_STUDY_REPORT_SCHEMA):
            with self.subTest(schema=schema):
                payload = self._extended_study_payload()
                payload['schema'] = schema
                payload['review_context'] = {'selected_case_ids': ['case-0001']}
                payload['dataset_generation'] = {'status': 'succeeded', 'path': '/tmp/dataset'}
                restored = StudyReport.from_dict(json.loads(json.dumps(payload)))
                self.assertEqual(restored.adaptive, payload['adaptive'])
                self.assertEqual(restored.postprocessing, payload['postprocessing'])
                self.assertEqual(restored.to_dict(), payload)
                self.assertEqual(StudyReport.from_dict(restored.to_dict()).to_dict(), payload)
                self.assertEqual(validate_public_payload(restored.to_dict()), schema)

    def test_study_report_extensions_are_independent_copies(self):
        payload = self._extended_study_payload()
        payload['future_metadata'] = {'values': [1]}
        original = copy.deepcopy(payload)
        restored = StudyReport.from_dict(payload)
        payload['adaptive']['decision_log'].clear()
        payload['postprocessing']['artifacts'].clear()
        payload['future_metadata']['values'].append(2)
        self.assertEqual(restored.to_dict(), original)

        exported = restored.to_dict()
        exported['adaptive']['decision_log'].clear()
        exported['postprocessing']['artifacts'].clear()
        exported['future_metadata']['values'].append(3)
        self.assertEqual(restored.to_dict(), original)

        restored.status = 'completed_with_issues'
        restored.adaptive['workflow']['stage'] = 'review_required'
        self.assertEqual(restored.to_dict()['status'], 'completed_with_issues')
        self.assertEqual(restored.to_dict()['adaptive']['workflow']['stage'], 'review_required')
        self.assertEqual(restored.to_dict()['future_metadata'], original['future_metadata'])

    def test_study_report_reads_legacy_schema_less_and_optional_extensions(self):
        for value in (None, {}):
            with self.subTest(value=value):
                payload = self._extended_study_payload()
                payload.pop('schema')
                payload['adaptive'] = copy.deepcopy(value)
                payload['postprocessing'] = copy.deepcopy(value)
                result = StudyReport.from_dict(payload).to_dict()
                self.assertEqual(result, {**payload, 'schema': STUDY_REPORT_SCHEMA})
        payload.pop('adaptive')
        payload.pop('postprocessing')
        restored = StudyReport.from_dict(payload)
        self.assertIsNone(restored.adaptive)
        self.assertIsNone(restored.postprocessing)

    def test_study_report_rejects_incompatible_schemas_and_invalid_extensions(self):
        for schema in ('pyscf-agent.study-report.v2', 'pyscf-agent.adaptive-study-report.v2', TASK_REPORT_SCHEMA):
            payload = self._extended_study_payload()
            payload['schema'] = schema
            with self.subTest(schema=schema), self.assertRaisesRegex(ValueError, 'Unsupported StudyReport schema'):
                StudyReport.from_dict(payload)
        for field in ('adaptive', 'postprocessing'):
            for value in ([], 'invalid', False):
                payload = self._extended_study_payload()
                payload[field] = value
                with self.subTest(field=field, value=value), self.assertRaisesRegex(TypeError, 'object or null'):
                    StudyReport.from_dict(payload)

    def test_adaptive_report_requires_its_extension_and_base_envelope(self):
        payload = self._extended_study_payload()
        payload['schema'] = ADAPTIVE_STUDY_REPORT_SCHEMA
        del payload['adaptive']
        with self.assertRaisesRegex(ValueError, 'missing required field'):
            StudyReport.from_dict(payload)
        payload['adaptive'] = None
        with self.assertRaisesRegex(ValueError, 'requires an adaptive object'):
            StudyReport.from_dict(payload)
        payload['adaptive'] = {}
        del payload['cases']
        with self.assertRaisesRegex(ValueError, 'missing required field'):
            StudyReport.from_dict(payload)

    def test_unknown_field_named_like_internal_storage_round_trips_safely(self):
        payload = self._extended_study_payload()
        self.assertNotIn('_extra_fields', payload)
        payload['_extra_fields'] = {'status': 'must not replace report status'}
        payload['future_metadata'] = {'schema': 'opaque-evidence.v1'}
        restored = StudyReport.from_dict(payload)
        self.assertEqual(restored.to_dict(), payload)
        self.assertEqual(restored.status, 'succeeded')

    def test_study_spec_and_plan_reject_incompatible_schema_versions(self):
        spec_payload = StudySpec(
            name='study',
            objective='test',
            system_type='molecular',
        ).to_dict()
        plan_payload = StudyPlan(
            study_id='study-1',
            name='study',
            objective='test',
            system_type='molecular',
            cases=[],
            observables=['energy'],
        ).to_dict()

        self.assertEqual(StudySpec.from_dict(spec_payload).to_dict(), spec_payload)
        self.assertEqual(StudyPlan.from_dict(plan_payload).to_dict(), plan_payload)
        spec_payload['schema'] = 'pyscf-agent.study-spec.v2'
        plan_payload['schema'] = 'pyscf-agent.study-plan.v2'
        with self.assertRaisesRegex(ValueError, 'Expected schema'):
            StudySpec.from_dict(spec_payload)
        with self.assertRaisesRegex(ValueError, 'Expected schema'):
            StudyPlan.from_dict(plan_payload)

    def test_manifest_covers_executor_and_scientific_contracts(self):
        manifest = public_schema_manifest()
        schemas = {item['schema'] for item in manifest['schemas']}

        self.assertEqual(manifest['contract_version'], PUBLIC_CONTRACT_VERSION)
        self.assertEqual(validate_public_payload(manifest), manifest['schema'])
        self.assertEqual(schemas, set(public_schema_ids()))
        self.assertIn(TASK_SPEC_SCHEMA, schemas)
        self.assertIn(TASK_REPORT_SCHEMA, schemas)
        self.assertIn(ADAPTIVE_STUDY_REPORT_SCHEMA, schemas)
        self.assertIn(JOB_HANDLE_SCHEMA, schemas)

        handle = JobHandle(
            job_id='job-1',
            executor_id='local',
            run_id='run-1',
            work_dir='/tmp/run-1',
            submitted_at='2026-08-10T00:00:00+00:00',
        )
        self.assertEqual(validate_public_payload(handle.to_dict()), JOB_HANDLE_SCHEMA)

    def test_missing_required_field_is_rejected(self):
        payload = task_spec_to_dict(TaskSpec())
        del payload['method']

        with self.assertRaisesRegex(ValueError, 'missing required field'):
            validate_public_payload(payload)


if __name__ == '__main__':
    unittest.main()
