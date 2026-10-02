"""Source-checkout examples load through the ordinary saved-Study API."""

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from computational_study_agent.application import StudyApplicationService
from computational_study_agent.application import example_studies
from computational_study_agent.web_api import handle_example_study_request


class ExampleStudiesTests(unittest.TestCase):
    def test_web_import_opens_all_examples_and_resolves_saved_figures(self):
        service = StudyApplicationService()
        with tempfile.TemporaryDirectory() as folder, patch(
                'computational_study_agent.web_api.get_study_application_service', return_value=service):
            status, _, body = handle_example_study_request(b'{}', action='list')
            self.assertEqual(status, 200)
            examples = json.loads(body)['examples']
            self.assertEqual(len(examples), 4)
            for item in examples:
                with self.subTest(study=item['study_id']):
                    status, _, body = handle_example_study_request(json.dumps({
                        'study_id': item['study_id'], 'work_dir': folder,
                    }).encode(), action='import')
                    self.assertEqual(status, 200, body)
                    imported = json.loads(body)
                    self.assertEqual(imported['execution_target'], 'local')
                    saved = service.open_study(imported['study_id'], work_dir=imported['work_dir'])
                    report = saved['report']
                    self.assertEqual(len(report['comparison_table']), item['cases'])
                    self.assertEqual(Path(report['work_dir']), Path(folder).resolve() / item['study_id'])
                    self.assertTrue((Path(report['work_dir']) / 'study-plan.json').is_file())
                    self.assertFalse((Path(report['work_dir']) / 'study-plan.template.json').exists())
                    plots = [a for a in report['artifacts'] if a['kind'] == 'postprocess-plot']
                    self.assertTrue(plots)
                    for plot in plots:
                        self.assertTrue(service.read_artifact(report, plot['path']).content.startswith(b'\x89PNG'))
                    if report['grid_refinement']:
                        self.assertTrue(report['grid_refinement']['cells'])

    def test_import_preserves_local_edits_and_refuses_unrelated_directories(self):
        service = StudyApplicationService()
        study_id = service.list_examples()[0]['study_id']
        with tempfile.TemporaryDirectory() as folder:
            destination = Path(folder) / study_id
            destination.mkdir()
            with self.assertRaisesRegex(ValueError, 'unrelated Study'):
                service.import_example(study_id, work_dir=folder)
            destination.rmdir()
            service.import_example(study_id, work_dir=folder)
            report_path = destination / 'study-report.json'
            report = json.loads(report_path.read_text())
            report['summary'] = 'User annotation'
            report_path.write_text(json.dumps(report))
            service.import_example(study_id, work_dir=folder)
            self.assertEqual(json.loads(report_path.read_text())['summary'], 'User annotation')
            with self.assertRaisesRegex(ValueError, 'Unknown bundled example'):
                service.import_example('../outside', work_dir=folder)

    def test_wheel_without_source_examples_returns_an_empty_catalog(self):
        with tempfile.TemporaryDirectory() as folder, patch.object(
                example_studies, 'EXAMPLES_ROOT', Path(folder)):
            self.assertEqual(StudyApplicationService().list_examples(), [])
