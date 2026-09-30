from __future__ import annotations

import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from computational_study_agent.application import StudyApplicationService
from pyscf_agent.paths import default_runs_root, resolve_work_dir, study_search_roots
from pyscf_agent.executors import JobHandle, LocalExecutor
from pyscf_agent.web.api import handle_task_view_request

SPEC = {
    'name': 'path migration',
    'system_type': 'molecular',
    'base_task': {'atom': 'H 0 0 0; H 0 0 0.74', 'basis': 'sto-3g', 'method': 'hf'},
    'sweep': {'basis': ['sto-3g']},
    'observables': ['energy'],
}


class RuntimePathTests(unittest.TestCase):
    def test_default_task_monitor_can_read_legacy_root_but_custom_root_stays_confined(
        self,
    ):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {}, clear=True),
        ):
            checkout = Path(tmp).resolve() / 'checkout'
            checkout.mkdir()
            handle = JobHandle(
                'job',
                'local',
                'job',
                str(checkout / 'runs'),
                '2026-09-25T00:00:00+00:00',
            )
            payload = json.dumps({'handle': handle.to_dict()}).encode()
            with (
                patch('pathlib.Path.cwd', return_value=checkout),
                patch.dict(os.environ, {'XDG_DATA_HOME': tmp}),
            ):
                with patch('pyscf_agent.web.api._execution_service') as service:
                    service.return_value.inspect_task.return_value = {'run_id': 'job'}
                    code, headers, _ = handle_task_view_request(
                        payload, task_executor=LocalExecutor()
                    )
                    self.assertEqual(code, 200)
                    self.assertEqual(headers['Cache-Control'], 'no-store')
                    code, _, _ = handle_task_view_request(
                        payload,
                        task_executor=LocalExecutor(),
                        work_dir=str(checkout / 'private'),
                    )
                    self.assertEqual(code, 400)
                    service.return_value.inspect_task.assert_called_once()

    def test_default_and_overrides_do_not_create_directories(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {}, clear=True),
        ):
            with patch('pathlib.Path.home', return_value=Path(tmp)):
                expected = Path(tmp) / '.local/share/pyscf-agent/runs'
                self.assertEqual(default_runs_root(), expected)
                self.assertFalse(expected.exists())
            with patch.dict(os.environ, {'XDG_DATA_HOME': tmp}):
                self.assertEqual(default_runs_root(), Path(tmp) / 'pyscf-agent/runs')
            with patch.dict(os.environ, {'PYSCF_AGENT_RUNS_DIR': tmp}):
                self.assertEqual(default_runs_root(), Path(tmp))
                self.assertEqual(study_search_roots(), (Path(tmp),))
            self.assertEqual(resolve_work_dir(tmp), Path(tmp))
            self.assertEqual(resolve_work_dir(tmp + '/run-1', 'run-1'), Path(tmp))

    def test_old_study_discovery_and_new_preparation_have_distinct_roots(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {}, clear=True),
        ):
            checkout = Path(tmp).resolve() / 'checkout'
            checkout.mkdir()
            data = Path(tmp).resolve() / 'data'
            with (
                patch('pathlib.Path.cwd', return_value=checkout),
                patch.dict(os.environ, {'XDG_DATA_HOME': str(data)}),
            ):
                service = StudyApplicationService()
                old = service.prepare_study(SPEC, work_dir=str(checkout / 'runs'))
                new = service.prepare_study(SPEC)
                opened = service.open_study(old['study_id'])
                self.assertEqual(Path(opened['work_dir']), checkout / 'runs')
                self.assertEqual(
                    Path(service.open_study(new['study_id'])['work_dir']),
                    default_runs_root(),
                )
                listed = {item['study_id']: item for item in service.list_studies()}
                self.assertEqual(set(listed), {old['study_id'], new['study_id']})
                self.assertEqual(
                    Path(listed[old['study_id']]['work_dir']), checkout / 'runs'
                )
                # The UI sends the displayed default root explicitly.
                self.assertEqual(
                    service.open_study(
                        old['study_id'], work_dir=str(default_runs_root())
                    )['study_id'],
                    old['study_id'],
                )
                self.assertEqual(
                    service.list_studies(work_dir=str(checkout / 'runs'))[0][
                        'study_id'
                    ],
                    old['study_id'],
                )
                with self.assertRaises(FileNotFoundError):
                    service.open_study(
                        old['study_id'], work_dir=str(Path(tmp) / 'isolated')
                    )
                with self.assertRaises(ValueError):
                    service.open_study('../outside')

    def test_collect_legacy_study_passes_its_original_root_to_executor(self):
        with (
            tempfile.TemporaryDirectory() as tmp,
            patch.dict(os.environ, {}, clear=True),
        ):
            checkout = Path(tmp).resolve() / 'checkout'
            checkout.mkdir()
            with (
                patch('pathlib.Path.cwd', return_value=checkout),
                patch.dict(os.environ, {'XDG_DATA_HOME': tmp}),
            ):
                service = StudyApplicationService()
                old = service.prepare_study(SPEC, work_dir=str(checkout / 'runs'))
                with patch(
                    'computational_study_agent.application.execution._backend_collect_study'
                ) as collect:
                    service.collect_study(old['study_id'])
                self.assertEqual(
                    collect.call_args.kwargs['work_dir'], str(checkout / 'runs')
                )
