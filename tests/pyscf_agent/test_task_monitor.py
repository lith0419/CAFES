from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit

from pyscf_agent.application import CalculationApplicationService
from pyscf_agent.executors import JobHandle, JobStatus, JobState, LocalExecutor
from pyscf_agent.executors.inspection import LOG_TAIL_BYTES
from pyscf_agent.providers.block2.progress import parse_sweep_progress
from pyscf_agent.pyscf_agent_web_api import handle_task_view_request
from pyscf_agent.remote.rpc_cli import dispatch_rpc
from pyscf_agent.workbench import LocalWorkbench
from tests.pyscf_agent import test_mcp_server as mcp_tests
from tests.pyscf_agent.test_mcp_server import HAS_MCP, with_client, H2


def header(index, bond=2000):
    return f'Sweep = {index} | Direction = forward | Bond dimension = {bond} | Noise = 0.00e+00 | Dav threshold = 1.00e-09\n'


RESULT = 'Time elapsed = 706.791 | E = -7.3614016097 | DE = -2.05e-05 | DW = 6.44344e-04\n'


class TaskInspectionTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name).resolve()
        self.run = self.root / 'job'
        self.run.mkdir()
        self.handle = JobHandle('job', 'local', 'job', str(self.root), '2026-09-20T12:00:00+00:00')
        self.executor = LocalExecutor(request_runner=lambda *args, **kwargs: self.fail('Unexpected calculation'))
        self.service = CalculationApplicationService(task_executor=self.executor)
        self.status = JobStatus(self.handle, JobState.RUNNING, '2026-09-20T12:00:01+00:00',
                                started_at='2026-09-20T12:00:01+00:00')
        self.executor._write_status(self.status)
        self.log = self.run / 'solver-dmet' / 'log-libdmet-output.log'
        self.log.parent.mkdir()

    def test_parser_keeps_active_and_completed_sweeps_separate_and_resets_stage(self):
        parsed = parse_sweep_progress(header(7) + RESULT + header(8))
        self.assertEqual(parsed['observed_sweep']['sweep_index'], 8)
        self.assertEqual(parsed['last_completed_sweep']['sweep_index'], 7)
        self.assertAlmostEqual(parsed['last_completed_sweep']['energy_change'], 2.05e-5)
        self.assertIsNone(parse_sweep_progress(header(7) + RESULT + header(0))['last_completed_sweep'])
        self.assertIsNone(parse_sweep_progress(RESULT))
        self.assertIsNone(parse_sweep_progress('cycle= 2 E= -3 delta_E= 1e-4'))

    def test_running_view_is_bounded_read_only_and_ignores_partial_last_line(self):
        self.log.write_text('x' * (LOG_TAIL_BYTES * 3) + '\n' + header(7) + RESULT + header(8) + RESULT.rstrip())
        before = {p: p.read_bytes() for p in self.run.rglob('*') if p.is_file()}
        with patch.object(self.executor, 'fetch', side_effect=AssertionError('Do not collect')):
            view = self.service.inspect_task(self.handle)
        self.assertFalse(view['status']['terminal'])
        self.assertIsNone(view['summary'])
        self.assertEqual(view['solver_progress']['observed_sweep']['sweep_index'], 8)
        self.assertEqual(view['solver_progress']['last_completed_sweep']['sweep_index'], 7)
        self.assertLessEqual(len(view['logs'][0]['tail']), 6000)
        self.assertTrue(view['logs'][0]['truncated'])
        self.assertEqual(before, {p: p.read_bytes() for p in self.run.rglob('*') if p.is_file()})

    def test_completed_process_can_have_failed_science_and_no_final_energy(self):
        from dataclasses import replace
        status = replace(self.status, state=JobState.COMPLETED, task_status='failed', report_available=True,
                         completed_at='2026-09-20T12:01:01+00:00')
        self.executor._write_status(status)
        report = {'run_id': 'job', 'execution_status': 'failed', 'errors': [{'message': 'DMRG did not converge'}],
                  'compact_results': {}, 'raw_stdout': 'Must not be included'}
        (self.run / 'job-task-report.json').write_text(json.dumps(report))
        self.log.write_text(header(7) + RESULT)
        view = self.service.inspect_task(self.handle)
        self.assertEqual(view['elapsed_seconds'], 60)
        self.assertEqual(view['status']['state'], 'completed')
        self.assertEqual(view['summary']['execution_status'], 'failed')
        self.assertIsNone(view['summary']['energy'])
        self.assertNotIn('raw_stdout', view['summary'])
        self.assertEqual(view['summary']['errors'][0]['message'], 'DMRG did not converge')
        report['run_id'] = 'other'
        (self.run / 'job-task-report.json').write_text(json.dumps(report))
        view = self.service.inspect_task(self.handle)
        self.assertIsNone(view['summary'])
        self.assertIn('does not match', view['notices'][0])

    def test_no_logs_symlinks_and_wrong_handles(self):
        outside = self.root / 'outside.log'
        outside.write_text(header(0) + RESULT)
        self.log.symlink_to(outside)
        view = self.service.inspect_task(self.handle)
        self.assertEqual(view['logs'], [])
        self.assertIsNone(view['solver_progress'])
        wrong = {**self.handle.to_dict(), 'job_id': 'different'}
        with self.assertRaises(Exception):
            self.service.inspect_task(wrong)

    def test_http_and_rpc_use_same_inspection_without_execution(self):
        self.log.write_text(header(7) + RESULT)
        code, headers, body = handle_task_view_request(json.dumps({'handle': self.handle.to_dict()}).encode(),
                                                     task_executor=self.executor, work_dir=str(self.root))
        self.assertEqual(code, 200)
        self.assertEqual(headers['Cache-Control'], 'no-store')
        view = json.loads(body)
        rpc = dispatch_rpc('inspect-task', {'handle': self.handle.to_dict()},
                           executor=self.executor, work_root=str(self.root))
        self.assertEqual(view['solver_progress'], rpc['solver_progress'])
        code, _, _ = handle_task_view_request(json.dumps({'handle': self.handle.to_dict()}).encode(),
                                              task_executor=self.executor, work_dir=str(self.root / 'elsewhere'))
        self.assertEqual(code, 400)

    def test_executor_without_inspection_returns_explicit_status_only_view(self):
        class StatusOnly:
            def status(inner, handle):
                return self.status
        view = CalculationApplicationService(task_executor=StatusOnly()).inspect_task(self.handle)
        self.assertIsNone(view['solver_progress'])
        self.assertIn('status only', view['notices'][0])

    def test_workbench_fragment_retains_exact_handle_and_target(self):
        wb = LocalWorkbench(self.root, [], 'amarel')
        with patch.object(wb, 'open', return_value={'url': 'http://127.0.0.1:1234/computational-study/', 'started': False}):
            url = urlsplit(wb.open_task(self.handle)['url'])
        self.assertEqual(url.path, '/task-monitor/')
        self.assertFalse(url.query)
        values = parse_qs(url.fragment)
        self.assertEqual(json.loads(values['handle'][0]), self.handle.to_dict())
        self.assertEqual(values['execution_target'], ['amarel'])


@unittest.skipUnless(HAS_MCP, 'optional MCP SDK is not installed')
class TaskViewMCPTests(unittest.IsolatedAsyncioTestCase):
    setUp = mcp_tests.MCPProtocolTests.setUp
    call = mcp_tests.MCPProtocolTests.call

    @with_client
    async def test_card_resource_fallback_and_reconnect_do_not_submit(self):
        from pyscf_agent.mcp_server.task_view import TASK_VIEW_URI
        tools = {tool.name: tool.model_dump(by_alias=True) for tool in (await self.client.list_tools()).tools}
        self.assertEqual(tools['show_task']['_meta']['ui']['resourceUri'], TASK_VIEW_URI)
        self.assertEqual(tools['refresh_task_view']['_meta']['ui']['visibility'], ['app'])
        resource = (await self.client.read_resource(TASK_VIEW_URI)).contents[0]
        self.assertEqual(resource.mime_type, 'text/html;profile=mcp-app')
        self.assertIn('ui/initialize', resource.text)
        self.assertNotIn('/* TASK_MONITOR_', resource.text)
        job = await self.call('submit_task', task_spec=H2, run_id='monitor')
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        for name in ('show_task', 'refresh_task_view'):
            result = await self.call(name, handle=job['handle'])
            self.assertEqual(result['summary']['energy'], -1.1)
            self.assertTrue(result['status']['terminal'])
            self.assertFalse(result['can_open_workbench'])
        self.assertEqual(len(self.executed), 1)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        outside = {**job['handle'], 'work_dir': str(self.root.parent)}
        self.assertTrue((await self.client.call_tool('show_task', {'handle': outside})).is_error)

    @with_client
    async def test_open_monitor_validates_handle_and_never_submits(self):
        from mcp import Client
        from pyscf_agent.mcp_server.server import create_server
        job = await self.call('submit_task', task_spec=H2, run_id='monitor-open')
        workbench = Mock()
        workbench.open_task.return_value = {'url': 'http://127.0.0.1:1234/task-monitor/'}
        server = create_server(self.service, work_dir=self.root, executor_description=self.executor.describe(),
                               workbench=workbench)
        async with Client(server) as client:
            result = await client.call_tool('open_task_monitor', {'handle': job['handle']})
            self.assertFalse(result.is_error)
            workbench.open_task.assert_called_once_with(JobHandle.from_dict(job['handle']))
            result = await client.call_tool('open_task_monitor', {'handle': {**job['handle'], 'job_id': 'wrong'}})
            self.assertTrue(result.is_error)
        self.assertEqual(workbench.open_task.call_count, 1)
        self.assertEqual(len(self.executed), 1)


@unittest.skipUnless(shutil.which('node'), 'Node is required for monitor tests')
class MonitorBrowserTests(unittest.TestCase):
    def test_polling_and_mcp_lifecycle(self):
        result = subprocess.run(['node', str(Path(__file__).with_name('task_monitor_bridge.cjs'))],
                                capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
