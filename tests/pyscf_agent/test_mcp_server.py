from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
import time
import unittest
from functools import wraps
from pathlib import Path
from unittest.mock import Mock, patch

from pyscf_agent.application import CalculationApplicationService
from pyscf_agent.contracts import TaskReport
from pyscf_agent.executors import JobHandle, JobState, JobStatus, LocalExecutor, LocalProcessExecutor
from pyscf_agent.mcp_server.cli import build_parser, main


H2 = {'task_type': 'molecular', 'atom': 'H 0 0 0; H 0 0 0.74',
      'basis': 'sto-3g', 'method': 'hf', 'outputs': ['energy']}
HAS_MCP = importlib.util.find_spec('mcp') is not None
HAS_PYSCF = importlib.util.find_spec('pyscf') is not None


def with_client(function):
    @wraps(function)
    async def run(self):
        from mcp import Client
        # AnyIO scopes must enter and exit within the same asyncio task.
        async with Client(self.server) as self.client:
            await function(self)
    return run


class StructuredTaskPreparationTests(unittest.TestCase):
    def test_shorthand_and_public_spec_use_the_same_validation_without_execution(self):
        executor = Mock()
        service = CalculationApplicationService(task_executor=executor)
        with tempfile.TemporaryDirectory() as root, patch.dict(os.environ, {'PYSCF_AGENT_RUNS_DIR': root}):
            prepared = service.validate_task_spec(H2)
            roundtrip = service.validate_task_spec(prepared['task_spec'])
            self.assertEqual(list(Path(root).iterdir()), [])
        self.assertTrue(prepared['valid'])
        self.assertEqual(roundtrip['task_spec'], prepared['task_spec'])
        self.assertTrue(prepared['workflow_configuration'])
        self.assertEqual(executor.mock_calls, [])

    def test_missing_geometry_and_unsupported_method_are_invalid(self):
        service = CalculationApplicationService()
        self.assertFalse(service.validate_task_spec({})['valid'])
        invalid = service.validate_task_spec(dict(H2, method='invented_method'))
        self.assertFalse(invalid['valid'])
        self.assertTrue(any(e['code'] == 'unsupported_method' for e in invalid['errors']))

    def test_unknown_contract_version_is_rejected_before_normalization(self):
        with self.assertRaises(ValueError):
            CalculationApplicationService().validate_task_spec(dict(H2, schema='task.v999'))


class MCPCommandTests(unittest.TestCase):
    def test_parser_shares_existing_remote_options(self):
        args = build_parser().parse_args(['--executor', 'remote', '--remote', 'amarel', '--work-dir', '/tmp/tasks'])
        self.assertEqual(args.remote, 'amarel')
        self.assertEqual(args.work_dir, '/tmp/tasks')

    @unittest.skipUnless(HAS_MCP, 'optional MCP SDK is not installed')
    def test_local_cli_uses_supervised_process_executor(self):
        with patch('pyscf_agent.mcp_server.cli.LocalProcessExecutor') as executor, \
                patch('pyscf_agent.mcp_server.server.create_server') as create:
            self.assertEqual(main(['--executor', 'local', '--local-wall-time-seconds', '30']), 0)
        executor.assert_called_once_with(wall_time_seconds=30.0)
        create.return_value.run.assert_called_once_with(transport='stdio')


@unittest.skipUnless(HAS_MCP, 'optional MCP SDK is not installed')
class MCPProtocolTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from pyscf_agent.mcp_server.server import create_server
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name).resolve()
        self.executed = []

        def run(request, **kwargs):
            self.executed.append((request, kwargs))
            return {'task_report': TaskReport(
                run_id=kwargs['run_id'], work_dir=kwargs['work_dir'],
                execution_status='succeeded', compact_results={'energy': -1.1},
                analysis_summary='Fixture result', raw_stdout='verbose solver output',
            ).to_dict()}

        self.executor = LocalExecutor(request_runner=run)
        self.service = CalculationApplicationService(task_executor=self.executor)
        self.server = create_server(self.service, work_dir=self.root, executor_description=self.executor.describe())

    async def call(self, name, **arguments):
        result = await self.client.call_tool(name, arguments)
        self.assertFalse(result.is_error, result.content)
        return result.structured_content

    @with_client
    async def test_discovery_validation_and_wiki_resources(self):
        listed = await self.client.list_tools()
        self.assertEqual({tool.name for tool in listed.tools}, {
            'get_capabilities', 'validate_task', 'submit_task',
            'get_task_status', 'collect_task', 'cancel_task', 'show_task', 'refresh_task_view'})
        for tool in listed.tools:
            self.assertEqual(tool.input_schema['type'], 'object')
        capabilities = await self.call('get_capabilities', namespace='molecular.method')
        self.assertTrue(capabilities['entries'])
        valid = await self.call('validate_task', task_spec=H2)
        self.assertTrue(valid['valid'])
        wiki = await self.client.read_resource('pyscf://wiki')
        pages = json.loads(wiki.contents[0].text)
        self.assertGreater(len(pages), 0)
        page = await self.client.read_resource(pages[0]['uri'])
        self.assertTrue(page.contents[0].text.strip())
        self.assertEqual(self.executed, [])

    @with_client
    async def test_collect_and_resources_survive_server_restart_without_resubmission(self):
        from mcp import Client
        from pyscf_agent.mcp_server.server import create_server
        submission = await self.call('submit_task', task_spec=H2, run_id='fixture-h2')
        handle = submission['handle']
        self.assertEqual(self.executed[0][1]['channel'], 'mcp')
        normalized_input = json.loads(self.executed[0][0])
        self.assertEqual(normalized_input['system']['basis'], 'sto-3g')
        fresh_executor = LocalExecutor(request_runner=lambda *a, **kw: self.fail('Unexpected resubmission'))
        server = create_server(CalculationApplicationService(task_executor=fresh_executor),
                               work_dir=self.root, executor_description=fresh_executor.describe())
        async with Client(server) as client:
            for _ in range(2):
                status = await client.call_tool('get_task_status', {'handle': handle})
                self.assertEqual(status.structured_content['state'], 'completed')
                result = await client.call_tool('collect_task', {'handle': handle})
                self.assertEqual(result.structured_content['summary']['execution_status'], 'succeeded')
                self.assertNotIn('raw_stdout', result.structured_content['summary'])
                self.assertNotIn('schema', result.structured_content['summary'])
            full = await client.read_resource(submission['report_uri'])
            self.assertEqual(json.loads(full.contents[0].text)['raw_stdout'], 'verbose solver output')
            full_tool = await client.call_tool('collect_task', {'handle': handle, 'include_report': True})
            self.assertEqual(full_tool.structured_content['report']['raw_stdout'], 'verbose solver output')
        self.assertEqual(len(self.executed), 1)

    @with_client
    async def test_invalid_input_and_outside_handles_do_not_execute(self):
        result = await self.call('submit_task', task_spec={}, run_id='invalid')
        self.assertFalse(result['submitted'])
        for run_id in ('../escape', '/escape', self.root.name):
            rejected = await self.client.call_tool('submit_task', {'task_spec': H2, 'run_id': run_id})
            self.assertTrue(rejected.is_error)
        handle = JobHandle(job_id='escape', executor_id='local', run_id='escape',
                           work_dir=str(self.root.parent), submitted_at='now').to_dict()
        for name in ('get_task_status', 'collect_task', 'cancel_task'):
            rejected = await self.client.call_tool(name, {'handle': handle})
            self.assertTrue(rejected.is_error)
        self.assertEqual(self.executed, [])

    @with_client
    async def test_scientific_failure_is_a_report_not_a_protocol_error(self):
        with patch.object(self.service, 'collect_request', return_value=TaskReport(
            execution_status='unconverged', errors=[{'code': 'scf_unconverged'}],
            approval={'required': True, 'kind': 'active_space'},
        ).to_dict()):
            handle = JobHandle(job_id='failed', executor_id='local', run_id='failed',
                               work_dir=str(self.root), submitted_at='now').to_dict()
            result = await self.call('collect_task', handle=handle)
        self.assertEqual(result['summary']['execution_status'], 'unconverged')
        self.assertEqual(result['summary']['errors'][0]['code'], 'scf_unconverged')
        self.assertTrue(result['summary']['approval']['required'])

    @with_client
    async def test_submission_errors_are_actionable_and_never_retried(self):
        from pyscf_agent.remote.protocol import RemoteProtocolError
        with patch.object(self.service, 'submit_request', side_effect=RemoteProtocolError('SSH interrupted')) as submit:
            result = await self.client.call_tool('submit_task', {'task_spec': H2, 'run_id': 'unknown'})
        self.assertTrue(result.is_error)
        self.assertIn('SSH interrupted', result.content[0].text)
        self.assertIn('No automatic retry', result.content[0].text)
        self.assertEqual(submit.call_count, 1)
        self.assertEqual(self.executed, [])

    @with_client
    async def test_duplicate_local_submission_does_not_overwrite_existing_job(self):
        await self.call('submit_task', task_spec=H2, run_id='duplicate')
        result = await self.client.call_tool('submit_task', {'task_spec': H2, 'run_id': 'duplicate'})
        self.assertTrue(result.is_error)
        self.assertIn('JobConflictError', result.content[0].text)
        self.assertEqual(len(self.executed), 1)

    @with_client
    async def test_cancel_and_not_ready_use_existing_executor_semantics(self):
        from mcp import Client
        from pyscf_agent.mcp_server.server import create_server
        from tests.pyscf_agent.test_local_process_executor import FakeProcess
        process = FakeProcess()
        executor = LocalProcessExecutor(popen_factory=lambda *a, **kw: process)
        server = create_server(CalculationApplicationService(task_executor=executor),
                               work_dir=self.root, executor_description=executor.describe())
        async with Client(server) as client:
            submitted = await client.call_tool('submit_task', {'task_spec': H2, 'run_id': 'cancel-me'})
            arguments = {'handle': submitted.structured_content['handle']}
            not_ready = await client.call_tool('collect_task', arguments)
            self.assertTrue(not_ready.is_error)
            self.assertIn('JobNotReadyError', not_ready.content[0].text)
            cancelled = await client.call_tool('cancel_task', arguments)
            self.assertEqual(cancelled.structured_content['state'], 'cancelled')
            repeated = await client.call_tool('cancel_task', arguments)
            self.assertEqual(repeated.structured_content['state'], 'cancelled')
            self.assertTrue(process.terminated)

    @with_client
    async def test_remote_handle_paths_are_resolved_by_configured_executor(self):
        from mcp import Client
        from pyscf_agent.mcp_server.server import create_server
        handle = JobHandle(job_id='remote-h2', executor_id='slurm', run_id='remote-h2',
                           work_dir='/scratch/remote/tasks', submitted_at='now', backend_job_id='123')
        remote = Mock()
        remote.status.return_value = JobStatus(handle=handle, state=JobState.QUEUED, updated_at='now')
        server = create_server(CalculationApplicationService(task_executor=remote), work_dir=self.root,
                               executor_description={'location': 'remote_slurm_cluster'})
        async with Client(server) as client:
            status = await client.call_tool('get_task_status', {'handle': handle.to_dict()})
            self.assertFalse(status.is_error, status.content)
            self.assertEqual(status.structured_content['state'], 'queued')
        remote.status.assert_called_once_with(handle)


@unittest.skipUnless(HAS_MCP, 'optional MCP SDK is not installed')
class MCPStdioTests(unittest.IsolatedAsyncioTestCase):
    def parameters(self, root):
        from mcp import StdioServerParameters
        return StdioServerParameters(
            command=sys.executable,
            args=['-m', 'pyscf_agent.mcp_server', '--executor', 'local', '--work-dir', root],
            env=dict(os.environ, OMP_NUM_THREADS='1', OPENBLAS_NUM_THREADS='1',
                     MKL_NUM_THREADS='1', MPLCONFIGDIR=str(Path(root) / '.matplotlib')),
        )

    async def test_stdio_discovery_and_validation(self):
        from mcp import Client
        with tempfile.TemporaryDirectory() as root:
            for mode in ('auto', 'legacy'):
                async with Client(self.parameters(root), mode=mode) as client:
                    listed = await client.list_tools()
                    self.assertTrue({'show_task', 'refresh_task_view', 'open_task_monitor'}.issubset(
                        {tool.name for tool in listed.tools}))
                    self.assertIn('open_workbench', {tool.name for tool in listed.tools})
                    result = await client.call_tool('validate_task', {'task_spec': H2})
                    self.assertFalse(result.is_error, result.content)
                    self.assertTrue(result.structured_content['valid'])
            self.assertEqual(list(Path(root).iterdir()), [])

    async def test_cancel_running_worker_after_stdio_server_restart(self):
        from mcp import Client, StdioServerParameters
        fixture_server = '''
import sys
from pathlib import Path
from pyscf_agent.application import CalculationApplicationService
from pyscf_agent.executors import LocalProcessExecutor
from pyscf_agent.mcp_server.server import create_server
from tests.pyscf_agent.test_local_process_executor import LocalProcessLifecycleTests
executor = LocalProcessExecutor(popen_factory=LocalProcessLifecycleTests.launch_fixture,
                                terminate_timeout=0.3, wall_time_seconds=20)
create_server(CalculationApplicationService(task_executor=executor),
              work_dir=Path(sys.argv[1]), executor_description=executor.describe()).run()
'''
        with tempfile.TemporaryDirectory() as root:
            parameters = StdioServerParameters(command=sys.executable, args=['-c', fixture_server, root],
                                                env=dict(os.environ))
            handle = None
            try:
                async with Client(parameters) as client:
                    result = await client.call_tool('submit_task', {'task_spec': H2, 'run_id': 'cancel-restart'})
                    self.assertFalse(result.is_error, result.content)
                    handle = result.structured_content['handle']
                    deadline = time.monotonic() + 10
                    while not (Path(root) / 'cancel-restart' / 'fixture-ready').exists():
                        self.assertLess(time.monotonic(), deadline, 'Worker did not start')
                        await asyncio.sleep(0.05)
                async with Client(self.parameters(root)) as client:
                    arguments = {'handle': handle}
                    status = await client.call_tool('get_task_status', arguments)
                    self.assertEqual(status.structured_content['state'], 'running')
                    cancelled = await client.call_tool('cancel_task', arguments)
                    self.assertFalse(cancelled.is_error, cancelled.content)
                    self.assertEqual(cancelled.structured_content['state'], 'cancelled')
                    status = await client.call_tool('get_task_status', arguments)
                    self.assertEqual(status.structured_content['state'], 'cancelled')
            finally:
                if handle is not None:
                    LocalProcessExecutor().cancel(handle)

    @unittest.skipUnless(HAS_PYSCF and os.environ.get('PYSCF_AGENT_TEST_MCP_NUMERICAL') == '1',
                         'set PYSCF_AGENT_TEST_MCP_NUMERICAL=1 for real H2 acceptance')
    async def test_real_h2_collect_after_stdio_server_restart(self):
        from mcp import Client
        with tempfile.TemporaryDirectory() as root:
            async with Client(self.parameters(root)) as client:
                result = await client.call_tool('submit_task', {'task_spec': H2, 'run_id': 'accept-h2'})
                self.assertFalse(result.is_error, result.content)
                submission = result.structured_content
            # Starting another protocol server must not start another calculation.
            async with Client(self.parameters(root)) as client:
                handle = submission['handle']
                deadline = time.monotonic() + 90
                while True:
                    result = await client.call_tool('get_task_status', {'handle': handle})
                    self.assertFalse(result.is_error, result.content)
                    status = result.structured_content
                    if status['terminal']:
                        break
                    self.assertLess(time.monotonic(), deadline, status)
                    await asyncio.sleep(0.2)
                self.assertTrue(status['report_available'], status)
                collected = await client.call_tool('collect_task', {'handle': handle, 'include_report': True})
                self.assertFalse(collected.is_error, collected.content)
                report = collected.structured_content['report']
                self.assertEqual(report['execution_status'], 'succeeded', report['errors'])
                energy = report['compact_results']['energy']
                self.assertAlmostEqual(energy, -1.1167593074, places=7)
                print(json.dumps({'acceptance': 'mcp_stdio_restart_h2', 'energy_hartree': energy,
                                  'execution_status': report['execution_status']}))
                resource = await client.read_resource(submission['report_uri'])
                self.assertEqual(json.loads(resource.contents[0].text)['run_id'], 'accept-h2')
                repeated = await client.call_tool('collect_task', {'handle': handle})
                self.assertEqual(repeated.structured_content['summary']['compact_results']['energy'], energy)
            states = list(Path(root).rglob('job-state.json'))
            self.assertEqual(len(states), 1)


if __name__ == '__main__':
    unittest.main()
