from __future__ import annotations

import asyncio
import json
import os
from pathlib import Path
import signal
import tempfile
import time
import unittest
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlsplit
from urllib.request import Request, urlopen

from pyscf_agent.mcp_server.cli import build_parser
from pyscf_agent.workbench import LocalWorkbench
from tests.pyscf_agent import test_codex_plugin as plugin_tests
from tests.pyscf_agent.test_mcp_server import HAS_MCP
from tests.pyscf_agent.test_mcp_studies import STUDY_SPEC


class WorkbenchConfigurationTests(unittest.TestCase):
    def test_configuration_preserves_paths_targets_and_separates_profiles(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = root / 'executor config.ini'
            config.write_text('[fixture]\n')
            remote = LocalWorkbench.from_args(build_parser().parse_args([
                '--executor', 'remote', '--remote', 'Amarel', '--remote-config', str(config),
                '--work-dir', str(root / 'runs with spaces'), '--local-wall-time-seconds', '90']))
            self.assertEqual(remote.execution_target, 'amarel')
            self.assertIn(str(config.resolve()), remote.arguments)
            self.assertEqual(remote.arguments[-2:], ['--local-wall-time-seconds', '90.0'])
            slurm = LocalWorkbench.from_args(build_parser().parse_args([
                '--executor', 'slurm', '--slurm-config', str(config), '--slurm-profile', 'small',
                '--work-dir', str(remote.root)]))
            self.assertEqual(slurm.execution_target, 'slurm')
            self.assertNotEqual(remote.state_path, slurm.state_path)
            self.assertFalse(remote.directory.exists())
            state = {'port': 12345}
            query = parse_qs(urlsplit(remote.url(state, 'saved-study')).query)
            self.assertEqual(query['work_dir'], [str(remote.root)])
            self.assertEqual(query['execution_target'], ['amarel'])
            self.assertEqual(query['study'], ['saved-study'])

    def test_failed_launch_has_actionable_log_and_is_not_retried(self):
        with tempfile.TemporaryDirectory() as root:
            workbench = LocalWorkbench(root, ['--executor', 'local'], 'local')
            process = Mock()
            process.poll.return_value = 2
            with patch('pyscf_agent.workbench.subprocess.Popen', return_value=process) as launch:
                with self.assertRaisesRegex(OSError, 'Workbench exited during startup. See'):
                    workbench.open()
            launch.assert_called_once()
            self.assertTrue(launch.call_args.kwargs['start_new_session'])
            self.assertEqual(launch.call_args.args[0][4:8], ['--host', '127.0.0.1', '--port', '0'])


@unittest.skipUnless(HAS_MCP, 'optional MCP SDK is not installed')
class WorkbenchPluginTests(unittest.IsolatedAsyncioTestCase):
    setUp = plugin_tests.CodexPluginNumericalTests.setUp
    parameters = plugin_tests.CodexPluginNumericalTests.parameters
    call = plugin_tests.CodexPluginNumericalTests.call

    def stop(self, workbench):
        state = workbench.running()
        if state:
            os.kill(state['pid'], signal.SIGTERM)
            for _ in range(40):
                if workbench.running() is None:
                    break
                time.sleep(0.05)

    def post(self, url, route, payload):
        base = urlsplit(url)
        request = Request(f'{base.scheme}://{base.netloc}/api/{route}',
            data=json.dumps(payload).encode(), headers={'Content-Type': 'application/json'})
        with urlopen(request, timeout=5) as response:
            return json.load(response)

    async def test_cached_plugin_opens_reuses_and_recovers_workbench_without_execution(self):
        from mcp import Client
        root = self.root / "runs with spaces & 'quotes'"
        params = self.parameters(str(root))
        workbench = LocalWorkbench(root, ['--executor', 'local'], 'local')
        self.addCleanup(self.stop, workbench)
        async with Client(params) as client:
            self.assertTrue({'show_task', 'refresh_task_view', 'open_task_monitor'}.issubset(
                {tool.name for tool in (await client.list_tools()).tools}))
            prepared = await self.call(client, 'prepare_study', study_spec=STUDY_SPEC)
            study_id = prepared['study_id']
            self.assertNotIn('workbench_url', prepared)
            self.assertFalse(workbench.directory.exists())
            for invalid in ('../outside', 'absent'):
                result = await client.call_tool('open_workbench', {'study_id': invalid})
                self.assertTrue(result.is_error)
            self.assertFalse(workbench.directory.exists())
            # Independent MCP calls racing to start share one Web process.
            results = await asyncio.gather(*[
                self.call(client, 'open_workbench', study_id=study_id) for _ in range(2)])
            self.assertEqual(sorted(item['started'] for item in results), [False, True])
            self.assertEqual(results[0]['pid'], results[1]['pid'])
            opened = results[0]
        # The launcher survives the plugin/MCP process exiting.
        with urlopen(opened['url'], timeout=5) as response:
            self.assertIn(b'Computational Study Planner', response.read())
        saved = self.post(opened['url'], 'study-open', {'study_id': study_id})
        self.assertEqual(saved['study_id'], study_id)
        self.assertEqual(saved['work_dir'], str(root.resolve()))
        async with Client(params) as client:
            status = await self.call(client, 'get_study_status', study_id=study_id)
            self.assertEqual(status['workbench_url'], opened['url'])
            self.assertEqual(status['status'], 'prepared')
            reused = await self.call(client, 'open_workbench', study_id=study_id)
            self.assertEqual(reused['pid'], opened['pid'])
            self.assertFalse(reused['started'])
            self.stop(workbench)
            recovered = await self.call(client, 'open_workbench', study_id=study_id)
            self.assertTrue(recovered['started'])
            self.assertNotEqual(recovered['pid'], opened['pid'])
            home = await self.call(client, 'open_workbench')
            self.assertEqual(home['pid'], recovered['pid'])
            self.assertNotIn('study', parse_qs(urlsplit(home['url']).query))
        self.assertEqual(list(root.rglob('job-state.json')), [])


if __name__ == '__main__':
    unittest.main()
