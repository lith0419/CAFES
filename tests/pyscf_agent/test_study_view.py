from __future__ import annotations

import copy
import shutil
import subprocess
import unittest
from pathlib import Path
from unittest.mock import Mock

from tests.pyscf_agent import test_mcp_studies as studies
from tests.pyscf_agent.test_mcp_server import HAS_MCP, with_client


@unittest.skipUnless(HAS_MCP, 'optional MCP SDK is not installed')
class StudyViewTests(unittest.IsolatedAsyncioTestCase):
    setUp = studies.MCPStudyTests.setUp
    call = studies.MCPStudyTests.call

    @with_client
    async def test_wire_metadata_resource_and_text_only_fallback(self):
        from pyscf_agent.mcp_server.study_view import STUDY_VIEW_URI
        tools = {tool.name: tool.model_dump(by_alias=True) for tool in (await self.client.list_tools()).tools}
        self.assertEqual(tools['show_study']['_meta']['ui']['resourceUri'], STUDY_VIEW_URI)
        self.assertEqual(tools['refresh_study_view']['_meta']['ui'], {'visibility': ['app']})
        self.assertTrue(tools['show_study']['annotations']['readOnlyHint'])
        self.assertNotIn('_meta', {k: v for k, v in tools['get_study_status'].items() if v is not None})
        resource = (await self.client.read_resource(STUDY_VIEW_URI)).contents[0].model_dump(by_alias=True)
        self.assertEqual(resource['mimeType'], 'text/html;profile=mcp-app')
        self.assertEqual(resource['_meta']['ui']['csp'], {'connectDomains': [], 'resourceDomains': []})
        self.assertIn('ui/initialize', resource['text'])
        saved = await self.call('prepare_study', study_spec=studies.STUDY_SPEC)
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        shown = await self.client.call_tool('show_study', {'study_id': saved['study_id']})
        self.assertFalse(shown.is_error, shown.content)
        self.assertTrue(shown.content)  # Ordinary MCP clients still get text.
        view = shown.structured_content
        self.assertEqual(view['execution_status'], 'prepared')
        self.assertEqual(view['task_status_counts'], {'not_executed': 2})
        self.assertIsNone(view['rows'][0]['converged'])
        self.assertFalse(view['can_open_workbench'])
        await self.call('refresh_study_view', study_id=saved['study_id'])
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})
        self.assertEqual(self.executor.submit_count, 0)
        for invalid in ('../outside', 'absent'):
            result = await self.client.call_tool('show_study', {'study_id': invalid})
            self.assertTrue(result.is_error)

    @with_client
    async def test_completed_view_does_not_collect_or_start_workbench(self):
        from pyscf_agent.mcp_server.study_view import register_study_view
        from pyscf_agent.mcp_server.server import _tool_errors
        # Existing server uses no workbench. Re-register on a separate server with a spy.
        from mcp.server import MCPServer
        wb = Mock()
        server = MCPServer('card-only')
        register_study_view(server, self.service, self.root, _tool_errors, lambda x: x, workbench=wb)
        saved = await self.call('prepare_study', study_spec=studies.STUDY_SPEC)
        await self.call('submit_study', study_id=saved['study_id'])
        await self.call('collect_study', study_id=saved['study_id'])
        before = {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()}
        count = self.executor.submit_count
        from mcp import Client
        async with Client(server) as client:
            result = await client.call_tool('show_study', {'study_id': saved['study_id']})
        self.assertFalse(result.is_error, result.content)
        self.assertEqual(result.structured_content['study_status'], 'succeeded')
        self.assertTrue(result.structured_content['can_open_workbench'])
        self.assertEqual(wb.mock_calls, [])
        self.assertEqual(self.executor.submit_count, count)
        self.assertEqual(before, {p: p.read_bytes() for p in self.root.rglob('*') if p.is_file()})

    def test_projection_preserves_false_missing_old_results_and_adaptive_scope(self):
        from pyscf_agent.mcp_server.study_view import study_view
        cases = [{'case_id': str(i), 'label': '<script>hello</script>',
                  'request': {'method': 'casscf', 'active_space': {'ncas': 4, 'nelecas': [2, 2]}},
                  'task_report': {'execution_status': 'failed', 'compact_results':
                                  {'energy': 0.0, 'converged': False}}} for i in range(25)]
        saved = {'study_id': 'adaptive', 'mode': 'adaptive', 'initial_scan_plan': {'name': 'CAS'},
                 'report': {'cases': cases, 'status': 'pending_review'}}
        execution = {'status': 'running', 'task_status_counts': {'pending': 2, 'succeeded': 23},
                     'receipts': [{'summary': {'state_counts': {'running': 2}}}]}
        before = copy.deepcopy((saved, execution))
        view = study_view(saved, execution)
        self.assertEqual((saved, execution), before)
        self.assertEqual(len(view['rows']), 20)
        self.assertEqual(view['case_count'], 25)
        self.assertTrue(view['review_required'])
        self.assertEqual(view['execution_status'], 'running')
        self.assertEqual(view['study_status'], 'pending_review')
        self.assertEqual(view['scheduler_state_counts'], {'running': 2})
        self.assertEqual(view['rows'][0]['energy'], 0)
        self.assertEqual(view['rows'][0]['energy_unit'], 'Ha')
        self.assertIs(view['rows'][0]['converged'], False)
        self.assertEqual(view['rows'][0]['active_space'], {'ncas': 4, 'nelecas': [2, 2]})
        for energy in (float('nan'), float('inf'), True, [-1, -2]):
            cases[0]['task_report']['compact_results'] = {'energy': energy}
            row = study_view(saved, {})['rows'][0]
            self.assertIsNone(row['energy'])
            self.assertIsNone(row['converged'])
        self.assertEqual(study_view(saved, {})['count_source'], 'saved_report')
        cases[0]['request'] = {'task_type': 'model_hamiltonian'}
        cases[0]['task_report']['compact_results'] = {'energy': -4, 'energy_unit': 'eV'}
        self.assertEqual(study_view(saved, {})['rows'][0]['energy_unit'], 'eV')


@unittest.skipUnless(shutil.which('node'), 'Node is required for card bridge tests')
class StudyCardBridgeTests(unittest.TestCase):
    def test_host_lifecycle_actions_errors_and_safe_rendering(self):
        script = Path(__file__).with_name('study_card_bridge.cjs')
        result = subprocess.run(['node', str(script)], text=True, capture_output=True)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
