"""MCP and HTTP share persisted Studies, including review and selective retry."""
from __future__ import annotations

import json
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from pathlib import Path
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from computational_study_agent.application import StudyApplicationService
from computational_study_agent.web_ui import DEFAULT_STUDY_SPEC
from pyscf_agent.application import CalculationApplicationService
from pyscf_agent.web.server import AgentWebHandler
from tests.computational_study_agent.test_retry_collection import Remote
from tests.computational_study_agent.test_study_background import inline_launcher
from tests.pyscf_agent.test_mcp_server import HAS_MCP, with_client
from tests.pyscf_agent.test_mcp_studies import STUDY_SPEC


@unittest.skipUnless(HAS_MCP, 'optional MCP SDK is not installed')
class StudyWorkbenchTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from pyscf_agent.mcp_server.server import create_server
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.executor = Remote(fail_first=False)
        self.service = StudyApplicationService(task_executor=self.executor, execution_config={'execution_target': 'remote'})
        self.service._study_launcher = inline_launcher(self.service)
        self.server = create_server(CalculationApplicationService(task_executor=self.executor),
            study_service=self.service, work_dir=self.root, executor_description=self.executor.describe())
        # A separate service instance reads only disk artifacts from the MCP side.
        web_service = StudyApplicationService(task_executor=self.executor, execution_config={'execution_target': 'remote'})
        web_service._study_launcher = inline_launcher(web_service)
        services = patch('computational_study_agent.application._STUDY_APPLICATION_SERVICES', {'fixture': web_service})
        services.start()
        self.addCleanup(services.stop)
        handler = type('WorkbenchHandler', (AgentWebHandler,), {'study_work_dir': str(self.root)})
        self.http = ThreadingHTTPServer(('127.0.0.1', 0), handler)
        thread = threading.Thread(target=self.http.serve_forever, daemon=True)
        thread.start()
        self.addCleanup(self.http.server_close)
        self.addCleanup(self.http.shutdown)
        self.url = 'http://127.0.0.1:{0}'.format(self.http.server_port)

    async def call(self, name, **arguments):
        result = await self.client.call_tool(name, arguments)
        self.assertFalse(result.is_error, result.content)
        return result.structured_content

    def post(self, route, **arguments):
        request = Request(self.url + '/api/study-' + route,
            data=json.dumps({'execution_target': 'fixture', **arguments}).encode(),
            headers={'Content-Type': 'application/json'})
        try:
            response = urlopen(request, timeout=10)
        except HTTPError as exc:
            response = exc
        with response:
            return response.status, json.load(response)

    @with_client
    async def test_mcp_prepare_web_start_and_saved_subset_retry(self):
        prepared = await self.call('prepare_study', study_spec=STUDY_SPEC)
        study_id = prepared['study_id']
        first, second = [case['case_id'] for case in prepared['plan']['cases']]
        code, listing = self.post('list')
        self.assertEqual(code, 200)
        self.assertEqual([item['study_id'] for item in listing['studies']], [study_id])
        for _ in range(2):
            code, opened = self.post('open', study_id=study_id)
            self.assertEqual(code, 200)
            self.assertEqual(opened['plan'], prepared['plan'])
            self.assertIsNone(opened['report'])
        self.assertEqual(self.executor.submit_count, 0)
        code, started = self.post('start', study_id=study_id)
        self.assertEqual(code, 202)
        self.assertTrue(started['started'])
        before = (await self.call('collect_study', study_id=study_id, include_report=True))['report']
        code, reviewed = self.post('review-action', study_id=study_id,
            action_id='increase_recovery_max_cycle', case_ids=[first], plan_kind='static',
            # Browser copies cannot replace the saved plan or invent extra cases.
            plan={'study_id': 'wrong-study', 'cases': []}, study_state={'study_id': 'wrong-study'})
        self.assertEqual(code, 200, reviewed)
        self.assertTrue(reviewed['can_run'])
        self.assertEqual(self.executor.submit_count, 1)
        reopened = self.post('open', study_id=study_id)[1]
        self.assertEqual(reopened['report']['pending_review']['case_ids'], [first])
        self.assertTrue((await self.call('get_study_status', study_id=study_id))['pending_review']['can_run'])
        self.post('start', study_id=study_id)
        code, collected = self.post('execution-collect', study_id=study_id, saved_study=True)
        self.assertEqual(code, 200, collected)
        report = collected['report']
        self.assertEqual([case['attempt_count'] for case in report['cases']], [2, 1])
        self.assertEqual(report['cases'][1]['case_id'], second)
        self.assertEqual(report['cases'][1], before['cases'][1])
        self.assertEqual(report['cases'], self.service.load_report(study_id, work_dir=str(self.root))['cases'])
        self.post('execution-collect', study_id=study_id, saved_study=True)
        self.assertEqual(self.executor.submit_count, 2)

    @with_client
    async def test_cost_approval_survives_reopen_without_execution(self):
        spec = dict(STUDY_SPEC, resource_policy={'review_work_estimates': True, 'total_work_review_threshold': 1})
        prepared = await self.call('prepare_study', study_spec=spec)
        study_id = prepared['study_id']
        self.assertEqual(self.post('start', study_id=study_id)[0], 409)
        code, review = self.post('review-action', study_id=study_id, action_id='approve_cost_estimate', plan_kind='static')
        self.assertEqual(code, 200, review)
        self.assertTrue(self.post('open', study_id=study_id)[1]['report']['pending_review']['can_run'])
        self.assertEqual(self.executor.submit_count, 0)
        await self.call('submit_study', study_id=study_id)
        self.assertEqual(self.executor.submit_count, 1)

    @with_client
    async def test_adaptive_preparation_opens_and_collects_without_browser_spec(self):
        prepared = await self.call('prepare_study', study_spec=dict(STUDY_SPEC, study_mode='adaptive'))
        study_id = prepared['study_id']
        opened = self.post('open', study_id=study_id)[1]
        self.assertEqual(opened['mode'], 'adaptive')
        self.assertIn('initial_scan_plan', opened)
        code, started = self.post('start', study_id=study_id)
        self.assertEqual(code, 202, started)
        code, collected = self.post('execution-collect', study_id=study_id, saved_study=True)
        self.assertEqual(code, 200, collected)
        self.assertIn('adaptive', collected['report'])
        self.assertEqual(collected['report']['study_id'], study_id)

    @with_client
    async def test_web_prepares_static_and_adaptive_studies_for_mcp_execution(self):
        for mode, spec in (('static', STUDY_SPEC), ('adaptive', STUDY_SPEC), ('static', DEFAULT_STUDY_SPEC)):
            with self.subTest(mode=mode, system=spec['system_type']):
                before = self.executor.submit_count
                code, saved = self.post('prepare', study_spec=dict(spec, study_mode=mode))
                self.assertEqual(code, 201, saved)
                self.assertEqual(saved['mode'], mode)
                self.assertEqual(saved['work_dir'], str(self.root))
                self.assertIsNone(saved['report'])
                self.assertEqual(self.executor.submit_count, before)
                study_id = saved['study_id']
                opened = self.post('open', study_id=study_id)[1]
                self.assertEqual(opened.get('plan'), saved.get('plan'))
                self.assertEqual(opened.get('initial_scan_plan'), saved.get('initial_scan_plan'))
                status = await self.call('get_study_status', study_id=study_id)
                self.assertEqual(status['status'], 'prepared')
                await self.call('submit_study', study_id=study_id)
                collected = await self.call('collect_study', study_id=study_id, include_report=True)
                self.assertEqual(collected['report']['study_id'], study_id)
                self.assertEqual(self.post('open', study_id=study_id)[1]['report']['cases'], collected['report']['cases'])

    @with_client
    async def test_web_dataset_preparation_is_saved_without_running_trajectories(self):
        code, saved = self.post('prepare', planner_template='hamiltonian_dataset',
            dataset_spec={'dataset_id': 'web-dataset', 'name': 'Web dataset', 'target_molecule_count': 2},
            seed_geometries=[{'molecule_id': 'mol-' + str(index), 'geometry_id': 'seed',
                'atomic_numbers': [1, 1], 'positions': [[0, 0, 0], [0, 0, 0.74]]} for index in range(2)])
        self.assertEqual(code, 201, saved)
        self.assertEqual(saved['plan']['comparison']['mode'], 'hamiltonian_dataset_assembly')
        self.assertEqual(len(saved['plan']['cases']), 2)
        resource = await self.client.read_resource('pyscf://studies/' + saved['study_id'] + '/plan')
        self.assertEqual(json.loads(resource.contents[0].text), saved['plan'])
        self.assertEqual(self.executor.submit_count, 0)

    @with_client
    async def test_web_analysis_reads_and_saves_the_shared_report(self):
        from unittest.mock import Mock
        study_id = self.post('prepare', study_spec=STUDY_SPEC)[1]['study_id']
        await self.call('submit_study', study_id=study_id)
        await self.call('collect_study', study_id=study_id)
        before = self.executor.submit_count
        builder = Mock()
        builder.build_result_analysis.return_value = 'Saved results were analyzed.'
        with patch('pyscf_agent.web.server.llm_request_builder', builder):
            code, analyzed = self.post('result-analysis', study_id=study_id,
                report={'study_id': 'wrong', 'cases': []}, plan={'study_id': 'wrong'})
        self.assertEqual(code, 200, analyzed)
        self.assertEqual(analyzed['result_analysis'], 'Saved results were analyzed.')
        self.assertEqual(analyzed['report']['study_id'], study_id)
        self.assertEqual(len(analyzed['report']['cases']), 2)
        self.assertEqual(self.post('open', study_id=study_id)[1]['report'], analyzed['report'])
        resource = await self.client.read_resource('pyscf://studies/' + study_id + '/report')
        self.assertEqual(json.loads(resource.contents[0].text), analyzed['report'])
        self.assertEqual(self.executor.submit_count, before)
        builder.build_result_analysis.assert_called_once()

    def test_web_preparation_routes_missing_cas_space_to_saved_probe(self):
        code, saved = self.post('prepare', study_spec=dict(STUDY_SPEC,
            base_task={**STUDY_SPEC['base_task'], 'method': 'casscf'}))
        self.assertEqual(code, 201, saved)
        self.assertEqual(saved['preparation_status'], 'active_space_probe')
        self.assertEqual(self.post('open', study_id=saved['study_id'])[1]['plan'], saved['plan'])
        self.assertEqual(self.executor.submit_count, 0)

    def test_invalid_preparation_saves_nothing_and_explicit_rebuild_keeps_both_studies(self):
        code, error = self.post('prepare', study_spec={'system_type': 'molecular', 'base_task': {'method': 'invented'}})
        self.assertEqual(code, 400, error)
        self.assertTrue(error['validation_issues'])
        self.assertEqual(list(self.root.iterdir()), [])
        first = self.post('prepare', study_spec=STUDY_SPEC)[1]
        second = self.post('prepare', study_spec=STUDY_SPEC)[1]
        self.assertNotEqual(first['study_id'], second['study_id'])
        self.assertEqual(self.post('open', study_id=first['study_id'])[1]['plan'], first['plan'])
        self.assertEqual(self.executor.submit_count, 0)

    def test_invalid_id_unknown_target_and_missing_study_do_not_submit(self):
        for route in ('open', 'start'):
            self.assertEqual(self.post(route, study_id='../outside')[0], 400)
            self.assertEqual(self.post(route, study_id='absent')[0], 404)
            self.assertEqual(self.post(route, study_id='absent', execution_target='unknown')[0], 400)
        self.assertEqual(self.executor.submit_count, 0)

    def test_web_page_uses_configured_root_and_lists_damaged_study(self):
        with urlopen(self.url + '/computational-study/') as response:
            html = response.read().decode()
        self.assertIn(str(self.root), html)
        damaged = self.root / 'broken-study'
        damaged.mkdir()
        (damaged / 'study-plan.json').write_text('{')
        self.assertEqual(self.post('list')[1]['studies'][0]['status'], 'unreadable')


if __name__ == '__main__':
    unittest.main()
