from __future__ import annotations

import asyncio
import json
import os
import tempfile
import time
import unittest
from pathlib import Path

from computational_study_agent.application import StudyApplicationService
from pyscf_agent.application import CalculationApplicationService
from tests.computational_study_agent.test_retry_collection import Remote
from tests.computational_study_agent.test_study_background import inline_launcher
from tests.pyscf_agent.test_mcp_server import H2, HAS_MCP, HAS_PYSCF, with_client
from tests.pyscf_agent import test_mcp_server as task_tests


STUDY_SPEC = {'name': 'H2 basis comparison', 'objective': 'compare_results',
              'system_type': 'molecular', 'base_task': H2,
              'sweep': {'basis': ['sto-3g', '6-31g']}, 'observables': ['energy']}


@unittest.skipUnless(HAS_MCP, 'optional MCP SDK is not installed')
class MCPStudyTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        from pyscf_agent.mcp_server.server import create_server
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.executor = Remote(fail_first=False)
        self.service = StudyApplicationService(task_executor=self.executor, execution_config={'execution_target': 'remote'})
        self.service._study_launcher = inline_launcher(self.service)
        self.server = create_server(CalculationApplicationService(task_executor=self.executor),
                                    study_service=self.service, work_dir=self.root,
                                    executor_description=self.executor.describe())

    async def call(self, name, **arguments):
        result = await self.client.call_tool(name, arguments)
        self.assertFalse(result.is_error, result.content)
        return result.structured_content

    @with_client
    async def test_dmet_continuation_tools_delegate_to_saved_study_lifecycle(self):
        from computational_study_agent.schema import StudyCase
        from tests.computational_study_agent.test_dmet_branches import BranchRemote, record
        from tests.computational_study_agent.test_retry_collection import plan_for_service

        remote = BranchRemote()
        self.service._task_executor = remote
        plan = plan_for_service(2)
        plan.cases = [StudyCase.from_dict(record(f'case-{i+1:04d}', i)) for i in range(2)]
        self.service.run_study(plan, work_dir=str(self.root))
        preview = await self.call('prepare_dmet_continuation', study_id=plan.study_id)
        self.assertEqual(preview['max_runs'], 4)
        self.assertEqual(len(remote.submissions), 1)
        started = await self.call('start_dmet_continuation', study_id=plan.study_id, action_id=preview['action_id'])
        self.assertTrue(started['started'])
        self.assertEqual(len(remote.submissions), 3)
        analyzed = await self.call('analyze_dmet_branches', study_id=plan.study_id)
        self.assertEqual(len(analyzed['analysis']['points']), 2)
        self.assertEqual(analyzed['analysis']['transfer'], 'mean_field_density_only')

    @with_client
    async def test_prepare_submit_collect_and_review_roundtrip(self):
        prepared = await self.call('prepare_study', study_spec=STUDY_SPEC)
        study_id = prepared['study_id']
        resource = await self.client.read_resource(prepared['plan_uri'])
        self.assertEqual(json.loads(resource.contents[0].text), prepared['plan'])
        status = await self.call('get_study_status', study_id=study_id)
        self.assertEqual(status['status'], 'prepared')
        self.assertEqual(status['task_status_counts'], {'not_executed': 2})
        self.assertEqual(self.executor.submit_count, 0)
        submission = await self.call('submit_study', study_id=study_id)
        self.assertTrue(submission['started'])
        repeated = await self.call('submit_study', study_id=study_id)
        self.assertTrue(repeated['started'])
        self.assertEqual(self.executor.submit_count, 1)
        collected = await self.call('collect_study', study_id=study_id, include_report=True)
        self.assertEqual(collected['report']['status'], 'succeeded')
        self.assertNotIn('schema', collected['summary'])
        reviewed = await self.call('review_study', study_id=study_id, action_id='increase_recovery_max_cycle',
                                   case_ids=[prepared['plan']['cases'][0]['case_id']], plan_kind='static')
        self.assertTrue(reviewed['saved'])
        self.assertEqual(reviewed['case_ids'], [prepared['plan']['cases'][0]['case_id']])
        self.assertEqual(self.executor.submit_count, 1)
        await self.call('submit_study', study_id=study_id)
        report = (await self.call('collect_study', study_id=study_id, include_report=True))['report']
        self.assertEqual([case['attempt_count'] for case in report['cases']], [2, 1])
        self.assertEqual(report['cases'][1], collected['report']['cases'][1])
        self.assertEqual(self.executor.submit_count, 2)

    @with_client
    async def test_cost_gate_is_an_actionable_error_and_review_does_not_execute(self):
        spec = dict(STUDY_SPEC, resource_policy={'review_work_estimates': True, 'total_work_review_threshold': 1})
        prepared = await self.call('prepare_study', study_spec=spec)
        study_id = prepared['study_id']
        blocked = await self.client.call_tool('submit_study', {'study_id': study_id})
        self.assertTrue(blocked.is_error)
        self.assertIn('CostApprovalRequired', str(blocked.content))
        self.assertEqual(self.executor.submit_count, 0)
        review = await self.call('review_study', study_id=study_id, action_id='approve_cost_estimate', plan_kind='static')
        self.assertEqual(self.executor.submit_count, 0)
        await self.call('submit_study', study_id=study_id)
        self.assertEqual(self.executor.submit_count, 1)

    @with_client
    async def test_submit_uses_only_study_id_and_bad_ids_are_rejected(self):
        prepared = await self.call('prepare_study', study_spec=STUDY_SPEC)
        tools = (await self.client.list_tools()).tools
        submission = next(tool for tool in tools if tool.name == 'submit_study')
        self.assertNotIn('prepared_plan', submission.input_schema['properties'])
        for tool in ('submit_study', 'get_study_status', 'collect_study'):
            result = await self.client.call_tool(tool, {'study_id': '../elsewhere'})
            self.assertTrue(result.is_error)
        self.assertEqual(self.executor.submit_count, 0)

    @with_client
    async def test_analysis_reuses_agent_diagnostics_and_never_submits(self):
        prepared = await self.call('prepare_study', study_spec=STUDY_SPEC)
        await self.call('submit_study', study_id=prepared['study_id'])
        before = self.executor.submit_count
        result = await self.call('analyze_study', study_id=prepared['study_id'], include_report=True, postprocess=True)
        self.assertIn('scan_path_diagnostics', result)
        self.assertIn('plot_specs', result)
        self.assertIn('postprocessing', result['report'])
        self.assertNotIn('result_analysis', result)
        saved = self.service.load_report(prepared['study_id'], work_dir=str(self.root))
        self.assertEqual(saved['scan_path_diagnostics'], result['scan_path_diagnostics'])
        self.assertEqual(self.executor.submit_count, before)

    @with_client
    async def test_invalid_spec_creates_no_study(self):
        for spec in ({'base_task': {'method': 'invented'}},):
            result = await self.client.call_tool('prepare_study', {'study_spec': spec})
            self.assertTrue(result.is_error)
        self.assertEqual(list(self.root.iterdir()), [])


@unittest.skipUnless(HAS_MCP and HAS_PYSCF and os.environ.get('PYSCF_AGENT_TEST_MCP_NUMERICAL') == '1',
                     'set PYSCF_AGENT_TEST_MCP_NUMERICAL=1 for real stdio/PySCF Study acceptance')
class MCPStudyNumericalTests(unittest.IsolatedAsyncioTestCase):
    parameters = task_tests.MCPStdioTests.parameters

    async def call(self, client, name, **arguments):
        result = await client.call_tool(name, arguments)
        self.assertFalse(result.is_error, result.content)
        return result.structured_content

    async def collect_when_ready(self, client, study_id):
        deadline = time.monotonic() + 90
        while True:
            status = await self.call(client, 'get_study_status', study_id=study_id)
            if status['can_collect']:
                return (await self.call(client, 'collect_study', study_id=study_id, include_report=True))['report']
            self.assertLess(time.monotonic(), deadline, status)
            await asyncio.sleep(0.2)

    async def test_one_start_finishes_h2_study_after_client_exit_and_subset_retry(self):
        from mcp import Client
        with tempfile.TemporaryDirectory() as root:
            async with Client(self.parameters(root)) as client:
                prepared = await self.call(client, 'prepare_study', study_spec=STUDY_SPEC)
                study_id = prepared['study_id']
                submitted = await self.call(client, 'submit_study', study_id=study_id)
                self.assertTrue(submitted['started'])
            async with Client(self.parameters(root)) as client:
                completed = await self.collect_when_ready(client, study_id)
                self.assertEqual(completed['status'], 'succeeded')
                self.assertEqual(len(list(Path(root).rglob('job-state.json'))), 2)
                energy = completed['cases'][0]['task_report']['compact_results']['energy']
                self.assertAlmostEqual(energy, -1.1167593074, places=7)
                resource = await client.read_resource(prepared['report_uri'])
                self.assertEqual(json.loads(resource.contents[0].text)['cases'], completed['cases'])
                review = await self.call(client, 'review_study', study_id=study_id,
                                          action_id='increase_recovery_max_cycle',
                                          case_ids=[completed['cases'][0]['case_id']], plan_kind='static')
            async with Client(self.parameters(root)) as client:
                await self.call(client, 'submit_study', study_id=study_id)
                final = await self.collect_when_ready(client, study_id)
                self.assertEqual(final['status'], 'succeeded')
                self.assertNotIn('pending_review', final)
                self.assertEqual([case['attempt_count'] for case in final['cases']], [2, 1])
                self.assertEqual(final['cases'][1], completed['cases'][1])
                self.assertNotEqual(final['cases'][0]['execution']['run_id'], completed['cases'][0]['execution']['run_id'])
                await self.call(client, 'collect_study', study_id=study_id)
                self.assertEqual(len(list(Path(root).rglob('job-state.json'))), 3)
                report_path = Path(root) / study_id / 'study-report.json'
                self.assertGreater(report_path.stat().st_mtime_ns,
                                   max(path.stat().st_mtime_ns for path in (Path(root) / study_id / 'cases').rglob('job-task-report.json')))
                print(json.dumps({'acceptance': 'mcp_agent_study_single_start_restart_retry_h2',
                                  'energy_hartree': energy, 'cases': 2, 'runs': 3, 'attempts': [2, 1]}))

    async def test_direct_casscf_probe_and_review_use_the_same_agent_service(self):
        from mcp import Client
        with tempfile.TemporaryDirectory() as root:
            spec = dict(STUDY_SPEC, base_task=dict(H2, method='casscf'), sweep={'basis': ['sto-3g']})
            async with Client(self.parameters(root)) as client:
                prepared = await self.call(client, 'prepare_study', study_spec=spec)
                self.assertEqual(prepared['status'], 'active_space_probe')
                study_id = prepared['study_id']
                await self.call(client, 'submit_study', study_id=study_id)
            async with Client(self.parameters(root)) as client:
                report = await self.collect_when_ready(client, study_id)
                self.assertEqual(report['status'], 'pending_review')
                self.assertEqual(report['adaptive']['mode'], 'direct_casscf_review')
                self.assertEqual(report['workflow']['stage'], 'active_space_review_required')
                review = await self.call(client, 'review_study', study_id=study_id,
                    action_id='approve_active_space', plan_kind='direct_casscf_review',
                    case_ids=[case['case_id'] for case in report['cases']])
            async with Client(self.parameters(root)) as client:
                await self.call(client, 'submit_study', study_id=study_id)
                final = await self.collect_when_ready(client, study_id)
                self.assertEqual(final['status'], 'succeeded')
                self.assertNotIn('pending_review', final)
                self.assertEqual(final['cases'][0]['request']['method'], 'casscf')
                self.assertEqual(final['lifecycle']['entity_id'], study_id)
                self.assertEqual(final['cases'][0]['attempt_count'], 2)
                again = await self.call(client, 'collect_study', study_id=study_id, include_report=True)
                self.assertEqual(again['report']['status'], 'succeeded')
                self.assertEqual(len(list(Path(root).rglob('job-state.json'))), 2)

    async def test_dataset_uses_existing_md_runner_and_manifest_after_restart(self):
        from mcp import Client
        with tempfile.TemporaryDirectory() as root:
            async with Client(self.parameters(root)) as client:
                prepared = await self.call(client, 'prepare_dataset', dataset_spec={
                    'dataset_id': 'h2-smoke', 'name': 'H2 MD smoke', 'target_molecule_count': 1,
                    'geometries_per_molecule': 2,
                    'molecular_dynamics': {'steps': 2, 'sample_stride': 1, 'sample_offset': 0, 'time_step_au': 1.0}},
                    seed_geometries=[{'molecule_id': 'h2', 'geometry_id': 'seed', 'atomic_numbers': [1, 1],
                                      'positions': [[0, 0, 0], [0, 0, 0.74]]}])
                study_id = prepared['study_id']
                self.assertFalse(list(Path(root).rglob('job-state.json')))
                await self.call(client, 'submit_study', study_id=study_id)
            async with Client(self.parameters(root)) as client:
                report = await self.collect_when_ready(client, study_id)
                self.assertEqual(report['status'], 'succeeded', [(c['task_report'].get('errors'), c['task_report'].get('analysis_summary')) for c in report['cases']])
                manifest = report['dataset_manifest']
                self.assertEqual(manifest['accepted_structure_count'], 2)
                self.assertEqual(manifest['rejected_structure_count'], 0)
                resource = await client.read_resource(prepared['dataset_uri'])
                self.assertEqual(json.loads(resource.contents[0].text), manifest)
                await self.call(client, 'collect_study', study_id=study_id)
                self.assertEqual(len(list(Path(root).rglob('job-state.json'))), 1)

    async def test_agent_runs_adaptive_stages_after_one_start_and_client_exit(self):
        from mcp import Client
        with tempfile.TemporaryDirectory() as root:
            async with Client(self.parameters(root)) as client:
                prepared = await self.call(client, 'prepare_study', study_spec=dict(STUDY_SPEC, study_mode='adaptive'))
                self.assertEqual(prepared['mode'], 'adaptive')
                self.assertFalse(list(Path(root).rglob('job-state.json')))
                study_id = prepared['study_id']
                await self.call(client, 'submit_study', study_id=study_id)
            async with Client(self.parameters(root)) as client:
                report = await self.collect_when_ready(client, study_id)
                self.assertEqual(report['study_id'], study_id)
                self.assertEqual(report['status'], 'succeeded', report.get('summary'))
                self.assertEqual(len(report['adaptive']['initial_scan_decisions']), 2)
                self.assertEqual(len(report['adaptive']['refined_report']['cases']), 2)
                states = list(Path(root).rglob('job-state.json'))
                self.assertGreaterEqual(len(states), 4)
                await self.call(client, 'collect_study', study_id=study_id)
                self.assertEqual(len(list(Path(root).rglob('job-state.json'))), len(states))
                print(json.dumps({'acceptance': 'mcp_agent_adaptive_single_start',
                                  'study_id': study_id, 'cases': 2, 'runs': len(states), 'status': report['status']}))


if __name__ == '__main__':
    unittest.main()
