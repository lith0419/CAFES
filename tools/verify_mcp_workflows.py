"""Opt-in real MCP acceptance: ten tiny H2 Task runs, isolated output only.

Uses the installed SDK and production MCP server. The only injected failure is
terminating this harness's own Study coordinator after its two handles are saved.
The low-cycle DFT case produces a real unconverged result before reviewed retry.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.metadata
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from mcp import Client, StdioServerParameters


H2 = {'atom': 'H 0 0 0; H 0 0 1.2', 'basis': 'def2-svp', 'method': 'dft', 'xc': 'b3lyp',
      'runtime': {'conv_tol': 1e-12}, 'outputs': ['energy']}
DATASET = {'dataset_id': 'mcp-h2-md', 'name': 'MCP H2 MD acceptance', 'target_molecule_count': 1,
           'geometries_per_molecule': 2,
           'molecular_dynamics': {'steps': 2, 'sample_stride': 1, 'sample_offset': 0, 'time_step_au': 1.0}}
GEOMETRY = {'molecule_id': 'h2', 'geometry_id': 'seed', 'atomic_numbers': [1, 1],
            'positions': [[0, 0, 0], [0, 0, 0.74]]}


class Campaign:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.work_dir).resolve()
        self.root.mkdir(parents=True, exist_ok=False)
        self.output = Path(args.output).resolve()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        command_args = ['-m', 'pyscf_agent.mcp_server', '--work-dir', str(self.root), '--executor', args.executor]
        if args.executor == 'remote':
            command_args += ['--remote', args.profile, '--remote-config', str(Path(args.remote_config).resolve())]
        self.parameters = StdioServerParameters(command=sys.executable, args=command_args,
            cwd=str(Path(args.source_root).resolve()), env=dict(os.environ, OMP_NUM_THREADS='1',
            OPENBLAS_NUM_THREADS='1', MKL_NUM_THREADS='1', MPLCONFIGDIR='/private/tmp/pyscf-agent-mcp-matplotlib'))
        self.data = {'status': 'running', 'executor': args.executor, 'studies': {}, 'checks': [],
                     'planned_task_runs': 0, 'versions': {name: importlib.metadata.version(name) for name in ('mcp', 'pyscf')}}
        self.persist()

    def persist(self):
        self.output.write_text(json.dumps(self.data, indent=2, ensure_ascii=False) + '\n')

    def check(self, name, condition):
        self.data['checks'].append({'name': name, 'passed': bool(condition)})
        self.persist()
        if not condition:
            raise AssertionError(name)
        print('PASS ' + name, flush=True)

    async def call(self, name, **arguments):
        # Each call uses a new protocol server: no client memory can own execution.
        async with Client(self.parameters) as client:
            result = await client.call_tool(name, arguments)
        if result.is_error:
            raise RuntimeError(str(result.content))
        return result.structured_content

    async def prepare(self, name, **kwargs):
        result = await self.call('prepare_study', **kwargs)
        self.data['studies'][name] = {'study_id': result['study_id']}
        self.persist()
        return result['study_id']

    async def start(self, study_id, expected_runs):
        count = self.data['planned_task_runs'] + expected_runs
        if count > 10:
            raise ValueError('Campaign refuses more than ten planned Task runs')
        self.data['planned_task_runs'] = count
        self.persist()
        try:
            return await self.call('submit_study', study_id=study_id)
        except RuntimeError as exc:
            if 'CostApprovalRequired' not in str(exc):
                raise
            # This opt-in campaign authorizes only the bounded H2 requests above.
            # Changed retry inputs may require a fresh cost review.
            await self.call('review_study', study_id=study_id, action_id='approve_cost_estimate')
            return await self.call('submit_study', study_id=study_id)

    async def collect(self, study_id):
        deadline = time.monotonic() + self.args.timeout
        while time.monotonic() < deadline:
            status = await self.call('get_study_status', study_id=study_id)
            if status['can_collect']:
                return (await self.call('collect_study', study_id=study_id, include_report=True))['report']
            error = (status.get('agent') or {}).get('error')
            if error and not status.get('receipt_count'):
                raise RuntimeError(json.dumps(status))
            await asyncio.sleep(10 if self.args.executor == 'remote' else 0.3)
        raise TimeoutError(json.dumps(status))

    def runs(self, study_id):
        runs = set()
        for path in (self.root / study_id).rglob('execution-receipt.json'):
            receipt = json.loads(path.read_text())
            handles = list((receipt.get('jobs') or {}).values())
            for batch in receipt.get('batches') or []:
                handles.extend(batch.get('jobs') or [])
            # Run IDs are scoped to their executor directory (adaptive stages
            # can both use case-0001); count persisted handles, not bare IDs.
            runs.update((handle['work_dir'], handle['run_id']) for handle in handles)
        return runs

    async def interrupt_coordinator(self, study_id, pid):
        deadline = time.monotonic() + 120
        while len(self.runs(study_id)) != 2 and time.monotonic() < deadline:
            await asyncio.sleep(0.2)
        self.check('two_handles_saved_before_interruption', len(self.runs(study_id)) == 2)
        result = subprocess.run(['ps', '-p', str(pid), '-o', 'args='], capture_output=True, text=True, check=True)
        if 'computational_study_agent.application.background' not in result.stdout or study_id not in result.stdout:
            raise ValueError('Refusing to signal a process outside this acceptance Study')
        os.kill(pid, signal.SIGTERM)
        for _ in range(100):
            invocation = json.loads((self.root / study_id / 'study-invocation.json').read_text())
            status = await self.call('get_study_status', study_id=study_id)
            if not (status.get('agent') or {}).get('running'):
                break
            await asyncio.sleep(0.1)
        self.check('reconnected_client_detects_coordinator_exit', status['agent']['interrupted'])

    async def run(self):
        study_id = await self.prepare('retry', study_spec={
            'name': 'MCP mixed H2 convergence', 'system_type': 'molecular', 'base_task': H2,
            'case_design': {'mode': 'cases', 'cases': [
                {'label': 'bounded failure', 'request_updates': {'runtime': {'max_cycle': 1, 'conv_tol': 1e-12}}},
                {'label': 'normal reference', 'request_updates': {'runtime': {'max_cycle': 50, 'conv_tol': 1e-12}}}]},
            'observables': ['energy'], 'resource_policy': {'review_work_estimates': True, 'total_work_review_threshold': 1}})
        try:
            await self.call('submit_study', study_id=study_id)
        except RuntimeError as exc:
            self.check('initial_cost_gate', 'CostApprovalRequired' in str(exc))
        else:
            raise AssertionError('Expected initial cost approval')
        await self.call('review_study', study_id=study_id, action_id='approve_cost_estimate', plan_kind='static')
        self.check('cost_review_saves_without_submission', not self.runs(study_id))
        started = await self.start(study_id, 2)
        if self.args.executor == 'remote':
            await self.interrupt_coordinator(study_id, started['pid'])
        initial = await self.collect(study_id)
        statuses = [case['task_report']['execution_status'] for case in initial['cases']]
        self.data['studies']['retry']['initial_statuses'] = statuses
        self.check('mixed_real_convergence_results', statuses == ['unconverged', 'succeeded'])
        self.check('recovery_collect_does_not_resubmit', len(self.runs(study_id)) == 2)
        await self.call('review_study', study_id=study_id, action_id='increase_recovery_max_cycle',
                        case_ids=[initial['cases'][0]['case_id']], plan_kind='static')
        pending = await self.call('get_study_status', study_id=study_id)
        self.check('review_survives_protocol_restart', pending['pending_review']['can_run'])
        await self.start(study_id, 1)
        final = await self.collect(study_id)
        self.check('selected_retry_succeeds', final['status'] == 'succeeded')
        self.check('unselected_result_is_unchanged', final['cases'][1] == initial['cases'][1])
        self.check('three_runs_and_attempts_2_1', len(self.runs(study_id)) == 3 and
                   [case['attempt_count'] for case in final['cases']] == [2, 1])
        analysis = await self.call('analyze_study', study_id=study_id, postprocess=True, include_report=True)
        self.check('analysis_and_plots_without_llm_or_new_tasks', 'scan_path_diagnostics' in analysis and
                   'postprocessing' in analysis and len(self.runs(study_id)) == 3)
        self.data['studies']['retry']['report'] = str(self.root / study_id / 'study-report.json')

        adaptive_id = await self.prepare('adaptive', study_spec={
            'name': 'MCP adaptive H2', 'system_type': 'molecular', 'study_mode': 'adaptive',
            'base_task': {'atom': 'H 0 0 0; H 0 0 0.74', 'method': 'hf', 'basis': 'sto-3g'},
            'sweep': {'basis': ['sto-3g', '6-31g']}, 'observables': ['energy']})
        await self.start(adaptive_id, 4)
        adaptive = await self.collect(adaptive_id)
        self.check('adaptive_completes_after_one_start', adaptive['status'] == 'succeeded' and len(self.runs(adaptive_id)) == 4)

        dataset = await self.call('prepare_dataset', dataset_spec=DATASET, seed_geometries=[GEOMETRY])
        dataset_id = dataset['study_id']
        self.data['studies']['dataset'] = {'study_id': dataset_id}
        self.persist()
        await self.start(dataset_id, 1)
        report = await self.collect(dataset_id)
        manifest = report['dataset_manifest']
        self.check('dataset_has_two_accepted_frames', manifest['accepted_structure_count'] == 2 and
                   manifest['rejected_structure_count'] == 0 and len(self.runs(dataset_id)) == 1)
        async with Client(self.parameters) as client:
            resource = await client.read_resource(dataset['dataset_uri'])
            self.check('dataset_resource_matches_report', json.loads(resource.contents[0].text) == manifest)
        self.data['studies']['dataset']['manifest'] = manifest

        cas_id = await self.prepare('casscf', study_spec={
            'name': 'MCP CASSCF approval', 'system_type': 'molecular',
            'base_task': {'atom': 'H 0 0 0; H 0 0 0.74', 'method': 'casscf', 'basis': 'sto-3g'}, 'observables': ['energy']})
        await self.start(cas_id, 1)
        probe = await self.collect(cas_id)
        self.check('probe_publishes_review', probe['status'] == 'pending_review')
        await self.call('review_study', study_id=cas_id, action_id='approve_active_space',
                        plan_kind='direct_casscf_review', case_ids=[case['case_id'] for case in probe['cases']])
        await self.start(cas_id, 1)
        casscf = await self.collect(cas_id)
        self.check('approved_casscf_uses_same_study', casscf['study_id'] == cas_id and casscf['status'] == 'succeeded'
                   and casscf['cases'][0]['request']['method'] == 'casscf' and len(self.runs(cas_id)) == 2)
        await self.call('collect_study', study_id=cas_id)
        actual = sum(len(self.runs(value['study_id'])) for value in self.data['studies'].values())
        self.check('ten_task_runs_total', actual == 10)
        self.data.update(status='passed', actual_task_runs=actual)
        self.persist()


async def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--executor', choices=('local', 'remote'), default='local')
    parser.add_argument('--source-root', default='.')
    parser.add_argument('--remote-config')
    parser.add_argument('--profile', default='amarel')
    parser.add_argument('--work-dir', required=True)
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=float, default=900)
    args = parser.parse_args()
    campaign = Campaign(args)
    try:
        await campaign.run()
    except BaseException as exc:
        campaign.data.update(status='failed', error=str(exc))
        campaign.persist()
        raise


if __name__ == '__main__':
    asyncio.run(main())
