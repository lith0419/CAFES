"""Opt-in Slurm acceptance using tiny H2 jobs and client-side fault injection.

Run with PYTHONPATH=. python tools/verify_retry_lifecycle.py --help.
No injected fault changes scientific requests or the production executor.
"""
from __future__ import annotations

import argparse
import copy
import json
import shlex
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

from computational_study_agent.application import StudyApplicationService
from computational_study_agent.execution_receipts import StudyExecutionBusy, StudyExecutionInterrupted
from computational_study_agent.schema import StudyCase, StudyPlan
from pyscf_agent.executors import BatchHandle
from pyscf_agent.executors.remote_config import load_remote_profile
from pyscf_agent.executors.ssh_slurm import SshSlurmExecutor


class Campaign:
    def __init__(self, args):
        self.args = args
        self.root = Path(args.work_dir).expanduser().resolve()
        self.root.mkdir(parents=True, exist_ok=args.resume_initial or args.resume_after_retry_fix or args.resume_collection)
        self.profile = load_remote_profile(args.profile, args.remote_config)
        self.data = {'started_at': datetime.now(timezone.utc).isoformat(),
                     'fault_model': 'Real Slurm jobs; injected client exceptions after submit RPC or before waiting.',
                     'submissions': [], 'checks': [], 'status': 'running'}
        if args.resume_initial:
            self.data = json.loads(Path(args.output).read_text())
            if len(self.data['submissions']) != 1 or self.data['checks']:
                raise ValueError('Resume is limited to the completed initial batch before any checks')
            self.data['harness_correction'] = self.data.pop('error', None)
            self.data['status'] = 'running'
        if args.resume_after_retry_fix:
            self.data = json.loads(Path(args.output).read_text())
            if len(self.data['submissions']) != 3 or self.data['status'] != 'failed':
                raise ValueError('Expected the retained three-submission duplicate-retry failure')
            self.data.setdefault('findings', []).append({
                'issue': self.data.pop('error'),
                'failed_checks': [c for c in self.data['checks'] if not c['passed']],
                'fix': 'Reject concurrent Study mutations instead of queuing another explicit Run.',
            })
            self.data['checks'] = [c for c in self.data['checks'] if c['passed']]
            self.data['status'] = 'running'
        if args.resume_collection:
            self.data = json.loads(Path(args.output).read_text())
            if len(self.data['submissions']) != 5 or self.data['status'] != 'failed':
                raise ValueError('Expected the retained five-submission delayed-report failure')
            self.data.setdefault('findings', []).append({
                'issue': self.data.pop('error'),
                'failed_checks': [c for c in self.data['checks'] if not c['passed']],
                'fix': 'Keep completed jobs pending collection until TaskReport becomes visible.',
            })
            self.data['checks'] = [c for c in self.data['checks'] if c['passed']]
            self.data['status'] = 'running'
        self.persist()

    def persist(self):
        Path(self.args.output).write_text(json.dumps(self.data, ensure_ascii=False, indent=2) + '\n')

    def check(self, name, condition):
        self.data['checks'].append({'name': name, 'passed': bool(condition)})
        self.persist()
        if not condition:
            raise AssertionError(name)
        print('PASS ' + name, flush=True)

    def executor(self, *, lose_ack=False, interrupt_wait=False, cancel_first=False):
        campaign = self
        class ObservedExecutor(SshSlurmExecutor):
            def _rpc(self, operation, payload=None):
                if operation == 'submit-independent':
                    submitted = sum(len(s['tasks']) for s in campaign.data['submissions'])
                    if submitted + len(payload['tasks']) > 10:
                        raise ValueError('Acceptance campaign refuses to exceed ten submitted tasks')
                result = super()._rpc(operation, payload)
                if operation == 'submit-independent':
                    campaign.data['submissions'].append({
                        'tasks': [dict(task_id=t['task_id'], run_id=t['run_id']) for t in payload['tasks']],
                        'batches': result['batches'],
                    })
                    campaign.persist()  # Keep cleanup handles even when the client loses its acknowledgement.
                    if lose_ack:
                        raise OSError('Acceptance injection: lost SSH submit acknowledgement')
                return result

            def submit_independent_tasks(self, tasks, **kwargs):
                batches = super().submit_independent_tasks(tasks, **kwargs)
                if cancel_first:
                    self.cancel(batches[0].jobs[0])
                return batches

            def wait_for_independent_tasks(self, batches):
                if interrupt_wait:
                    raise OSError('Acceptance injection: client disconnected before waiting')
                return super().wait_for_independent_tasks(batches)
        return ObservedExecutor(self.profile)

    def plan(self, name, count=2):
        saved = self.root / name / 'study-plan.json'
        if saved.is_file():
            # Recovery retains the executed resource policy as well as case
            # inputs, rather than replacing it with a fresh unbound plan.
            return StudyPlan.from_dict(json.loads(saved.read_text()))
        plan = StudyPlan(study_id=name, name=name, objective='Retry lifecycle acceptance',
                         system_type='molecular', observables=['energy'], cases=[
            StudyCase(case_id='case-{0:04d}'.format(i + 1), label='H2 {0}'.format(i + 1),
                      variables={'bond_length': 0.70 + i * 0.04}, request={
                          'task_type': 'molecular', 'method': 'hf', 'basis': 'sto-3g',
                          'atom': 'H 0 0 0; H 0 0 {0:.2f}'.format(0.70 + i * 0.04)})
            for i in range(count)])
        (self.root / (name + '-plan.json')).write_text(json.dumps(plan.to_dict(), indent=2))
        return plan

    def interrupt(self, service, plan, **kwargs):
        try:
            service.run_study(plan, work_dir=str(self.root), **kwargs)
        except StudyExecutionInterrupted:
            return
        raise AssertionError('Expected the injected interruption')

    def wait(self, executor, submissions):
        jobs = [job for entry in submissions for raw in entry['batches']
                for job in BatchHandle.from_dict(raw).jobs]
        deadline = time.monotonic() + self.args.timeout
        last = None
        while time.monotonic() < deadline:
            statuses = executor.status_many(jobs)
            states = [status.state.value for status in statuses]
            if states != last:
                print('Slurm states: ' + ', '.join(states), flush=True)
                last = states
            if all(status.terminal for status in statuses):
                return
            time.sleep(3)
        raise TimeoutError('Slurm acceptance deadline exceeded')

    def run(self):
        normal = self.executor()
        self.data['runtime_identity'] = normal.remote_capabilities().get('runtime_identity')
        self.data.setdefault('runtime_identities', []).append(self.data['runtime_identity'])
        service = StudyApplicationService(task_executor=normal)
        if self.args.resume_collection:
            unknown = self.plan('lost-submit-ack', 1)
            recovered = self.collect_ready(lambda: service.collect_study(unknown, work_dir=str(self.root)))
            again = service.collect_study(unknown, work_dir=str(self.root))
            self.check('reconciliation and repeated Collect recover the original jobs',
                       recovered.status == again.status == 'succeeded' and
                       len(self.data['submissions']) == 5 and again.cases[0]['attempt_count'] == 1)
            self.run_followups(service, normal, 5)
            return
        mixed = self.plan('mixed-cancel-and-retry', 3)
        if not (self.args.resume_initial or self.args.resume_after_retry_fix):
            self.interrupt(StudyApplicationService(task_executor=self.executor(
                interrupt_wait=True, cancel_first=True)), mixed)
        self.wait(normal, self.data['submissions'])
        with ThreadPoolExecutor(max_workers=2) as pool:
            reports = list(pool.map(lambda _: self.collect_ready(
                lambda: service.collect_study(mixed, work_dir=str(self.root))), range(2)))
        first = reports[0]
        if not self.args.resume_after_retry_fix:
            self.check('mixed success and scheduler cancellation collected',
                       first.status == 'completed_with_issues' and
                       [c['task_report']['execution_status'] for c in first.cases] == ['failed', 'succeeded', 'succeeded'])
            self.check('concurrent Collect never submits or increments attempts',
                       len(self.data['submissions']) == 1 and all(
                           [c['attempt_count'] for c in r.cases] == [1, 1, 1] for r in reports))
        expected_submissions = len(self.data['submissions']) + 1
        expected_attempt = first.cases[0]['attempt_count'] + 1
        def retry():
            try:
                return service.run_study(mixed, work_dir=str(self.root), rerun_case_ids=['case-0001'])
            except StudyExecutionBusy:
                return None
        with ThreadPoolExecutor(max_workers=2) as pool:
            retries = list(pool.map(lambda _: retry(), range(2)))
        final = next(r for r in retries if r is not None)
        self.check('concurrent selected retry submits one fresh attempt',
                   len(self.data['submissions']) == expected_submissions and retries.count(None) == 1 and
                   [t['task_id'] for t in self.data['submissions'][-1]['tasks']] == ['case-0001'] and
                   [c['attempt_count'] for c in final.cases] == [expected_attempt, 1, 1] and final.status == 'succeeded')
        self.check('successful cases and earlier failed report remain distinct',
                   first.cases[1:][0]['task_report'] == final.cases[1:][0]['task_report'] and
                   first.cases[2]['task_report'] == final.cases[2]['task_report'] and
                   first.cases[0]['task_report']['run_id'] != final.cases[0]['task_report']['run_id'])
        service.collect_study(mixed, work_dir=str(self.root))
        self.check('Collect after retry remains submission free', len(self.data['submissions']) == expected_submissions)

        unknown = self.plan('lost-submit-ack', 1)
        self.interrupt(StudyApplicationService(task_executor=self.executor(lose_ack=True)), unknown)
        expected_submissions += 1
        unknown_submission = copy.deepcopy(self.data['submissions'][-1])
        self.check('lost acknowledgement remains submission_unknown',
                   service.inspect_execution(unknown.study_id, work_dir=str(self.root))['status'] == 'submission_unknown')
        try:
            service.run_study(unknown, work_dir=str(self.root))
        except ValueError:
            pass
        else:
            raise AssertionError('Unknown submission must block another Run')
        self.check('Run cannot duplicate an unknown submission', len(self.data['submissions']) == expected_submissions)
        evidence_path = str(Path(unknown_submission['batches'][0]['jobs'][0]['work_dir']).parent / 'submission.json')
        for _ in range(2):
            service.reconcile_execution(unknown, work_dir=str(self.root), submission_path=evidence_path)
        self.wait(normal, [unknown_submission])
        for _ in range(2):
            recovered = self.collect_ready(lambda: service.collect_study(unknown, work_dir=str(self.root)))
        self.check('reconciliation and repeated Collect recover the original jobs',
                   recovered.status == 'succeeded' and len(self.data['submissions']) == expected_submissions and
                   [c['attempt_count'] for c in recovered.cases] == [1])

        self.run_followups(service, normal, expected_submissions)

    def collect_ready(self, operation):
        deadline = time.monotonic() + self.args.timeout
        while True:
            try:
                return operation()
            except StudyExecutionInterrupted:
                if time.monotonic() >= deadline:
                    raise
                print('Waiting for existing TaskReports to become visible; no submission.', flush=True)
                time.sleep(3)

    def run_followups(self, service, normal, expected_submissions):
        sequential = self.plan('sequential-reconnect', 2)
        self.interrupt(StudyApplicationService(task_executor=self.executor(interrupt_wait=True)),
                       sequential, batch_independent=False)
        expected_submissions += 1
        self.wait(normal, [self.data['submissions'][-1]])
        partial = self.collect_ready(lambda: service.collect_study(sequential, work_dir=str(self.root)))
        self.check('sequential Collect does not start the next case', len(self.data['submissions']) == expected_submissions and
                   [c['attempt_count'] for c in partial.cases] == [1, 0])
        finished = service.run_study(sequential, work_dir=str(self.root), batch_independent=False)
        expected_submissions += 1
        self.check('explicit Run resumes only the pending sequential case',
                   len(self.data['submissions']) == expected_submissions and finished.status == 'succeeded' and
                   self.data['submissions'][-1]['tasks'][0]['task_id'] == 'case-0002')

        adaptive_spec = {'name': 'adaptive-reconnect', 'objective': 'collect diagnostics',
                         'system_type': 'molecular', 'base_task': {
                             'task_type': 'molecular', 'atom': 'H 0 0 0; H 0 0 0.74',
                             'basis': 'sto-3g', 'method': 'mp2'}, 'observables': ['energy']}
        try:
            StudyApplicationService(task_executor=self.executor(interrupt_wait=True)).run_adaptive_study(
                adaptive_spec, work_dir=str(self.root), requested_study_id='adaptive-reconnect')
        except StudyExecutionInterrupted:
            pass
        else:
            raise AssertionError('Expected interrupted adaptive initial scan')
        before = len(self.data['submissions'])
        self.wait(normal, [self.data['submissions'][-1]])
        adaptive = self.collect_ready(lambda: service.collect_adaptive_study(
            adaptive_spec, study_id='adaptive-reconnect', work_dir=str(self.root)))
        self.check('adaptive Collect prepares refinement without submitting it',
                   len(self.data['submissions']) == before and adaptive['adaptive']['refined_report'] is None and
                   adaptive['adaptive']['execution_pending_plan_kind'] == 'refined')
        ids = [raw['backend_job_id'] for s in self.data['submissions'] for raw in s['batches']]
        command = shlex.join(['sacct', '-j', ','.join(ids), '--format=JobIDRaw,State,ExitCode', '--parsable2', '--noheader'])
        accounting = subprocess.run(normal._ssh_command('capabilities')[:-1] + [command],
                                    capture_output=True, text=True, check=True)
        self.data['slurm_accounting'] = accounting.stdout.strip().splitlines()
        self.data['status'] = 'passed'
        self.data['completed_at'] = datetime.now(timezone.utc).isoformat()
        self.persist()

    def cleanup(self):
        executor = self.executor()
        for submission in self.data['submissions']:
            for raw in submission['batches']:
                for job in BatchHandle.from_dict(raw).jobs:
                    try:
                        if not executor.status(job).terminal:
                            executor.cancel(job)
                    except Exception as exc:
                        self.data.setdefault('cleanup_errors', []).append(str(exc))
        self.persist()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--remote-config', required=True)
    parser.add_argument('--profile', default='amarel')
    parser.add_argument('--work-dir', required=True, help='New, isolated local acceptance directory')
    parser.add_argument('--output', required=True)
    parser.add_argument('--timeout', type=float, default=600)
    parser.add_argument('--resume-initial', action='store_true', help='Reuse the initial persisted batch after a harness failure')
    parser.add_argument('--resume-after-retry-fix', action='store_true', help='Retain the duplicate-retry finding and verify its fix')
    parser.add_argument('--resume-collection', action='store_true', help='Continue after restoring the delayed-report acceptance checkpoint')
    parser.add_argument('--submit', action='store_true', required=True, help='Authorize up to 10 tiny Slurm tasks')
    args = parser.parse_args()
    campaign = Campaign(args)
    try:
        campaign.run()
    except BaseException as exc:
        campaign.data.update(status='failed', error='{0}: {1}'.format(type(exc).__name__, exc))
        campaign.cleanup()
        raise


if __name__ == '__main__':
    main()
