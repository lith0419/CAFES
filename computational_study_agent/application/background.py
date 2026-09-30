"""Run one existing Study application call outside the lifetime of its client.

This module owns process launch only. Scientific execution and recovery stay in
StudyApplicationService.run_study/run_adaptive_study and their existing runners.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess
import sys

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.executors.factory import create_task_executor
from pyscf_agent.executors import LocalProcessExecutor

from ..locking import LOCK_FILENAME as LOCK_FILENAME, study_lock, study_is_locked, adopt_study_lock
from ..study_state import read_json


INVOCATION_FILENAME = 'study-invocation.json'


def _now():
    return datetime.now(timezone.utc).isoformat()


def _write(directory, invocation):
    default_artifact_repository().write_json(
        Path(directory) / INVOCATION_FILENAME, invocation, kind='study-invocation', atomic=True,
    )


@contextmanager
def invocation_lock(directory):
    with study_lock(directory, reentrant=False, create=False) as stream:
        yield stream


def inspect_invocation(directory):
    directory = Path(directory)
    path = directory / INVOCATION_FILENAME
    if not path.is_file():
        return None
    running = study_is_locked(directory)
    invocation = read_json(path)
    return {'running': running, 'started_at': invocation.get('started_at'),
            'finished_at': invocation.get('finished_at'), 'error': invocation.get('error'),
            'interrupted': not running and not invocation.get('finished_at'),
            'mode': invocation['request']['mode'],
            'cancelled': invocation.get('cancelled', False),
            'stop_requested_at': invocation.get('stop_requested_at')}


def launch_study(directory, request, execution_config, *, popen_factory=subprocess.Popen, lock=None):
    """Persist a call and launch it once. No polling or per-task submission loop."""
    directory = Path(directory).resolve()
    with (invocation_lock(directory) if lock is None else nullcontext(lock)) as lock:
        invocation = {'schema': 'pyscf-agent.study-invocation.v1',
                      'study_id': directory.name, 'request': request,
                      'execution_config': execution_config,
                      'started_at': _now(), 'finished_at': None, 'error': None}
        _write(directory, invocation)
        try:
            with (directory / 'study-worker.log').open('ab') as log:
                process = popen_factory([
                    sys.executable, '-m', 'computational_study_agent.application.background',
                    '--study-directory', str(directory), '--lock-fd', str(lock.fileno()),
                ], pass_fds=(lock.fileno(),), stdin=subprocess.DEVNULL,
                    stdout=log, stderr=log, start_new_session=True)
        except OSError as exc:
            invocation.update(finished_at=_now(), error={'code': type(exc).__name__, 'message': str(exc)})
            _write(directory, invocation)
            raise
        # Retain no numerical data or task state in the protocol process.
        # Popen's standard child cleanup reaps a completed launcher on later starts.
        return {'study_id': directory.name, 'started': True, 'running': True,
                'pid': process.pid, 'mode': request['mode']}


def run_invocation(directory, *, service=None):
    with study_lock(directory):
        return _run_invocation(directory, service=service)


def _run_invocation(directory, *, service=None):
    directory = Path(directory).resolve()
    invocation = read_json(directory / INVOCATION_FILENAME)
    try:
        if service is None:
            from .service import StudyApplicationService
            config = invocation['execution_config']
            executor = (LocalProcessExecutor(wall_time_seconds=config.get('local_wall_time_seconds'))
                        if config.get('execution_target', 'local') == 'local' else create_task_executor(**config))
            service = StudyApplicationService(task_executor=executor)
        request = invocation['request']
        if request['mode'] == 'analysis':
            service._analyze_saved_results(directory.name, work_dir=str(directory.parent),
                                           review=request['review'], **request['kwargs'])
        elif request['mode'] == 'adaptive':
            if request.get('initial_cost_approved'):
                from ..adaptive.executor import save_adaptive_cost_approval
                save_adaptive_cost_approval(directory, request['spec'], request['options'])
            service.run_adaptive_study(request['spec'], options=request['options'],
                                       work_dir=str(directory.parent), resume_study_id=directory.name,
                                       **request['kwargs'])
        elif request['mode'] == 'retry':
            from .retries import run_retry_invocation
            run_retry_invocation(service, directory, request)
        elif request['mode'] == 'dmet_continuation':
            from .dmet_continuation import run_continuation_invocation
            run_continuation_invocation(service, directory, request)
        else:
            service.run_study(request['plan'], work_dir=str(directory.parent), **request['kwargs'])
        if request.get('consume_review'):
            service.complete_saved_review(directory.name, work_dir=str(directory.parent))
    except Exception as exc:
        error = {'code': type(exc).__name__, 'message': str(exc)}
        for name in ('estimate', 'receipt'):
            if hasattr(exc, name):
                error[name] = getattr(exc, name)
        invocation['error'] = error
    finally:
        invocation['finished_at'] = _now()
        _write(directory, invocation)


def main(argv=None):
    parser = argparse.ArgumentParser(description='Run a saved Study application call')
    parser.add_argument('--study-directory', required=True)
    parser.add_argument('--lock-fd', type=int, required=True)
    args = parser.parse_args(argv)
    # Do not propagate this lock to numerical workers. They already own their
    # Task/Run receipts and can be recovered if this coordinating process exits.
    os.set_inheritable(args.lock_fd, False)
    try:
        with adopt_study_lock(args.study_directory, args.lock_fd):
            run_invocation(args.study_directory)
    finally:
        os.close(args.lock_fd)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
