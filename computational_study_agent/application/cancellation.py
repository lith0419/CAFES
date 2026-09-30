"""Stop a local Study coordinator before cancelling its persisted workers."""
from __future__ import annotations

import os
from pathlib import Path
import shlex
import signal
import subprocess

from pyscf_agent.executors import JobHandle
from ..execution_receipts import discover_receipts, jobs_from_receipt, load_receipt
from ..locking import study_is_locked, study_lock
from ..study_state import read_json
from .background import INVOCATION_FILENAME, _now, _write


def _coordinator_pid(directory):
    # Match the exact module and Study directory, including legacy invocations
    # written before coordinator PIDs were recorded. Never signal a saved PID alone.
    output = subprocess.check_output(['ps', '-axo', 'pid=,command='], text=True)
    matches = []
    for line in output.splitlines():
        fields = line.strip().split(None, 1)
        if len(fields) != 2:
            continue
        try:
            args = shlex.split(fields[1])
            module = args.index('-m')
            study = args.index('--study-directory')
            if (args[module + 1] == 'computational_study_agent.application.background'
                    and Path(args[study + 1]).resolve() == directory):
                matches.append(int(fields[0]))
        except (ValueError, IndexError):
            continue
    if len(matches) != 1:
        raise ValueError('Could not identify this Study coordinator safely; refresh and try again.')
    return matches[0]


def stop_study(self, study_id, *, work_dir=None):
    directory = self._study_directory(study_id, work_dir)
    invocation = read_json(directory / INVOCATION_FILENAME)
    if (invocation['execution_config'].get('execution_target', 'local') != 'local'
            or self._task_executor is None
            or self._task_executor.describe().get('executor_id') != 'local-process'):
        raise ValueError('Stop Calculation currently supports local Studies only.')
    started_at = invocation.get('started_at')
    if study_is_locked(directory):
        pid = _coordinator_pid(directory)
        if pid == os.getpid():
            raise ValueError('Cannot stop the serving process.')
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    # Holding the Study lock prevents another start while workers are cancelled.
    # The coordinator must release it before we enumerate workers, so no next
    # refinement batch or sequential case can be created after cancellation.
    with study_lock(directory, reentrant=False, create=False, wait_seconds=5):
        invocation = read_json(directory / INVOCATION_FILENAME)
        if invocation.get('started_at') != started_at:
            raise ValueError('The Study restarted during cancellation; refresh and try again.')
        handles = {}
        for path in discover_receipts(directory):
            for handle in jobs_from_receipt(load_receipt(path)).values():
                handles[(handle.work_dir, handle.run_id)] = handle
        # Cover the startup window before the coordinator persists its receipt.
        for path in directory.rglob('job-state.json'):
            state = read_json(path)
            if state.get('handle'):
                handle = JobHandle.from_dict(state['handle'])
                handles[(handle.work_dir, handle.run_id)] = handle
        results, errors = [], []
        for handle in handles.values():
            try:
                if handle.executor_id != 'local-process':
                    raise ValueError('Refusing a nonlocal worker in a local Study')
                (Path(handle.work_dir).resolve() / handle.run_id).resolve().relative_to(directory)
                status = self._task_executor.status(handle)
                if not status.terminal:
                    status = self._task_executor.cancel(handle)
                results.append(status.to_dict())
                if not status.terminal:
                    errors.append({'run_id': handle.run_id, 'message': 'Cancellation has not completed.'})
            except Exception as exc:
                errors.append({'run_id': handle.run_id, 'message': str(exc)})
        invocation.update(finished_at=_now(), stop_requested_at=_now(),
                          cancelled=not errors, cancellation_errors=errors)
        invocation['error'] = ({'code': 'StudyCancellationIncomplete',
                                'message': 'Some workers could not be confirmed stopped.'} if errors else None)
        _write(directory, invocation)
        # Reconcile terminal workers into the ordinary saved results without
        # advancing refinement or retrying any case.
        if not errors and (directory / 'study-state.json').is_file():
            try:
                self.collect_study(study_id, work_dir=str(directory.parent))
            except Exception as exc:
                invocation['collection_warning'] = str(exc)
                _write(directory, invocation)
        return {'study_id': study_id, 'status': 'stop_incomplete' if errors else 'cancelled',
                'cancelled': not errors, 'jobs': results, 'errors': errors,
                'message': ('Some workers could not be confirmed stopped. Retry Stop Calculation.'
                            if errors else 'Calculation stopped. Existing results and logs are preserved.')}
