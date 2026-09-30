"""Durable identity and process-group control for local task workers (POSIX)."""
from __future__ import annotations

import json
import os
import signal
import subprocess
import time
from pathlib import Path
from typing import Any, Dict, Optional

from .base import ExecutorContractError
from .contracts import JobHandle
from .local import _write_json_atomic


PROCESS_FILENAME = 'job-process.json'
CHILD_PROCESS_FILENAME = 'job-calculation-process.json'
CANCEL_FILENAME = 'job-cancel.json'
PROCESS_SCHEMA = 'pyscf-agent.local-process.v1'


def process_snapshot(pid: int) -> Optional[Dict[str, Any]]:
    """Use both start time and a per-run command token, never PID alone."""
    if pid <= 1:
        raise ExecutorContractError('Invalid local worker PID')
    result = subprocess.run(
        ['ps', '-p', str(pid), '-o', 'lstart=', '-o', 'pgid=', '-o', 'stat=', '-o', 'args='],
        capture_output=True, text=True, env={**os.environ, 'LC_ALL': 'C'},
    )
    line = result.stdout.strip()
    if result.returncode == 1 and not line:
        return None
    if result.returncode != 0 or not line:
        raise ExecutorContractError('Cannot inspect local worker process identity')
    parts = line.split(None, 7)
    if len(parts) != 8:
        raise ExecutorContractError('Unrecognized local worker process identity')
    return {'pid': pid, 'started': ' '.join(parts[:5]), 'pgid': int(parts[5]),
            'state': parts[6], 'command': parts[7]}


def write_process_identity(path: Path, handle: JobHandle, pid: int, token: str) -> None:
    # A newly forked child can still show its parent's command before exec.
    deadline = time.monotonic() + 2.0
    while True:
        snapshot = process_snapshot(pid)
        if snapshot is None:
            return
        if token in snapshot['command'] and snapshot['pgid'] == pid:
            break
        if time.monotonic() >= deadline:
            break
        time.sleep(0.02)
    if not token or token not in snapshot['command'] or snapshot['pgid'] != pid:
        raise ExecutorContractError('Local worker was not launched in its own verified process group')
    _write_json_atomic(path, {
        'schema': PROCESS_SCHEMA, 'handle': handle.to_dict(), 'token': token,
        'pid': pid, 'pgid': pid, 'started': snapshot['started'],
    }, kind='job-state')


def read_process_identity(path: Path, handle: JobHandle) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding='utf-8'))
        if (data['schema'] != PROCESS_SCHEMA or JobHandle.from_dict(data['handle']) != handle
                or not isinstance(data['pid'], int) or data['pid'] <= 1
                or data['pgid'] != data['pid']
                or not isinstance(data['token'], str) or not data['token']
                or not isinstance(data['started'], str) or not data['started']):
            raise ValueError('Mismatched process identity')
        return data
    except (OSError, KeyError, TypeError, ValueError) as exc:
        raise ExecutorContractError('Local worker identity is unreadable or mismatched') from exc


def identity_is_running(identity: Dict[str, Any]) -> bool:
    snapshot = process_snapshot(identity['pid'])
    if snapshot is None or snapshot['state'].startswith('Z'):
        return False
    if (snapshot['started'] != identity['started'] or snapshot['pgid'] != identity['pgid']
            or identity['token'] not in snapshot['command']):
        raise ExecutorContractError('Local worker PID was reused or its identity changed; refusing process control')
    return True


def group_is_running(pgid: int) -> bool:
    result = subprocess.run(['ps', '-e', '-o', 'pgid=', '-o', 'stat='],
                            capture_output=True, text=True)
    if result.returncode != 0:
        raise ExecutorContractError('Cannot confirm that the calculation process group stopped')
    return any(len(parts) == 2 and parts[0] == str(pgid) and not parts[1].startswith('Z')
               for parts in (line.split() for line in result.stdout.splitlines()))


def stop_process_group(pgid: int, grace: float) -> None:
    if pgid <= 1 or pgid == os.getpgrp():
        raise ExecutorContractError('Refusing to signal an unisolated process group')
    for sig in (signal.SIGTERM, signal.SIGKILL):
        if not group_is_running(pgid):
            return
        try:
            os.killpg(pgid, sig)
        except ProcessLookupError:
            return
        deadline = time.monotonic() + grace
        while time.monotonic() < deadline:
            if not group_is_running(pgid):
                return
            time.sleep(0.05)
    if group_is_running(pgid):
        raise ExecutorContractError('Calculation processes did not stop; cancellation is unconfirmed')
