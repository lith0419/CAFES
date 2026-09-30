"""Launch or reconnect to the existing local WebUI; never execute a Study."""

from __future__ import annotations

import fcntl
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from urllib.parse import urlencode
from urllib.request import ProxyHandler, build_opener

from pyscf_agent.paths import resolve_work_dir
from .executors.remote_config import resolve_remote_config_path
from .executors.slurm_config import resolve_slurm_config_path


class LocalWorkbench:
    def __init__(self, work_dir, arguments, execution_target):
        self.root = Path(work_dir).expanduser().resolve()
        self.arguments = ['--work-dir', str(self.root), *arguments]
        self.execution_target = execution_target
        # Preserve the venv interpreter, including after plugin-cache relocation.
        self.python = os.path.abspath(sys.executable)
        identity = [self.python, self.arguments]
        for flag in ('--remote-config', '--slurm-config'):
            if flag in arguments:
                path = Path(arguments[arguments.index(flag) + 1])
                identity.append(hashlib.sha256(path.read_bytes()).hexdigest())
        self.identity = hashlib.sha256(json.dumps(identity).encode()).hexdigest()[:20]
        self.directory = self.root / '.workbench'
        self.state_path = self.directory / (self.identity + '.json')
        self.log_path = self.directory / (self.identity + '.log')

    @classmethod
    def from_args(cls, args):
        arguments = ['--executor', args.executor]
        target = args.executor
        if args.executor == 'remote':
            arguments += ['--remote-config', str(resolve_remote_config_path(args.remote_config)),
                          '--remote', args.remote]
            target = str(args.remote).strip().lower()
        elif args.executor == 'slurm':
            arguments += ['--slurm-config', str(resolve_slurm_config_path(args.slurm_config))]
            if args.slurm_profile:
                arguments += ['--slurm-profile', args.slurm_profile]
        if args.local_wall_time_seconds is not None:
            arguments += ['--local-wall-time-seconds', str(args.local_wall_time_seconds)]
        return cls(resolve_work_dir(args.work_dir), arguments, target)

    def running(self):
        """A cached port is usable only while the matching WebUI answers there."""
        try:
            state = json.loads(self.state_path.read_text())
            port = int(state['port'])
            if not 0 < port < 65536:
                return None
            # Loopback probes must not go through HTTP proxy configuration.
            with build_opener(ProxyHandler({})).open(
                f'http://127.0.0.1:{port}/api/workbench', timeout=0.5,
            ) as response:
                live = json.load(response)
            if live == {'workbench_id': self.identity, 'pid': state['pid']}:
                return state
        except (OSError, ValueError, KeyError, TypeError):
            pass
        return None

    def url(self, state, study_id=None):
        parameters = {'work_dir': str(self.root), 'execution_target': self.execution_target}
        if study_id is not None:
            parameters['study'] = study_id
        return f"http://127.0.0.1:{state['port']}/computational-study/?" + urlencode(parameters)

    def existing_url(self, study_id):
        state = self.running()
        return self.url(state, study_id) if state else None

    def open_task(self, handle):
        """Keep the exact handle in the URL fragment, without a second job store."""
        from urllib.parse import urlsplit
        result = self.open()
        base = urlsplit(result['url'])
        fragment = urlencode({'handle': json.dumps(handle.to_dict(), separators=(',', ':')),
                              'execution_target': self.execution_target})
        result['url'] = f'{base.scheme}://{base.netloc}/task-monitor/#{fragment}'
        return result

    def open(self, study_id=None):
        self.directory.mkdir(parents=True, exist_ok=True)
        # Serializes startup across MCP connections, without locking calculations.
        with self.state_path.with_suffix('.lock').open('a+b') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            state = self.running()
            started = state is None
            if started:
                with self.log_path.open('ab') as log:
                    process = subprocess.Popen([
                        self.python, '-u', '-m', 'pyscf_agent.pyscf_agent_web',
                        '--host', '127.0.0.1', '--port', '0', '--no-open-browser',
                        '--workbench-state', str(self.state_path),
                        '--workbench-id', self.identity, *self.arguments,
                    ], stdin=subprocess.DEVNULL, stdout=log, stderr=log,
                       start_new_session=True, close_fds=True)
                deadline = time.monotonic() + 15
                while time.monotonic() < deadline:
                    if process.poll() is not None:
                        raise OSError(f'Workbench exited during startup. See {self.log_path}')
                    state = self.running()
                    if state:
                        break
                    time.sleep(0.1)
                else:
                    process.terminate()
                    process.wait(timeout=5)
                    raise OSError(f'Workbench did not become ready. See {self.log_path}')
            return {'url': self.url(state, study_id), 'started': started, 'pid': state['pid'],
                    'work_dir': str(self.root), 'execution_target': self.execution_target,
                    'log_path': str(self.log_path)}
