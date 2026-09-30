"""Real subprocess fixture; never part of an executable platform capability."""
import json
import signal
import subprocess
import sys
import time
from pathlib import Path

from pyscf_agent.executors.local import _write_json_atomic
from pyscf_agent.executors.local_process_worker import build_parser, run_request_file


def main():
    args = build_parser().parse_args()
    if args.request_file:
        def launch(command, **kwargs):
            command = list(command)
            command[2] = __name__ if __name__ != '__main__' else 'tests.pyscf_agent.local_process_fixture'
            return subprocess.Popen(command, **kwargs)
        run_request_file(args.request_file, args.worker_token, popen_factory=launch)
        return
    path = Path(args.execute_request_file)
    payload = json.loads(path.read_text())
    request = payload['request']
    if isinstance(request, str):
        request = json.loads(request)
    if request.get('fixture') == 'tree':
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        child = subprocess.Popen([sys.executable, '-c',
            'import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); time.sleep(60)'])
        (path.parent / 'fixture-descendant.pid').write_text(str(child.pid))
    (path.parent / 'fixture-ready').write_text('ready')
    time.sleep(request.get('duration', 60))
    _write_json_atomic(path.parent / 'job-task-report.json', {
        'run_id': payload['handle']['run_id'], 'execution_status': 'succeeded',
        'structured_results': {'fixture': True},
    }, kind='task-report')


if __name__ == '__main__':
    main()
