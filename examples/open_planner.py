"""Install the archived examples into a local work directory and open Planner."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import shutil
import sys
import webbrowser


EXAMPLES = Path(__file__).resolve().parent
ROOT = EXAMPLES.parent
sys.path.insert(0, str(ROOT))

from computational_study_agent.example_storage import materialize_study, read_example_json  # noqa: E402


def verify_sources() -> int:
    """Verify all public source copies without accessing the author's archives."""
    manifest = json.loads((EXAMPLES / 'manifest.json').read_text())
    verified = set()
    for study in manifest['studies']:
        for source in study['sources']:
            path = (EXAMPLES / source['path']).resolve()
            path.relative_to(EXAMPLES.resolve())
            data = path.read_bytes()
            if hashlib.sha256(data).hexdigest() != source['sha256']:
                raise ValueError(f'Source checksum mismatch: {source["path"]}')
            read_example_json(path)
            verified.add(source['path'])
    return len(verified)


def prepare(work_dir: Path) -> list[str]:
    """Materialize portable Study files, preserving existing local copies."""
    work_dir = work_dir.expanduser().resolve()
    work_dir.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((EXAMPLES / 'manifest.json').read_text())
    ids = []
    for item in manifest['studies']:
        study_id = item['study_id']
        destination = work_dir / study_id
        marker = destination / 'example-origin.json'
        if destination.exists():
            if not marker.is_file():
                raise FileExistsError(f'Refusing to replace an existing Study: {destination}')
        else:
            shutil.copytree(EXAMPLES / 'studies' / study_id, destination)
            materialize_study(destination)
            report_path = destination / 'study-report.json'
            report = json.loads(report_path.read_text())
            report['work_dir'] = str(destination)
            report_path.write_text(json.dumps(report, indent=2) + '\n')
            marker.write_text(json.dumps(item, indent=2) + '\n')
        ids.append(study_id)
    return ids


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--work-dir', type=Path, default=ROOT / 'runs' / 'planner-examples')
    parser.add_argument('--port', type=int, default=8767)
    parser.add_argument('--prepare-only', action='store_true')
    parser.add_argument('--no-open-browser', action='store_true')
    parser.add_argument('--verify-sources', action='store_true',
                        help='Verify bundled source checksums and exit.')
    args = parser.parse_args()
    if args.verify_sources:
        print(f'Verified {verify_sources()} bundled source files.', flush=True)
        return 0
    work_dir = args.work_dir.expanduser().resolve()
    ids = prepare(work_dir)
    base = f'http://127.0.0.1:{args.port}/computational-study/'
    for study_id in ids:
        print(f'{study_id}: {base}?study={study_id}', flush=True)
    if args.prepare_only:
        return 0
    from pyscf_agent.web.server import serve
    if not args.no_open_browser:
        webbrowser.open(base)
    return serve(['--host', '127.0.0.1', '--port', str(args.port),
                  '--work-dir', str(work_dir), '--no-open-browser'])


if __name__ == '__main__':
    raise SystemExit(main())
