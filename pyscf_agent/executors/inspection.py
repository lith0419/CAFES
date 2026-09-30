"""Bounded, read-only projections of executor-owned task evidence."""

from __future__ import annotations

from datetime import datetime, timezone
import json

from ..providers.block2.progress import parse_sweep_progress


LOG_TAIL_BYTES = 64 * 1024
MAX_LOGS = 8
MAX_REPORT_BYTES = 16 * 1024 * 1024


def status_view(status):
    now = datetime.now(timezone.utc)
    elapsed = None
    if status.started_at:
        try:
            end = datetime.fromisoformat(status.completed_at) if status.completed_at else now
            elapsed = max(0.0, (end - datetime.fromisoformat(status.started_at)).total_seconds())
        except (TypeError, ValueError):
            pass  # Older executor timestamps may not contain a timezone.
    return {'handle': status.handle.to_dict(), 'status': status.to_dict(),
            'checked_at': now.isoformat(), 'elapsed_seconds': elapsed,
            'summary': None, 'solver_progress': None, 'logs': [], 'notices': []}


def _inside(path, directory):
    return path.resolve().is_relative_to(directory)


def inspect_saved_task(executor, handle):
    """Use the executor's authoritative status, without fetch/collect or recovery."""
    status = executor.status(handle)  # Also verifies persisted handle identity.
    view = status_view(status)
    directory = executor._job_dir(status.handle).resolve()
    report_path = executor._report_path(status.handle)
    if status.report_available:
        try:
            if not _inside(report_path, directory):
                raise ValueError('Report path is outside this Run')
            if report_path.stat().st_size > MAX_REPORT_BYTES:
                raise ValueError('Report exceeds the monitor read limit; use collect_task for the full report')
            report = json.loads(report_path.read_text(encoding='utf-8'))
            if report.get('run_id') != status.handle.run_id:
                raise ValueError('Report run_id does not match this Run')
            compact = report.get('compact_results') or {}
            view['summary'] = {
                'execution_status': report.get('execution_status'),
                'energy': compact.get('energy'), 'energy_unit': compact.get('energy_unit'),
                'converged': compact.get('converged'),
                'analysis_summary': str(report.get('analysis_summary') or '')[:4000],
                'errors': [{'message': str(error.get('message') or '')[:2000],
                            'code': error.get('code')} for error in (report.get('errors') or [])[:8]],
            }
        except (OSError, ValueError) as exc:
            view['notices'].append('Saved report unavailable: ' + str(exc))

    # Provider/worker text logs only. Never traverse MPS/checkpoint trees or read
    # arbitrary artifact paths from a request. Existing runs need no migration.
    candidates = set()
    for pattern in ('*.log', '*.out', '*.err', '*/*.log', '*/*/*.log'):
        for path in directory.glob(pattern):
            if _inside(path, directory) and path.is_file():
                candidates.add(path)
    readable = []
    for path in candidates:
        try:
            stat = path.stat()
            if stat.st_size:
                readable.append((stat.st_mtime, path, stat.st_size))
        except FileNotFoundError:
            continue  # A worker may rotate/remove its own output during a read.
    for modified, path, size in sorted(readable, key=lambda item: (item[0], str(item[1])), reverse=True)[:MAX_LOGS]:
        try:
            with path.open('rb') as stream:
                stream.seek(max(0, size - LOG_TAIL_BYTES))
                raw = stream.read(LOG_TAIL_BYTES)
            if size > LOG_TAIL_BYTES:
                raw = raw.partition(b'\n')[2]  # Drop a partial first line.
            text = raw.decode('utf-8', errors='replace')
            # A line being written is not evidence of a completed metric.
            complete = text[:text.rfind('\n') + 1]
            source = str(path.relative_to(directory))
            updated = datetime.fromtimestamp(modified, timezone.utc).isoformat()
            progress = parse_sweep_progress(complete)
            if progress and view['solver_progress'] is None:
                view['solver_progress'] = {**progress, 'source': source, 'logged_at': updated}
            view['logs'].append({'path': source, 'updated_at': updated,
                                 'tail': '\n'.join(complete.splitlines()[-30:])[-6000:],
                                 'truncated': size > LOG_TAIL_BYTES or len(complete) > 6000})
        except OSError as exc:
            view['notices'].append('Log unavailable: ' + str(exc))
    return view
