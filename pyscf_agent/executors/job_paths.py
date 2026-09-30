from __future__ import annotations
from pathlib import Path
from .base import JobNotFoundError
from .contracts import JobHandle, JobState, JobStatus

def collectable_status(status: JobStatus) -> bool:
    return bool(
        status.terminal
        and (
            status.state != JobState.COMPLETED
            or status.report_available
        )
    )



def job_directory(handle: JobHandle) -> Path:
    root = Path(handle.work_dir).expanduser().resolve()
    directory = (root / handle.run_id).resolve()
    try:
        directory.relative_to(root)
    except ValueError as exc:
        raise JobNotFoundError('JobHandle run_id escapes its work directory') from exc
    return directory
