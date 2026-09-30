"""Stable run identifiers shared by task and Study preparation."""
from datetime import datetime, timezone
import uuid


def make_run_id() -> str:
    timestamp = datetime.now(timezone.utc).strftime('%Y%m%d-%H%M%S')
    return '{0}-{1}'.format(timestamp, uuid.uuid4().hex[:8])
