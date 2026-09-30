"""Shared run-root policy; preserve logical paths across cluster mounts."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

RUNS_DIR_ENV = 'PYSCF_AGENT_RUNS_DIR'


def default_runs_root() -> Path:
    configured = os.environ.get(RUNS_DIR_ENV)
    if configured:
        return Path(os.path.abspath(os.path.expanduser(configured)))
    data = os.environ.get('XDG_DATA_HOME') or str(Path.home() / '.local' / 'share')
    return Path(os.path.abspath(os.path.expanduser(data))) / 'pyscf-agent' / 'runs'


def resolve_work_dir(work_dir: Any = None, run_id: Any = None) -> Path:
    if isinstance(work_dir, str) and work_dir.strip():
        root = Path(os.path.abspath(os.path.expanduser(work_dir)))
        if isinstance(run_id, str) and run_id.strip() and root.name == run_id.strip():
            return root.parent
        return root
    return default_runs_root()


def study_search_roots(work_dir: Any = None) -> tuple[Path, ...]:
    """Discover old checkout runs only when browsing the ordinary default.

    Explicit custom roots and the environment override remain authoritative.
    This reads old locations; it never moves or creates directories.
    """
    root = resolve_work_dir(work_dir)
    legacy = Path.cwd() / 'runs'
    if os.environ.get(RUNS_DIR_ENV) or root != default_runs_root() or root == legacy:
        return (root,)
    return (root, legacy)
