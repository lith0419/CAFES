"""Bounded control-plane commands; scientific jobs have separate lifetimes."""
from __future__ import annotations

import subprocess
from typing import Any, Sequence

CONTROL_TIMEOUT_SECONDS = 60.0
DEPLOYMENT_TIMEOUT_SECONDS = 300.0
BULK_TIMEOUT_SECONDS = 1800.0


class CommandTimeoutError(TimeoutError):
    """The remote outcome is unknown; a timeout never proves job failure."""


def command_timeout(operation: str, exc: subprocess.TimeoutExpired) -> CommandTimeoutError:
    return CommandTimeoutError(
        f'{operation} timed out after {exc.timeout:g}s; status is temporarily unreadable. '
        'The remote operation may still be running; inspect before retrying submission.'
    )


def run_bounded(args: Sequence[str], *, timeout: float = CONTROL_TIMEOUT_SECONDS, **kwargs: Any) -> subprocess.CompletedProcess:
    try:
        return subprocess.run(list(args), timeout=timeout, **kwargs)
    except subprocess.TimeoutExpired as exc:
        raise command_timeout(str(args[0]), exc) from exc
