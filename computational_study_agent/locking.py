"""One advisory file lock for Study mutation, across threads and processes."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import threading
import time

from .execution_receipts import StudyExecutionBusy

LOCK_FILENAME = '.study-invocation.lock'
_OWNED = threading.local()


def _owned():
    # A fork must not inherit Python ownership; workers adopt their passed fd.
    if getattr(_OWNED, 'pid', None) != os.getpid():
        _OWNED.pid = os.getpid()
        _OWNED.streams = {}
    return _OWNED.streams


def _acquire_writer(stream, *, wait_seconds=0.0):
    deadline = time.monotonic() + 5.0
    writer_deadline = time.monotonic() + wait_seconds
    while True:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return
        except BlockingIOError:
            # Readers only inspect the lock. Do not mistake them for a worker.
            try:
                fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                if time.monotonic() < writer_deadline:
                    time.sleep(0.01)
                    continue
                raise StudyExecutionBusy('This Study already has an operation in progress; inspect its status.') from exc
            fcntl.flock(stream, fcntl.LOCK_UN)
            if time.monotonic() >= deadline:
                raise TimeoutError('Study status readers did not release the lock')
            time.sleep(0.001)


@contextmanager
def study_lock(directory, *, reentrant=True, create=True, wait_seconds=0.0):
    directory = Path(directory).resolve()
    key = str(directory)
    owned = _owned()
    if reentrant and key in owned:
        yield owned[key]
        return
    if create:
        directory.mkdir(parents=True, exist_ok=True)
    with (directory / LOCK_FILENAME).open('a+b') as stream:
        _acquire_writer(stream, wait_seconds=wait_seconds)
        owned[key] = stream
        try:
            yield stream
        finally:
            owned.pop(key, None)
            # Close only. LOCK_UN would release a child's inherited ownership.


@contextmanager
def adopt_study_lock(directory, fd):
    directory = Path(directory).resolve()
    expected = (directory / LOCK_FILENAME).stat()
    actual = os.fstat(fd)
    if (expected.st_dev, expected.st_ino) != (actual.st_dev, actual.st_ino):
        raise ValueError('Inherited descriptor does not belong to this Study')
    with os.fdopen(os.dup(fd), 'a+b') as stream:
        _acquire_writer(stream)
        owned = _owned()
        key = str(directory)
        previous = owned.get(key)
        owned[key] = stream
        try:
            yield stream
        finally:
            if previous is None:
                owned.pop(key, None)
            else:
                owned[key] = previous


def study_is_locked(directory):
    """Observe with a shared lock; execution distinguishes readers from writers."""
    path = Path(directory) / LOCK_FILENAME
    try:
        stream = path.open('rb')
    except FileNotFoundError:
        return False
    with stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_SH | fcntl.LOCK_NB)
        except BlockingIOError:
            return True
    return False
