from __future__ import annotations

from typing import Any, Dict, List, Mapping, Optional, Protocol, Sequence, runtime_checkable

from .contracts import BatchExecutionResult, BatchTask, JobHandle, JobStatus


class ExecutorContractError(RuntimeError):
    """Raised when an executor does not return the required task report."""


class JobError(RuntimeError):
    """Base error for submitted-job lifecycle operations."""


class JobNotFoundError(JobError):
    """Raised when an executor cannot resolve a supplied JobHandle."""


class JobNotReadyError(JobError):
    """Raised when a submitted job does not yet have a TaskReport."""


class JobConflictError(JobError):
    """Raised when a submitted job identifier already exists."""


class JobCancellationUnsupportedError(JobError):
    """Raised when an executor cannot cancel the referenced job."""


@runtime_checkable
class TaskExecutor(Protocol):
    """Execution port shared by local, scheduler, and remote implementations."""

    executor_id: str
    execution_mode: str

    def execute_task(
        self,
        request: Any,
        *,
        channel: str = 'agent',
        locale: str = 'en',
        work_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        resource_profile: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Execute one task and return its TaskReport payload."""

    def submit_task(
        self,
        request: Any,
        *,
        channel: str = 'agent',
        locale: str = 'en',
        work_dir: Optional[str] = None,
        run_id: Optional[str] = None,
        resource_profile: Optional[str] = None,
    ) -> JobHandle:
        """Submit one task and return a stable handle."""

    def status(self, handle: Any) -> JobStatus:
        """Return the current scheduler-level state for a job."""

    def cancel(self, handle: Any) -> JobStatus:
        """Request cancellation and return the resulting scheduler state."""

    def fetch(self, handle: Any) -> Dict[str, Any]:
        """Return the completed TaskReport payload."""

    def logs(self, handle: Any) -> List[Dict[str, Any]]:
        """Return structured TaskReport log entries."""

    def artifacts(self, handle: Any) -> List[Dict[str, Any]]:
        """Return TaskReport artifact references."""

    def describe(self) -> Dict[str, Any]:
        """Return stable execution metadata for reports and diagnostics."""


@runtime_checkable
class TaskInspectingExecutor(TaskExecutor, Protocol):
    """Optional read-only port for status plus bounded solver/report evidence."""

    def inspect_task(self, handle: Any) -> Dict[str, Any]:
        """Read an existing Run without collection, execution or recovery."""


@runtime_checkable
class BatchTaskExecutor(TaskExecutor, Protocol):
    """Optional port for executing ready, mutually independent tasks in batches."""

    def execute_independent_tasks(
        self,
        tasks: Sequence[BatchTask],
        *,
        resource_profile: Optional[str] = None,
    ) -> BatchExecutionResult:
        """Execute independent tasks and return reports keyed by BatchTask.task_id."""


@runtime_checkable
class FileCollectingExecutor(TaskExecutor, Protocol):
    """Optional port for copying executor-owned artifacts to explicit local paths."""

    def collect_files(self, paths: Mapping[str, str]) -> List[Dict[str, Any]]:
        """Copy source-path keys to local destination-path values."""


@runtime_checkable
class DatasetGeneratingExecutor(TaskExecutor, Protocol):
    """Optional port for materializing a dataset on the execution target."""

    def generate_hamiltonian_dataset(
        self,
        request: Mapping[str, Any],
    ) -> Dict[str, Any]:
        """Generate a complete dataset in executor-owned storage."""


__all__ = [
    'BatchTaskExecutor',
    'DatasetGeneratingExecutor',
    'ExecutorContractError',
    'FileCollectingExecutor',
    'JobCancellationUnsupportedError',
    'JobConflictError',
    'JobError',
    'JobNotFoundError',
    'JobNotReadyError',
    'TaskExecutor',
    'TaskInspectingExecutor',
]
