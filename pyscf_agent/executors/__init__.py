from __future__ import annotations

from .base import (
    BatchTaskExecutor,
    DatasetGeneratingExecutor,
    ExecutorContractError,
    FileCollectingExecutor,
    JobCancellationUnsupportedError,
    JobConflictError,
    JobError,
    JobNotFoundError,
    JobNotReadyError,
    TaskExecutor,
)
from .contracts import (
    BATCH_EXECUTION_RESULT_SCHEMA,
    BATCH_HANDLE_SCHEMA,
    BATCH_TASK_SCHEMA,
    BatchExecutionResult,
    BatchHandle,
    BatchTask,
    JOB_HANDLE_SCHEMA,
    JOB_STATUS_SCHEMA,
    SLURM_BATCH_MANIFEST_SCHEMA,
    JobHandle,
    JobState,
    JobStatus,
)
from .local import (
    JOB_REPORT_FILENAME,
    JOB_STATE_FILENAME,
    LocalExecutor,
    get_local_executor,
)
from .local_process import LocalProcessExecutor
from .factory import (
    EXECUTION_TARGETS,
    add_executor_arguments,
    create_task_executor,
    create_task_executor_from_args,
)
from .remote_config import (
    DEFAULT_REMOTE_CONFIG_PATH,
    RemoteConfigurationError,
    SshRemoteProfile,
    load_remote_profile,
    resolve_remote_config_path,
)
from .slurm import (
    SLURM_BATCH_HANDLE_FILENAME,
    SlurmExecutor,
)
from .slurm_config import (
    DEFAULT_SLURM_CONFIG_PATH,
    SlurmConfigurationError,
    SlurmExecutorConfig,
    SlurmRemoteConfig,
    load_slurm_executor_config,
    resolve_slurm_config_path,
)
from .ssh_slurm import SshSlurmExecutor


__all__ = [
    'BATCH_EXECUTION_RESULT_SCHEMA',
    'BATCH_HANDLE_SCHEMA',
    'BATCH_TASK_SCHEMA',
    'BatchExecutionResult',
    'BatchHandle',
    'BatchTask',
    'BatchTaskExecutor',
    'DatasetGeneratingExecutor',
    'DEFAULT_SLURM_CONFIG_PATH',
    'DEFAULT_REMOTE_CONFIG_PATH',
    'EXECUTION_TARGETS',
    'ExecutorContractError',
    'FileCollectingExecutor',
    'JOB_HANDLE_SCHEMA',
    'JOB_REPORT_FILENAME',
    'JOB_STATE_FILENAME',
    'JOB_STATUS_SCHEMA',
    'JobCancellationUnsupportedError',
    'JobConflictError',
    'JobError',
    'JobHandle',
    'JobNotFoundError',
    'JobNotReadyError',
    'JobState',
    'JobStatus',
    'LocalExecutor',
    'LocalProcessExecutor',
    'RemoteConfigurationError',
    'SLURM_BATCH_HANDLE_FILENAME',
    'SLURM_BATCH_MANIFEST_SCHEMA',
    'SlurmConfigurationError',
    'SlurmExecutor',
    'SlurmExecutorConfig',
    'SlurmRemoteConfig',
    'SshRemoteProfile',
    'SshSlurmExecutor',
    'TaskExecutor',
    'add_executor_arguments',
    'create_task_executor',
    'create_task_executor_from_args',
    'get_local_executor',
    'load_slurm_executor_config',
    'load_remote_profile',
    'resolve_remote_config_path',
    'resolve_slurm_config_path',
]
