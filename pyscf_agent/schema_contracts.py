from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Tuple


PUBLIC_CONTRACT_VERSION = '1.0'
PUBLIC_SCHEMA_MANIFEST_SCHEMA = 'pyscf-agent.public-schema-manifest.v1'

BENCHMARK_MANIFEST_SCHEMA = 'pyscf-agent.benchmark-manifest.v1'
BENCHMARK_RESULT_SCHEMA = 'pyscf-agent.benchmark-result.v1'
INSTALL_VERIFICATION_SCHEMA = 'pyscf-agent.install-verification.v1'
REMOTE_VERIFICATION_SCHEMA = 'pyscf-agent.remote-verification.v1'
RUNTIME_IDENTITY_SCHEMA = 'pyscf-agent.runtime-identity.v1'

TASK_SPEC_SCHEMA = 'pyscf-agent.task-spec.v1'
TASK_REPORT_SCHEMA = 'pyscf-agent.task-report.v1'
STUDY_SPEC_SCHEMA = 'pyscf-agent.study-spec.v1'
STUDY_PLAN_SCHEMA = 'pyscf-agent.study-plan.v1'
STUDY_REPORT_SCHEMA = 'pyscf-agent.study-report.v1'
ADAPTIVE_STUDY_REPORT_SCHEMA = 'pyscf-agent.adaptive-study-report.v1'
JOB_HANDLE_SCHEMA = 'pyscf-agent.job-handle.v1'
JOB_STATUS_SCHEMA = 'pyscf-agent.job-status.v1'
BATCH_TASK_SCHEMA = 'pyscf-agent.batch-task.v1'
BATCH_HANDLE_SCHEMA = 'pyscf-agent.batch-handle.v1'
BATCH_EXECUTION_RESULT_SCHEMA = 'pyscf-agent.batch-execution-result.v1'
SLURM_BATCH_MANIFEST_SCHEMA = 'pyscf-agent.slurm-batch-manifest.v1'
REMOTE_RPC_SCHEMA = 'pyscf-agent.remote-rpc.v1'
STUDY_EXECUTION_RECEIPT_SCHEMA = 'pyscf-agent.study-execution-receipt.v1'
STUDY_EXECUTION_STATUS_SCHEMA = 'pyscf-agent.study-execution-status.v1'


_PUBLIC_SCHEMA_FIELDS: Dict[str, Tuple[str, ...]] = {
    PUBLIC_SCHEMA_MANIFEST_SCHEMA: (
        'contract_version',
        'compatibility_policy',
        'schemas',
    ),
    BENCHMARK_MANIFEST_SCHEMA: ('benchmarks',),
    BENCHMARK_RESULT_SCHEMA: (
        'status',
        'created_at',
        'environment',
        'benchmark_count',
        'passed_count',
        'results',
    ),
    INSTALL_VERIFICATION_SCHEMA: (
        'status',
        'dependency_mode',
        'checks',
        'wheel',
        'environment',
    ),
    REMOTE_VERIFICATION_SCHEMA: (
        'status',
        'profile_id',
        'checks',
        'smoke_submitted',
        'client_contract_version',
    ),
    RUNTIME_IDENTITY_SCHEMA: (
        'environment_id',
        'release_id',
        'source_revision',
        'source_state',
        'source_fingerprint',
        'public_contract_version',
        'created_at',
    ),
    TASK_SPEC_SCHEMA: (
        'task_type',
        'system',
        'method',
        'job',
        'analysis',
        'runtime',
    ),
    TASK_REPORT_SCHEMA: (
        'execution_status',
        'task_spec',
        'attempts',
        'errors',
        'artifacts',
    ),
    STUDY_SPEC_SCHEMA: (
        'name',
        'objective',
        'system_type',
        'base_task',
        'observables',
    ),
    STUDY_PLAN_SCHEMA: (
        'study_id',
        'name',
        'objective',
        'system_type',
        'cases',
        'observables',
    ),
    STUDY_REPORT_SCHEMA: (
        'study_id',
        'name',
        'objective',
        'system_type',
        'status',
        'cases',
        'comparison_table',
        'artifacts',
    ),
    JOB_HANDLE_SCHEMA: (
        'job_id',
        'executor_id',
        'run_id',
        'work_dir',
        'submitted_at',
    ),
    JOB_STATUS_SCHEMA: ('handle', 'state', 'terminal', 'updated_at'),
    BATCH_TASK_SCHEMA: ('task_id', 'request', 'channel', 'locale'),
    BATCH_HANDLE_SCHEMA: (
        'batch_id',
        'executor_id',
        'backend_job_id',
        'submitted_at',
        'profile_id',
        'manifest_path',
        'jobs',
    ),
    BATCH_EXECUTION_RESULT_SCHEMA: ('reports', 'batches', 'artifacts'),
    SLURM_BATCH_MANIFEST_SCHEMA: ('batch_id', 'profile_id', 'tasks'),
    REMOTE_RPC_SCHEMA: ('ok', 'operation'),
    STUDY_EXECUTION_RECEIPT_SCHEMA: (
        'study_id',
        'study_fingerprint',
        'task_fingerprints',
        'executor',
        'status',
        'batches',
    ),
    STUDY_EXECUTION_STATUS_SCHEMA: (
        'status',
        'expected',
        'terminal',
        'report_available',
        'receipts',
    ),
}

# The existing adaptive report shares the study envelope and adds its stage
# evidence. Keep its schema identity when reading it through StudyReport.
_PUBLIC_SCHEMA_FIELDS[ADAPTIVE_STUDY_REPORT_SCHEMA] = (
    *_PUBLIC_SCHEMA_FIELDS[STUDY_REPORT_SCHEMA],
    'adaptive',
)


def public_schema_ids() -> Tuple[str, ...]:
    """Return the stable top-level schemas exposed to clients and executors."""

    return tuple(_PUBLIC_SCHEMA_FIELDS)


def validate_public_payload(
    payload: Mapping[str, Any],
    *,
    expected_schema: Optional[str] = None,
    allow_missing_schema: bool = False,
) -> str:
    """Validate the version and required fields of a public payload.

    Unknown fields are intentionally accepted so a v1 reader can consume
    additive v1 updates. A new or incompatible layout must use a new schema ID.
    """

    if not isinstance(payload, Mapping):
        raise TypeError('Public payload must be a mapping')
    raw_schema = payload.get('schema')
    if raw_schema is None and allow_missing_schema and expected_schema:
        schema = expected_schema
    else:
        schema = str(raw_schema or '').strip()
    if not schema:
        raise ValueError('Public payload is missing its schema identifier')
    if expected_schema is not None and schema != expected_schema:
        raise ValueError(
            'Expected schema {0}, received {1}'.format(expected_schema, schema)
        )
    required = _PUBLIC_SCHEMA_FIELDS.get(schema)
    if required is None:
        raise ValueError('Unsupported public schema: {0}'.format(schema))
    missing = [field for field in required if field not in payload]
    if missing:
        raise ValueError(
            '{0} is missing required field(s): {1}'.format(
                schema,
                ', '.join(missing),
            )
        )
    return schema


def public_schema_manifest() -> Dict[str, Any]:
    """Return a machine-readable inventory of the stable public contracts."""

    return {
        'schema': PUBLIC_SCHEMA_MANIFEST_SCHEMA,
        'contract_version': PUBLIC_CONTRACT_VERSION,
        'compatibility_policy': {
            'same_schema': 'Readers accept additive fields within the same schema ID.',
            'breaking_change': 'Breaking changes require a new schema ID.',
            'missing_schema': 'Top-level public payloads must include schema.',
        },
        'schemas': [
            {
                'schema': schema,
                'required_fields': list(required_fields),
            }
            for schema, required_fields in _PUBLIC_SCHEMA_FIELDS.items()
        ],
    }


def public_schema_requirements() -> Dict[str, Tuple[str, ...]]:
    """Return a defensive copy for documentation and contract tests."""

    return copy.deepcopy(_PUBLIC_SCHEMA_FIELDS)


__all__ = [
    'ADAPTIVE_STUDY_REPORT_SCHEMA',
    'BENCHMARK_MANIFEST_SCHEMA',
    'BENCHMARK_RESULT_SCHEMA',
    'BATCH_EXECUTION_RESULT_SCHEMA',
    'BATCH_HANDLE_SCHEMA',
    'BATCH_TASK_SCHEMA',
    'JOB_HANDLE_SCHEMA',
    'JOB_STATUS_SCHEMA',
    'INSTALL_VERIFICATION_SCHEMA',
    'PUBLIC_CONTRACT_VERSION',
    'PUBLIC_SCHEMA_MANIFEST_SCHEMA',
    'REMOTE_VERIFICATION_SCHEMA',
    'RUNTIME_IDENTITY_SCHEMA',
    'REMOTE_RPC_SCHEMA',
    'SLURM_BATCH_MANIFEST_SCHEMA',
    'STUDY_EXECUTION_RECEIPT_SCHEMA',
    'STUDY_EXECUTION_STATUS_SCHEMA',
    'STUDY_PLAN_SCHEMA',
    'STUDY_REPORT_SCHEMA',
    'STUDY_SPEC_SCHEMA',
    'TASK_REPORT_SCHEMA',
    'TASK_SPEC_SCHEMA',
    'public_schema_ids',
    'public_schema_manifest',
    'public_schema_requirements',
    'validate_public_payload',
]
