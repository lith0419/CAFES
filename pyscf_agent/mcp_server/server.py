"""MCP tools translate transport data into calculation application calls."""

from __future__ import annotations

import json
import re
from functools import wraps
from pathlib import Path
from typing import Any, Optional

from mcp.server import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from computational_study_agent.application import StudyApplicationError
from computational_study_agent.execution_receipts import StudyExecutionInterrupted
from computational_study_agent.costing import CostApprovalRequired

from ..application import CalculationApplicationService
from ..application.calculation_service import CalculationApplicationError
from ..executors import JobHandle
from ..executors.base import ExecutorContractError, JobError
from ..remote.protocol import RemoteProtocolError
from .resources import register_resources, report_uri


def _tool_errors(function):
    """Keep anticipated application errors actionable for the MCP caller."""
    @wraps(function)
    def call(*args, **kwargs):
        try:
            return function(*args, **kwargs)
        except (ValueError, TypeError, OSError, CalculationApplicationError,
                StudyApplicationError, StudyExecutionInterrupted, CostApprovalRequired,
                ExecutorContractError, JobError, RemoteProtocolError) as exc:
            message = '{0}: {1}'.format(type(exc).__name__, exc)
            if function.__name__ == 'submit_task' and isinstance(exc, (OSError, RemoteProtocolError, ExecutorContractError)):
                message += ' No automatic retry was performed; inspect executor evidence before resubmitting if submission may have started.'
            raise ToolError(message) from exc
    return call


def create_server(
    service: CalculationApplicationService,
    *,
    work_dir: Path,
    executor_description: dict[str, Any],
    study_service: Any = None,
    workbench: Any = None,
) -> MCPServer:
    """Build an adapter with one configured execution target and no in-memory job store."""
    root = Path(work_dir).expanduser().resolve()
    server = MCPServer(
        'pyscf-agent',
        instructions=(
            'Use get_capabilities and validate_task before submit_task. Pass structured '
            'TaskSpec data, retain the returned JobHandle, and use status/collect to follow '
            'existing work. Scheduler completion and scientific success are separate. '
            'Scientific review gates remain enforced by the runtime. submit_task starts a '
            'new calculation; it is not idempotent. After a submission transport error, '
            'inspect existing executor evidence before submitting again. Read full reports '
            'and wiki pages through the returned resource URIs. For Studies, prepare a plan, '
            'start the agent once, then inspect and collect its results. The agent owns '
            'static task execution and adaptive orchestration. Review saves a decision '
            'for an explicit later start by Study ID. Analyze and dataset preparation '
            'also use existing agent services; MCP implements no scientific workflow.'
        ),
    )
    read_only = ToolAnnotations(readOnlyHint=True, idempotentHint=True)

    def checked_handle(payload: dict[str, Any]) -> JobHandle:
        handle = JobHandle.from_dict(payload)
        # SSH handles refer to the configured remote host's filesystem. Local
        # and direct-Slurm handles must remain within this server's output root.
        if executor_description.get('location') != 'remote_slurm_cluster':
            directory = (Path(handle.work_dir).expanduser() / handle.run_id).resolve()
            try:
                directory.relative_to(root)
            except ValueError as exc:
                raise ValueError('JobHandle is outside the configured work directory') from exc
        return handle

    @server.tool(annotations=read_only)
    @_tool_errors
    def get_capabilities(namespace: Optional[str] = None) -> dict[str, Any]:
        """List Registry namespaces, or full entries in one namespace (e.g. molecular.method).

        Includes the configured executor. Registered provider support does not
        imply an optional provider is installed; runtime validation still applies.
        """
        entries = service.capabilities()['entries']
        namespaces = sorted({entry['namespace'] for entry in entries})
        result = {'namespaces': namespaces, 'executor': executor_description,
                  'registry_uri': 'pyscf://capabilities', 'schemas_uri': 'pyscf://schemas',
                  'wiki_uri': 'pyscf://wiki'}
        if namespace is not None:
            if namespace not in namespaces:
                raise ValueError('Unknown capability namespace: ' + namespace)
            result['entries'] = [entry for entry in entries if entry['namespace'] == namespace]
        return result

    @server.tool(annotations=read_only)
    @_tool_errors
    def validate_task(task_spec: dict[str, Any], locale: str = 'en') -> dict[str, Any]:
        """Normalize, validate and compile a TaskSpec without running a calculation or LLM.

        Accepts nested TaskSpec fields or existing shorthand, e.g.
        {"task_type":"molecular","atom":"H 0 0 0; H 0 0 0.74","basis":"sto-3g","method":"hf"}.
        Returns valid, normalized task_spec, errors, defaults and compiled workflow.
        Validation does not grant scientific approval or test numerical convergence.
        """
        return service.validate_task_spec(task_spec, locale=locale)

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False))
    @_tool_errors
    def submit_task(
        task_spec: dict[str, Any],
        run_id: str,
        resource_profile: Optional[str] = None,
        locale: str = 'en',
    ) -> dict[str, Any]:
        """Validate and submit a new calculation; return its JobHandle without waiting.

        Choose a unique run_id (letters, digits, underscore, hyphen; up to 120 chars).
        Save handle for status, collect and cancel, including after server restart.
        Repeating submit can create another remote job: never retry blindly after
        a transport error. Resource profiles use the server's existing executor config.
        """
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', run_id):
            raise ValueError('run_id must be 1-120 letters, digits, underscores or hyphens, starting with a letter or digit')
        if run_id == root.name:
            raise ValueError('run_id must differ from the work directory name')
        prepared = service.validate_task_spec(task_spec, locale=locale)
        if not prepared['valid']:
            return {'submitted': False, 'validation': prepared}
        handle = service.submit_request(
            json.dumps(prepared['task_spec'], ensure_ascii=False), channel='mcp',
            locale=locale, work_dir=str(root), run_id=run_id, resource_profile=resource_profile,
        )
        # Do not query status here: a subsequent connection failure must not
        # hide a successfully returned submission handle from the caller.
        return {'submitted': True, 'handle': handle.to_dict(), 'report_uri': report_uri(handle)}

    @server.tool(annotations=read_only)
    @_tool_errors
    def get_task_status(handle: dict[str, Any]) -> dict[str, Any]:
        """Query the exact JobHandle returned by submit_task; never submits or retries.

        state is the scheduler state; task_status is scientific status. Collect
        only when a report is available. A cancelled job may have no TaskReport.
        """
        return service.job_status(checked_handle(handle)).to_dict()

    @server.tool(annotations=read_only)
    @_tool_errors
    def collect_task(handle: dict[str, Any], include_report: bool = False) -> dict[str, Any]:
        """Collect an existing TaskReport without submitting, retrying, or LLM feedback.

        Default output is a compact scientific summary plus artifact references.
        Read report_uri or set include_report=true for the complete report. A
        not-ready error means query status later, not resubmit. Scientific failures
        are reported as execution_status and errors, not as MCP transport failures.
        """
        normalized = checked_handle(handle)
        report = service.collect_request(normalized, '', include_llm_feedback=False)
        keys = ('run_id', 'execution_status', 'compact_results', 'analysis_summary',
                'errors', 'validation_errors', 'clarification_questions', 'approval',
                'lifecycle', 'artifacts')
        result = {'handle': normalized.to_dict(), 'report_uri': report_uri(normalized),
                  'summary': {key: report[key] for key in keys if key in report}}
        if include_report:
            result['report'] = report
        return result

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=True, idempotentHint=True))
    @_tool_errors
    def cancel_task(handle: dict[str, Any]) -> dict[str, Any]:
        """Request cancellation of the exact submitted JobHandle. Does not delete results.

        Returns the executor state; cancellation support and completion semantics
        come from that executor. Repeating cancellation of a terminal job is safe.
        """
        return service.cancel_job(checked_handle(handle)).to_dict()

    register_resources(server, service, checked_handle)
    from .task_view import register_task_view
    register_task_view(server, service, checked_handle, _tool_errors, workbench=workbench)
    if study_service is not None:
        from .studies import register_studies
        register_studies(server, study_service, root, _tool_errors, workbench=workbench)
    return server
