"""Read-only MCP Apps presentation of the agent's saved Study and status."""

from __future__ import annotations

from collections import Counter
from datetime import datetime, timezone
from importlib.resources import files
from math import isfinite
from typing import Any

from mcp.types import ToolAnnotations


STUDY_VIEW_URI = 'ui://pyscf-agent/study-v1.html'
ROW_LIMIT = 20


def _text(value):
    return str(value)[:240] if value is not None else None


def _case_row(case):
    request = case.get('request') or {}
    task = case.get('task_report') or {}
    compact = task.get('compact_results') or {}
    active = request.get('active_space') or {}
    unit = (compact.get('energy_unit') or (task.get('structured_results') or {}).get('energy_unit')
            or (case.get('model_spec') or {}).get('energy_unit'))
    if not unit:
        unit = {'molecular': 'Ha', 'periodic': 'Ha/cell'}.get(request.get('task_type', 'molecular'))
    energy = compact.get('energy')
    if isinstance(energy, bool) or not isinstance(energy, (int, float)) or not isfinite(energy):
        energy = None
    return {
        'case_id': _text(case.get('case_id')), 'label': _text(case.get('label') or case.get('case_id')),
        'method': _text(request.get('method')), 'basis': _text(request.get('basis')),
        'active_space': {key: active.get(key) for key in ('nelecas', 'ncas')},
        'execution_status': _text(task.get('execution_status') or 'not_executed'),
        'energy': energy, 'energy_unit': _text(unit),
        'converged': compact.get('converged') if isinstance(compact.get('converged'), bool) else None,
    }


def study_view(saved, execution, *, workbench_available=False):
    """Project existing evidence; never decide convergence, collect or run work."""
    plan = saved.get('plan') or saved.get('initial_scan_plan') or {}
    report = saved.get('report') or {}
    cases = report.get('cases') or plan.get('cases') or []
    rows = [_case_row(case) for case in cases[:ROW_LIMIT]]
    # These are the agent's checkpoint counts, not reconstructed scheduler states.
    counts = execution.get('task_status_counts')
    if counts is None:
        counts = dict(Counter((case.get('task_report') or {}).get('execution_status', 'not_executed')
                              for case in cases))
        count_source = 'saved_report' if report.get('cases') else 'saved_plan'
    else:
        count_source = 'agent_task_state'
    # Live scheduler totals are separate: adaptive receipts may span multiple stages.
    scheduler_counts = Counter()
    for receipt in execution.get('receipts') or []:
        scheduler_counts.update((receipt.get('summary') or {}).get('state_counts') or {})
    workflow = execution.get('workflow') or report.get('workflow') or (report.get('adaptive') or {}).get('workflow') or {}
    return {
        'study_id': saved['study_id'], 'name': _text(plan.get('name') or saved['study_id']),
        'mode': saved.get('mode'), 'checked_at': datetime.now(timezone.utc).isoformat(),
        'execution_status': execution.get('status'), 'study_status': report.get('status'),
        'task_status_counts': counts, 'count_source': count_source,
        'scheduler_state_counts': dict(scheduler_counts),
        'review_required': bool(report.get('pending_review')) or report.get('status') == 'pending_review'
                           or execution.get('status') == 'review_required',
        'workflow_stage': _text(workflow.get('stage')),
        'rows': rows, 'case_count': len(cases), 'rows_source': 'saved_report' if report.get('cases') else 'saved_plan',
        'can_open_workbench': workbench_available,
        'plan_uri': 'pyscf://studies/' + saved['study_id'] + '/plan',
        'report_uri': 'pyscf://studies/' + saved['study_id'] + '/report',
    }


def register_study_view(server, service, root, tool_errors, checked_id, *, workbench=None):
    read_only = ToolAnnotations(readOnlyHint=True, idempotentHint=True)

    def snapshot(study_id):
        checked_id(study_id)
        saved = service.open_study(study_id, work_dir=str(root))
        execution = service.inspect_execution(study_id, work_dir=str(root))
        return study_view(saved, execution, workbench_available=workbench is not None)

    @server.tool(annotations=read_only, meta={'ui': {'resourceUri': STUDY_VIEW_URI, 'visibility': ['model']}})
    @tool_errors
    def show_study(study_id: str) -> dict[str, Any]:
        """Show a read-only Study card with status and up to 20 saved case results.

        UI-capable hosts render the card; other hosts receive the same structured
        summary. Refresh never collects, reviews or runs calculations. Energies
        and convergence are from the saved report and can precede a running retry.
        Use open_workbench for the full Study, or when inline UI is unavailable.
        """
        return snapshot(study_id)

    @server.tool(annotations=read_only, meta={'ui': {'visibility': ['app']}})
    @tool_errors
    def refresh_study_view(study_id: str) -> dict[str, Any]:
        """Refresh the Study card from existing agent state and saved report; never execute work."""
        return snapshot(study_id)

    @server.resource(STUDY_VIEW_URI, name='Study card', mime_type='text/html;profile=mcp-app',
                     meta={'ui': {'prefersBorder': True, 'csp': {'connectDomains': [], 'resourceDomains': []}}})
    def study_card() -> str:
        """Self-contained Study card; all data and actions pass through the host's MCP bridge."""
        return files('pyscf_agent.web_assets').joinpath('study-card.html').read_text(encoding='utf-8')
