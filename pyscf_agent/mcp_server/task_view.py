"""Presentation-only tools over the calculation service's task inspection."""

from typing import Any

from mcp.types import ToolAnnotations

from ..task_monitor import task_monitor_html
from .resources import report_uri


TASK_VIEW_URI = 'ui://pyscf-agent/task-v1.html'


def register_task_view(server, service, checked_handle, tool_errors, *, workbench=None):
    read_only = ToolAnnotations(readOnlyHint=True, idempotentHint=True)

    def snapshot(handle):
        normalized = checked_handle(handle)
        view = service.inspect_task(normalized)
        return {**view, 'report_uri': report_uri(normalized),
                'can_open_workbench': workbench is not None}

    @server.tool(annotations=read_only, meta={'ui': {'resourceUri': TASK_VIEW_URI, 'visibility': ['model']}})
    @tool_errors
    def show_task(handle: dict[str, Any]) -> dict[str, Any]:
        """Show an existing Task's status, report summary and observed solver progress.

        Retain the exact JobHandle. Compatible MCP Apps hosts render a monitor
        that refreshes while running. Other hosts receive structured data;
        use open_task_monitor for a browser view. Never submits or retries.
        """
        return snapshot(handle)

    @server.tool(annotations=read_only, meta={'ui': {'visibility': ['app']}})
    @tool_errors
    def refresh_task_view(handle: dict[str, Any]) -> dict[str, Any]:
        """Read current task evidence for the monitor; never collect or execute."""
        return snapshot(handle)

    @server.resource(TASK_VIEW_URI, name='Task monitor', mime_type='text/html;profile=mcp-app',
                     meta={'ui': {'prefersBorder': True, 'csp': {'connectDomains': [], 'resourceDomains': []}}})
    def task_card() -> str:
        return task_monitor_html()

    if workbench is not None:
        @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
        @tool_errors
        def open_task_monitor(handle: dict[str, Any]) -> dict[str, Any]:
            """Open a read-only task monitor with automatic refresh in the local WebUI.

            Uses the exact existing JobHandle and configured executor. Starts or
            reuses only the Web service, never a calculation. Open the returned
            URL in the host browser. Remote solver details need an updated runtime.
            """
            normalized = checked_handle(handle)
            service.job_status(normalized)
            return workbench.open_task(normalized)
