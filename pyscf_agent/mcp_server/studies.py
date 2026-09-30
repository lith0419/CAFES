"""MCP projections over the existing Study application service and artifacts."""

from __future__ import annotations

import json
import re
from typing import Any, Optional

from mcp.server.mcpserver.exceptions import ResourceError
from mcp.types import ToolAnnotations


def register_studies(server, service, root, tool_errors, *, workbench=None):
    read_only = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
    writes = ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=False)

    def checked_id(study_id):
        if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,119}', study_id):
            raise ValueError('study_id must be 1-120 letters, digits, underscores or hyphens')
        return study_id

    def uris(study_id):
        result = {'plan_uri': 'pyscf://studies/' + study_id + '/plan',
                  'report_uri': 'pyscf://studies/' + study_id + '/report'}
        url = workbench.existing_url(study_id) if workbench is not None else None
        if url:
            result['workbench_url'] = url
        return result

    from .study_view import register_study_view
    register_study_view(server, service, root, tool_errors, checked_id, workbench=workbench)

    if workbench is not None:
        @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
        @tool_errors
        def open_workbench(study_id: Optional[str] = None) -> dict[str, Any]:
            """Start or reuse the local Study WebUI and return a browser URL.

            Pass a saved Study ID to open its plan, reviews and results, or omit
            it for the workbench home. Uses this MCP server's Python environment,
            work directory and executor. Runs no calculations. The loopback Web
            server stays available after MCP disconnects; opening a URL is a
            separate browser action. Call again if an old link stops responding.
            """
            if study_id is not None:
                service.load_study_preparation(checked_id(study_id), work_dir=str(root))
            return workbench.open(study_id)

    @server.tool(annotations=writes)
    @tool_errors
    def prepare_study(study_spec: dict[str, Any], resource_profile: Optional[str] = None,
                      options: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Ask the agent to prepare and save a static or adaptive Study without executing it.

        Returns the complete plan, stable study_id, cost estimate and resource URIs.
        study_spec.study_mode selects static or adaptive. options uses the agent's
        existing adaptive options. Keep study_id and use submit_study to execute.
        Preparing again creates a new Study.
        """
        prepared = service.prepare_study(study_spec, work_dir=str(root), resource_profile=resource_profile, options=options)
        return {**prepared, **uris(prepared['study_id'])}

    @server.tool(annotations=writes)
    @tool_errors
    def submit_study(study_id: str,
                     rerun_case_ids: Optional[list[str]] = None, rerun_statuses: Optional[list[str]] = None,
                     max_case_attempts: int = 2, resource_profile: Optional[str] = None,
                     locale: str = 'en') -> dict[str, Any]:
        """Start the agent's complete Study workflow in the background and return immediately.

        The agent runs all static tasks, or its existing adaptive stages, until
        completion, required review or interruption. Do not submit one task at a
        time. Repeating while running returns the existing invocation. Reconnect
        with the same Study ID to inspect/collect. Explicit later starts use the
        agent's existing resume/retry rules and approval gates. review_study
        saves the chosen action in the agent; start it using the same Study ID.
        """
        checked_id(study_id)
        result = service.start_study(study_id, work_dir=str(root), locale=locale,
                                      rerun_case_ids=rerun_case_ids, rerun_statuses=rerun_statuses,
                                      max_case_attempts=max_case_attempts, resource_profile=resource_profile)
        return {**result, **uris(study_id)}

    @server.tool(annotations=read_only)
    @tool_errors
    def get_study_status(study_id: str) -> dict[str, Any]:
        """Inspect the agent invocation, task receipts and review requirements.

        While the agent is running, wait for it to complete or request review.
        can_collect is false during orchestration. Scheduler and scientific
        outcomes remain separate; read study_status and the full StudyReport.
        """
        checked_id(study_id)
        service.load_study_preparation(study_id, work_dir=str(root))
        return {**service.inspect_execution(study_id, work_dir=str(root)), **uris(study_id)}

    @server.tool(annotations=writes)
    @tool_errors
    def analyze_dmet_branches(study_id: str, policy: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        """Compare all saved DMET candidates by final order parameters and energy.

        Writes phase maps, energy curves, and candidate crossing brackets. Retains
        independent and continuation Run provenance. No calculations are submitted.
        Thresholds use absolute sublattice half-differences; missing quality evidence
        cannot qualify a Run. Numerical failures are not physical branch endpoints.
        """
        checked_id(study_id)
        return {'study_id': study_id, 'analysis': service.analyze_dmet_branches(
            study_id, policy=policy, work_dir=str(root)), **uris(study_id)}

    @server.tool(annotations=writes)
    @tool_errors
    def prepare_dmet_continuation(study_id: str, case_ids: Optional[list[str]] = None,
                                  branches: Optional[list[str]] = None, mode: str = 'bidirectional',
                                  policy: Optional[dict[str, Any]] = None, mixing: float = 0.2,
                                  max_iterations: int = 200) -> dict[str, Any]:
        """Preview a bounded density-only continuation sweep on an existing Study.

        Defaults to AFM increasing V and CDW decreasing V, at fixed U. Nearest mode
        selects compatible neighbors in U/V. Donors are reselected from qualified
        final states before each step, including historical Runs. Saves a reviewable
        action, maximum Run count, and available cost estimates; submits no calculations. Keep action_id and start
        it with start_dmet_continuation. Only density is transferred; u starts zero.
        """
        checked_id(study_id)
        return {**service.prepare_dmet_continuation(study_id, case_ids=case_ids, branches=branches,
            mode=mode, policy=policy, mixing=mixing, max_iterations=max_iterations, work_dir=str(root)), **uris(study_id)}

    @server.tool(annotations=writes)
    @tool_errors
    def start_dmet_continuation(study_id: str, action_id: str, approve_cost: bool = False,
                                locale: str = 'en') -> dict[str, Any]:
        """Start a prepared density-only sweep using the Study's existing retry runner.

        Repeated starts of the same action do not resubmit. Inspect and collect
        through get_study_status/collect_study. If interrupted, collect existing
        Runs before previewing another sweep. Set approve_cost only after approval
        of the prepared sweep estimate. Some impurity solvers have no resource estimate.
        """
        checked_id(study_id)
        return {**service.start_dmet_continuation(study_id, action_id, approve_cost=approve_cost,
            locale=locale, work_dir=str(root)), **uris(study_id)}

    @server.tool(annotations=ToolAnnotations(readOnlyHint=False, destructiveHint=False, idempotentHint=True))
    @tool_errors
    def collect_study(study_id: str, include_report: bool = False, locale: str = 'en') -> dict[str, Any]:
        """Read the agent's complete StudyReport, or recover existing runs after interruption.

        Does not launch another Study invocation. While the agent is running,
        inspect later. Scientific
        failures remain in the report. Read report_uri for the complete report.
        """
        checked_id(study_id)
        report = service.collect_saved_study(study_id, work_dir=str(root), locale=locale)
        keys = ('study_id', 'status', 'summary', 'comparison_table', 'lifecycle', 'workflow', 'gate_decisions', 'artifacts', 'dataset_manifest', 'pending_review')
        result = {'study_id': study_id, 'summary': {key: report[key] for key in keys if key in report}, **uris(study_id)}
        if include_report:
            result['report'] = report
        return result

    @server.tool(annotations=writes)
    @tool_errors
    def review_study(study_id: str, action_id: str, case_ids: Optional[list[str]] = None,
                     approval_token: Optional[str] = None, plan_kind: Optional[str] = None) -> dict[str, Any]:
        """Save an existing review action on the Study; never execute calculations.

        Use displayed action IDs and case IDs. Approval actions require the user's
        decision. The agent stores the reviewed plan and workflow. Call submit_study
        with the Study ID to execute after can_run becomes true, even after reconnect.
        """
        checked_id(study_id)
        result = service.review_study(study_id, action_id, work_dir=str(root), case_ids=case_ids,
                                      approval_token=approval_token, plan_kind=plan_kind)
        return {'study_id': study_id, 'action_id': result['action_id'], 'message': result['message'],
                'saved': True, 'can_run': result['can_run'],
                'case_ids': result['case_ids'], 'workflow': result['workflow'], 'lifecycle': result['lifecycle'],
                **uris(study_id)}

    @server.tool(annotations=writes)
    @tool_errors
    def analyze_study(study_id: str, postprocess: bool = False,
                      include_report: bool = False, locale: str = 'en') -> dict[str, Any]:
        """Ask the agent to diagnose saved results without an LLM or new calculations.

        Returns scientific diagnostics, continuation/review proposals and suggested
        plots. postprocess also writes the agent's existing plot/data artifacts.
        Results and review evidence are saved in StudyReport. Approve displayed
        follow-up actions through review_study, then start them explicitly.
        """
        checked_id(study_id)
        result = service.analyze_study(study_id, work_dir=str(root), locale=locale, postprocess=postprocess)
        return {'study_id': study_id, **{key: value for key, value in result.items()
                if include_report or key != 'report'}, **uris(study_id)}

    @server.tool(annotations=writes)
    @tool_errors
    def prepare_dataset(dataset_spec: dict[str, Any], seed_geometries: list[dict[str, Any]],
                        resource_profile: Optional[str] = None) -> dict[str, Any]:
        """Use the agent's HamiltonianDatasetSpec planner to save one Study for MD sampling.

        seed_geometries uses MolecularGeometry objects (molecule_id, geometry_id,
        atomic_numbers, positions, coordinate_unit). Preparation runs no MD.
        Review the cost estimate, then submit_study/collect_study using its ID.
        The existing runner assembles accepted/rejected samples and the manifest.
        """
        prepared = service.prepare_dataset(dataset_spec, seed_geometries,
            work_dir=str(root), resource_profile=resource_profile)
        return {**prepared, **uris(prepared['study_id']),
                'dataset_uri': 'pyscf://studies/' + prepared['study_id'] + '/dataset'}

    @server.resource('pyscf://studies/{study_id}/dataset', mime_type='application/json')
    def study_dataset(study_id: str) -> str:
        """Read the dataset manifest already assembled by the existing Study runner."""
        try:
            return json.dumps(service.load_dataset(checked_id(study_id), work_dir=str(root)), ensure_ascii=False)
        except (ValueError, OSError) as exc:
            raise ResourceError(str(exc)) from exc

    @server.resource('pyscf://studies/{study_id}/plan', mime_type='application/json')
    def study_plan(study_id: str) -> str:
        """Read the saved static plan or adaptive preparation. Saved review decisions are in the report."""
        try:
            prepared = service.load_study_preparation(checked_id(study_id), work_dir=str(root))
            return json.dumps(prepared.get('plan', prepared), ensure_ascii=False)
        except (ValueError, OSError) as exc:
            raise ResourceError(str(exc)) from exc

    @server.resource('pyscf://studies/{study_id}/report', mime_type='application/json')
    def study_report(study_id: str) -> str:
        """Read the saved StudyReport. Call collect_study to refresh it from existing runs."""
        try:
            return json.dumps(service.load_report(checked_id(study_id), work_dir=str(root)), ensure_ascii=False)
        except (ValueError, OSError) as exc:
            raise ResourceError(str(exc)) from exc
