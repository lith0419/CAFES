from __future__ import annotations

import copy
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.executors import TaskExecutor
from pyscf_agent.lifecycle import (
    ensure_lifecycle,
    synchronize_study_report_lifecycle,
    transition_lifecycle,
)

from ..adaptive import (
    analyze_dmrg_state_tracking,
    build_adaptive_initial_scan_plan,
    build_entanglement_active_space_review_plan,
    run_adaptive_study,
)
from ..adaptive.path_diagnostics import analyze_scan_path
from ..adaptive.mps_continuation import (
    build_mps_continuation_plan,
    merge_mps_continuation_report,
)
from ..adaptive.refinement import (
    build_casscf_active_space_probe_plan,
    build_casscf_active_space_review_plan,
    build_casscf_active_space_review_from_probe,
    build_path_refinement_plan,
    build_path_window_restart_plans,
    evaluate_path_window_restarts,
    merge_path_window_restart_reports,
)
from ..executor import run_study
from computational_study_agent.datasets.hamiltonian.contracts import (
    HamiltonianDatasetManifest,
)
from computational_study_agent.datasets.hamiltonian.finalize import (
    finalize_hamiltonian_dataset,
)
from computational_study_agent.datasets.hamiltonian.planner import (
    build_hamiltonian_dataset_study_plan,
)
from ..planner import build_study_plan
from ..postprocessing import run_postprocessing, suggest_plot_specs
from ..schema import StudyPlan, StudyReport, StudySpec
from ..validation import (
    ValidationIssue,
    has_errors,
    validate_study_plan,
    validate_study_spec,
)





from .types import (
    StudyApplicationError as StudyApplicationError,
    StudyApplicationValidationError,
    StudyFeatureUnavailableError as StudyFeatureUnavailableError,
    StudyArtifactError as StudyArtifactError,
    ArtifactContent as ArtifactContent,
    StudyPlanResult as StudyPlanResult,
)
from . import (
    preparation,
    saved_studies,
    execution,
    datasets,
    postprocessing,
    analysis,
    path_recovery,
    mps_continuation,
    retries,
    dmet_continuation,
    cancellation,
)




class StudyApplicationService:
    """Protocol-neutral application boundary for computational studies.

    Web, CLI, MCP, and background workers should call this service instead of
    importing planner or executor internals directly.  Constructor injection
    keeps the use cases testable and provides the future boundary for local and
    remote execution implementations.
    """

    def __init__(
        self,
        *,
        plan_builder: Callable[[Any], StudyPlan] = build_study_plan,
        study_runner: Callable[..., StudyReport] = run_study,
        adaptive_plan_builder: Callable[
            ..., Dict[str, Any]
        ] = build_adaptive_initial_scan_plan,
        adaptive_runner: Callable[..., Dict[str, Any]] = run_adaptive_study,
        postprocessor: Callable[..., Dict[str, Any]] = run_postprocessing,
        postprocess_suggester: Callable[..., List[Dict[str, Any]]] = suggest_plot_specs,
        path_analyzer: Callable[[Dict[str, Any]], Dict[str, Any]] = analyze_scan_path,
        path_refinement_builder: Callable[
            ..., Optional[Dict[str, Any]]
        ] = build_path_refinement_plan,
        path_restart_builder: Callable[
            ..., Optional[Dict[str, Any]]
        ] = build_path_window_restart_plans,
        path_restart_evaluator: Callable[
            ..., Dict[str, Any]
        ] = evaluate_path_window_restarts,
        path_restart_merger: Callable[
            ..., Dict[str, Any]
        ] = merge_path_window_restart_reports,
        mps_continuation_builder: Callable[
            ..., Optional[Dict[str, Any]]
        ] = build_mps_continuation_plan,
        mps_continuation_merger: Callable[
            ..., Dict[str, Any]
        ] = merge_mps_continuation_report,
        dmrg_state_tracker: Callable[
            [Dict[str, Any]], Dict[str, Any]
        ] = analyze_dmrg_state_tracking,
        entanglement_active_space_builder: Callable[
            ..., Optional[Dict[str, Any]]
        ] = build_entanglement_active_space_review_plan,
        active_space_probe_builder: Callable[
            ..., Optional[Dict[str, Any]]
        ] = build_casscf_active_space_probe_plan,
        active_space_review_builder: Callable[
            ..., Optional[Dict[str, Any]]
        ] = build_casscf_active_space_review_plan,
        active_space_probe_review_builder: Callable[
            ..., Optional[Dict[str, Any]]
        ] = build_casscf_active_space_review_from_probe,
        spec_validator: Callable[
            [StudySpec], List[ValidationIssue]
        ] = validate_study_spec,
        plan_validator: Callable[
            [StudyPlan], List[ValidationIssue]
        ] = validate_study_plan,
        hamiltonian_dataset_plan_builder: Callable[
            ..., StudyPlan
        ] = build_hamiltonian_dataset_study_plan,
        hamiltonian_dataset_finalizer: Callable[
            ..., HamiltonianDatasetManifest
        ] = finalize_hamiltonian_dataset,
        task_executor: Optional[TaskExecutor] = None,
        execution_config: Optional[Dict[str, Any]] = None,
        study_launcher: Optional[Callable[..., Dict[str, Any]]] = None,
    ):
        self._plan_builder = plan_builder
        self._study_runner = study_runner
        self._adaptive_plan_builder = adaptive_plan_builder
        self._adaptive_runner = adaptive_runner
        self._postprocessor = postprocessor
        self._postprocess_suggester = postprocess_suggester
        self._path_analyzer = path_analyzer
        self._path_refinement_builder = path_refinement_builder
        self._path_restart_builder = path_restart_builder
        self._path_restart_evaluator = path_restart_evaluator
        self._path_restart_merger = path_restart_merger
        self._mps_continuation_builder = mps_continuation_builder
        self._mps_continuation_merger = mps_continuation_merger
        self._dmrg_state_tracker = dmrg_state_tracker
        self._entanglement_active_space_builder = entanglement_active_space_builder
        self._active_space_probe_builder = active_space_probe_builder
        self._active_space_review_builder = active_space_review_builder
        self._active_space_probe_review_builder = active_space_probe_review_builder
        self._spec_validator = spec_validator
        self._plan_validator = plan_validator
        self._hamiltonian_dataset_plan_builder = hamiltonian_dataset_plan_builder
        self._hamiltonian_dataset_finalizer = hamiltonian_dataset_finalizer
        self._task_executor = task_executor
        self._execution_config = copy.deepcopy(execution_config)
        self._study_launcher = study_launcher

    @staticmethod
    def _coerce_spec(value: Any) -> StudySpec:
        if isinstance(value, StudySpec):
            return value
        if isinstance(value, dict):
            return StudySpec.from_dict(value)
        raise TypeError('StudySpec must be a StudySpec object or dictionary')

    @staticmethod
    def _coerce_plan(value: Any) -> StudyPlan:
        if isinstance(value, StudyPlan):
            return value
        if isinstance(value, dict):
            return StudyPlan.from_dict(value)
        raise TypeError('StudyPlan must be a StudyPlan object or dictionary')

    @staticmethod
    def _require_report(report: Any) -> Dict[str, Any]:
        if not isinstance(report, dict):
            raise TypeError('StudyReport must be a dictionary')
        return report

    @staticmethod
    def _begin_approved_refinement(
        report: Dict[str, Any],
        *,
        source: str,
        case_ids: Optional[Sequence[str]] = None,
    ) -> Dict[str, Any]:
        value = ensure_lifecycle(
            report.get('lifecycle'),
            'study',
            entity_id=report.get('study_id'),
        )
        details = {
            'source': source,
            'case_ids': [str(item) for item in (case_ids or [])],
        }
        if value['stage'] in ('planned', 'executing'):
            value = synchronize_study_report_lifecycle(
                value,
                entity_id=report.get('study_id'),
                decisions=report.get('gate_decisions') or [],
                details=details,
            )
        if value['stage'] == 'completed':
            value = transition_lifecycle(value, 'review_requested', details=details)
        if value['stage'] == 'review_required':
            value = transition_lifecycle(value, 'approval_granted', details=details)
        if value['stage'] == 'retry_ready':
            value = transition_lifecycle(value, 'refinement_started', details=details)
        report['lifecycle'] = value
        return value

    @staticmethod
    def _attach_parent_lifecycle(
        plan: Dict[str, Any],
        lifecycle: Dict[str, Any],
    ) -> Dict[str, Any]:
        payload = copy.deepcopy(plan)
        payload['lifecycle'] = copy.deepcopy(lifecycle)
        return payload

    @staticmethod
    def _raise_for_issues(stage: str, issues: Sequence[ValidationIssue]) -> None:
        if has_errors(issues):
            raise StudyApplicationValidationError(stage, issues)

    _execution_resource_policy = preparation._execution_resource_policy

    _bind_spec_resources = preparation._bind_spec_resources

    _bind_plan_resources = preparation._bind_plan_resources

    validate_spec = preparation.validate_spec

    validate_plan = preparation.validate_plan

    prepare_plan = preparation.prepare_plan

    build_plan = preparation.build_plan

    _study_directory = saved_studies._study_directory

    load_plan = saved_studies.load_plan

    load_report = saved_studies.load_report

    prepare_study = preparation.prepare_study

    _save_prepared_plan = saved_studies._save_prepared_plan

    _saved_report_path = saved_studies._saved_report_path

    _save_report = saved_studies._save_report

    complete_saved_review = saved_studies.complete_saved_review

    load_study_preparation = saved_studies.load_study_preparation

    start_study = saved_studies.start_study
    retry_context = retries.retry_context
    prepare_retry = retries.prepare_retry
    start_retry = retries.start_retry
    analyze_dmet_branches = dmet_continuation.analyze_dmet_branches
    prepare_dmet_continuation = dmet_continuation.prepare_dmet_continuation
    start_dmet_continuation = dmet_continuation.start_dmet_continuation

    open_study = saved_studies.open_study

    list_studies = saved_studies.list_studies

    _list_studies_in_root = saved_studies._list_studies_in_root

    _start_saved_study = saved_studies._start_saved_study

    collect_saved_study = saved_studies.collect_saved_study

    review_study = saved_studies.review_study

    _review_saved_study = saved_studies._review_saved_study

    prepare_dataset = datasets.prepare_dataset

    load_dataset = datasets.load_dataset

    build_hamiltonian_dataset_plan = datasets.build_hamiltonian_dataset_plan

    finalize_hamiltonian_dataset = datasets.finalize_hamiltonian_dataset

    compile_study_spec = preparation.compile_study_spec

    run_study = execution.run_study

    _finish_probe_review = execution._finish_probe_review

    build_adaptive_plan = preparation.build_adaptive_plan

    apply_review_action = saved_studies.apply_review_action

    run_adaptive_study = execution.run_adaptive_study

    reconcile_execution = execution.reconcile_execution

    collect_study = execution.collect_study

    collect_adaptive_study = execution.collect_adaptive_study

    stop_study = cancellation.stop_study
    inspect_execution = execution.inspect_execution

    prepare_active_space = preparation.prepare_active_space

    build_active_space_review = preparation.build_active_space_review

    build_active_space_probe = preparation.build_active_space_probe

    build_active_space_review_from_probe = (
        preparation.build_active_space_review_from_probe
    )

    @staticmethod
    def _review_report_scaffold(plan: Dict[str, Any]) -> Dict[str, Any]:
        return {
            'study_id': plan.get('study_id'),
            'name': plan.get('name'),
            'objective': plan.get('objective'),
            'system_type': plan.get('system_type'),
            'status': 'pending_review',
            'summary': 'Scientific review is required before the selected calculation can run.',
            'cases': [],
            'comparison_table': [],
            'artifacts': [],
        }

    _install_review_plan = preparation._install_review_plan

    run_postprocessing = postprocessing.run_postprocessing

    suggest_postprocessing = postprocessing.suggest_postprocessing

    postprocessing_context = postprocessing.postprocessing_context

    read_artifact = postprocessing.read_artifact

    prepare_result_analysis = analysis.prepare_result_analysis

    _attach_dmrg_state_tracking = analysis._attach_dmrg_state_tracking

    @staticmethod
    def _report_payload(value: Any) -> Dict[str, Any]:
        if isinstance(value, StudyReport):
            return value.to_dict()
        if isinstance(value, dict):
            return copy.deepcopy(value)
        raise TypeError('Study runner must return a StudyReport or dictionary')

    @staticmethod
    def _write_result_analysis_artifact(
        path: Path,
        payload: Dict[str, Any],
        *,
        kind: str,
        description: str,
    ) -> Dict[str, Any]:
        return default_artifact_repository().write_json(
            path,
            payload,
            kind=kind,
            description=description,
            atomic=True,
        )

    run_result_path_restarts = path_recovery.run_result_path_restarts

    build_result_path_refinement = path_recovery.build_result_path_refinement

    _completed_path_refinement_matches = staticmethod(
        path_recovery._completed_path_refinement_matches
    )

    build_result_entanglement_active_space = (
        path_recovery.build_result_entanglement_active_space
    )

    run_result_mps_continuation = mps_continuation.run_result_mps_continuation

    diagnose_results = analysis.diagnose_results

    analyze_results = analysis.analyze_results

    _interpret_result_analysis = analysis._interpret_result_analysis

    analyze_study = analysis.analyze_study

    _analyze_saved_results = analysis._analyze_saved_results

    _saved_case_checkpoints = saved_studies._saved_case_checkpoints
