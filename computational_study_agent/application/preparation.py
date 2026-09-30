"""Preparation use cases, composed by StudyApplicationService."""

from __future__ import annotations

import copy
from typing import Any, Dict, List, Optional, Tuple
from pyscf_agent.identifiers import make_run_id
from pyscf_agent.registry import (
    default_registry as default_platform_capability_registry,
)
from pyscf_agent.workflow_modules import compile_study_workflow
from pyscf_agent.workflow_gates import compile_study_gates
from ..gates.presentation import build_report_workflow
from ..normalization import (
    normalize_study_mode,
    normalize_system_type,
    supported_study_modes,
)
from ..costing import ensure_plan_cost_estimate
from ..gates.runtime import validate_study_gate_runtime
from ..module_runtime import validate_study_module_runtime
from ..adaptive.executor import save_adaptive_preparation
from ..schema import StudyPlan, StudySpec
from ..validation import ValidationIssue
from .types import StudyApplicationValidationError, StudyPlanResult
from pyscf_agent.artifacts import default_artifact_repository


ACTIVE_SPACE_REVIEW_ISSUE_CODES = {
    'missing_active_space_nelecas',
    'missing_active_space_ncas',
    'active_space_not_approved',
}


def _execution_resource_policy(
    self,
    resource_profile: Optional[str],
) -> Dict[str, Any]:
    """Return scheduler limits advertised by the selected executor profile."""

    if self._task_executor is None:
        return {}
    try:
        remote_capabilities = getattr(self._task_executor, 'remote_capabilities', None)
        metadata = (
            remote_capabilities()
            if callable(remote_capabilities)
            else self._task_executor.describe()
        )
    except Exception:
        # Planning remains available when a remote capability probe is
        # temporarily unreachable. Submission will still validate the
        # selected profile through the executor.
        return {}
    if not isinstance(metadata, dict):
        return {}
    selected_profile = str(resource_profile or 'auto').strip() or 'auto'
    limits = metadata.get('default_resource_limits')
    limits = copy.deepcopy(limits) if isinstance(limits, dict) else {}
    if selected_profile.lower() != 'auto':
        options = metadata.get('resource_profile_options')
        for item in options if isinstance(options, list) else []:
            if (
                isinstance(item, dict)
                and str(item.get('id') or '').strip() == selected_profile
            ):
                limits = copy.deepcopy(item)
                break
    try:
        memory_limit_mb = int(limits.get('memory_mb'))
    except (TypeError, ValueError):
        memory_limit_mb = 0
    if memory_limit_mb <= 0:
        return {}
    return {
        'memory_limit_mb': memory_limit_mb,
        'memory_limit_source': 'slurm_profile',
        'resource_profile': selected_profile,
    }


def _bind_spec_resources(
    self,
    spec: StudySpec,
    resource_profile: Optional[str],
) -> StudySpec:
    bound = copy.deepcopy(spec)
    selected_profile = str(resource_profile or 'auto').strip() or 'auto'
    if (
        bound.resource_policy.get('memory_limit_source') == 'slurm_profile'
        and bound.resource_policy.get('memory_limit_mb')
        and bound.resource_policy.get('resource_profile') == selected_profile
    ):
        return bound
    bound.resource_policy = {
        **copy.deepcopy(bound.resource_policy),
        **self._execution_resource_policy(resource_profile),
    }
    return bound


def _bind_plan_resources(
    self,
    plan: StudyPlan,
    resource_profile: Optional[str],
) -> StudyPlan:
    envelope = self._execution_resource_policy(resource_profile)
    if envelope:
        plan.resource_policy = {
            **copy.deepcopy(plan.resource_policy),
            **envelope,
        }
    ensure_plan_cost_estimate(plan)
    return plan


def validate_spec(self, spec: Any) -> Tuple[StudySpec, List[ValidationIssue]]:
    study_spec = self._coerce_spec(spec)
    issues = list(self._spec_validator(study_spec))
    return study_spec, issues


def validate_plan(self, plan: Any) -> Tuple[StudyPlan, List[ValidationIssue]]:
    study_plan = self._coerce_plan(plan)
    issues = list(self._plan_validator(study_plan))
    return study_plan, issues


def prepare_plan(
    self,
    spec: Any,
    *,
    resource_profile: Optional[str] = None,
) -> StudyPlanResult:
    study_spec, spec_issues = self.validate_spec(spec)
    self._raise_for_issues('study_spec', spec_issues)
    study_spec = self._bind_spec_resources(study_spec, resource_profile)
    try:
        plan = self._plan_builder(study_spec)
    except ValueError as exc:
        # Resolved case inputs are checked by their owning operation engine.
        # Preserve the same reviewable validation outcome for every adapter.
        raise StudyApplicationValidationError(
            'study_plan',
            [
                ValidationIssue(
                    severity='error',
                    code='study_plan_build_failed',
                    message=str(exc),
                )
            ],
        ) from exc
    ensure_plan_cost_estimate(plan)
    plan_issues = list(self._plan_validator(plan))
    self._raise_for_issues('study_plan', plan_issues)
    return StudyPlanResult(
        plan=plan,
        validation_issues=tuple(spec_issues + plan_issues),
    )


def build_plan(self, spec: Any, *, resource_profile: Optional[str] = None) -> StudyPlan:
    return self.prepare_plan(spec, resource_profile=resource_profile).plan


def prepare_study(
    self, spec: Any, *, work_dir=None, resource_profile=None, options=None
) -> Dict[str, Any]:
    """Expose the agent's existing static/adaptive planning as a saved use case."""
    study_spec = self._bind_spec_resources(self._coerce_spec(spec), resource_profile)
    if normalize_study_mode(study_spec.study_mode) == 'adaptive':
        prepared = self.build_adaptive_plan(
            study_spec, options=options, resource_profile=resource_profile
        )
        study_id = save_adaptive_preparation(
            study_spec.to_dict(), options, prepared, work_dir=work_dir
        )
        return {
            'study_id': study_id,
            'mode': 'adaptive',
            'plan': prepared,
            'validation_issues': [],
        }
    extra = {}
    try:
        result = self.prepare_plan(study_spec, resource_profile=resource_profile)
    except StudyApplicationValidationError as exc:
        extra = self.prepare_active_space(study_spec, exc.issues, options=options)
        if extra is None:
            raise
        plan = self._bind_plan_resources(
            StudyPlan.from_dict(extra['plan']), resource_profile
        )
        plan.study_id = make_run_id()
        plan.lifecycle['entity_id'] = plan.study_id
        extra['plan'] = plan.to_dict()
        for key in ('active_space_review_plan', 'active_space_probe_plan'):
            if key in extra:
                extra[key] = plan.to_dict()
        if extra.get('report'):
            report = extra['report']
            report.update(
                study_id=plan.study_id,
                work_dir=str(self._study_directory(plan.study_id, work_dir)),
            )
            report['lifecycle'] = copy.deepcopy(plan.lifecycle)
            report['adaptive']['direct_active_space_review_plan'] = plan.to_dict()
        result = StudyPlanResult(plan=plan, validation_issues=())
    directory = self._save_prepared_plan(result.plan, work_dir=work_dir)
    if extra.get('report'):
        default_artifact_repository().write_json(
            directory / 'study-report.json',
            extra['report'],
            kind='study-report',
            atomic=True,
        )
    return {
        **extra,
        'study_id': result.plan.study_id,
        'mode': 'static',
        'plan': result.plan.to_dict(),
        'validation_issues': [issue.to_dict() for issue in result.validation_issues],
    }


def compile_study_spec(
    self,
    spec: Any,
    *,
    study_mode: Optional[str] = None,
) -> Dict[str, Any]:
    """Compile study modules without executing any numerical cases."""

    study_spec = self._coerce_spec(spec)
    registry = default_platform_capability_registry()
    mode = normalize_study_mode(study_mode or study_spec.study_mode)
    if mode not in supported_study_modes(study_spec.system_type):
        raise ValueError(
            'Model Hamiltonian studies support static scans only; choose an explicit solver for the planned cases.'
        )
    configuration = compile_study_workflow(
        study_spec.to_dict(),
        registry.modules_for_template('study.template.' + mode),
        study_mode=mode,
    ).to_dict()
    validate_study_module_runtime(configuration)
    gate_configuration = compile_study_gates(
        study_spec.to_dict(),
        registry.gates_for(scope='study'),
    ).to_dict()
    validate_study_gate_runtime(gate_configuration)
    configuration['gate_configuration'] = gate_configuration
    return configuration


def build_adaptive_plan(
    self,
    spec: Any,
    *,
    options: Optional[Dict[str, Any]] = None,
    resource_profile: Optional[str] = None,
) -> Dict[str, Any]:
    study_spec = self._bind_spec_resources(self._coerce_spec(spec), resource_profile)
    if normalize_system_type(study_spec.system_type) != 'molecular':
        self._raise_for_issues(
            'adaptive_study',
            [
                ValidationIssue(
                    severity='error',
                    code='unsupported_study_mode',
                    message='Model Hamiltonian studies support static scans only; choose an explicit solver for the planned cases.',
                    path='study_mode',
                )
            ],
        )
    payload = self._adaptive_plan_builder(study_spec, options or {})
    issues: List[ValidationIssue] = []
    initial_scan_spec = payload.get('initial_scan_spec')
    if isinstance(initial_scan_spec, dict):
        _normalized_spec, stage_issues = self.validate_spec(initial_scan_spec)
        issues.extend(stage_issues)
    initial_scan_plan = payload.get('initial_scan_plan')
    if isinstance(initial_scan_plan, dict):
        _normalized_plan, stage_issues = self.validate_plan(initial_scan_plan)
        issues.extend(stage_issues)
    self._raise_for_issues('adaptive_initial_scan', issues)
    return payload


def prepare_active_space(
    self, spec: Any, validation_issues, *, options=None, study_state=None
):
    """Use the existing direct-CAS review or probe when this is the only missing input."""
    errors = [issue for issue in validation_issues if issue.severity == 'error']
    if not errors or any(
        issue.code not in ACTIVE_SPACE_REVIEW_ISSUE_CODES for issue in errors
    ):
        return None
    spec = self._coerce_spec(spec).to_dict()
    review = self.build_active_space_review(
        spec, options=options, study_state=study_state
    )
    if review and (review.get('active_space_review_plan') or {}).get('cases'):
        return {
            'status': 'active_space_review',
            'message': 'CASSCF/CASCI needs an approved active space. Review the ActiveSpaceAudit proposal and cost estimate.',
            'study_spec': spec,
            'plan': review['active_space_review_plan'],
            'active_space_review_plan': review['active_space_review_plan'],
            'active_space_review_decisions': review.get('active_space_review_decisions')
            or [],
            'report': review.get('report'),
            'validation_issues': [],
        }
    probe = self.build_active_space_probe(spec, options=options)
    if probe and (probe.get('active_space_probe_plan') or {}).get('cases'):
        return {
            'status': 'active_space_probe',
            'message': 'Run the molecular active-space probe to obtain case-specific orbital evidence, then review the ActiveSpaceAudit candidates.',
            'study_spec': spec,
            'plan': probe['active_space_probe_plan'],
            'active_space_probe_plan': probe['active_space_probe_plan'],
            'validation_issues': [],
        }
    return None


def build_active_space_review(
    self,
    spec: Any,
    *,
    options: Optional[Dict[str, Any]] = None,
    study_state: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    review = self._active_space_review_builder(spec, options or {})
    if not isinstance(review, dict):
        return None
    plan = review.get('active_space_review_plan')
    if not isinstance(plan, dict) or not isinstance(plan.get('cases'), list):
        return review
    review['report'] = self._install_review_plan(
        study_state,
        plan=plan,
        decisions=review.get('active_space_review_decisions'),
        plan_key='direct_active_space_review_plan',
        decision_key='direct_active_space_review_decisions',
        mode='direct_casscf_review',
    )
    return review


def build_active_space_probe(
    self,
    spec: Any,
    *,
    options: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    return self._active_space_probe_builder(spec, options or {})


def build_active_space_review_from_probe(
    self,
    probe_plan: Any,
    probe_report: Dict[str, Any],
    *,
    options: Optional[Dict[str, Any]] = None,
) -> Optional[Dict[str, Any]]:
    review = self._active_space_probe_review_builder(
        probe_plan,
        probe_report,
        options or {},
    )
    if not isinstance(review, dict):
        return None
    plan = review.get('active_space_review_plan')
    if not isinstance(plan, dict) or not isinstance(plan.get('cases'), list):
        return review
    review['report'] = self._install_review_plan(
        probe_report,
        plan=plan,
        decisions=review.get('active_space_review_decisions'),
        plan_key='direct_active_space_review_plan',
        decision_key='direct_active_space_review_decisions',
        mode='direct_casscf_review',
    )
    return review


def _install_review_plan(
    self,
    report: Optional[Dict[str, Any]],
    *,
    plan: Dict[str, Any],
    decisions: Any,
    plan_key: str,
    decision_key: str,
    mode: str,
) -> Dict[str, Any]:
    payload = (
        copy.deepcopy(report)
        if isinstance(report, dict)
        else self._review_report_scaffold(plan)
    )
    payload['status'] = 'pending_review'
    adaptive = (
        copy.deepcopy(payload.get('adaptive'))
        if isinstance(payload.get('adaptive'), dict)
        else {}
    )
    adaptive['mode'] = mode
    adaptive[plan_key] = copy.deepcopy(plan)
    adaptive[decision_key] = copy.deepcopy(decisions or [])
    payload['adaptive'] = adaptive
    adaptive['workflow'] = build_report_workflow(
        payload,
        analysis_requested=True,
    )
    return payload
