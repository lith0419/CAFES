from __future__ import annotations

from pyscf_agent.serialization import json_default

import copy
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

from pyscf_agent.artifacts import default_artifact_repository
from pyscf_agent.identifiers import make_run_id
from pyscf_agent.paths import resolve_work_dir
from pyscf_agent.executors import TaskExecutor
from pyscf_agent.lifecycle import begin_study_execution
from pyscf_agent.schema_contracts import ADAPTIVE_STUDY_REPORT_SCHEMA
from pyscf_agent.workflow_gates import gate_artifact_documents

from .active_space_contracts import (
    active_space_contract_from_case_report as _active_space_contract_from_case_report,
    structured_results_from_case_report as _structured_results_from_case_report,
)
from .initial_scan import (
    _as_study_spec_payload,
    _dedupe,
    _normalize_options,
    build_adaptive_initial_scan_plan,
    build_initial_scan_study_spec as build_initial_scan_study_spec,
    normalize_adaptive_options as normalize_adaptive_options,
)
from .refinement import (
    _merge_recovery_report,
    _recovery_solver,
    build_recovery_plan_for_unconverged_refined_cases as build_recovery_plan_for_unconverged_refined_cases,
    build_recovery_plan_for_unresolved_refined_cases,
    build_refined_plan_from_decisions,
)
from .path_diagnostics import deferred_scan_path_diagnostics
from ..gates.presentation import build_report_workflow
from ..costing import CostApprovalRequired
from ..executor import collect_study, run_study
from ..normalization import normalize_system_type
from ..schema import StudyPlan
from ..locking import study_lock


def _deepcopy_json(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False, default=json_default, allow_nan=False))


def _adaptive_study_id_for_resume(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    token = str(value).strip()
    if not token:
        return None
    if Path(token).name != token or token in ('.', '..'):
        raise ValueError('resume_study_id must be a study identifier, not a path')
    return token


_ADAPTIVE_REQUEST_SCHEMA = 'pyscf-agent.adaptive-request.v1'
_ADAPTIVE_REQUEST_FILENAME = 'adaptive-study-request.json'
_ADAPTIVE_DECISION_LOG_SCHEMA = 'pyscf-agent.adaptive-decision-log.v1'
_ADAPTIVE_STUDY_REPORT_SCHEMA = ADAPTIVE_STUDY_REPORT_SCHEMA


def _adaptive_request_fingerprint(study_spec_payload: Dict[str, Any], adaptive_options: Dict[str, Any]) -> str:
    """Fingerprint the computational request before reusing an adaptive run directory."""
    canonical = json.dumps(
        {
            'study_spec': study_spec_payload,
            'adaptive_options': adaptive_options,
        },
        ensure_ascii=False,
        sort_keys=True,
        separators=(',', ':'),
        default=json_default, allow_nan=False,
    )
    return hashlib.sha256(canonical.encode('utf-8')).hexdigest()


def _adaptive_resume_matches_request(adaptive_work_dir: Path, fingerprint: str) -> bool:
    manifest_path = adaptive_work_dir / _ADAPTIVE_REQUEST_FILENAME
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding='utf-8'))
    except (OSError, ValueError):
        return False
    return (
        isinstance(manifest, dict)
        and manifest.get('schema') == _ADAPTIVE_REQUEST_SCHEMA
        and manifest.get('input_fingerprint') == fingerprint
    )


def _prepare_adaptive_work_dir(
    work_dir: Optional[str],
    resume_study_id: Optional[str],
    requested_study_id: Optional[str],
    study_spec_payload: Dict[str, Any],
    adaptive_options: Dict[str, Any],
    *, persist: bool = True,
) -> tuple[str, Path]:
    root = resolve_work_dir(work_dir)
    fingerprint = _adaptive_request_fingerprint(study_spec_payload, adaptive_options)
    requested_id = _adaptive_study_id_for_resume(resume_study_id)
    requested_new_id = _adaptive_study_id_for_resume(requested_study_id)
    if requested_id:
        candidate_dir = root / requested_id
        if _adaptive_resume_matches_request(candidate_dir, fingerprint):
            adaptive_id = requested_id
            adaptive_work_dir = candidate_dir
        else:
            adaptive_id = make_run_id()
            adaptive_work_dir = root / adaptive_id
    elif requested_new_id and not (root / requested_new_id).exists():
        adaptive_id = requested_new_id
        adaptive_work_dir = root / adaptive_id
    else:
        adaptive_id = make_run_id()
        adaptive_work_dir = root / adaptive_id
    if persist:
        with study_lock(adaptive_work_dir):
            _write_adaptive_request(adaptive_work_dir, study_spec_payload, adaptive_options)
    return adaptive_id, adaptive_work_dir


def _write_adaptive_request(directory, study_spec_payload, adaptive_options):
    _write_json(
        directory / _ADAPTIVE_REQUEST_FILENAME,
        {
            'schema': _ADAPTIVE_REQUEST_SCHEMA,
            'input_fingerprint': _adaptive_request_fingerprint(study_spec_payload, adaptive_options),
            'study_spec': study_spec_payload,
            'adaptive_options': adaptive_options,
        },
        kind='adaptive-request-manifest',
        description='Fingerprint used to validate safe adaptive-study resume requests',
    )


def save_adaptive_preparation(spec, options, prepared, *, work_dir=None):
    """Save the already validated input needed to start or resume by Study ID."""
    study_id, directory = _prepare_adaptive_work_dir(
        work_dir, None, None, _as_study_spec_payload(spec), _normalize_options(options),
    )
    _write_json(directory / 'adaptive-initial-scan-plan.json', prepared['initial_scan_plan'],
                kind='adaptive-initial-scan-plan')
    return study_id


def save_adaptive_cost_approval(directory, spec, options):
    """Persist an approved initial request under its existing Study identity."""
    directory = Path(directory)
    saved = json.loads((directory / _ADAPTIVE_REQUEST_FILENAME).read_text(encoding='utf-8'))
    before, after = copy.deepcopy(saved['study_spec']), copy.deepcopy(spec)
    for item in (before, after):
        item.setdefault('resource_policy', {}).pop('approved', None)
    if before != after or saved['adaptive_options'] != options:
        raise ValueError('Initial cost approval cannot change the saved adaptive scientific request')
    saved.update(study_spec=spec, input_fingerprint=_adaptive_request_fingerprint(spec, options))
    _write_json(directory / _ADAPTIVE_REQUEST_FILENAME, saved, kind='adaptive-request-manifest')


def _write_json(path: Path, payload: Any, *, kind: str, description: str = '') -> Dict[str, Any]:
    return default_artifact_repository().write_json(
        path,
        payload,
        kind=kind,
        description=description,
    )


def _append_gate_artifacts(
    report: Dict[str, Any],
    output_dir: Path,
    artifacts: List[Dict[str, Any]],
) -> None:
    """Persist the same gate contract documents for every adaptive exit path."""

    for document in gate_artifact_documents(report):
        artifact = _write_json(
            output_dir / document['filename'],
            document['payload'],
            kind=document['kind'],
            description=document['description'],
        )
        artifacts[:] = [item for item in artifacts if item.get('path') != artifact['path']]
        artifacts.append(artifact)


def _case_report_by_id(report_payload: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    cases = report_payload.get('cases') if isinstance(report_payload.get('cases'), list) else []
    return {
        str(item.get('case_id')): item
        for item in cases
        if isinstance(item, dict) and item.get('case_id') is not None
    }


def _initial_scan_method(case_report: Dict[str, Any], row: Dict[str, Any]) -> Optional[str]:
    for value in (
        row.get('solver') if isinstance(row, dict) else None,
        row.get('method') if isinstance(row, dict) else None,
        (case_report.get('request') or {}).get('solver') if isinstance(case_report, dict) and isinstance(case_report.get('request'), dict) else None,
        (case_report.get('request') or {}).get('method') if isinstance(case_report, dict) and isinstance(case_report.get('request'), dict) else None,
    ):
        normalized = str(value or '').strip().lower()
        if normalized:
            return normalized
    return None


def _adaptive_initial_scan_artifacts(
    adaptive_work_dir: Path,
    initial_scan_payload: Dict[str, Any],
    initial_scan_report: Dict[str, Any],
) -> List[Dict[str, Any]]:
    return [
        _write_json(
            adaptive_work_dir / 'adaptive-initial-scan-plan.json',
            initial_scan_payload['initial_scan_plan'],
            kind='adaptive-initial-scan-plan',
            description='Initial diagnostic scan plan used before adaptive method selection',
        ),
        _write_json(
            adaptive_work_dir / 'adaptive-initial-scan-report.json',
            initial_scan_report,
            kind='adaptive-initial-scan-report',
            description='Initial-scan diagnostics used for adaptive method selection',
        ),
    ]


def _case_report_system_type(case_report: Dict[str, Any]) -> str:
    structured = _structured_results_from_case_report(case_report)
    if structured.get('task_type'):
        return normalize_system_type(structured.get('task_type'))
    if isinstance(structured.get('strong_correlation_diagnostics'), dict):
        return 'model_hamiltonian'
    request = case_report.get('request') if isinstance(case_report, dict) else {}
    if isinstance(request, dict):
        if request.get('task_type'):
            return normalize_system_type(request.get('task_type'))
        if isinstance(request.get('model_hamiltonian'), dict):
            return 'model_hamiltonian'
    return 'molecular'


def _diagnostics_from_case_report(case_report: Dict[str, Any]) -> Dict[str, Any]:
    structured = _structured_results_from_case_report(case_report)
    diagnostics = structured.get('strong_correlation_diagnostics') if isinstance(structured, dict) else {}
    if isinstance(diagnostics, dict) and diagnostics:
        return diagnostics
    diagnostics = structured.get('correlation_diagnostics') if isinstance(structured, dict) else {}
    if isinstance(diagnostics, dict) and diagnostics:
        return diagnostics
    return {}


def _is_molecular_diagnostics(diagnostics: Dict[str, Any]) -> bool:
    if not isinstance(diagnostics, dict):
        return False
    kind = str(diagnostics.get('kind') or '').strip().lower()
    return kind in ('molecular_correlation_diagnostics', 'molecular_correlation_risk') or isinstance(diagnostics.get('molecular_correlation_risk'), dict)


def _active_space_contract_inconsistent(active_space_contract: Optional[Dict[str, Any]]) -> bool:
    if not isinstance(active_space_contract, dict):
        return False
    audit_summary = active_space_contract.get('audit_summary')
    if isinstance(audit_summary, dict) and audit_summary.get('ncas_nelecas_consistent') is False:
        return True
    audit = active_space_contract.get('audit')
    consistency = audit.get('ncas_nelecas_consistency') if isinstance(audit, dict) else None
    return bool(isinstance(consistency, dict) and consistency.get('consistent') is False)


def _case_report_site_count(case_report: Dict[str, Any], diagnostics: Dict[str, Any] = None) -> Optional[int]:
    parameter_summary = diagnostics.get('parameter_summary') if isinstance(diagnostics, dict) and isinstance(diagnostics.get('parameter_summary'), dict) else {}
    candidates = [
        parameter_summary.get('site_count'),
        _structured_results_from_case_report(case_report).get('site_count'),
    ]
    request = case_report.get('request') if isinstance(case_report, dict) else {}
    model_spec = {}
    if isinstance(request, dict):
        model_hamiltonian = request.get('model_hamiltonian')
        if isinstance(model_hamiltonian, dict) and isinstance(model_hamiltonian.get('spec'), dict):
            model_spec = model_hamiltonian['spec']
    if isinstance(model_spec.get('sites'), list):
        candidates.append(len(model_spec['sites']))
    for value in candidates:
        try:
            return int(value)
        except (TypeError, ValueError):
            continue
    return None


def _initial_scan_failure_reasons(case_report: Dict[str, Any]) -> List[str]:
    task_report = case_report.get('task_report') if isinstance(case_report, dict) else {}
    if not isinstance(task_report, dict):
        return ['Initial-scan case did not produce a TaskReport.']
    reasons: List[str] = []
    validation_errors = task_report.get('validation_errors')
    if isinstance(validation_errors, list):
        reasons.extend(str(item) for item in validation_errors if item)
    raw_stderr = task_report.get('raw_stderr')
    if isinstance(raw_stderr, str) and raw_stderr.strip():
        for line in raw_stderr.splitlines():
            if line.strip() and line.strip() not in reasons:
                reasons.append(line.strip())
    analysis_summary = task_report.get('analysis_summary')
    if isinstance(analysis_summary, str) and analysis_summary.strip() and analysis_summary.strip() not in reasons:
        reasons.append(analysis_summary.strip())
    return reasons or ['Initial-scan case did not succeed, so no reliable adaptive method choice is available.']


def _decision_tags(level: str, score: Optional[float]) -> List[str]:
    tags = [level or 'unknown']
    if level in ('moderate', 'strong'):
        tags.append('needs_refined_method')
    if score is not None and (0.3 <= score <= 0.4 or 0.62 <= score <= 0.72):
        tags.append('near_decision_boundary')
    return tags


def _recommended_solver(
    level: str,
    diagnostics: Dict[str, Any],
    options: Dict[str, Any],
    site_count: Optional[int] = None,
) -> str:
    if level not in ('weak', 'moderate', 'strong'):
        raise ValueError('Cannot route an unknown correlation level: {0}'.format(level))
    policy = options['method_policy']
    parameter_summary = diagnostics.get('parameter_summary') if isinstance(diagnostics.get('parameter_summary'), dict) else {}
    site_count_value = site_count if site_count is not None else parameter_summary.get('site_count')
    try:
        site_count_int = int(site_count_value)
    except (TypeError, ValueError):
        site_count_int = options['max_fci_sites'] + 1
    if level == 'strong':
        return policy['strong_small'] if site_count_int <= options['max_fci_sites'] else policy['strong_large']
    if level == 'moderate':
        return policy['moderate']
    return policy['weak']


def _recommended_molecular_method(level: str, diagnostics: Dict[str, Any], options: Dict[str, Any]) -> str:
    if level not in ('weak', 'moderate', 'strong'):
        raise ValueError('Cannot route an unknown correlation level: {0}'.format(level))
    policy = options['molecular_method_policy']
    active_space = diagnostics.get('active_space_contract') if isinstance(diagnostics.get('active_space_contract'), dict) else {}
    ncas = active_space.get('ncas')
    try:
        ncas_int = int(ncas)
    except (TypeError, ValueError):
        ncas_int = None
    if level == 'strong':
        if ncas_int is not None and ncas_int > options['max_cas_orbitals']:
            return policy['strong_large']
        return policy['strong_small']
    if level == 'moderate':
        return policy['moderate']
    return policy['weak']


def _molecular_decision_tags(
    level: str,
    score: Optional[float],
    active_space_contract: Optional[Dict[str, Any]],
    options: Dict[str, Any],
) -> List[str]:
    tags = _decision_tags(level, score)
    if level == 'strong':
        tags.append('review_active_space')
        tags.append('requires_active_space_approval')
        if not active_space_contract:
            tags.append('missing_active_space_candidate')
    if isinstance(active_space_contract, dict):
        if _active_space_contract_inconsistent(active_space_contract):
            tags.append('active_space_inconsistent')
            tags.append('review_active_space')
            tags.append('requires_active_space_approval')
        try:
            if int(active_space_contract.get('ncas')) > options['max_cas_orbitals']:
                tags.append('large_active_space_resource_review_required')
        except (TypeError, ValueError):
            pass
    return _dedupe(tags)


def _decision_reasons(diagnostics: Dict[str, Any]) -> List[str]:
    reasons = []
    if diagnostics.get('summary'):
        reasons.append(str(diagnostics['summary']))
    if _is_molecular_diagnostics(diagnostics):
        risk = diagnostics.get('molecular_correlation_risk') if isinstance(diagnostics.get('molecular_correlation_risk'), dict) else diagnostics
        level = risk.get('level') or diagnostics.get('level')
        reasons.append('MolecularCorrelationRisk={0}; routing used internal near-degeneracy and solver-stress diagnostics.'.format(
            level or 'unknown',
        ))
        for component in list(risk.get('physics_components') or []) + list(risk.get('solver_stress_components') or []):
            if not isinstance(component, dict):
                continue
            try:
                component_score = float(component.get('score'))
            except (TypeError, ValueError):
                component_score = 0.0
            if component_score >= 0.35:
                reasons.append('{0}: {1}'.format(component.get('name'), component.get('interpretation')))
        recommendation = risk.get('method_recommendation')
        if isinstance(recommendation, dict) and recommendation.get('next_step'):
            reasons.append(str(recommendation['next_step']))
        return reasons[:6]
    for item in diagnostics.get('diagnostics') or []:
        if not isinstance(item, dict):
            continue
        severity = item.get('severity')
        if severity in ('moderate', 'strong'):
            reasons.append('{0}: {1}'.format(item.get('name'), item.get('interpretation')))
    return reasons[:6]


def _molecular_risk_levels(diagnostics: Dict[str, Any]) -> Dict[str, Any]:
    if not _is_molecular_diagnostics(diagnostics):
        return {}
    risk = (
        diagnostics.get('molecular_correlation_risk')
        if isinstance(diagnostics.get('molecular_correlation_risk'), dict)
        else diagnostics
    )
    return {
        'diagnostic_level': risk.get('level') or diagnostics.get('level'),
        'physics_level': risk.get('physics_level') or diagnostics.get('physics_level'),
        'solver_stress_level': risk.get('solver_stress_level') or diagnostics.get('solver_stress_level'),
        'routing_level': risk.get('routing_level') or diagnostics.get('routing_level'),
    }


def analyze_initial_scan_report(report_payload: Dict[str, Any], options: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    adaptive_options = _normalize_options(options)
    decisions = []
    case_reports = _case_report_by_id(report_payload)
    rows = report_payload.get('comparison_table') if isinstance(report_payload.get('comparison_table'), list) else []
    for row in rows:
        if not isinstance(row, dict):
            continue
        case_id = str(row.get('case_id') or '')
        case_report = case_reports.get(case_id, {})
        initial_scan_status = row.get('status')
        diagnostics = _diagnostics_from_case_report(case_report)
        system_type = _case_report_system_type(case_report)
        initial_scan_method = _initial_scan_method(case_report, row)
        active_space_contract = _active_space_contract_from_case_report(case_report)
        if initial_scan_status == 'unconverged':
            try:
                score = float(diagnostics.get('score')) if diagnostics.get('score') is not None else None
            except (TypeError, ValueError):
                score = None
            if system_type == 'molecular' or _is_molecular_diagnostics(diagnostics):
                solver = adaptive_options['molecular_method_policy']['moderate']
            else:
                site_count = _case_report_site_count(case_report, diagnostics)
                solver = _recovery_solver(row.get('solver'), site_count, adaptive_options)
            reasons = _initial_scan_failure_reasons(case_report)
            reasons.insert(0, 'Initial-scan solver did not converge; retrying this region with {0}.'.format(solver.upper()))
            tags = ['initial_scan_unconverged', 'needs_method_recovery']
            if (system_type == 'molecular' or _is_molecular_diagnostics(diagnostics)) and active_space_contract:
                tags.append('review_active_space')
                reasons.insert(
                    1,
                    'The unconverged probe still produced a consistent ActiveSpaceAudit CAS({0}, {1}); preserve it for any multireference recovery.'.format(
                        active_space_contract.get('nelecas'),
                        active_space_contract.get('ncas'),
                    ),
                )
            risk_levels = _molecular_risk_levels(diagnostics)
            decisions.append({
                'case_id': case_id,
                'label': row.get('label') or case_id,
                'variables': copy.deepcopy(case_report.get('variables') or {}),
                'initial_scan_status': initial_scan_status,
                'initial_scan_method': initial_scan_method,
                'initial_scan_run_dir': row.get('run_dir'),
                'level': str(diagnostics.get('level') or 'initial_scan_unconverged'),
                **risk_levels,
                'score': score,
                'confidence': diagnostics.get('confidence') or 'low',
                'recommended_solver': solver,
                'recommended_method': solver,
                'tags': _dedupe(tags),
                'reasons': reasons[:6],
                'diagnostics': diagnostics,
                'active_space_contract': copy.deepcopy(active_space_contract) if active_space_contract else None,
            })
            continue
        risk_levels = _molecular_risk_levels(diagnostics)
        missing_diagnostics = initial_scan_status == 'succeeded' and (
            diagnostics.get('level') not in ('weak', 'moderate', 'strong')
            or (risk_levels.get('routing_level') is not None
                and risk_levels['routing_level'] not in ('weak', 'moderate', 'strong'))
        )
        if initial_scan_status != 'succeeded' or missing_diagnostics:
            reasons = (['Initial scan has no usable diagnostic classification; correlation strength is unknown.']
                       if missing_diagnostics else _initial_scan_failure_reasons(case_report))
            decisions.append({
                'case_id': case_id,
                'label': row.get('label') or case_id,
                'variables': copy.deepcopy(case_report.get('variables') or {}),
                'initial_scan_status': initial_scan_status,
                'initial_scan_method': initial_scan_method,
                'initial_scan_run_dir': row.get('run_dir'),
                'level': 'initial_scan_blocked',
                'score': None,
                'confidence': 'none',
                'recommended_solver': None,
                'recommended_method': None,
                'tags': ['initial_scan_blocked'],
                'reasons': reasons[:6],
                'diagnostics': copy.deepcopy(diagnostics),
            })
            continue
        diagnostic_level = str(diagnostics['level'])
        risk_levels = _molecular_risk_levels(diagnostics)
        level = str(risk_levels.get('routing_level') or diagnostic_level)
        try:
            score = float(diagnostics.get('score')) if diagnostics.get('score') is not None else None
        except (TypeError, ValueError):
            score = None
        if system_type == 'molecular' or _is_molecular_diagnostics(diagnostics):
            diagnostics_for_routing = copy.deepcopy(diagnostics)
            if active_space_contract:
                diagnostics_for_routing['active_space_contract'] = copy.deepcopy(active_space_contract)
            solver = _recommended_molecular_method(level, diagnostics_for_routing, adaptive_options)
            tags = _molecular_decision_tags(level, score, active_space_contract, adaptive_options)
            reasons = _decision_reasons(diagnostics)
            if diagnostic_level != level:
                reasons.insert(
                    0,
                    'Overall diagnostic level is {0}, but physical evidence is {1}; route through {2} before multireference promotion.'.format(
                        diagnostic_level,
                        risk_levels.get('physics_level') or 'unknown',
                        solver.upper(),
                    ),
                )
            if active_space_contract and level == 'strong':
                reasons.insert(0, 'ActiveSpaceAudit proposed CAS({0}, {1}); user approval is required before CASSCF.'.format(
                    active_space_contract.get('nelecas'),
                    active_space_contract.get('ncas'),
                ))
        else:
            solver = _recommended_solver(level, diagnostics, adaptive_options, _case_report_site_count(case_report, diagnostics))
            tags = _decision_tags(level, score)
            reasons = _decision_reasons(diagnostics)
        decisions.append({
            'case_id': case_id,
            'label': row.get('label') or case_id,
            'variables': copy.deepcopy(case_report.get('variables') or {}),
            'initial_scan_status': row.get('status'),
            'initial_scan_method': initial_scan_method,
            'initial_scan_run_dir': row.get('run_dir'),
            'level': level,
            **risk_levels,
            'score': score,
            'confidence': diagnostics.get('confidence'),
            'recommended_solver': solver,
            'recommended_method': solver,
            'tags': tags,
            'reasons': reasons[:6],
            'diagnostics': diagnostics,
            'active_space_contract': copy.deepcopy(active_space_contract) if active_space_contract else None,
        })
    return decisions


def _initial_scan_issues_from_decisions(decisions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    issues = []
    for item in decisions:
        if not isinstance(item, dict) or 'initial_scan_blocked' not in (item.get('tags') or []):
            continue
        issues.append({
            'case_id': item.get('case_id'),
            'label': item.get('label'),
            'variables': copy.deepcopy(item.get('variables') or {}),
            'initial_scan_status': item.get('initial_scan_status'),
            'initial_scan_method': item.get('initial_scan_method'),
            'initial_scan_run_dir': item.get('initial_scan_run_dir'),
            'reasons': copy.deepcopy(item.get('reasons') or []),
        })
    return issues


def _adaptive_pending_report(
    *,
    adaptive_id: str,
    adaptive_work_dir: Path,
    study_spec_payload: Dict[str, Any],
    system_type: str,
    adaptive_options: Dict[str, Any],
    initial_scan_payload: Dict[str, Any],
    initial_scan_report_payload: Dict[str, Any],
    initial_scan_decisions: List[Dict[str, Any]],
    initial_scan_artifacts: List[Dict[str, Any]],
    refined_payload: Dict[str, Any],
    plan_kind: str,
    refined_report_payload: Optional[Dict[str, Any]] = None,
    recovery_payload: Optional[Dict[str, Any]] = None,
    recovery_decisions: Optional[List[Dict[str, Any]]] = None,
    lifecycle: Optional[Dict[str, Any]] = None,
    collection_paused: bool = False,
) -> Dict[str, Any]:
    """Persist a resumable adaptive checkpoint when only cost approval is pending."""
    artifacts: List[Dict[str, Any]] = []
    artifacts.extend(initial_scan_report_payload.get('artifacts') or [])
    artifacts.extend(initial_scan_artifacts)
    if refined_report_payload:
        artifacts.extend(refined_report_payload.get('artifacts') or [])
    active_plan = recovery_payload['recovery_plan'] if plan_kind == 'recovery' and recovery_payload else refined_payload['refined_plan']
    source_report = refined_report_payload or initial_scan_report_payload
    report = {
        'schema': _ADAPTIVE_STUDY_REPORT_SCHEMA,
        'study_id': adaptive_id,
        'name': study_spec_payload.get('name') or 'adaptive-study',
        'objective': study_spec_payload.get('objective') or 'adaptive scan',
        'system_type': system_type,
        'status': 'pending_review',
        'work_dir': str(adaptive_work_dir),
        'summary': 'Adaptive scan paused before {0} execution because the estimated resource cost requires approval.'.format(plan_kind),
        'lifecycle': copy.deepcopy(lifecycle or {}),
        'adaptive': {
            'mode': 'adaptive_scan',
            'options': adaptive_options,
            'initial_scan_plan': initial_scan_payload['initial_scan_plan'],
            'initial_scan_cost_estimate': initial_scan_payload.get('initial_scan_cost_estimate'),
            'initial_scan_report': initial_scan_report_payload,
            'initial_scan_method': initial_scan_payload.get('initial_scan_method'),
            'initial_scan_decisions': initial_scan_decisions,
            'refined_plan': refined_payload['refined_plan'],
            'decision_log': refined_payload.get('decision_log') or [],
            'study_active_space_policy': refined_payload.get('study_active_space_policy'),
            'refined_report': refined_report_payload,
            'recovery_plan': recovery_payload['recovery_plan'] if recovery_payload else None,
            'recovery_decisions': recovery_decisions or [],
            'recovery_report': None,
            'cost_review_plan_kind': plan_kind,
            'cost_review_estimate': copy.deepcopy(active_plan.get('cost_estimate') or {}),
        },
        'cases': source_report.get('cases') or [],
        'comparison_table': source_report.get('comparison_table') or [],
        'artifacts': artifacts,
    }
    if collection_paused:
        report['summary'] = 'Existing results collected; {0} plan is ready for a separate Run action.'.format(plan_kind)
        report['adaptive'].pop('cost_review_plan_kind', None)
        report['adaptive'].pop('cost_review_estimate', None)
        report['adaptive']['execution_pending_plan_kind'] = plan_kind
    report['adaptive']['workflow'] = build_report_workflow(report)
    _append_gate_artifacts(report, adaptive_work_dir, artifacts)
    artifacts.append(_write_json(
        adaptive_work_dir / 'adaptive-study-report.json',
        report,
        kind='adaptive-study-report',
        description='Adaptive scan checkpoint awaiting the next Run or required approval',
    ))
    report['artifacts'] = artifacts
    return _deepcopy_json(report)


def run_adaptive_study(
    spec: Any,
    *,
    options: Optional[Dict[str, Any]] = None,
    work_dir: str = None,
    locale: str = 'en',
    resume_study_id: Optional[str] = None,
    requested_study_id: Optional[str] = None,
    task_executor: Optional[TaskExecutor] = None,
    resource_profile: Optional[str] = None,
    lifecycle: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    return _run_adaptive_study(
        spec, options=options, work_dir=work_dir, locale=locale,
        resume_study_id=resume_study_id, requested_study_id=requested_study_id,
        task_executor=task_executor, resource_profile=resource_profile, lifecycle=lifecycle,
    )


def collect_adaptive_study(spec: Any, *, study_id: str, options=None, work_dir=None,
                           locale='en', task_executor=None, resource_profile=None, lifecycle=None):
    return _run_adaptive_study(spec, options=options, work_dir=work_dir, locale=locale,
                               resume_study_id=study_id, task_executor=task_executor,
                               resource_profile=resource_profile, lifecycle=lifecycle, collect_only=True)


def _run_adaptive_study(
    spec: Any,
    *,
    options: Optional[Dict[str, Any]] = None,
    work_dir: str = None,
    locale: str = 'en',
    resume_study_id: Optional[str] = None,
    requested_study_id: Optional[str] = None,
    task_executor: Optional[TaskExecutor] = None,
    resource_profile: Optional[str] = None,
    lifecycle: Optional[Dict[str, Any]] = None,
    collect_only: bool = False,
) -> Dict[str, Any]:
    adaptive_options = _normalize_options(options)
    study_spec_payload = _as_study_spec_payload(spec)
    system_type = normalize_system_type(study_spec_payload.get('system_type'))
    if system_type not in ('model_hamiltonian', 'molecular'):
        raise ValueError('Adaptive scan currently supports molecular or model_hamiltonian studies only')
    study_spec_payload['system_type'] = system_type

    if collect_only:
        adaptive_id = _adaptive_study_id_for_resume(resume_study_id)
        if not adaptive_id:
            raise ValueError('An existing study_id is required for collection')
        adaptive_work_dir = resolve_work_dir(work_dir).resolve() / adaptive_id
        if not _adaptive_resume_matches_request(adaptive_work_dir, _adaptive_request_fingerprint(study_spec_payload, adaptive_options)):
            raise ValueError('Collection request does not match the persisted adaptive study')
    else:
        adaptive_id, adaptive_work_dir = _prepare_adaptive_work_dir(
            work_dir,
            resume_study_id,
            requested_study_id,
            study_spec_payload,
            adaptive_options,
            persist=False,
        )
    with study_lock(adaptive_work_dir, wait_seconds=30.0 if collect_only else 0.0):
        if not collect_only:
            _write_adaptive_request(adaptive_work_dir, study_spec_payload, adaptive_options)
        stage_runner = collect_study if collect_only else run_study
        adaptive_lifecycle = begin_study_execution(
            lifecycle,
            entity_id=adaptive_id,
            details={'source': 'adaptive_study_executor'},
        )
        initial_scan_payload = build_adaptive_initial_scan_plan(study_spec_payload, adaptive_options)
        initial_scan_cost = initial_scan_payload.get('initial_scan_cost_estimate')
        if not collect_only and isinstance(initial_scan_cost, dict) and initial_scan_cost.get('approval_required') and not initial_scan_cost.get('approved'):
            raise CostApprovalRequired(initial_scan_cost)
        initial_scan_plan = StudyPlan.from_dict(initial_scan_payload['initial_scan_plan'])
        initial_scan_report = stage_runner(
            initial_scan_plan,
            work_dir=str(adaptive_work_dir / 'initial-scan'),
            locale=locale,
            task_executor=task_executor,
            batch_independent=True,
            resource_profile=resource_profile,
        )
        initial_scan_report_payload = initial_scan_report.to_dict()
        initial_scan_decisions = analyze_initial_scan_report(initial_scan_report_payload, adaptive_options)
        initial_scan_issues = _initial_scan_issues_from_decisions(initial_scan_decisions)
        initial_scan_artifacts = _adaptive_initial_scan_artifacts(
            adaptive_work_dir,
            initial_scan_payload,
            initial_scan_report_payload,
        )

        if initial_scan_issues:
            artifacts: List[Dict[str, Any]] = []
            artifacts.extend(initial_scan_report_payload.get('artifacts') or [])
            artifacts.extend(initial_scan_artifacts)
            artifacts.append(_write_json(
                adaptive_work_dir / 'adaptive-decision-log.json',
                {
                    'schema': _ADAPTIVE_DECISION_LOG_SCHEMA,
                    'initial_scan_decisions': initial_scan_decisions,
                    'initial_scan_issues': initial_scan_issues,
                    'refined_decisions': [],
                    'adaptive_options': adaptive_options,
                },
                kind='adaptive-decision-log',
                description='Adaptive scan stopped because the initial scan did not produce valid diagnostics',
            ))
            summary = (
                'Adaptive Study stopped after initial scan; initial_scan_cases={0}; '
                'initial_scan_issues={1}; refined_cases=0; reason=initial_scan_not_succeeded'
            ).format(
                len(initial_scan_report_payload.get('comparison_table') or []),
                len(initial_scan_issues),
            )
            report = {
                'schema': _ADAPTIVE_STUDY_REPORT_SCHEMA,
                'study_id': adaptive_id,
                'name': study_spec_payload.get('name') or 'adaptive-study',
                'objective': study_spec_payload.get('objective') or 'adaptive scan',
                'system_type': system_type,
                'status': 'completed_with_issues',
                'work_dir': str(adaptive_work_dir),
                'summary': summary,
                'lifecycle': copy.deepcopy(adaptive_lifecycle),
                'adaptive': {
                    'mode': 'adaptive_scan',
                    'options': adaptive_options,
                    'initial_scan_plan': initial_scan_payload['initial_scan_plan'],
                    'initial_scan_cost_estimate': initial_scan_payload.get('initial_scan_cost_estimate'),
                    'initial_scan_report': initial_scan_report_payload,
                    'initial_scan_method': initial_scan_payload.get('initial_scan_method'),
                    'initial_scan_decisions': initial_scan_decisions,
                    'initial_scan_issues': initial_scan_issues,
                    'refined_plan': None,
                    'decision_log': [],
                    'refined_report': None,
                },
                'cases': initial_scan_report_payload.get('cases') or [],
                'comparison_table': initial_scan_report_payload.get('comparison_table') or [],
                'artifacts': artifacts,
            }
            report['adaptive']['workflow'] = build_report_workflow(report)
            _append_gate_artifacts(report, adaptive_work_dir, artifacts)
            artifacts.append(_write_json(
                adaptive_work_dir / 'adaptive-study-report.json',
                report,
                kind='adaptive-study-report',
                description='Adaptive scan report stopped after a blocked initial scan',
            ))
            report['artifacts'] = artifacts
            return _deepcopy_json(report)

        refined_payload = build_refined_plan_from_decisions(study_spec_payload, initial_scan_decisions, adaptive_options)
        refined_plan = StudyPlan.from_dict(refined_payload['refined_plan'])
        refined_cost = refined_plan.cost_estimate
        refinement_unstarted = collect_only and not (adaptive_work_dir / refined_plan.study_id / 'study-state.json').exists()
        if refinement_unstarted or (not collect_only and isinstance(refined_cost, dict) and refined_cost.get('approval_required') and not refined_cost.get('approved')):
            return _adaptive_pending_report(
                adaptive_id=adaptive_id,
                adaptive_work_dir=adaptive_work_dir,
                study_spec_payload=study_spec_payload,
                system_type=system_type,
                adaptive_options=adaptive_options,
                initial_scan_payload=initial_scan_payload,
                initial_scan_report_payload=initial_scan_report_payload,
                initial_scan_decisions=initial_scan_decisions,
                initial_scan_artifacts=initial_scan_artifacts,
                refined_payload=refined_payload,
                plan_kind='refined',
                collection_paused=refinement_unstarted,
                lifecycle=adaptive_lifecycle,
            )
        refined_report = stage_runner(
            refined_plan,
            work_dir=str(adaptive_work_dir),
            locale=locale,
            task_executor=task_executor,
            # Refined cases are still ordinary independent tasks unless their
            # request declares a cross-case continuation dependency.  run_study
            # keeps those dependent cases on the sequential path while batching
            # the rest for scheduler job arrays.
            batch_independent=True,
            resource_profile=resource_profile,
        )
        refined_report_payload = refined_report.to_dict()
        decision_log = refined_payload['decision_log']
        recovery_payload = build_recovery_plan_for_unresolved_refined_cases(
            refined_plan,
            refined_report_payload,
            decision_log,
            adaptive_options,
        )
        recovery_report_payload = None
        recovery_decisions: List[Dict[str, Any]] = []
        if recovery_payload:
            recovery_plan = StudyPlan.from_dict(recovery_payload['recovery_plan'])
            recovery_cost = recovery_plan.cost_estimate
            recovery_unstarted = collect_only and not (adaptive_work_dir / recovery_plan.study_id / 'study-state.json').exists()
            if recovery_unstarted or (not collect_only and isinstance(recovery_cost, dict) and recovery_cost.get('approval_required') and not recovery_cost.get('approved')):
                return _adaptive_pending_report(
                    adaptive_id=adaptive_id,
                    adaptive_work_dir=adaptive_work_dir,
                    study_spec_payload=study_spec_payload,
                    system_type=system_type,
                    adaptive_options=adaptive_options,
                    initial_scan_payload=initial_scan_payload,
                    initial_scan_report_payload=initial_scan_report_payload,
                    initial_scan_decisions=initial_scan_decisions,
                    initial_scan_artifacts=initial_scan_artifacts,
                    refined_payload=refined_payload,
                    refined_report_payload=refined_report_payload,
                    recovery_payload=recovery_payload,
                    recovery_decisions=recovery_payload.get('recovery_decisions') or [],
                    lifecycle=adaptive_lifecycle,
                    plan_kind='recovery',
                    collection_paused=recovery_unstarted,
                )
            recovery_report = stage_runner(
                recovery_plan,
                work_dir=str(adaptive_work_dir),
                locale=locale,
                task_executor=task_executor,
                batch_independent=True,
                resource_profile=resource_profile,
            )
            recovery_report_payload = recovery_report.to_dict()
            recovery_decisions = recovery_payload['recovery_decisions']
            refined_report_payload = _merge_recovery_report(
                refined_report_payload,
                recovery_report_payload,
                decision_log,
                recovery_decisions,
            )

        # Adaptive execution ends with independent case results. Cross-case
        # continuity checks and projected-1RDM propagation are analysis actions and
        # begin only after the user explicitly requests Analyze Results.
        path_diagnostics = deferred_scan_path_diagnostics()
        path_window_restart_payload = None
        path_window_restart_reports: Dict[str, Dict[str, Any]] = {}
        path_window_restart_decisions: List[Dict[str, Any]] = []
        path_window_restart_validation = None
        path_refinement_payload = None
        path_refinement_decisions: List[Dict[str, Any]] = []

        artifacts: List[Dict[str, Any]] = []
        artifacts.extend(initial_scan_report_payload.get('artifacts') or [])
        artifacts.extend(initial_scan_artifacts)
        artifacts.extend(refined_report_payload.get('artifacts') or [])
        if recovery_report_payload:
            artifacts.extend(recovery_report_payload.get('artifacts') or [])
        for restart_report_payload in path_window_restart_reports.values():
            artifacts.extend(restart_report_payload.get('artifacts') or [])
        artifacts.append(_write_json(
            adaptive_work_dir / 'adaptive-refined-plan.json',
            refined_payload['refined_plan'],
            kind='adaptive-refined-plan',
            description='Refined plan with solver choices selected from initial-scan diagnostics',
        ))
        artifacts.append(_write_json(
            adaptive_work_dir / 'adaptive-decision-log.json',
            {
                'schema': _ADAPTIVE_DECISION_LOG_SCHEMA,
                'initial_scan_decisions': initial_scan_decisions,
                'refined_decisions': decision_log,
                'recovery_decisions': recovery_decisions,
                'path_window_restart_decisions': path_window_restart_decisions,
                'path_window_restart_validation': path_window_restart_validation,
                'path_refinement_decisions': path_refinement_decisions,
                'adaptive_options': adaptive_options,
                'study_active_space_policy': refined_payload.get('study_active_space_policy'),
            },
            kind='adaptive-decision-log',
            description='Adaptive scan initial-scan labels and refined method choices',
        ))
        artifacts.append(_write_json(
            adaptive_work_dir / 'scan-path-diagnostics.json',
            path_diagnostics,
            kind='scan-path-diagnostics',
            description='Deferred cross-case continuity analysis state',
        ))
        if isinstance(path_window_restart_payload, dict):
            artifacts.append(_write_json(
                adaptive_work_dir / 'adaptive-path-window-restart-plan.json',
                path_window_restart_payload,
                kind='adaptive-path-window-restart-plan',
                description='Bidirectional same-method continuation candidates using projected one-particle density matrices',
            ))
        if isinstance(path_window_restart_validation, dict):
            artifacts.append(_write_json(
                adaptive_work_dir / 'adaptive-path-window-restart-validation.json',
                path_window_restart_validation,
                kind='adaptive-path-window-restart-validation',
                description='Consistency checks for independently restarted local path candidates',
            ))
        if isinstance(path_refinement_payload, dict):
            artifacts.append(_write_json(
                adaptive_work_dir / 'adaptive-path-refinement-plan.json',
                path_refinement_payload['path_refinement_plan'],
                kind='adaptive-path-refinement-plan',
                description='Case-local CASSCF refinement plan proposed from scan-path continuity diagnostics',
            ))

        succeeded = refined_report_payload.get('status') == 'succeeded'
        strong_count = sum(1 for item in initial_scan_decisions if item.get('level') == 'strong')
        moderate_count = sum(1 for item in initial_scan_decisions if item.get('level') == 'moderate')
        summary = (
            'Adaptive Study completed; initial_scan_cases={0}; initial_scan_method={1}; refined_cases={2}; '
            'strong={3}; moderate={4}; refined_status={5}; recovery_cases={6}; path_analysis=deferred'
        ).format(
            len(initial_scan_report_payload.get('comparison_table') or []),
            str(initial_scan_payload.get('initial_scan_method') or 'unknown').upper(),
            len(refined_report_payload.get('comparison_table') or []),
            strong_count,
            moderate_count,
            refined_report_payload.get('status'),
            len(recovery_decisions),
        )
        report = {
            'schema': _ADAPTIVE_STUDY_REPORT_SCHEMA,
            'study_id': adaptive_id,
            'name': study_spec_payload.get('name') or 'adaptive-study',
            'objective': study_spec_payload.get('objective') or 'adaptive scan',
            'system_type': system_type,
            'status': (
                'succeeded' if succeeded else 'completed_with_issues'
            ),
            'work_dir': str(adaptive_work_dir),
            'summary': summary,
            'lifecycle': copy.deepcopy(adaptive_lifecycle),
            'adaptive': {
                'mode': 'adaptive_scan',
                'options': adaptive_options,
                'initial_scan_plan': initial_scan_payload['initial_scan_plan'],
                'initial_scan_cost_estimate': initial_scan_payload.get('initial_scan_cost_estimate'),
                'initial_scan_report': initial_scan_report_payload,
                'initial_scan_method': initial_scan_payload.get('initial_scan_method'),
                'initial_scan_decisions': initial_scan_decisions,
                'refined_plan': refined_payload['refined_plan'],
                'decision_log': decision_log,
                'study_active_space_policy': refined_payload.get('study_active_space_policy'),
                'refined_report': refined_report_payload,
                'recovery_plan': recovery_payload['recovery_plan'] if recovery_payload else None,
                'recovery_decisions': recovery_decisions,
                'recovery_report': recovery_report_payload,
                'path_diagnostics': path_diagnostics,
                'path_window_restart_plans': (
                    path_window_restart_payload.get('path_window_restart_plans')
                    if isinstance(path_window_restart_payload, dict)
                    else None
                ),
                'path_window_restart_approval': (
                    path_window_restart_payload.get('path_window_restart_approval')
                    if isinstance(path_window_restart_payload, dict)
                    else None
                ),
                'path_window_restart_decisions': path_window_restart_decisions,
                'path_window_restart_reports': path_window_restart_reports,
                'path_window_restart_validation': path_window_restart_validation,
                'path_refinement_plan': path_refinement_payload['path_refinement_plan'] if path_refinement_payload else None,
                'path_refinement_decisions': path_refinement_decisions,
                'path_refinement_report': None,
            },
            'cases': refined_report_payload.get('cases') or [],
            'comparison_table': refined_report_payload.get('comparison_table') or [],
            'artifacts': artifacts,
        }
        report['adaptive']['workflow'] = build_report_workflow(report)
        _append_gate_artifacts(report, adaptive_work_dir, artifacts)
        artifacts.append(_write_json(
            adaptive_work_dir / 'adaptive-study-report.json',
            report,
            kind='adaptive-study-report',
            description='Full adaptive scan report',
        ))
        report['artifacts'] = artifacts
        return _deepcopy_json(report)
