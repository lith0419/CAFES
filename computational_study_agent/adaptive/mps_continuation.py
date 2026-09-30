from __future__ import annotations

from pyscf_agent.serialization import json_default

import copy
import hashlib
import json
import math
from typing import Any, Dict, Iterable, List, Optional

from ..costing import ensure_plan_cost_estimate
from ..schema import StudyCase, StudyPlan


MPS_CONTINUATION_APPROVAL_SCHEMA = 'pyscf-agent.mps-continuation-approval.v1'


def _solver_name(request: Dict[str, Any]) -> str:
    raw = request.get('solver') if isinstance(request, dict) else None
    if isinstance(raw, dict):
        raw = raw.get('name')
    value = str(raw or '').strip().lower().replace('-', '_')
    return 'block2_dmrg' if value in ('block2', 'dmrg') else value


def _solver_options(request: Dict[str, Any]) -> Dict[str, Any]:
    raw = request.get('solver') if isinstance(request, dict) else None
    return copy.deepcopy(raw.get('options') or {}) if isinstance(raw, dict) else {}


def _method_name(request: Dict[str, Any]) -> str:
    raw = request.get('method') if isinstance(request, dict) else None
    if isinstance(raw, dict):
        raw = raw.get('name')
    return str(raw or '').strip().lower().replace('-', '_')


def _dmrg_result(case_record: Dict[str, Any]) -> Dict[str, Any]:
    task_report = case_record.get('task_report') if isinstance(case_record, dict) else None
    structured = task_report.get('structured_results') if isinstance(task_report, dict) else None
    structured = structured if isinstance(structured, dict) else {}
    direct = structured.get('dmrg_result')
    if isinstance(direct, dict):
        return direct
    cas_result = structured.get('cas_result') if isinstance(structured.get('cas_result'), dict) else {}
    result = cas_result.get('dmrg_result') if isinstance(cas_result.get('dmrg_result'), dict) else {}
    return result


def _recovery_recommendation(case_record: Dict[str, Any]) -> Dict[str, Any]:
    result = _dmrg_result(case_record)
    recommendation = result.get('recovery_recommendation') if isinstance(result, dict) else None
    if isinstance(recommendation, dict):
        return recommendation
    task_report = case_record.get('task_report') if isinstance(case_record, dict) else None
    errors = task_report.get('errors') if isinstance(task_report, dict) else None
    for error in reversed(errors or []):
        details = error.get('details') if isinstance(error, dict) else None
        recommendation = details.get('recovery_recommendation') if isinstance(details, dict) else None
        if isinstance(recommendation, dict):
            return recommendation
    return {}


def _memory_failure(case_record: Dict[str, Any]) -> bool:
    return _recovery_recommendation(case_record).get('failure_class') == 'memory_exhausted'


def _state_tracking(report: Dict[str, Any]) -> Dict[str, Any]:
    direct = report.get('dmrg_state_tracking') if isinstance(report, dict) else None
    if isinstance(direct, dict):
        return direct
    adaptive = report.get('adaptive') if isinstance(report, dict) else None
    value = adaptive.get('dmrg_state_tracking') if isinstance(adaptive, dict) else None
    return value if isinstance(value, dict) else {}


def _checkpoint_manifest(case_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    result = _dmrg_result(case_record)
    manifest = result.get('checkpoint_manifest') if isinstance(result, dict) else None
    if isinstance(manifest, dict) and manifest.get('schema') == 'pyscf-agent.block2-mps-manifest.v1':
        return copy.deepcopy(manifest)
    return None


def _joint_checkpoint(manifest: Dict[str, Any]) -> bool:
    context = manifest.get('orbital_context') if isinstance(manifest, dict) else None
    return bool(
        isinstance(context, dict)
        and context.get('external_restart_supported') is True
        and context.get('kind') == 'casscf_optimized_orbitals_and_mps'
    )


def _manifest_artifact(case_record: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    task_report = case_record.get('task_report') if isinstance(case_record, dict) else None
    artifacts = task_report.get('artifacts') if isinstance(task_report, dict) else None
    for artifact in artifacts or ():
        if isinstance(artifact, dict) and artifact.get('kind') == 'block2_mps_manifest':
            return copy.deepcopy(artifact)
    return None


def _status(case_record: Dict[str, Any]) -> str:
    task_report = case_record.get('task_report') if isinstance(case_record, dict) else None
    return str(task_report.get('execution_status') if isinstance(task_report, dict) else '').strip().lower()


def _numeric_variables(case_record: Dict[str, Any]) -> Dict[str, float]:
    values = case_record.get('variables') if isinstance(case_record.get('variables'), dict) else {}
    result: Dict[str, float] = {}
    for key, value in values.items():
        try:
            result[str(key)] = float(value)
        except (TypeError, ValueError):
            continue
    return result


def _case_distance(left: Dict[str, Any], right: Dict[str, Any]) -> float:
    left_values = _numeric_variables(left)
    right_values = _numeric_variables(right)
    keys = sorted(set(left_values).intersection(right_values))
    if not keys:
        return math.inf
    terms = []
    for key in keys:
        scale = max(abs(left_values[key]), abs(right_values[key]), 1.0)
        terms.append(((left_values[key] - right_values[key]) / scale) ** 2)
    return math.sqrt(sum(terms) / len(terms))


def _relative_direction(target: Dict[str, Any], anchor: Dict[str, Any]) -> str:
    target_values = _numeric_variables(target)
    anchor_values = _numeric_variables(anchor)
    for key in sorted(set(target_values).intersection(anchor_values)):
        delta = anchor_values[key] - target_values[key]
        if abs(delta) > 1e-14:
            return 'higher_parameter_side' if delta > 0 else 'lower_parameter_side'
    return 'same_or_multidimensional_position'


def _electron_partition(total: Any, spin: Any = 0) -> Optional[List[int]]:
    if isinstance(total, (list, tuple)) and len(total) == 2:
        try:
            return [int(total[0]), int(total[1])]
        except (TypeError, ValueError):
            return None
    try:
        total_value = int(total)
        spin_value = int(spin or 0)
    except (TypeError, ValueError):
        return None
    if total_value < 0 or abs(spin_value) > total_value or (total_value + spin_value) % 2:
        return None
    return [(total_value + spin_value) // 2, (total_value - spin_value) // 2]


def _target_sector(case: StudyCase, system_type: str) -> Dict[str, Any]:
    request = case.request if isinstance(case.request, dict) else {}
    options = _solver_options(request)
    orbital_processing = request.get('orbital_processing') if isinstance(request.get('orbital_processing'), dict) else {}
    orbital_ordering = str(
        options.get('orbital_ordering')
        or orbital_processing.get('orbital_ordering')
        or 'canonical'
    ).strip().lower()
    orbital_order = list(
        options.get('orbital_order')
        or orbital_processing.get('orbital_order')
        or []
    )
    symmetry = str(options.get('symmetry') or 'auto').strip().lower()
    if symmetry == 'auto':
        symmetry = 'su2'
    try:
        nroots = max(1, int(options.get('nroots') or 1))
    except (TypeError, ValueError):
        nroots = 1
    if str(system_type).strip().lower() == 'model_hamiltonian':
        model = request.get('model_hamiltonian') if isinstance(request.get('model_hamiltonian'), dict) else {}
        spec = model.get('spec') if isinstance(model.get('spec'), dict) else case.model_spec
        spec = spec if isinstance(spec, dict) else {}
        sites = spec.get('sites') if isinstance(spec.get('sites'), list) else []
        nelec = _electron_partition(spec.get('nelec'))
        spin = (nelec[0] - nelec[1]) if nelec else None
        return {
            'n_orbitals': len(sites) or None,
            'n_electrons': nelec,
            'spin': spin,
            'symmetry': symmetry,
            'nroots': nroots,
            'orbital_ordering': orbital_ordering,
            'orbital_order': orbital_order,
        }
    active_space = request.get('active_space') if isinstance(request.get('active_space'), dict) else {}
    system = request.get('system') if isinstance(request.get('system'), dict) else request
    spin = system.get('spin', request.get('spin', 0)) if isinstance(system, dict) else request.get('spin', 0)
    nelec = _electron_partition(active_space.get('nelecas'), spin)
    return {
        'n_orbitals': active_space.get('ncas'),
        'n_electrons': nelec,
        'spin': (nelec[0] - nelec[1]) if nelec else None,
        'symmetry': symmetry,
        'nroots': nroots,
        'orbital_ordering': orbital_ordering,
        'orbital_order': orbital_order,
    }


def _compatibility(manifest: Dict[str, Any], sector: Dict[str, Any]) -> Dict[str, Any]:
    source_ordering = manifest.get('orbital_ordering') if isinstance(manifest.get('orbital_ordering'), dict) else {}
    checks = {
        'n_orbitals': manifest.get('n_orbitals') == sector.get('n_orbitals'),
        'n_electrons': list(manifest.get('n_electrons') or []) == list(sector.get('n_electrons') or []),
        'spin': manifest.get('spin') == sector.get('spin'),
        'symmetry': str(manifest.get('symmetry') or '') == str(sector.get('symmetry') or ''),
        'nroots': int(manifest.get('nroots') or 1) == int(sector.get('nroots') or 1),
        'checkpoint_files': bool(manifest.get('files')),
        'orbital_ordering_request': (
            str(source_ordering.get('requested_method') or 'canonical') == str(sector.get('orbital_ordering') or 'canonical')
            and list(source_ordering.get('requested_order') or []) == list(sector.get('orbital_order') or [])
        ),
    }
    return {
        'compatible': all(checks.values()),
        'checks': checks,
        'source_sector': {
            key: copy.deepcopy(manifest.get(key))
            for key in ('n_orbitals', 'n_electrons', 'spin', 'symmetry', 'nroots')
        },
        'target_sector': copy.deepcopy(sector),
        'source_orbital_ordering': copy.deepcopy(source_ordering),
    }


def _approval_token(items: Iterable[Dict[str, Any]]) -> str:
    payload = [
        {
            'case_id': item.get('case_id'),
            'source_case_id': item.get('source_case_id'),
            'compatibility': item.get('compatibility'),
        }
        for item in items
    ]
    encoded = json.dumps(payload, sort_keys=True, separators=(',', ':'), default=json_default, allow_nan=False).encode('utf-8')
    return hashlib.sha256(encoded).hexdigest()


def build_mps_continuation_plan(
    plan: Any,
    report: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    """Build same-method retries seeded from the nearest compatible succeeded MPS."""

    try:
        study_plan = plan if isinstance(plan, StudyPlan) else StudyPlan.from_dict(plan)
    except (TypeError, ValueError):
        return None
    report_cases = [item for item in report.get('cases') or [] if isinstance(item, dict)]
    if not report_cases:
        return None
    plan_cases = {case.case_id: case for case in study_plan.cases}
    execution = report.get('execution') if isinstance(report.get('execution'), dict) else {}
    source_executor = execution.get('executor') if isinstance(execution.get('executor'), dict) else {}
    report_by_id = {str(item.get('case_id')): item for item in report_cases if item.get('case_id') is not None}
    state_tracking = _state_tracking(report)
    ambiguous_anchors = {
        str(case_id)
        for case_id in (state_tracking.get('ambiguous_case_ids') or [])
    }
    anchors = [
        item for item in report_cases
        if _status(item) == 'succeeded'
        and _solver_name(item.get('request') or {}) == 'block2_dmrg'
        and _checkpoint_manifest(item)
        and str(item.get('case_id') or '') not in ambiguous_anchors
    ]
    targets = [
        item for item in report_cases
        if _status(item) in ('unconverged', 'failed')
        and _solver_name(item.get('request') or {}) == 'block2_dmrg'
        and not _memory_failure(item)
    ]
    if not anchors or not targets:
        return None

    retry_cases: List[StudyCase] = []
    decisions: List[Dict[str, Any]] = []
    for target in targets:
        case_id = str(target.get('case_id') or '')
        source_case = plan_cases.get(case_id)
        if not source_case:
            source_case = StudyCase.from_dict(target)
        sector = _target_sector(source_case, study_plan.system_type)
        target_method = _method_name(source_case.request)
        candidates = []
        for anchor in anchors:
            manifest = _checkpoint_manifest(anchor)
            source_method = _method_name(anchor.get('request') or {})
            if (
                str(study_plan.system_type).strip().lower() == 'molecular'
                and source_method != target_method
            ):
                continue
            if target_method == 'casscf' and (
                source_method != 'casscf'
                or not _joint_checkpoint(manifest or {})
            ):
                continue
            compatibility = _compatibility(manifest or {}, sector)
            if compatibility['compatible']:
                candidates.append((_case_distance(target, anchor), anchor, manifest, compatibility))
        if not candidates:
            continue
        candidates.sort(key=lambda item: item[0])
        distance, anchor, manifest, compatibility = candidates[0]
        candidate_summary = [
            {
                'source_case_id': str(candidate[1].get('case_id') or ''),
                'source_variables': copy.deepcopy(candidate[1].get('variables') or {}),
                'normalized_distance': None if math.isinf(candidate[0]) else candidate[0],
                'relative_direction': _relative_direction(target, candidate[1]),
                'state_tracking_status': 'trusted_anchor',
            }
            for candidate in candidates[:4]
        ]
        request = copy.deepcopy(source_case.request)
        solver_options = _solver_options(request)
        solver_options.update({
            'restart_manifest': copy.deepcopy(manifest),
            'restart_required': True,
            'restart_provenance': {
                'source_case_id': anchor.get('case_id'),
                'source_artifact': _manifest_artifact(anchor),
                'selection': 'nearest_compatible_state_trusted_succeeded_case',
                'normalized_distance': None if math.isinf(distance) else distance,
                'source_executor': copy.deepcopy(source_executor),
            },
        })
        request['solver'] = {'name': 'block2_dmrg', 'options': solver_options}
        retry_cases.append(StudyCase(
            case_id=source_case.case_id,
            label=source_case.label,
            request=request,
            variables=copy.deepcopy(source_case.variables),
            operations=copy.deepcopy(source_case.operations),
            model_spec=copy.deepcopy(source_case.model_spec),
        ))
        decisions.append({
            'case_id': source_case.case_id,
            'label': source_case.label,
            'variables': copy.deepcopy(source_case.variables),
            'source_case_id': str(anchor.get('case_id') or ''),
            'source_label': anchor.get('label'),
            'source_variables': copy.deepcopy(anchor.get('variables') or {}),
            'source_artifact': _manifest_artifact(anchor),
            'source_scratch_directory': manifest.get('scratch_directory'),
            'normalized_distance': None if math.isinf(distance) else distance,
            'compatibility': compatibility,
            'transport': 'executor_local_checkpoint',
            'checkpoint_components': list(manifest.get('checkpoint_components') or ['mps']),
            'orbital_context': copy.deepcopy(manifest.get('orbital_context') or {}),
            'source_executor': copy.deepcopy(source_executor),
            'compatible_anchor_candidates': candidate_summary,
            'state_tracking': {
                'schema': state_tracking.get('schema'),
                'status': 'trusted_anchor',
                'source_case_ambiguous': False,
                'excluded_ambiguous_case_ids': sorted(ambiguous_anchors),
            },
            'reason': (
                'Retry the same block2 DMRG-CASSCF method from the nearest compatible optimized-orbital plus MPS checkpoint.'
                if target_method == 'casscf'
                else 'Retry the same block2 method from the nearest compatible succeeded MPS before changing method or bond dimension.'
            ),
        })
    if not retry_cases:
        return None

    continuation_plan = StudyPlan(
        study_id='{0}-mps-continuation'.format(study_plan.study_id),
        name='{0}-mps-continuation'.format(study_plan.name),
        objective='Same-method block2 recovery from compatible neighboring MPS checkpoints',
        system_type=study_plan.system_type,
        cases=retry_cases,
        observables=copy.deepcopy(study_plan.observables),
        comparison=copy.deepcopy(study_plan.comparison),
        capability_snapshot=copy.deepcopy(study_plan.capability_snapshot),
        resource_policy={**copy.deepcopy(study_plan.resource_policy), 'approved': True},
    )
    ensure_plan_cost_estimate(continuation_plan)
    token = _approval_token(decisions)
    approval = {
        'schema': MPS_CONTINUATION_APPROVAL_SCHEMA,
        'kind': 'mps_continuation_approval',
        'status': 'approval_required',
        'approval_token': token,
        'case_ids': [item['case_id'] for item in decisions],
        'summary': 'State-tracked compatible block2 checkpoints can retry {0} unresolved case(s) without changing the target method.'.format(len(decisions)),
        'items': copy.deepcopy(decisions),
    }
    return {
        'mps_continuation_plan': continuation_plan.to_dict(),
        'mps_continuation_decisions': decisions,
        'mps_continuation_approval': approval,
    }


def merge_mps_continuation_report(
    report: Dict[str, Any],
    continuation_report: Dict[str, Any],
    decisions: List[Dict[str, Any]],
) -> Dict[str, Any]:
    merged = copy.deepcopy(report)
    decision_by_case = {str(item.get('case_id')): item for item in decisions}

    def merge_rows(old_rows: Any, new_rows: Any) -> List[Dict[str, Any]]:
        updates = {str(item.get('case_id')): item for item in new_rows or [] if isinstance(item, dict)}
        result = []
        seen = set()
        for item in old_rows or []:
            if not isinstance(item, dict):
                continue
            case_id = str(item.get('case_id') or '')
            replacement = copy.deepcopy(updates.get(case_id) or item)
            if case_id in decision_by_case:
                replacement['mps_restart_source_case'] = decision_by_case[case_id].get('source_case_id')
            result.append(replacement)
            seen.add(case_id)
        for case_id, item in updates.items():
            if case_id not in seen:
                result.append(copy.deepcopy(item))
        return result

    merged['cases'] = merge_rows(merged.get('cases'), continuation_report.get('cases'))
    merged['comparison_table'] = merge_rows(
        merged.get('comparison_table'),
        continuation_report.get('comparison_table'),
    )
    merged['status'] = (
        'succeeded'
        if merged.get('comparison_table')
        and all(str(item.get('status') or '').lower() == 'succeeded' for item in merged['comparison_table'])
        else 'completed_with_issues'
    )
    merged['artifacts'] = list(merged.get('artifacts') or []) + list(continuation_report.get('artifacts') or [])
    adaptive = copy.deepcopy(merged.get('adaptive') or {})
    adaptive['mps_continuation_decisions'] = copy.deepcopy(decisions)
    adaptive['mps_continuation_report'] = copy.deepcopy(continuation_report)
    merged['adaptive'] = adaptive
    return merged


__all__ = [
    'MPS_CONTINUATION_APPROVAL_SCHEMA',
    'build_mps_continuation_plan',
    'merge_mps_continuation_report',
]
