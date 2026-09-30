from __future__ import annotations

from pyscf_agent.serialization import json_default

import copy
import json
from typing import Any, Dict, List, Optional

from pyscf_agent.registry import default_registry
from pyscf_agent.contracts import normalize_orbital_processing
from pyscf_agent.input_validation import integer, reject_unknown_fields
from pyscf_agent.providers.block2.config import normalized_block2_provider_options
from pyscf_agent.backend.active_space_probe import (
    DEFAULT_MOLECULAR_PROBE_OUTPUTS,
    active_space_probe_strategy,
    build_molecular_active_space_probe_request,
)

from ..planner import build_study_plan
from ..normalization import normalize_system_type
from ..gates.presentation import build_initial_plan_workflow
from ..schema import StudySpec
from .policies import adaptive_routing_policy


DEFAULT_MODEL_INITIAL_SCAN_OBSERVABLES = ('energy', 'strong_correlation_diagnostics')
DEFAULT_MOLECULAR_INITIAL_SCAN_OBSERVABLES = DEFAULT_MOLECULAR_PROBE_OUTPUTS
_ROUTING_POLICY = adaptive_routing_policy()
_RESOURCE_LIMITS = _ROUTING_POLICY['resource_limits']
DEFAULT_METHOD_POLICY = copy.deepcopy(_ROUTING_POLICY['model_method_policy'])
DEFAULT_MOLECULAR_METHOD_POLICY = copy.deepcopy(_ROUTING_POLICY['molecular_method_policy'])
DEFAULT_MAX_FCI_SITES = int(_RESOURCE_LIMITS['max_fci_sites'])
DEFAULT_MAX_MOLECULAR_FCI_ORBITALS = int(_RESOURCE_LIMITS['max_molecular_fci_orbitals'])
DEFAULT_MAX_MOLECULAR_FCI_ELECTRONS = int(_RESOURCE_LIMITS['max_molecular_fci_electrons'])
DEFAULT_MAX_CAS_ORBITALS = int(_RESOURCE_LIMITS['max_cas_orbitals'])
RECOVERY_SOLVER_ORDER = tuple(_ROUTING_POLICY['recovery_solver_order'])
MOLECULAR_SINGLE_REFERENCE_METHODS = tuple(_ROUTING_POLICY['molecular_single_reference_methods'])

_INITIAL_SCAN_CAPABILITIES = default_registry().capabilities(
    namespace='study.adaptive_initial_scan',
    backend_allowed=True,
)
INITIAL_SCAN_STRATEGY_METHODS = {
    item.id: str(item.metadata.get('execution_method') or '').strip().lower()
    for item in _INITIAL_SCAN_CAPABILITIES
    if str(item.metadata.get('execution_method') or '').strip()
}
DEFAULT_INITIAL_SCAN_STRATEGY = next(
    (item.id for item in _INITIAL_SCAN_CAPABILITIES if item.metadata.get('default')),
    next(iter(INITIAL_SCAN_STRATEGY_METHODS), 'auto'),
)


def _dedupe(items: List[Any]) -> List[Any]:
    result = []
    seen = set()
    for item in items:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True, default=json_default, allow_nan=False)
        if key in seen:
            continue
        seen.add(key)
        result.append(item)
    return result


def _as_study_spec_payload(spec: Any) -> Dict[str, Any]:
    study_spec = spec if isinstance(spec, StudySpec) else StudySpec.from_dict(spec)
    return study_spec.to_dict()


def _normalize_strategy_name(value: Any) -> str:
    raw = str(value or '').strip().lower()
    canonical = default_registry().canonical_capability_id(
        raw, namespace='study.adaptive_initial_scan'
    )
    if canonical:
        return canonical
    normalized = raw.replace('-', '_').replace(' ', '_')
    return (
        default_registry().canonical_capability_id(
            normalized, namespace='study.adaptive_initial_scan'
        )
        or normalized
    )


def _normalize_active_space_orbital_processing(value: Any) -> Dict[str, Any]:
    if value is None or value == {}:
        return {}
    return normalize_orbital_processing(value, default_scope='active_space')


def _normalize_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    if options is not None and not isinstance(options, dict):
        raise ValueError('adaptive options must be an object')
    payload = copy.deepcopy(options or {})
    limits = {
        'max_fci_sites': DEFAULT_MAX_FCI_SITES,
        'max_molecular_fci_orbitals': DEFAULT_MAX_MOLECULAR_FCI_ORBITALS,
        'max_molecular_fci_electrons': DEFAULT_MAX_MOLECULAR_FCI_ELECTRONS,
        'max_cas_orbitals': DEFAULT_MAX_CAS_ORBITALS,
    }
    reject_unknown_fields(payload, set(limits) | {
        'method_policy', 'molecular_method_policy', 'model_solver_options',
        'initial_scan_strategy', 'active_space_solver', 'block2_dmrg_options',
        'active_space_orbital_processing',
    }, 'adaptive options')
    for field, defaults in (
        ('method_policy', DEFAULT_METHOD_POLICY),
        ('molecular_method_policy', DEFAULT_MOLECULAR_METHOD_POLICY),
    ):
        raw = payload.get(field, {})
        if not isinstance(raw, dict):
            raise ValueError('{0} must be an object'.format(field))
        reject_unknown_fields(raw, defaults, field)
        policy = copy.deepcopy(defaults)
        for key, value in raw.items():
            if not isinstance(value, str) or not value.strip():
                raise ValueError('{0}.{1} must name a method'.format(field, key))
            policy[key] = value.strip().lower()
        payload[field] = policy
    model_options = payload.get('model_solver_options', {})
    if not isinstance(model_options, dict) or any(not isinstance(v, dict) for v in model_options.values()):
        raise ValueError('model_solver_options must map solver names to option objects')
    payload['model_solver_options'] = model_options
    raw_initial_scan_strategy = payload.get('initial_scan_strategy')
    initial_scan_strategy = _normalize_strategy_name(raw_initial_scan_strategy)
    if not initial_scan_strategy:
        initial_scan_strategy = DEFAULT_INITIAL_SCAN_STRATEGY
    elif initial_scan_strategy not in INITIAL_SCAN_STRATEGY_METHODS:
        raise ValueError(
            'Unsupported initial_scan_strategy: {0}. Choose one of: {1}.'.format(
                raw_initial_scan_strategy,
                ', '.join(sorted(INITIAL_SCAN_STRATEGY_METHODS)),
            )
        )
    payload['initial_scan_strategy'] = initial_scan_strategy
    for field, default in limits.items():
        value = integer(payload.get(field, default), field)
        if value <= 0:
            raise ValueError('{0} must be positive'.format(field))
        payload[field] = value
    active_space_solver = str(payload.get('active_space_solver') or 'auto').strip().lower().replace('-', '_')
    if active_space_solver in ('block2', 'dmrg'):
        active_space_solver = 'block2_dmrg'
    if active_space_solver not in ('auto', 'fci', 'block2_dmrg'):
        raise ValueError('Unsupported active_space_solver: {0}'.format(active_space_solver))
    payload['active_space_solver'] = active_space_solver
    dmrg_options = payload.get('block2_dmrg_options')
    payload['block2_dmrg_options'] = (
        normalized_block2_provider_options(dmrg_options, include_defaults=False) if dmrg_options is not None else {}
    )
    payload['active_space_orbital_processing'] = _normalize_active_space_orbital_processing(
        payload.get('active_space_orbital_processing')
    )
    return payload


def normalize_adaptive_options(options: Optional[Dict[str, Any]]) -> Dict[str, Any]:
    return _normalize_options(options)


def _set_model_solver(study_spec: Dict[str, Any], solver: str) -> None:
    base_task = study_spec.setdefault('base_task', {})
    if isinstance(base_task, dict):
        base_task['solver'] = solver
    base_model_spec = study_spec.get('base_model_spec')
    if isinstance(base_model_spec, dict):
        base_model_spec['solver'] = solver
    case_design = study_spec.get('case_design')
    if isinstance(case_design, dict) and case_design:
        template = case_design.setdefault('template', {})
        if isinstance(template, dict):
            request_updates = template.setdefault('request_updates', {})
            if isinstance(request_updates, dict):
                request_updates['solver'] = solver
        cases = case_design.get('cases')
        if isinstance(cases, list):
            for item in cases:
                if not isinstance(item, dict):
                    continue
                request_updates = item.setdefault('request_updates', {})
                if isinstance(request_updates, dict):
                    request_updates['solver'] = solver
        overrides = case_design.get('overrides')
        if isinstance(overrides, list):
            for item in overrides:
                if not isinstance(item, dict):
                    continue
                request_updates = item.setdefault('request_updates', {})
                if isinstance(request_updates, dict):
                    request_updates['solver'] = solver


def _molecular_initial_scan_request_updates(
    method: str,
    *,
    strategy: str = DEFAULT_INITIAL_SCAN_STRATEGY,
    target_solver: Optional[str] = None,
    target_solver_options: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    probe = build_molecular_active_space_probe_request(
        {'task_type': 'molecular'},
        strategy=strategy,
        probe_method=method,
        target_solver=target_solver,
        target_solver_options=target_solver_options,
    )['request']
    probe.pop('task_type', None)
    return probe


def _merge_dict_update(target: Dict[str, Any], update: Dict[str, Any]) -> None:
    for key, value in update.items():
        if isinstance(value, dict) and isinstance(target.get(key), dict):
            _merge_dict_update(target[key], value)
        else:
            target[key] = copy.deepcopy(value)


def _set_molecular_initial_scan_method(
    study_spec: Dict[str, Any],
    method: str,
    *,
    strategy: str = DEFAULT_INITIAL_SCAN_STRATEGY,
    target_solver: Optional[str] = None,
    target_solver_options: Optional[Dict[str, Any]] = None,
) -> None:
    updates = _molecular_initial_scan_request_updates(
        method,
        strategy=strategy,
        target_solver=target_solver,
        target_solver_options=target_solver_options,
    )
    base_task = study_spec.setdefault('base_task', {})
    if isinstance(base_task, dict):
        _merge_dict_update(base_task, updates)
        base_task.pop('solver', None)
    sweep = study_spec.get('sweep')
    if isinstance(sweep, dict):
        for key in ('method', 'solver', 'xc', 'restricted', 'active_space', 'post_cas', 'analysis'):
            sweep.pop(key, None)
    case_design = study_spec.get('case_design')
    if isinstance(case_design, dict) and case_design:
        template = case_design.setdefault('template', {})
        if isinstance(template, dict):
            request_updates = template.setdefault('request_updates', {})
            if isinstance(request_updates, dict):
                _merge_dict_update(request_updates, updates)
                request_updates.pop('solver', None)
        cases = case_design.get('cases')
        if isinstance(cases, list):
            for item in cases:
                if not isinstance(item, dict):
                    continue
                request_updates = item.setdefault('request_updates', {})
                if isinstance(request_updates, dict):
                    _merge_dict_update(request_updates, updates)
                    request_updates.pop('solver', None)
        overrides = case_design.get('overrides')
        if isinstance(overrides, list):
            for item in overrides:
                if not isinstance(item, dict):
                    continue
                request_updates = item.setdefault('request_updates', {})
                if isinstance(request_updates, dict):
                    _merge_dict_update(request_updates, updates)
                    request_updates.pop('solver', None)


def _ensure_initial_scan_observables(study_spec: Dict[str, Any], system_type: str) -> None:
    observables = list(study_spec.get('observables') or [])
    if system_type == 'model_hamiltonian':
        defaults = DEFAULT_MODEL_INITIAL_SCAN_OBSERVABLES
    else:
        defaults = DEFAULT_MOLECULAR_INITIAL_SCAN_OBSERVABLES
    study_spec['observables'] = _dedupe(observables + list(defaults))


def build_initial_scan_study_spec(
    spec: Any,
    options: Optional[Dict[str, Any]] = None,
    *,
    solver: Optional[str] = None,
) -> Dict[str, Any]:
    payload = _as_study_spec_payload(spec)
    adaptive_options = _normalize_options(options)
    system_type = normalize_system_type(payload.get('system_type'))
    if system_type not in ('model_hamiltonian', 'molecular'):
        raise ValueError('Adaptive scan currently supports molecular or model_hamiltonian studies only')
    payload['system_type'] = system_type

    stage_solver = str(
        solver
        or initial_scan_method_for_options(
            adaptive_options,
            system_type=system_type,
        )
    ).strip().lower()
    if not stage_solver:
        raise ValueError('Initial scan strategy does not define a solver.')
    payload['name'] = '{0}-initial-scan'.format(payload.get('name') or 'adaptive-study')
    payload['objective'] = 'Initial diagnostic scan ({0}) for {1}'.format(
        stage_solver.upper(),
        payload.get('objective') or 'adaptive study',
    )
    if system_type == 'model_hamiltonian':
        _set_model_solver(payload, stage_solver)
    else:
        active_space_solver = adaptive_options.get('active_space_solver')
        explicit_target_solver = (
            active_space_solver
            if active_space_solver in ('fci', 'block2_dmrg')
            else None
        )
        _set_molecular_initial_scan_method(
            payload,
            stage_solver,
            strategy=adaptive_options['initial_scan_strategy'],
            target_solver=explicit_target_solver,
            target_solver_options=(
                adaptive_options.get('block2_dmrg_options')
                if explicit_target_solver == 'block2_dmrg'
                else None
            ),
        )
    _ensure_initial_scan_observables(payload, system_type)
    comparison = payload.setdefault('comparison', {})
    if isinstance(comparison, dict):
        notes = comparison.get('notes')
        if system_type == 'model_hamiltonian':
            initial_scan_note = 'Internal adaptive model scan runs {0} diagnostics on every planned case.'.format(stage_solver.upper())
        else:
            initial_scan_note = 'Adaptive molecular scan runs {0} initial diagnostics on every planned case, then chooses refined methods from MolecularCorrelationRisk and ActiveSpaceAudit candidates.'.format(stage_solver.upper())
        if isinstance(notes, list):
            if initial_scan_note not in notes:
                notes.append(initial_scan_note)
        elif isinstance(notes, str) and notes.strip():
            comparison['notes'] = [notes, initial_scan_note]
        else:
            comparison['notes'] = [initial_scan_note]
    return payload


def initial_scan_method_for_options(
    options: Optional[Dict[str, Any]] = None,
    *,
    system_type: Optional[str] = None,
) -> str:
    adaptive_options = _normalize_options(options)
    strategy = adaptive_options['initial_scan_strategy']
    if system_type is not None and normalize_system_type(system_type) == 'model_hamiltonian':
        # Preserve the internal model helper's MP2 probe; the shared HF-first
        # strategy is molecular-only. Public model studies remain static scans.
        if strategy == 'auto':
            return DEFAULT_METHOD_POLICY['weak']
        return INITIAL_SCAN_STRATEGY_METHODS[strategy]
    return str(active_space_probe_strategy(strategy)['probe_method'])


def build_adaptive_initial_scan_plan(spec: Any, options: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    adaptive_options = _normalize_options(options)
    spec_payload = _as_study_spec_payload(spec)
    system_type = normalize_system_type(spec_payload.get('system_type'))
    initial_scan_method = initial_scan_method_for_options(
        adaptive_options,
        system_type=system_type,
    )
    initial_scan_spec = build_initial_scan_study_spec(
        spec_payload,
        adaptive_options,
        solver=initial_scan_method,
    )
    plan = build_study_plan(initial_scan_spec)
    plan.study_id = 'initial-scan'
    initial_scan_plan = plan.to_dict()
    return {
        'initial_scan_spec': initial_scan_spec,
        'initial_scan_plan': initial_scan_plan,
        'initial_scan_strategy': adaptive_options['initial_scan_strategy'],
        'initial_scan_method': initial_scan_method,
        'initial_scan_cost_estimate': copy.deepcopy(initial_scan_plan.get('cost_estimate')),
        'adaptive_options': adaptive_options,
        'workflow': build_initial_plan_workflow(
            initial_scan_plan,
            initial_scan_plan.get('cost_estimate'),
        ),
    }
