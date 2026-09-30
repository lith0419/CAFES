from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional

from ..registry import default_registry


ACTIVE_SPACE_PROBE_SCHEMA = 'pyscf-agent.active-space-probe.v1'
ACTIVE_SPACE_PROBE_MODULE_ID = 'molecular.active_space_probe'
ACTIVE_SPACE_PROBE_REFINEMENT_MODULE_ID = 'molecular.active_space_probe_refinement'
DEFAULT_ACTIVE_SPACE_OCCUPATION_WINDOW = (0.02, 1.98)
DEFAULT_MOLECULAR_PROBE_OUTPUTS = ('energy', 'homo_lumo', 'dipole')


def active_space_probe_strategy(strategy: Any = 'auto') -> Dict[str, Any]:
    """Resolve one Registry-backed molecular active-space probe strategy."""

    registry = default_registry()
    raw = str(strategy or 'auto').strip().lower().replace('-', '_').replace(' ', '_')
    strategy_id = registry.canonical_capability_id(
        raw,
        namespace='molecular.active_space_probe_strategy',
    )
    if not strategy_id:
        raise ValueError('Unsupported active-space probe strategy: {0}.'.format(strategy))
    capability = registry.capability(
        strategy_id,
        namespace='molecular.active_space_probe_strategy',
    )
    if capability is None or not capability.backend_allowed:
        raise ValueError('Active-space probe strategy is unavailable: {0}.'.format(strategy_id))
    method = str(capability.metadata.get('execution_method') or '').strip().lower()
    if method not in ('hf', 'mp2', 'fci'):
        raise ValueError(
            "Active-space probe strategy '{0}' has no supported execution method.".format(
                strategy_id
            )
        )
    return {
        'requested_strategy': strategy_id,
        'probe_method': method,
        'refinement_method': str(
            capability.metadata.get('refinement_method') or ''
        ).strip().lower() or None,
        'label': capability.label,
    }


def _merge_probe_outputs(request: Dict[str, Any]) -> None:
    outputs = list(request.get('outputs') or [])
    for item in DEFAULT_MOLECULAR_PROBE_OUTPUTS:
        if item not in outputs:
            outputs.append(item)
    if outputs:
        request['outputs'] = outputs

    analysis = copy.deepcopy(request.get('analysis')) if isinstance(request.get('analysis'), Mapping) else {}
    analysis_outputs = list(analysis.get('outputs') or [])
    for item in DEFAULT_MOLECULAR_PROBE_OUTPUTS:
        if item not in analysis_outputs:
            analysis_outputs.append(item)
    analysis['outputs'] = analysis_outputs
    request['analysis'] = analysis


def active_space_probe_workflow(
    value: Any,
    contract: Mapping[str, Any],
) -> Dict[str, Any]:
    """Compile probe orchestration without activating the target CAS solver."""

    workflow = copy.deepcopy(value) if isinstance(value, Mapping) else {}
    raw_modules = workflow.get('modules')
    modules = [
        str(module_id).strip()
        for module_id in (raw_modules if isinstance(raw_modules, (list, tuple)) else [])
        if str(module_id).strip()
        and not str(module_id).strip().startswith('solver.')
    ]
    modules.append(ACTIVE_SPACE_PROBE_MODULE_ID)

    raw_config = workflow.get('module_config')
    module_config = {
        str(module_id): copy.deepcopy(configuration)
        for module_id, configuration in (
            raw_config.items() if isinstance(raw_config, Mapping) else []
        )
        if not str(module_id).strip().startswith('solver.')
        and isinstance(configuration, Mapping)
    }
    module_config[ACTIVE_SPACE_PROBE_MODULE_ID] = {
        'requested_strategy': str(contract['requested_strategy']),
        'probe_method': str(contract['probe_method']),
        'occupation_window': list(contract['occupation_window']),
    }
    if contract.get('refinement_method'):
        module_config[ACTIVE_SPACE_PROBE_MODULE_ID]['refinement_method'] = str(
            contract['refinement_method']
        )
        modules.append(ACTIVE_SPACE_PROBE_REFINEMENT_MODULE_ID)
        module_config[ACTIVE_SPACE_PROBE_REFINEMENT_MODULE_ID] = {
            'initial_method': str(contract['probe_method']),
            'refinement_method': str(contract['refinement_method']),
        }
    return {
        'modules': list(dict.fromkeys(modules)),
        'module_config': module_config,
        'dependency_policy': str(
            workflow.get('dependency_policy') or 'auto'
        ).strip().lower(),
    }


def build_molecular_active_space_probe_request(
    request: Mapping[str, Any],
    *,
    strategy: Any = 'auto',
    probe_method: Optional[str] = None,
    target_method: Optional[str] = None,
    target_solver: Optional[str] = None,
    target_solver_options: Optional[Mapping[str, Any]] = None,
    selection_method: Optional[str] = None,
) -> Dict[str, Any]:
    """Build the single numerical probe contract shared by Assistant and Planner."""

    if not isinstance(request, Mapping):
        raise TypeError('Molecular active-space probe request must be a mapping.')
    probe_request = copy.deepcopy(dict(request))
    task_type = str(probe_request.get('task_type') or 'molecular').strip().lower()
    if task_type != 'molecular':
        raise ValueError('Active-space probes currently require a molecular task.')

    strategy_contract = active_space_probe_strategy(strategy)
    method = str(probe_method or strategy_contract['probe_method']).strip().lower()
    if method not in ('hf', 'mp2', 'fci'):
        raise ValueError('Unsupported molecular active-space probe method: {0}.'.format(method))
    refinement_method = (
        strategy_contract.get('refinement_method')
        if method == strategy_contract['probe_method']
        else None
    )

    original_active_space = (
        copy.deepcopy(probe_request.get('active_space'))
        if isinstance(probe_request.get('active_space'), Mapping)
        else {}
    )
    requested_target_method = str(
        target_method
        or original_active_space.get('target_method')
        or probe_request.get('method')
        or 'casscf'
    ).strip().lower()
    if requested_target_method not in ('casci', 'casscf'):
        requested_target_method = 'casscf'

    solver_payload = probe_request.get('solver')
    solver_name = ''
    solver_options: Dict[str, Any] = {}
    if isinstance(solver_payload, Mapping):
        solver_name = str(solver_payload.get('name') or '').strip().lower()
        if isinstance(solver_payload.get('options'), Mapping):
            solver_options = copy.deepcopy(dict(solver_payload['options']))
    elif isinstance(solver_payload, str):
        solver_name = solver_payload.strip().lower()
    requested_target_solver = str(
        target_solver
        or original_active_space.get('target_solver')
        or solver_name
        or 'fci'
    ).strip().lower().replace('-', '_')
    requested_target_solver_options = copy.deepcopy(
        dict(target_solver_options)
        if isinstance(target_solver_options, Mapping)
        else original_active_space.get('target_solver_options')
        if isinstance(original_active_space.get('target_solver_options'), Mapping)
        else solver_options
    )

    requested_selection = str(
        selection_method or original_active_space.get('selection_method') or ''
    ).strip().lower().replace('-', '_')
    use_avas = requested_selection == 'avas' and bool(original_active_space.get('avas_targets'))
    effective_selection = 'avas' if use_avas else 'occupation_window'
    occupation_window = list(
        original_active_space.get('occupation_window')
        or DEFAULT_ACTIVE_SPACE_OCCUPATION_WINDOW
    )
    active_space = copy.deepcopy(original_active_space)
    active_space.update({
        'enabled': True,
        'selection_method': effective_selection,
        'ncas': None,
        'nelecas': None,
        'orbital_indices': [],
        'occupation_window': occupation_window,
        'target_method': requested_target_method,
        'target_solver': requested_target_solver,
        'target_solver_options': requested_target_solver_options,
        'approved': False,
    })
    if not use_avas:
        active_space['avas_targets'] = []
        active_space.pop('initial_mo_coeff', None)

    contract = {
        'schema': ACTIVE_SPACE_PROBE_SCHEMA,
        'requested_strategy': strategy_contract['requested_strategy'],
        'probe_method': method,
        'refinement_method': refinement_method,
        'target_method': requested_target_method,
        'target_solver': requested_target_solver,
        'target_solver_options': requested_target_solver_options,
        'selection_method': effective_selection,
        'occupation_window': occupation_window,
    }
    probe_request.update({
        'task_type': 'molecular',
        'method': method,
        'xc': None,
        'restricted': False,
        'active_space': active_space,
        'post_cas': {
            'sc_nevpt2': {'enabled': False, 'root': 0, 'density_fit': True},
        },
    })
    probe_request.pop('solver', None)
    density_fitting = probe_request.get('density_fitting')
    if isinstance(density_fitting, dict) and density_fitting.get('apply_to') == 'scf_and_casscf':
        density_fitting['apply_to'] = 'scf'
    _merge_probe_outputs(probe_request)
    probe_request['workflow'] = active_space_probe_workflow(
        probe_request.get('workflow'),
        contract,
    )
    return {
        'request': probe_request,
        'contract': contract,
    }


__all__ = [
    'ACTIVE_SPACE_PROBE_MODULE_ID',
    'ACTIVE_SPACE_PROBE_REFINEMENT_MODULE_ID',
    'ACTIVE_SPACE_PROBE_SCHEMA',
    'DEFAULT_ACTIVE_SPACE_OCCUPATION_WINDOW',
    'DEFAULT_MOLECULAR_PROBE_OUTPUTS',
    'active_space_probe_strategy',
    'active_space_probe_workflow',
    'build_molecular_active_space_probe_request',
]
