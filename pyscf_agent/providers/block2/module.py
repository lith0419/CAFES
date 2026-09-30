from __future__ import annotations

import copy
from typing import Any, Dict

from ...contracts import task_spec_from_dict
from ...workflow_modules.runtime import ModuleInvocation
from .availability import block2_availability
from .config import (
    block2_options_for_orbital_processing,
    block2_options_for_outputs,
    normalized_block2_provider_options,
)


def prepare_block2_provider(state: Dict[str, Any], invocation: ModuleInvocation) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    availability = block2_availability()
    task_spec = task_spec_from_dict(state.get('task_spec') or {})
    provider = dict(availability)
    requested_options = block2_options_for_outputs(
        task_spec.solver.options,
        task_spec.analysis.outputs,
    )
    if task_spec.task_type == 'molecular' and task_spec.method.name in ('casci', 'casscf'):
        requested_options = block2_options_for_orbital_processing(
            requested_options, task_spec.orbital_processing,
        )
    requested_options.update(invocation.configuration or {})
    # Configuration describes the request even on a client without block2.
    # The existing availability gate decides whether it can execute here.
    provider['configuration'] = normalized_block2_provider_options(requested_options)
    provider['configuration_source'] = {
        'solver_options': copy.deepcopy(task_spec.solver.options or {}),
        'module_configuration': copy.deepcopy(invocation.configuration or {}),
    }
    state['solver_provider'] = provider
    return state


def _configure_block2_feature(
    state: Dict[str, Any],
    invocation: ModuleInvocation,
    defaults: Dict[str, Any],
) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    provider = dict(state.get('solver_provider') or {})
    if provider.get('provider') != 'block2_dmrg':
        raise ValueError('block2 feature modules require the solver.block2.dmrg provider module.')
    options = dict(provider.get('configuration') or {})
    options.update(defaults)
    options.update(invocation.configuration or {})
    provider['configuration'] = normalized_block2_provider_options(options)
    state['solver_provider'] = provider
    state.setdefault('block2_feature_configuration', {})[invocation.module_id] = {
        'runtime_id': invocation.runtime_id,
        'configuration': copy.deepcopy(invocation.configuration),
        'effective_provider_options': copy.deepcopy(provider['configuration']),
    }
    return state


def configure_entanglement_diagnostics(state: Dict[str, Any], invocation: ModuleInvocation) -> Dict[str, Any]:
    return _configure_block2_feature(state, invocation, {
        'compute_entanglement': True,
        'compute_mutual_information': False,
        'compute_bipartite_entanglement': True,
    })


def configure_bond_dimension_planning(state: Dict[str, Any], invocation: ModuleInvocation) -> Dict[str, Any]:
    return _configure_block2_feature(state, invocation, {})


def configure_excited_states(state: Dict[str, Any], invocation: ModuleInvocation) -> Dict[str, Any]:
    state = copy.deepcopy(state)
    provider = dict(state.get('solver_provider') or {})
    if provider.get('provider') != 'block2_dmrg':
        raise ValueError('block2 feature modules require the solver.block2.dmrg provider module.')

    options = dict(provider.get('configuration') or {})
    requested = dict(invocation.configuration or {})
    requested.setdefault('excited_state_mode', 'state_averaged')

    # The provider configuration already contains TaskSpec solver options.  A
    # workflow-module override may replace nroots, but a stale weight vector
    # must not survive that replacement.  Let the common block2 normalizer
    # regenerate equal weights when no explicit replacement was requested.
    if 'nroots' in requested and 'state_average_weights' not in requested:
        try:
            roots_changed = int(requested['nroots']) != int(options.get('nroots') or 1)
        except (TypeError, ValueError):
            roots_changed = True
        if roots_changed:
            options.pop('state_average_weights', None)
    options.update(requested)
    provider['configuration'] = normalized_block2_provider_options(options)
    state['solver_provider'] = provider
    state.setdefault('block2_feature_configuration', {})[invocation.module_id] = {
        'runtime_id': invocation.runtime_id,
        'configuration': copy.deepcopy(invocation.configuration),
        'effective_provider_options': copy.deepcopy(provider['configuration']),
    }
    return state


def configure_mps_continuation(state: Dict[str, Any], invocation: ModuleInvocation) -> Dict[str, Any]:
    return _configure_block2_feature(state, invocation, {'restart_required': True})


def configure_symmetry_analysis(state: Dict[str, Any], invocation: ModuleInvocation) -> Dict[str, Any]:
    return _configure_block2_feature(state, invocation, {'compute_symmetry_analysis': True})


__all__ = [
    'configure_bond_dimension_planning',
    'configure_entanglement_diagnostics',
    'configure_excited_states',
    'configure_mps_continuation',
    'configure_symmetry_analysis',
    'prepare_block2_provider',
]
