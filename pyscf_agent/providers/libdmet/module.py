from __future__ import annotations

import copy
from typing import Any, Dict

from ...backend.state import append_log
from ...embedding.reference_density import normalize_reference_density_guess
from ..block2.availability import block2_availability
from .availability import libdmet_availability
from .dmet import normalize_dmet_options, validate_dmet_model_request


def configure_reference_density(state: Dict[str, Any], invocation: Any) -> Dict[str, Any]:
    """Materialize one registered PM/AF/FM/CDW reference-density strategy."""

    state = copy.deepcopy(state)
    task_spec = state.get('task_spec') if isinstance(state.get('task_spec'), dict) else {}
    solver = task_spec.get('solver') if isinstance(task_spec.get('solver'), dict) else {}
    if str(solver.get('name') or '').strip().lower().replace('-', '_') != 'dmet':
        return state
    strategy = normalize_reference_density_guess(
        str(getattr(invocation, 'module_id', '')).rsplit('.', 1)[-1]
    )
    options = solver.get('options') if isinstance(solver.get('options'), dict) else {}
    options = copy.deepcopy(options)
    options['reference_density_guess'] = strategy
    options['initial_correlation_potential'] = 'zero'
    solver['options'] = options
    task_spec['solver'] = solver
    state['task_spec'] = task_spec
    append_log(state, 'info', 'provider.libdmet.reference_density_configured', {
        'strategy': strategy,
        'initial_correlation_potential': 'zero',
    })
    return state


def prepare_dmet_provider(state: Dict[str, Any], invocation: Any) -> Dict[str, Any]:
    """Validate libDMET and materialize the composite DMET solver contract."""

    if getattr(invocation, 'configuration', None):
        raise ValueError(
            'embedding.libdmet.dmet does not accept module configuration; '
            'configure DMET through solver.options.'
        )
    state = copy.deepcopy(state)
    task_spec = state.get('task_spec') if isinstance(state.get('task_spec'), dict) else {}
    solver = task_spec.get('solver') if isinstance(task_spec.get('solver'), dict) else {}
    solver_name = str(solver.get('name') or '').strip().lower().replace('-', '_')
    if solver_name != 'dmet':
        return state
    availability = libdmet_availability()
    if not availability.get('available'):
        raise RuntimeError(
            'DMET requires the optional libDMET provider: {0}'.format(
                availability.get('reason') or 'provider unavailable'
            )
        )
    model = task_spec.get('model_hamiltonian') if isinstance(task_spec.get('model_hamiltonian'), dict) else {}
    model_spec = model.get('spec') if isinstance(model.get('spec'), dict) else {}
    options = normalize_dmet_options(solver.get('options') if isinstance(solver.get('options'), dict) else {})
    if options['impurity_solver'] == 'block2_dmrg':
        block2 = block2_availability()
        if not block2.get('available'):
            raise RuntimeError(
                'DMET with block2_dmrg requires the optional block2 provider: {0}'.format(
                    block2.get('reason') or 'provider unavailable'
                )
            )
    errors, resolved = validate_dmet_model_request(model_spec, options)
    if errors or resolved is None:
        raise ValueError('; '.join(errors))
    solver['name'] = 'dmet'
    solver['options'] = options
    task_spec['solver'] = solver
    state['task_spec'] = task_spec
    state['solver_provider'] = {
        'provider': 'libdmet',
        'method': 'dmet',
        'configuration': resolved,
    }
    append_log(state, 'info', 'provider.libdmet.dmet_prepared', {
        'impurity_solver': resolved['impurity_solver'],
        'impurity_solver_options': resolved['impurity_solver_options'],
        'lattice_shape': resolved['lattice_shape'],
        'impurity_shape': resolved['impurity_shape'],
        'execution_mode': resolved['execution_mode'],
        'translation_backend': resolved['translation_backend'],
        'fragment_count': resolved['fragment_count'],
        'fragment_site_counts': resolved['fragment_site_counts'],
        'interacting_bath': resolved['interacting_bath'],
        'initial_correlation_potential': resolved['initial_correlation_potential'],
        'reference_density_guess': resolved['reference_density_guess'],
        'reference': resolved['reference'],
        'libdmet_sz': resolved['libdmet_sz'],
    })
    return state


__all__ = ['configure_reference_density', 'prepare_dmet_provider']
