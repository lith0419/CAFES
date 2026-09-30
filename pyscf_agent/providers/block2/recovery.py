from __future__ import annotations

import copy
from typing import Any, Dict, Optional


BLOCK2_RECOVERY_SCHEMA = 'pyscf-agent.block2-recovery-recommendation.v1'

_MEMORY_MARKERS = (
    'out of memory',
    'out_of_memory',
    'memoryerror',
    'cannot allocate memory',
    'bad_alloc',
    'oom-kill',
    'oom kill',
    'oom_kill',
    'exceeded memory limit',
)


def classify_block2_exception(error: Any) -> Optional[Dict[str, Any]]:
    message = str(error or '')
    lowered = message.lower()
    if not any(marker in lowered for marker in _MEMORY_MARKERS):
        return None
    return {
        'schema': BLOCK2_RECOVERY_SCHEMA,
        'status': 'review_required',
        'failure_class': 'memory_exhausted',
        'automatic_retry_safe': False,
        'recommended_action': 'increase_memory_profile',
        'alternatives': [
            'disable_optional_entanglement_or_2rdm_outputs',
            'reduce_independent_task_concurrency',
        ],
        'summary': (
            'block2 exhausted the available memory. Select a larger execution '
            'memory profile; optional 2-RDM or entanglement outputs may also be '
            'disabled when they are not scientifically required.'
        ),
    }


def block2_scheduler_recovery(
    scheduler_state: Any,
    message: Any = '',
) -> Dict[str, Any]:
    """Translate a terminal scheduler failure into a reviewable DMRG action."""

    state = str(scheduler_state or '').strip().upper().split('+', 1)[0]
    memory = classify_block2_exception('{0} {1}'.format(state, message))
    if memory is not None:
        return memory
    if state in ('TIMEOUT', 'DEADLINE'):
        return {
            'schema': BLOCK2_RECOVERY_SCHEMA,
            'status': 'review_required',
            'failure_class': 'walltime_exhausted',
            'automatic_retry_safe': False,
            'recommended_action': 'increase_walltime_profile',
            'alternatives': [
                'continue_from_compatible_mps',
                'reduce_requested_sweeps',
            ],
            'summary': (
                'The scheduler wall-time limit expired. Select a longer wall-time '
                'profile and continue from a compatible MPS checkpoint when one is '
                'available.'
            ),
        }
    if state in ('BOOT_FAIL', 'NODE_FAIL', 'PREEMPTED', 'REVOKED'):
        return {
            'schema': BLOCK2_RECOVERY_SCHEMA,
            'status': 'automatic_retry_available',
            'failure_class': 'scheduler_interruption',
            'automatic_retry_safe': True,
            'recommended_action': 'retry_same_profile',
            'alternatives': ['select_another_resource_profile'],
            'summary': (
                'The scheduler interrupted the calculation independently of the '
                'DMRG convergence result. Retry the same task and reuse a compatible '
                'MPS checkpoint if one was written.'
            ),
        }
    if state == 'CANCELLED':
        return {
            'schema': BLOCK2_RECOVERY_SCHEMA,
            'status': 'review_required',
            'failure_class': 'scheduler_cancelled',
            'automatic_retry_safe': False,
            'recommended_action': 'review_cancellation',
            'alternatives': ['retry_same_profile'],
            'summary': (
                'The scheduler cancelled the DMRG task. Review the cancellation '
                'reason before submitting it again.'
            ),
        }
    return {
        'schema': BLOCK2_RECOVERY_SCHEMA,
        'status': 'review_required',
        'failure_class': 'scheduler_failure',
        'automatic_retry_safe': False,
        'recommended_action': 'inspect_scheduler_logs',
        'alternatives': ['retry_same_profile', 'select_another_resource_profile'],
        'summary': (
            'The scheduler ended the DMRG task before it produced a TaskReport. '
            'Inspect the scheduler output and resource profile before retrying.'
        ),
    }


def _automatic_recovery_policy():
    from ...registry import default_registry
    return default_registry().capability(
        'block2_dmrg', namespace='molecular.active_space_solver',
    ).metadata['automatic_recovery']


def block2_convergence_recovery(result: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(result, dict) or result.get('converged') is not False:
        return None
    convergence = result.get('convergence') if isinstance(result.get('convergence'), dict) else {}
    adaptive = result.get('adaptive_schedule') if isinstance(result.get('adaptive_schedule'), dict) else {}
    energy_ok = bool(convergence.get('energy_converged'))
    discarded_ok = bool(convergence.get('discarded_weight_converged'))
    maximum_bond_dimension = adaptive.get('maximum_bond_dimension')
    limit = _automatic_recovery_policy()['max_bond_dimension']
    at_supported_limit = (
        maximum_bond_dimension is not None
        and int(maximum_bond_dimension) >= limit
    )
    if not discarded_ok and at_supported_limit:
        action = 'review_resource_profile'
        summary = (
            'The discarded-weight target was not reached at the supported automatic '
            'bond-dimension limit M={0}. Review the resource profile and scientific '
            'tolerance before another retry.'
        ).format(limit)
    elif not discarded_ok:
        action = 'increase_bond_dimension'
        summary = (
            'The discarded-weight target was not reached. Increase the maximum '
            'bond dimension and continue from the compatible MPS checkpoint.'
        )
    elif not energy_ok:
        action = 'extend_sweeps'
        summary = (
            'The sweep energy is still changing. Extend the sweep schedule and '
            'continue from the compatible MPS checkpoint before changing method.'
        )
    else:
        action = 'review_dmrg_convergence'
        summary = 'The block2 convergence contract was not met; inspect the sweep history.'
    return {
        'schema': BLOCK2_RECOVERY_SCHEMA,
        'status': 'review_required' if not discarded_ok and at_supported_limit else 'automatic_retry_available',
        'failure_class': (
            'discarded_weight_not_converged'
            if not discarded_ok
            else 'sweep_energy_not_converged'
        ),
        'automatic_retry_safe': not (not discarded_ok and at_supported_limit),
        'recommended_action': action,
        'alternatives': ['retry_from_compatible_mps', 'review_resource_profile'],
        'current': {
            'final_energy_change': convergence.get('final_energy_change'),
            'energy_tolerance': convergence.get('energy_tolerance'),
            'final_discarded_weight': convergence.get('final_discarded_weight'),
            'discarded_weight_tolerance': convergence.get('discarded_weight_tolerance'),
            'final_bond_dimension': adaptive.get('final_bond_dimension'),
            'maximum_bond_dimension': maximum_bond_dimension,
            'budget_exhausted': adaptive.get('budget_exhausted'),
        },
        'summary': summary,
    }


def escalated_block2_options(options: Any, recommendation: Dict[str, Any]) -> Dict[str, Any]:
    payload = copy.deepcopy(options or {}) if isinstance(options, dict) else {}
    action = str(recommendation.get('recommended_action') or '')
    from .config import normalize_block2_options
    configuration = normalize_block2_options(payload)
    policy = _automatic_recovery_policy()
    current_max = configuration.max_bond_dimension
    if action == 'increase_bond_dimension':
        proposed = max(current_max + 1, int(current_max * configuration.bond_dimension_growth_factor))
        payload['max_bond_dimension'] = max(current_max, min(policy['max_bond_dimension'], proposed))
    if action in ('increase_bond_dimension', 'extend_sweeps'):
        payload['adaptive_schedule'] = True
        proposed_sweeps = max(
            policy['minimum_adaptive_sweeps'],
            configuration.adaptive_sweeps + policy['adaptive_sweep_increment'],
        )
        payload['adaptive_sweeps'] = max(
            configuration.adaptive_sweeps, min(policy['max_adaptive_sweeps'], proposed_sweeps),
        )
        payload['max_adaptive_stages'] = max(
            configuration.max_adaptive_stages,
            min(policy['max_adaptive_stages'], configuration.max_adaptive_stages + 1),
        )
        payload['restart_required'] = bool(payload.get('restart_manifest'))
    return payload


__all__ = [
    'BLOCK2_RECOVERY_SCHEMA',
    'block2_convergence_recovery',
    'block2_scheduler_recovery',
    'classify_block2_exception',
    'escalated_block2_options',
]
