from __future__ import annotations

import copy
from typing import Any, Dict, Mapping

from pyscf_agent.registry import default_registry


def _policy_metadata(policy_id: str) -> Dict[str, Any]:
    capability = default_registry().capability(policy_id, namespace='study.policy')
    if capability is None or not isinstance(capability.metadata, Mapping):
        raise RuntimeError("Registry study policy '{0}' is unavailable".format(policy_id))
    return copy.deepcopy(dict(capability.metadata))


def adaptive_routing_policy() -> Dict[str, Any]:
    policy = _policy_metadata('adaptive_routing')
    required = {
        'model_method_policy',
        'molecular_method_policy',
        'resource_limits',
        'recovery_solver_order',
        'molecular_single_reference_methods',
    }
    missing = sorted(required.difference(policy))
    if missing:
        raise RuntimeError('Registry adaptive routing policy is incomplete: {0}'.format(', '.join(missing)))
    return policy


def scan_path_continuity_policy() -> Dict[str, Any]:
    policy = _policy_metadata('scan_path_continuity')
    required = {
        'minimum_sample_count',
        'slope_mismatch_ratio',
        'energy_residual_hartree',
        'strong_slope_mismatch_ratio',
        'molecular_method_levels',
    }
    missing = sorted(required.difference(policy))
    if missing:
        raise RuntimeError('Registry scan-path policy is incomplete: {0}'.format(', '.join(missing)))
    if int(policy['minimum_sample_count']) < 4:
        raise RuntimeError('Registry scan-path policy requires at least four samples')
    return policy


__all__ = ['adaptive_routing_policy', 'scan_path_continuity_policy']
