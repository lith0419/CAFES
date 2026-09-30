from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

from pyscf_agent.registry import (
    PlatformRegistry,
    default_registry as default_platform_registry,
)


@dataclass
class StudyCapabilityView:
    platform_registry: PlatformRegistry

    @property
    def molecular_methods(self) -> List[str]:
        return [item.id for item in self.platform_registry.capabilities(
            namespace='molecular.method', planner_allowed=True
        )]

    @property
    def molecular_basis_sets(self) -> List[str]:
        return [item.id for item in self.platform_registry.option_values(
            'options.molecular.basis', planner_allowed=True
        )]

    @property
    def molecular_xc_functionals(self) -> List[str]:
        return [item.id for item in self.platform_registry.option_values(
            'options.molecular.xc', planner_allowed=True
        )]

    @property
    def molecular_observables(self) -> List[str]:
        return [item.id for item in self.platform_registry.observables(
            namespace='molecular.observable', planner_allowed=True
        )]

    @property
    def model_hamiltonian_models(self) -> List[str]:
        return [item.id for item in self.platform_registry.capabilities(
            namespace='model_hamiltonian.model', planner_allowed=True
        )]

    @property
    def model_hamiltonian_solvers(self) -> List[str]:
        return [item.id for item in self.platform_registry.capabilities(
            namespace='model_hamiltonian.solver', planner_allowed=True
        )]

    @property
    def model_hamiltonian_parameters(self) -> List[str]:
        return [item.id for item in self.platform_registry.parameters(
            namespace='model_hamiltonian.parameter', planner_allowed=True
        )]

    @property
    def model_hamiltonian_operations(self) -> List[str]:
        return [item.id for item in self.platform_registry.capabilities(
            namespace='model_hamiltonian.operation', planner_allowed=True
        )]

    @property
    def model_hamiltonian_observables(self) -> List[str]:
        outputs = [item.id for item in self.platform_registry.observables(
            namespace='model_hamiltonian.observable', planner_allowed=True
        )]
        outputs.extend(
            item.id for item in self.platform_registry.observables(
                namespace='postprocessing.metric', system_type='model_hamiltonian',
                backend_allowed=True,
            ) if self.model_observable_output(item.id)
        )
        return list(dict.fromkeys(outputs))

    def as_dict(self) -> Dict[str, Any]:
        return {
            'molecular_methods': list(self.molecular_methods),
            'molecular_basis_sets': list(self.molecular_basis_sets),
            'molecular_xc_functionals': list(self.molecular_xc_functionals),
            'molecular_observables': list(self.molecular_observables),
            'model_hamiltonian_models': list(self.model_hamiltonian_models),
            'model_hamiltonian_solvers': list(self.model_hamiltonian_solvers),
            'model_hamiltonian_parameters': list(self.model_hamiltonian_parameters),
            'model_hamiltonian_operations': list(self.model_hamiltonian_operations),
            'model_hamiltonian_observables': list(self.model_hamiltonian_observables),
            'platform_catalog': self.platform_registry.as_dict(),
        }

    def supports_molecular_method(self, method: str) -> bool:
        return self.platform_registry.capability_is_allowed(
            method, namespace='molecular.method', action='planner'
        )

    def canonical_molecular_method(self, method: Any) -> Optional[str]:
        return self.platform_registry.canonical_capability_id(method, namespace='molecular.method')

    def molecular_method_capability(self, method: Any) -> Any:
        return self.platform_registry.capability(method, namespace='molecular.method')

    def molecular_workflow_option_capability(self, option: Any) -> Any:
        return self.platform_registry.capability(option, namespace='molecular.workflow_option')

    def supports_model_solver(self, solver: str) -> bool:
        return self.platform_registry.capability_is_allowed(
            solver, namespace='model_hamiltonian.solver', action='planner'
        )

    def canonical_model_solver(self, solver: Any) -> Optional[str]:
        return self.platform_registry.canonical_capability_id(
            solver, namespace='model_hamiltonian.solver'
        )

    def supports_model_parameter(self, parameter: str) -> bool:
        capability = self.platform_registry.parameter(
            parameter, namespace='model_hamiltonian.parameter'
        )
        return bool(capability and capability.planner_allowed)

    def supports_model_observable(self, observable: str) -> bool:
        return self.model_observable_output(observable) is not None

    def model_observable_output(self, observable: str) -> Optional[str]:
        """Resolve a Study metric request to its registered executable output."""
        capability = self.platform_registry.observable(
            observable, namespace='model_hamiltonian.observable'
        )
        if capability and capability.planner_allowed:
            return capability.id
        metric = self.platform_registry.observable(observable, namespace='postprocessing.metric')
        if not metric or not metric.backend_allowed:
            return None
        if 'model_hamiltonian' not in metric.metadata.get('system_types', []):
            return None
        output = metric.metadata.get('study_request_output')
        if not output:
            return None
        capability = self.platform_registry.observable(output, namespace='model_hamiltonian.observable')
        if capability and capability.planner_allowed and capability.backend_allowed and capability.requestable_output:
            return capability.id
        return None

    def supports_molecular_observable(self, observable: str) -> bool:
        capability = self.platform_registry.observable(
            observable, namespace='molecular.observable'
        )
        return bool(capability and capability.planner_allowed)


def default_study_capabilities() -> StudyCapabilityView:
    return StudyCapabilityView(default_platform_registry())


def study_capability_snapshot(registry: StudyCapabilityView = None) -> Dict[str, Any]:
    return copy.deepcopy((registry or default_study_capabilities()).as_dict())


def _request_spin(request: Dict[str, Any]) -> int:
    system = request.get('system') if isinstance(request.get('system'), dict) else {}
    try:
        return int(request.get('spin', system.get('spin', 0)) or 0)
    except (TypeError, ValueError):
        return 0


def _request_restricted_value(request: Dict[str, Any]) -> Optional[bool]:
    value = request.get('restricted')
    if value is None:
        return None
    if isinstance(value, str):
        return value.strip().lower() in ('1', 'true', 'yes', 'on')
    return bool(value)


def molecular_reference_mode(request: Dict[str, Any], capability: Any) -> str:
    metadata = getattr(capability, 'metadata', {}) if capability is not None else {}
    policy = metadata.get('reference_policy') if isinstance(metadata, dict) else {}
    policy = policy if isinstance(policy, dict) else {}
    spin = _request_spin(request)
    restricted = _request_restricted_value(request)
    if restricted is None:
        return str(policy.get('open_shell_default' if spin else 'closed_shell_default') or 'unknown')
    if not restricted:
        return str(policy.get('unrestricted') or 'unrestricted')
    if spin:
        return str(policy.get('restricted_open_shell') or 'unsupported')
    return str(policy.get('closed_shell_default') or 'restricted')


def molecular_request_contract_issues(
    request: Dict[str, Any],
    registry: StudyCapabilityView = None,
) -> List[Tuple[str, str]]:
    """Validate registry-declared molecular method and post-CAS contracts."""
    capability_registry = registry or default_study_capabilities()
    method = capability_registry.canonical_molecular_method(request.get('method') or 'hf')
    if not method:
        return []
    method_capability = capability_registry.molecular_method_capability(method)
    metadata = getattr(method_capability, 'metadata', {}) if method_capability is not None else {}
    metadata = metadata if isinstance(metadata, dict) else {}
    issues: List[Tuple[str, str]] = []
    reference_mode = molecular_reference_mode(request, method_capability)
    if reference_mode == 'unsupported':
        issues.append((
            'restricted_open_shell',
            '{0} does not support a restricted open-shell reference; use unrestricted execution or a CAS method with the spin-adapted ROHF default.'.format(method.upper()),
        ))

    method_contract = metadata.get('request_contract') if isinstance(metadata.get('request_contract'), dict) else {}
    active_space = request.get('active_space') if isinstance(request.get('active_space'), dict) else {}
    for field in method_contract.get('requires_active_space', []):
        value = active_space.get(field)
        missing = (
            value is None
            or value is False
            or (field == 'ncas' and (not isinstance(value, int) or value <= 0))
            or (isinstance(value, (list, dict, str)) and not value)
        )
        if missing:
            code = 'active_space_not_approved' if field == 'approved' else 'missing_active_space_{0}'.format(field)
            issues.append((
                code,
                '{0} requires active_space.{1} before the plan can run.'.format(method.upper(), field),
            ))

    post_cas = request.get('post_cas') if isinstance(request.get('post_cas'), dict) else {}
    sc_nevpt2 = post_cas.get('sc_nevpt2') if isinstance(post_cas.get('sc_nevpt2'), dict) else {}
    if not sc_nevpt2.get('enabled'):
        return issues
    option_capability = capability_registry.molecular_workflow_option_capability('sc_nevpt2')
    option_metadata = getattr(option_capability, 'metadata', {}) if option_capability is not None else {}
    option_contract = option_metadata.get('request_contract') if isinstance(option_metadata, dict) else {}
    option_contract = option_contract if isinstance(option_contract, dict) else {}
    allowed_methods = set(option_contract.get('requires_methods') or [])
    if allowed_methods and method not in allowed_methods:
        issues.append((
            'sc_nevpt2_requires_cas',
            'SC-NEVPT2 requires method={0}.'.format(' or '.join(sorted(allowed_methods))),
        ))
    if option_contract.get('requires_active_space_approval') and not active_space.get('approved'):
        issues.append((
            'sc_nevpt2_requires_approved_active_space',
            'SC-NEVPT2 requires an approved active_space contract.',
        ))
    allowed_references = set(option_contract.get('allowed_references') or [])
    if allowed_references and reference_mode not in allowed_references:
        issues.append((
            'sc_nevpt2_reference_not_supported',
            'SC-NEVPT2 requires a spin-adapted restricted/ROHF-style CAS reference.',
        ))
    root = sc_nevpt2.get('root', 0)
    try:
        normalized_root = int(root)
    except (TypeError, ValueError):
        normalized_root = None
    allowed_roots = set(option_contract.get('allowed_roots') or [])
    if normalized_root is None or (allowed_roots and normalized_root not in allowed_roots):
        issues.append((
            'sc_nevpt2_root_not_supported',
            'SC-NEVPT2 currently supports root=0 only.',
        ))
    return issues
