from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional, Protocol, Tuple, runtime_checkable


REGISTRY_SCHEMA = 'pyscf-agent.registry.v3'

ENTRY_KIND_CAPABILITY = 'capability'
ENTRY_KIND_PARAMETER = 'parameter'
ENTRY_KIND_OBSERVABLE = 'observable'
ENTRY_KIND_OPTION_SET = 'option_set'
ENTRY_KIND_TEMPLATE = 'template'
ENTRY_KIND_MODULE = 'module'
ENTRY_KIND_GATE = 'gate'
ENTRY_KIND_PROVIDER = 'provider'
ENTRY_KIND_BINDING = 'binding'
ENTRY_KIND_ARTIFACT = 'artifact'

ENTRY_KINDS = (
    ENTRY_KIND_CAPABILITY,
    ENTRY_KIND_PARAMETER,
    ENTRY_KIND_OBSERVABLE,
    ENTRY_KIND_OPTION_SET,
    ENTRY_KIND_TEMPLATE,
    ENTRY_KIND_MODULE,
    ENTRY_KIND_GATE,
    ENTRY_KIND_PROVIDER,
    ENTRY_KIND_BINDING,
    ENTRY_KIND_ARTIFACT,
)


@runtime_checkable
class RegistryContract(Protocol):
    def to_dict(self) -> Dict[str, Any]:
        ...


@dataclass(frozen=True)
class ProviderContract:
    provider_id: str
    label: str
    status: str = 'executable'
    optional_dependency: str = ''
    description: str = ''
    aliases: Tuple[str, ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CatalogItemContract:
    id: str
    label: str
    domain: str
    status: str
    description: str = ''
    limitations: List[str] = field(default_factory=list)
    ui_visibility: str = 'visible'
    planner_allowed: bool = False
    backend_allowed: bool = False
    wiki_slug: str = ''
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class CapabilityContract(CatalogItemContract):
    """Executable or design-time capability exposed by the platform."""

    required_provider_ids: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ParameterContract(CatalogItemContract):
    value_type: str = 'number'
    unit: str = ''
    scopes: Tuple[str, ...] = ()
    sweepable: bool = False
    choices: Tuple[str, ...] = ()


@dataclass(frozen=True)
class ObservableContract(CatalogItemContract):
    result_fields: Tuple[str, ...] = ()
    source_fields: Tuple[str, ...] = ()
    comparison_field: str = ''
    result_shape: str = ''
    unit: str = ''
    requestable_output: bool = False
    supports_plot: Tuple[str, ...] = ()
    requires_solvers: Tuple[str, ...] = ()


@dataclass(frozen=True)
class OptionValueContract:
    """One lightweight canonical value inside an option set."""

    id: str
    label: str = ''
    description: str = ''
    limitations: Tuple[str, ...] = ()
    aliases: Tuple[str, ...] = ()
    constraints: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class OptionSet:
    option_set_id: str
    label: str = ''
    domain: str = ''
    status: str = 'executable'
    description: str = ''
    ui_visibility: str = 'visible'
    planner_allowed: bool = False
    backend_allowed: bool = False
    wiki_slug: str = ''
    default: Optional[str] = None
    aliases: Dict[str, str] = field(default_factory=dict)
    items: Tuple[OptionValueContract, ...] = ()

    @property
    def values(self) -> Tuple[str, ...]:
        return tuple(item.id for item in self.items)

    @staticmethod
    def _normalize(value: object) -> str:
        return str(value or '').strip().lower()

    def canonical_id(self, value: object) -> Optional[str]:
        normalized = self._normalize(value)
        alias = self.aliases.get(normalized)
        if alias is not None:
            return alias
        for item in self.items:
            if any(
                self._normalize(identifier) == normalized
                for identifier in (item.id,) + tuple(item.aliases)
            ):
                return item.id
        return None

    def contains(self, value: object) -> bool:
        return self.canonical_id(value) is not None

    def to_dict(self) -> Dict[str, Any]:
        return {
            'option_set_id': self.option_set_id,
            'label': self.label,
            'domain': self.domain,
            'status': self.status,
            'description': self.description,
            'ui_visibility': self.ui_visibility,
            'planner_allowed': self.planner_allowed,
            'backend_allowed': self.backend_allowed,
            'wiki_slug': self.wiki_slug,
            'default': self.default,
            'aliases': dict(self.aliases),
            'items': [item.to_dict() for item in self.items],
        }


@dataclass(frozen=True)
class ArtifactContract:
    """Versioned artifact family exchanged across runtime boundaries."""

    id: str
    label: str
    status: str = 'executable'
    description: str = ''
    artifact_kinds: Tuple[str, ...] = ()
    artifact_kind_patterns: Tuple[str, ...] = ()
    payload_schemas: Dict[str, str] = field(default_factory=dict)
    binary_artifact_kinds: Tuple[str, ...] = ()
    result_fields: Tuple[str, ...] = ()
    numeric_container: str = ''
    required_reference_fields: Tuple[str, ...] = ()
    continuation_contract: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ProviderBinding:
    binding_id: str
    module_id: str
    provider_id: str
    runtime_id: str
    priority: int = 100
    description: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WorkflowTemplateContract:
    template_id: str
    label: str
    scope: str
    status: str = 'executable'
    required_module_ids: Tuple[str, ...] = ()
    optional_module_ids: Tuple[str, ...] = ()
    entry_capability_ids: Tuple[str, ...] = ()
    description: str = ''

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RegistryEntry:
    entry_id: str
    kind: str
    namespace: str
    local_id: str
    contract: RegistryContract
    aliases: Tuple[str, ...] = ()
    tags: Tuple[str, ...] = ()
    source_family: str = ''
    order: int = 0

    def __post_init__(self) -> None:
        if self.kind not in ENTRY_KINDS:
            raise ValueError("Unsupported registry entry kind: '{0}'".format(self.kind))
        if not self.entry_id or not self.namespace or not self.local_id:
            raise ValueError('Registry entries require entry_id, namespace, and local_id.')
        if not isinstance(self.contract, RegistryContract):
            raise TypeError("Registry contract for '{0}' must provide to_dict().".format(self.entry_id))

    def to_dict(self) -> Dict[str, Any]:
        return {
            'id': self.entry_id,
            'kind': self.kind,
            'namespace': self.namespace,
            'local_id': self.local_id,
            'aliases': list(self.aliases),
            'tags': list(self.tags),
            'source_family': self.source_family,
            'order': self.order,
            'contract': self.contract.to_dict(),
        }


@dataclass(frozen=True)
class RegistryIssue:
    code: str
    message: str
    entry_id: str = ''
    severity: str = 'error'
    details: Tuple[Tuple[str, str], ...] = ()

    def to_dict(self) -> Dict[str, Any]:
        return {
            'code': self.code,
            'message': self.message,
            'entry_id': self.entry_id,
            'severity': self.severity,
            'details': dict(self.details),
        }


__all__ = [
    'ENTRY_KINDS',
    'ENTRY_KIND_ARTIFACT',
    'ENTRY_KIND_BINDING',
    'ENTRY_KIND_CAPABILITY',
    'ENTRY_KIND_PARAMETER',
    'ENTRY_KIND_OBSERVABLE',
    'ENTRY_KIND_GATE',
    'ENTRY_KIND_MODULE',
    'ENTRY_KIND_OPTION_SET',
    'ENTRY_KIND_PROVIDER',
    'ENTRY_KIND_TEMPLATE',
    'ArtifactContract',
    'CapabilityContract',
    'CatalogItemContract',
    'ObservableContract',
    'OptionSet',
    'OptionValueContract',
    'ParameterContract',
    'ProviderBinding',
    'ProviderContract',
    'REGISTRY_SCHEMA',
    'RegistryContract',
    'RegistryEntry',
    'RegistryIssue',
    'WorkflowTemplateContract',
]
