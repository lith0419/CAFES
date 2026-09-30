from __future__ import annotations

from collections import defaultdict
from typing import DefaultDict, Dict, List, Optional, Set, Tuple

from ..workflow_gates.contracts import GateContract
from ..workflow_modules.contracts import ModuleContract

from .contracts import (
    ENTRY_KIND_ARTIFACT,
    ENTRY_KIND_BINDING,
    ENTRY_KIND_CAPABILITY,
    ENTRY_KIND_GATE,
    ENTRY_KIND_MODULE,
    ENTRY_KIND_OBSERVABLE,
    ENTRY_KIND_OPTION_SET,
    ENTRY_KIND_PARAMETER,
    ENTRY_KIND_PROVIDER,
    ENTRY_KIND_TEMPLATE,
    ArtifactContract,
    CapabilityContract,
    ObservableContract,
    OptionSet,
    ParameterContract,
    REGISTRY_SCHEMA,
    ProviderBinding,
    ProviderContract,
    RegistryEntry,
    RegistryIssue,
    WorkflowTemplateContract,
)


def _normalize(value: object) -> str:
    return str(value or '').strip().lower().replace('-', '_')


class UnifiedRegistryIndex:
    """One typed index over capabilities, option sets, workflows, and runtimes."""

    def __init__(self) -> None:
        self._entries: Dict[str, RegistryEntry] = {}
        self._aliases: DefaultDict[str, Set[str]] = defaultdict(set)
        self._issues: List[RegistryIssue] = []

    def register(self, entry: RegistryEntry) -> None:
        normalized = _normalize(entry.entry_id)
        if normalized in self._entries:
            self._issues.append(RegistryIssue(
                'duplicate_registry_entry',
                "Registry entry '{0}' is registered more than once.".format(entry.entry_id),
                entry.entry_id,
            ))
            return
        self._entries[normalized] = entry
        for alias in (entry.entry_id, entry.local_id) + tuple(entry.aliases):
            if str(alias or '').strip():
                self._aliases[_normalize(alias)].add(normalized)

    def entry(self, entry_id: object, *, kind: str = '') -> Optional[RegistryEntry]:
        normalized = _normalize(entry_id)
        direct = self._entries.get(normalized)
        if direct is not None and (not kind or direct.kind == kind):
            return direct
        candidates = [
            self._entries[candidate]
            for candidate in sorted(self._aliases.get(normalized, set()))
            if not kind or self._entries[candidate].kind == kind
        ]
        return candidates[0] if len(candidates) == 1 else None

    def entries(
        self,
        *,
        kind: str = '',
        namespace: str = '',
        source_family: str = '',
    ) -> List[RegistryEntry]:
        values = list(self._entries.values())
        if kind:
            values = [entry for entry in values if entry.kind == kind]
        if namespace:
            prefix = namespace.rstrip('.') + '.'
            values = [
                entry for entry in values
                if entry.namespace == namespace or entry.namespace.startswith(prefix)
            ]
        if source_family:
            values = [entry for entry in values if entry.source_family == source_family]
        return sorted(values, key=lambda entry: (entry.source_family, entry.order, entry.entry_id))

    def _missing_reference_issue(self, source: RegistryEntry, reference: str, relation: str) -> RegistryIssue:
        return RegistryIssue(
            'missing_registry_reference',
            "Registry entry '{0}' {1} unknown entry '{2}'.".format(
                source.entry_id,
                relation,
                reference,
            ),
            source.entry_id,
            details=(('reference', reference), ('relation', relation)),
        )

    @staticmethod
    def _contract_matches_kind(entry: RegistryEntry) -> bool:
        contract = entry.contract
        if entry.kind == ENTRY_KIND_CAPABILITY:
            return isinstance(contract, CapabilityContract)
        if entry.kind == ENTRY_KIND_PARAMETER:
            return isinstance(contract, ParameterContract)
        if entry.kind == ENTRY_KIND_OBSERVABLE:
            return isinstance(contract, ObservableContract)
        if entry.kind == ENTRY_KIND_OPTION_SET:
            return isinstance(contract, OptionSet)
        if entry.kind == ENTRY_KIND_TEMPLATE:
            return isinstance(contract, WorkflowTemplateContract)
        if entry.kind == ENTRY_KIND_MODULE:
            return isinstance(contract, ModuleContract)
        if entry.kind == ENTRY_KIND_GATE:
            return isinstance(contract, GateContract)
        if entry.kind == ENTRY_KIND_PROVIDER:
            return isinstance(contract, ProviderContract)
        if entry.kind == ENTRY_KIND_BINDING:
            return isinstance(contract, ProviderBinding)
        if entry.kind == ENTRY_KIND_ARTIFACT:
            return isinstance(contract, ArtifactContract)
        return False

    @staticmethod
    def _contract_kind_issue(entry: RegistryEntry) -> RegistryIssue:
        return RegistryIssue(
            'registry_contract_kind_mismatch',
            "Registry entry '{0}' uses contract '{1}' for kind '{2}'.".format(
                entry.entry_id,
                type(entry.contract).__name__,
                entry.kind,
            ),
            entry.entry_id,
            details=(
                ('contract_type', type(entry.contract).__name__),
                ('entry_kind', entry.kind),
            ),
        )

    def validate(self) -> Tuple[RegistryIssue, ...]:
        issues = list(self._issues)
        bindings_by_module: DefaultDict[str, List[RegistryEntry]] = defaultdict(list)
        capability_owners: DefaultDict[str, List[RegistryEntry]] = defaultdict(list)
        for binding_entry in self.entries(kind=ENTRY_KIND_BINDING):
            contract = binding_entry.contract
            if isinstance(contract, ProviderBinding):
                bindings_by_module[_normalize(contract.module_id)].append(binding_entry)
        for module_entry in self.entries(kind=ENTRY_KIND_MODULE):
            contract = module_entry.contract
            if not isinstance(contract, ModuleContract):
                continue
            for capability_id in contract.capability_ids:
                capability_owners[_normalize(capability_id)].append(module_entry)
        for entry in self.entries():
            if not self._contract_matches_kind(entry):
                issues.append(self._contract_kind_issue(entry))
            if entry.kind == ENTRY_KIND_TEMPLATE:
                contract = entry.contract
                if isinstance(contract, WorkflowTemplateContract):
                    module_references = contract.required_module_ids + contract.optional_module_ids
                    duplicate_module_ids = sorted(
                        set(contract.required_module_ids).intersection(contract.optional_module_ids)
                    )
                    if duplicate_module_ids:
                        issues.append(RegistryIssue(
                            'duplicate_workflow_module_role',
                            "Workflow template '{0}' lists modules as both required and optional.".format(
                                entry.entry_id
                            ),
                            entry.entry_id,
                            details=(('module_ids', ','.join(duplicate_module_ids)),),
                        ))
                    template_modules: Set[str] = set()
                    for reference in module_references:
                        module_entry = self.entry(reference, kind=ENTRY_KIND_MODULE)
                        if module_entry is None:
                            issues.append(self._missing_reference_issue(entry, reference, 'references module'))
                        else:
                            template_modules.add(module_entry.entry_id)
                            expected_namespace = contract.scope + '.module'
                            if not module_entry.namespace.startswith(expected_namespace):
                                issues.append(RegistryIssue(
                                    'workflow_module_scope_mismatch',
                                    "Workflow template '{0}' includes a module outside its '{1}' scope.".format(
                                        entry.entry_id,
                                        contract.scope,
                                    ),
                                    entry.entry_id,
                                    details=(('module_id', module_entry.entry_id),),
                                ))
                    for module_id in sorted(template_modules):
                        module_entry = self.entry(module_id, kind=ENTRY_KIND_MODULE)
                        if module_entry is None or not isinstance(module_entry.contract, ModuleContract):
                            continue
                        module_scope = module_entry.namespace.split('.', 1)[0]
                        for dependency in module_entry.contract.requires_modules:
                            dependency_id = '{0}.module.{1}'.format(module_scope, dependency)
                            dependency_entry = self.entry(dependency_id, kind=ENTRY_KIND_MODULE)
                            if dependency_entry is not None and dependency_entry.entry_id not in template_modules:
                                issues.append(RegistryIssue(
                                    'workflow_module_dependency_omitted',
                                    "Workflow template '{0}' omits a registered module dependency.".format(
                                        entry.entry_id
                                    ),
                                    entry.entry_id,
                                    details=(
                                        ('module_id', module_entry.entry_id),
                                        ('dependency_id', dependency_entry.entry_id),
                                    ),
                                ))
                    for reference in contract.entry_capability_ids:
                        if self.entry(reference, kind='capability') is None:
                            issues.append(self._missing_reference_issue(entry, reference, 'exposes capability'))
            elif entry.kind == ENTRY_KIND_BINDING:
                contract = entry.contract
                if isinstance(contract, ProviderBinding):
                    if self.entry(contract.module_id, kind='module') is None:
                        issues.append(self._missing_reference_issue(entry, contract.module_id, 'binds module'))
                    if self.entry(contract.provider_id, kind='provider') is None:
                        issues.append(self._missing_reference_issue(entry, contract.provider_id, 'uses provider'))
                    if not contract.runtime_id.strip():
                        issues.append(RegistryIssue(
                            'missing_provider_runtime',
                            "Provider binding '{0}' does not name a runtime adapter.".format(
                                entry.entry_id
                            ),
                            entry.entry_id,
                        ))
            elif entry.kind == ENTRY_KIND_CAPABILITY:
                contract = entry.contract
                if isinstance(contract, CapabilityContract):
                    for provider_id in contract.required_provider_ids:
                        if self.entry(provider_id, kind='provider') is None:
                            issues.append(self._missing_reference_issue(
                                entry,
                                provider_id,
                                'requires provider',
                            ))
                owners = capability_owners.get(_normalize(entry.entry_id), [])
                if not owners and isinstance(contract, CapabilityContract) and contract.status == 'executable':
                    issues.append(RegistryIssue(
                        'missing_capability_owner',
                        "Executable capability '{0}' does not belong to a module.".format(
                            entry.entry_id
                        ),
                        entry.entry_id,
                    ))
                elif len(owners) > 1:
                    issues.append(RegistryIssue(
                        'ambiguous_capability_owner',
                        "Capability '{0}' belongs to multiple modules.".format(entry.entry_id),
                        entry.entry_id,
                        details=(('module_count', str(len(owners))),),
                    ))
                elif len(owners) == 1 and isinstance(contract, CapabilityContract):
                    owner = owners[0]
                    owner_bindings = bindings_by_module.get(_normalize(owner.entry_id), [])
                    if len(owner_bindings) == 1:
                        provider_id = owner_bindings[0].contract.provider_id
                        if contract.required_provider_ids and provider_id not in contract.required_provider_ids:
                            issues.append(RegistryIssue(
                                'capability_provider_mismatch',
                                "Capability '{0}' requires a provider not bound to its owning module.".format(
                                    entry.entry_id
                                ),
                                entry.entry_id,
                                details=(
                                    ('module_id', owner.entry_id),
                                    ('bound_provider_id', provider_id),
                                    ('required_provider_ids', ','.join(contract.required_provider_ids)),
                                ),
                            ))
            elif entry.kind == ENTRY_KIND_MODULE:
                contract = entry.contract
                if isinstance(contract, ModuleContract):
                    scope = entry.namespace.split('.', 1)[0]
                    for capability_id in contract.capability_ids:
                        if self.entry(capability_id, kind=ENTRY_KIND_CAPABILITY) is None:
                            issues.append(self._missing_reference_issue(
                                entry,
                                capability_id,
                                'owns capability',
                            ))
                    for module_id in contract.requires_modules:
                        reference = '{0}.module.{1}'.format(scope, module_id)
                        if self.entry(reference, kind=ENTRY_KIND_MODULE) is None:
                            issues.append(self._missing_reference_issue(
                                entry,
                                reference,
                                'requires module',
                            ))
                bindings = bindings_by_module.get(_normalize(entry.entry_id), [])
                if not bindings:
                    issues.append(RegistryIssue(
                        'missing_provider_binding',
                        "Module '{0}' has no provider runtime binding.".format(entry.entry_id),
                        entry.entry_id,
                    ))
                elif len(bindings) > 1:
                    issues.append(RegistryIssue(
                        'ambiguous_provider_binding',
                        "Module '{0}' has multiple provider runtime bindings.".format(entry.entry_id),
                        entry.entry_id,
                        details=(('binding_count', str(len(bindings))),),
                    ))
        unique: Dict[Tuple[str, str, str], RegistryIssue] = {}
        for issue in issues:
            unique[(issue.code, issue.entry_id, issue.message)] = issue
        return tuple(unique[key] for key in sorted(unique))

    def capability_owner(self, capability_id: object) -> Optional[RegistryEntry]:
        capability = self.entry(capability_id, kind=ENTRY_KIND_CAPABILITY)
        if capability is None:
            return None
        owners = [
            entry
            for entry in self.entries(kind=ENTRY_KIND_MODULE)
            if isinstance(entry.contract, ModuleContract)
            and capability.entry_id in entry.contract.capability_ids
        ]
        return owners[0] if len(owners) == 1 else None

    def capabilities_for_module(self, module_id: object) -> Tuple[RegistryEntry, ...]:
        module = self.entry(module_id, kind=ENTRY_KIND_MODULE)
        if module is None or not isinstance(module.contract, ModuleContract):
            return ()
        return tuple(
            capability
            for capability_id in module.contract.capability_ids
            for capability in (self.entry(capability_id, kind=ENTRY_KIND_CAPABILITY),)
            if capability is not None
        )

    def hierarchy(self) -> Dict[str, object]:
        module_entries = self.entries(kind=ENTRY_KIND_MODULE)
        binding_by_module = {
            entry.contract.module_id: entry.contract
            for entry in self.entries(kind=ENTRY_KIND_BINDING)
            if isinstance(entry.contract, ProviderBinding)
        }
        assigned_capabilities = {
            capability.entry_id
            for module in module_entries
            for capability in self.capabilities_for_module(module.entry_id)
        }
        provider_modules: DefaultDict[str, List[str]] = defaultdict(list)
        for binding in binding_by_module.values():
            provider_modules[binding.provider_id].append(binding.module_id)
        return {
            'workflows': [
                {
                    'id': entry.entry_id,
                    'required_module_ids': list(entry.contract.required_module_ids),
                    'optional_module_ids': list(entry.contract.optional_module_ids),
                    'entry_capability_ids': list(entry.contract.entry_capability_ids),
                }
                for entry in self.entries(kind=ENTRY_KIND_TEMPLATE)
                if isinstance(entry.contract, WorkflowTemplateContract)
            ],
            'modules': [
                {
                    'id': entry.entry_id,
                    'capability_ids': list(entry.contract.capability_ids),
                    'required_module_ids': [
                        '{0}.module.{1}'.format(
                            entry.namespace.split('.', 1)[0],
                            module_id,
                        )
                        for module_id in entry.contract.requires_modules
                    ],
                    'provider_id': (
                        binding_by_module[entry.entry_id].provider_id
                        if entry.entry_id in binding_by_module
                        else ''
                    ),
                    'runtime_id': (
                        binding_by_module[entry.entry_id].runtime_id
                        if entry.entry_id in binding_by_module
                        else ''
                    ),
                }
                for entry in module_entries
                if isinstance(entry.contract, ModuleContract)
            ],
            'capabilities': [
                {
                    'id': entry.entry_id,
                    'module_id': (
                        owner.entry_id
                        if (owner := self.capability_owner(entry.entry_id)) is not None
                        else ''
                    ),
                    'required_provider_ids': list(entry.contract.required_provider_ids),
                }
                for entry in self.entries(kind=ENTRY_KIND_CAPABILITY)
                if isinstance(entry.contract, CapabilityContract)
            ],
            'providers': [
                {
                    'id': entry.entry_id,
                    'module_ids': sorted(provider_modules.get(entry.entry_id, [])),
                }
                for entry in self.entries(kind=ENTRY_KIND_PROVIDER)
                if isinstance(entry.contract, ProviderContract)
            ],
            'unassigned_capability_ids': [
                entry.entry_id
                for entry in self.entries(kind=ENTRY_KIND_CAPABILITY)
                if entry.entry_id not in assigned_capabilities
            ],
        }

    def as_dict(self) -> Dict[str, object]:
        issues = self.validate()
        return {
            'schema': REGISTRY_SCHEMA,
            'entries': [entry.to_dict() for entry in self.entries()],
            'hierarchy': self.hierarchy(),
            'validation': {
                'status': 'valid' if not any(issue.severity == 'error' for issue in issues) else 'invalid',
                'issues': [issue.to_dict() for issue in issues],
            },
        }


__all__ = ['UnifiedRegistryIndex']
