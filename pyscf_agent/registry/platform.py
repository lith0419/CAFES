from __future__ import annotations

from functools import lru_cache

import copy
import fnmatch
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from ..workflow_modules import (
    ModuleContract,
    ResolvedModuleContract,
    build_default_module_contracts,
    build_default_study_module_contracts,
)
from ..workflow_gates import (
    GateContract,
    build_default_study_gate_contracts,
    build_default_task_gate_contracts,
)
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
    CatalogItemContract,
    ObservableContract,
    OptionSet,
    OptionValueContract,
    ParameterContract,
    ProviderBinding,
    ProviderContract,
    RegistryEntry,
    RegistryIssue,
    WorkflowTemplateContract,
)
from .index import UnifiedRegistryIndex
from .catalog import (
    build_default_provider_bindings,
    build_default_provider_contracts,
    build_default_workflow_templates,
    study_gate_entry_id,
    study_module_entry_id,
    task_gate_entry_id,
    task_module_entry_id,
    OPTION_SET_IDS,
)
from .defaults import build_default_artifact_contracts, build_default_catalog_families


CAPABILITY_EXECUTABLE = 'executable'
CAPABILITY_DESIGN_ONLY = 'design_only'
CAPABILITY_PLANNED = 'planned'
CAPABILITY_REFERENCE_ONLY = 'reference_only'


def _normalize_capability_id(value: Any) -> str:
    return str(value or '').strip().lower().replace('-', '_')


def _catalog_item(
    capability_id: str,
    label: str,
    domain: str,
    status: str = CAPABILITY_EXECUTABLE,
    *,
    description: str = '',
    limitations: Optional[Iterable[str]] = None,
    ui_visibility: str = 'visible',
    planner_allowed: Optional[bool] = None,
    backend_allowed: Optional[bool] = None,
    wiki_slug: str = '',
    metadata: Optional[Dict[str, Any]] = None,
    required_provider_ids: Iterable[str] = (),
) -> CapabilityContract:
    executable = status == CAPABILITY_EXECUTABLE
    return CapabilityContract(
        id=capability_id,
        label=label,
        domain=domain,
        status=status,
        description=description,
        limitations=list(limitations or []),
        ui_visibility=ui_visibility,
        planner_allowed=executable if planner_allowed is None else bool(planner_allowed),
        backend_allowed=executable if backend_allowed is None else bool(backend_allowed),
        wiki_slug=wiki_slug,
        metadata=copy.deepcopy(metadata or {}),
        required_provider_ids=tuple(str(value) for value in required_provider_ids),
    )


@dataclass
class PlatformRegistry:
    _entry_index: UnifiedRegistryIndex = field(default_factory=UnifiedRegistryIndex, init=False, repr=False)
    _family_order: Tuple[str, ...] = field(default_factory=tuple, init=False, repr=False)

    def parameters(
        self,
        *,
        namespace: str = '',
        scope: str = '',
        sweepable: Optional[bool] = None,
        planner_allowed: Optional[bool] = None,
        backend_allowed: Optional[bool] = None,
    ) -> List[ParameterContract]:
        """Return parameters by semantic contract instead of catalog family."""

        parameters = [
            entry.contract
            for entry in self.entries_for(kind=ENTRY_KIND_PARAMETER, namespace=namespace)
            if isinstance(entry.contract, ParameterContract)
        ]
        if scope:
            parameters = [parameter for parameter in parameters if scope in parameter.scopes]
        if sweepable is not None:
            parameters = [parameter for parameter in parameters if parameter.sweepable == sweepable]
        if planner_allowed is not None:
            parameters = [parameter for parameter in parameters if parameter.planner_allowed == planner_allowed]
        if backend_allowed is not None:
            parameters = [parameter for parameter in parameters if parameter.backend_allowed == backend_allowed]
        return parameters

    def parameter(self, value: Any, *, namespace: str = '') -> Optional[ParameterContract]:
        entry = self._semantic_entry(value, kind=ENTRY_KIND_PARAMETER, namespace=namespace)
        return entry.contract if entry is not None and isinstance(entry.contract, ParameterContract) else None

    def observables(
        self,
        *,
        namespace: str = '',
        system_type: str = '',
        requestable: Optional[bool] = None,
        plottable: Optional[bool] = None,
        planner_allowed: Optional[bool] = None,
        backend_allowed: Optional[bool] = None,
    ) -> List[ObservableContract]:
        """Return observables by their output contract and compatibility."""

        observables = [
            entry.contract
            for entry in self.entries_for(
                kind=ENTRY_KIND_OBSERVABLE,
                namespace=namespace,
            )
            if isinstance(entry.contract, ObservableContract)
        ]
        if system_type:
            observables = [
                observable
                for observable in observables
                if not observable.metadata.get('system_types')
                or system_type in observable.metadata.get('system_types', [])
            ]
        if requestable is not None:
            observables = [observable for observable in observables if observable.requestable_output == requestable]
        if plottable is not None:
            observables = [observable for observable in observables if bool(observable.supports_plot) == plottable]
        if planner_allowed is not None:
            observables = [observable for observable in observables if observable.planner_allowed == planner_allowed]
        if backend_allowed is not None:
            observables = [observable for observable in observables if observable.backend_allowed == backend_allowed]
        return observables

    def observable(self, value: Any, *, namespace: str = '') -> Optional[ObservableContract]:
        entry = self._semantic_entry(value, kind=ENTRY_KIND_OBSERVABLE, namespace=namespace)
        return entry.contract if entry is not None and isinstance(entry.contract, ObservableContract) else None

    def capabilities(
        self,
        *,
        namespace: str = '',
        status: str = '',
        planner_allowed: Optional[bool] = None,
        backend_allowed: Optional[bool] = None,
        ui_visible: Optional[bool] = None,
    ) -> List[CapabilityContract]:
        """Return platform capabilities by semantic namespace."""

        entries = self.entries_for(kind=ENTRY_KIND_CAPABILITY, namespace=namespace)
        capabilities = [
            entry.contract
            for entry in entries
            if isinstance(entry.contract, CapabilityContract)
        ]
        if status:
            capabilities = [item for item in capabilities if item.status == status]
        if planner_allowed is not None:
            capabilities = [item for item in capabilities if item.planner_allowed == planner_allowed]
        if backend_allowed is not None:
            capabilities = [item for item in capabilities if item.backend_allowed == backend_allowed]
        if ui_visible is not None:
            capabilities = [item for item in capabilities if (item.ui_visibility != 'hidden') == ui_visible]
        return capabilities

    def capability(self, value: Any, *, namespace: str = '') -> Optional[CapabilityContract]:
        entry = self._semantic_entry(value, kind=ENTRY_KIND_CAPABILITY, namespace=namespace)
        return entry.contract if entry is not None and isinstance(entry.contract, CapabilityContract) else None

    def canonical_capability_id(self, value: Any, *, namespace: str = '') -> Optional[str]:
        entry = self._semantic_entry(value, kind=ENTRY_KIND_CAPABILITY, namespace=namespace)
        return entry.local_id if entry is not None else None

    def capability_is_allowed(self, value: Any, *, namespace: str = '', action: str = 'backend') -> bool:
        capability = self.capability(value, namespace=namespace)
        if capability is None:
            return False
        if action == 'planner':
            return capability.planner_allowed
        if action == 'ui':
            return capability.ui_visibility != 'hidden'
        return capability.backend_allowed

    def option_values(
        self,
        option_set_id: Any,
        *,
        planner_allowed: Optional[bool] = None,
        backend_allowed: Optional[bool] = None,
        ui_visible: Optional[bool] = None,
    ) -> List[OptionValueContract]:
        option_set = self.option_set(option_set_id)
        if option_set is None:
            return []
        if planner_allowed is not None and option_set.planner_allowed != planner_allowed:
            return []
        if backend_allowed is not None and option_set.backend_allowed != backend_allowed:
            return []
        if ui_visible is not None and (option_set.ui_visibility != 'hidden') != ui_visible:
            return []
        return list(option_set.items)

    def option_value(self, option_set_id: Any, value: Any) -> Optional[OptionValueContract]:
        option_set = self.option_set(option_set_id)
        canonical = option_set.canonical_id(value) if option_set is not None else None
        if canonical is None:
            return None
        return next((item for item in option_set.items if item.id == canonical), None)

    def canonical_option_id(self, option_set_id: Any, value: Any) -> Optional[str]:
        option_set = self.option_set(option_set_id)
        return option_set.canonical_id(value) if option_set is not None else None

    def as_dict(self) -> Dict[str, Any]:
        return {
            'schema': 'pyscf-agent.platform-registry.v3',
            'status_values': [
                CAPABILITY_EXECUTABLE,
                CAPABILITY_DESIGN_ONLY,
                CAPABILITY_PLANNED,
                CAPABILITY_REFERENCE_ONLY,
            ],
            'registry': self._entry_index.as_dict(),
        }

    @staticmethod
    def _catalog_entry_kind(family: str, capability: CatalogItemContract) -> str:
        if family == 'model_hamiltonian_parameters':
            return ENTRY_KIND_PARAMETER
        if family.endswith('_observables') or family == 'postprocessing_metrics':
            return ENTRY_KIND_OBSERVABLE
        return ENTRY_KIND_CAPABILITY

    @staticmethod
    def _artifact_catalog_projection(contract: ArtifactContract) -> CatalogItemContract:
        metadata: Dict[str, Any] = {}
        if contract.artifact_kinds:
            metadata['artifact_kinds'] = list(contract.artifact_kinds)
        if contract.artifact_kind_patterns:
            metadata['artifact_kind_patterns'] = list(contract.artifact_kind_patterns)
        if contract.payload_schemas:
            metadata['payload_schemas'] = dict(contract.payload_schemas)
        if contract.binary_artifact_kinds:
            metadata['binary_artifact_kinds'] = list(contract.binary_artifact_kinds)
        if contract.result_fields:
            metadata['result_fields'] = list(contract.result_fields)
        if contract.numeric_container:
            metadata['numeric_container'] = contract.numeric_container
        if contract.required_reference_fields:
            metadata['required_fields'] = list(contract.required_reference_fields)
        if contract.continuation_contract:
            metadata['continuation_contract'] = contract.continuation_contract
        return CatalogItemContract(
            id=contract.id,
            label=contract.label,
            domain='artifact.contract',
            status=contract.status,
            description=contract.description,
            ui_visibility='hidden',
            planner_allowed=False,
            backend_allowed=contract.status == CAPABILITY_EXECUTABLE,
            metadata=metadata,
        )

    @staticmethod
    def _option_catalog_projection(
        option_set: OptionSet,
        item: OptionValueContract,
    ) -> CatalogItemContract:
        metadata = copy.deepcopy(item.constraints)
        if item.aliases:
            metadata['aliases'] = list(item.aliases)
        return CatalogItemContract(
            id=item.id,
            label=item.label,
            domain=option_set.domain,
            status=option_set.status,
            description=item.description,
            limitations=list(item.limitations),
            ui_visibility=option_set.ui_visibility,
            planner_allowed=option_set.planner_allowed,
            backend_allowed=option_set.backend_allowed,
            wiki_slug=option_set.wiki_slug,
            metadata=metadata,
        )

    @staticmethod
    def _catalog_compatibility_projection(entry: RegistryEntry) -> CatalogItemContract:
        contract = entry.contract
        if not isinstance(contract, CatalogItemContract):
            raise TypeError("Entry '{0}' is not catalog-compatible.".format(entry.entry_id))
        metadata = copy.deepcopy(contract.metadata)
        if entry.aliases:
            metadata['aliases'] = list(entry.aliases)
        payload = contract.to_dict()
        payload['metadata'] = metadata
        if type(contract) is not CatalogItemContract:
            allowed = CatalogItemContract.__dataclass_fields__
            payload = {key: value for key, value in payload.items() if key in allowed}
        if isinstance(contract, ObservableContract):
            if contract.requires_solvers:
                payload['metadata']['requires_solver'] = list(contract.requires_solvers)
        return CatalogItemContract(**payload)

    @staticmethod
    def _catalog_entry_id(capability: CatalogItemContract) -> str:
        return '{0}.{1}'.format(capability.domain, capability.id)

    @staticmethod
    def _parameter_contract(capability: CatalogItemContract) -> ParameterContract:
        value_types = {
            'boundary': 'string',
            'cell_offset': 'integer_vector',
            'electrons_per_cell': 'number',
            'kmesh': 'integer_vector',
            'lattice_vectors': 'matrix',
            'nelec': 'electron_count',
            'representation': 'string',
        }
        energy_parameters = {'t', 'U', 'V', 'epsilon', 'effective_t', 'effective_V'}
        scopes = tuple(str(value) for value in capability.metadata.get('scopes', []))
        payload = capability.to_dict()
        payload.pop('required_provider_ids', None)
        metadata = copy.deepcopy(payload.get('metadata') or {})
        choices = tuple(str(value) for value in metadata.pop('values', []))
        metadata.pop('scopes', None)
        metadata.pop('aliases', None)
        metadata.pop('module_id', None)
        metadata.pop('provider_id', None)
        metadata.pop('requires_solver', None)
        payload['metadata'] = metadata
        return ParameterContract(
            **payload,
            value_type=value_types.get(capability.id, 'number'),
            unit='energy' if capability.id in energy_parameters else '',
            scopes=scopes,
            sweepable='sweep' in scopes,
            choices=choices,
        )

    @staticmethod
    def _observable_contract(capability: CatalogItemContract) -> ObservableContract:
        payload = capability.to_dict()
        payload.pop('required_provider_ids', None)
        metadata = copy.deepcopy(payload.get('metadata') or {})
        result_fields = list(metadata.pop('result_fields', []) or [])
        source_field = str(metadata.pop('result_field', '') or '')
        comparison_field = str(metadata.pop('comparison_summary', '') or '')
        source_fields: List[str] = []
        if comparison_field:
            if source_field:
                source_fields.append(source_field)
            result_fields.append(comparison_field)
        elif source_field:
            result_fields.append(source_field)
        result_shape = str(metadata.pop('result_shape', '') or '')
        unit = str(metadata.pop('unit', '') or '')
        requestable_output = bool(metadata.pop('requestable_output', False))
        supports_plot = tuple(str(value) for value in metadata.pop('supports_plot', []))
        requires_solvers = tuple(str(value) for value in metadata.pop('requires_solver', []))
        metadata.pop('module_id', None)
        metadata.pop('provider_id', None)
        metadata.pop('aliases', None)
        payload['metadata'] = metadata
        return ObservableContract(
            **payload,
            result_fields=tuple(dict.fromkeys(str(value) for value in result_fields)),
            source_fields=tuple(dict.fromkeys(source_fields)),
            comparison_field=comparison_field,
            result_shape=result_shape,
            unit=unit,
            requestable_output=requestable_output,
            supports_plot=supports_plot,
            requires_solvers=requires_solvers,
        )

    @staticmethod
    def _capability_contract(capability: CatalogItemContract) -> CapabilityContract:
        payload = capability.to_dict()
        metadata = copy.deepcopy(payload.get('metadata') or {})
        for key in ('aliases', 'module_id', 'provider_id', 'requires_solver'):
            metadata.pop(key, None)
        payload['metadata'] = metadata
        return CapabilityContract(**payload)

    @classmethod
    def _typed_catalog_contract(cls, family: str, capability: CatalogItemContract) -> CatalogItemContract:
        kind = cls._catalog_entry_kind(family, capability)
        if kind == ENTRY_KIND_PARAMETER:
            return cls._parameter_contract(capability)
        if kind == ENTRY_KIND_OBSERVABLE:
            return cls._observable_contract(capability)
        return cls._capability_contract(capability)

    def rebuild_entry_index(
        self,
        catalog_families: Mapping[str, Sequence[CatalogItemContract]],
        *,
        workflow_modules: Sequence[ModuleContract],
        study_workflow_modules: Sequence[ModuleContract],
        workflow_gates: Sequence[GateContract],
        study_workflow_gates: Sequence[GateContract],
        artifact_contracts: Sequence[ArtifactContract],
        workflow_templates: Sequence[WorkflowTemplateContract],
        providers: Sequence[ProviderContract],
        provider_bindings: Sequence[ProviderBinding],
    ) -> UnifiedRegistryIndex:
        """Build the sole typed store for catalog items and runtime contracts."""

        index = UnifiedRegistryIndex()
        self._family_order = tuple(catalog_families) + ('artifact_contracts',)
        for family, capabilities in catalog_families.items():
            if family in OPTION_SET_IDS:
                aliases: Dict[str, str] = {}
                option_values: List[OptionValueContract] = []
                for capability in capabilities:
                    item_aliases = tuple(
                        str(alias) for alias in capability.metadata.get('aliases', [])
                    )
                    for alias in item_aliases:
                        aliases[str(alias).strip().lower()] = capability.id
                    constraints = copy.deepcopy(capability.metadata)
                    constraints.pop('aliases', None)
                    constraints.pop('default', None)
                    option_values.append(OptionValueContract(
                        id=capability.id,
                        label=capability.label,
                        description=capability.description,
                        limitations=tuple(capability.limitations),
                        aliases=item_aliases,
                        constraints=constraints,
                    ))
                option_set_id = OPTION_SET_IDS[family]
                representative = capabilities[0] if capabilities else None
                if representative is not None:
                    policy_fields = (
                        'domain',
                        'status',
                        'ui_visibility',
                        'planner_allowed',
                        'backend_allowed',
                        'wiki_slug',
                    )
                    for capability in capabilities[1:]:
                        mismatches = [
                            field_name
                            for field_name in policy_fields
                            if getattr(capability, field_name) != getattr(representative, field_name)
                        ]
                        if mismatches:
                            raise ValueError(
                                "Option family '{0}' has inconsistent family policy on '{1}': {2}.".format(
                                    family,
                                    capability.id,
                                    ', '.join(mismatches),
                                )
                            )
                option_set = OptionSet(
                    option_set_id=option_set_id,
                    label=family.replace('_', ' ').title(),
                    domain=representative.domain if representative is not None else option_set_id,
                    status=representative.status if representative is not None else CAPABILITY_EXECUTABLE,
                    description='Registered values for {0}.'.format(family.replace('_', ' ')),
                    ui_visibility=representative.ui_visibility if representative is not None else 'visible',
                    planner_allowed=bool(representative.planner_allowed) if representative is not None else False,
                    backend_allowed=bool(representative.backend_allowed) if representative is not None else False,
                    wiki_slug=representative.wiki_slug if representative is not None else '',
                    default=next((
                        capability.id
                        for capability in capabilities
                        if capability.metadata.get('default')
                    ), None),
                    aliases=aliases,
                    items=tuple(option_values),
                )
                index.register(RegistryEntry(
                    entry_id=option_set_id,
                    kind=ENTRY_KIND_OPTION_SET,
                    namespace=option_set_id.rsplit('.', 1)[0],
                    local_id=option_set_id.rsplit('.', 1)[-1],
                    contract=option_set,
                    aliases=(family,),
                    tags=(family,),
                    source_family=family,
                ))
                continue

            for order, capability in enumerate(capabilities):
                typed_contract = self._typed_catalog_contract(family, capability)
                entry_id = self._catalog_entry_id(capability)
                index.register(RegistryEntry(
                    entry_id=entry_id,
                    kind=self._catalog_entry_kind(family, capability),
                    namespace=capability.domain,
                    local_id=capability.id,
                    contract=typed_contract,
                    aliases=tuple(str(value) for value in capability.metadata.get('aliases', [])),
                    tags=(family,),
                    source_family=family,
                    order=order,
                ))

        for order, contract in enumerate(artifact_contracts):
            index.register(RegistryEntry(
                entry_id='artifact.contract.{0}'.format(contract.id),
                kind=ENTRY_KIND_ARTIFACT,
                namespace='artifact.contract',
                local_id=contract.id,
                contract=contract,
                aliases=(contract.id,),
                tags=('artifact_contracts',),
                source_family='artifact_contracts',
                order=order,
            ))

        for order, provider in enumerate(providers):
            index.register(RegistryEntry(
                entry_id=provider.provider_id,
                kind=ENTRY_KIND_PROVIDER,
                namespace=provider.provider_id.rsplit('.', 1)[0],
                local_id=provider.provider_id.rsplit('.', 1)[-1],
                contract=provider,
                aliases=provider.aliases,
                tags=('execution_provider',),
                source_family='providers',
                order=order,
            ))

        for scope, modules, entry_id_factory in (
            ('task', workflow_modules, task_module_entry_id),
            ('study', study_workflow_modules, study_module_entry_id),
        ):
            for order, module in enumerate(modules):
                entry_id = entry_id_factory(module.module_id)
                index.register(RegistryEntry(
                    entry_id=entry_id,
                    kind=ENTRY_KIND_MODULE,
                    namespace='{0}.module'.format(scope),
                    local_id=module.module_id,
                    contract=module,
                    aliases=(module.module_id,),
                    tags=(scope, module.default_stage),
                    source_family='{0}_workflow_modules'.format(scope),
                    order=order,
                ))

        for scope, gates, entry_id_factory in (
            ('task', workflow_gates, task_gate_entry_id),
            ('study', study_workflow_gates, study_gate_entry_id),
        ):
            for order, gate in enumerate(gates):
                entry_id = entry_id_factory(gate.gate_id)
                index.register(RegistryEntry(
                    entry_id=entry_id,
                    kind=ENTRY_KIND_GATE,
                    namespace='{0}.gate'.format(scope),
                    local_id=gate.gate_id,
                    contract=gate,
                    aliases=(gate.gate_id,),
                    tags=(scope, gate.default_hook),
                    source_family='{0}_workflow_gates'.format(scope),
                    order=order,
                ))

        for order, template in enumerate(workflow_templates):
            index.register(RegistryEntry(
                entry_id=template.template_id,
                kind=ENTRY_KIND_TEMPLATE,
                namespace=template.template_id.rsplit('.', 1)[0],
                local_id=template.template_id.rsplit('.', 1)[-1],
                contract=template,
                aliases=(template.label,),
                tags=(template.scope, 'non_limiting'),
                source_family='workflow_templates',
                order=order,
            ))

        for order, binding in enumerate(provider_bindings):
            index.register(RegistryEntry(
                entry_id=binding.binding_id,
                kind=ENTRY_KIND_BINDING,
                namespace=binding.binding_id.rsplit('.', 1)[0],
                local_id=binding.binding_id.rsplit('.', 1)[-1],
                contract=binding,
                tags=('provider_binding',),
                source_family='provider_bindings',
                order=order,
            ))
        self._entry_index = index
        return index

    def entry(self, entry_id: Any, *, kind: str = '') -> Optional[RegistryEntry]:
        return self._entry_index.entry(entry_id, kind=kind)

    def _semantic_entry(
        self,
        value: Any,
        *,
        kind: str,
        namespace: str = '',
    ) -> Optional[RegistryEntry]:
        entry = self.entry(value, kind=kind)
        if entry is not None and (not namespace or entry.namespace == namespace):
            return entry
        normalized = _normalize_capability_id(value)
        matches = []
        for candidate in self.entries_for(kind=kind, namespace=namespace):
            if namespace and candidate.namespace != namespace:
                continue
            identifiers = (candidate.entry_id, candidate.local_id) + tuple(candidate.aliases)
            if any(_normalize_capability_id(identifier) == normalized for identifier in identifiers):
                matches.append(candidate)
        return matches[0] if len(matches) == 1 else None

    def entries_for(
        self,
        *,
        kind: str = '',
        namespace: str = '',
        source_family: str = '',
    ) -> List[RegistryEntry]:
        return self._entry_index.entries(
            kind=kind,
            namespace=namespace,
            source_family=source_family,
        )

    def option_set(self, option_set_id: Any) -> Optional[OptionSet]:
        reference = OPTION_SET_IDS.get(str(option_set_id or '').strip(), option_set_id)
        entry = self.entry(reference, kind=ENTRY_KIND_OPTION_SET)
        return entry.contract if entry is not None else None

    def artifact_contract(self, artifact_contract_id: Any) -> Optional[ArtifactContract]:
        entry = self.entry(artifact_contract_id, kind=ENTRY_KIND_ARTIFACT)
        return entry.contract if entry is not None and isinstance(entry.contract, ArtifactContract) else None

    def artifact_contracts(self, *, status: str = '') -> List[ArtifactContract]:
        contracts = [
            entry.contract
            for entry in self.entries_for(kind=ENTRY_KIND_ARTIFACT)
            if isinstance(entry.contract, ArtifactContract)
        ]
        if status:
            contracts = [contract for contract in contracts if contract.status == status]
        return contracts

    def registry_issues(self) -> Tuple[RegistryIssue, ...]:
        return self._entry_index.validate()

    def registry_payload(self) -> Dict[str, Any]:
        """Return the serialized typed index without exposing its storage."""

        return self._entry_index.as_dict()

    def module(self, module_id: Any) -> Optional[ResolvedModuleContract]:
        entry = self.entry(module_id, kind=ENTRY_KIND_MODULE)
        return self._bound_module_contract(entry)

    def _bound_module_contract(self, entry: Optional[RegistryEntry]) -> Optional[ResolvedModuleContract]:
        if entry is None or not isinstance(entry.contract, ModuleContract):
            return None
        binding = self.provider_binding(entry.entry_id)
        if binding is None:
            return None
        return ResolvedModuleContract(
            **{
                contract_field.name: getattr(entry.contract, contract_field.name)
                for contract_field in fields(ModuleContract)
            },
            provider_id=binding.provider_id,
            runtime_id=binding.runtime_id,
        )

    def modules_for(
        self,
        stage: str = None,
        task_type: str = None,
        *,
        scope: str = 'task',
    ) -> List[ResolvedModuleContract]:
        source_family = '{0}_workflow_modules'.format(scope)
        registered = self.entries_for(kind=ENTRY_KIND_MODULE, source_family=source_family)
        modules = [
            module
            for entry in registered
            for module in (self._bound_module_contract(entry),)
            if module is not None
        ]
        if stage is not None:
            modules = [module for module in modules if stage in module.permitted_stages]
        if task_type is not None:
            modules = [
                module for module in modules
                if not module.compatibility.task_types or task_type in module.compatibility.task_types
            ]
        return modules

    def gate(self, gate_id: Any) -> Optional[GateContract]:
        entry = self.entry(gate_id, kind=ENTRY_KIND_GATE)
        return entry.contract if entry is not None else None

    def gates_for(
        self,
        hook: str = None,
        task_type: str = None,
        *,
        scope: str = 'task',
    ) -> List[GateContract]:
        source_family = '{0}_workflow_gates'.format(scope)
        registered = self.entries_for(kind=ENTRY_KIND_GATE, source_family=source_family)
        gates = [entry.contract for entry in registered]
        if hook is not None:
            gates = [gate for gate in gates if hook in gate.permitted_hooks]
        if task_type is not None:
            gates = [
                gate for gate in gates
                if not gate.compatibility.task_types or task_type in gate.compatibility.task_types
            ]
        return gates

    def providers_for(self, *, status: str = '') -> List[ProviderContract]:
        providers = [
            entry.contract
            for entry in self.entries_for(kind=ENTRY_KIND_PROVIDER)
            if isinstance(entry.contract, ProviderContract)
        ]
        if status:
            providers = [provider for provider in providers if provider.status == status]
        return providers

    def templates_for(self, *, scope: str = '', status: str = '') -> List[WorkflowTemplateContract]:
        templates = [
            entry.contract
            for entry in self.entries_for(kind=ENTRY_KIND_TEMPLATE)
            if isinstance(entry.contract, WorkflowTemplateContract)
        ]
        if scope:
            templates = [template for template in templates if template.scope == scope]
        if status:
            templates = [template for template in templates if template.status == status]
        return templates

    def provider_binding(self, module_id: Any) -> Optional[ProviderBinding]:
        module = self.entry(module_id, kind=ENTRY_KIND_MODULE)
        canonical_module = module.entry_id if module is not None else str(module_id or '').strip()
        matches = [
            entry.contract
            for entry in self.entries_for(kind=ENTRY_KIND_BINDING)
            if entry.contract.module_id == canonical_module
        ]
        return matches[0] if len(matches) == 1 else None

    def capabilities_for_module(self, module_id: Any) -> List[CapabilityContract]:
        return [
            entry.contract
            for entry in self._entry_index.capabilities_for_module(module_id)
            if isinstance(entry.contract, CapabilityContract)
        ]

    def module_for_capability(self, capability_id: Any) -> Optional[ModuleContract]:
        owner = self._entry_index.capability_owner(capability_id)
        return self._bound_module_contract(owner)

    def modules_for_template(
        self,
        template_id: Any,
        *,
        include_optional: bool = True,
    ) -> List[ModuleContract]:
        template_entry = self.entry(template_id, kind=ENTRY_KIND_TEMPLATE)
        if template_entry is None or not isinstance(
            template_entry.contract,
            WorkflowTemplateContract,
        ):
            return []
        module_ids = list(template_entry.contract.required_module_ids)
        if include_optional:
            module_ids.extend(template_entry.contract.optional_module_ids)
        return [
            contract
            for module_id in module_ids
            for module in (self.entry(module_id, kind=ENTRY_KIND_MODULE),)
            for contract in (self._bound_module_contract(module),)
            if contract is not None
        ]


@lru_cache(maxsize=1)
def default_registry() -> PlatformRegistry:
    catalog_families = build_default_catalog_families(
        _catalog_item,
        capability_design_only=CAPABILITY_DESIGN_ONLY,
        capability_planned=CAPABILITY_PLANNED,
    )
    workflow_modules = build_default_module_contracts()
    study_workflow_modules = build_default_study_module_contracts()
    workflow_gates = build_default_task_gate_contracts()
    study_workflow_gates = build_default_study_gate_contracts()
    artifact_contracts = build_default_artifact_contracts()
    workflow_templates = build_default_workflow_templates()
    providers = build_default_provider_contracts()
    provider_bindings = build_default_provider_bindings(
        workflow_modules,
        study_workflow_modules,
    )
    registry = PlatformRegistry()
    registry.rebuild_entry_index(
        catalog_families,
        workflow_modules=workflow_modules,
        study_workflow_modules=study_workflow_modules,
        workflow_gates=workflow_gates,
        study_workflow_gates=study_workflow_gates,
        artifact_contracts=artifact_contracts,
        workflow_templates=workflow_templates,
        providers=providers,
        provider_bindings=provider_bindings,
    )
    return registry


def normalize_molecular_method(value: Any, *, default: Optional[str] = None) -> Optional[str]:
    """Return a registered molecular method ID while preserving unknown input for validation."""

    if value is None or not str(value).strip():
        return default
    canonical = _DEFAULT_REGISTRY.canonical_capability_id(
        value, namespace='molecular.method'
    )
    if canonical is not None:
        return canonical
    return str(value).strip().lower().replace('-', '_').replace(' ', '_')


def _capability_ids(
    registry: PlatformRegistry,
    namespace: str,
    *,
    backend_only: bool = True,
) -> List[str]:
    return [
        item.id
        for item in registry.capabilities(
            namespace=namespace,
            backend_allowed=True if backend_only else None,
        )
    ]


def _option_ids(
    registry: PlatformRegistry,
    option_set_id: str,
    *,
    backend_only: bool = True,
) -> List[str]:
    return [
        item.id
        for item in registry.option_values(
            option_set_id,
            backend_allowed=True if backend_only else None,
        )
    ]


def registered_artifact_kinds(registry: PlatformRegistry = None) -> List[str]:
    capability_registry = registry or default_registry()
    artifact_kinds: List[str] = []
    for contract in capability_registry.artifact_contracts(status=CAPABILITY_EXECUTABLE):
        for artifact_kind in contract.artifact_kinds:
            normalized = str(artifact_kind or '').strip()
            if normalized and normalized not in artifact_kinds:
                artifact_kinds.append(normalized)
    return artifact_kinds


def artifact_payload_schema(
    artifact_kind: Any,
    registry: PlatformRegistry = None,
) -> Optional[str]:
    capability_registry = registry or default_registry()
    normalized_kind = str(artifact_kind or '').strip()
    if not normalized_kind:
        return None
    for contract in capability_registry.artifact_contracts(status=CAPABILITY_EXECUTABLE):
        if contract.payload_schemas.get(normalized_kind):
            return str(contract.payload_schemas[normalized_kind])
    return None


def artifact_kind_is_registered(
    artifact_kind: Any,
    registry: PlatformRegistry = None,
) -> bool:
    capability_registry = registry or default_registry()
    normalized_kind = str(artifact_kind or '').strip()
    if not normalized_kind:
        return False
    if normalized_kind in registered_artifact_kinds(capability_registry):
        return True
    for contract in capability_registry.artifact_contracts(status=CAPABILITY_EXECUTABLE):
        for pattern in contract.artifact_kind_patterns:
            if fnmatch.fnmatchcase(normalized_kind, str(pattern)):
                return True
    return False


def density_fitting_scopes(registry: PlatformRegistry = None) -> List[str]:
    capability = (registry or default_registry()).capability(
        'density_fitting', namespace='molecular.workflow_option'
    )
    return list(capability.metadata.get('apply_to', [])) if capability else []


def density_fitting_auxbasis_recommendations(registry: PlatformRegistry = None) -> Dict[str, List[str]]:
    capability_registry = registry or default_registry()
    capabilities = capability_registry.option_values(
        'options.molecular.auxbasis', backend_allowed=True
    )
    fallback_order = [
        item
        for item in ('def2-universal-jkfit', 'weigend+etb')
        if any(capability.id == item for capability in capabilities)
    ]
    basis_order = _option_ids(capability_registry, 'options.molecular.basis')
    for capability in capabilities:
        for basis in capability.constraints.get('recommended_for_basis', []):
            if basis not in basis_order:
                basis_order.append(basis)

    recommendations: Dict[str, List[str]] = {}
    for basis in basis_order:
        exact_matches = [
            capability.id
            for capability in capabilities
            if basis in capability.constraints.get('recommended_for_basis', [])
        ]
        ordered = []
        for auxbasis in exact_matches + fallback_order:
            if auxbasis not in ordered:
                ordered.append(auxbasis)
        recommendations[basis] = ordered
    return recommendations


def public_registry_payload(registry: PlatformRegistry = None) -> Dict[str, Any]:
    platform_registry = registry or default_registry()
    return platform_registry.registry_payload()


def result_analysis_output_contracts(registry: PlatformRegistry = None, *, system_type: str = '') -> Dict[str, Any]:
    capability_registry = registry or default_registry()
    contracts: Dict[str, Dict[str, Any]] = {}
    capabilities: List[CatalogItemContract] = []
    for namespace in (
        'molecular.method',
        'molecular.workflow_option',
        'periodic.method',
        'molecular.orbital_processing',
        'embedding.method',
    ):
        capabilities.extend(capability_registry.capabilities(
            namespace=namespace, backend_allowed=True
        ))
    for namespace in (
        'molecular.observable',
        'periodic.observable',
        'model_hamiltonian.observable',
        'postprocessing.metric',
    ):
        capabilities.extend(capability_registry.observables(
            namespace=namespace, backend_allowed=True
        ))

    for capability in capabilities:
        family = capability.domain.split('.')[0]
        if system_type and family in ('molecular', 'periodic', 'model_hamiltonian') and family != system_type:
            continue
        systems = capability.metadata.get('system_types') or []
        if system_type and systems and system_type not in systems:
            continue
        if isinstance(capability, ObservableContract):
            result_fields = list(dict.fromkeys(capability.source_fields + capability.result_fields))
            unit = capability.unit
            result_shape = capability.result_shape
        else:
            result_fields = list(capability.metadata.get('result_fields') or [])
            if capability.metadata.get('result_field'):
                result_fields.append(str(capability.metadata['result_field']))
            result_fields = list(dict.fromkeys(result_fields))
            unit = str(capability.metadata.get('unit') or '')
            result_shape = str(capability.metadata.get('result_shape') or '')
        for result_field in result_fields:
            contract = contracts.setdefault(result_field, {
                'source_capabilities': [],
                'compact_result': True,
                'summary_fields': [],
                'preview_tables': {},
                'artifact_kinds': [],
                'generated_output_rules': [],
                'comparison_summary_fields': [],
                'units': [],
                'result_shapes': [],
            })
            contract['source_capabilities'].append(capability.id)
            if capability.metadata.get('compact_result') is False:
                contract['compact_result'] = False
            if unit and unit not in contract['units']:
                contract['units'].append(unit)
            if result_shape and result_shape not in contract['result_shapes']:
                contract['result_shapes'].append(result_shape)
            for field_name in capability.metadata.get('summary_fields', []):
                if field_name not in contract['summary_fields']:
                    contract['summary_fields'].append(field_name)
            for field_name in capability.metadata.get('comparison_summary_fields', []):
                if field_name not in contract['comparison_summary_fields']:
                    contract['comparison_summary_fields'].append(field_name)
            for table_name, table_spec in capability.metadata.get('preview_tables', {}).items():
                existing = contract['preview_tables'].get(table_name, {})
                limit = table_spec.get('limit') if isinstance(table_spec, dict) else None
                if limit is not None:
                    existing_limit = existing.get('limit')
                    existing['limit'] = limit if existing_limit is None else min(existing_limit, limit)
                contract['preview_tables'][table_name] = existing
            for artifact_kind in capability.metadata.get('artifact_kinds', []):
                if artifact_kind not in contract['artifact_kinds']:
                    contract['artifact_kinds'].append(artifact_kind)
            output_rule = capability.metadata.get('generated_output_rule')
            if output_rule and output_rule not in contract['generated_output_rules']:
                contract['generated_output_rules'].append(output_rule)
    return contracts


def result_analysis_contract_prompt(registry: PlatformRegistry = None) -> str:
    contracts = result_analysis_output_contracts(registry)
    if not contracts:
        return ''
    lines = ['Registered output contracts:']
    for result_field, contract in sorted(contracts.items()):
        parts = ['- {0}'.format(result_field)]
        if contract.get('units'):
            parts.append('units={0}'.format(', '.join(contract['units'])))
        if contract.get('result_shapes'):
            parts.append('shapes={0}'.format(', '.join(contract['result_shapes'])))
        if contract.get('summary_fields'):
            parts.append('fields={0}'.format(', '.join(contract['summary_fields'])))
        if contract.get('comparison_summary_fields'):
            parts.append('comparison_fields={0}'.format(', '.join(contract['comparison_summary_fields'])))
        if contract.get('artifact_kinds'):
            parts.append('artifacts={0}'.format(', '.join(contract['artifact_kinds'])))
        if contract.get('generated_output_rules'):
            parts.append('rules={0}'.format(' '.join(contract['generated_output_rules'])))
        lines.append('; '.join(parts))
    return '\n'.join(lines)


def _markdown_cell(value: Any) -> str:
    return str(value).replace('|', '\\|').replace('\n', '<br>')


def capabilities_markdown(registry: PlatformRegistry = None) -> str:
    capability_registry = registry or default_registry()
    lines = [
        '# PySCF Agent Capability Registry',
        '',
        '<!-- Generated from pyscf_agent.registry. Do not edit by hand. -->',
        '',
        'This document summarizes executable, design-only, planned, and reference-only capabilities exposed by the runtime registry.',
        '',
    ]
    for family in capability_registry._family_order:  # pylint: disable=protected-access
        entries = capability_registry.entries_for(source_family=family)
        capabilities: List[CatalogItemContract] = []
        option_entry = next(
            (entry for entry in entries if entry.kind == ENTRY_KIND_OPTION_SET),
            None,
        )
        if option_entry is not None:
            capabilities = [
                capability_registry._option_catalog_projection(option_entry.contract, item)  # pylint: disable=protected-access
                for item in option_entry.contract.items
            ]
        elif family == 'artifact_contracts':
            capabilities = [
                capability_registry._artifact_catalog_projection(entry.contract)  # pylint: disable=protected-access
                for entry in entries
                if isinstance(entry.contract, ArtifactContract)
            ]
        else:
            capabilities = [
                capability_registry._catalog_compatibility_projection(entry)  # pylint: disable=protected-access
                for entry in entries
                if isinstance(entry.contract, CatalogItemContract)
            ]
        lines.extend([
            '## {0}'.format(family.replace('_', ' ').title()),
            '',
            '| id | status | planner | backend | notes |',
            '| --- | --- | --- | --- | --- |',
        ])
        for item in capabilities:
            notes = '; '.join(item.limitations) or item.description
            lines.append(
                '| {id} | {status} | {planner} | {backend} | {notes} |'.format(
                    id=_markdown_cell(item.id),
                    status=_markdown_cell(item.status),
                    planner='yes' if item.planner_allowed else 'no',
                    backend='yes' if item.backend_allowed else 'no',
                    notes=_markdown_cell(notes),
                )
            )
        lines.append('')
    return '\n'.join(lines).rstrip() + '\n'


def write_capabilities_markdown(path: str, registry: PlatformRegistry = None) -> str:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(capabilities_markdown(registry), encoding='utf-8')
    return str(output_path)


_DEFAULT_REGISTRY = default_registry()
DENSITY_FITTING_AUXBASIS_RECOMMENDATIONS = density_fitting_auxbasis_recommendations(_DEFAULT_REGISTRY)

SUPPORTED_TASK_TYPES = tuple(_capability_ids(_DEFAULT_REGISTRY, 'task_type'))
SUPPORTED_METHODS = tuple(_capability_ids(_DEFAULT_REGISTRY, 'molecular.method'))
SUPPORTED_JOBS = tuple(_capability_ids(_DEFAULT_REGISTRY, 'molecular.job'))
SUPPORTED_ANALYSIS = tuple(
    item.id
    for item in _DEFAULT_REGISTRY.observables(
        namespace='molecular.observable',
        requestable=True,
        backend_allowed=True,
    )
)
SUPPORTED_PERIODIC_METHODS = tuple(_capability_ids(_DEFAULT_REGISTRY, 'periodic.method'))
SUPPORTED_PERIODIC_JOBS = tuple(_capability_ids(_DEFAULT_REGISTRY, 'periodic.job'))
SUPPORTED_PERIODIC_ANALYSIS = tuple(
    item.id
    for item in _DEFAULT_REGISTRY.observables(
        namespace='periodic.observable',
        requestable=True,
        backend_allowed=True,
    )
)
SUPPORTED_PERIODIC_BASIS_SETS = tuple(_option_ids(_DEFAULT_REGISTRY, 'options.periodic.basis'))
DEFAULT_PERIODIC_BASIS_SET = 'gth-szv-molopt-sr'
SUPPORTED_PERIODIC_PSEUDOPOTENTIALS = tuple(_option_ids(_DEFAULT_REGISTRY, 'options.periodic.pseudopotential'))
SUPPORTED_PERIODIC_XC_FUNCTIONALS = tuple(_option_ids(_DEFAULT_REGISTRY, 'options.periodic.xc'))
SUPPORTED_PERIODIC_DENSITY_FITTING_METHODS = tuple(_option_ids(_DEFAULT_REGISTRY, 'options.periodic.density_fitting'))
SUPPORTED_PERIODIC_KPOINT_SCHEMES = tuple(_option_ids(_DEFAULT_REGISTRY, 'options.periodic.kpoint_scheme'))
SUPPORTED_PERIODIC_BAND_PATH_MODES = tuple(_option_ids(_DEFAULT_REGISTRY, 'options.periodic.band_path_mode'))
SUPPORTED_PERIODIC_SMEARING_METHODS = tuple(_option_ids(_DEFAULT_REGISTRY, 'options.periodic.smearing'))
SUPPORTED_PERIODIC_EXXDIV_OPTIONS = tuple(_option_ids(_DEFAULT_REGISTRY, 'options.periodic.exxdiv'))
SUPPORTED_MODEL_HAMILTONIANS = tuple(_capability_ids(_DEFAULT_REGISTRY, 'model_hamiltonian.model'))
SUPPORTED_MODEL_SOLVERS = tuple(_capability_ids(_DEFAULT_REGISTRY, 'model_hamiltonian.solver'))
COMMON_XC_FUNCTIONALS = tuple(_option_ids(_DEFAULT_REGISTRY, 'options.molecular.xc'))
SUPPORTED_AUXBASIS_SETS = tuple(_option_ids(_DEFAULT_REGISTRY, 'options.molecular.auxbasis'))
DEFAULT_ANALYSIS = ('energy', 'homo_lumo', 'dipole')
