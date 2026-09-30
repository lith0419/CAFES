from __future__ import annotations

import unittest

import pyscf_agent.registry as registry_api
from pyscf_agent.registry.contracts import (
    ENTRY_KIND_ARTIFACT,
    ENTRY_KIND_BINDING,
    ENTRY_KIND_CAPABILITY,
    ENTRY_KIND_MODULE,
    ENTRY_KIND_OPTION_SET,
    ENTRY_KIND_PROVIDER,
    ArtifactContract,
    CapabilityContract,
    CatalogItemContract,
    OptionSet,
    OptionValueContract,
    ProviderBinding,
    ProviderContract,
    RegistryEntry,
    WorkflowTemplateContract,
)
from pyscf_agent.registry.index import UnifiedRegistryIndex
from pyscf_agent.registry.platform import default_registry
from pyscf_agent.workflow_modules.contracts import ModuleContract


class RegistryContractFoundationTests(unittest.TestCase):
    @staticmethod
    def _register_provider(index, provider_id):
        index.register(RegistryEntry(
            entry_id=provider_id,
            kind=ENTRY_KIND_PROVIDER,
            namespace='provider',
            local_id=provider_id.rsplit('.', 1)[-1],
            contract=ProviderContract(provider_id=provider_id, label=provider_id),
        ))

    @staticmethod
    def _register_module(index, module_id, *, capabilities=(), requires=(), provider='provider.internal'):
        entry_id = 'task.module.' + module_id
        index.register(RegistryEntry(
            entry_id=entry_id,
            kind=ENTRY_KIND_MODULE,
            namespace='task.module',
            local_id=module_id,
            contract=ModuleContract(
                module_id=module_id,
                version='1',
                permitted_stages=('task.prepare',),
                default_stage='task.prepare',
                capability_ids=tuple(capabilities),
                requires_modules=tuple(requires),
            ),
        ))
        index.register(RegistryEntry(
            entry_id='task.binding.' + module_id,
            kind=ENTRY_KIND_BINDING,
            namespace='task.binding',
            local_id=module_id,
            contract=ProviderBinding(
                binding_id='task.binding.' + module_id,
                module_id=entry_id,
                provider_id=provider,
                runtime_id='runtime.' + module_id,
            ),
        ))

    def test_target_contracts_are_available_from_the_registry_api(self):
        self.assertIs(registry_api.ArtifactContract, ArtifactContract)
        self.assertIs(registry_api.OptionValueContract, OptionValueContract)

    def test_option_value_contract_supports_canonical_ids_and_local_aliases(self):
        options = OptionSet(
            option_set_id='option.test.values',
            default='canonical',
            items=(OptionValueContract(
                id='canonical',
                label='Canonical value',
                aliases=('legacy',),
                constraints={'system_types': ('molecular',)},
            ),),
        )

        self.assertEqual(options.canonical_id('canonical'), 'canonical')
        self.assertEqual(options.canonical_id('LEGACY'), 'canonical')
        self.assertEqual(options.values, ('canonical',))
        self.assertEqual(
            options.to_dict()['items'][0]['constraints'],
            {'system_types': ('molecular',)},
        )

    def test_artifact_contract_owns_payload_and_reference_fields(self):
        contract = ArtifactContract(
            id='study',
            label='Study artifacts',
            artifact_kinds=('study-report',),
            artifact_kind_patterns=('observable-*',),
            payload_schemas={'study-report': 'pyscf-agent.study-report.v1'},
            required_reference_fields=('kind', 'path', 'size_bytes'),
        )

        payload = contract.to_dict()
        self.assertEqual(payload['artifact_kinds'], ('study-report',))
        self.assertEqual(
            payload['payload_schemas']['study-report'],
            'pyscf-agent.study-report.v1',
        )
        self.assertEqual(payload['required_reference_fields'], ('kind', 'path', 'size_bytes'))

    def test_index_reports_contract_kind_mismatches(self):
        index = UnifiedRegistryIndex()
        index.register(RegistryEntry(
            entry_id='provider.invalid',
            kind=ENTRY_KIND_PROVIDER,
            namespace='provider',
            local_id='invalid',
            contract=OptionSet(option_set_id='option.invalid'),
        ))

        issues = index.validate()
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, 'registry_contract_kind_mismatch')
        self.assertEqual(issues[0].entry_id, 'provider.invalid')

    def test_artifact_entries_require_the_typed_contract_after_migration(self):
        index = UnifiedRegistryIndex()
        index.register(RegistryEntry(
            entry_id='artifact.contract.target',
            kind=ENTRY_KIND_ARTIFACT,
            namespace='artifact.contract',
            local_id='target',
            contract=ArtifactContract(id='target', label='Target artifact'),
        ))
        index.register(RegistryEntry(
            entry_id='artifact.contract.transitional',
            kind=ENTRY_KIND_ARTIFACT,
            namespace='artifact.contract',
            local_id='transitional',
            contract=CatalogItemContract(
                id='transitional',
                label='Transitional artifact',
                domain='artifact.contract',
                status='executable',
            ),
        ))
        index.register(RegistryEntry(
            entry_id='option.test.values',
            kind=ENTRY_KIND_OPTION_SET,
            namespace='option.test',
            local_id='values',
            contract=OptionSet(
                option_set_id='option.test.values',
                items=(OptionValueContract(id='value'),),
            ),
        ))

        issues = index.validate()
        self.assertEqual(len(issues), 1)
        self.assertEqual(issues[0].code, 'registry_contract_kind_mismatch')
        self.assertEqual(issues[0].entry_id, 'artifact.contract.transitional')

    def test_default_registry_remains_valid_during_contract_migration(self):
        self.assertEqual(default_registry().registry_issues(), ())

    def test_default_registry_assigns_every_executable_capability(self):
        registry = default_registry()
        hierarchy = registry.registry_payload()['hierarchy']

        for capability_id in hierarchy['unassigned_capability_ids']:
            capability = registry.entry(capability_id, kind=ENTRY_KIND_CAPABILITY)
            self.assertIn(capability.contract.status, ('design_only', 'planned'))

    def test_index_rejects_executable_capability_without_owner_module(self):
        index = UnifiedRegistryIndex()
        index.register(RegistryEntry(
            entry_id='capability.unowned.execute',
            kind=ENTRY_KIND_CAPABILITY,
            namespace='capability.unowned',
            local_id='execute',
            contract=CapabilityContract(
                id='execute',
                label='Execute',
                domain='capability.unowned',
                status='executable',
            ),
        ))

        issue_codes = {issue.code for issue in index.validate()}
        self.assertIn('missing_capability_owner', issue_codes)

    def test_index_rejects_provider_mismatch_between_capability_and_owner(self):
        index = UnifiedRegistryIndex()
        self._register_provider(index, 'provider.internal')
        self._register_provider(index, 'provider.external')
        index.register(RegistryEntry(
            entry_id='capability.external.solve',
            kind=ENTRY_KIND_CAPABILITY,
            namespace='capability.external',
            local_id='solve',
            contract=CapabilityContract(
                id='solve',
                label='Solve',
                domain='capability.external',
                status='executable',
                required_provider_ids=('provider.external',),
            ),
        ))
        self._register_module(
            index,
            'solver.example',
            capabilities=('capability.external.solve',),
            provider='provider.internal',
        )

        issue_codes = {issue.code for issue in index.validate()}
        self.assertIn('capability_provider_mismatch', issue_codes)

    def test_index_rejects_workflow_that_omits_a_module_dependency(self):
        index = UnifiedRegistryIndex()
        self._register_provider(index, 'provider.internal')
        self._register_module(index, 'core.dependency')
        self._register_module(index, 'core.consumer', requires=('core.dependency',))
        index.register(RegistryEntry(
            entry_id='task.template.incomplete',
            kind='template',
            namespace='task.template',
            local_id='incomplete',
            contract=WorkflowTemplateContract(
                template_id='task.template.incomplete',
                label='Incomplete workflow',
                scope='task',
                required_module_ids=('task.module.core.consumer',),
            ),
        ))

        issue_codes = {issue.code for issue in index.validate()}
        self.assertIn('workflow_module_dependency_omitted', issue_codes)

    def test_default_template_returns_provider_bound_runtime_contracts(self):
        registry = default_registry()
        modules = registry.modules_for_template('task.template.default')

        self.assertTrue(modules)
        for module in modules:
            entry_id = 'task.module.' + module.module_id
            stored_module = registry.entry(entry_id, kind=ENTRY_KIND_MODULE)
            binding = registry.provider_binding(entry_id)
            self.assertIsNotNone(binding)
            self.assertFalse(hasattr(stored_module.contract, 'runtime_id'))
            self.assertEqual(module.provider_id, binding.provider_id)
            self.assertEqual(module.runtime_id, binding.runtime_id)


if __name__ == '__main__':
    unittest.main()
