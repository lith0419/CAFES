from __future__ import annotations

import importlib.util
import unittest
from pathlib import Path

import computational_study_agent
import pyscf_agent
import pyscf_agent.contracts as contracts
import pyscf_agent.registry as registry


REPO_ROOT = Path(__file__).resolve().parents[2]


class PublicApiBoundaryTests(unittest.TestCase):
    def test_calculation_root_exports_only_core_concepts(self):
        self.assertEqual(set(pyscf_agent.__all__), {
            'CalculationApplicationService',
            'PlatformRegistry',
            'TaskReport',
            'TaskSpec',
            'default_registry',
        })
        self.assertIs(pyscf_agent.TaskSpec, contracts.TaskSpec)
        self.assertIs(pyscf_agent.TaskReport, contracts.TaskReport)
        for internal_name in (
            'LocalExecutor',
            'append_log',
            'build_workflow',
            'repair_or_retry',
            'result_extractor',
            'runner',
            'spec_builder',
        ):
            self.assertFalse(hasattr(pyscf_agent, internal_name), internal_name)

    def test_registry_root_exports_contracts_not_catalog_implementation(self):
        self.assertEqual(set(registry.__all__), {
            'ArtifactContract',
            'CapabilityContract',
            'ObservableContract',
            'OptionSet',
            'OptionValueContract',
            'ParameterContract',
            'PlatformRegistry',
            'ProviderBinding',
            'ProviderContract',
            'RegistryEntry',
            'RegistryIssue',
            'WorkflowTemplateContract',
            'default_registry',
        })
        for implementation_name in (
            'SUPPORTED_METHODS',
            'DEFAULT_ANALYSIS',
            'ENTRY_KIND_MODULE',
            'UnifiedRegistryIndex',
            'build_default_provider_contracts',
            'public_registry_payload',
        ):
            self.assertFalse(hasattr(registry, implementation_name), implementation_name)

    def test_study_root_exports_only_service_and_study_contracts(self):
        self.assertEqual(set(computational_study_agent.__all__), {
            'StudyApplicationService',
            'StudyCase',
            'StudyPlan',
            'StudyReport',
            'StudySpec',
        })
        for operation_name in (
            'build_study_plan',
            'get_study_application_service',
            'run_postprocessing',
            'run_study',
            'suggest_plot_specs',
        ):
            self.assertFalse(hasattr(computational_study_agent, operation_name), operation_name)

    def test_legacy_backend_facade_has_been_removed(self):
        self.assertIsNone(importlib.util.find_spec('pyscf_agent.pyscf_agent_backend'))

    def test_product_entry_points_do_not_import_compatibility_backend(self):
        product_sources = (
            REPO_ROOT / 'pyscf_agent' / 'pyscf_agent_cli.py',
            REPO_ROOT / 'pyscf_agent' / 'pyscf_agent_web.py',
            REPO_ROOT / 'pyscf_agent' / 'benchmarks' / 'fcdmft_si_g0w0.py',
            REPO_ROOT / 'pyscf_agent' / 'benchmarks' / 'suite.py',
        )
        for source in product_sources:
            self.assertNotIn('pyscf_agent_backend', source.read_text(encoding='utf-8'), str(source))


if __name__ == '__main__':
    unittest.main()
