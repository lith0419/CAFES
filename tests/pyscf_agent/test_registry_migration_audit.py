from __future__ import annotations

import unittest
from pathlib import Path

from pyscf_agent.registry.audit import (
    REGISTRY_MIGRATION_AUDIT_SCHEMA,
    build_registry_migration_audit,
    registry_migration_markdown,
    scan_legacy_registry_consumers,
)
from pyscf_agent.registry.platform import default_registry


REPO_ROOT = Path(__file__).resolve().parents[2]


class RegistryMigrationAuditTests(unittest.TestCase):
    def setUp(self):
        self.registry = default_registry()

    def test_audit_is_read_only_and_captures_the_current_typed_baseline(self):
        before = self.registry.registry_payload()
        audit = build_registry_migration_audit(self.registry)
        after = self.registry.registry_payload()

        self.assertEqual(before, after)
        self.assertEqual(audit['schema'], REGISTRY_MIGRATION_AUDIT_SCHEMA)
        self.assertEqual(audit['registry_schema'], 'pyscf-agent.registry.v3')
        self.assertEqual(audit['validation_issues'], [])
        self.assertEqual(audit['entry_count'], 326)
        self.assertEqual(audit['entry_counts_by_kind'], {
            'artifact': 22,
            'binding': 43,
            'capability': 120,
            'gate': 12,
            'module': 43,
            'observable': 47,
            'option_set': 12,
            'parameter': 14,
            'provider': 5,
            'template': 8,
        })

    def test_audit_confirms_typed_migration_is_complete(self):
        audit = build_registry_migration_audit(self.registry)
        relationships = {
            (item['contract_id'], item['key'])
            for item in audit['legacy_relation_metadata']
        }
        self.assertEqual(relationships, set())

        observations = audit['contract_shape_observations']
        self.assertEqual(
            len(observations['artifact_entries_using_catalog_contract']),
            0,
        )
        self.assertEqual(observations['option_values_using_catalog_contract'], [])
        self.assertEqual(
            observations['contract_types_by_kind']['artifact'],
            {'ArtifactContract': 22},
        )
        self.assertEqual(len(observations['artifact_entries_using_artifact_contract']), 22)
        self.assertEqual(len(observations['option_values_using_option_value_contract']), 136)
        self.assertEqual(audit['legacy_public_payload_keys'], [])

    def test_source_scan_separates_runtime_and_test_consumers(self):
        consumers = scan_legacy_registry_consumers(REPO_ROOT)
        runtime = [item for item in consumers if item['scope'] == 'runtime']
        compatibility = [item for item in consumers if item['scope'] == 'registry_compatibility']
        self.assertEqual(runtime, [])
        self.assertEqual(compatibility, [])

    def test_markdown_is_a_report_not_a_generated_runtime_contract(self):
        audit = build_registry_migration_audit(self.registry, source_root=REPO_ROOT)
        markdown = registry_migration_markdown(audit)
        self.assertIn('# Registry Migration Audit', markdown)
        self.assertIn('Entries: 326', markdown)
        self.assertIn('Metadata relationships to type:', markdown)
        self.assertIn('Runtime Family Consumers', markdown)


if __name__ == '__main__':
    unittest.main()
