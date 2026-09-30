from __future__ import annotations

import argparse
import ast
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from .contracts import (
    ArtifactContract,
    CatalogItemContract,
    OptionSet,
    OptionValueContract,
    RegistryEntry,
)


REGISTRY_MIGRATION_AUDIT_SCHEMA = 'pyscf-agent.registry-migration-audit.v1'

LEGACY_RELATION_METADATA_KEYS = (
    'module_id',
    'provider_id',
    'requires_solver',
)

LEGACY_FAMILY_QUERY_METHODS = (
    'item',
    'item_ids',
    'items',
    'resolve_id',
)

LEGACY_PUBLIC_PAYLOAD_KEYS = (
    'catalog',
    'workflow_modules',
    'study_workflow_modules',
    'workflow_gates',
    'study_workflow_gates',
)


def _contract_status(entry: RegistryEntry) -> str:
    return str(getattr(entry.contract, 'status', '') or '')


def _catalog_contracts(entry: RegistryEntry) -> Iterable[Tuple[str, Any]]:
    if isinstance(entry.contract, CatalogItemContract):
        yield entry.entry_id, entry.contract
    elif isinstance(entry.contract, OptionSet):
        for item in entry.contract.items:
            yield '{0}::{1}'.format(entry.entry_id, item.id), item


def _metadata_relationships(entries: Sequence[RegistryEntry]) -> List[Dict[str, Any]]:
    relationships: List[Dict[str, Any]] = []
    for entry in entries:
        for contract_id, contract in _catalog_contracts(entry):
            metadata = (
                contract.constraints
                if isinstance(contract, OptionValueContract)
                else contract.metadata
            )
            for key in LEGACY_RELATION_METADATA_KEYS:
                if key not in metadata:
                    continue
                relationships.append({
                    'entry_id': entry.entry_id,
                    'contract_id': contract_id,
                    'source_family': entry.source_family,
                    'key': key,
                    'value': metadata[key],
                })
    return sorted(
        relationships,
        key=lambda item: (item['source_family'], item['contract_id'], item['key']),
    )


def _contract_shape_observations(entries: Sequence[RegistryEntry]) -> Dict[str, Any]:
    catalog_artifacts = [
        entry.entry_id
        for entry in entries
        if entry.kind == 'artifact' and isinstance(entry.contract, CatalogItemContract)
    ]
    option_values = [
        '{0}::{1}'.format(entry.entry_id, item.id)
        for entry in entries
        if isinstance(entry.contract, OptionSet)
        for item in entry.contract.items
        if isinstance(item, CatalogItemContract)
    ]
    typed_artifacts = [
        entry.entry_id
        for entry in entries
        if entry.kind == 'artifact' and isinstance(entry.contract, ArtifactContract)
    ]
    typed_option_values = [
        '{0}::{1}'.format(entry.entry_id, item.id)
        for entry in entries
        if isinstance(entry.contract, OptionSet)
        for item in entry.contract.items
        if isinstance(item, OptionValueContract)
    ]
    contract_types: Dict[str, Counter] = defaultdict(Counter)
    for entry in entries:
        contract_types[entry.kind][type(entry.contract).__name__] += 1
    return {
        'artifact_entries_using_catalog_contract': sorted(catalog_artifacts),
        'option_values_using_catalog_contract': sorted(option_values),
        'artifact_entries_using_artifact_contract': sorted(typed_artifacts),
        'option_values_using_option_value_contract': sorted(typed_option_values),
        'contract_types_by_kind': {
            kind: dict(sorted(counts.items()))
            for kind, counts in sorted(contract_types.items())
        },
    }


def _call_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        return node.attr
    return ''


def _constant_string(node: ast.AST) -> str:
    if isinstance(node, ast.Constant) and isinstance(node.value, str):
        return node.value
    return ''


def _subscript_key(node: ast.Subscript) -> str:
    return _constant_string(node.slice)


def _is_test_path(path: Path) -> bool:
    return any(part == 'tests' for part in path.parts) or path.name.startswith('test_')


def _consumer_scope(path: Path) -> str:
    if _is_test_path(path):
        return 'test'
    if path.parts[:2] == ('pyscf_agent', 'registry'):
        return 'registry_compatibility'
    return 'runtime'


def scan_legacy_registry_consumers(source_root: Path) -> List[Dict[str, Any]]:
    """Locate Python call sites that still consume family or legacy payload APIs."""

    root = Path(source_root).resolve()
    consumers: List[Dict[str, Any]] = []
    ignored_parts = {'.git', '.venv', 'build', 'dist', 'runs', '__pycache__'}
    for path in sorted(root.rglob('*.py')):
        relative = path.relative_to(root)
        if ignored_parts.intersection(relative.parts):
            continue
        try:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(relative))
        except (OSError, SyntaxError, UnicodeDecodeError):
            continue
        scope = _consumer_scope(relative)
        for node in ast.walk(tree):
            if isinstance(node, ast.Call):
                name = _call_name(node.func)
                if name in LEGACY_FAMILY_QUERY_METHODS and node.args:
                    family = _constant_string(node.args[0])
                    if family:
                        consumers.append({
                            'path': relative.as_posix(),
                            'line': int(node.lineno),
                            'api': name,
                            'family': family,
                            'scope': scope,
                            'test_only': scope == 'test',
                        })
                elif name in ('catalog_families', 'public_catalog_payload'):
                    consumers.append({
                        'path': relative.as_posix(),
                        'line': int(node.lineno),
                        'api': name,
                        'family': '',
                        'scope': scope,
                        'test_only': scope == 'test',
                    })
            elif isinstance(node, ast.Name) and node.id == 'public_catalog_payload':
                consumers.append({
                    'path': relative.as_posix(),
                    'line': int(node.lineno),
                    'api': node.id,
                    'family': '',
                    'scope': scope,
                    'test_only': scope == 'test',
                })
            elif isinstance(node, ast.Subscript) and _subscript_key(node) == 'catalog':
                consumers.append({
                    'path': relative.as_posix(),
                    'line': int(node.lineno),
                    'api': 'catalog_payload_access',
                    'family': '',
                    'scope': scope,
                    'test_only': scope == 'test',
                })
    unique = {
        (item['path'], item['line'], item['api'], item['family']): item
        for item in consumers
    }
    return [unique[key] for key in sorted(unique)]


def build_registry_migration_audit(
    registry: Optional[Any] = None,
    *,
    source_root: Optional[Path] = None,
) -> Dict[str, Any]:
    """Return a deterministic, read-only inventory for staged registry migration."""

    if registry is None:
        from .platform import default_registry  # pylint: disable=import-outside-toplevel

        registry = default_registry()
    entries = tuple(registry.entries_for())
    kind_counts = Counter(entry.kind for entry in entries)
    family_entries: Dict[str, List[str]] = defaultdict(list)
    for entry in entries:
        family_entries[entry.source_family].append(entry.entry_id)
    full_payload = registry.as_dict()
    legacy_payload_keys = [
        key for key in LEGACY_PUBLIC_PAYLOAD_KEYS
        if key in full_payload
    ]
    consumers = (
        scan_legacy_registry_consumers(Path(source_root))
        if source_root is not None
        else []
    )
    return {
        'schema': REGISTRY_MIGRATION_AUDIT_SCHEMA,
        'registry_schema': registry.registry_payload().get('schema'),
        'validation_issues': [issue.to_dict() for issue in registry.registry_issues()],
        'entry_count': len(entries),
        'entry_counts_by_kind': dict(sorted(kind_counts.items())),
        'entries': [
            {
                'id': entry.entry_id,
                'kind': entry.kind,
                'namespace': entry.namespace,
                'source_family': entry.source_family,
                'status': _contract_status(entry),
            }
            for entry in entries
        ],
        'source_families': {
            family: sorted(entry_ids)
            for family, entry_ids in sorted(family_entries.items())
        },
        'legacy_relation_metadata': _metadata_relationships(entries),
        'contract_shape_observations': _contract_shape_observations(entries),
        'legacy_public_payload_keys': legacy_payload_keys,
        'legacy_family_consumers': consumers,
    }


def registry_migration_markdown(audit: Mapping[str, Any]) -> str:
    """Format the audit as a concise migration checklist without changing state."""

    lines = [
        '# Registry Migration Audit',
        '',
        'Schema: `{0}`'.format(audit.get('schema', '')),
        '',
        '## Current Baseline',
        '',
        '- Entries: {0}'.format(audit.get('entry_count', 0)),
        '- Validation issues: {0}'.format(len(audit.get('validation_issues') or [])),
    ]
    for kind, count in sorted((audit.get('entry_counts_by_kind') or {}).items()):
        lines.append('- `{0}`: {1}'.format(kind, count))
    relationships = list(audit.get('legacy_relation_metadata') or [])
    observations = dict(audit.get('contract_shape_observations') or {})
    consumers = list(audit.get('legacy_family_consumers') or [])
    runtime_consumers = [item for item in consumers if item.get('scope') == 'runtime']
    compatibility_consumers = [
        item for item in consumers if item.get('scope') == 'registry_compatibility'
    ]
    test_consumers = [item for item in consumers if item.get('scope') == 'test']
    lines.extend([
        '',
        '## Migration Inventory',
        '',
        '- Metadata relationships to type: {0}'.format(len(relationships)),
        '- Artifact entries using `CatalogItemContract`: {0}'.format(
            len(observations.get('artifact_entries_using_catalog_contract') or [])
        ),
        '- Artifact entries using `ArtifactContract`: {0}'.format(
            len(observations.get('artifact_entries_using_artifact_contract') or [])
        ),
        '- Option values using `CatalogItemContract`: {0}'.format(
            len(observations.get('option_values_using_catalog_contract') or [])
        ),
        '- Option values using `OptionValueContract`: {0}'.format(
            len(observations.get('option_values_using_option_value_contract') or [])
        ),
        '- Runtime family consumers: {0}'.format(len(runtime_consumers)),
        '- Registry compatibility call sites: {0}'.format(len(compatibility_consumers)),
        '- Test-only family consumers: {0}'.format(len(test_consumers)),
        '- Legacy public payload keys: {0}'.format(
            ', '.join(audit.get('legacy_public_payload_keys') or []) or 'none'
        ),
        '',
        '## Runtime Family Consumers',
        '',
    ])
    if runtime_consumers:
        for item in runtime_consumers:
            suffix = ' `{0}`'.format(item['family']) if item.get('family') else ''
            lines.append(
                '- `{0}:{1}` `{2}`{3}'.format(
                    item['path'], item['line'], item['api'], suffix
                )
            )
    else:
        lines.append('- None')
    return '\n'.join(lines).rstrip() + '\n'


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description='Inspect the Registry migration baseline.')
    parser.add_argument('--source-root', default='', help='Repository root for legacy consumer scanning.')
    parser.add_argument('--format', choices=('json', 'markdown'), default='markdown')
    args = parser.parse_args(argv)
    audit = build_registry_migration_audit(
        source_root=Path(args.source_root) if args.source_root else None,
    )
    if args.format == 'json':
        print(json.dumps(audit, indent=2, sort_keys=True))
    else:
        print(registry_migration_markdown(audit), end='')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())


__all__ = [
    'LEGACY_FAMILY_QUERY_METHODS',
    'LEGACY_PUBLIC_PAYLOAD_KEYS',
    'LEGACY_RELATION_METADATA_KEYS',
    'REGISTRY_MIGRATION_AUDIT_SCHEMA',
    'build_registry_migration_audit',
    'registry_migration_markdown',
    'scan_legacy_registry_consumers',
]
