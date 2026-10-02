"""Local architecture checks; independent of CI and scientific providers."""

from __future__ import annotations

import ast
import importlib.util
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


class ArchitectureBoundaryTests(unittest.TestCase):
    def test_task_core_does_not_import_study_or_inbound_adapters(self):
        # Composition roots may know both applications. Kernels, contracts,
        # storage, providers and registry must not import executable adapters.
        core = [
            'backend',
            'artifacts',
            'providers',
            'registry',
            'workflow_modules',
            'workflow_gates',
            'embedding',
            'executors',
            'application',
        ]
        forbidden = (
            'computational_study_agent',
            'pyscf_agent.web',
            'pyscf_agent.mcp_server',
            'pyscf_agent.cli',
        )
        # These are composition roots, not core modules.
        # Moving them to apps is a separate migration.
        adapters = {
            '__main__.py', 'cli.py', 'configure.py', 'verification.py', 'workbench.py',
        }
        paths = [path for area in core
                 for path in (ROOT / 'pyscf_agent' / area).rglob('*.py')]
        paths.extend(path for path in (ROOT / 'pyscf_agent').glob('*.py')
                     if path.name not in adapters)
        violations = []
        for path in paths:
            module = '.'.join(path.relative_to(ROOT).with_suffix('').parts)
            package = module.rsplit('.', 1)[0]
            for node in ast.walk(ast.parse(path.read_text())):
                targets = []
                if isinstance(node, ast.Import):
                    targets = [alias.name for alias in node.names]
                elif isinstance(node, ast.ImportFrom):
                    name = '.' * node.level + (node.module or '')
                    target = (
                        importlib.util.resolve_name(name, package)
                        if node.level
                        else name
                    )
                    targets = [target] + [
                        target + '.' + alias.name for alias in node.names
                    ]
                for target in targets:
                    if any(
                        target == prefix or target.startswith(prefix + '.')
                        for prefix in forbidden
                    ):
                        violations.append(
                            f'{path.relative_to(ROOT)}:{node.lineno}: {target}'
                        )
        self.assertEqual(violations, [])
