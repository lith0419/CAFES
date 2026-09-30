from __future__ import annotations

import unittest

from pyscf_agent.application import ExecutionTargetRegistry
from pyscf_agent.executors import LocalExecutor


def _executor(marker: str) -> LocalExecutor:
    def run(request, **_kwargs):
        return {'task_report': {'marker': marker, 'request': request}}

    return LocalExecutor(request_runner=run)


class ExecutionTargetRegistryTests(unittest.TestCase):
    def test_resolves_default_and_explicit_targets(self):
        local = _executor('local')
        remote = _executor('remote')
        registry = ExecutionTargetRegistry(
            {'local': local, 'amarel': remote},
            default_target='amarel',
            labels={'local': 'Local', 'amarel': 'Amarel'},
        )

        self.assertIs(registry.resolve(), remote)
        self.assertIs(registry.resolve('local'), local)
        self.assertEqual(registry.public_dict()['default_target'], 'amarel')
        self.assertEqual(
            [item['label'] for item in registry.public_dict()['targets']],
            ['Local', 'Amarel'],
        )

    def test_rejects_unknown_target(self):
        registry = ExecutionTargetRegistry(
            {'local': _executor('local')},
            default_target='local',
        )

        with self.assertRaisesRegex(ValueError, 'Available targets: local'):
            registry.resolve('missing')


if __name__ == '__main__':
    unittest.main()
