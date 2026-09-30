from __future__ import annotations

import runpy
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pyscf_agent.resources.text import read_text_resource


class TextResourceTests(unittest.TestCase):
    def test_templates_and_preview_work_without_package_resource_lookup(self):
        # Exercise each caller's real fallback path, including both templates
        # that used to look in the nonexistent web_assets/web_assets directory.
        root = Path(__file__).resolve().parents[2]
        modules = (
            ('pyscf_agent.web_assets.template', {
                'HTML_PAGE': root / 'pyscf_agent/web_assets/assistant_index.html',
            }),
            ('computational_study_agent.web_assets.template', {
                'HTML_PAGE': root / 'computational_study_agent/web_assets/planner_index.html',
            }),
            ('pyscf_agent.molecular_structure_preview_ui', {
                'MOLECULAR_STRUCTURE_PREVIEW_CSS': root / 'pyscf_agent/web_assets/molecular-preview.css',
                'MOLECULAR_STRUCTURE_PREVIEW_JS': root / 'pyscf_agent/web_assets/molecular-preview.js',
            }),
        )
        for error in (FileNotFoundError, ModuleNotFoundError):
            for module, exports in modules:
                with self.subTest(module=module, error=error):
                    with patch('pyscf_agent.resources.text.resources.files', side_effect=error):
                        source_module = root.joinpath(*module.split('.')).with_suffix('.py')
                        namespace = runpy.run_path(str(source_module), run_name=module)
                    for name, source in exports.items():
                        self.assertEqual(namespace[name], source.read_text(encoding='utf-8'))

    def test_package_resources_take_precedence_over_source_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory)
            source.joinpath('assistant_index.html').write_text('fallback', encoding='utf-8')
            content = read_text_resource(
                'pyscf_agent.web_assets', 'assistant_index.html', source_dir=source,
            )
        self.assertIn('<html', content.lower())
        self.assertNotEqual(content, 'fallback')

    def test_unexpected_resource_errors_are_not_hidden_by_fallback(self):
        with patch('pyscf_agent.resources.text.resources.files', side_effect=PermissionError):
            with self.assertRaises(PermissionError):
                read_text_resource('pyscf_agent.web_assets', 'assistant_index.html', source_dir=Path('.'))


if __name__ == '__main__':
    unittest.main()
