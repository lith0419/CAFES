from __future__ import annotations

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


REPO_ROOT = Path(__file__).resolve().parents[2]


@unittest.skipUnless(shutil.which('node'), 'Wiki curation requires Node.js')
class WikiCurationTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.wiki = self.root / 'agent_knowledge'
        (self.wiki / 'sources').mkdir(parents=True)
        shutil.copy2(REPO_ROOT / 'agent_knowledge/curate_wiki.js', self.wiki)
        self.structure = {
            'sections': [{'name': 'Methods', 'pages': [
                {'title': 'Runtime Guide', 'source': 'Runtime Guide', 'kind': 'rule'},
            ]}],
            'mergeInto': {}, 'seeAlso': {}, 'pageGuidance': {},
        }
        self.source = self.wiki / 'sources/guide.md'
        self.source.write_text('# Runtime Guide\n\nCurrent reviewed behavior.\n')
        self.export = self.root / 'computational_study_agent/knowledge/curated_wiki.json'

    def curate(self):
        (self.wiki / 'curated_structure.json').write_text(json.dumps(self.structure))
        subprocess.run(
            [shutil.which('node'), str(self.wiki / 'curate_wiki.js')],
            cwd=self.root, check=True, capture_output=True, text=True, timeout=30,
        )
        return json.loads(self.export.read_text())['pages'][0]

    def test_reviewed_source_wins_over_old_candidates_and_exports_on_every_build(self):
        candidates = self.wiki / '.llmwiki/candidates'
        candidates.mkdir(parents=True)
        (candidates / 'old.json').write_text(json.dumps({
            'id': 'old', 'title': 'Runtime Guide', 'body': 'Obsolete candidate behavior.',
            'sources': ['old.md'],
        }))
        concepts = self.wiki / 'wiki/concepts'
        concepts.mkdir(parents=True)
        (concepts / 'runtime-guide.md').write_text(
            '---\ntitle: Runtime Guide\n---\n\nObsolete compiled behavior.\n'
        )
        self.export.parent.mkdir(parents=True)
        self.export.write_text(json.dumps({'pages': [{
            'title': 'Runtime Guide', 'slug': 'runtime-guide',
            'body': 'Obsolete published behavior.',
        }]}))
        for text in ('Current reviewed behavior.', 'Updated reviewed behavior.'):
            self.source.write_text('# Runtime Guide\n\n' + text + '\n')
            page = self.curate()
            self.assertIn(text, page['body'])
            self.assertNotIn('Obsolete', page['body'])
            self.assertIn('`guide.md`', page['body'])
            self.assertNotIn('`old.md`', page['body'])

    def test_sources_can_build_without_generated_state(self):
        self.assertFalse(self.export.exists())
        self.assertIn('Current reviewed behavior.', self.curate()['body'])

    def test_explicit_editorial_guidance_still_overrides_source_body(self):
        self.structure['pageGuidance']['Runtime Guide'] = {
            'summary': 'Reviewed action summary.',
            'sections': [{'heading': 'Scope', 'items': ['Explicit compact guidance.']}],
        }
        page = self.curate()
        self.assertIn('Explicit compact guidance.', page['body'])
        self.assertNotIn('Current reviewed behavior.', page['body'])
        self.assertIn('`guide.md`', page['body'])


if __name__ == '__main__':
    unittest.main()
