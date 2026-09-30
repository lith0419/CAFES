from __future__ import annotations

import hashlib
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from pyscf_agent.resources.web import version_asset_urls


class SharedWebAssetTests(unittest.TestCase):
    def test_unchanged_files_are_hashed_once_and_edits_invalidate_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'shared.js'
            asset.write_text('first', encoding='utf-8')
            html = '<script src="/assets/shared.js"></script>'
            reads = []
            original = Path.read_bytes

            def read(path):
                reads.append(path)
                return original(path)

            with patch('pyscf_agent.resources.web.resources.files', return_value=root), patch.object(Path, 'read_bytes', read):
                first = version_asset_urls(html)
                self.assertEqual(version_asset_urls(html), first)
                self.assertEqual(len(reads), 1)
                asset.write_text('other', encoding='utf-8')
                self.assertNotEqual(version_asset_urls(html), first)
                self.assertEqual(len(reads), 2)

    def test_asset_versions_follow_content_and_replace_old_manual_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            asset = root / 'shared.js'
            asset.write_text('first', encoding='utf-8')
            html = '<script src="/assets/shared.js?v=old"></script>'
            with patch('pyscf_agent.resources.web.resources.files', return_value=root):
                first = version_asset_urls(html)
                self.assertIn(hashlib.sha256(b'first').hexdigest()[:12], first)
                self.assertEqual(version_asset_urls(first), first)
                asset.write_text('second', encoding='utf-8')
                second = version_asset_urls(first)
            self.assertNotEqual(first, second)
            self.assertNotIn('?v=old', second)

    @unittest.skipUnless(shutil.which('node'), 'Node is required for UI helper tests')
    def test_shared_helpers_escape_session_labels_and_isolate_page_state(self):
        source = (
            Path(__file__).resolve().parents[2] / 'pyscf_agent/web_assets/shared.js'
        )
        script = r"""
const assert = require('node:assert/strict'), fs = require('node:fs'), vm = require('node:vm');
const context = {}; vm.createContext(context);
vm.runInContext(fs.readFileSync(process.argv[1], 'utf8'), context);
const ui = context.AgentUI, sessions = [{id: 'a"', status: 'running', label: '<script>'}];
assert(Object.isFrozen(ui));
assert.equal(ui.findSession(sessions, 'a"'), sessions[0]);
assert.equal(ui.findSession(sessions, 'missing'), null);
const html = ui.sessionOptions(sessions, 'a"', s => s.label, s => s);
assert(html.includes('value="a&quot;" selected'));
assert(html.includes('&lt;script&gt;'));
assert(!html.includes('<script>'));
assert.equal(sessions[0].label, '<script>');
assert.match(ui.makeRunId(), /^\d{8}-\d{6}-[0-9a-f]{8}$/);
assert.match(ui.makeSessionId('planner-task'), /^planner-task-/);
"""
        result = subprocess.run(
            ['node', '-e', script, str(source)], capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
