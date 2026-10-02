from __future__ import annotations

import hashlib
import os
import subprocess
import tarfile
import tempfile
import unittest
from pathlib import Path


class DistributionScriptTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.repo_root = Path(__file__).resolve().parents[2]

    def test_shell_scripts_have_valid_syntax(self):
        for relative_path in (
            'clean.sh',
            'distribution/package.sh',
            'distribution/offline/install.sh',
            'distribution/offline/generate-locks.sh',
            'distribution/offline/package.sh',
        ):
            path = self.repo_root / relative_path
            result = subprocess.run(
                ('bash', '-n', str(path)),
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(os.access(path, os.X_OK))

    def test_clean_script_separates_generated_and_user_state(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'clean.sh'
            script.write_bytes((self.repo_root / 'clean.sh').read_bytes())
            script.chmod(0o755)

            for path in (
                root / 'build',
                root / 'dist',
                root / 'package.egg-info',
                root / 'module' / '__pycache__',
                root / 'runs',
                root / '.venv',
                root / 'agent_knowledge' / 'node_modules',
                root / 'agent_knowledge' / 'wiki',
                root / 'block2-scratch',
                root / '.pyscf-agent',
            ):
                path.mkdir(parents=True, exist_ok=True)
            (root / '.pyscf-agent' / 'remote.ini').write_text(
                '[remote]\n', encoding='utf-8'
            )

            preview = subprocess.run(
                (str(script), '--check'), capture_output=True, text=True, check=False,
            )
            self.assertEqual(preview.returncode, 1, preview.stderr)
            self.assertIn('  - build', preview.stdout)
            self.assertNotIn('  - dist', preview.stdout)
            self.assertTrue((root / 'build').is_dir())

            result = subprocess.run(
                (str(script), '--yes'),
                cwd=str(root),
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / 'build').exists())
            self.assertTrue((root / 'dist').is_dir())
            self.assertTrue((root / 'agent_knowledge' / 'wiki').is_dir())
            self.assertTrue((root / 'block2-scratch').is_dir())
            self.assertFalse((root / 'package.egg-info').exists())
            self.assertFalse((root / 'module' / '__pycache__').exists())
            self.assertTrue((root / 'runs').is_dir())
            self.assertTrue((root / '.venv').is_dir())
            self.assertTrue((root / 'agent_knowledge' / 'node_modules').is_dir())
            self.assertTrue((root / '.pyscf-agent' / 'remote.ini').is_file())

            result = subprocess.run(
                (str(script), '--yes', '--all', '--runs', '--dist', '--wiki', '--scratch'),
                cwd=str(root),
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertFalse((root / 'runs').exists())
            self.assertFalse((root / 'dist').exists())
            self.assertFalse((root / 'agent_knowledge' / 'wiki').exists())
            self.assertFalse((root / 'block2-scratch').exists())
            self.assertFalse((root / '.venv').exists())
            self.assertFalse((root / 'agent_knowledge' / 'node_modules').exists())
            self.assertTrue((root / '.pyscf-agent' / 'remote.ini').is_file())

    def test_clean_refuses_to_delete_git_backups_before_any_mutation(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'clean.sh'
            script.write_bytes((self.repo_root / 'clean.sh').read_bytes())
            backup = root / 'runs' / 'history' / 'before.bundle'
            backup.parent.mkdir(parents=True)
            backup.write_bytes(b'backup evidence')
            (root / 'build').mkdir()
            result = subprocess.run(
                ('bash', str(script), '--yes', '--runs'),
                capture_output=True, text=True, check=False,
            )
            self.assertEqual(result.returncode, 2, result.stderr)
            self.assertIn('Git bundle backup exists', result.stderr)
            self.assertEqual(backup.read_bytes(), b'backup evidence')
            self.assertTrue((root / 'build').is_dir())

    def test_release_metadata_has_one_core_langgraph_contract(self):
        metadata = (self.repo_root / 'pyproject.toml').read_text(encoding='utf-8')

        self.assertRegex(metadata, r'(?m)^version = "[0-9]+\.[0-9]+\.[0-9]+"$')
        self.assertIn('requires-python = ">=3.10"', metadata)
        self.assertIn('"pyscf>=2.13,<2.14"', metadata)
        self.assertEqual(metadata.count('"langgraph>=0.6.11,<1.0"'), 1)
        self.assertIn('embedding-runtime = [', metadata)
        self.assertIn('dmft-runtime = [', metadata)
        self.assertIn(
            'pyscf-agent-configure = "pyscf_agent.configure:main"',
            metadata,
        )

    def test_supported_python_offline_dependencies_are_fully_pinned(self):
        offline_root = self.repo_root / 'distribution' / 'offline'
        for compact in ('310', '311', '312', '313', '314'):
            with self.subTest(python=compact):
                lock = (
                    offline_root / f'requirements-linux-x86_64-py{compact}.lock'
                ).read_text(encoding='utf-8')
                requirements = [
                    line.strip()
                    for line in lock.splitlines()
                    if line.strip() and not line.lstrip().startswith('#')
                ]
                self.assertTrue(
                    any(line.startswith('pyscf==2.13.') for line in requirements)
                )
                self.assertTrue(all(line.count('==') == 1 for line in requirements))
                self.assertEqual(
                    any(line.startswith('exceptiongroup==') for line in requirements),
                    compact == '310',
                )

        py314 = (
            offline_root / 'requirements-linux-x86_64-py314.lock'
        ).read_text(encoding='utf-8')
        self.assertIn('glibc 2.28+', py314)
        self.assertFalse(
            (offline_root / 'requirements-linux-x86_64-py315.lock').exists()
        )

    def test_offline_installer_verifies_hashes_before_installing(self):
        installer = (
            self.repo_root / 'distribution' / 'offline' / 'install.sh'
        ).read_text(encoding='utf-8')

        verification = installer.index('verifying wheelhouse SHA-256 checksums')
        installation = installer.index('installing bundled dependencies')
        self.assertLess(verification, installation)
        self.assertIn('sha256sum --check', installer)
        self.assertIn('BUNDLE_TARGET.env', installer)
        self.assertIn('TARGET_PYTHON_VERSION', installer)
        self.assertIn('TARGET_MIN_GLIBC', installer)
        self.assertGreaterEqual(installer.count('--no-deps'), 2)
        self.assertIn('--force-reinstall', installer)
        self.assertIn('PROJECT_WHEELS=', installer)
        self.assertIn('sysconfig.get_path("scripts")', installer)
        self.assertIn('missing console command', installer)

    def test_offline_bundle_uses_standard_build_frontend(self):
        packager = (
            self.repo_root / 'distribution' / 'offline' / 'package.sh'
        ).read_text(encoding='utf-8')

        self.assertIn("-c 'import build.__main__'", packager)
        self.assertIn('cd "$SCRIPT_DIR"', packager)
        self.assertIn('cd "$TEMP_ROOT"', packager)
        self.assertIn('"$PYTHON_COMMAND" -m build', packager)
        self.assertIn('--python-version', packager)
        self.assertIn('requirements-linux-x86_64-py${TARGET_PYTHON}.lock', packager)
        self.assertIn('BUNDLE_TARGET.env', packager)
        self.assertIn('-m pip download', packager)
        self.assertIn('--no-deps', packager)
        self.assertNotIn('-m pip wheel', packager)

    def test_offline_bundle_rejects_unreleased_python_target(self):
        result = subprocess.run(
            (
                str(self.repo_root / 'distribution' / 'offline' / 'package.sh'),
                '--python-version',
                '3.15',
            ),
            cwd=str(self.repo_root),
            capture_output=True,
            check=False,
            text=True,
        )

        self.assertEqual(result.returncode, 2)
        self.assertIn('between 3.10 and 3.14', result.stderr)

    def test_source_snapshot_excludes_local_numerical_packages(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            script = root / 'distribution' / 'package.sh'
            script.parent.mkdir()
            script.write_bytes((self.repo_root / 'distribution' / 'package.sh').read_bytes())
            (root / 'pyproject.toml').write_text('version = "0.0.0"\n')
            excluded = (
                'datasets/QM9_CCSD_upload/dataset.h5',
                'datasets/QM9_CCSD_upload_5shards/part-001.h5',
                'datasets/qm9-ccsd500-v2-dict/dataset.h5',
                'docs/si/honeycomb-current/support/server-snapshot.json',
                'docs/si/honeycomb-current/table.csv',
                'manuscript/correlation-policy-si.tex',
                'manuscript/lutein-si/lutein-si.tex',
                'docs/si/another-study/table.csv',
            )
            retained = ('datasets/reference/README.md', 'docs/reference/table.csv')
            for name in (*excluded, *retained):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text('fixture\n')
            result = subprocess.run(
                ('bash', str(script), '--name', 'test-snapshot'),
                cwd=root, capture_output=True, text=True, timeout=30,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            with tarfile.open(root / 'dist' / 'test-snapshot.tar.gz', 'r:gz') as bundle:
                names = set(bundle.getnames())
            for name in excluded:
                self.assertNotIn('test-snapshot/' + name, names)
                self.assertTrue((root / name).exists())
            for name in retained:
                self.assertIn('test-snapshot/' + name, names)

    def test_distribution_archive_is_clean_and_self_configuring(self):
        with tempfile.TemporaryDirectory() as directory:
            result = subprocess.run(
                (
                    str(self.repo_root / 'distribution' / 'package.sh'),
                    '--output-dir',
                    directory,
                    '--name',
                    'pyscf-agent-test',
                ),
                cwd=str(self.repo_root),
                capture_output=True,
                check=False,
                text=True,
            )
            self.assertEqual(result.returncode, 0, result.stderr)
            archive = Path(directory) / 'pyscf-agent-test.tar.gz'
            checksum = archive.with_suffix(archive.suffix + '.sha256')
            self.assertTrue(archive.is_file())
            self.assertTrue(checksum.is_file())
            with tarfile.open(archive, mode='r:gz') as bundle:
                names = bundle.getnames()

            required = {
                'pyscf-agent-test/README.md',
                'pyscf-agent-test/pyscf_agent/configure.py',
                'pyscf-agent-test/config/environments/conda.yml',
                'pyscf-agent-test/pyscf_agent/resources/templates/llm.env',
                'pyscf-agent-test/pyscf_agent/resources/templates/remote.ini',
                'pyscf-agent-test/pyscf_agent/resources/templates/server-slurm.ini',
                'pyscf-agent-test/distribution/package.sh',
                'pyscf-agent-test/distribution/offline/install.sh',
                'pyscf-agent-test/distribution/offline/package.sh',
                'pyscf-agent-test/distribution/offline/generate-locks.sh',
                'pyscf-agent-test/distribution/offline/requirements-linux-x86_64.in',
                'pyscf-agent-test/computational_study_agent/knowledge/curated_wiki.json',
                'pyscf-agent-test/pyscf_agent/benchmarks/data/n2_contextual_vqe_comparison.tsv',
                'pyscf-agent-test/DISTRIBUTION_MANIFEST.txt',
            }
            required.update({
                f'pyscf-agent-test/distribution/offline/requirements-linux-x86_64-py{compact}.lock'
                for compact in ('310', '311', '312', '313', '314')
            })
            self.assertTrue(required.issubset(names))
            self.assertFalse(any('/runs/' in name for name in names))
            self.assertFalse(any('/.git/' in name for name in names))
            self.assertFalse(any('/.pyscf-agent/' in name for name in names))
            self.assertFalse(any('/__pycache__/' in name for name in names))
            self.assertFalse(any('/._' in name for name in names))
            self.assertNotIn('pyscf-agent-test/llm.env', names)

            expected = checksum.read_text(encoding='utf-8').split()[0]
            actual = hashlib.sha256(archive.read_bytes()).hexdigest()
            self.assertEqual(actual, expected)


if __name__ == '__main__':
    unittest.main()
