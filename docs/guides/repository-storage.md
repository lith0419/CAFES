# Repository storage

Storage layout after the 2026-10-01 cleanup:

| Location | Contents and handling |
| --- | --- |
| `pyscf_agent/`, `computational_study_agent/`, `model_hamiltonian_ui/` | Runtime source |
| `tests/`, `tools/`, `config/`, `distribution/`, `plugins/` | Tests, maintained tools and installation definitions |
| `docs/`, `agent_knowledge/` | Documentation, wiki sources and generated pages |
| `examples/studies/` | Four portable Planner result snapshots; AFM and CDW are separate |
| `reports/` | Research evidence and verification receipts; retained at existing paths |
| `runs/` | Local calculation records and installed example copies |
| `dist/` | Prepared installation packages; preserved by ordinary cleanup |
| `.pyscf-agent/` | Private configuration, execution state and archive receipts |
| `.git/` | Repository history; no history rewriting was performed |

The Git history backup `runs/repository-consolidation-20260930/before.bundle`
is retained. `clean.sh --runs` refuses to proceed while any `.bundle` file is
present under `runs/`. Ordinary `./clean.sh --yes` removes caches and build
intermediates; use `./clean.sh --check` to preview that scope.

## QM9 archive

The former local `datasets/` directory was removed after SHA-256 verification.
The archive location is written as `your_username@cluster.example.edu`, under
`/home/your_username/datasets/`. These are placeholders; use the actual archive
account, host, and path when restoring data:

- `qm9_ccsdt_pilot_40/` and `qm9_ccsdt_seeds_500/`: selected structures,
  original source files, selection scripts and provenance.
- `qm9-ccsd500-v2-dict/dataset.h5`: full dataset, 8,318,986,977 bytes.
  SHA-256: `0ed5db1628311985dc654b4a905a2da1a250cf4d37edbfdc101fd6f437b56a64`.
- `QM9_CCSD_upload/`: delivery files, including a verified copy of the full HDF5.
- `qm9-preparation-assets-20261001/`: former local `tmp/` contents, including
  source archive/index, selection dependencies and rendering checks. Its macOS
  dependencies are archival material, not a Linux environment.

The unused five-shard copy was deleted at the user's request. The two seed ZIPs
retain original source files alongside XYZ files. The full HDF5 is not a
replacement for these original geometry archives.

Dataset verification covered 1,156 files; preparation-asset verification covered
1,006 files. Receipts are stored locally under
`.pyscf-agent/dataset-archive-20261001/` and `.pyscf-agent/tmp-archive-20261001/`.
The server manifest is `local-archive-20261001-manifest.json` at the dataset root;
preparation receipts are
`_archive-manifest.json` / `_archive-verification.json` inside the preparation archive.
These are historical verification records, not live server status checks.

## Use or restore the seeds

On Amarel, set the input path directly to
`/home/your_username/datasets/qm9_ccsdt_seeds_500/`. To restore a local working copy
outside this checkout:

```sh
mkdir -p "$HOME/cafes-data/qm9_ccsdt_seeds_500"
rsync -av your_username@cluster.example.edu:/home/your_username/datasets/qm9_ccsdt_seeds_500/ \
  "$HOME/cafes-data/qm9_ccsdt_seeds_500/"
```

Verify restored files against the archive manifest before use. Historical
reports may still use `datasets/...` as their original input paths; map that
prefix to the archive or restored data root. Local `datasets/` is ignored by Git
if an older script requires restoring that layout. Existing tracked deletions
still need to be recorded in a later commit.

The QM9 Markdown runbooks include the archive-location update. The September 21
Word runbook remains a historical export; regenerate it from `build_guide.py`
when a new Word copy is needed.
