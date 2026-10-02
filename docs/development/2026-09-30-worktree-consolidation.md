# Worktree consolidation — 2026-09-30

Commit `6acf26a` integrates the pending DMET density continuation, branch
analysis, spin-bath completion and model-grid convergence barriers on `main`.
The documentation update adds the corresponding Wiki pages and corrects obsolete
DMET convergence guidance. The molecular diagnostic-policy SI and current
honeycomb manuscript files were subsequently removed from Git tracking at the
author's request. All manuscript and SI directories, saved run manuscript text,
and temporary manuscript/literature extracts were then moved into the desktop
folder `pyscf-agent-论文材料-20260930`, preserving their relative paths. The archive
includes a per-file checksum manifest and available supporting snapshots and plots.
Three Lutein snapshot dependencies were already missing before the move and are
listed in that manifest. Calculation data and software verification reports remain
in the repository workspace.

## Worktree audit

- All 27 changed or untracked files in worktree `7bb1` match the already merged
  preservation commit `24624ca` byte for byte. The worktree remains available
  to its Lutein task.
- Archived snapshot `b3081a9` has the same tree as the already merged `8b60ec2`.
- No additional branch or worktree changes required merging.
- A local Git bundle, source-file snapshot and inventory are retained under
  `runs/repository-consolidation-20260930/`.

Local QM9 upload packages and manuscript SI files remain on disk and are excluded
from publication and source-distribution archives. A packaging regression test
checks these exclusions while retaining ordinary reference documentation and tables.

## Wiki generation and validation

The maintained sources and curated structure were rebuilt with `wiki:curate`,
producing 46 pages, the review export and the runtime Wiki JSON. Wiki lint reports
zero errors and three existing citation warnings in the density-fitting roadmap,
multireference roadmap and scientific benchmark protocol pages.

- Full unittest discovery: 1,498 tests, 77 skipped, no failures or errors
  (218.848 seconds).
- Distribution tests after the final archive-exclusion update: 9 passed.
- Wiki tests: 57 passed. Bounded retrieval was also checked for density
  continuation, convergence barriers and molecular score definitions.
- Ruff, configured Mypy checks, JavaScript syntax and Git whitespace checks passed.

Test logs are stored locally in `/tmp/pyscf-consolidation-20260930/`. Optional
solver tests retain their skips; these checks do not replace production solver
validation. No production calculations were submitted or changed.
