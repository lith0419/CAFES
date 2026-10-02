# Worktree consolidation — 2026-09-29

The integration branch combines the primary checkout with the saved changes in
worktrees `7bb1` and `920f`, plus the archived `deee` snapshot. Each source has a
preserved Git commit; the secondary working directories have not been rewritten.

## Sources

| Source | Base | Preserved change commit |
| --- | --- | --- |
| Primary checkout | `7284347` | `c5ff8f5` |
| `7bb1` | `bb5ee41` | `abe05de` |
| `920f` | `c620f3b` | `8b60ec2` |
| Archived `deee` | `bb5ee41` | `16c6bf7` |

The primary branch also carries the earlier grid-refinement and Study recovery
work that had not reached `main`. The combined changes include DMET damping,
bath diagnostics and FCI controls; local Study cancellation and its UI;
orbital/MPS continuation; CCSD label artifacts; plotting rules and styles; and
Lutein manuscript notes.

## Conflict resolution

- Retained the newer curated Wiki generation timestamp.
- Retained optional loading of file-backed checkpoint reports, including the
  compact status path that does not load complete numerical reports.
- Combined orbital-array externalization with CCSD label artifact persistence.
- The archived damping changes were already present in the primary checkout.
- Updated integration tests for the stop-button DOM, plot selection with repeated
  coordinates, plotting styles, retained unit metadata, and the additional CCSD
  observable in the registry audit.

## Calculation state

This integration does not deploy a runtime or move a Study coordinator. Existing
Slurm jobs continue with their submitted runtime. The ignored `runs/` directory
is absent from the archived `deee` Git snapshot; recovering source history does
not recover that Study's scheduling state.

## Validation

- `python -m ruff check .`: passed.
- `python -m mypy`: passed for the five configured source files.
- Full unittest discovery: 1,453 tests, 73 skipped, no failures or errors
  (79.623 seconds). Optional solver/runtime tests retain their existing skips.
- The full run used a temporary `PYSCF_AGENT_RUNS_DIR` and permitted localhost
  servers and test-process inspection. An earlier sandboxed run could not bind
  sockets or inspect worker processes.

## Final consolidation into main

The final pass also includes native DMET energy/mean-field-1RDM convergence,
CDW density seeds (`bc482a6`), and the new Lutein supplementary-information
sources, coordinate archive and production evidence from `7bb1` (`24624ca`).
`920f` had no changes beyond its earlier saved snapshot. The primary preservation
commit `23f3564` has exactly the same tree as `c5ff8f5`; its ancestry is retained
without replacing the later integration changes. Two unused imports in the SI
helper scripts were removed.

Before cleanup, all local and remote branch tips were checked as ancestors of
the final integration. A complete Git bundle and the branch/worktree inventory
are saved locally under `runs/repository-consolidation-20260929/`. The two
secondary worktrees remain in place because other tasks still use them.

Final validation: Ruff and Mypy passed; unittest discovery ran 1,456 tests in
93.175 seconds, with 74 skips and no failures or errors. The test output is saved
in `/private/tmp/pyscf-consolidation-final-20260929/tests.log`.
