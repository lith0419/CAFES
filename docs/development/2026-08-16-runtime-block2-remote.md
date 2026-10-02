# Development Note: Runtime, block2, and Remote Execution Consolidation

Date: 2026-08-16

## Scope

This version consolidates the block2 scientific path, active-space probe
boundary, Web execution state, and local/remote execution adapters. The changes
are general runtime features and do not contain molecule- or scan-specific
special cases.

## Scientific Runtime

- Added same-spin/symmetry-sector state-averaged DMRG-CASSCF with normalized
  root weights, root energies, root RDMs, and a state-average orbital objective.
- Added exact fixed-`(Nalpha, Nbeta)` Schmidt-rank planning for block2 bond
  dimensions. Oversized schedules are trimmed before execution while energy
  change and discarded weight still control convergence.
- Kept entanglement diagnostics on root 0 and separated targeted DMRG roots
  from a claim of a complete excited-state spectrum.
- Preserved the actually executed model solver in structured results and UI
  summaries, avoiding stale FCI metadata on block2 runs.
- Isolated the MP2 active-space probe as a registered workflow role. The probe
  now preserves the requested CASCI/CASSCF target, block2 solver, and solver
  options through approval and execution.

## Application and Web Runtime

- Added reusable execution-target and resource-profile selection for the
  Calculation Assistant and Computational Study Planner.
- Kept scheduler target/profile data outside scientific task specifications.
- Clarified prepared, stale, pending, running, and completed UI states; blocked
  missing molecular structures before execution; and removed the non-executable
  Holstein-Hubbard Builder option.
- Standardized the current runtime interface and block2 analysis text in
  English.

## Slurm and Distribution

- Consolidated server configuration into one private `server-slurm.ini` with
  named `[server:<name>]` sections and server-scoped
  `[profile:<server>:<profile>]` resource profiles.
- Added explicit `--slurm-profile` propagation through SSH RPC and cluster-id
  verification so a remote profile cannot silently use another server's queue
  settings.
- Kept independent ready tasks eligible for Slurm arrays while adaptive
  refinement, recovery, and state-continuation tasks remain sequential.
- Updated clean/offline packaging rules for the new configuration template and
  continued excluding `runs/` and private `.pyscf-agent/` configuration.

## Boundaries

- Different-spin state averaging, transition-RDM/direct-wavefunction root
  tracking, state-specific projected-root optimization, unrestricted block2
  active spaces, and block2 SC-NEVPT2 are not implemented. Cross-case targeted-
  root tracking from root 1RDMs or natural occupations was added in the
  subsequent low-energy workflow update.
- Server resource profiles are selected by id in the UI; connection details,
  queue names, memory, and wall time remain private INI configuration.
- Client and remote RPC code must use the same package revision.

## Verification

- Full Python regression suite after the subsequent low-energy workflow update:
  575 tests passed on 2026-08-16.
- Deterministic wiki curation wrote 40 pages plus the runtime JSON exports.
- Wiki lint reported 0 errors and 8 non-blocking citation-density/source-
  fingerprint warnings.
- All changed JavaScript files passed `node --check`.
- `git diff --check` passed with no whitespace errors.
- Release commit `13780f9` was pushed to GitHub `main`.
- The same source and private server profile were synchronized to Amarel.
- Read-only Amarel verification passed SSH RPC capability, execution-mode,
  cluster-profile, and public-contract checks. The redacted machine-readable
  evidence is `reports/verification/amarel-2026-08-16.json`.
