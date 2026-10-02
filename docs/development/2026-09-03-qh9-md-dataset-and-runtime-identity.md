# QH9 Molecular Dynamics Datasets And Runtime Identity

Source baseline: `33c7610` (`0.2.0`), merged into `main` on 2026-09-04.

## Scope

This change extends the existing task and study layers with molecular
Hamiltonian dataset production and source-matched remote execution. It keeps
scientific execution inside the task kernel and large numerical arrays in
executor-owned artifacts.

## Implemented

- Added `HamiltonianDatasetSpec`, molecular geometry, AO metadata, sample,
  rejection, matrix-reference, and manifest contracts. A dataset compiles into
  one ordinary Study case per seed molecule; each case runs a complete NVE
  trajectory and retains its sampled frames in typed artifacts.
- Registered the molecular-dynamics job, trajectory output, artifact families,
  and separate dataset generation/collection actions. Planner, CLI, application
  services, and execution receipts use the existing task/executor boundary.
- Added deterministic velocity seeds, explicit frame indexes, QH9 AO ordering,
  Fock/overlap arrays, geometry and energy units, convergence metadata, and
  matrix shape, finiteness, and symmetry checks.
- Added sample/rejection JSONL indexes and dataset manifests. Default splits
  keep all frames of a molecule together; geometry-random and molecule-size
  out-of-distribution protocols must be selected explicitly.
- Kept finalization lightweight for remote tasks. `generate_hamiltonian_dataset`
  materializes a portable dataset on the selected execution target and returns
  a receipt. `collect_hamiltonian_dataset` separately retrieves that generated
  dataset when a local copy is requested.
- Added immutable remote source releases, private worktree bindings, and
  `require_runtime_match` checks. Stale local source or a mismatched remote
  environment, release, source snapshot, or public contract blocks submission.
- Added Web result/session and execution-target integration for these flows.

## Scientific Boundary

The executable MD profile is restricted B3LYP/def2-SVP for neutral closed-shell
H/C/N/O/F molecules, with symmetry disabled, standard SCF, and no density
fitting or active-space/orbital-processing modules. Sampling defaults are
300 K initial velocities, NVE, a 50-atomic-time-unit step, 100 frames, and frame
indexes `9, 19, ..., 99`. NVE does not thermostat the trajectory to 300 K.

The current `qh9_relaxed_scf` profile fixes `conv_tol=1e-8`,
`conv_tol_grad=3.16e-5`, `grid_level=3`, and `diis_space=8`. The dataset
contract retains the reference `qh9` settings with `conv_tol=1e-13`, but the
task validator currently accepts only the relaxed profile. Reference metadata
must not be advertised as an additional executable profile. Outputs record
the runtime PySCF version, reference version 2.2.1, the tolerance deviation,
and `bitwise_reproduction=false`.

The stored Fock matrix is an AO mean-field Hamiltonian label paired with its
overlap matrix. It is not the full many-electron Hamiltonian. The 100 x 10
default describes a plan, not an already generated or validated dataset.
Trajectory energy conservation, full-campaign failure rates, and downstream
learning performance need their own acceptance evidence.

## Design Audit

- Registry entries determine executable jobs, outputs, and postprocessing
  actions; dataset contracts describe the request and persisted evidence.
- Task execution owns the trajectory and numerical arrays. Study finalization
  integrates complete task reports; generation and collection do not introduce
  another scheduler or scientific execution path.
- Resource review remains required before campaign execution. A manifest,
  generation receipt, and collected dataset describe different stages and
  must not be presented as interchangeable completion signals.
- Source fingerprints belong to deployment identity. Common artifact references
  use kind, path, size, MIME type, and description; dataset generation and
  collection do not promise content-checksum verification.
- Study modules still declare ownership by the established orchestrator.
  Typed stage transitions and revisioned parent-report integration remain
  migration work under the existing architecture rules.

## Verification

The 2026-09-04 source-tree suite ran 888 tests in 225.165 seconds: 887 passed
and the opt-in 18-site libDMET benchmark was skipped. The run used Python
3.9.6 and PySCF 2.13.0; the package declares Python >=3.10, so this is not a
supported-version installation acceptance run. Its original output is archived
at `reports/verification/unittest-2026-09-04.log`.

`test_molecular_dynamics.py` includes a real one-frame H2 calculation and
dataset finalization. Dataset contract/postprocessing tests cover the 100-case
plan, failed-trajectory accounting, remote metadata finalization, executor-side
generation, and separate collection. Runtime identity/deployment tests use
controlled transports; they do not establish a new live-cluster baseline.
No complete 100-molecule dataset or new remote deployment is claimed here.

See [the current platform report](../status/strong-correlation-agent-platform.md)
for the retained numerical benchmarks and remaining platform limitations.
