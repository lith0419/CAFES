# Molecular Hamiltonian Dataset Workflow

## Scope

The QH9-style molecular dataset workflow compiles one dataset specification
into the existing Study and Task layers. Each seed molecule owns one complete
NVE trajectory; sampled frames are dataset samples, not independent scheduler
cases. Registry validation determines the executable boundary.

## Executable Profile

- Use restricted B3LYP/def2-SVP for neutral closed-shell H, C, N, O, and F
  molecules. Disable symmetry, density fitting, and active-space/orbital
  processing. The registered job is `molecular_dynamics`, with `trajectory`
  output and standard SCF.
- Select `qh9` for reference SCF `conv_tol=1e-13`, or `qh9_relaxed_scf`
  for `conv_tol=1e-8`. Both executable profiles use `conv_tol_grad=3.16e-5`,
  `grid_level=3`, and `diis_space=8`. The Registry owns these settings;
  explicit conflicting tolerances are rejected, not silently overwritten.
- Default sampling starts from Maxwell-Boltzmann velocities at 300 K and uses
  NVE with a 50-atomic-time-unit step and 100 frames. Retain zero-based indexes
  `9, 19, ..., 99`; do not confuse sampled frames with MD work performed.
  NVE does not maintain a constant temperature through a thermostat.
- Sampling is independent of the SCF profile. QH9-dynamic-300k uses
  `time_step_au=50`, `steps=100`, and keeps every frame; explicitly set
  `sample_stride=1`, `sample_offset=0`, and `geometries_per_molecule=100`
  for that sampling schedule. The historical default keeps only ten frames.
- The default campaign is 100 molecules x 10 samples. It is a planning
  default; never report those samples as produced without completed evidence.
- Preserve deterministic per-case velocity seeds and explicit molecule,
  seed-geometry, frame, and sample identities. User-supplied seed geometries
  must match the declared molecule count and coordinate unit.

## Layer Ownership

`StudyApplicationService.build_hamiltonian_dataset_plan()` compiles a
`HamiltonianDatasetSpec` and seed geometries into an ordinary `StudyPlan`.
Resource review is initially unapproved. Each case runs a complete molecular
MD task through the selected executor and produces a `TaskReport`.

The task kernel captures positions, sampled potential/kinetic/total energies,
temperature, time, velocities when requested, and QH9-ordered Fock/overlap
matrices. AO metadata records the permutation, phase convention, and atom
slices. Arrays remain in registered NPZ artifacts; compact metadata carries
units, shape, finiteness, symmetry, and convergence evidence.

Fock is a mean-field one-electron Hamiltonian label, not a complete
many-electron interaction tensor. Preserve overlap and AO metadata whenever
Fock matrices are exported or consumed.

## Finalize, Generate, Collect

1. Finalization integrates complete task reports into `samples.jsonl`,
   `rejections.jsonl`, and `dataset-manifest.json`. Account for every expected
   frame, including failed or invalid trajectories. Rejected structures do not
   count as accepted training samples.
2. For remote results, finalization can use self-describing trajectory metadata
   without downloading executor-owned NPZ arrays. A lightweight manifest does
   not establish local availability or validate remote file contents.
3. The explicit registered `generate_hamiltonian_dataset` action materializes
   and validates a portable dataset on the selected execution target. Validate
   file presence, array keys/indexes, and matrix shapes, rewrite matrix paths
   relative to the dataset root, and retain the generation receipt.
4. `collect_hamiltonian_dataset` requires successful generation and retrieves
   that dataset when a local copy is requested. For SSH execution, generation
   returns only a lightweight receipt; collection owns the array transfer.
5. Derive terminal/successful/failed trajectory counts, available/failed/pending
   structures, and generated/collected states from the existing Study plan,
   execution receipts, and manifests. Do not add another mutable job lifecycle.

Neither generation nor collection requires content-checksum validation. Source
fingerprints used for remote deployment are a separate contract.

## Split And Compatibility Rules

- Default `molecule_random` splitting keeps all frames of one molecule in one
  split, with train/validation/test fractions 0.8/0.1/0.1 and a persisted seed.
- `geometry_random` is an explicit geometry-generalization protocol;
  `molecule_size_ood` is an explicit molecule-size protocol. Report which
  generalization question the selected split measures.
- Preserve the selected profile, runtime PySCF version, reference version
  2.2.1, actual SCF setting deviations (empty for strict `qh9`), and
  `bitwise_reproduction=false`. QH9 format and
  convention compatibility do not imply numerical or bitwise reproduction.
- A one-frame numerical smoke test establishes basic execution/artifact
  integration. It does not establish trajectory energy conservation, complete
  campaign quality, or downstream learning accuracy. Those need separate
  numerical acceptance records.

## Related Pages

- [[CAFES Architecture]]
- [[Public Data Contracts]]
- [[Planner Agent Approval Boundary]]
- [[Planner Agent Postprocessing]]
- [[Installation And Remote Verification]]
- [[Scientific Benchmark Protocol]]

## Explicit Trajectory Profile

Dataset planning propagates `compatibility_profile` (`qh9` or
`qh9_relaxed_scf`) to `molecular_dynamics.profile` on each trajectory Task.
Standalone requests must select a profile explicitly;
a generic MD request does not authorize QH9 basis, functional, or SCF defaults.
The current provider still supports only the documented QH9-compatible slice.
Profile scientific defaults are declared in the molecular job registry entry.
Older saved MD requests need this field before re-execution; their reports
remain readable without migration.

## Failed Trajectory Evidence

An integration failure preserves the full PySCF output in `log-pyscf-output.log`
and writes `result-molecular-md-failure.json`. The latter identifies the failed
frame, integrated versus retained frame counts, geometry, velocity seed,
SCF convergence flag and last observed SCF iteration (energy change, orbital
gradient and density change). An earlier iteration's frame index stays explicit
if a later frame fails before starting SCF. Failed trajectories do not emit
successful trajectory arrays or accepted dataset samples.

PySCF's MD message `Gradients did not converge!` checks the gradient scanner's
underlying electronic calculation. For the supported RKS workflow it means SCF
non-convergence; inspect the saved SCF evidence before choosing recovery.
Increasing `runtime.max_cycle` does not require relaxing `conv_tol`. Keep the
same molecule and velocity seed when retrying a whole trajectory in a new Run.
Do not treat the saved failure geometry as a supported MD restart checkpoint.
