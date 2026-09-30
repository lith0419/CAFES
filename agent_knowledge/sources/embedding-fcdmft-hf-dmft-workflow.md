# fcDMFT HF+DMFT Workflow

## Executable Boundary

The optional fcDMFT provider executes periodic HF+DMFT only after a correlated
subspace and localized Hamiltonian have been prepared, reviewed, and approved.
Its canonical request uses `task_type=periodic`, `method.name=hf`,
`solver.name=hf_dmft`, and `embedding.provider=fcdmft`.

The provider consumes registered HDF5 artifacts containing the periodic HF
one-body Hamiltonian, Fock matrix, 1RDM, overlap, localized-orbital definition,
and localized two-electron integrals. Restricted data use one spin channel and
one ERI block. Unrestricted data use alpha/beta one-body channels and ERI block
order `aa_bb_ab`. The correlated orbitals must form the contiguous fcDMFT
window `[ncore, nval)`.

For unapproved POSCAR/CIF requests, the provider runs periodic HF and derives an
IAO or IAO+PAO candidate, localized HF/Fock/density matrices, and unit-cell ERIs.
It emits a correlated-subspace audit and stops at explicit user review. The
approved follow-up executes fcDMFT from those registered artifacts. Automatic
preparation requires integer occupations, a gamma-centered mesh, and zero
k-point shift.

## Runtime Composition

`embedding.fcdmft.prepare` validates provider availability, options, and
reference/impurity-solver compatibility. Normal periodic PySCF execution then
computes the HF reference. `embedding.fcdmft.prepare_subspace` creates the
localized proposal when approval is absent. `embedding.fcdmft.hf_dmft`
consumes approved numerical artifacts, runs fcDMFT, and owns the terminal task
status. The task is not marked successful merely because the intermediate HF
reference converged.

fcDMFT is imported lazily during execution because its solver module initializes
MPI. Capability discovery must use the side-effect-free provider availability
probe.

## Solver And Resource Contract

Supported impurity solvers are `cc`, `ucc`, and `fci`. A restricted reference
may use CC or FCI; the current unrestricted route requires UCC. Bath size,
discretization, frequency window, broadening, hybridization tolerance, damping,
chemical-potential control, thread count, and memory are normalized by the
provider. Requested threads and memory may not exceed the active Slurm
allocation; absent explicit limits inherit the allocation conservatively.
The `opt`, `direct`, and `log` bath grids require at least two bath energies.
Approved inputs are rejected before fcDMFT if the projected lattice has no
resolvable hybridization, as occurs for a full-cell Gamma-only isolated limit.

An unconverged HF+DMFT task is retried only after increasing
`solver.options.max_iterations`. Changing the generic HF `runtime.max_cycle`
does not constitute a DMFT recovery. Each attempt remains in task provenance,
and retry JSON, NPZ, and provider-log artifacts use `retry-N` filenames.

Under Slurm, explicit `n_threads` and `max_memory_mb` values are rejected when
they exceed the active allocation. Without overrides, the adapter inherits
`SLURM_CPUS_PER_TASK` and uses 75% of the allocated memory. The study resource
gate may enforce the selected profile's hard memory limit, but there is not yet
a calibrated HF+DMFT working-set estimator. Do not present an uncalibrated
proxy as a memory prediction.

## Results And Artifacts

The structured result reports DMFT convergence, chemical potential, target
occupancy, correlated window, bath configuration, impurity solver, spin and
k-point dimensions, resource provenance, and available numerical arrays. It
does not report a DMFT total energy. The periodic HF energy is retained only as
`reference_energy` and `mean_field_energy`; `reference_converged` records the HF
status separately from the terminal DMFT convergence status. Failed provider
runs likewise clear `energy` and `final_energy` instead of presenting the HF
reference as a DMFT result.

Registered artifacts are:

- `hf_dmft_result`: compact convergence and solver metadata;
- `hf_dmft_arrays`: NPZ hybridization, self-energy, frequencies, and weights;
- `hf_dmft_output_log`: complete provider stdout/stderr;
- `hf_dmft_checkpoint`: fcDMFT HDF5 restart data.

HF+DMFT generated scripts replay the compiled workflow rather than calling the
periodic HF kernel alone.

## Non-Executable Boundary

GW+DMFT is available through its separate staged provider contract. Cluster
DMFT, excited roots, a total-energy estimator, Wannier localization, Planner
scans, and adaptive routing remain unavailable. A
k-resolved physical benchmark and restart validation are required before this
method can participate in automatic recommendations. A one-orbital Gamma-only
atomic limit is useful for preparation tests but is not a meaningful DMFT bath
benchmark.

## Related Pages

- [[Unified Registry Runtime Boundary]]
- [[libDMET Hubbard Workflow]]
- [[PySCF Method Capability Matrix]]
