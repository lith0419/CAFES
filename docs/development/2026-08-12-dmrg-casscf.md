# Development Note: DMRG-CASSCF

Date: 2026-08-12

## Added

- Registered optional block2 as the active-space solver for restricted/ROHF,
  single-state molecular CASSCF.
- Added a PySCF FCI-protocol adapter that returns block2 spin-free 1/2-RDMs to
  the PySCF orbital optimizer.
- Reuses the preceding internal MPS across CASSCF solver calls while keeping a
  noncanonical active-orbital permutation fixed throughout the optimization.
- Defers requested entanglement and symmetry analysis until the final optimized
  active Hamiltonian.
- Preserves optimized MO coefficients, macroiteration history, solver-call
  trace, convergence components, orbital ordering, and DMRG checkpoint
  provenance as structured results and artifacts.
- Routes approved large molecular active spaces to DMRG-CASSCF rather than
  fixed-orbital DMRG-CASCI.
- Added a formal H2/6-31G CAS(2,2) benchmark against PySCF FCI-CASSCF.
- Added a joint optimized-orbital/MPS checkpoint for compatible cross-task
  DMRG-CASSCF continuation.
- Added bounded adaptive bond-dimension/sweep escalation and a heuristic DMRG
  energy-error estimate based on discarded-weight extrapolation when possible.
- Added entanglement- and occupation-driven ActiveSpaceAudit expansion. The
  expanded calculation reuses optimized orbitals, starts a fresh MPS, and
  requires explicit active-space approval.
- Added `block2-adaptive-workflows`, a direct numerical benchmark covering the
  joint checkpoint, adaptive bond dimension, error estimate, and active-space
  review contracts.

## Benchmark Record

The maintained 2026-08-12 run passed all three workflow checks. Joint H2/6-31G
DMRG-CASSCF continuation agreed with a cold run within `3.418e-10 Ha`. An
adaptive eight-site Hubbard-ring calculation increased `M` from 16 to 128 and
agreed with exact diagonalization within `1.296e-9 Ha`. The generated CAS(2,3)
proposal remained unapproved, included a cost estimate, reused optimized
orbitals, and deliberately started a fresh MPS. The machine-readable report is
`reports/benchmarks/block2-adaptive-workflows-2026-08-12.json`.

## Current Boundary

- Single-state and same-spin/symmetry-sector state-averaged optimization are
  supported. Different-spin averaging and state-specific projected-root
  optimization remain outside the current contract.
- Restricted or ROHF references only; unrestricted active spaces are rejected.
- SC-NEVPT2 is not attached to the block2 route.
- Cross-task DMRG-CASSCF restart is accepted only when optimized orbitals and a
  compatible MPS are restored together.
- Entanglement inside the current active space cannot directly measure omitted
  orbitals, so expansion remains a bounded boundary-saturation heuristic.
