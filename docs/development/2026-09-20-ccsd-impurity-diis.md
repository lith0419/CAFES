# CCSD impurity ADIIS/CDIIS

## Problem and implementation

Native libDMET CCSD could continue after its preliminary impurity UHF reached
the cycle limit. The validated September 14 ADIIS/CDIIS experiment was still
outside production. The agent now installs that policy on each native CCSD
solver instance, for both restricted and unrestricted references.

`pyscf_agent/providers/libdmet/impurity_scf.py` subclasses the native SCF
driver and configures each RHF/UHF object as libDMET creates it. Integral
construction, SCF iteration, CCSD amplitudes, lambda equations and RDMs remain
native. No process-wide SCF class changes or duplicate SCF kernel are added.

The mixing metric is the RMS Fock-density commutator in an orthonormal basis,
including spin-resolved overlaps. Native ADIIS and CDIIS receive the same
chronological history. Above `1e-2`, ADIIS supplies the Fock matrix; below
`1e-4`, CDIIS does; a linear blend connects them. The native default DIIS
space is 12 and every impurity solve gets fresh history.

CCSD now uses ordinary SCF without Newton for both reference types. There is
no automatic level-shift retry or relaxed tolerance. The existing
`impurity_scf_diis=false` option disables the hybrid completely. FCI and block2
behavior is unchanged; neither lattice nor outer DMET DIIS is modified.

After preliminary SCF, acceptance requires native convergence, finite energy
and density, and an unshifted orbital-gradient norm below `conv_tol_grad`
(or the native default `sqrt(conv_tol)`). Failure stops before CCSD starts for
that impurity call. The exception identifies the stage, cycle count, gradient
and tolerance. The existing output log is retained on failure.

The existing `impurity_solver_details` result records all preliminary SCF
calls and final CCSD/lambda convergence flags. The shared impurity-convergence
quality check consumes these observations. Final CCSD flags are not a stored
history of every earlier CCSD/lambda solve, and these fields do not certify
lattice SCF or a global energy minimum.

## Original-case acceptance

The original saved TaskSpec was executed without experimental hooks or option
changes: 32-site periodic honeycomb, U/t=4, 2x2 primitive-cell impurity (8
sites), 16 embedding orbitals/electrons, unrestricted CCSD.

| Observation | September 14 native default | Current production |
| --- | ---: | ---: |
| First impurity UHF | Failed after 200 cycles | Converged in 24 cycles |
| Accepted impurity SCF calls | 9/10 | 9/9 |
| DMET iterations | 10 | 9 |
| Energy per site | -0.784961065855 | -0.784962399074 |
| Final density-fit residual | 8.6257731e-5 | 5.5250541e-5 |

The current energy exactly matches the archived experimental hybrid result.
The maximum accepted SCF gradient is `2.4421e-7`, below `3.1623e-6`. The local
run took 7.71 seconds; this is not a cross-machine performance comparison.

## Verification and limits

- Two-electron Hubbard CCSD agrees with the exact energy within `1e-7` for
  RHF/UHF, with DIIS enabled and disabled; density traces and 2RDMs pass.
- Forced one-cycle SCF failure prevents CCSD construction, including with
  DIIS disabled. Repeated solves retain independent histories and leave native
  SCF classes unchanged.
- The DMET/recovery/numerical suite ran 76 tests: 75 passed. The remaining
  Registry ownership check reports the pre-existing missing module owner for
  `molecular.orbital_processing.frozen_orbitals`, outside this DIIS change.
- Seven recovery/contract tests also pass under Python 3.12.14.
- Numerical tests use the existing source-regression environment, Python
  3.9.6 / PySCF 2.13.0 / libDMET 0.5, with one thread per process. Python 3.9
  is below the package installation minimum of 3.10; this is not supported-
  environment installation acceptance.
- Runtime wiki was regenerated; lint reports zero errors and three unrelated
  citation warnings.

Evidence: verification record (author archive: `reports/verification/ccsd-impurity-diis-2026-09-20.json`).
Full local output: `runs/dmet-ccsd-diis-20260920/`.
