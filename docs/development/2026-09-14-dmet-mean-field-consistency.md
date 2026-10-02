# DMET Mean-Field Consistency

Inspection date: 2026-09-14.

Follow-up: the [cross-solver audit](2026-09-14-dmet-solver-fitting-audit.md)
confirmed these corrections cover all three impurity solvers in translational
DMET and identified the same fitting-base defect in the separate finite-graph
fragment entries. The linked report now records their production correction:
all three fitting entries share one preparation helper, with 65 regression
tests and 14 cross-solver mapping checks passing.

## Problem And Result

Two provider defects prevented the native translational density fit from
representing the same mean field used by the DMET loop:

- With `use_hcore_as_emb_ham=True`, lattice HF diagonalizes `hcore + vcor`,
  but libDMET's `FitVcorFull` always reads `getFock()`. The agent's `vcor`
  already contains the physical baseline and auxiliary correction, so fitting
  against the interacting-bath physical Fock adds an extra HF potential.
  Fitting now receives a shallow lattice copy whose Fock accessors expose
  `hcore`. The original physical Fock remains available to the bath. The
  `use_hcore_as_emb_ham=False` path continues to fit against the physical Fock.
- For cell-local interactions, the physical HF potential must be constructed
  from the local density `rho(R=0) = mean_k(rho(k))`. Previously the cache
  contracted the interaction separately with each k-point density, creating
  an unphysical k-dependent potential. It now contracts once and broadcasts
  the potential across k points, with the existing RHF spin-trace convention.

Both changes are in the existing
[libDMET provider](../../pyscf_agent/providers/libdmet/dmet.py). They preserve
the native fitting optimizer, convergence criteria, impurity solvers and
public options.

## Verification

The compact evidence (author archive: `reports/verification/dmet-mean-field-consistency-2026-09-14.json`)
records the regression and full-calculation results. Numerical artifacts,
TaskSpecs, checkpoints and observation scripts remain in ignored local runs.

The two new numerical regression tests cover eight RHF/UHF subcases. Four
subcases fail on the old implementation and all pass after the corrections.
The fitting residual is compared with an independent lattice-HF density
evaluation, including preservation of the original Fock and potential.
The bath Fock is compared with libDMET's native local J/K routine for one
and three k points, with nonuniform k-point densities and cache mirrors.

The complete DMET test module passes **64 tests, with no skips**, including
the opt-in 1D interacting-bath benchmark and the existing finite-graph,
intersite-interaction, FCI, CCSD and block2 checks:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
PYSCF_AGENT_RUN_LIBDMET_BENCHMARKS=1 \
python3 -B -m unittest tests.pyscf_agent.test_libdmet_dmet
```

The original periodic 4×4 honeycomb case has 32 lattice sites, a 2×2-cell
impurity containing 8 sites, and 16 embedding orbitals/electrons at U/t=4.
Two full runs use the corrected production provider and the same original
TaskSpec, with unchanged tolerances: energy `1e-6`, mean-field density
`1e-4`, density fit `1e-4`, and CCSD `1e-10`.

| SCF setup | DMET iterations | Final fit residual | Impurity UHF converged | CCSD / lambda converged | Energy per site |
| --- | --- | --- | --- | --- | --- |
| Experimental ADIIS/CDIIS acceptance hooks | 9 | 5.5250541e-5 | 9/9 | 9/9 | -0.7849623991 |
| Production default, observation only | 10 | 8.6257731e-5 | 9/10 | 10/10 | -0.7849610659 |

The ADIIS/CDIIS run uses no Newton or shifted retry. Its energy, complete
iteration history and all 12 finite numerical arrays exactly reproduce the
earlier isolated two-fix control. The harness does not patch either corrected
provider function. This is convergence evidence for this case, not a proof
of a global energy minimum.

Verification uses the existing Python 3.9.6 / PySCF 2.13.0 / libDMET 0.5
source-regression environment with one thread per process. Python 3.9 remains
below the package's supported installation minimum of 3.10.

## Remaining SCF Boundary

Historical boundary as of September 14. The [September 20 follow-up](2026-09-20-ccsd-impurity-diis.md)
promotes the validated ADIIS/CDIIS policy into production and stops failed
preliminary SCF before CCSD. The measurements below retain their original date.

The experimental SCF strategy remains in local acceptance scripts. These two
provider corrections do not promote it into the production solver.

In the production-default run, the first impurity UHF reaches 200 cycles
without converging; its unshifted orbital-gradient norm is `0.0184645`.
libDMET still proceeds to CCSD. The following nine UHF solves and all ten
CCSD amplitude/lambda solves converge, and the DMET outer criteria pass.
The current reported quality checks describe the outer DMET result and do
not certify every intermediate impurity SCF solve. Consequently this run
does not establish a fully converged inner-solver history.

The next stability change should integrate the validated impurity SCF
strategy and propagate an exhausted SCF failure before starting CCSD.
