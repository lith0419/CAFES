# DMET Density-Fit Audit Across Impurity Solvers

Inspection date: 2026-09-14. This follows the
[native mean-field consistency correction](2026-09-14-dmet-mean-field-consistency.md).

The finite-graph defect identified below is now corrected in production.
All three fitting entries share the same lattice preparation; verification
of the correction is recorded at the end of this report.

## Finding And Scope

FCI, CCSD and block2 DMRG share the same DMET outer loop. The native
translational fitting and interacting-bath cache corrections therefore apply
to all three impurity solvers; changing the solver cannot bypass an outer-loop
Hamiltonian error.

The audit found a remaining instance of the fitting-base defect in
`fit_fragment_local_correlation_potential`. Both its translation-tied and
independent-fragment branches called `slater.FitVcorFull` directly with the
original lattice. Finite-graph lattice HF uses `hcore + vcor`, while the fitter
reads the physical `Fock + vcor`. With an interacting bath and a nonuniform HF
potential, these calls optimize the wrong density map. The preceding fix
covered the translational entry point but missed these two fragment entries.

The finite-graph bath has one k point, so the separate per-k contraction bug
does not change its numerical cache. Its fitting-base defect still matters.
Noninteracting-bath paths skip the physical Fock-cache update and did not show
this mismatch in the controls. A uniform HF potential can shift energies
without changing the density, masking the defect in symmetric fixtures.

## Pre-Fix Numerical Verification

The compact evidence (author archive: `reports/verification/dmet-solver-fitting-audit-2026-09-14.json`)
records 14 small cases and 14 isolated controls, with 26 native fit calls in
each variant. Each case runs one DMET iteration on a six-site Hubbard ring
with two-site fragments and four embedding orbitals. All three solvers run
the translational, translation-tied finite-graph and independent finite-graph
routes using an AF unrestricted reference and interacting bath. FCI adds
noninteracting-bath and nonuniform restricted-reference controls.

For every native fit call, an independent lattice-HF evaluation computes the
density at the same potential, filling and fitting temperature. Its residual
uses the exact fitted fragment indices and spin normalization. This checks
the fit objective rather than relying on a solver's convergence flag.

The translation-tied finite-graph examples show the remaining defect:

| Impurity solver | Reported residual after fitting | Actual lattice-HF residual at the fitted potential |
| --- | --- | --- |
| FCI | 1.9941e-7 | 0.2105022 |
| CCSD | 6.0384e-7 | 0.2111943 |
| block2 DMRG | 8.3425e-7 | 0.2105026 |

All six unrestricted finite-graph interacting-bath cases reproduce the
defect, including the independent-fragment route. The nonuniform restricted
FCI case also reproduces it. The corrected production translational cases
agree to at most `2.78e-17`; noninteracting-bath controls also agree at that
scale.

In the isolated controls, `FitVcorFull` receives a shallow lattice copy with
`getFock()` exposing `hcore` whenever lattice HF uses `hcore`. All 14 cases
then agree to at most `2.78e-17`. The physical Fock remains on the original
lattice. Production source was not changed during the audit itself.

These are single-iteration checks of the density map, not full convergence
benchmarks. They use Python 3.9.6 / PySCF 2.13.0 / libDMET 0.5 / block2 0.5.3
in the existing local source-regression environment, with one thread per
process. Python 3.9 is below the package installation minimum.

## Production Correction And Verification

`prepare_density_fit_lattice` in the existing
[fragment-fitting module](../../pyscf_agent/providers/libdmet/fragment_fitting.py)
now prepares the lattice for the native translational entry and both fragment
entries. When lattice HF uses `hcore`, the helper returns a shallow copy whose
Fock accessors expose `hcore`, retaining the original physical Fock for bath
construction. When lattice HF uses the physical Fock, it returns the original
lattice. The fitting optimizer, thresholds and fragment parameterizations
continue to use the existing implementations.

The production verification (author archive: `reports/verification/dmet-fragment-fitting-fix-2026-09-14.json`)
records **65 tests passed, with no skips**, including the opt-in 1D
interacting-bath benchmark. The density-fit regression now checks 12
combinations across all three entries, RHF/UHF and hcore/Fock modes. The
eight new finite-graph combinations include four that fail before the fix.
Every independent-fragment fit is compared against actual lattice-HF
densities, and the original physical Fock and input potential are checked
for preservation.

All 14 cross-solver cases were rerun against production without the isolated
base-correction hook. Across 26 native fitting calls, the maximum discrepancy
between the reported and independently evaluated residual is `2.78e-17`.
All fit values and energies per site exactly match the isolated corrected
controls. These remain single-iteration mapping checks rather than full
DMET convergence benchmarks.

The separate impurity-SCF stability/recovery issue described in the preceding
mean-field report remains outside this correction.

The registered model solvers outside DMET, molecular DF-CASSCF and fcDMFT do
not call these local DMET correlation-potential fitting functions. Their
independent numerical algorithms were not benchmarked in this audit.

## September 15 Pre-Commit Verification

The complete DMET module was rerun with the opt-in interacting-bath benchmark:
**65 tests passed, with no skips**, in 40.043 seconds. This uses the existing
Python 3.9.6 / PySCF 2.13.0 numerical environment described above; it is not
supported-version installation evidence.

In Python 3.12.14 / PySCF 2.13.1, the DMET contract tests, result-quality tests
and Planner Wiki tests passed **82 tests, with no skips**, in 2.979 seconds:

```sh
OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
python -B -m unittest \
  tests.pyscf_agent.test_libdmet_dmet.LibDmetContractTests \
  tests.pyscf_agent.test_result_quality \
  tests.computational_study_agent.test_llm_wiki
```

This second environment does not contain libDMET, so its checks cover contracts,
quality evaluation and Wiki integration, not libDMET numerical execution.
Maintained Wiki sources and guidance were synchronized with the corrected
fitting semantics and remaining impurity-SCF boundary, then regenerated into
43 runtime pages. Wiki lint reported no errors and three existing citation
warnings. The historical cross-solver and full honeycomb results above were
retained; those campaigns were not repeated during this commit preparation.
