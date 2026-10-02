# CCSD-owned DMET beta — September 25, 2026

`solver.options.impurity_solver_options.beta` remains the single input for
CCSD-DMET. Its Registry default is now **1000.0**. An explicit value must be
positive and finite; invalid/null values are rejected. The adapter passes the
same value to lattice RHF/UHF, density fitting and native CCSD preliminary
impurity SCF, including impurity chemical-potential trials. Both translational
and finite-graph execution use this policy. The Fermi width is `1/beta` in the
Hamiltonian's energy units. CCSD/RDM electron counts remain integer; the
correlated calculation remains a ground-state calculation.

FCI and block2 are unchanged: lattice SCF retains the native infinite-beta
limit, density fitting uses 1000, and FCI preliminary SCF retains its native
zero-temperature/Newton policy. The block2 bridge has no preliminary impurity
SCF. Their forms do not accept the CCSD beta field. No DMET-wide top-level beta
option is introduced.

The Assistant and Planner still save beta under CCSD impurity options and show
the field only for CCSD. Blank means the Registry default 1000. Old saved
explicit values still load; they now control all three CCSD-DMET stages.
Old saved CCSD tasks with no beta resolve to the new default when executed.
Existing calculation artifacts are not rewritten. Reproduction of the former
mixed-temperature policy requires its original code revision and environment.

`dmet_result.smearing` records beta, width, and all three effective scales for
CCSD; iteration history records the actual density-fitting beta. Native impurity
SCF call evidence continues to record observed beta. UI summaries only display
new shared-policy metadata when present, without inventing it for old reports.

## Numerical limitation

The six-site Hubbard ring with `nelec=[4,2]`, U=4, a two-site impurity and a
paramagnetic unrestricted initial reference retains a degenerate smeared
reference at beta=1000. Native integer-occupation CCSD develops nonfinite
amplitudes and fails. This previously documented case no longer uses the old
zero-temperature default. Its regression now checks that the calculation raises
and does not return an accepted result. Balanced-spin CCSD execution, integer
sectors/RDMs and exact two-electron energy checks remain separately tested.
There is no automatic solver substitution, beta change or tolerance relaxation.

## Verification

Behavioral tests cover default and custom CCSD beta, invalid input, UI round trips,
and unchanged FCI/block2 policy. Real libDMET tests inspect the values passed to
lattice SCF, native FitVcorFull and impurity constructors in both execution modes
and restricted/unrestricted references. The open-shell noninteracting FCI square
continues to exercise the original fractional-occupation bath and quality gates.

The optional-provider run used Python 3.12.14, PySCF 2.13.1, NumPy 2.5.3,
SciPy 1.18.1 and local libDMET 0.5 with one numerical thread. Of 76 tests,
73 passed, 2 were skipped and the existing noninteracting FCI square failed
its convergence assertion. The same unchanged assertion fails on the pristine
`e67f49e` checkout in this environment (separate process); it remains unchanged
rather than weakening its quality gate. All new eight-way execution-policy
checks and the CCSD SCF tests passed. This is not a claim that the entire
optional-provider suite is green.

The standard development-environment suite passes: 1,376 tests, 67 optional/
environment skips. Ruff, the configured five-module mypy scope, JavaScript
syntax checking and whitespace checks pass.
