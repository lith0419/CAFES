# Study recovery actions — 2026-09-17

The Honeycomb DMET scan `20260917-230528-1c555977` exposed an ineffective
manual retry. An impurity UHF/DIIS exception was treated like an iteration-limit
result, and the retry changed `runtime.max_cycle`. That field does not control
DMET or its preliminary impurity SCF, so the same numerical failure repeated.

## Recovery behavior

The existing Study execution-quality gate now offers a concrete proposal when
the failed stage supports one. A traceback identifying a `LinAlgError` in native
libDMET FCI/CCSD preliminary SCF DIIS produces **Prepare Retry without Impurity
SCF DIIS**, with the reason and exact change:

```json
{"solver": {"options": {"impurity_scf_diis": false}}}
```

`impurity_scf_diis` is registered with default `true`. The provider applies it
to each native impurity solver instance, including repeated chemical-potential
fitting calls, in both finite-graph and translated execution. It does not alter
lattice/outer-loop DIIS or global PySCF classes. The block2 bridge has no such
preliminary SCF and rejects disabling the option.

Preparing the action creates a selected-case plan within the same Study. Run
Plan uses the existing execution and result-merge path; preparing alone does
not submit a calculation. The Hamiltonian, impurity solver, tolerances and
other case results remain intact. The action is rejected when its specific
failure evidence is absent or the proposed option is already disabled.

New TaskReport errors carry the provider recommendation. Historical reports
can obtain the same recommendation from their saved traceback. The frontend
shows the backend proposal and does not implement another error classifier.
Unclassified errors retain their diagnostic evidence rather than receiving an
unsupported parameter change.

Iteration-only retries of execution exceptions remain rejected. An unconverged
DMET outer loop instead offers `solver.options.max_iterations`; more outer
iterations do not guarantee recovery from a density-fit plateau. The separate
`solver_max_cycle` is passed to the correlated impurity solver, not preliminary
impurity SCF. Ordinary molecular runtime retries canonicalize shorthand aliases
before modifying the effective runtime setting.

## Numerical verification

The failed `case-0003` (`U=1, V=2`) was rerun locally in an isolated verification
directory with only impurity SCF DIIS disabled. It completed in **53.37 s** and
converged after **13 DMET iterations**, with total energy
**33.23230606402897 a.u.**

| Criterion | Final value | Required limit |
| --- | ---: | ---: |
| Total-energy change | 6.3173e-7 | 1e-6 |
| Maximum consecutive mean-field density change | 6.3112e-5 | 1e-4 |
| Correlated-to-auxiliary density residual | 1.5597e-5 | 1e-4 |

All **208 FCI calls converged**. **150 preliminary SCF calls did not converge**;
this is not evidence that every intermediate UHF reference was repaired.
Full CI can still solve the complete embedding space in those orbitals. The
verification establishes the reported FCI/DMET convergence for this case, not
universal SCF convergence or convergence of other grid points.

A small Hubbard-ring regression also compares DIIS on/off in both finite-graph
and translated modes, verifies the option reaches every impurity SCF call,
and checks that FCI-based energies agree to seven decimal places. Saved-Study
integration exercises preparation without submission, a third attempt after
an earlier retry, and preservation of the other case's result.

Validation: 96 recovery/UI/saved-Study tests and one error-reporting test passed
on Python 3.12; 49 libDMET contract/numerical/recovery tests passed on Python
3.9. JavaScript syntax and diff whitespace checks passed. Wiki lint reported
zero errors and three citation warnings on other pages.

The compact result, selected-case plan and action are in
`reports/verification/dmet-diis-recovery-2026-09-17/`. Full numerical output is
in `runs/verification/dmet-diis-recovery-2026-09-17/`. The original Study reports
were not overwritten. Earlier iteration-boundary verification remains in
`reports/verification/study-recovery-actions-2026-09-17.json`.

## Loading the changes

An already running WebUI must restart to load the new backend option and
recovery handler. Then reopen the saved Study and collect its existing results
to refresh recovery actions; this does not resubmit jobs. A page refresh alone
only reloads frontend assets. The current port-8001 process was left running;
its launch environment is not reproduced by a project-local `llm.env` file.

## Historical comparison

A separate replay uses the complete September 15 snapshot `c620f3b` and the
original failed U=1, V=2 TaskSpec in the current numerical environment
(PySCF 2.13.0 / SciPy 1.13.1 / NumPy 2.0.2). It fails after 61.32 seconds with
the same impurity DIIS `LinAlgError: Internal Error.` All 13,683 logged SCF
cycle lines are numerically identical to the original failed run, including
the last cycle 17; 482 preceding FCI calls converge in both. This failure does
not require the source cleanup made after that snapshot. It is not a replay
of the old dependency environment or a diagnosis of the internal LAPACK cause.

The successful four-case Study `20260917-232825-cb0f013f` covers U=2,4 and
V=0,0.5 with a PM reference-density seed. The successful four-case Study
`20260918-000404-bbac2e3b` covers U=2,4 and V=0.5,1 with an AF seed. Neither
contains the failed U=1,V=2 point. The September 14 conservative ADIIS/CDIIS
SCF experiment also remains an acceptance-script strategy, not the production
default. Its successful results therefore do not establish that the current
production SCF path already contains that stability strategy.

Evidence: `reports/verification/dmet-history-comparison-2026-09-17.json`.
