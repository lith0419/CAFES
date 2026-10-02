# CCSD impurity SCF beta — September 24, 2026

> Historical policy and numerical evidence. The default and parameter scope were
> superseded by [CCSD/DMET shared beta](2026-09-25-ccsd-dmet-shared-beta.md).
> The original experiments below have not been reinterpreted or overwritten.

## Meaning and scope

Use the existing nested impurity-solver options:

```json
{"impurity_solver": "ccsd", "impurity_solver_options": {"beta": 1000}}
```

This lives under `solver.options`. The option is registered on
`embedding.impurity_solver.ccsd`, validated once during DMET normalization,
and passed through the existing solver constructor. Beta must be positive and
finite; omit it to preserve native zero-temperature SCF. Per-call result
evidence records numeric beta, or JSON `null` for the infinite-beta limit.
The comparison below found a real nonzero-spin regression with a fixed global
beta, so the final implementation keeps this an explicit option. FCI and
block2 settings are unaffected.

Passing `beta=1000.0` to native libDMET `CCSD` applies Fermi smearing of
width `0.001` in model-energy units to its preliminary RHF/UHF. The existing
ADIIS/CDIIS policy and SCF acceptance checks remain in place. Native libDMET
sets integer occupied counts for CCSD and integer occupations for correlated
RDM evaluation. This is not finite-temperature CCSD, and does not change the
lattice mean-field or correlation-potential fitting beta.

Implementation was checked against the installed libDMET 0.5 `solver/cc.py`
constructor, `run` SCF call, integer `nocc` branch and RDM occupation branch,
and `solver/scf.py::SCF.HF` Fermi-smearing setup.

## WebUI follow-up

The Calculation Assistant and Study Planner now show **CCSD Impurity SCF β
(optional)** only for DMET + CCSD. The blank default omits beta; entering
1000 sends `impurity_solver_options: {"beta": 1000}`. The hint distinguishes
zero-temperature SCF from finite-beta Fermi smearing and states the energy
units. Invalid explicit input is preserved for the backend to reject rather
than silently reverting to the blank default.

Task snapshots, restored TaskSpecs and saved Study preparations retain beta.
Switching to FCI/block2 or another model solver disables/hides the control and
does not leak the CCSD option into that solver's request. Changing beta
invalidates the assistant's existing prepared request/preview just like the
other solver controls. Result summaries show beta from observed SCF calls,
including the zero-temperature limit, and leave legacy missing evidence absent.

Verification: all 65 focused WebUI/API tests pass. Two new behavior tests run
the real JavaScript collection/restoration functions, exercise blank/numeric/
invalid values and solver switching, validate the resulting TaskSpec, and
prepare/reload a saved Study through the existing application service.
Seven modified JavaScript assets pass syntax checks. Browser checks verify
the CCSD-only controls, assistant task switching, and Planner draft updates.
After updating the maintained Wiki guidance and regenerating its export, all
55 Wiki/planner tests pass; lint retains zero errors and three existing warnings.
No new scientific calculation is required for this UI follow-up.

## Same-input Hubbard comparison

The archived September 20 TaskSpec was replayed through
`CalculationApplicationService` and `LocalExecutor` in separate Runs. Its inline
model is retained; only the obsolete original-workspace `input_file` reference
was cleared in both runs. No original report was changed. Both runs used the
same serialized input, tolerances and single-thread environment. The baseline
ran on the unmodified `bb5ee41` source before the beta change.

Model: 4x4 periodic honeycomb primitive cells, 32 sites, half filling
`nelec=[16,16]`, hopping `t=-1`, `U=4`, `V=0`, 2x2 primitive-cell impurity
(8 sites), interacting bath, unrestricted reference with AF initial density.

| Observation | Original beta=infinity | beta=1000 |
| --- | ---: | ---: |
| Total energy / \|t\| | -25.118796770358 | -25.118796825137 |
| Energy per site / \|t\| | -0.784962399074 | -0.784962400786 |
| DMET iterations | 9 | 9 |
| Accepted impurity SCF calls | 9/9 | 9/9 |
| First impurity SCF cycles | 24 | 31 |
| Total impurity SCF cycles | 185 | 257 |
| Final density-fit residual | 5.5250541e-5 | 5.5056251e-5 |
| Final CCSD / lambda convergence | true / true | true / true |

The energy difference is `-1.7118482e-9 |t|` per site, below the scale of the
outer convergence threshold. Maximum site-density and magnetization changes
are `2.8247e-8` and `1.3302e-7`. This case retains essentially the same numerical
result but requires 38.9% more impurity SCF cycles; it does not demonstrate a
convergence-speed improvement. The baseline reproduces the archived September
20 energy exactly. One elapsed-time measurement per version includes startup
overhead and is not a controlled timing comparison.

## Nonzero-spin regression

The existing six-site periodic Hubbard-ring regression with `nelec=[4,2]`,
two-site impurity, unrestricted CCSD and one outer iteration passes on pristine
`bb5ee41`, exported independently with `git archive`.

With beta=1000, its preliminary SCF converges but retains an alpha HOMO-LUMO
gap of approximately `2.22e-16` (beta gap approximately `1.0`). Integer CCSD
uses `(3,1)` occupied orbitals in `(5,5)` embedding orbitals. The first CCSD
update grows to order `1e15`; amplitudes become nonfinite and lambda DIIS
eventually raises `ValueError: array must not contain infs or NaNs`.
Thus converged smeared SCF is not sufficient to establish a suitable
single-reference CCSD starting point. No automatic level shift, zero-temperature
retry or tolerance relaxation was introduced to hide this result.

## Evidence and verification

- Numerical comparison and provenance (author archive: `reports/verification/ccsd-impurity-beta-2026-09-24/comparison.json`)
- Reusable task input (author archive: `reports/verification/ccsd-impurity-beta-2026-09-24/task-spec.json`)
- Application replay driver (author archive: `reports/verification/ccsd-impurity-beta-2026-09-24/run_case.py`)
- Baseline and changed Runs: `runs/ccsd-impurity-beta-20260924/honeycomb-before/`
  and `honeycomb-after/`. They retain TaskReports, DMET results, iteration
  histories, numerical arrays and complete provider logs.
- Initial fixed-beta DMET/SCF/recovery suite: 77 tests, 75 passed, one opt-in
  benchmark skipped, and the nonzero-spin regression above failed. The original
  regression passes independently. The focused SCF/recovery/Registry/Wiki suite
  passes 50 tests, including genuine fractional-occupation RHF/UHF and integer
  CCSD/RDM sector checks. The failing test expectation was not weakened.
- Final optional-parameter implementation: 117 tests, 116 passed and one
  opt-in benchmark skipped, with no failures. The original nonzero-spin
  regression passes unchanged. Both application Runs were repeated on the
  final source: `honeycomb-default-final` omits beta and
  `honeycomb-beta-1000-final` explicitly sets it to 1000. Both energies match
  their respective initial comparison results exactly. Every final SCF call
  records the expected beta and passes its convergence checks.
- To reproduce using new Run labels, execute `run_case.py LABEL` for the
  original behavior or `run_case.py LABEL --beta 1000` for smearing, under
  the recorded single-thread scientific environment. The driver saves the
  effective input and exact JobHandle; existing Run IDs are not overwritten.
- Deterministic Wiki curation produces 44 pages. Lint reports zero errors and
  the same three pre-existing citation warnings. All 20 curation/retrieval
  tests pass after regenerating the final Wiki.
- Numerical environment: Python 3.9.6, PySCF 2.13.0, libDMET 0.5, one thread.
  This is source numerical evidence; Python 3.9 is below the supported
  installation minimum. No cluster deployment or cluster calculation was used.
