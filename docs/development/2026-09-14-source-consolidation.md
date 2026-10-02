# Source Consolidation And Verification

Inspection date: 2026-09-14. Package version remains `0.2.0`.

## Result And Scope

This consolidation collects the accumulated follow-ups after `48a0478` into
one reviewable source state, with maintained Wiki guidance and current campaign
outcomes. It does not restart the cancelled lutein calculation or introduce a
new scientific workflow. Numerical optimization is deferred.

## Consolidated Implementation

- **Study → Task → Run:** stable Study and Task identities, new Runs only for
  selected retries, full-plan retention and shared checkpoint/report updates.
  Adaptive/postprocessing extensions survive current-result collection. Status
  and Collect inspect existing work; execution stays in the application runner.
- **MCP:** thirteen stdio tools and report/Wiki resources delegate to existing
  calculation and Study services. A detached application invocation survives
  client exit and handles static/adaptive execution, saved review, analysis and
  dataset preparation. MCP owns no numerical or adaptive orchestration loop.
- **Molecular CAS:** explicit DF-CASSCF scope, reference-density restart using
  the actual UHF policy, optional SCF stability analysis, and genuine UNO
  candidates carrying the full orbital transformation. Automatic SCF retry
  preserves the requested algorithm; Newton remains an explicit choice.
- **block2:** task orbital settings resolve before provider defaults, explicit
  overrides retain precedence, and requested configuration survives a host
  without block2. Per-call M/sweep/order evidence distinguishes orbital
  optimization from the optional final fixed-orbital solve.
- **Execution and quality:** restart-aware local cancellation/timeout reports,
  DMET option validation and finite quality evidence, and the user-requested
  default Slurm node `your_node`.

The implementation keeps the shared Study helpers and block2 option resolver
instead of restoring duplicate Web/MCP paths. Final cleanup normalizes a CAS
conditional and source spacing. The fcDMFT test fixtures now explicitly skip
when their required optional libDMET transformation provider is absent; the
other tests continue to execute.

## Documentation And Artifact Boundaries

The platform report and README describe the current interfaces and DF/CAS
scope. The lutein status report (author archive: `reports/lutein-workflow-status-2026-09-14.md`)
and compact JSON evidence record both completed CAS(20,20) runs, the ordering
defect/correction, remote regression and cancelled CAS(8,14) experiment. Earlier
development and paper-comparison records keep their dated observations with
links to the final outcome.

Maintained Wiki sources and `curated_structure.json` are updated first, then
`node agent_knowledge/curate_wiki.js` generates the 43-page runtime export.
Guidance distinguishes final M from orbital M, numerical convergence from
execution completion, and full-AO orbital cost from active-space solver cost.
Stale roadmap entries for already executable CASSCF/GW/DMFT routes are corrected.

Git ignores the entire local `.pyscf-agent/` state directory and root `tmp/`.
Public reports retain compact evidence; private bindings, full numerical
reports/orbital arrays, checkpoints and cached paper files are excluded.

## Verification

Final results are recorded in the
verification summary (author archive: `reports/verification/consolidation-2026-09-14-summary.json`)
and accompanying `consolidation-2026-09-14-*` files.

| Check | Result | Boundary |
| --- | --- | --- |
| Python 3.12.14 / PySCF 2.13.1 / MCP 2.2.0 full suite | 1038 tests: 985 passed, 53 skipped; no errors/failures | Supported core/MCP environment; block2/libDMET unavailable. |
| Python 3.9.6 / PySCF 2.13.0 / block2 0.5.3 full suite | 1038 tests: 1016 passed, 22 skipped; no errors/failures | Existing source-regression environment includes libDMET but lacks MCP; Python 3.9 is below the installation minimum. |
| Wheel and sdist | Both built successfully | Version 0.2.0; no release tag or package publication. |
| Fresh Python 3.12 wheel installation outside checkout | All five installed-package checks passed, including packaged resources and N2/Hubbard-dimer smoke benchmarks | Fresh dependencies and wheel, no editable source import; no offline wheelhouse claim. |
| Installed stdio MCP | 13 tools, 43 Wiki pages; H2 energy −1.1167593073964255 Ha; reconnect and repeated collection with exactly one Task Run | Real local process; no remote job. |
| Wiki | 43 pages; 0 lint errors, 3 existing citation warnings | Maintained sources and guidance regenerated through curation. |
| Source hygiene | Diff whitespace, changed JavaScript/shell syntax, and 88 documentation links passed | Wheel, sdist and source snapshot inspected; private state and root `tmp/` excluded. |

The two full suites use `python -B -m unittest discover -s tests -t . -p
'test*.py'` with OMP/OpenBLAS/MKL thread counts set to one. The clean package
check builds with `python -m build`, installs the wheel with its `[mcp]` extra
in a fresh virtual environment, and runs `pyscf-agent-configure verify-install`
outside the checkout. The installed stdio check additionally submits and
collects one H2 task after a client/server reconnect.

The first restricted Python 3.9 run hit seven process-control failures because
the sandbox blocked `ps`, plus an audit false positive caused by a temporary
verification virtual environment under the source tree. Moving that environment
outside the checkout and allowing process inspection made the 18 affected
process/audit tests pass. The initial Python 3.12 run exposed thirteen fixture
errors caused by absent optional libDMET; those fixtures now report explicit
skips rather than failing the core-only installation.

## Remaining Boundaries

Lutein final discarded-weight convergence, higher-M orbital validation and DF
orbital-response performance remain unverified. Full orbital matrices can
still inflate TaskReport JSON; artifact-only transport is future work. This
consolidation does not add an offline platform matrix, new large numerical
campaign, release tag or automatic deployment.
