# DMET neighbor-density warm starts

Finite-graph DMET can initialize from a converged neighboring Run's full
spin-resolved mean-field 1RDM. This is a density warm start: the target builds
its physical mean-field baseline with its own Hamiltonian, and starts the
auxiliary correlation-potential correction at zero. It does not resume the
donor's correlation potential, chemical potential, DIIS history, or impurity
wavefunction.

## Saved Study retry

Use the existing retry context, preview, and start APIs on the same Study.
Add this override for the selected target case:

```json
{
  "case_overrides": {
    "target-case-id": {
      "solver.options.reference_density_source": {
        "source_case_id": "neighbor-case-id"
      },
      "solver.options.correlation_potential_mixing": 0.1,
      "solver.options.diis_enabled": false,
      "solver.options.max_iterations": 400
    }
  }
}
```

The caller supplies actual saved case IDs and the remaining RetryAction fields
from `retry_context`. This low-level override chooses an explicit donor. The branch-aware service
below can select donors and execute bounded sweeps automatically. Initial seed
labels never certify a final branch.

The executor resolves the donor from its saved successful TaskReport, requires
`converged=true`, and pins its Run ID, density artifact path, and SHA-256.
Optionally provide `source_run_id` with `source_case_id` to select that exact
current or historical candidate Run; a missing Run is rejected. Deferred dependencies are included in retry dependency accounting, and
dependent cases are excluded from the initial independent batch. Only selected
cases receive new Runs; the source result and unrelated cases are retained.

A fixed artifact may instead be supplied as `reference_density_source` with
an absolute `path`, `sha256`, and optional expected `source_run_id` and
`source_case_id`. It has no live dependency on later donor retries. Missing or
incompatible artifacts cause a clear failure rather than falling back to an
analytic seed. A source overrides the configured analytic density seed;
`reference_density_initialization.strategy` reports
`neighbor_mean_field_density` when used.

## Artifact and scientific checks

New finite-graph Runs save:

- `mean_field_density_matrix` in the existing numerical arrays, with shape
  `(2, nsite, nsite)` in sorted site-ID order, including off-diagonal entries;
- a compact `dmet_mean_field_state` NPZ containing that density and JSON metadata,
  with source Run ID, convergence and required-quality status, model and fragment
  compatibility information, and source U/V values.

Restricted densities are stored as two identical per-spin blocks. The density
comes from the final evaluated iteration, before its next potential update.
Unconverged Runs can retain diagnostic state, but cannot act as warm-start
donors. The original `reference_density_seed` is not the final density.

The loader verifies the content hash and source Run, convergence and required
quality checks, site basis/geometry/hopping, fragment partition, reference and
spin particle counts, bath policy, and impurity solver/options. Only on-site U
and intersite V may differ, including their Builder `globals` fields. Density
shape, finite entries, Hermiticity, per-spin traces, and occupations in `[0,1]`
are checked without rescaling or discarding off-diagonal elements. This first
version accepts real finite-graph site densities only.

The target records source and target parameters, signed per-site/per-bond
differences and their maximum absolute values, source Case/Run IDs, artifact
hash, and the baseline reconstruction policy. The density-fit residual remains
a diagnostic; it is not a new required convergence gate.

## Local and server execution

SSH submission attaches a locally available density file and rewrites its path
on the server; the receiver verifies the original checksum. Paths from a prior
Run on the same shared server filesystem are used directly. Cross-server paths
must first be made available to the receiving execution target. The server
runtime must contain this implementation.

Old Runs that lack `dmet_mean_field_state` cannot be used directly. They must be
recomputed with a runtime that saves the final mean-field density; an initial
seed or correlated embedding density is not silently substituted.

## Branch selection and bidirectional sweeps

`StudyApplicationService` and the MCP server expose:

- `analyze_dmet_branches(study_id, policy=None)`: compare saved candidates and
  write a state map, branch energy curves, and crossing evidence. No jobs run.
- `prepare_dmet_continuation(study_id, case_ids=None, branches=None,
  mode="bidirectional", policy=None, mixing=0.2, max_iterations=200)`: save a
  bounded preview and cost estimate. The default branches are `afm` and `cdw`.
- `start_dmet_continuation(study_id, action_id, approve_cost=False)`: start that
  saved action through the existing Study worker and retry lifecycle.

Python service callers can pass `work_dir`; MCP uses its configured Study root.
The Planner result section also has analysis and continuation-preview controls.
The corresponding web actions are `study-dmet-branches-analyze`,
`study-dmet-continuation-prepare`, and `study-dmet-continuation-start`.

Each branch gets at most one new Run per selected physical point. Cases sharing
one Hamiltonian, fragment/reference/bath/impurity policy, and U/V coordinate are
one point even if their analytic seeds or mixing differ. Preparation keeps the
first selected case at that point as the target Task. Continuations retain its
Task ID and create distinct Run IDs. No grid points are added.

Bidirectional mode visits AFM targets in increasing V, then CDW targets in
decreasing V, at fixed U. Before **each** step it chooses the nearest qualified
compatible donor on the intended branch and side, using all saved candidates.
This includes Runs just completed earlier in the sweep and historical Runs
superseded in the current task view. If a solution changes branch, its final
label controls its later eligibility. When no donor exists the step is recorded
as skipped; it does not silently use an analytic seed. `mode="nearest"` uses
Euclidean U/V distance with configurable positive `distance_scales` (defaults
U=1, V=1), and also supports `mixed` or `near_unordered` branches. Ties use energy,
then stable Case/Run IDs. Identical physical points are excluded as neighbors.
Automatic selection currently requires uniform U and V and inline model specs.

Sweeps use mixing 0.2, outer DIIS off, bath policy `max`, and zero initial u.
Mixing and iteration limits can be supplied at preparation. This does not add
an automatic fallback seed or a second damping retry. Independent AF/CDW
calculations remain separate candidates. Existing donor compatibility includes
impurity options such as shared temperature/smearing settings. Old incompatible
bath settings are excluded. The provider still verifies the density bytes and
compatibility before executing.

A whole-sweep preview covers the maximum number of Runs and uses the existing
cost estimator. FCI/CCSD DMET currently has no work/memory estimate there; that
unavailability is shown explicitly and is not a runtime or memory upper bound.
Its Study revision and
execution target are checked again at start. Step selection and retry action IDs
are persisted in `dmet-continuations/<action_id>.json`. Submission, collection,
receipts, locking, and cancellation use existing Study infrastructure. Starting
the same action again does not resubmit. An interrupted action stops; collect
its existing Runs before preparing another sweep. A new preview is a new bounded
sweep and may repeat earlier completed points, preserving all their candidates.

## Candidate history and scientific summary

A retried DMET checkpoint retains previous results in `candidate_runs`, with
original requests and Run IDs. Historical TaskReports are externalized alongside
current reports in `case-reports/`; reload and collection restore the index.
Failed and unconverged numerical attempts remain diagnostic candidates. A blocked
dependency does not replace a previously completed numerical Run. Reports saved
before this feature are not retroactively scanned for unindexed old runs.

`dmet_phase_analysis` in the StudyReport and `dmet-phase-analysis.json` contain:

- independent/continuation provenance, original seed, source Case/Run/hash;
- final correlated charge and magnetic order parameters;
- qualified candidates, branch minima, and the minimum energy per site at each
  physical point, with the selected Run ID and nearly tied Runs;
- possible-hysteresis markers, sampled AFM/CDW coexistence points, and adjacent
  energy-crossing brackets with optional linear interpolation.

The ordinary task table continues to describe current Runs. The DMET competing
states table and phase map use the qualified minimum over **all** candidates,
so a failed/latest retry cannot hide an earlier lower-energy solution. Only
successful, explicitly converged results with passed required quality checks
and finite, consistently normalized energies qualify. Missing legacy quality
information remains unqualified; a density-fit diagnostic alone is not a veto.

The default charge and magnetic thresholds are both 0.001, applied to absolute
sublattice **half-differences**. A 20% band around either threshold is labeled
`ambiguous`. Both orders above threshold means `mixed`, neither means
`near_unordered`, and missing/invalid observations mean `unknown`. All thresholds
are saved and configurable through `policy` (or plan
`comparison.dmet_branch_policy`). No classification uses the initial seed name.

Distinct qualified states whose order magnitudes differ by more than
`state_tolerance=0.01` mark possible hysteresis; reversing a symmetry-related
order sign alone does not. Phase-map rings show those points. The energy tie
tolerance defaults to 1e-6 per site. Crossings require both AFM and CDW at adjacent
sampled V values; gaps are not bridged. A lower competing third state is flagged.
Crossings are candidate finite-model boundaries, conditional on the sampled
branches and DMET approximation. Numerical failure and the extremes of sampled
coexistence are not reported as physical spinodal endpoints.

## Validation

The complete local suite ran 1,491 tests with zero failures and 77 optional
skips. All eight density-restart tests also passed with the local libDMET runtime
enabled. Ruff, the configured mypy checks, and whitespace checks passed. Phase
map layout was rendered and visually checked.

Tests cover final-state classification rather than seed labels, nearest and
directional selection, incompatible models and missing density artifacts,
required versus diagnostic quality checks, historical Run selection after a
failed retry, checkpoint round trips, energy minima/crossings/gaps, sign symmetry,
whole-sweep cost review, interruption/collection without resubmission, and public
Study/MCP/Planner actions. Native libDMET tests verify density-only transfer,
target-Hamiltonian baseline reconstruction, and zero initial u. No cluster jobs
are submitted by these tests.
