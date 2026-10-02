# Automatic model-Hamiltonian parameter-grid refinement

Model Studies support one-dimensional midpoint refinement and local rectangular-cell
refinement of two-dimensional scans (since 2026-09-25). This is an
optional sampling policy for fixed-solver `study_mode=static` plans. Existing
molecular adaptive method selection remains a separate workflow.

## Configuration and inheritance

Set `StudySpec.grid_refinement.enabled=true` and choose one or two `axes`.
Direct sweeps support `U`, `V`, `t`, `epsilon`; grid templates also support
names such as `U_value`, provided they only change those coefficient values.
Other variables define independent fixed-parameter groups. Particle sectors,
solver settings, reference choice, DMET fragments and impurity-SCF beta are
inherited from the frozen source. Explicit case lists and per-case overrides
are rejected when refinement is enabled, because they do not define how to
construct an intermediate calculation.

A complete executable example is
[`model_uv_grid_refinement.json`](../../computational_study_agent/examples/model_uv_grid_refinement.json).
The optional policy defaults are:

- `max_new_points=64`: total extra calculation points, across groups and axes;
- `max_rounds=4`: midpoint rounds in 1D, local probe levels in 2D (coarse cells are level 1);
- `min_spacing={axis: 1e-4}`: smallest permitted new coordinate spacing;
- `max_interval={}`: optional maximum accepted gap between sampled coordinates;
- required `energy_per_site`, optional `mean_double_occupancy`, each with
  `atol=0.001`, `rtol=0.01`.

The absolute energy tolerance uses the declared model energy unit per site.
Other supported quantities are nearest-neighbor spin/charge correlations,
staggered magnetization and sublattice charge imbalance. A required quantity
must be measured and comparable; unavailable optional quantities are recorded
but do not invalidate available required evidence.

## Sampling rule

Every initial interval gets a midpoint probe if its endpoints have sufficient
valid data. For a measured triple `(left, middle, right)`, the residual is
`abs(y_middle - (y_left + y_right)/2)`. The tolerance is
`atol + rtol * max(abs(y_left), abs(y_middle), abs(y_right))` per quantity.
An excess residual or maximum-interval constraint requests probes in both
halves. Purely linear steep curves therefore stop after midpoint validation.
Refinement candidates retain their triggering metric residuals and tolerances.
Within an axis, excess-error candidates precede untested seed intervals.

In two dimensions, adjacent coordinates of the rectangular seed grid define
local cells. Each eligible cell first gets one center probe. Its measured value
is compared with the bilinear prediction from its four corners, using
`atol + rtol * max(abs(corners), abs(probe))`. Existing edge-midpoint results
are also checked with the corresponding bilinear weights. An excess residual
or `max_interval` constraint requests the missing edge midpoints, then splits
only that cell into four children. Neighboring cells share points; unrelated
parts of the parameter plane receive no complete rows or columns.

The budget charges each distinct new calculation once. All missing edge points
for one split must fit before that split is persisted. Children receive their
own center probes on the next pass. A bilinear cell can therefore stop after
four seed corners plus one center. `min_spacing` must allow bisection on both
axes; `max_rounds=1` allows coarse center checks but no child cells. Fixed
particle sectors and other nonscanned settings remain independent groups.

Unconverged calculations, invalid metric values and incompatible local-density
coverage are excluded from the corresponding comparison. Missing quantities
are never zero-filled. A CC/MP2 calculation also requires its reference to be
converged when that information is present. Large correlation scores and solver
stress labels alone do not request a new parameter value or select a method.

Midpoint agreement is a local sampling criterion. It does not bound solver
error, prove the absence of narrower unsampled features or establish a phase
transition. A center probe can miss cancellations (for example, a saddle whose
center equals its corner mean); a denser seed grid or `max_interval` is needed
when narrower features matter. The stopping statuses explicitly distinguish `satisfied`,
`budget_exhausted`, `max_rounds_reached`, `min_spacing_reached`, and
`insufficient_evidence`.

## Persistence and execution

`grid_refinement_source` in the plan stores the resolved source specification,
so subsequent rounds do not reread a mutable input file. Original case IDs are
retained; added IDs follow the separate `grid-0001`, `grid-0002`, … sequence. The
`grid-refinement-state.json` artifact records the frozen contract, initial
cases, insertion transactions, refinement levels and interval/cell evidence.
For local refinement it also stores the complete leaf-cell topology. Each
complete insertion is persisted before submission. Restart can recover it even
if the coordinating process stops before the next plan write.

New two-dimensional Studies use `strategy=local_cells`. Existing saved states
without this field keep the legacy alternating-axis, whole-row/column strategy,
including on resume. Running calculations are not migrated or resubmitted.

All rounds use the same Study, the existing per-case execution receipts and the
normal resource checks. A prior cost approval is not reused for a changed cost
fingerprint. A pending insertion and its concrete plan remain available for
resource review. Run resumes sampling; Collect never proposes or submits new
calculations. Retrying selected cases keeps the full report, and a subsequent
Run reassesses changed results. Changing the saved scientific/sampling contract
requires a new Study.

The shared `run_study` entry point serves Python, MCP, background execution and
WebUI. Reports include the sampling status, initial/added point counts,
per-point origin and level, trigger evidence, and unresolved regions.
Planner controls restore the saved policy and show the stopping reason.
Exported curves distinguish initial/added points. Heatmaps show only the color
mesh, without sampling-point markers or their legend, using actual nonuniform
coordinate spacing. Each axis uses sparse, rounded major ticks (up to five
intervals), independently of the sampled points, without minor ticks. Exported
plot data retains point provenance. Local 2D heatmaps draw the actual leaf-cell
rectangles. Colors use a valid center result when present; cells without a
computed center use the bilinear corner mean, explicitly identified in exported
data. A failed/unconverged center or missing corner leaves the cell gray, never
silently interpolated. Multiple fixed-parameter groups cannot be overlaid.

## Verification

The numerical artifact (author archive: `reports/verification/grid-refinement-2026-09-24.json`)
records the original alternating-axis implementation’s actual PySCF FCI calculations for a two-site half-filled Hubbard model
with `t=-1`. At every executed point, energies and double occupancies agree
with the analytic extended-dimer solution within `1e-10`.

| Scan | Initial points | Final points | Absolute metric tolerance | Maximum dense energy interpolation error/site |
| --- | ---: | ---: | ---: | ---: |
| U in [0,12], V=0 | 3 | 29 | 0.003 | 0.000625574 |
| U in [0,8], V in [0,4] | 9 | 117 (13×9) | 0.03 | 0.00921349 |

Dense validation evaluates the analytic reference on 257 points and 257×65
points respectively, using linear/bilinear interpolation of the calculated
samples. These errors are observed for this benchmark, not general guarantees.
Both completed Studies resumed with zero additional solver executions.

Regression coverage includes linear and curved signals, bilinear interior
probes, localized curvature, shared-edge budgets, independent particle sectors, missing data,
DMET coverage identity and beta/fragment inheritance, insertion interruption,
resource-review recovery, source-file freezing, invalid input and UI round trips.
The browser check configured an 8-point budget, saved/ran a real FCI Study and
reopened it with the same settings and a complete five-point report.

### Sequential grid identifiers

Refined points use a separate sequence: `grid-0001`, `grid-0002`, and so on.
Seed points keep their `case-0001` identifiers. The allocator resumes after the
largest saved numeric grid ID, and persisted insertion IDs are reused after an
interruption. Legacy `grid-<hash>` IDs are retained for existing calculations.
Already running coordinators retain the implementation loaded when they started.

### Local-cell regression coverage (2026-09-25)

Synthetic localized curvature verifies that only the affected quadrant acquires
smaller cells. Checks cover area preservation, unique shared edges, complete
split transactions under budget, maximum depth/spacing, unavailable results,
restart with stable IDs, legacy-state recovery, live progress, and heatmaps
with masked failed centers, reversed axes, no markers and sparse ticks.
These tests validate sampling mechanics; they are not physical reference labels.

The local FCI verification (author archive: `reports/verification/local-grid-refinement-2026-09-25.json`)
uses the same extended Hubbard dimer and U=[0,4,8], V=[0,2,4] seeds.
With absolute energy/site and double-occupancy tolerances of 0.03, it stops at
90 points and 37 leaf cells (two local depths). All calculated quantities agree
with the analytic solution to 4.3e-15. The observed maximum bilinear energy
interpolation error on 17×17 samples per leaf is 0.024708 per site; this is an
empirical interpolation check, distinct from the heatmap's piecewise constant
cell colors. Resuming executes zero additional calculations.
