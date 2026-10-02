# Postprocessing Contract Cleanup

Implementation date: 2026-09-17. This follows the
[WebUI context and postprocessing audit](2026-09-17-webui-context-postprocessing-audit.md).
The preceding audit remains a record of the earlier implementation.

## Design And Behavior

PostprocessContext remains the owner of scientific row eligibility. It now
excludes explicit failed/nonterminal statuses even on legacy rows without a
publication flag, checks available TaskReport status, and records excluded rows
with case identifiers and reasons. It exposes eligible rows, scientific
columns, numeric columns and case variables through the existing application
service. The suggestions API can return this context without generating plot
suggestions. WebUI consumes it and no longer duplicates the status filter or
parses numeric prefixes. Responses from an earlier report/task are ignored.

PlotSpec normalization uses the existing scalar-input validation helpers.
Invalid explicit options no longer become defaults: zero/fractional DPI,
invalid sizes, nonfinite heatmap centers, nonexistent grouping fields and
unknown options are rejected. Supported boolean aliases are normalized
losslessly, so `sort_by_x="false"` means false. Plot input errors return HTTP
400. Documented defaults apply when options are omitted.

Units come from numeric cells or the actual case TaskSpec, including its
canonical `system.unit`. A stored TaskSpec takes precedence over a stale
request's unit. Missing evidence does not imply Angstrom. A selected column
with different units, or a mixture of specified and unspecified units, produces
an explicit error. This change does not introduce automatic unit conversion.

The selected plot data is resolved before rendering, and the same numeric rows
are used for the PNG and the TSV/JSON export. Missing numeric or grouping values
carry exclusion reasons in the data JSON and API response; raw Study rows stay
unchanged. Plot requests in a batch are normalized before any rendering, and
output names must be distinct after filename normalization.

Heatmap coordinates must be unique after numeric parsing. Two rows with the
same x/y cannot overwrite one another. Users must choose a series or aggregate
data explicitly; grouped heatmaps are rejected until faceting is implemented.
The existing Matplotlib appearance and signed color scale remain in use.

## Result Analysis Context

The shared LLM result-context function selects output contracts from actual
result field names and artifact kinds. Registry contract construction can be
scoped by the known system type, avoiding molecular/periodic instructions in a
model-Hamiltonian analysis. Selected contracts appear once in the user payload;
the system prompt no longer repeats the entire output catalog. Structured
prepared-request data stays structured instead of being nested as escaped JSON.

All scientific comparison rows remain present. Plotting, plot suggestions and
dataset generation/collection remain deterministic and do not call an LLM.
Optional single-task execution feedback shares the context selection, no
longer applies Chinese substring filters, and records model failures under
`llm_feedback` without changing numerical status or energy.

## Token Replay

The same offline audit utility and scientific fixtures were replayed through
the real WebUI request functions and Python handlers. Model replies remain
fixtures; no live inference or remote calculation was submitted. Counts below
are message-content tokens using o200k_base, not provider billing.

| Analysis input | Before | After | Reduction |
| --- | ---: | ---: | ---: |
| Synthetic 1-case DMET table | 12,879 | 752 | 94.2% |
| Synthetic 16-case DMET table | 13,735 | 1,548 | 88.7% |
| Synthetic 64-case DMET table | 16,471 | 4,092 | 75.2% |
| Synthetic 256-case DMET table | 27,415 | 14,268 | 48.0% |
| Existing saved 2-case report | 13,120 | 1,113 | 91.5% |

The 16-case request still contains every comparison row. The previous complete
75-field catalog remains available to Registry consumers; it is no longer
model-bound context. The audit's catalog reference measurements therefore do
not represent current transmitted overhead. `analysis-request.json` shows the
actual selected request.

Evidence is in `reports/verification/postprocessing-fix-2026-09-17/`;
the original baseline remains in `webui-context-audit-2026-09-17/`.

## Verification And Limits

Thirteen new regressions cover legacy/current eligibility, shared browser
projection and stale responses, actual nested TaskSpec units, mixed-unit
rejection, duplicate coordinates in both row orders, invalid explicit options,
visible exclusions, agreement between plotted and exported data, preflight and
output collisions, relevant output/artifact contracts, language-independent
feedback and optional feedback failures. Existing real molecular dissociation
and Matplotlib artifact tests also exercise the changed boundary.

The final full suite ran 1,138 tests: 1,081 passed and 57 skipped, with no
failures or errors. Python 3.9 passed 111 focused tests; a separate 18-test
browser/postprocessing run passed. Both changed browser assets passed syntax
checks, and `git diff --check` passed. Totals are recorded in the accompanying
`verification.json`.
Wiki sources, curated guidance and the 43-page runtime export were updated;
lint reports no errors and three pre-existing citation warnings.

Large Study tables still grow with case count: this change removes irrelevant
contract overhead without introducing lossy row sampling or an additional
recovery framework. Scoped large-result analysis and historical provider-usage
aggregation remain separate work. Optional root-RDM loading and other numerical
diagnostic evidence paths outside postprocessing were not changed.

The changes are local. A running Python WebUI service must be restarted and
its page refreshed to load the updated backend and browser code. No production
service restart, remote deployment or GitHub push was performed.
