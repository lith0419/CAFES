# Study MCP Submission And Collection

Inspection date: 2026-09-06. Local continuation of the
[single-task MCP implementation](2026-09-06-mcp-single-task-interface.md).
The source tree also contains the earlier uncommitted Study/Task/Run changes.

> Historical implementation record. The per-task submission path described
> below was replaced by the [agent interface cleanup](2026-09-07-mcp-agent-interface.md).
> Use the current MCP guide for client behavior.

## Problem And Result

The existing Study Run operation waits for numerical work. Calling it directly
from an MCP tool would hold a protocol request open and make reconnects awkward.
The application service now offers nonblocking submission using the same Study
checkpoint, task selection, retry counters and execution receipts as Run.

The stdio server exposes eleven tools: the original six single-task tools plus
`prepare_study`, `submit_study`, `get_study_status`, `collect_study` and
`review_study`. Resources provide the saved full StudyPlan and StudyReport.
The adapter owns transport fields, projections and error presentation;
application services and executors continue to own execution policy.

## Execution And Persisted Evidence

Preparation validates and saves a new static plan and its cost estimate without
running or approving calculations. Submission shares Run's selection logic and
returns after persisting handles. Repeating it while any current run is pending
returns the existing execution without polling, waiting or resubmitting.

LocalProcessExecutor advances one ready task per call, preserving local serial
execution without adding a background scheduler. Recoverable Slurm executors
retain their independent-batch submission. After collection, the caller submits
the next ready task or batch explicitly. Projected-1RDM continuation resolves
the preceding task's artifact using the existing dependency logic.

The full plan, `study-state.json` and existing execution receipts remain the
only persisted Study execution records. Receipts retain their v1 schema and
Slurm `batches`, with an additive `jobs` mapping for local case-to-JobHandle
associations. Local runs do not impersonate Slurm batch IDs. Submitted and
collected results share the same case/run acceptance before checkpoint updates.
Historical batch receipts remain readable; this change requires no bulk data
migration or run-directory moves.

Collection never submits unstarted tasks or retries failures. It rebuilds the
full task view, leaves unstarted tasks as `not_executed`, and preserves adaptive
and postprocessing extensions. A not-ready collection does not mark an ordinary
running job as a broken connection. Status includes current task status counts,
so readiness to collect submitted runs is distinct from full Study completion.

LocalProcessExecutor now produces its existing failure report in `fetch`, using
the saved request when a terminal worker has no numerical report. Timeouts and
cancellations can therefore close their reserved attempt after a client or
executor restart. Synchronous execution uses this same path.

## Review And Retries

Review loads the saved plan/report and calls the existing review-action service.
It returns a draft without executing or saving it. The caller explicitly passes
that prepared plan to submit; submission persists its changed requests in the
full plan. A client retains the draft until then. Approval actions still require
the user's decision, and submission retains resource and scientific gates.

Subset review keeps the Study and case IDs. Selected tasks receive new run IDs
and attempt counts, while other task results remain intact. Ordinary submission
reuses successes and applies the existing failed/unconverged retry limit of two
total attempts. Explicit case/status selections retain their existing override
rules. No MCP-specific retry loop or result database was introduced.

## Verification

- Python 3.12.14, MCP SDK 2.1.1 and PySCF 2.13.1: 108 focused tests passed.
  Coverage includes existing task MCP, local process control, Study retry and
  result acceptance, application services, and 17 new Study tests.
- Real stdio acceptance performs a two-case H2 basis comparison across server
  restarts, partial collection, reviewed single-case retry, preservation of
  the unselected result and repeated collection without resubmission. Final
  counts are two task IDs, three run records and attempts `[2, 1]`. The STO-3G
  HF energy is `-1.1167593073964255` Hartree. The final Study report file is
  written after the local executor TaskReports.
- An independently installed wheel passed the same prepare/submit/restart/
  collect/review/subset workflow through its installed console command from
  outside the source checkout. Eleven tools and all 43 packaged wiki pages
  are available. Numerical artifacts remain under
  `runs/mcp-study-acceptance-20260906/`; the verification JSON records the exact
  package path, versions, wheel checksum and final report path.
- Wiki regeneration retains 43 pages. Lint reports zero errors and the same
  three pre-existing citation warnings.
- The existing Python 3.9.6 source environment ran 1,009 tests: 991 passed and
  18 skipped (17 optional MCP checks and the pre-existing opt-in benchmark).
  This is source-regression evidence; Python 3.9 is below the declared package
  minimum. Supported-version installation evidence comes from the Python 3.12
  focused suite and the installed-wheel acceptance above.

Evidence files use `reports/verification/mcp-study-2026-09-06-*`: focused tests,
source regression, installed-wheel acceptance and wiki lint. See the
[MCP guide](../guides/mcp.md) for tool arguments and client sequencing.

## Remaining Boundaries

Automatic adaptive-stage orchestration, dedicated scientific continuation
tools, binary artifact transfer, Streamable HTTP and MCP Tasks are outside this
increment. Advanced review actions without an executable prepared plan still
use their existing adaptive application workflow.

Unknown submissions remain unknown until existing executor evidence is
reconciled. Remote reconciliation is available through the existing Python
service but is not a new MCP tool. Same-process operations share the existing
Study lock; separate server processes should not mutate the same Study root.

No Git commit, GitHub publication or Amarel submission was performed. Remote
batch routing and recovery are covered by contract fixtures, not new cluster
acceptance. The earlier supported-environment full-suite limitations involving
optional libDMET fixtures and block2 remain documented in the single-task
implementation record; this increment does not claim that suite is green.
