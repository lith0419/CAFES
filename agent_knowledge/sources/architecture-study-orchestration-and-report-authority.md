# Study Orchestration And Report Authority

## Scope

The study layer coordinates complete `TaskSpec` executions. It owns task
relationships, approvals, cross-case analysis, refinement, and report updates.
Providers and the trusted task kernel remain responsible for individual
calculations.

## Study, Task, And Run

The task organization is `Study -> Task (case) -> current Run`. A case identifies
a scientific calculation point within its Study. A run identifies one execution
attempt and its artifacts. Retrying a selected case preserves the Study ID and
case ID, increments its attempt count, and creates a distinct run directory.
Several tasks may share a submitted batch and receipt; that grouping does not
create a child Study.

The September 6 correction removes independent IDs for review/retry subsets,
parent registration, exported task-to-child references, and the separate
`integrate_review_execution` operation. Approved subset plans retain their
Study identity. Run and Collect directly return the complete Study report.

## Persisted Responsibilities

- `study-plan.json` retains the complete task set and the latest executed
  task contracts across subset executions.
- `study-state.json` owns the task map, current run/receipt associations,
  attempt counters, TaskReport references, compact task-status summaries, and
  lifecycle. Result files are written before the checkpoint is replaced. Its optional `review` field
  records scientific stage kind and selected case IDs, without another
  execution identity or ownership hierarchy.
- Execution receipts own submitted handles and transport status.
- StudyReport owns the complete comparison, scientific analysis, and artifact
  manifest. Existing adaptive and postprocessing extensions survive updates.
- Browser state is a presentation cache; it does not replace persisted results.

Status describes scheduler progress, TaskReport describes the scientific
outcome, and gates describe review requirements. These roles remain distinct.
There is no new database, report-revision protocol, or additional workflow-state
framework in this correction.

New checkpoints keep collected reports in `case-reports/`, addressed by
`task_report_ref`, rather than embedding whole reports in `study-state.json`.
`load_checkpoint(..., include_reports=False)` reads status without opening
reports or numerical arrays. Collection loads report JSON explicitly; arrays
remain file references. Legacy inline checkpoints are still readable. Existing
inline Study input fingerprints remain unchanged; historical data is not
silently migrated or resubmitted. StudyReport retains its public case/result
view for review and analysis.

## Execution And Recovery

`StudyApplicationService` separates Run, inspection, and collection. Run can
execute an approved subset or an explicit case/status selection. It preserves
other task results and contracts, including when the caller supplies an older
full plan. Run first recovers pending executions across the Study; replaying an
interrupted invocation does not automatically launch another attempt.

Background Start calls the existing static/adaptive application runner once.
The runner owns all task and scientific stage sequencing until completion,
required review or interruption. MCP callers do not submit subsequent tasks
manually. The earlier submit-only executor branch has been removed. Existing
receipts store scheduler batches or local case-to-JobHandle associations.

MCP prepares static/adaptive inputs, starts, inspects, collects and prepares
review actions through StudyApplicationService. Direct-CAS preparation and
probe-to-review conversion are shared with Web. Review saves a decision in StudyReport.pending_review; callers explicitly
start it using the Study ID. Study ID and execution
root are sufficient for post-restart status and collection.

study-invocation.json is the latest background application call and completion
metadata. A coordinator-held POSIX file lock prevents overlapping background
starts/reviews/collections across MCP processes, surviving client exit. It is
not another task state machine. Direct Python/Web Run retains its existing
locking and should not overlap a background writer in the same Study.

Collect can use the Study ID and execution root alone. It loads the persisted
full plan, collects current saved handles, and rebuilds the full report without
submitting failed or unstarted tasks. Adaptive collection can prepare a later
stage, but Run must start it separately. A completed scheduler job without a
visible TaskReport remains pending collection.

Remote collection and direct execution share result acceptance. The complete
expected case set and current run identities are checked before updating any
case checkpoint. Receipts become collected after result acceptance. Missing or
corrupt state and unknown submission acknowledgements cannot be interpreted as
an empty new execution. Explicit submission reconciliation uses existing
server-side evidence and never submits work.

## Report Updates

The full comparison is derived from current checkpoint TaskReports. Cached
report numbers and status are not execution results. Unchanged scientific
annotations are preserved while numerical fields are refreshed. Narrower retry
plans do not remove the full plan's observables or unrelated comparison rows.
Report updates preserve adaptive decisions, postprocessing evidence, and other
accepted additive fields. Stage result snapshots, such as `recovery_report`,
remain scientific evidence within the same Study and do not own separate tasks.

The complete report is marked successful only when all current task outcomes
and comparison rows succeed. Pending execution marks an existing report as
awaiting updated results. Final reports and checkpoints use atomic writes, and
the final report is written after current task results. Same-process Run/Collect
operations share the Study lock. Background Start/review/Collect additionally
share the coordinator process lock across MCP servers; direct Python/Web Run
must not overlap that writer, as described above.

## Historical Data

Historical reports remain readable. Completed legacy child/stage checkpoint
records can be imported into the original Study checkpoint when it is next
loaded. Existing run directories and artifacts are retained; this does not
submit or copy calculations. Pending or unresolved historical child execution
must be collected before migration; it is not silently replaced by a new run.
A browser-only historical numerical report cannot seed execution state.

This correction changes the review/retry task organization. Specialized adaptive
initial/refined stages, MPS continuation, and bidirectional candidate evaluation
retain their scientific orchestration and existing stage artifacts. They have
not all been migrated to a universal stage engine. Typed stage adapters may be
introduced independently when useful; a revision engine is not a prerequisite
for ordinary task retries.

## Related Pages

- [[Current Agent Architecture]]
- [[Public Data Contracts]]
- [[Design Philosophy And Layer Boundaries]]
- [[Adaptive Scan Workflow State Machine]]

## Saved Review, Analysis And Dataset Calls

StudyReport.pending_review contains the latest selected action, plan, workflow,
case IDs and any existing continuation approval. Review never submits work;
Start uses this saved decision by Study ID. The worker consumes it after its
application call returns. Failed launches and interrupted invocations retain
it; ordinary receipts and Task/Run resume rules still own recovery.

The agent's diagnose_results performs scientific analysis without an LLM;
analyze_results optionally adds language interpretation. MCP analyze_study
loads/saves by Study ID and proposes continuation without executing it. An
approved MPS/path continuation is a background call to the same existing
application use case. Postprocessing uses the existing plotting service.

prepare_dataset accepts the existing dataset/geometry contracts and validates
through the existing Task service. It saves an ordinary MD Study. The runner
assembles accepted/rejected samples and the dataset manifest; MCP only exposes
that result. Runtime-supported MD methods remain authoritative even for custom
dataset specifications.
