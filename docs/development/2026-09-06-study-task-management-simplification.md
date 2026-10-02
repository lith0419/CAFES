# Study Task-Management Simplification

Superseded by the [Study/Task/Run correction](2026-09-06-study-task-run-organization.md).
This record and its verification describe the earlier local implementation;
independent review child Studies and the separate merge API have been removed.

Inspection date: 2026-09-06. Local follow-up to the child-task reference change.

## Problem And Result

The preceding merge interface received a parent report, a child plan, and a
child report, checked their cached case identities, then replaced most of those
inputs with disk records. It also compared child-report run IDs with checkpoint
run IDs and reread checkpoints while deriving parent completion. Result identity
validation was split between remote collection and checkpoint updates.

Integration now takes `study_id` and `work_dir`. It loads the registered parent,
saved plan, and current case records. The child checkpoint saves `review_kind`
at Run time alongside its existing `parent_study` link, so Web Run/Collect can
restore the parent after restart without a browser-owned parent snapshot.
The Planner no longer caches that snapshot in its recovery context or sends it
with Status/Collect requests. Initial review Run still supplies the parent for
registration, including direct-review scaffolds.

All referenced checkpoints are loaded once per integration. Their current
TaskReports supply case data and comparison rows. Pending executions produce
pending rows instead of reusing an older successful result. An unstarted case
remains `not_executed`. A replaced child cannot select itself again by merging
its result. A pending selected child still requires collection; another pending
child remains visible and prevents overall success.
Unchanged cases outside the selected subset retain their original comparison
rows and annotations. Selected, changed, or missing rows are regenerated using
their own saved execution plan's observables, so a narrower retry plan cannot
remove another case's output columns.

The persisted child StudyReport continues to supply stage evidence and its
artifact manifest. Its cached case values, comparison rows, status, and summary
are not merge inputs; those are rebuilt from current records. Current
TaskReport artifacts are also retained in the parent manifest. Adaptive and
postprocessing evidence stays intact. Existing scientific selection algorithms
in specialized MPS/path/recovery writers remain unchanged.

## Shared Result Acceptance

Remote collection and direct execution both call `_finish_cases` to accept a
result batch. It checks the complete expected case set, result shape, current
run identity, and submitted handles where applicable before updating any case.
Legacy adapters that omit a run ID are normalized against the known submitted
execution here. A malformed later result cannot leave earlier cases marked
complete in the checkpoint. Remote receipts are marked collected after result
acceptance succeeds.

`load_checkpoint` is the common persisted-data boundary for execution and
integration. It validates the checkpoint and normalizes known legacy omissions;
the former duplicate report-identity and second checkpoint traversal helpers
have been removed. Existing legacy receipt adoption remains at execution load,
using saved evidence rather than resubmitting jobs.

## Compatibility And Scope

The three-object `integrate_review_execution(parent, plan, report)` call is
replaced by `integrate_review_execution(child_id, work_dir=...)`. Repository
callers use the new signature. There is no parallel in-memory merge path.
Historical reports remain readable through StudyReport. A registered older
execution without `review_kind` can be identified by its static parent location
or persisted adaptive stage-report identity. Without sufficient saved ownership
or execution context, it remains read-only; no ownership is guessed from a
browser payload and no calculation is launched for migration.

The change reuses child Study containers, the existing checkpoint, execution
receipts, and parent case references. It does not move adaptive parent control
into a new state store. Submission-unknown handling, distinct attempt
directories, atomic persistence, process-local mutual exclusion, and executor
cancellation identity checks retain their existing purposes. Locks do not
coordinate separate Python processes. Attached plots are preserved rather than
automatically regenerated.

## Verification

The focused suite covers real checkpoint execution/collection, old-attempt
replay, replacement children, pending and partial execution, concurrent subset
integration, cached child-report disagreement, restart without a browser parent
report, legacy review-kind loading, preservation of unselected comparison
evidence, and all-or-nothing acceptance of malformed batches in both recoverable
and direct executor paths. Existing report-merging policy tests remain, while
persistence tests now use registered execution fixtures rather than free-form
report arguments.

Verification outputs use the
`reports/verification/study-task-simplification-2026-09-06-*` prefix:

- Focused suite: 101 tests passed.
- Full suite: 983 tests ran in 103.591 seconds; 982 passed and 1 skipped.
- Wiki: 43 pages regenerated; lint reported 0 errors and 3 existing citation
  warnings.
- JavaScript syntax and `git diff --check` passed; 31 local documentation links
  resolved with no missing targets.
- The five affected production files have 57 fewer lines in total than at the
  start of this task, including the preceding local changes in that baseline.
  The file-by-file comparison is recorded in the source JSON output.

Tests ran from the source tree with the available Python 3.9.6. This does not
establish installed-package support on the declared Python >=3.10 versions or
new remote-cluster acceptance. These changes are local and have not been
committed or pushed.
