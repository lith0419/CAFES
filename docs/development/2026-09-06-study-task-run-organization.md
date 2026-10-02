# Study, Task, And Run Organization

Inspection date: 2026-09-06. Local correction following the task-management
simplification and the user's rejection of independent review child Studies.

## Problem And Result

The interim implementation gave each approved retry subset a separate Study ID,
registered it against parent cases, and integrated its report in a second API
call. The executor already supported task retries within one Study. Reusing a
Study as a retry container added unnecessary ownership and merging rules.

Review/retry organization is now `Study -> Task (case) -> current Run`.
Prepared review plans retain the original Study ID and selected case IDs.
A retry changes the selected tasks' run IDs and increments their counters in
that Study's existing checkpoint. Old run directories and artifacts remain.
Batch receipts group scheduler submissions without another scientific owner.

## Execution And Reports

Run executes the selected plan or explicit case/status selection. The saved
`study-plan.json` remains complete, including updated task requests and the
union of required observables. A stale full plan used to retry another task
cannot restore an unselected task's earlier request.

The existing checkpoint directly holds every task's current run association
and TaskReport. The executor rebuilds the full report and comparison TSV from
those records. Cached report values cannot override current task results or
introduce extra tasks. Unselected results, unchanged annotations, scientific
analysis, adaptive extensions, and postprocessing evidence are preserved.

The separate integration API, child registration, task-to-child references, and
parent links are removed from the review execution path. Scientific stage
snapshots such as `recovery_report` remain evidence attached to the same Study;
they do not own task execution. Checkpoint `review {kind, case_ids}` metadata
retains the stage context across collection without introducing a run group ID.

Existing reports are marked awaiting results before execution. Final success
requires successful current TaskReports and comparison rows for all tasks;
the complete report is atomically written after task results. Full-plan report
construction prevents a subset report from overwriting the complete task set.

## Recovery And API Changes

Both Python and Web Run/Collect return the full Study report directly. Python
`collect_study(study_id, work_dir=...)` reloads the saved plan; the existing
plan-object input remains supported. Web Collect no longer requires a plan or
report snapshot. A review scaffold can enter Run as `study_report=`, with the
same identity as the execution plan. Existing saved results take precedence.
`parent_report=` and `integrate_review_execution()` are removed.

Run first recovers pending runs across the Study. Repeating an interrupted Run
does not automatically create another attempt. Collect never submits tasks,
including unstarted sequential tasks; a later Run can execute those tasks.
Remote and direct execution retain the shared complete-batch acceptance
boundary and existing unknown-submission recovery. Same-process Run/Collect
mutations share a Study lock; cross-process coordination is unchanged.

## Historical Data And Remaining Scope

Completed legacy child/stage checkpoint records can be imported when the
original Study is next loaded, retaining TaskReports, run identities, counters,
and artifact paths. Subsequent retries create runs under the original Study.
This is a loading migration, not a bulk rewrite or a calculation. Pending
historical child execution must be collected before migration; this version
does not resume an old child as a new Study. Browser-only historical numerical
reports remain readable but cannot initialize execution state.

Specialized adaptive initial/refined stage execution, MPS continuation, and
bidirectional candidate evaluation retain their scientific orchestration and
stage artifacts. This change removes the independent child identity introduced
for review/retry subsets; it does not replace every scientific stage executor.
No database, additional workflow-state framework, or report-revision engine is
introduced.

## Verification

Regression coverage includes same-Study subset execution, accumulated attempts,
preserved historical files, full-plan persistence, changed requests, explicit
unselected contract preservation, restart by Study ID, pending-run recovery,
malformed batch acceptance, stale cached rows, report extensions, complete TSV
output, legacy completed-child loading, partial sequential execution, and Web
Run/Collect. Existing scientific report-policy tests remain; obsolete tests of
child registration and the removed merge API were replaced by task/run tests.

Verification outputs use `reports/verification/study-task-run-2026-09-06-*`.
The focused suite passed 107 tests. Wiki curation generated 43 pages; lint
reported 0 errors and 3 existing citation warnings. Both changed JavaScript
files passed syntax checks, `git diff --check` passed, and all 35 checked local
documentation links resolved. The affected production files have 123 fewer
lines than at the start of this task, including preceding local changes in the
baseline; the source JSON output records the file-by-file comparison.

The full suite ran 976 tests in 105.228 seconds: 975 passed and 1 skipped.
Tests used the available source-tree Python 3.9.6 runtime, below the declared
Python >=3.10 support minimum. This is not a supported-version installed-package
or new remote-cluster acceptance. Changes remain local and have not been
committed or pushed.
