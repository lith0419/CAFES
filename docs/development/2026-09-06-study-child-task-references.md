# Study Child-Task References And Current-Result Merging

Superseded by the [Study/Task/Run correction](2026-09-06-study-task-run-organization.md).
This record and its verification describe the earlier local implementation;
independent review child Studies and the separate merge API have been removed.

Inspection date: 2026-09-06. Local changes on top of `48a0478` and the
DMET option/quality validation follow-up.

Subsequent local follow-up: the [task-management simplification](2026-09-06-study-task-management-simplification.md)
replaces full-report integration arguments with an execution ID, consolidates
result acceptance, and removes the cached-child-report identity comparisons.
The rules below describe the original reference implementation and its archived
verification; use the Python API guide for current calls.

## Problem And Result

Reloading the latest parent StudyReport preserved updates to different cases,
but still allowed a late result from an older attempt to replace the current
result for the same case. A result containing an unexpected case could also
append it to the parent. The repair uses the existing Study/case hierarchy and
execution checkpoint as the source of current results.

```text
parent Study
  -> case_id + task_reference {study_id, checkpoint}
     -> existing child study-state.json / cases[case_id]
        -> execution.run_id + receipt_path + TaskReport
```

Ordinary StudyReports export each case's task reference. Review/retry execution
registers the selected cases with the parent before submission and saves the
reverse `parent_study {study_id, report_path}` link in the child's existing
checkpoint. Run can restore this link after a restart. Collect only reads and
collects the existing execution; it never rebinds an obsolete child.
If another child has replaced one of its parent references, Run requires a new
review subset too. An old full plan cannot reclaim an unselected case while
retrying another case.

Prepared review subsets now get separate child Study IDs. This also prevents a
static retry from writing its subset report over the original parent's report
file. Existing stable case IDs, run directories, execution receipts, scientific
gates, and cost approvals retain their roles.

## Integration Rules

- Reload the current parent report under the shared process-local lock.
- Require unique case IDs and the same case set in the submitted child plan,
  child report, and comparison rows. Selected cases must belong to the parent.
- Follow the parent's selected child checkpoint and verify child identity,
  persisted parent ownership when present, and TaskReport/current-run identity.
  A caller's cached numerical values and attempt counters are not merge inputs.
- Require the persisted child StudyReport to reflect the same runs, then use
  its current artifacts and the checkpoint TaskReports to rebuild the subset.
  A pending current run cannot be replaced with its previous TaskReport.
- Preserve other parent cases and additive report extensions. An old attempt
  replay uses the current child result; a report from a replaced child is
  rejected without changing the parent.
- Mark the parent successful only after every case succeeds and all referenced
  snapshots match their current executions. Registration marks the parent as
  awaiting updated results, retaining its previous numerical snapshot. Other
  pending children prevent premature success when one subset finishes first.
- Atomically write the final parent after the child TaskReports and StudyReport.
  Partial sequential collection retains `not_executed` cases without starting
  them, so it produces an incomplete parent rather than a false success.

## Compatibility And Scope

These are additive fields in existing v1 reports/checkpoints. The current run
counter is not duplicated in a new Study state or report revision. The
application service and Web Run/Collect review path use the same integration
boundary. Base reports with additive adaptive evidence still use their base
report filename; adaptive-schema parents use the adaptive filename.

Old unbound in-memory reports remain compatible when their plan/report case
identities match and they do not replace an identified task with another run.
Old checkpoint-backed child reports without parent references are not silently
adopted during merge. New executions must register through
`run_study(..., parent_report=...)`. No bulk migration or automatic rerun of
historical calculations is performed.

Scientific selection in adaptive recovery, MPS continuation, and path restarts
is unchanged. Their copied case records retain references exported by ordinary
Study execution, but this change does not centralize every specialized writer
or introduce typed stage adapters, StudyWorkflowState, or backend revision
history. Locks coordinate one Python process only. Existing plots and attached
postprocessing evidence are preserved; automatic plot regeneration is separate.

## Verification

Regression tests exercise old-attempt replay, replacement children, pending
retries, service restart and collection, malformed/foreign cases, corrupt or
mismatched checkpoints, independent case updates, direct-review scaffolds,
base/adaptive report filenames, separate child directories, ordinary case
reference export, partial sequential collection, and completion while another
child is pending, including attempts to rerun a replaced full plan. Controlled
executor fixtures use actual local checkpoints and reports; they do not submit
cluster calculations.

The focused suite passed 84 tests. The final full suite ran 977 tests in 103.173
seconds: 976 passed and one opt-in numerical benchmark was skipped. Wiki
curation generated 43 pages; lint reports zero errors and the same three
citation warnings. `git diff --check` passed, and all 28 local links in the
updated documentation entry points resolved. Verification outputs use the
`reports/verification/study-task-references-2026-09-06-*` prefix.

The source-tree runtime is Python 3.9.6, below the declared supported minimum
of Python 3.10. These checks do not establish supported-version wheel acceptance
or a new live-cluster campaign. Changes remain local and have not been committed
or pushed as part of this follow-up.
