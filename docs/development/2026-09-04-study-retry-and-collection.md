# Study Retry, Collection, And Migration Status

Inspection date: 2026-09-04. This describes local working-tree changes on top of
`33c7610`, including the preceding report-merge, dataset-generation, and local
run-id fixes. It is not a deployed remote release.

## Problem And Result

Previously, repeating a failed remote study could increment `attempt_count`
while collecting the same failed batch again. Conversely, the Web Collect
endpoint called the execution runner, so collecting a mixed successful/failed
batch could submit the failed subset. An unreadable checkpoint silently became
an empty state and could execute already completed cases again.

The application service now separates `inspect_execution`, `collect_study` /
`collect_adaptive_study`, and Run. Collect fetches persisted handles or uses
checkpointed TaskReports, then rebuilds the report. It never invokes submission,
local execution, or automatic scientific recovery. An unfinished adaptive next
stage may be prepared as a plan, but execution waits for a separate Run action
and existing gates still determine any approval requirements.

Run retains bounded automatic case retry. Explicit case/status selection limits
execution to that subset and retains the other case reports, including failures.
Successful cases remain reusable. A new execution receives a new run directory;
its predecessor's TaskReport and numerical artifacts remain available. Resuming
a pending remote execution reuses its handles even when the caller repeats
`resume=False`.

## Persistence And Compatibility

The existing `study-state.v1` case checkpoint gains an additive `execution`
association containing `run_id`, `receipt_path`, `pending`, and `attempt_count`.
The state's `execution_associations_version=1` distinguishes new records from
legacy records that did not persist these associations. This is execution
bookkeeping, not another approval or lifecycle state machine.

For a new remote attempt, the coordinator first persists its association and a
submission-intent receipt. Acknowledged handles are then written atomically,
and the case attempt count is recorded before waiting. If acknowledgement is
lost, the receipt remains `submission_unknown`: the reserved attempt is visible,
but an unconfirmed submission does not count as a confirmed new attempt. The
coordinator refuses to submit again until the existing job is reconciled.
Local synchronous invocation records its attempt before calling the executor.
Task-internal numerical retries keep their existing Task-level counters.

Collection validates complete TaskReports and matching run identities before
marking a receipt collected. Status inspection does not modify receipts, reopen
collected jobs, or count superseded attempts as current work. Concurrent Run and
Collect requests in one process serialize by study directory and identity.

Compatible legacy v1 checkpoints remain readable. When a legacy checkpoint
lacks execution associations, matching persisted batch handles can be adopted
without resubmission. Existing completed case counts are retained. An invalid
checkpoint, missing association in a new-format checkpoint, corrupt receipt,
changed pending request, or conflicting executor is an explicit error; it does
not silently become a new run. There is no bulk rewrite of users' saved runs.

## Migration Status

Follow-up on 2026-09-05: the public report conversion gap recorded below has
been fixed. StudyReport now preserves adaptive/postprocessing and additive
fields across both existing schemas. See the current
[Python API guide](../guides/python-api.md#study-report-round-trips).

| Area | Current state | Remaining boundary |
| --- | --- | --- |
| Registry runtime | Migrated: 317 typed entries, zero validation issues, runtime family consumers, compatibility call sites, or legacy public payload keys. | Source-family construction helpers are internal cleanup, not a second runtime catalog. |
| Retry and collection use cases | Implemented for static and adaptive service/Web paths; shared task collection and checkpoint integration. | No live-cluster acceptance run was performed here. |
| Existing execution persistence | Additive case associations, fresh retry run directories, conservative legacy receipt adoption, and explicit unknown-submission handling. | Missing/corrupt authoritative records require reconciliation. Local synchronous process death without a recoverable handle is not automatically repaired. |
| Study report merge | The prior local fix loads the latest parent under a process-local lock, merges by case ID, and requires all cases to succeed before claiming success. | Cross-process ownership and consumed-action deduplication are not implemented. |
| Public report contract | Follow-up on September 5 preserves case associations and top-level adaptive/postprocessing/additive evidence across both existing schemas. | Typed review and stage contracts remain migration work. |
| Study orchestration | Existing Registry/Gates and procedural orchestrators remain the execution owners. | Typed stage adapters, shared `StudyWorkflowState`, and `ReviewExecutionContext` remain unimplemented. |
| Review/UI authority | Collect now uses an explicit backend use case; the UI receives the prepared next plan. | Parent-report transport and private plan-kind fields remain; no monotonic backend revision history was added. |

## Next Work

1. Completed on September 5: the common report conversion preserves adaptive,
   postprocessing, review, and continuation evidence, with migration fixtures.
2. Exercise interrupted submission, collection, selected retry, and cancellation
   on the real execution target. Reconcile the ambiguous-acknowledgement case
   before considering an executor-owned idempotent submission facility.
3. Move one Study stage at a time behind typed inputs/outputs after those
   persistence and execution boundaries have acceptance evidence. Keep the
   current small merge rule; broad report repositories or an additional retry
   workflow are not prerequisites for this fix.

## Verification

Fourteen new regressions exercise the public service and Web collection path,
remote mixed outcomes, selected subsets, independent and sequential interrupted
execution, repeat/concurrent collection, local retry directories and limits,
legacy receipt adoption, corrupt/missing state, unknown submission, incomplete
or misidentified reports, and adaptive refinement/recovery collection.

The final suite ran 910 tests in 64.476 seconds: 909 passed and one opt-in
numerical benchmark was skipped. Registry audit found 317 entries and no
migration/validation issues. Wiki curation generated 43 pages; lint reported
zero errors and the same three pre-existing inferred-citation warnings.
`git diff --check` passed. Python was 3.9.6, below the declared >=3.10 support
floor, so this remains source-tree regression evidence.

The final full-suite, Registry audit, and wiki validation outputs are archived
under `reports/verification/` with the `study-retry-2026-09-04` prefix. These tests
combine local numerical cases and controlled executor/transport fixtures; they
do not establish supported-Python wheel acceptance or new live-cluster evidence.
