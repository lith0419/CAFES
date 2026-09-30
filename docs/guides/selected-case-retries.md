# Retrying saved Study cases

A retry is an operation on a saved Study, represented by `RetryAction`. It does
not create a StudySpec or replace the full scan. The complete saved plan and
report remain the source for result tables and plots.

## Using the planner

Open a saved Study, choose **Retry this point** on a result row, choose a
parameter and value, and select **Preview Retry**. The preview lists the exact
execution scope, parameter differences, conditional dependencies, and resource
estimate. **Confirm Retry** submits that saved action. When cost approval is
needed, the button explicitly includes it; a hard resource limit cannot be
approved away.

Deferred dependencies are optional: select the checkbox to also update points
that consume new parent results. Fixed artifact references remain pinned to
their historical Run and do not create a dependency cascade. A failed parent
blocks its dependent cases with a reason instead of submitting them blindly.

The saved-Study conversation uses intent routing before StudySpec drafting.
For example, `retry case-0003 with max_iterations 100` produces a retry preview
for that DMET case. The model may suggest parameter changes, but case IDs are
resolved by code from explicit IDs, selected IDs supplied by the client, or
saved failed/unconverged statuses. Ambiguous or empty selections require
clarification and never fall through to a full scan.

## Application boundary

`StudyApplicationService.retry_context(study_id, work_dir=...)` returns the
saved revision. Pass that revision to `prepare_retry`:

```json
{
  "schema": "pyscf-agent.retry-action.v1",
  "study_id": "saved-study-id",
  "case_ids": ["case-0003"],
  "case_overrides": {
    "case-0003": {"solver.options.max_iterations": 100}
  },
  "reason": "Increase DMET outer iteration limit",
  "include_dependents": false,
  "base_study_fingerprint": "revision returned by retry_context"
}
```

`prepare_retry` validates the request, derives a subset from the authoritative
saved plan, and evaluates it with the executor's case-context resolver. It
returns a persisted preview with `action_id`, `execution_case_ids`, `contexts`,
`changes`, and `cost_estimate`. `start_retry(study_id, action_id, ...)` launches
only that saved action. The HTTP equivalents are `/api/study-retry-context`,
`/api/study-retry-prepare`, and `/api/study-retry-start`.

Overrides are request paths restricted to numerical methods and settings:
`runtime`, `solver`, `method`, `xc`, `active_space`, and `post_cas`. They cannot
change shared `base_task`, sweep design, geometry, Hamiltonian, case identity,
or another case. Existing structured DMET/SCF/FCI recovery buttons pass through
this same boundary.

## Guarantees and recovery

- Empty/duplicate/unknown IDs and overrides outside declared IDs are rejected.
- Plan and checkpoint revisions are rechecked at launch and inside the executor
  before mutations. Changed case fields are included in stale-preview errors.
- Execution cannot exceed the previewed selected cases and explicitly included
  deferred descendants. Dependency failures can reduce the submitted set; their
  blocked results explain why.
- An action ledger under `retry-actions/` prevents duplicate UI submissions and
  duplicate worker invocations. An uncertain launch is not automatically replayed.
  Inspect/collect any pending execution, then prepare a new action if necessary.
- A failed retry retains other cases and their plotting data. Existing per-Run
  artifacts preserve earlier attempts.
- Frontend `plannerState.pendingAction` is separate from the full plan and report.
  Full-plan assignments use `installFullStudyPlan`; old session variables remain
  compatibility accessors. A reviewed subset cannot replace the complete plan.

Local regression tests: `test_retry_actions`, `test_retry_action_ui`,
`test_retry_collection`, `test_study_task_runs`, `test_saved_study_ui`, and
`test_study_background` under `tests/computational_study_agent`.
