# Local Process Control And Retry Acceptance

Inspection date: 2026-09-06. These are local working-tree changes on top of
`33c7610` and the preceding retry/StudyReport fixes. An isolated Amarel source
release has been deployed; the default remote profile is not replaced.

## Problem And Result

Local Web jobs previously depended on the submitting process's in-memory
Popen objects. After restart, a cancellation could change persisted status
without controlling the original numerical process. The termination grace
period also provided no independent wall-time limit.

Each new local job now starts an independent POSIX supervisor. Existing run
directories retain the request, status, worker logs, supervisor identity,
calculation identity, and cancellation request. Identity includes PID, process
group, start time, and a per-run command token. Reconnected clients verify this
identity before controlling a job; mismatches are explicit errors.

The supervisor alone owns normal status transitions. It stops the numerical
process group, escalates from TERM to KILL when necessary, and confirms group
termination before publishing cancellation or timeout. A dead supervisor's
verified live calculation can be cleaned up by status inspection. Unverifiable
remaining descendants do not become a falsely confirmed terminal result.
Launch failure is persisted as failure. Normal TaskReports remain fetchable
after submitting clients exit.

`wall_time_seconds` is optional, finite, and positive. It is saved at submission
and measured by the supervisor, independently of Web/client polling. Configure
it through `--local-wall-time-seconds` or
`PYSCF_AGENT_LOCAL_WALL_TIME_SECONDS`; this also selects supervised execution
for synchronous local CLI/Study calls. A synchronous timeout produces a failed
TaskReport and closes the Study attempt, so subsequent collection does not
mistake it for unfinished work.

## Submission Reconciliation

The SSH batch endpoint writes server-owned `submission.json` evidence after
receiving all batch handles and before returning the acknowledgement. Request
fingerprints are computed before rewriting attached input paths. The new
read-only `recover-submission` RPC is confined to the configured submission
root.

`StudyApplicationService.reconcile_execution` associates an unknown receipt
only when the plan/executor, original requests, case IDs, pending run IDs, and
receipt paths agree. The operation shares the existing study lock with Run and
Collect. Live acceptance found that queuing two simultaneous explicit Run
requests produced two new attempts. Concurrent mutations now return
`StudyExecutionBusy` instead of waiting to execute again; Collect remains
serialized against checkpoint writes. A later explicit Run still retains its
existing meaning and can exceed the automatic retry limit. Repeating the same association is harmless. It never submits a task,
increments an attempt, or performs collection.

The CLI exposes `--saved-plan`, `--reconcile-submission`, optional
`--execution-receipt`, and `--collect-only`. Recovery actions require the saved
plan to retain study identity. See the [execution guide](../guides/remote-execution.md)
and [Python API guide](../guides/python-api.md).

## Live Findings And Fixes

Real acceptance exposed two gaps that controlled fixtures had not covered:

1. Concurrent explicit Run requests were serialized and each created a fresh
   attempt. Same-study concurrent mutations now return `StudyExecutionBusy`.
   The original automatic retry limit and later explicit rerun behavior remain
   unchanged. The failed pre-fix check and extra attempt are retained in
   `retry-live-2026-09-06-duplicate-retry-before-fix.json`.
2. Slurm could report `COMPLETED` before its TaskReport became visible on the
   shared filesystem. Collection treated every terminal scheduler state as
   permission to synthesize a failure, then cached that premature result.
   Inspection now requires a visible report for completed jobs. Slurm collection
   leaves them pending and only synthesizes reports for scheduler failure or
   cancellation. The failure and affected acceptance checkpoint are preserved
   in `retry-live-2026-09-06-delayed-report-before-fix.json`. Only this known
   acceptance checkpoint and its recorded pre-collection lifecycle were
   restored to collect its original successful job;
   no replacement job was submitted. Existing user runs are not bulk rewritten; a prematurely collected failure
   from an older version requires explicit evidence-based repair.

The harness also initially intercepted the wrong wait method. That error was
corrected, and the initial real jobs were reused. Its original output is kept
in `retry-live-2026-09-06-initial-harness.json`; it is not presented as a
production executor defect.

## Compatibility And Boundaries

- Existing JobHandle, JobStatus, TaskReport, StudyReport, and study receipt
  schemas remain unchanged. The new records are executor bookkeeping.
- The current-process LocalExecutor stays synchronous without hard limits.
  No default scientific timeout is invented.
- Old local jobs without verified identity remain inspectable; cancellation
  after reconnect cannot claim success without proof of process termination.
- Process-group control requires POSIX and process-inspection permission. A
  deliberate new session, host reboot, or total loss of process evidence needs
  separate reconciliation. This is not a host-wide process-management service.
- Server evidence requires the updated endpoint. Old submissions and failures
  before all server handles are persisted remain unknown. Explicit evidence
  selection is required; no automatic submission search or idempotency service
  is introduced.
- Study locking remains within one Python service process. Concurrent Run requests are rejected as busy in that service; multi-process Study execution ownership remains deferred.

## Verification

Real-process regressions cover submitter exit, restart cancellation, descendants
that ignore TERM, supervisor crash, identity mismatch, timeout without polling,
normal completion, and synchronous Study timeout. Request/reconciliation tests
also reject wrong fingerprints, different attempts, and paths outside the
study/server roots while permitting repeated recovery and collection.

The local H2/HF/STO-3G smoke calculation succeeded with energy
`-1.1167593073964255` hartree; its TaskReport is archived as
`reports/verification/local-h2-2026-09-06.json`. Regression and audit outputs use
the `reports/verification/local-retry-2026-09-06-*` prefix. The final suite ran
932 tests in 73.892 seconds: 931 passed and one opt-in benchmark was skipped.
The focused suite passed 42 tests; Registry audit reports 317 entries and zero
validation issues. Wiki curation generated 43 pages; lint reports zero errors
and the three existing inferred-citation warnings. The source-tree runtime is Python 3.9.6, below
the declared >=3.10 support floor; this does not replace supported-wheel tests.

SSH/RPC preflight against Amarel passed and is archived as
`reports/verification/retry-live-2026-09-06-preflight.json`. After explicit upload authorization, the frozen source was deployed to
`retry-acceptance-20260906-20260906-223104z-cda6ad4871`, then updated after the
concurrent-Run finding to `retry-acceptance-20260906-20260906-223922z-0444cb55c8`,
then updated for the report-visibility fix to
`retry-acceptance-20260906-20260906-224720z-ec36780431`, with matching
frozen-source/server runtime identities. The live campaign passed; results are recorded in
`reports/verification/retry-live-2026-09-06-campaign.json`.

`tools/verify_retry_lifecycle.py` is an opt-in acceptance runner for at most ten
tiny H2 tasks in an isolated environment. The completed campaign verified real mixed
success/cancellation, concurrent Collect, selected retry, lost acknowledgement,
explicit reconciliation, sequential reconnect, and adaptive collection. Faults
are injected at the client transport/wait boundary around real Slurm calls;
they must not be described as a physical network outage. It archives returned
handles before injecting failures and only cancels its own unfinished jobs on
failure. The campaign completed with 10 tasks across 8 submissions and all
11 final checks passing. No recorded job remains in the Slurm queue. Nine
launched tasks have `COMPLETED/0:0` accounting; the first array member was
intentionally cancelled before numerical execution.

| Scenario | Slurm submission IDs | Result |
| --- | --- | --- |
| Mixed success/cancellation and concurrent Collect | `61268802` | One cancelled case, two successful cases; no retry during Collect. |
| Concurrent retry before and after the fix | `61268831`, `61268835`, `61268867` | The first pair exposed the duplicate; after the fix only one concurrent request submitted. |
| Lost acknowledgement and explicit recovery | `61268871` | Original job recovered and collected; attempt remains 1. |
| Sequential reconnect and explicit continuation | `61269835`, `61269838` | Collect did not submit the next case; subsequent Run did. |
| Adaptive initial-stage collection | `61269843` | Refined plan prepared without a refined submission. |

The sequential and adaptive collection phases both observed completed jobs
whose reports were not yet visible. Repeated Collect waited for those original
reports and succeeded without synthesizing failures or resubmitting.
CLI `--collect-only` and repeated `--reconcile-submission` also passed. Recovery
must retain the executed plan's resource policy: the CLI rejected a fresh
unbound plan, then accepted the original policy after exact receipt-fingerprint
verification. The harness now loads the saved executed plan when resuming.

The compact result is `reports/verification/retry-live-2026-09-06-summary.json`;
the full campaign, pre-fix findings, deployment identity, CLI checks, queue
cleanup, and logs are adjacent. The final runtime Python source matches the
deployed snapshot. Subsequent local changes update documentation, generated
wiki text, acceptance evidence, and the verification harness only.

The remote binding is isolated at
`runs/verification/retry-binding-2026-09-06/remote.ini`; the project's default
remote profile is unchanged. This verifies small-task lifecycle behavior, not
large DMRG/FCI jobs, complete MD datasets, physical network outages, or
multi-process Study ownership.
