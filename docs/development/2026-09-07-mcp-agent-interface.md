# Thin MCP Agent Interface

Inspection date: 2026-09-07. This replaces the per-task submission mechanism in
[the preceding Study MCP increment](2026-09-06-mcp-study-interface.md). Earlier
uncommitted Study/Task/Run and scientific validation changes remain in the tree.

## Problem And Result

The first nonblocking Study interface made the client submit each local task
or scheduler wave. That duplicated execution control at the protocol boundary
and required Codex to advance work already owned by the agent.

The MCP tool `submit_study` now calls `StudyApplicationService.start_study`.
One background process invokes the existing `run_study` or `run_adaptive_study`
through completion, required review or interruption. Static task ordering,
adaptive initial scans, diagnostics, method routing, refinement, recovery,
dependencies, cost gates, retries and report merging remain in the existing
application services and runners.

The core `submit_study`, `submit_only` branch, submit-only summary and no-wait
execution branch were deleted, along with tests for manual per-task advancement.
The thin adapter still exposes eleven tools. No scientific task loop was added
to MCP or the background launcher.

## Shared Application Behavior

Preparation saves static plans or the complete adaptive input/options needed to
resume by Study ID. Adaptive initial cost approval updates that saved request
without changing its identity. Changing a prepared adaptive resource profile
requires a new preparation.

Direct CASSCF/CASCI preparation now routes to the existing active-space review
or probe in the application service. Web only presents its response. The same
service converts completed probe results into a saved ActiveSpaceAudit review;
MCP, Web and Python therefore see the same report. Approved subsequent tasks
keep the Study ID and use the existing subset Run/Collect path. Fresh probe/review
preparations also replace the template lifecycle identity with that Study ID.

Real adaptive acceptance exposed a normalization bug: missing orbital-processing
options first normalized to `{}`, then to a populated default object. Repeated
normalization changed the input fingerprint and silently created another Study.
Empty options now remain empty, making normalization idempotent and preserving
resume identity.

## Process And Report Ownership

`study-invocation.json` contains the latest application call, executor connection
configuration, start/finish times and any error. It is registered as an execution
artifact. The detached coordinator holds `.study-invocation.lock` through an
inherited POSIX descriptor; closing the MCP client does not release that lock.
Numerical workers do not inherit it. `study-worker.log` captures coordinator
output without contaminating MCP stdout.

This is process metadata, not another task state machine. Existing Study plans,
checkpoints, TaskReports and execution receipts remain authoritative. Local
case-to-JobHandle receipts and terminal-worker failure reports from the previous
increment are retained because ordinary Run/Collect also use them for recovery.

Repeated start while running creates no new invocation. Status returns
`agent.running`; collection/review waits until the coordinator exits. Collection
then reads or recovers existing results without starting a stage or retry.
After an interruption, explicit start follows the ordinary runner's recovery
rules. Scientific failure and review requirements remain in StudyReport;
`agent.error` describes an application invocation failure.

The background lock coordinates these start/review/collect calls across MCP
servers. It does not extend direct Python/Web Run into a distributed locking
system; callers should avoid overlapping those writers in the same Study.

## Verification

- Python 3.12.14, MCP 2.1.1, PySCF 2.13.1: 192 focused tests passed, including
  real stdio/PySCF acceptance, existing local process cancellation/timeouts,
  application/Web parity, Registry/artifacts, adaptive execution/recovery,
  retry collection and Study task/run behavior.
- A two-case H2 static Study finishes after one start and client exit. Reviewing
  and rerunning one case produces three runs with attempts `[2, 1]`; the other
  case is unchanged. HF/STO-3G energy: `-1.1167593073964255` Hartree.
- A two-case adaptive H2 Study finishes its existing initial and refined stages
  after one start and client exit, with four runs under the same prepared Study.
- Direct H2 CASSCF starts with a probe, returns a saved review, then executes
  CASSCF after explicit case approval. Repeated collection preserves success
  and does not restart the two completed runs.
- A fresh wheel installed outside the checkout passes all three Study workflows
  through the installed console command. Evidence records package path,
  versions, checksum, eleven tools, 43 wiki pages and report paths. Numerical
  artifacts remain in `runs/mcp-agent-acceptance-20260907/`.
- The existing Python 3.9.6 source environment ran 1,010 tests: 990 passed and
  20 skipped (19 optional MCP tests plus the opt-in benchmark). Local process
  tests require OS process inspection/control permissions. This is source
  regression evidence, not supported-version installation certification.
- Wiki regeneration retains 43 pages; lint has zero errors and the same three
  existing citation warnings.

Evidence: `reports/verification/mcp-agent-2026-09-07-focused.log`,
`mcp-agent-2026-09-07-wheel.json`, `mcp-agent-2026-09-07-wiki.log` and the source
regression log. The existing Python 3.9 environment is below the package's
supported minimum; its source regression is recorded separately from supported
Python 3.12 installation evidence.

## Remaining Scope

The saved-review, analysis and dataset follow-up is recorded in
[the next implementation record](2026-09-07-mcp-study-workflows.md); the scope
below describes this earlier increment.

Review actions return drafts for an explicit start; draft persistence can be
simplified in a later application-service change. Dedicated analysis and
scientific continuation interfaces should wrap existing services. Dataset tools,
remote unknown-submission reconciliation tools, binary artifact transfer,
Streamable HTTP and MCP Tasks are not added here. Existing specialized adaptive
stage directories have not been rewritten into a new orchestration engine.

No Git commit, GitHub publication or Amarel submission was performed in this
increment. Remote routing/recovery use contract fixtures, not new cluster
acceptance. Earlier Python 3.12 full-suite limitations involving optional
libDMET fixtures and block2 remain documented in the single-task MCP record;
this increment does not reclassify them as passing.
