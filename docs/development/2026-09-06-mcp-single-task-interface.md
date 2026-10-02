# Single-Task MCP Interface

Inspection date: 2026-09-06. Local implementation on top of the uncommitted
Study/Task/Run correction; no deployment or GitHub publication was performed.

Follow-up: [Study MCP submission and collection](2026-09-06-mcp-study-interface.md)
implements the next increment. The scope and verification below record the
original single-task phase.

## Problem And Result

External agents previously needed Python or CLI calls to use the calculation
application service. The optional `pyscf-agent-mcp` command now exposes six
single-task tools over stdio: capability lookup, deterministic validation,
submission, status, collection and cancellation. Resources expose the full
Registry, public schema manifest, packaged wiki and existing TaskReports.

The adapter is a protocol boundary. It has no job database, scheduler, second
retry policy, LLM request builder or additional scientific approval mechanism.
TaskSpec, JobHandle, JobStatus and TaskReport retain their existing contracts.
The official Python MCP SDK owns protocol handling and client negotiation.

## Application And Execution Boundaries

`CalculationApplicationService.validate_task_spec` exposes the existing
builder, validator and workflow/gate compiler without numerical execution or
LLM calls. It accepts nested public contracts and existing shorthand and
checks an explicit schema before normalization. Errors, applied defaults,
clarification questions and compiled configuration remain reviewable.

MCP calls `submit_request`, `job_status`, `collect_request` and `cancel_job`.
The local CLI uses LocalProcessExecutor so submission returns promptly and
workers remain manageable after the protocol server exits. Direct Slurm and
SSH targets use the existing executor factory and profiles. Submission does
not query status before returning the handle, since a later connection error
must not hide successful submission from the caller.

Status, collection and resource reads never submit work. Runtime numerical
retries and scientific gates remain unchanged. Collection returns a named
summary projection, artifact references and a report URI; the complete
TaskReport retains its schema and is available through the resource or the
`include_report` tool option. Resource URIs reversibly encode existing handles
without introducing another job store or domain identifier.

Expected validation/executor errors map to actionable MCP tool errors.
Scientific failures remain failed/unconverged TaskReports. CLI/service logs
use stderr and numerical output remains in worker log files. Local report
handles are constrained to the configured output root; SSH handles retain
remote paths and are resolved by the configured remote executor.

## Scope And Remaining Work

First-version callers retain JobHandles across reconnects. Submission is not
idempotent; the adapter does not automatically retry it. In particular, a
remote submission error before a handle arrives requires inspecting existing
scheduler/executor evidence. The first version does not implement job listing
or expose Study's unknown-submission reconciliation through single-task tools.

Study MCP tools, nonblocking Study submission, approved subset execution,
dedicated continuation operations, binary artifact transfer, HTTP hosting and
MCP Tasks extensions are subsequent work. Amarel profile routing is available
through the existing factory, but this change did not submit cluster jobs or
perform a new remote acceptance campaign.

## Verification

- Python 3.12.14, MCP SDK 2.1.1, PySCF 2.13.1: 87 focused tests passed,
  including all 16 new tests and existing application, workflow, public-contract
  and distribution checks. Protocol checks include modern/legacy stdio,
  nonexecuting validation, complete reports, reconnect collection, real worker
  cancellation after server restart, duplicate local submission, remote handle
  routing, and submission-error propagation without automatic retry.
- Real stdio H2/STO-3G HF acceptance produced energy
  `-1.1167593073964255` Hartree. After server restart, full-report reading and
  repeated collection retained one execution.
- A wheel was built, installed in an isolated Python 3.12 environment, and
  exercised from outside the source checkout through its installed console
  command. The same H2/reconnect/collection acceptance passed. Its numerical
  artifacts and saved client handle remain under
  `runs/mcp-acceptance-20260906/`; the tracked verification JSON records package
  versions, installed path and wheel checksum.
- The prior Python 3.9.6 source environment ran 992 tests: 979 passed and
  13 skipped (12 optional MCP tests plus the pre-existing opt-in benchmark).
  Python 3.9 is below the declared package minimum and this is source-regression
  evidence, not supported-version installation verification.
- The exploratory full Python 3.12 run was not green: 13 fcDMFT fixture errors
  required the absent optional libDMET provider, and one block2 root-count
  assertion failed. Representative fcDMFT and block2 failures reproduce on
  Git baseline `48a0478` with the same environment. An additional source-archive
  exclusion failure was fixed by explicitly excluding root `__pycache__`;
  the distribution tests pass in the final focused suite. Scientific provider
  code was not changed for this MCP implementation.
- Wiki generation retains 43 pages; lint reports 0 errors and the same 3
  existing citation warnings.

Evidence files use `reports/verification/mcp-2026-09-06-*`: focused tests,
exploratory full supported-environment tests, source regression, baseline
comparison, installed-wheel acceptance and wiki lint. See the
[MCP guide](../guides/mcp.md) for installation and client configuration.
