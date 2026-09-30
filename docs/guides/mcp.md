# MCP Interface

The optional stdio MCP server exposes calculation and Study application services
to an external agent. It uses existing task, Study and execution contracts.
A Study starts one detached invocation of the existing agent runner. Local
work runs in supervised child processes; direct Slurm and SSH-to-Slurm reuse the existing executor configuration.

## Installation And Startup

Use Python 3.10 or newer and install the extra from this checkout:

```bash
python -m pip install -e '.[mcp]'
pyscf-agent-mcp --work-dir /absolute/path/to/runs
```

The extra uses the [official MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk),
with a `>=2.1,<3` dependency bound. The SDK owns protocol negotiation, schemas,
tool dispatch and stdio transport. The core package does not require MCP.

The equivalent module command is `python -m pyscf_agent.mcp_server`.
When launched in a terminal, the server waits for protocol input; it is meant
to be started by an MCP client. Standard output is reserved for MCP messages.
Service logging goes to stderr. The Study coordinator writes `study-worker.log`,
and numerical workers use the existing run log files. Background Study execution
uses POSIX processes and file locks on macOS/Linux.

Choose one execution target when starting each server:

```bash
pyscf-agent-mcp --executor local --work-dir /absolute/path/to/runs \
  --local-wall-time-seconds 600
pyscf-agent-mcp --executor remote --remote amarel \
  --remote-config /absolute/path/to/remote.ini
pyscf-agent-mcp --executor slurm --slurm-profile default \
  --slurm-config /absolute/path/to/server-slurm.ini
```

The remote command runs on the client machine and reaches the cluster through
SSH. Remote work directories remain owned by the remote executor. Resource
profiles and optional source/runtime matching retain their existing semantics.
See [remote execution](remote-execution.md). This interface serves a trusted
local client using the operating-system user's existing executor permissions;
it is not a multi-user hosted endpoint.

## Client Configuration

For Codex, the [local plugin](../../plugins/pyscf-agent/README.md) bundles this
server with workflow guidance. Configure its installed agent environment and
fixed output directory once, then install the plugin through a local personal
marketplace. It exposes the sixteen tools and resources described here (one
refresh tool is app-only);
the plugin adds no scientific execution logic. The [saved Study workbench](study-workbench.md)
can create or open the same Study ID in Codex's browser and use the same saved
start/review/collect and analysis methods. Use `open_workbench` to start or reuse
the local Web server and obtain a Study link without executing calculations.

For clients accepting an `mcpServers` configuration, use the installed command's
absolute path and an explicit output root:

```json
{
  "mcpServers": {
    "pyscf-agent": {
      "command": "/absolute/path/to/venv/bin/pyscf-agent-mcp",
      "args": ["--executor", "local", "--work-dir", "/absolute/path/to/runs"],
      "env": {"OMP_NUM_THREADS": "1"}
    }
  }
}
```

For a remote profile, replace the executor arguments with the remote command
above. No LLM credentials are needed by these tools: the external agent creates
structured requests, and application services validate and execute them.

## Tools

| Tool | Input | Result |
| --- | --- | --- |
| `get_capabilities` | Optional Registry namespace | Namespace index or full entries, configured executor, resource URIs |
| `validate_task` | `task_spec`, optional `locale` | Validity, normalized TaskSpec, errors, applied defaults, compiled workflow |
| `submit_task` | `task_spec`, unique `run_id`, optional `resource_profile` and `locale` | Submission result, JobHandle, report URI |
| `get_task_status` | Exact returned `handle` | Existing JobStatus |
| `collect_task` | `handle`, optional `include_report` | Compact scientific result, errors/review requirements, artifact references, report URI |
| `cancel_task` | `handle` | Executor cancellation result |
| `prepare_study` | `study_spec`, optional `resource_profile` and adaptive `options` | Saved static/adaptive preparation, cost estimate, study ID and resource URIs |
| `submit_study` | `study_id`, optional retry selection and resource settings | Whether a complete agent invocation was started, its mode and resource URIs |
| `get_study_status` | `study_id` | Agent invocation status, current execution receipts, collection readiness and available task counts |
| `collect_study` | `study_id`, optional `include_report` | Rebuilt full-Study summary, report URI and optional complete StudyReport |
| `review_study` | `study_id`, `action_id`, optional case IDs, approval token and plan kind | Saved review decision and workflow; no submission |
| `analyze_study` | `study_id`, optional `postprocess`, `include_report`, `locale` | Existing agent diagnostics, continuation proposals and plot specifications; no LLM or new calculations |
| `prepare_dataset` | `dataset_spec`, `seed_geometries`, optional resource profile | Saved MD Study plan and dataset resource URI; no calculation |
| `open_workbench` | Optional saved `study_id` | Starts/reuses the local WebUI; returns URL, process ID and log path; no calculation |
| `show_study` | Saved `study_id` | Read-only MCP Apps card and structured fallback; up to 20 saved case results |
| `refresh_study_view` | Saved `study_id` | App-only refresh of the same card data; no collection or execution |
| `show_task` | Exact saved JobHandle | Task monitor and structured fallback: executor state, scientific outcome, bounded log evidence |
| `refresh_task_view` | Exact saved JobHandle | App-only refresh of the same Task monitor; no collection or execution |
| `open_task_monitor` | Exact saved JobHandle | Starts/reuses the workbench and opens a standalone Task with automatic refresh |

The CLI supplies `open_workbench` with its existing runtime/executor settings.
See the [Study card guide](mcp-study-card.md) for UI metadata, host support and
the distinction between saved scientific results and current execution status.
Embedded `create_server` users can supply a `LocalWorkbench` explicitly; without
one, scientific and read-only card tools remain available; browser launcher
tools require a workbench. After the
workbench starts, Study tools add `workbench_url` to their results while the
matching service is reachable. No Web process starts merely because a Study
is prepared or inspected. See the [workbench guide](study-workbench.md) for
lifecycle and local-host boundaries.

See the [Task monitor guide](task-monitor.md) for standalone Runs, automatic
refresh, solver metrics, and remote-runtime requirements. It uses the same
calculation application service as the CLI and WebUI, without an extra task store.

`task_spec` accepts the nested public contract or existing shorthand:

```json
{
  "task_type": "molecular",
  "atom": "H 0 0 0; H 0 0 0.74",
  "basis": "sto-3g",
  "method": "hf",
  "outputs": ["energy"]
}
```

For capability details, request a namespace such as `molecular.method`, or
read the full Registry resource. Registry support does not certify that an
optional provider is installed on the configured target. Numerical execution
continues to validate provider availability and scientific prerequisites.

Validation uses the same deterministic builder, validator and workflow/gate
compiler as execution. It creates no calculation files and starts no numerical
work. Passing validation does not grant active-space or other scientific
approval; a blocked report retains its existing review requirements.

`collect_task` returns `handle`, `report_uri` and a `summary` projection. The
summary is not labeled as a complete TaskReport; `include_report=true` adds
the complete versioned contract under `report`.

`submit_task` validates before submitting and does not wait for completion.
Invalid scientific input returns `submitted=false` with validation details.
Malformed arguments and executor failures return an MCP tool error. Scientific
failure or nonconvergence is carried in the collected TaskReport.

## Following And Recovering Calculations

Save the returned JobHandle and report URI. After reconnecting or restarting
the MCP server with the same target and output root, pass that handle to
`get_task_status`, `collect_task` or `cancel_task`. A report URI encodes the
existing handle; it introduces no separate task identifier or persisted store.

Scheduler `state` and scientific `task_status` are separate. Query until a
TaskReport is available, then collect. For a terminal local process that exits
without a report, collection creates the existing failed TaskReport from its
saved request and terminal status. Cancellation and timeout therefore close
the reserved Study attempt even after a server restart.

Status, collection and resource reads never submit another calculation.
Repeated collection is supported. `submit_task` starts a new calculation;
it is not an idempotent recovery operation. Local duplicate run IDs are
rejected by the existing executor. Remote submissions may create different
scheduler jobs even when given the same run ID. The adapter never retries
submission automatically. If a transport fails before returning a handle,
inspect existing executor/scheduler evidence before submitting again.

The single-task API requires retaining the handle. It does not yet provide
job listing or Study's persisted unknown-submission reconciliation workflow.
Existing calculation-internal convergence retries are unchanged; MCP does not
introduce another retry loop.

## Study Submission, Collection And Review

For a small static example, call `prepare_study` with this `study_spec`:

```json
{
  "name": "H2 basis comparison",
  "objective": "compare_results",
  "system_type": "molecular",
  "base_task": {
    "task_type": "molecular",
    "atom": "H 0 0 0; H 0 0 0.74",
    "basis": "sto-3g",
    "method": "hf",
    "outputs": ["energy"]
  },
  "sweep": {"basis": ["sto-3g", "6-31g"]},
  "observables": ["energy"]
}
```

Static preparation saves `study-plan.json` and `cost-estimate.json`. Set
`study_mode: "adaptive"` in the same input to use the agent's existing molecular
adaptive workflow. Adaptive preparation saves the complete StudySpec and options
in `adaptive-study-request.json` and the initial plan in
`adaptive-initial-scan-plan.json`. Preparation performs no numerical work.

1. Call `submit_study(study_id)` once. It starts a background call to
   `StudyApplicationService.run_study` or `run_adaptive_study` and returns.
2. The agent executes the static task list or the adaptive initial scan,
   diagnostics, method routing, refinement and existing recovery stages. It
   stops on completion, required review or interruption. Dependencies, retry
   selection, cost gates and result merging use the existing runners.
3. Inspect `get_study_status(study_id)`. While `agent.running` is true,
   `can_collect` is false. After the runner returns, collect/read the report
   and inspect `study_status` and the workflow for scientific outcomes and
   required decisions. Status and collection never advance the workflow.

The MCP client may exit immediately after starting a Study. Reconnect with the
same executor and output root, using the same Study ID. Repeating start while
the agent is running returns `started=false` and creates no new invocation.
After it stops, an explicit start follows existing resume/retry rules: reuse
successes, recover pending handles, and retry failed/unconverged static tasks
up to `max_case_attempts` (default two total attempts). Explicit case/status
selection retains existing override behavior. Adaptive reviewed subsets use the saved review decision; arbitrary adaptive
case/status retries require reviewing the selected cases first.

`study-invocation.json` stores the latest application call, executor connection
options, timestamps and any invocation error. `.study-invocation.lock` is held
by the coordinator until it exits, including after the MCP server closes.
This is process metadata; TaskReports, Study checkpoints and execution receipts
remain the scientific and execution evidence. The lock coordinates background
start/review/collect and analysis calls across MCP processes and the saved Study
workbench. Browser Build Plan now uses saved preparation as well. Direct Python
and legacy HTTP Run endpoints retain their existing locking; the current
browser no longer calls those endpoints. Do not overlap legacy writers with a
background agent.

A terminated coordinator is shown as `agent.interrupted`; a failed invocation
exposes `agent.error` (including a saved receipt when applicable). Existing
numerical jobs can be recovered by collection or an explicit resume. Collection
never submits missing work. Readiness to collect does not imply scientific
success, and review-required adaptive reports are valid completed invocations.
Use the resource profile selected at preparation for adaptive resume; changing
that saved profile requires a new preparation.

Direct CASSCF/CASCI preparation also uses the application's existing active-space
review/probe path. A probe executes through the same Study runner, which saves
its ActiveSpaceAudit review before returning. Approval and subsequent execution
use the existing review actions and retain the Study identity.

To change a calculation, call `review_study` with a displayed action ID and its
selected case IDs. The application saves the decision, plan, approval evidence
and workflow in StudyReport.pending_review without changing current Task results
or starting work. When `can_run` is true, call `submit_study(study_id)`; no plan
or parent report crosses back from the client. This survives a server restart.
An unapproved or cancelled review cannot start. Cost approval uses this same
flow with `approve_cost_estimate` after the user's decision.

The worker consumes the saved review only after the application invocation
returns. Launcher/transport failures retain it for recovery. Collection retains
the saved review without executing it. Existing Task/Run resume rules prevent a
replayed invocation from duplicating completed work. The older MCP
`prepared_plan` submission argument is removed; Python's explicit-plan API
remains available for existing application callers.

Approved subsets stay under the original Study and task IDs. Submission saves
their changed requests into the full plan and creates new run IDs only for
selected tasks. Collection retains other task results and saved adaptive or
postprocessing extensions. There is no separate MCP queue, database or retry
state machine: `study-state.json` and existing execution receipts remain the
execution evidence. Receipts may contain Slurm `batches` or a local `jobs`
mapping from case ID to its existing JobHandle.

If submission is `submission_unknown`, inspect the persisted executor evidence.
The adapter never guesses a successful submission or automatically replaces it.
Remote reconciliation remains available through the existing Python application
service; it is not a new MCP tool in this release.

## Analysis And Datasets

`analyze_study(study_id)` calls the agent's deterministic result analysis. It
saves path diagnostics, state tracking and follow-up proposals in the report.
It neither requires an LLM nor starts calculations. `postprocess=true` also
uses the existing postprocessor to write plots and their data artifacts.
MPS/path continuation approval uses the displayed action, case IDs and token
through `review_study`; a separate `submit_study(study_id)` runs that approved
application use case. Skipping continuation allows ordinary analysis to resume.
The optional language interpretation remains available through Python/Web.

`prepare_dataset` accepts HamiltonianDatasetSpec and JSON MolecularGeometry
inputs. It calls the existing MD Study planner and validates its Task requests.
Use the same cost review, start, status and collection tools. The runner already
assembles accepted/rejected sample indexes and the manifest; MCP adds no dataset
execution loop. Read `dataset_manifest` in the collected report or `dataset_uri`.
The executable MD path currently uses B3LYP/def2-SVP and the existing QH9 SCF
settings. A custom dataset contract does not enable unsupported MD methods.
Remote matrix arrays remain executor-owned artifact references.

## Resources

| URI | Content |
| --- | --- |
| `pyscf://schemas` | Existing public contract manifest |
| `pyscf://capabilities` | Full runtime Registry payload |
| `pyscf://wiki` | Packaged wiki page index with resource URIs |
| `pyscf://wiki/{slug}` | One packaged wiki page |
| Report URI returned by submit/collect | Complete TaskReport, including existing logs and evidence |
| `pyscf://studies/{study_id}/plan` | Saved full static/reviewed StudyPlan, or adaptive input/options and initial plan |
| `pyscf://studies/{study_id}/report` | Saved StudyReport, including pending review and scientific analysis |
| `pyscf://studies/{study_id}/dataset` | Dataset manifest produced by the existing Study runner |

Clients should retain and use returned report URIs as opaque strings. Full
reports are read from the configured executor. Binary numerical artifacts
remain ordinary artifact references; the first version does not add a generic
file-download tool. A client without resource-reading support can request
`collect_task(..., include_report=true)` for the full report.

## Scope And Verification

The CLI exposes sixteen tools over stdio: six task tools, seven
Study/analysis/dataset tools, one local WebUI launcher and two card tools (one
app-only). The adapter calls application services for static/adaptive preparation,
background execution, subset retries and existing review actions. It implements
no scientific stage loop. Analysis, saved approval decisions and dataset preparation use existing agent
services. Approved MPS/path continuation runs through a background analysis
invocation using those existing services. Remote unknown-submission
reconciliation, binary transfer, Streamable HTTP and MCP Tasks remain separate
work.

Run protocol and numerical acceptance with the optional SDK installed:

```bash
python -m unittest tests.pyscf_agent.test_mcp_server tests.pyscf_agent.test_mcp_studies -v
PYSCF_AGENT_TEST_MCP_NUMERICAL=1 \
  python -m unittest tests.pyscf_agent.test_mcp_server tests.pyscf_agent.test_mcp_studies -v
```

The numerical check submits H2/STO-3G HF through a real stdio client, closes
the server, starts another server, collects the existing calculation, checks
the energy, and verifies repeated collection retains one job. The Study check
executes two H2 basis cases after one start and client exit, reviews and retries
one case, and verifies two task IDs retain three run records with attempts
`[2, 1]`. A second Study check runs the existing adaptive initial and refined
stages after one start; a direct-CASSCF check covers probe, review and execution. Local process
tests require OS process inspection/control permissions. Optional MCP tests
are skipped when the SDK is absent; normal package imports remain available.
