# Python API Boundaries

The package-root API expresses core scientific application concepts, not
internal workflow steps.

## Calculation API

Use the root package for one-task services and contracts:

```python
from pyscf_agent import (
    CalculationApplicationService,
    PlatformRegistry,
    TaskReport,
    TaskSpec,
    default_registry,
)
```

Use focused namespaces for specialized contracts and infrastructure:

```python
from pyscf_agent.contracts import task_spec_from_dict, task_spec_to_dict
from pyscf_agent.executors import LocalExecutor, SlurmExecutor
from pyscf_agent.registry import CapabilityContract, ProviderContract
from pyscf_agent.schema_contracts import public_schema_manifest
```

Workflow nodes, parsers, runners, retry helpers, registry catalog constants,
and executor implementations are not re-exported from `pyscf_agent`.

`CalculationApplicationService.validate_task_spec(payload)` accepts a nested
TaskSpec or existing shorthand, normalizes and validates it, and compiles its
workflow and gates without executing calculations or calling an LLM. Its result
contains `valid`, `task_spec`, `errors`, `applied_defaults`,
`clarification_questions`, and `workflow_configuration`. Scientific runtime
approval is separate from this preparation check.

The optional [MCP adapter](mcp.md) uses this operation and the existing
`submit_request`, `job_status`, `collect_request` and `cancel_job` services.
It has no independent task store or retry policy.

## Optional SCF Stability

Correlation diagnostics and UNO candidate generation do not run SCF stability
analysis by default. To request PySCF's internal and external checks:

```python
request.setdefault('workflow', {}).setdefault('module_config', {})[
    'molecular.correlation_diagnostics'
] = {'scf_stability': True}
```

Without this option, `structured_results.scf_stability` records
`status='not_requested'` and `stable=None`. A UHF occupation-window candidate
uses the spin-summed AO 1-RDM and overlap. Its audit retains actual UNO
occupations and the full `initial_mo_coeff`; canonical mappings are for display.
The candidate still requires active-space approval before CAS execution.

## Molecular Density Fitting

Add this field to an approved restricted/ROHF CASSCF TaskSpec (FCI or block2):

```python
request['density_fitting'] = {
    'enabled': True,
    'auxbasis': None,  # PySCF chooses the fitting basis.
    'apply_to': 'scf_and_casscf',
}
```

The shared backend fits both reference SCF and CASSCF integrals. Legacy
`apply_to='scf'` on restricted CASSCF normalizes to the combined scope to
preserve its earlier implicit PySCF behavior. Ordinary molecular reference
tasks use `scf`; density fitting remains disabled unless requested.

TaskReport.structured_results.density_fitting records actual `applied_to`,
`resolved_auxbasis`, and `casscf_implementation` alongside the requested fields.
Unrestricted DF-CASSCF is unsupported. The block2 CASCI route still transforms
conventional active two-electron integrals even if its reference uses DF.
MCP calls the same application and needs no additional DF tool.

## Study API

Use the study package root for the application service and study contracts:

```python
from computational_study_agent import (
    StudyApplicationService,
    StudyCase,
    StudyPlan,
    StudyReport,
    StudySpec,
)
```

Low-level operations remain available from their owning modules for focused
tests and internal integrations:

```python
from computational_study_agent.executor import run_study
from computational_study_agent.planner import build_study_plan
from computational_study_agent.postprocessing import run_postprocessing
```

Application clients should prefer `StudyApplicationService` over composing
these operations directly.

`prepare_study(spec, work_dir=..., options=...)` saves a static or adaptive
preparation and returns a dictionary with `study_id`, `mode` and `plan`.
`load_study_preparation(study_id, work_dir=...)` reads the saved inputs;
`load_plan(...)` reads a full static/reviewed plan and `load_report(...)` reads
its report. Missing direct-CAS active-space inputs use the same review/probe
preparation as Web.

`start_study(study_id, work_dir=...)` starts a detached call to the ordinary
`run_study` or `run_adaptive_study`. The runner owns the full task/stage sequence;
clients do not advance it task by task. Use `inspect_execution` and then
`collect_saved_study` to inspect/recover by Study ID. Collection never submits
work. The earlier application `submit_study`/core `submit_only` branch has been
removed; the MCP tool name `submit_study` now maps to `start_study`.

Background local execution uses LocalProcessExecutor. Remote/direct-Slurm
clients inject `execution_config` with the same keyword options accepted by
`create_task_executor`; the worker reconstructs that executor. The synchronous
Python `run_study`/`run_adaptive_study` APIs remain available for in-process use.

`review_study(study_id, action_id, work_dir=..., case_ids=...)` saves the selected
review decision in StudyReport.pending_review without executing it. A later
`start_study(study_id)` reads that plan/approval itself. Current Task results are
unchanged until the ordinary runner executes the selection; no additional task
store is introduced.

`diagnose_results(report, plan=...)` performs the existing scientific analysis
without an LLM. `analyze_results(..., llm_request_builder=...)` adds the optional
language explanation. `analyze_study(study_id, postprocess=False)` loads and
saves the scientific result by ID, proposing continuation without executing it.
Approved continuation actions are started in the background through the same
Study API. `prepare_dataset(dataset_spec, seed_geometries, work_dir=...)` uses
the existing dataset planner; the ordinary runner assembles its manifest.
`load_dataset(study_id)` reads that manifest from the saved report.

A Study owns its tasks (cases), and each task owns its current execution run.
An approved review subset retains the original `study_id` and `case_id` values;
it changes only the selected task contracts and runs. `study-plan.json` retains
the full task set, and `study-state.json` retains every task's current execution,
attempt count, and TaskReport.

```python
# Retry selected tasks under the same Study; the returned report is complete.
report = service.run_study(
    full_plan, work_dir="runs", rerun_case_ids=["case-0001"],
)

# Or execute an approved subset returned by apply_review_action().
# Its study_id is the original Study ID, and its cases are the selected tasks.
report = service.run_study(approved_plan, work_dir="runs", resume=False)

# After an interruption, reload the saved plan and collect by Study ID.
report = service.collect_study(report.study_id, work_dir="runs")
```

Run and Collect directly rebuild the complete report from current TaskReports.
There is no separate `integrate_review_execution` call or child registration.
The optional `study_report=` argument carries a review scaffold when one has
not yet been persisted; its identity must match the plan. Saved results take
precedence over a browser snapshot. The former `parent_report=` argument is
removed. Web Collect needs only `study_id` and `work_dir`, plus the execution
target when applicable.

New task attempts receive distinct run directories; previous artifacts remain.
Pending executions are recovered across the same Study before further retries.
Collect never submits tasks. Batch completeness and run identity are validated
before checkpoint updates. Final success requires successful current results
for all tasks. Full comparison rows, unselected task contracts, adaptive and
postprocessing evidence, and unchanged annotations survive subset updates.

Completed legacy child/stage results can be imported from persisted evidence
into the original Study checkpoint without recalculation. Pending historical
child executions require collection before migration. Historical reports remain
readable, but browser-only numerical snapshots cannot initialize execution
state. See the [architecture guide](../architecture.md) for ownership boundaries.

Local dataset generation publishes each successful attempt in a separate
`generation-*` subdirectory under the requested output directory. Use the
generation receipt's `dataset_root` to locate the portable dataset. Arrays are
copied into staging, validated, and published before the receipt is replaced;
failed retries preserve the previous dataset and receipt. Earlier successful
versions are retained. The low-level `materialize_hamiltonian_dataset` operation
requires a new destination directory and refuses to overwrite an existing one.

## Study Report Round Trips

`StudyReport.from_dict()` reads base `study-report.v1`, existing
`adaptive-study-report.v1`, and schema-less legacy base reports. It retains
`adaptive` and `postprocessing` as optional object fields, and preserves other
additive fields at their original top-level keys. `to_dict()` retains the input
schema; schema-less input receives the base v1 schema. Incompatible schemas and
malformed known extension fields are rejected.

```python
report = StudyReport.from_dict(payload)
report.summary = "Reviewed results"
saved_payload = report.to_dict()
# Existing adaptive/postprocessing evidence and additive fields are retained.
```

Input, object, and exported nested values are independent copies. Absent optional
extensions default to `None` and serialize as JSON `null`; existing empty
objects remain empty objects. The postprocessing service still returns its
result separately; this change preserves results already attached to a report.
Serialize through `to_dict()`, not raw `dataclasses.asdict()`.

This fixes the earlier data-loss boundary. The adaptive runner still returns a
versioned dictionary; typed stage execution, review context, and backend report
revisions remain separate migration work.

## Internal Workflow Modules

The former `pyscf_agent.pyscf_agent_backend` compatibility facade has been
removed. Import workflow implementation details only from their focused
internal modules; application clients should use the public services and
contracts shown above.

## Migration Examples

```python
# Old
from pyscf_agent import build_prepared_request

# New
from pyscf_agent.request_builder import build_prepared_request
```

```python
# Old
from computational_study_agent import build_study_plan, run_study

# New
from computational_study_agent.planner import build_study_plan
from computational_study_agent.executor import run_study
```

## Study Retry And Collection

Use the application service for these separate operations:

```python
# plan is the original validated StudyPlan; executor is its execution target.
service = StudyApplicationService(task_executor=executor)
status = service.inspect_execution(plan.study_id, work_dir="runs")
report = service.collect_study(plan, work_dir="runs")

# A separate Run action can execute only the selected case again.
report = service.run_study(plan, work_dir="runs", rerun_case_ids=["case-0001"])
```

Collect does not submit calculations or increment attempt counts. Adaptive
collection uses `collect_adaptive_study(spec, study_id=..., options=...,
work_dir=...)`; it can return the next prepared plan without executing it.
Use the original spec/options and execution target for collection.

Concurrent Run requests for one Study return `StudyExecutionBusy` rather than
queuing another execution; Collect still uses the existing checkpoint. A later
explicit Run remains a new user action and can override the automatic retry
limit. Run without an explicit subset retains bounded failed-case retry. New attempts
receive distinct run directories while stable case IDs remain suitable for
merging. Recovering an already pending execution reuses its handles. A missing
or corrupt checkpoint/receipt and an unknown submission outcome require
reconciliation rather than automatic resubmission. See the
[retry guide](selected-case-retries.md) for supported recovery actions.

For an unknown SSH submission, explicitly select the existing server evidence:

```python
service.reconcile_execution(
    plan, work_dir="runs",
    submission_path="/scratch/user/agent/remote-submissions/remote-.../submission.json",
    # receipt_file="runs/study-id/execution-receipts/batch-.../execution-receipt.json",
)
report = service.collect_study(plan, work_dir="runs")
```

Reconciliation compares the original request fingerprints, case IDs, and pending
run IDs before associating handles. It never submits, retries, or collects jobs.
Repeated reconciliation of the same evidence is harmless. It requires server
evidence written by the updated SSH endpoint; older or incomplete submissions
without that evidence remain unknown and require manual investigation.

## Supervised Local Execution

```python
from pyscf_agent.executors import LocalProcessExecutor

executor = LocalProcessExecutor(wall_time_seconds=600, terminate_timeout=5)
handle = executor.submit_task(request, work_dir="runs", run_id="local-task")
# Save handle.to_dict(); after the submitting process exits:
reconnected = LocalProcessExecutor()
status = reconnected.status(handle)
status = reconnected.cancel(handle)
```

On POSIX hosts, an independent supervisor records process identity and controls
the calculation's process group. The wall-time limit is persisted at submission
and remains active after the Web process exits; `terminate_timeout` is the grace
period for stopping processes. Omit `wall_time_seconds` for no automatic limit.
Cancellation becomes terminal only after calculation-group termination. A
mismatched PID/start time/token or an old job without identity is an explicit
unconfirmed-control error, never permission to signal an arbitrary PID.
The synchronous `execute_task` interface also uses supervision and turns a hard
timeout into a failed TaskReport, allowing a Study attempt to finish normally.
