# Public Data Contracts

## Purpose

The Python, CLI, Web, local-executor, Slurm, SSH-RPC, and MCP surfaces
must exchange the same versioned structures. `pyscf_agent.contracts` owns the
task dataclasses and task serialization helpers; `pyscf_agent.schema_contracts`
owns public schema identifiers, compatibility policy, and envelope validation.
Study dataclasses live in `computational_study_agent.schema`. An LLM prompt or
UI-specific payload is never a contract source of truth.

## Contract Version

The current public contract version is `1.0`. Every top-level public payload
uses a stable `schema` identifier ending in `.v1`. Readers accept additive
fields under the same identifier. Removing a field, changing its meaning, or
changing its type requires a new schema identifier and an explicit migration.

`public_schema_manifest()` returns the machine-readable contract inventory.
`validate_public_payload()` verifies the schema identifier and required
top-level fields while allowing additive fields.

## Stable Structures

- `TaskSpec` is the validated calculation request passed to one executor.
- `TaskReport` is the scientific result returned by every executor.
- `StudySpec` is a user-facing multi-case study definition.
- `StudyPlan` is the expanded, validated set of executable cases.
- `StudyReport` integrates case reports, comparisons, decisions, and artifacts.
- `JobHandle` and `JobStatus` describe scheduler identity and lifecycle. They do
  not replace scientific status in `TaskReport`.
- `BatchTask`, `BatchHandle`, and `BatchExecutionResult` describe independent
  task submission and collection.
- execution receipts preserve fingerprints and scheduler handles for resume.
- remote RPC envelopes, benchmark reports, and installation/remote
  verification reports also carry versioned schemas.
- `HamiltonianDatasetSpec`, `HamiltonianSample`, rejection, and dataset
  manifest contracts live in `computational_study_agent.datasets.hamiltonian.contracts`.
  `StudyReport.dataset_manifest` carries the typed dataset summary; numerical
  trajectory matrices stay in referenced artifacts.
- `RuntimeIdentity` records a deployed source snapshot and public contract;
  it belongs to execution infrastructure rather than scientific task data.

Large orbital guesses use `pyscf-agent.numeric-array.v1` references (NPY path,
shape, dtype and content identity) in the existing `initial_mo_coeff` field.
Numerical consumers load arrays explicitly; status and energy plotting do not.
Long presentation text is retained as an artifact with a bounded report preview.
JSON persistence streams into an atomic temporary file instead of constructing
another complete serialized copy in memory.

Artifact references remain embedded records inside report contracts. Each
contains `kind`, `path`, `size_bytes`, `mime_type`, and `description`;
JSON artifact payloads use their registered artifact-specific schemas.

The existing study checkpoint now has additive case execution associations
(run ID, receipt path, pending flag, and outer attempt number), distinguished by
`execution_associations_version=1`. Case dictionaries retain these fields in
StudyReport round trips. The report conversion fix described below also
preserves top-level extensions. Execution receipts can record `submission_unknown` before
handles are acknowledged; consumers must not interpret it as permission to
resubmit. Collect is a separate application-service operation with no submission
side effect, including in adaptive workflows.

The Study checkpoint directly owns each case's current execution association:
`execution {run_id, receipt_path, pending, attempt_count}` and a reference to
its collected TaskReport, with a compact status summary for inspection.
Review subsets keep the original Study ID. An optional checkpoint `review`
record carries stage kind and selected case IDs; it does not own task execution.
Run and Collect return the full StudyReport, without a child registration or
separate integration call. Collect accepts the Study ID to load its saved plan.
The full saved plan and checkpoint retain unselected tasks across retries.

The preceding local `task_reference` and `parent_study` fields are no longer
emitted by this path. Completed historical references can be read for migration;
run artifacts remain at their original paths. Report/checkpoint v1 schemas are
unchanged. Result-batch identity is checked before checkpoint updates, and
historical reports retain their additive-field compatibility.

## Public API Boundary

The optional stdio MCP adapter exposes single-task validation, submission,
status, collection and cancellation through CalculationApplicationService.
It retains the existing TaskSpec, JobHandle, JobStatus and TaskReport contracts.
`collect_task` returns a named summary projection and a full-report resource
URI; only the full report carries the TaskReport schema. Resource references
encode existing handles and introduce no additional task identity or store.
`inspect_task` is a read-only application projection over an executor's existing
JobStatus, saved TaskReport summary and bounded text-log evidence. The Web task
monitor and MCP `show_task` / app-only `refresh_task_view` consume that same
projection. They do not collect, submit, retry or infer scientific convergence.
`open_task_monitor` opens the same view with the exact JobHandle; browser reload
reconnects to that Run. Visible active monitors refresh without LLM calls,
stop at terminal status, and label solver metrics as last observed evidence.
SSH inspection uses the executor's `inspect-task` RPC and requires an updated
remote runtime. UI availability does not establish native host rendering support.
The Registry and packaged wiki are separate resources with their existing
authority boundaries. Numerical approval gates and executor retries remain
owned by the runtime; tool calls do not grant scientific approval.

Single-task callers retain the returned handle across reconnects. Single-task
submission is not idempotent and is never retried by the adapter.

Study MCP tools call StudyApplicationService for static/adaptive preparation,
one background agent invocation, inspection, collection and review. The tool
submit_study maps to application start_study; the former per-task submit-only
executor branch is removed. The existing run_study/run_adaptive_study own task
execution, diagnostics, method routing, dependencies, retry and report merging.
Direct-CAS preparation and probe-to-review conversion also belong to the service.

A saved Study ID is enough to inspect and collect after restart. Review loads the saved plan/report and saves its decision in
StudyReport.pending_review. Start reads that decision using only Study ID;
current Task results are unchanged until execution. The full plan, study-state.json and execution receipts remain task
execution evidence. Receipts contain Slurm batches or local case-to-JobHandle
associations under jobs; no new execution entity is introduced.

study-invocation.json records the latest background application call, executor
configuration, timestamps and error. An inherited POSIX file lock identifies a
running coordinator across MCP server restarts; numerical children do not keep
that lock. This metadata does not replace Study/Task reports. One start runs the
existing agent through completion, required review or interruption. During
orchestration can_collect is false; status and collection never advance work.
Approved subsets keep Study/task IDs and create runs only for selected tasks.
Summary projections carry no full-report schema. Unknown-submission
reconciliation remains in the existing Python application service.
See `docs/guides/mcp.md` for the installed command and client contract.

- `pyscf_agent` exposes `CalculationApplicationService`, `TaskSpec`,
  `TaskReport`, `PlatformRegistry`, and `default_registry`.
- `computational_study_agent` exposes `StudyApplicationService`, `StudySpec`,
  `StudyPlan`, `StudyCase`, and `StudyReport`.
- Executor contracts live under `pyscf_agent.executors`; registry contracts
  live under `pyscf_agent.registry`.
- Workflow nodes, request parsers, runners, retry helpers, provider adapters,
  and catalog-building constants are internal implementation APIs and are not
  re-exported from package roots.
- The removed `pyscf_agent_backend` and `backend.models` compatibility facades
  must not be recreated. Internal code imports task contracts from
  `pyscf_agent.contracts`.

## Boundary Rules

- Serialize dataclasses through their `to_dict()` methods.
- Reconstruct external payloads through `from_dict()` and contract validation.
- Preserve accepted additive fields when a payload is read and written again;
  accepting an extension and silently dropping it is not compatible round-trip
  behavior.
- Never infer scientific success from scheduler completion alone.
- Do not expose process-local PySCF objects in a public structure; persist
  numerical arrays as artifacts and pass registered references.
- Registry entries and workflow provenance may add fields, but cannot silently
  change the meaning of an existing schema.

## Study Review Boundary

`StudyReport` is the merged scientific record, rebuilt from current Task/Run
results. Saved review actions retain the original Study ID and selected case
IDs. The backend loads its saved plan and execution evidence; browser caches
do not decide result authority. Some specialized services still transport a
parent report and private `_adaptive_*` stage fields. These are implementation
details, not an additional public task hierarchy.

Complete reports are written after the current TaskReports and can be marked
successful only when all current task outcomes succeed. Typed review/stage
adapters may be added independently when useful. Ordinary subset retries do
not require a report-revision protocol or another state store.

The stable study contract must explicitly carry adaptive results, review
context, postprocessing, path diagnostics, and continuation evidence when those
features cross an API or persistence boundary. Free-form dictionaries may be
used inside one implementation function, but not as a second public study
result format.

## Study Report Compatibility And Remaining Migration

The September 2026 report conversion fix adds optional `adaptive` and
`postprocessing` objects to `StudyReport`. `from_dict()` accepts the base v1
schema, the existing adaptive v1 schema, and schema-less legacy base reports.
`to_dict()` preserves the input schema and accepted additive top-level fields;
schema-less input receives the base v1 schema. All nested values are copied at
both boundaries. Missing optional extensions default to null, while explicit
empty objects remain empty. Unknown schemas and malformed known extensions
are rejected; an adaptive-schema report requires an adaptive object.

This corrects the earlier silent extension loss. Scientific calculations,
postprocessing execution, and artifact contents are unchanged. The adaptive
runner still returns its existing versioned dictionary, now readable through
StudyReport without discarding its stage evidence.

Some review services still consume a caller-supplied `parent_report` and
private `_adaptive_*` transport fields. Shared typed review context and
executable stage adapters remain possible improvements. Preserving opaque
additive evidence does not itself validate the contents of those fields.

## Verification

Contract tests verify study extension/additive-field preservation, both existing
report schemas, legacy inputs, independent copies, and rejection of incompatible
versions or malformed known extensions. A numerical adaptive test also reads
the actual service result and persisted report through StudyReport.
Clean-wheel and remote
verification reuse the same manifest so packaging and cluster deployments are
checked against the API that local execution uses.

## Related Pages

- [[Design Philosophy And Layer Boundaries]]
- [[CAFES Architecture]]
- [[Capability Registry Runtime Boundary]]

## Saved Review, Analysis And Dataset Calls

StudyReport.pending_review contains the latest selected action, plan, workflow,
case IDs and any existing continuation approval. Review never submits work;
Start uses this saved decision by Study ID. The worker consumes it after its
application call returns. Failed launches and interrupted invocations retain
it; ordinary receipts and Task/Run resume rules still own recovery.

The agent's diagnose_results performs scientific analysis without an LLM;
analyze_results optionally adds language interpretation. MCP analyze_study
loads/saves by Study ID and proposes continuation without executing it. An
approved MPS/path continuation is a background call to the same existing
application use case. Postprocessing uses the existing plotting service.

prepare_dataset accepts the existing dataset/geometry contracts and validates
through the existing Task service. It saves an ordinary MD Study. The runner
assembles accepted/rejected samples and the dataset manifest; MCP only exposes
that result. Runtime-supported MD methods remain authoritative even for custom
dataset specifications.

## Scientific Request Validation

New calculation requests reject unknown owned fields and explicit invalid
numeric/boolean values before normalization can erase them. Defaults apply to
missing optional values. Integer conversions preserve the full value;
non-finite scientific inputs and booleans used as numbers are rejected.
Versioned report readers still support additive extensions. Provider options
are validated by the owning provider at the shared application boundary.

A Planner response is a complete StudySpec. Web adapters do not reinterpret
free text to override a solver, infer an operation parameter, rebuild a scan
from its seed, or change the requested study mode. Equivalent geometry aliases
and explicit numeric ranges normalize in the shared StudySpec contract.

## Canonical Runtime And Evidence Status

Task/Study normalization rejects lossy orbital indices, malformed adaptive
options, and invalid provider scalar/container values. The DF shorthand aliases
remain supported by the shared parser. String false normalizes to boolean false.
Sparse block2 adaptive options and orbital-only restart provenance survive
normalization without changing the saved request fingerprint on a second pass.

Finite-model reference calculations and generated scripts receive RuntimeSpec.
Without overrides they use max_cycle=50 and the upstream PySCF energy tolerance;
legacy implicit 200/1e-10 values require an explicit request for reproducibility.
Study cycle recovery consumes top-level runtime aliases before changing the
canonical runtime object. DMET continues to use its separate provider controls.

Evidence records are additive: reference-density capture failures appear in
evidence_errors; CAS natural-occupation evidence and T2-importance evidence
include status and reason. An out_of_bounds model NOON summary preserves raw
occupations and has null fractionality metrics. Unavailable evidence must not
be counted as evidence of weak correlation.
