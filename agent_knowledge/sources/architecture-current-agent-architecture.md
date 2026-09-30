# Current CAFES Architecture

## Scope

This page describes the executable architecture as of the current modular
source tree. The runtime capability registry and backend validation remain the
authority for what may execute.

## Three Surfaces

1. **Calculation Assistant** prepares, approves, runs, and analyzes one
   molecular, periodic, or model-Hamiltonian task.
2. **Model Hamiltonian Builder** creates lattice/model input artifacts. It does
   not run a calculation. Finite clusters leave correlated solver selection to
   the assistant/planner; Bloch inputs bind to the representation-specific
   `tight_binding` solver.
3. **Computational Study Planner** expands a scientific objective into cases,
   coordinates static or adaptive scans, and compares their results.

The single web server exposes these surfaces at `/`,
`/model-hamiltonian-builder/`, and `/computational-study/` respectively.

## Implementation Layers

| Layer | Modules | Responsibility |
| --- | --- | --- |
| Public contracts | `pyscf_agent/contracts.py`, `pyscf_agent/schema_contracts.py`, and `computational_study_agent/schema.py` | Stable task/study structures, serialization helpers, schema identifiers, and compatibility validation. |
| Task kernel | `pyscf_agent/backend/` | Task-spec normalization, validation, probe roles, execution, results, artifacts, and state. |
| Molecular correlation | `pyscf_agent/backend/correlation/` | Correlation diagnostics, ActiveSpaceAudit, CAS execution, and SC-NEVPT2. |
| Periodic electronic structure | `pyscf_agent/backend/periodic/` | POSCAR/CIF parsing, periodic validation, gamma/k-point HF/DFT, and cell summaries. |
| Model Hamiltonian | `pyscf_agent/backend/model_hamiltonian/` | Finite Hubbard solvers and diagnostics plus primitive-cell Bloch `h(k)`, bands, DOS, occupations, and artifacts. |
| Optional providers | `pyscf_agent/providers/` | Provider adapters such as block2 DMRG and libDMET, including normalized options, Hamiltonian conversion, structured outputs, convergence history, and checkpoint/artifact provenance. |
| Request preparation | `pyscf_agent/request_builder/` | Structured request preparation, optional LLM drafting, and UI modifications. |
| Unified registry | `pyscf_agent/registry/` | `PlatformRegistry` provides one typed index for capabilities, parameters, observables, option sets, templates, modules, gates, providers, bindings, artifacts, aliases, and execution boundaries. Domain defaults are split by system under `pyscf_agent/registry/defaults/`. |
| Application services | `pyscf_agent/application/` and `computational_study_agent/application/` | Protocol-neutral single-task and study use cases plus execution-target selection shared by inbound adapters. |
| Execution adapters | `pyscf_agent/executors/` | Shared task/batch ports, versioned job lifecycle, current-process `LocalExecutor`, cancellable Web `LocalProcessExecutor`, direct Slurm, and SSH-Slurm job arrays. |
| Runtime deployment | `pyscf_agent/runtime_identity.py`, `pyscf_agent/remote/deployment*.py` | Immutable remote source releases, private worktree bindings, and optional source/runtime identity checks. |
| Molecular datasets | `pyscf_agent/backend/molecular_dynamics.py`, `computational_study_agent/datasets/hamiltonian/` | Complete per-task trajectories, AO matrix artifacts, Study sample manifests, dataset generation, and collection. |
| Campaign planner | `computational_study_agent/` | Study schema, plan construction, validation, execution, postprocessing, and web API. |
| Adaptive workflow | `computational_study_agent/adaptive/` | Initial plans, routing, recovery plans, and explicit workflow state. |
| Web presentation | `pyscf_agent/web_assets/`, `computational_study_agent/web_assets/`, and `model_hamiltonian_ui/` | HTML shells, responsibility-scoped JavaScript, CSS, and shared molecular-preview assets served from package resources. |

## Current application organization

- `pyscf_agent/web/{server,api,ui}.py` owns HTTP composition and transport.
  Old Web module names and console aliases forward to the same implementations.
- `StudyApplicationService` is an explicitly composed facade over preparation,
  saved-study, execution, dataset, postprocessing, analysis, path-recovery and
  MPS-continuation use cases. They share constructor-injected dependencies and
  one report/execution context.
- `computational_study_agent/grid/` owns static parameter-grid refinement;
  `datasets/hamiltonian/` owns dataset planning, contracts and finalization.
- Task infrastructure imports neither executable Study code nor inbound
  adapters; local architecture tests enforce this direction. Study declarations
  in the shared platform catalog remain data, not executable Study imports.
- Run roots come from `pyscf_agent.paths`: explicit directory, environment
  override, then the user data directory. Default saved-Study discovery also
  reads an existing checkout's `runs/`, without moving its records.

## Web Asset Boundary

- Python UI builders inject capability-derived options and initial runtime data
  into small HTML templates; they do not own browser behavior or page styling.
- Calculation Assistant and Planner behavior is split into ordered classic
  scripts by responsibility. The scripts intentionally share one page-level
  state scope, so their load order in each HTML shell is part of the contract.
- `shared.js` exports a frozen `AgentUI` namespace for pure escaping, IDs,
  session-selector and table helpers. Page-specific scientific state is not shared.
- JS/CSS URLs are derived from package-content hashes, not hand-edited versions.
- Molecular structure preview CSS and JavaScript live once under
  `pyscf_agent/web_assets/` and are reused by both surfaces.
- `/assets/` serves Calculation Assistant/shared resources and
  `/computational-study/assets/` serves Planner-specific resources. Both are
  package-resource routes and reject absolute or parent-relative paths.
- Static HTML, CSS, and JavaScript files must remain in package-data manifests;
  source-tree-only asset lookup is not an acceptable deployment dependency.

## Core Rules

- Package-root APIs expose application services and task/study contracts, not
  workflow nodes, parsers, retry helpers, executors, or provider internals.
- The dependency direction is public API to application services to contracts
  and registry to workflow runtime to providers and executors. Contracts do
  not import backend implementations.
- The builder owns model structure. The assistant/planner own finite-cluster
  solver selection; representation-specific Bloch tight binding is selected by
  the input contract.
- Probe calculations are registered workflow roles, not target methods. The
  Calculation Assistant and Planner must resolve the same
  `molecular.active_space_probe_strategy` capability and call the shared
  molecular probe builder. Its versioned contract preserves the requested
  CASCI/CASSCF method, target solver, target solver options, selection method,
  and occupation window for the approved calculation. The registered `auto`
  strategy begins with HF/SCF and conditionally invokes the registered MP2
  refinement module when the preliminary chemical proposal is not review-ready
  or SCF diagnostics show instability, frontier degeneracy, or a routing-boundary
  score.
- The study layer sees complete task inputs and reports. Its active-space policy
  resolves one chemical target and CAS dimension across related cases, while
  leaving case-specific orbital mappings inside each task contract. It does not
  implement a second molecular solver path.
- The planner must execute each case through the single-calculation kernel,
  rather than creating a separate scientific execution path.
- Conversation and UI state are not execution truth. A normalized `TaskSpec` or
  `StudyPlan`, validation result, and explicit approval are required.
- Long output belongs in run artifacts. Workflow state carries compact status,
  summaries, artifact references, and approval context.
- Artifact references carry kind, path, size, MIME type, and description; JSON
  payload schemas are registered alongside the feature that writes them.
- The unified registry determines executability and runtime bindings. Wiki content explains why,
  but cannot enable a method by itself.

## Application Service Boundary

- `CalculationApplicationService` coordinates request preparation, one-task
  execution, result analysis, structure previews, capabilities, and Builder
  input persistence.
- `StudyApplicationService` coordinates `StudySpec`/`StudyPlan` validation,
  static and adaptive execution, postprocessing, artifact access, and
  study-level result analysis.
- Web handlers and the CLI are inbound adapters. They may parse transport
  fields and map errors, but scientific execution must enter through an
  application service rather than import planner, executor, or backend use
  cases directly.
- The services use dependency injection for their runners and request builders.
  `LocalExecutor`, direct `SlurmExecutor`, and SSH-backed `SshSlurmExecutor`
  currently reuse this boundary without granting direct access to backend
  internals. MCP can expose the same application use cases later.
- `ExecutionTargetRegistry` resolves a user-visible target such as `local` or a
  named remote cluster outside the scientific `TaskSpec`. A resource-profile id
  is likewise execution metadata: the server owns its CPU, memory, queue, and
  wall-time values, while the Web UI exposes only registered names.
- `TaskExecutor.execute_task` accepts one normalized task request plus execution
  context and returns its structured `TaskReport`. The same port also exposes
  `submit_task`, `status`, `cancel`, `fetch`, `logs`, and `artifacts` around a
  versioned `JobHandle`.
- `BatchTaskExecutor.execute_independent_tasks` is an optional extension used
  only for ready tasks without cross-case dependencies. It returns reports
  keyed by case id plus versioned batch evidence; it does not create a second
  scientific result format.
- Scheduler state and scientific outcome are separate. `JobStatus.state`
  records `queued`, `running`, `completed`, `failed`, or `cancelled`;
  convergence, validation, and blocked outcomes remain in
  `TaskReport.execution_status`.
- The Web server uses `LocalProcessExecutor` with an independent POSIX
  supervisor, durable PID/start-time/token identity, and process-group cleanup.
  Restarted clients retain status/cancellation control. An optional persisted
  wall-time limit is enforced without client polling; the termination grace
  period remains separate. Setting the local CLI wall-time flag also selects
  supervision. Current-process `LocalExecutor` does not support cancellation.
- `LocalExecutor` implements immediate submission in the current Python
  process. It atomically persists `job-state.json` and
  `job-task-report.json` under the task run directory. Submission therefore
  returns only after local execution, but the resulting handle can be
  serialized and queried by a later CLI or API process.
- Local execution does not claim a scheduler queue or cancellation of a
  running calculation. `SlurmExecutor` submits homogeneous independent tasks
  as throttled job arrays, polls scheduler state, supports cancellation, and
  maps every array index back to the original case id through a persisted
  manifest.
- Static studies and adaptive initial-scan stages may use independent-task
  Slurm job arrays. Adaptive refined and recovery plans are submitted one task
  at a time; projected-1RDM continuation tasks are also excluded from
  independent batching because their predecessor artifact is an execution
  dependency.
- Direct `SlurmExecutor` assumes the submission process and compute nodes share
  the configured work directory. `SshSlurmExecutor` instead uses stateless,
  versioned JSON RPC over SSH; the server materializes checked attachments,
  owns the work root and resource profiles, and requires no resident daemon.
- Remote connection profiles in client `remote.ini` select named
  `[server:<name>]` sections in one server-side `server-slurm.ini`. Resource
  profiles use `[profile:<server>:<profile>]`, preventing queue settings from
  one cluster from being applied to another. The RPC passes the selected server
  profile explicitly and verifies the expected cluster identity.
- A scheduler `COMPLETED` state is not collectable until its `TaskReport` is
  visible. Direct and SSH Slurm adapters use a bounded propagation interval so
  shared-filesystem delay is not misreported as a missing scientific result.
- Every submitted study batch immediately persists a versioned execution
  receipt containing the plan/task fingerprints, executor identity, and Slurm
  handles. Status inspection and result collection reuse that receipt, so a
  transport interruption does not imply resubmission. Independent batches and
  one-at-a-time refined tasks use the same recovery contract.
- Capability availability remains in the runtime registry. Moving orchestration
  into a service does not register or enable a new scientific method.

## Two-Level Workflow Boundary

The agent has two orchestration levels and must not add a third autonomous
research layer inside this package:

1. The task level compiles one `TaskSpec` into executable modules and gates,
   dispatches them through the task runtime, and produces one `TaskReport`.
2. The study level owns complete tasks, their dependencies, approvals,
   execution order, cross-case analysis, refinement, and one merged
   `StudyReport`.

The task runtime is already executable module dispatch. The current study
module runtime is still an ownership/validation registry: adaptive execution
and result analysis are coordinated by procedural application-service and
adaptive-orchestrator functions. New study features must not add more branches
to those functions as a permanent design. They should introduce explicit
study-state transitions and stage adapters until the registered study modules
are the actual execution path.

One study lifecycle is authoritative. Case scientific status, scheduler
status, gate decisions, and presentation workflow are separate observations or
projections; they must not become competing mutable state machines. Browser
session state is a view cache, not the merge authority for a parent report.
Saved subset execution identifies the original Study and its selected cases.
The backend reads the current Task/Run results before rebuilding the complete
report. New checkpoints keep TaskReport references and compact status summaries;
large MO guesses are file references. Existing inline records remain readable.
Typed stage adapters can be introduced independently when needed; a monotonic
report-revision protocol is not required for ordinary retries. Some specialized
review services still transport a parent report and private stage fields.

See [[Study Orchestration And Report Authority]].

## Molecular Dataset Boundary

The dataset planner compiles one seed molecule into one complete MD task,
using the existing Study resource review and executor. Task execution owns
the trajectory and QH9-ordered Fock/overlap arrays; Study finalization owns
sample/rejection indexes and the manifest. Explicit generation materializes
a portable dataset on the selected execution target; collection separately
downloads that generated dataset. The default 100 x 10 design is not evidence
of a completed numerical campaign. See [[Molecular Hamiltonian Dataset Workflow]].

## Study Revision And Adaptive State Boundary

Case-level revisions are part of the `StudySpec` contract. A planner edit that
changes a method, basis, or runtime at selected scan points must preserve the
parent scan and encode the change as `case_design.overrides`: each override has
a variable `selector` and `request_updates` and is applied only to matching
cases. The LLM receives the newest user instruction plus the current structured
specification; it must not reconstruct an empty explicit-case plan for a
point-level change. Deterministic normalization repairs that malformed draft
only when an existing scanned specification and a resolvable selector are both
present.

Adaptive scans are represented by `pyscf-agent.adaptive-workflow.v1`. The
workflow distinguishes full plans from subset recovery and scan-path refinement
plans. Unresolved errors use a current-case review queue because their recovery
causes can differ. Active-space candidates belonging to one executable plan are
instead reviewed as a batch, so their CAS sizes and orbital choices can be
compared before one approved subset run. After a subset rerun, the result must
merge back into the full adaptive report before analysis or postprocessing.
Adaptive execution stops after independent case calculations, including
case-local diagnostics and recovery. It records scan-path analysis as deferred
and does not transfer an AO 1RDM between cases. Only an explicit `Analyze
Results` request checks a one-dimensional final-energy path for non-smooth
transitions or endpoint risks. This path-level evidence complements, but does
not replace, single-point correlation diagnostics. The analysis workflow first
offers a bidirectional projected-1RDM continuation check and proposes local
method promotion only if continuation does not resolve the anomaly. Endpoint
validation and cross-method/reference seeds pause at a versioned review gate
that displays both sources and requires explicit approval.

For block2 results, the same explicit analysis step first tracks targeted roots
across ordered cases using root 1RDMs or natural occupations. It records root
reordering and excludes ambiguous assignments as MPS anchors. It may then
propose a same-method MPS continuation for unresolved cases, retaining up to
four compatible candidates and selecting a state-trusted source. The workflow
checks quantum-number/root sector, executor identity, and requested/executed
orbital ordering, then presents checkpoint and provenance in the reusable
continuation approval gate. Approved retries run sequentially and merge back
into the parent report. This does not alter or couple the initial independent
case execution.

DMRG-CASSCF checkpoints may contain both optimized AO-basis MO coefficients and
MPS files. A compatible same-CAS task can restore both components jointly after
approval/provenance checks. If entanglement and fractional boundary occupations
indicate that the active space is saturated, the workflow creates a costed,
unapproved ActiveSpaceAudit. That expansion restores the optimized orbitals but
deliberately starts a fresh MPS because the Hilbert space changed.

Once execution begins, the Planner session records two directory values: an
immutable displayed study directory and the execution root used for retries or
continuations. The displayed directory is locked in the UI to prevent
mid-calculation redirection, while the root prevents accidental nested study
directories on rerun. Task-session restore preserves this distinction.
The session also retains the active execution identity. Planner Conversation
can deterministically inspect or collect that execution through the application
service, and the review surface exposes the same actions without asking the LLM
to infer scheduler state.

See [[Adaptive Scan Workflow State Machine]].

## Deliberate Non-Goals

- Holstein-Hubbard remains non-executable and is not exposed by the Builder.
- Bloch tight binding is one-body only; it is not DMFT, DMET, Hartree-Fock, or
  an interacting periodic Hubbard solver.
- Same-spin/symmetry-sector state-averaged DMRG-CASSCF is runnable. Different-
  spin averaging, transition-RDM/direct-wavefunction root tracking, state-
  specific projected-root optimization, ab initio DMET including DMRG impurities,
  AFQMC, periodic MP2/CC and Planner scans, and unrestricted AVAS are not
  current runnable capabilities.
  Molecular DMRG-CASCI/DMRG-CASSCF, same-sector state averaging, cross-case
  root-signature tracking, finite-model block2 DMRG, and bounded model-Hubbard
  libDMET are registered optional capabilities. Periodic G0W0, HF+DMFT, and
  GW+DMFT are optional single-task fcDMFT workflows with explicit provider,
  artifact, approval, and convergence contracts; they are not yet adaptive
  Planner routes.
- The planner does not infer arbitrary PySCF APIs from user text or wiki
  retrieval.

## Related Pages

- [[Design Philosophy And Layer Boundaries]]
- [[Predefined Workflow Architecture]]
- [[Planner Agent for Computational Campaigns]]
- [[Capability Registry]]
