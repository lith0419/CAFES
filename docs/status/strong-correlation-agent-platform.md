# PySCF Strong-Correlation Agent Platform Report

Inspection date: 2026-09-21
Repository: `pyscf-agent` project root
Baseline: agent version `0.2.0`, integrated `main` at `c620f3b`, plus the
September 16–21 development on `codex/codex-plugin`: Codex plugin/workbench,
retrieval and recovery cleanup, block2/CASSCF improvements, and file-backed storage.
The dated sections below retain historical evidence;
superseded implementation descriptions are not the current architecture.
Current source checks and local tests are distinguished below from archived
scientific, installation, and remote evidence.
Scope: current source tree, runtime capability registry, maintained wiki source,
and regression-suite structure.

The optional stdio MCP CLI exposes calculation/capability and Study tools,
local workbench launchers, and read-only Study/Task views. Static/adaptive preparation and one background invocation
call the existing application runners. Status, collection and review are thin
service calls. The earlier per-task MCP submission branch has been removed.
See the [MCP guide](../guides/mcp.md) and the
current consolidation record (author archive: `reports/development-status-2026-09-21.md`)
for verification and supported-environment boundaries.

## Executive Summary

The September 24 CCSD impurity update adds the Registry-owned optional
`impurity_solver_options.beta`, passed to native libDMET for preliminary
RHF/UHF smearing, with integer CCSD sectors. Omitting it preserves native
zero-temperature SCF; explicit `beta=1000` is available for comparison.
The same-input 32-site honeycomb comparison converges in nine DMET iterations
before and after: energy changes by -1.712e-9 per site in units of |t|, while
total impurity SCF cycles increase from 185 to 257. This case shows no
convergence-speed improvement. A nonzero-spin ring instead develops a
degenerate smeared reference and divergent CCSD, so beta is not a global
default. See the
[beta comparison](../development/2026-09-24-ccsd-impurity-beta.md).

The September 21 storage change replaces large inline MO guesses with lossless
NPY references and keeps TaskReports outside new Study checkpoints. Status
reads use compact summaries; numerical code loads arrays explicitly. Selected
retries, legacy inline data and public StudyReport views remain supported.
Atomic JSON writes stream to disk, and plot preparation copies only its needed
fields. Historical large runs have not been bulk-migrated, and this latest
storage change has not been deployed to Amarel. See the
storage evidence (author archive: `reports/file-backed-task-storage-2026-09-21.md`).

The September 20 Task monitor adds an independent-Run view through
`CalculationApplicationService.inspect_task`. Browser and MCP Apps use the same
page, with active-only automatic refresh, separate scheduler/scientific states,
bounded log excerpts and observed block2 sweep metrics. Inspection does not
execute, collect or retry work. SSH uses the same projection through a new
read-only RPC; remote deployment and native Codex inline rendering are separate
acceptance boundaries. See the [Task monitor guide](../guides/task-monitor.md).

The first Codex plugin increment packages the existing thirteen MCP tools and
one workflow skill. It binds a configured Python installation, output root and
execution target outside the plugin cache. Actual local Codex installation,
stdio discovery and a two-case H2 Study after reconnect passed. The plugin
does not change scientific execution. See the
[implementation and verification record](../development/2026-09-16-codex-plugin.md).

The second increment opens saved Studies by ID in the existing WebUI, preserves
their URL and review state across refreshes, and delegates saved start/review/collect
to the MCP application methods. A real cross-interface H2 acceptance retained
two Tasks and three Runs after retrying one case; 91 focused tests passed with
no skips.
See the [workbench record](../development/2026-09-16-study-workbench.md) and
[usage guide](../guides/study-workbench.md).

The third increment moves browser Build Plan onto saved agent preparation for
static, adaptive and dataset studies, removes the frontend execution dispatcher,
and saves result-analysis diagnostics through the same service as MCP. LLM
drafts remain previews until Build Plan saves them. A real browser-created
four-case H2 Study completed with four Tasks and four Runs and was read and
recollected through the installed MCP plugin without further execution.
The final focused regression passed 98 tests with no skips.
See the [migration and verification record](../development/2026-09-16-saved-study-planning.md).

The fourth increment adds `open_workbench`, which starts or reuses the existing
WebUI on loopback with the MCP runtime's saved configuration. It returns a
direct Study link, survives MCP reconnects, and adds links to subsequent Study
responses without starting calculations. The plugin skill opens those links
in Codex's browser. See the [launcher record](../development/2026-09-16-workbench-launcher.md).

The fifth increment adds a read-only MCP Apps Study card and an app-only refresh
tool. It projects the existing saved plan/report and execution inspection,
without adding scientific decisions or orchestration. The card's workbench
button uses the existing launcher. Protocol and browser-preview acceptance passed;
actual Codex inline rendering remains unverified until a new task loads the tools.
See the [card record](../development/2026-09-16-mcp-study-card.md).

The September 14 consolidation retains one Study with stable Tasks and a new
Run for each selected retry. Shared checkpoint/report helpers preserve the full
Study and its adaptive/postprocessing evidence. MCP delegates to the same
application services as Web and CLI; it adds no scientific execution loop.

The DMET follow-up aligns correlation-potential fitting with the actual
lattice mean-field Hamiltonian across translational and both finite-graph
fitting routes, shared by FCI, CCSD and block2 impurity solvers. Interacting
baths now construct the local HF potential from the k-averaged density while
preserving the physical Fock cache. Numerical mapping checks and the DMET
regression suite pass. The September 20 follow-up adds the validated ADIIS/CDIIS
policy to CCSD impurity SCF, stops unconverged SCF before CCSD, and records
per-call SCF evidence. The original honeycomb case converges in nine DMET
iterations with all nine impurity SCFs accepted. This does not establish
universal convergence. See the [SCF follow-up](../development/2026-09-20-ccsd-impurity-diis.md),
[mean-field analysis](../development/2026-09-14-dmet-mean-field-consistency.md)
and [cross-solver audit and correction](../development/2026-09-14-dmet-solver-fitting-audit.md).

The molecular path supports DF-CASSCF, UHF-density restart, genuine UNO
coefficients, whole/split PM localization, Fiedler ordering, Conventional MPO,
and explicit frozen orbitals. The earlier unconverged lutein campaign is
historical. September 20–21 CAS(20,20) tasks converged at M=600 during orbital
optimization. A preceding unfrozen task also completed its fixed-orbital
M=2000 solve; it is not M=2000-optimized CASSCF. The five overnight frozen-core
tasks each passed three macroiterations and all 12 inner calls.
Freezing 42 core orbitals shifted the observed full-space energy by 0.0248 mHa.
The +3/+5 eV external virtual windows failed the agreed energy/NOON accuracy
targets, despite numerical convergence. Compressed MO integral storage agrees
with the original workspace at the same frozen constraints. See the
latest results (author archive: `reports/lutein-overnight-results-2026-09-21.md`).

The September 10 follow-up makes SCF stability opt-in through the existing
correlation-diagnostics module configuration. Default diagnostics record it as
not requested. UHF occupation-window candidates now use genuine spin-summed
1-RDM natural orbitals and preserve their full core-active-virtual coefficient
matrix. The old same-index alpha/beta occupation sum is no longer used as UNO
selection evidence. See the [verification and Amarel follow-up](../development/2026-09-10-uno-stability-policy.md).

The repository is a scientific workflow platform with a small-system
strong-correlation core and a new molecular Hamiltonian dataset workflow.
It has a validated single-task kernel, a model-builder surface, and a
multi-case planner that can perform
full-grid initial scans, method routing, ActiveSpaceAudit review, recovery, and
data-backed plotting.

The September integration adds QH9-format-compatible molecular dynamics,
dataset sample/rejection indexes, explicit executor-side dataset generation,
and separate local collection. One Study case owns one complete trajectory.
The default 100-molecule, ten-sample-per-molecule design is implemented and
contract-tested; a complete numerical campaign has not been archived.
Remote deployment can bind a local source snapshot to an immutable server
release and reject stale runtime identities before submission.

The current revision also adds a compiled module and quality-gate layer. Each
registered module declares its stage, typed input/output ports, method/task
compatibility, dependencies, conflicts, configuration, and runtime adapter.
The compiler rejects incompatible, cyclic, ambiguous, or incomplete graphs and
persists the resulting workflow configuration and provenance. Task execution
uses those compiled hooks directly; study modules currently describe and trace
the established orchestrator rather than replacing it with a free-form
numerical DAG.

The public Python/executor boundary is now explicitly versioned. Task, study,
scheduler, batch, receipt, RPC, benchmark, and verification structures share a
machine-readable schema manifest with additive-v1 compatibility rules. A clean
wheel check verifies package resources outside the source tree, and the curated
LLM Wiki is packaged for installed Planner use.

The September 18 Wiki alignment publishes 44 pages, including a diagnostic
interpretation guide. Maintained sources outrank historical compiler output;
explicit curated summaries are synchronized separately. Corrected guidance
separates static model scans, molecular adaptive routing, numerical stress,
scientific evidence, and the current MD/DF-CASSCF/periodic-provider boundaries.
Retrieval still supplies bounded English passages. See the
[alignment and verification record](../development/2026-09-18-wiki-code-alignment.md).

The core scientific boundary is also clear:

- Molecular systems can move from MP2 diagnostics to CCSD/CCSD(T), CASSCF, and
  optional ground-state SC-NEVPT2 after active-space approval.
- Finite Hubbard-style models use unrestricted mean-field references by default
  and support explicit MP2, CCSD, CCSD(T), FCI, and optional block2 DMRG
  solver selection. Model Planner studies use static full-grid scans with a
  specified solver; molecular automatic method routing does not apply to them.
  Optional libDMET supports self-consistent single-band Hubbard embedding with
  translated representative impurities or explicit finite-graph fragments,
  FCI/CCSD/block2 DMRG impurity solvers, and structured convergence/artifact
  records.
  Primitive-cell model inputs can also run one-body Bloch tight-binding band,
  DOS, and filling analysis.
- Single periodic HF/DFT tasks run from POSCAR or P1-form CIF at Gamma or on a
  regular k mesh. An optional fcDMFT backend can prepare IAO/IAO+PAO localized
  embedding artifacts and unit-cell ERIs, request correlated-subspace approval,
  and execute HF+DMFT while preserving hybridization, self-energy, convergence,
  logs, checkpoints, and per-retry provenance. The same provider now exposes a
  staged periodic G0W0 and GW+DMFT workflow with reusable lattice-GW and local
  double-counting artifacts. A real two-k-point Si/GTH HF+DMFT smoke run reaches
  the native fcDMFT CC kernel, but it is execution evidence rather than a
  converged physical benchmark. Periodic Planner scans, calibrated memory
  estimators, and correlated total-energy estimators remain outside the
  executable boundary.
- Optional block2 execution now covers fixed-orbital molecular DMRG-CASCI,
  single-state and same-spin/symmetry-sector state-averaged molecular
  DMRG-CASSCF, and finite-model DMRG. Approved molecular
  active spaces may use Boys or Pipek-Mezey localization followed by canonical,
  Fiedler, or explicit manual orbital ordering. DMRG-CASSCF uses PySCF orbital
  macroiterations with block2 1/2-RDMs and internal MPS continuation; optimized
  orbitals, macroiteration history, and solver-call provenance are preserved as
  artifacts. Fixed-orbital result analysis can still propose an approved retry
  from a compatible state-trusted successful MPS on the same execution target.
  Compatible DMRG-CASSCF tasks can restore optimized orbitals and MPS data
  jointly. A larger entanglement-driven active space instead reuses only the
  optimized orbitals and requires approval before constructing a fresh MPS.
  Exact fixed-particle-number Schmidt-rank planning trims oversized bond-
  dimension schedules before execution. Molecular and finite-model block2 runs
  share a result contract, and explicit result analysis tracks targeted roots
  across ordered cases using root 1RDMs or natural occupations before offering
  MPS continuation. Different-spin averaging, unrestricted DMRG-CASSCF,
  transition-RDM/direct-wavefunction tracking, ab initio DMET including DMRG
  impurities, AFQMC, and
  phonon Hilbert-space solvers are deliberately not runnable yet.

Remaining work includes deploying compact storage and deciding whether to
migrate historical large records, cross-geometry orbital projection before
the 23-geometry lutein scan, a full M=2000 orbital-optimization pilot, runtime
output port validation, scientific benchmark calibration,
research-grade active-space quality, DMET convergence and finite-graph
normalization benchmarks, cross-platform offline installation, timeout
calibration, complete MD dataset acceptance, and extending
periodic single-task support into Planner studies with validated correlated
periodic results.

## Current Completion Snapshot

| Area | Current maturity | Remaining boundary |
| --- | --- | --- |
| Task specification and validation | Versioned `TaskSpec`/study/report/scheduler contracts, deterministic normalization, registry-derived options, and explicit approval fields. | StudyReport now preserves adaptive/postprocessing and additive evidence across both existing schemas. Opaque extension transport does not replace typed review/stage contracts. |
| Single-task execution | Current-process local, cancellable Web child-process, direct Slurm, and SSH-Slurm execution share `TaskReport`. Remote deployments can require matching source/runtime identities. | Web cancellation survives restart through verified process identities, and an independent POSIX supervisor enforces an optional local wall-time limit. Legacy jobs without identity remain unconfirmed; the isolated Amarel campaign passed 11 final checks across 10 small tasks. Large-solver campaigns and multi-process Study ownership remain separate work. |
| Molecular calculations | HF/DFT, MP2, CCSD, CCSD(T), FCI, CASCI/CASSCF, SC-NEVPT2, diagnostics, orbital processing, and ActiveSpaceAudit are executable within documented limits. Calculation Assistant and Planner use one Registry-backed active-space probe contract. | Improve active-space chemistry and multireference diagnostics; extend reference calculations beyond the archived N2 curve. |
| Finite model Hamiltonians | Hubbard-family MP2/CCSD/CCSD(T)/FCI/block2 DMRG, optional self-consistent libDMET, exact finite-cluster observables, static full-grid studies, and Builder support are executable. | Add user-reviewed cross-case method revision, finite-size scaling protocols, and larger embedding/impurity solvers before making phase claims. |
| Periodic calculations | POSCAR/CIF HF/DFT, k meshes, smearing, density fitting, standardized/custom band paths, structured artifacts, approval-gated HF+DMFT, periodic G0W0, and staged GW+DMFT with automatic IAO/IAO+PAO subspace/ERI preparation are executable as single tasks. A native-example Si G0W0 benchmark definition is available as an explicit server run. | No archived converged GW/DMFT numerical baseline, calibrated correlated-solver memory estimate, Wannier preparation, correlated total energy, periodic Planner scan, DOS, force/stress, or interacting Bloch solver yet. |
| Molecular Hamiltonian datasets | Restricted B3LYP/def2-SVP NVE trajectories, QH9 AO matrices, deterministic sampling/splits, rejection accounting, dataset generation and collection through existing Study services. | Only the relaxed-SCF QH9 profile is executable; one-frame evidence does not validate a complete 100 x 10 dataset, trajectory energy conservation, or downstream learning quality. |
| Study Planner | Molecular static/adaptive and finite-model static studies with optional one-/two-axis grid refinement, saved review/recovery, shared Web/MCP runners, full reports across subset retries, and compact checkpoints with report references are executable. | Specialized scientific stages retain their existing adapters; some review calls still round-trip the parent report. Cross-geometry orbital projection and legacy large-file migration are separate work. A report-revision engine is not required for ordinary retries. |
| Modules and gates | Task modules and task/study quality gates are compiled, validated, executed or observed, and recorded in provenance/trace artifacts. | Study modules declare orchestrator ownership; runtime port values are not yet schema-validated after every handler. |
| Knowledge and packaging | Curated Wiki sources/runtime exports are packaged; environment/distribution scripts support local and cluster installation. Version 0.2.0 has supported-Python wheel and MCP evidence; see the current consolidation record. | Revalidate target-specific offline wheelhouses. Optional provider skips and the three existing Wiki citation warnings remain explicit. |

## Retry Reliability And Migration Update: 2026-09-04

The local working tree now separates Status, Collect, and Run at the application
service and Web boundaries. Collect uses persisted handles/checkpoints and never
submits failed cases or the next adaptive stage. New Run attempts receive new
run directories; explicit subsets retain all other results. Interrupted remote
execution reuses its handles, while missing/corrupt state and unknown submission
acknowledgement block automatic resubmission. Compatible legacy receipts are
adopted without recalculation.

The Registry audit currently reports 317 typed entries and zero validation
issues, runtime family consumers, compatibility call sites, or legacy public
payload keys. This completed Registry migration is distinct from the unfinished
Study engine migration: typed stage adapters, shared StudyWorkflowState and
ReviewExecutionContext and backend revision history are still open. The subsequent
September 5 conversion fix completes report-extension preservation. The September
6 task-organization correction replaces the interim child-reference design
with Study -> Task -> Run. Review subsets keep the original Study ID and update
selected tasks in its checkpoint. Run and Collect return the complete report;
Collect can reload the full saved plan from the Study ID. No child registration,
separate integration API, database, or report-revision protocol is added.

See the [retry and migration record](../development/2026-09-04-study-retry-and-collection.md)
and [Study/Task/Run correction](../development/2026-09-06-study-task-run-organization.md)
for behavior, historical-data handling, and remaining specialized-stage work.

## Current Architecture

| Layer | Main modules | Responsibility |
| --- | --- | --- |
| Single-task kernel | `pyscf_agent/backend/`, `pyscf_agent/request_builder/`, `pyscf_agent/registry/` | Normalize and validate `TaskSpec`, compile shared probe/target contracts, generate input, execute PySCF, persist artifacts, and summarize results. |
| Molecular correlation | `pyscf_agent/backend/correlation/` | Molecular diagnostics, SCF stability, ActiveSpaceAudit, CAS execution, and SC-NEVPT2 support. |
| Periodic electronic structure | `pyscf_agent/backend/periodic/` | POSCAR/CIF parsing, periodic validation, gamma/k-point HF/DFT, and periodic artifacts. |
| Model Hamiltonian | `pyscf_agent/backend/model_hamiltonian/` | Hubbard spec normalization, unrestricted-reference finite-cluster solvers, FCI observables, and one-body Bloch tight binding. |
| Embedding providers | `pyscf_agent/embedding/`, `pyscf_agent/providers/libdmet/`, `pyscf_agent/providers/fcdmft/` | Shared embedding contracts, translated/finite-graph libDMET execution, and optional periodic fcDMFT HF+DMFT, G0W0, and GW+DMFT execution from registered localized/GW artifacts. |
| Application services | `pyscf_agent/application/`, `computational_study_agent/application/` | Protocol-neutral single-task and study use cases shared by Web, CLI, and future external adapters. |
| Public Python API | `pyscf_agent/__init__.py`, `pyscf_agent/contracts.py`, `computational_study_agent/__init__.py` | Package roots expose only application services and core task/study contracts; executors, registry contracts, schemas, and low-level study operations use focused namespaces. |
| Workflow modules | `pyscf_agent/workflow_modules/`, `pyscf_agent/backend/module_*`, `computational_study_agent/module_runtime.py` | Compile module contracts, validate dependencies/compatibility, dispatch task hooks, and record execution traces. |
| Quality gates | `pyscf_agent/workflow_gates/`, `pyscf_agent/backend/gate_runtime.py`, `computational_study_agent/gates/` | Compile and evaluate task/study quality policies, approvals, resource review, execution quality, and path consistency. |
| Execution adapters | `pyscf_agent/executors/` | Shared task/batch contracts; current-process `LocalExecutor`, cancellable `LocalProcessExecutor`, direct Slurm, and stateless SSH-Slurm. |
| Runtime deployment | `pyscf_agent/runtime_identity.py`, `pyscf_agent/remote/deployment*.py` | Source fingerprints, immutable server releases, private worktree bindings, and client/server identity checks. |
| Molecular datasets | `pyscf_agent/backend/molecular_dynamics.py`, `backend/ao_conventions.py`, `computational_study_agent/hamiltonian_dataset_*.py` | One-task trajectories/AO arrays, Study plan compilation, sample manifests, executor-side generation, and collection. |
| Web application | `pyscf_agent/web/server.py`, `pyscf_agent/web_assets/`, `computational_study_agent/web_assets/` | Serves the Calculation Assistant, Builder, and Planner under one local server. |
| Study planner | `computational_study_agent/` | Case expansion, static/adaptive scans, recovery, report merging, postprocessing, and planner APIs. |
| Adaptive workflow | `computational_study_agent/adaptive/` | Initial-scan construction, diagnostic routing, recovery planning, and explicit workflow state. |
| Knowledge base | `agent_knowledge/` | Curated planning rules and capability boundaries; not the runtime authority for execution. |

The old flat backend filenames no longer exist. The split into `backend/`,
`backend/correlation/`, `backend/model_hamiltonian/`, `request_builder/`, and
`adaptive/` is now the implementation structure that documentation should use.

The former `pyscf_agent.pyscf_agent_backend` compatibility facade has been
removed. Web, CLI, benchmark, and library integrations use application
services or focused namespaces directly.

The Web API is an inbound adapter: it decodes HTTP payloads, maps typed service
errors to status codes, and serializes responses. `CalculationApplicationService`
owns single-task preparation, execution, result analysis, previews, and Builder
input persistence. `StudyApplicationService` owns plan validation, static and
adaptive execution, postprocessing, artifact access, and study-level analysis.
The application services inject one executor contract into every initial,
refined, recovery, and path-restart case. `LocalExecutor` runs the existing
agent workflow in the current process. The Web server uses
`LocalProcessExecutor` to run cancellable child processes. `SlurmExecutor`
submits directly from a cluster login node, while `SshSlurmExecutor` lets a
local CLI or Web UI call a stateless server-side RPC over SSH. These executors return the same validated
`TaskReport`; scheduler lifecycle remains a separate `JobStatus` contract.
Independent cases use Slurm job arrays, while dependency-bearing and refined
tasks are submitted one at a time.

## Public Contracts And Distribution Verification

`pyscf_agent.schema_contracts` is now the single version authority for public
payloads. Contract version `1.0` covers task/study specifications and reports,
scheduler and batch records, execution receipts, remote RPC envelopes,
benchmark reports, and installation/remote verification reports. Top-level
payloads carry stable `.v1` schema IDs. Envelope validation accepts additive
fields under the same ID; breaking changes require a new schema ID and migration.

`public_schema_manifest()` exposes the inventory for Python, CLI, Web, Slurm,
and MCP clients. `validate_public_payload()` checks schema identity and
required top-level fields. Core dataclasses now serialize through `to_dict()`
and reconstruct through `from_dict()` at process boundaries. Process-local
PySCF objects remain outside these structures; large numerical state is passed
through registered artifact references with kind, path, size, MIME type, and
description. Content digests are not required by the common reference contract;
source fingerprints used for deployment are a separate concern.

StudyReport now declares optional adaptive/postprocessing objects and retains
other accepted additive fields through `from_dict().to_dict()`. The reader
supports base v1, schema-less legacy reports, and the existing adaptive v1
schema; it preserves schema identity and rejects incompatible versions.
Nested values are independent copies. Missing optional extensions serialize as
null, and postprocessing execution continues to return its separate result.
The adaptive runner's dictionary API is unchanged. Typed review/stage contracts
and replacing parent-report transport remain separate migration work.

`python -m pyscf_agent.configure verify-install` builds a wheel, installs it in a temporary
environment, changes to a directory outside the checkout, and verifies import,
metadata, schema manifest, console entry points, Web assets, packaged curated
Wiki, benchmark data, and lightweight scientific benchmarks. The normal mode
reuses already installed compiled scientific dependencies while testing the
wheel itself; `--wheelhouse` provides a fully isolated offline mode.

The server-side `capabilities` RPC now includes its public contract manifest.
`python -m pyscf_agent.configure verify-remote` compares that manifest over stateless SSH and can
submit, poll, fetch, and validate a minimal H2/HF Slurm task with
`--submit-smoke`, or two independent H2/HF tasks in one scheduler batch with
`--submit-batch-smoke`. Transport failures produce a structured, redacted
report rather than only a traceback. Completed jobs wait for bounded shared-
filesystem report propagation before collection.

Before scheduler submission, the study runner writes a submission-intent
`execution-receipt.json`; acknowledged handles are then persisted atomically
along with plan/task fingerprints and executor identity. Planner status inspection uses one batched SSH RPC, and
collection reuses those handles to merge remote `TaskReport` payloads into the
original static or adaptive study without resubmitting completed tasks.

`python -m pyscf_agent.configure deploy-remote` now creates an immutable source release and a
private per-worktree binding. With `require_runtime_match=true`, submission
checks the local source fingerprint and remote environment/release identity,
source revision/fingerprint, and public contract. Reusing a public schema
version alone does not establish source parity. The setup and rejection paths have regression
tests; no new live deployment was performed during this report update.

## Compiled Modules And Quality Gates

The unified registry now contains 29 task modules, 13 study modules, five task
gates, and seven study gates, alongside capabilities, typed parameters and
observables, option sets, non-limiting workflow templates, providers, provider
bindings, and artifact contracts. A module contract records:

- stable module/runtime IDs and version;
- permitted/default stages;
- required and provided data ports with schema and merge policy;
- compatible task types, methods/solvers, and jobs;
- optional/default configuration;
- dependency, conflict, exclusivity, and ordering rules;
- deterministic activation rules.

`compile_task_workflow` and `compile_study_workflow` select requested and
automatically activated modules, insert unique dependencies under the `auto`
policy, validate configuration and compatibility, reject missing ports,
ambiguous providers, conflicts, backward-stage dependencies, and cycles, then
write a deterministic workflow ID, registry/spec hashes, execution order,
hooks, edges, and complete selection provenance.

Task execution dispatches the compiled hooks for input generation, numerical
execution, bounded retry, result extraction, molecular/model/periodic
diagnostics, result analysis, and final report construction. Numerical modules
use a process-local solver context so PySCF objects do not enter serialized
checkpoints. Their ordered invocations are stored in
`workflow-execution-trace.json` and copied into `TaskReport`.

Task gates validate compilation, executable input readiness, final execution
quality, and artifact integrity. Study gates expose compilation, resource,
initial-scan, active-space, continuation-state, unresolved-execution, and path
consistency decisions through one reusable approval/presentation contract.
Gate configuration, provenance, decisions, and execution traces are persisted
as structured artifacts.

The study module entries currently describe the existing Planner lifecycle
with `orchestrator_owned` bindings. They provide compatibility checks,
provenance, and observability, but the study orchestrator still owns adaptive
control flow and scheduler submission. This is intentional for the current
release and should not be described as arbitrary graph execution. The shared
typed `StudyWorkflowState`, `ReviewExecutionContext`, backend report revisions,
and executable study-stage adapters described in the design sources are
targets, not completed features of this revision.

Task handlers currently must return a state dictionary, but actual required
output values are not checked against every declared port schema. A completed
trace records declared output names and is not proof of full runtime port
validation.

## User-Facing Surfaces

1. **Calculation Assistant**: prepares one molecular, periodic, or
   model-Hamiltonian task; audits molecular geometry or parsed periodic cell
   data; uses the shared Registry-backed `Auto` active-space probe; requests
   structure or active-space approval where applicable; runs the calculation;
   and displays artifacts/results. Planner studies may additionally force the
   registered `MP2` or small-system `FCI` probe strategies.
2. **Model Hamiltonian Builder**: edits finite-cluster or primitive-cell
   lattices, sites, bonds, boundaries, and Hubbard-family parameters. Finite
   inputs leave the correlated solver to the agent; Bloch inputs declare the
   representation-specific `tight_binding` solver and reciprocal-space setup.
3. **Computational Study Planner**: converts a scientific request into a plan,
   then executes and compares case results. The Web UI is available at
   `/computational-study/` on the same server as the assistant.

## Adaptive Scan State Machine

The planner now exposes an explicit `pyscf-agent.adaptive-workflow.v1` state
instead of relying on chat text alone. Its meaningful states are:

```text
initial_plan_ready
  -> initial scan runs every planned case
  -> initial_scan_blocked              (invalid/missing diagnostic data)
  -> cost_review_required              (resource estimate needs approval)
  -> active_space_review_required      (CASSCF/CASCI candidate)
  -> recovery_review_required          (unconverged/failed refined case)
  -> completed | completed_with_issues
```

Unresolved failure recovery uses a case queue: only the current failed case is
proposed for method/runtime recovery, while successful and unrelated cases
remain intact. Active-space candidates already assembled into one executable
refined, recovery, or path-refinement plan are reviewed as one batch so their
CAS sizes, orbitals, and cost can be compared before approval. Candidates from
different plans are never combined. Active-space cost estimates appear in the
same review surface as the candidate itself, so approval is one scientific
decision rather than a second detached gate. A subset run merges its result
back into the full adaptive report, so analysis and plots continue to use the
full study rather than only the most recent recovery point.

## Recent Workflow Consolidation

The September 17 input/recovery cleanup rejects malformed explicit values and
unknown request fields while keeping documented defaults for absent fields.
Block2 validates provider options before submission and reads its presets and
automatic recovery limits from the Registry. Missing scan diagnostics stop
routing; missing legacy quality evidence has `publication_eligible=null`.
Optional artifact write failures retain valid energies and record missing
evidence. Browser recovery choices come from the backend workflow. See
[implementation and compatibility details](../development/2026-09-17-input-and-recovery-boundaries.md).

- **Shared model-edit contract**: Calculation Assistant and Planner now compile
  natural-language model changes into the same Registry-bounded site, bond,
  global, electron-count, boundary, and solver operations. The LLM proposes a
  structured edit; one deterministic backend applies it, validates the full
  model, preserves unmentioned fields, and emits a reviewable change audit.
- **General case-level revisions**: `case_design.overrides` applies validated
  `request_updates` or model operations to cases selected by their scan
  variables. The LLM returns the complete revised StudySpec, preserving both
  global and case-local edits. Transport adapters no longer reconstruct an
  incomplete scan or infer scientific choices from method keywords. Equivalent
  geometry aliases and explicit numeric ranges normalize in the shared contract;
  invalid or incomplete drafts remain errors.
- **Path-aware result refinement**: a non-smooth transition in a final
  one-dimensional energy path produces a local CASSCF/ActiveSpaceAudit plan.
  It is evidence about method stitching, not a replacement for a single-point
  correlation diagnosis.
- **Run-directory integrity**: the Planner locks the displayed concrete study
  directory at execution start, stores the execution root separately for retry
  and resume, and restores that lock per task session. Adaptive runs reserve a
  study id before dispatch so the visible directory is already exact.
- **Benchmark boundary**: the archived molecular dissociation comparison is a
  named regression/accuracy fixture. No production diagnostic threshold,
  active-space rule, scan router, or LLM repair is keyed to a particular
  molecule, equilibrium distance, or named benchmark.
- **Low-energy state workflows**: molecular FCI/CAS and finite-model FCI/block2
  accept explicit targeted root counts. The result contract separates requested
  roots from a complete spectrum, classifies low-energy degeneracy within the
  selected sector, and records whether more roots are needed.
- **DMRG state continuity and recovery**: explicit result analysis maps block2
  roots across cases from root 1RDM/natural-occupation evidence, excludes
  ambiguous anchors, and gates same-method MPS retries. Nonconvergence can
  increase bond dimension or sweeps within limits; Slurm OOM/timeout/node
  failures become per-case reports with resource-specific next actions.
- **Reference-versus-target convergence**: molecular reports distinguish the
  initial mean-field orbital guess from the requested correlated solver. SCF
  recovery preserves the requested SCF algorithm and reference choice and doubles the cycle limit,
  reusing the latest attempt's saved AO density when the reference policy is
  unchanged. Newton remains an explicit runtime choice. Unconverged source
  densities retain their convergence metadata.
  A converged CASSCF result is no longer marked failed solely because its
  initial UHF guess was unconverged; CASCI and single-reference methods retain
  the stricter reference-convergence requirement. Runtime recovery controls are
  preserved across ActiveSpaceAudit approval.
- **Registered study policies**: adaptive method/resource routing and path-
  continuity thresholds are hidden `study.policy` capabilities consumed by the
  Planner. The executable path no longer keeps a second constants table, and
  molecular FCI recovery requires PySCF-derived orbital and electron counts
  rather than atom- or benchmark-specific guesses.
- **Single DMET configuration authority**: the
  `model_hamiltonian.solver.dmet` Registry contract owns scientific defaults,
  validation metadata, and automatic-resolution policies. The composite
  `embedding.libdmet.dmet` module accepts no duplicate module configuration;
  all user choices travel through `TaskSpec.solver.options`.

## Molecular Hamiltonian Dataset Flow

MD tasks explicitly select `molecular_dynamics.profile=qh9` (SCF 1e-13) or
`qh9_relaxed_scf` (1e-8); the dataset planner propagates the selected
`compatibility_profile`. Existing standalone MD requests need this field
before another execution. Report reading remains compatible.

1. `HamiltonianDatasetSpec` and seed geometries compile through
   `StudyApplicationService` into one molecular-dynamics case per molecule.
   The default is 100 molecules with ten sampled frames each; resource review
   starts unapproved and covers trajectory work, not only retained samples.
2. The executable profile is restricted B3LYP/def2-SVP for neutral closed-shell
   H/C/N/O/F molecules, symmetry disabled, standard SCF, and no density fitting
   or active-space processing. NVE starts with 300 K Maxwell-Boltzmann
   velocities, uses a 50-atomic-time-unit step and 100 frames, and samples
   zero-based indexes `9, 19, ..., 99`. Initial temperature is not a thermostat.
3. Both `qh9` (`conv_tol=1e-13`) and `qh9_relaxed_scf` (`1e-8`) are executable,
   with `conv_tol_grad=3.16e-5`, `grid_level=3`, and `diis_space=8`. Outputs
   record the selected profile, runtime PySCF version, reference version 2.2.1,
   actual SCF setting deviations, and `bitwise_reproduction=false`.
   Sampling is independent of this choice: retaining the complete QH9-300k-style
   100-frame trajectory requires explicit `sample_stride=1`, `sample_offset=0`
   and dataset `geometries_per_molecule=100`.
4. Each task stores QH9 AO ordering, positions, sampled energies, temperature,
   times, velocities when requested, and Fock/overlap matrices in registered
   MD manifest/NPZ artifacts. The Fock matrix is a mean-field one-electron
   Hamiltonian label; overlap and AO metadata must accompany it.
5. Study finalization writes `samples.jsonl`, `rejections.jsonl`, and
   `dataset-manifest.json`. Every expected structure is accepted or rejected;
   rejected samples do not count as usable training data. Default molecule-level
   random splitting keeps all frames of a molecule in one split. Geometry-random
   and molecule-size out-of-distribution splits are explicit alternatives.
6. Finalization may consume remote trajectory metadata without opening or
   downloading the NPZ. The explicit `generate_hamiltonian_dataset` action
   materializes and validates a portable dataset on the selected execution
   target, rewrites matrix references relative to that dataset root, and returns
   a generation receipt. `collect_hamiltonian_dataset` uses successful generation
   evidence to retrieve a local copy only when requested. These are registered
   actions, not plot tools or a separate scheduler.
7. `select_energy_stratified_frames` selects a requested number of geometries
   per molecule from the accepted index, with a configurable time window and
   equal-population potential-energy bins. It exports a selection manifest,
   provenance JSONL and XYZ without loading matrices or starting calculations.
   The index must be accessible to the postprocessing process. The September 23
   pilot40 validation selected 780 frames from 39 successful molecules; see
   `reports/qm9_energy_selection_20260923/REPORT.md`.

A real one-frame H2 test covers task execution, typed matrix artifacts, and
finalization. Plan, rejection, remote metadata, generation, collection, and UI
contracts have regressions. No complete 100 x 10 numerical acceptance report,
trajectory energy-conservation benchmark, or downstream learning benchmark is
archived. See `docs/development/2026-09-03-qh9-md-dataset-and-runtime-identity.md`.

## Molecular Strong-Correlation Flow

1. Calculation Assistant active-space selection and Planner adaptive initial
   scans resolve the same Registry strategies and call the same molecular probe
   builder. `Auto` starts with HF/SCF chemical-valence or AVAS evidence and
   invokes the registered MP2 refinement when that proposal is unresolved or
   ambiguous, or SCF diagnostics show instability, frontier degeneracy, or a
   routing-boundary score. Explicit `MP2` and small-system `FCI` remain Planner choices.
   All strategies preserve the intended CASCI/CASSCF method, target solver, and
   solver options in `pyscf-agent.active-space-probe.v1`; temporary probe
   settings never become formal-calculation settings.
2. The backend produces HOMO/LUMO information, frontier degeneracy (including
   alpha/beta frontier comparison for unrestricted references), natural
   occupations when available, maximum T2 evidence, CCSD T1/D1 diagnostics,
   spin information, and an SCF stability artifact that records whether the
   optional check was requested.
3. `MolecularCorrelationRisk` keeps `physics_score` and `solver_stress_score`
   internal, and exposes separate `physics_level` and `solver_stress_level`
   alongside the overall level/confidence/recommendation. Nonconvergence is a
   hard solver-stress signal: it promotes the overall level to `strong` without
   rewriting the independently inferred physics level.
4. The router uses CCSD(T) for weak regions, CCSD for moderate regions, and
   CASSCF/CASCI candidates for strong regions or failed single-reference
   refinement. Diagnostic, physical, solver-stress, and routing levels remain
   separate: solver stress without strong physical evidence routes through
   CCSD first. The correlated occupation routing score is moderate from
   0.50 and strong from 0.80. Closed shells use 2/0 fractionality; open shells
   use deviations from the matching spin-adapted 2/1/0 determinant spectrum.
   Normal SOMOs remain CAS-selection evidence without forcing strong-correlation
   routing, and open-shell electron count alone is informational. Missing UNO
   evidence remains unavailable. Failed CCSD(T) cases retry with CCSD before CAS promotion. T1 >=
   0.05 or D1 >= 0.10 promotes a CCSD result directly to
   multireference review; smaller elevated values warn against perturbative
   triples. FCI recovery is guarded by conservative orbital/electron limits.
5. `ActiveSpaceAudit` records canonical orbital indices, energies, occupations,
   AO/atom contributions, atom-as-fragment contributions, localization metadata,
   selection reasons, `ncas`/`nelecas` consistency, multiple candidates, and
   approval details. Automatic proposals select the smallest valid chemically
   complete candidate first, sort reviewable alternatives by increasing `ncas`,
   and retain strong occupation/T2 evidence as an explicit next-expansion
   candidate. The policy has no element-specific or fixed-CAS branch. Ordinary
   CAS retains canonical or approved AVAS orbitals;
   approved block2 CASCI/CASSCF may explicitly localize its initial active block
   and apply a recorded canonical/Fiedler/manual solver ordering. CASSCF then
   optimizes the molecular orbitals through PySCF macroiterations.
6. Related molecular cases are resolved to one study-level chemical target and
   CAS dimension before approval. Each case keeps its own canonical-orbital
   mapping and initial orbitals; missing or conflicting cases return to the
   shared probe instead of receiving a fixed frontier space.
7. Ground-state SC-NEVPT2 is available after approved spin-adapted CASCI/CASSCF;
   it is blocked for unrestricted CAS and excited roots.
8. Molecular FCI, CASCI, and CASSCF accept `nroots`; multi-root CASSCF uses
   state-averaged orbital optimization in one configured spin/symmetry sector.
   Targeted roots are not presented as a complete spectrum.

Active-space candidates include manual, UHF natural-occupation-window,
chemical-valence baseline, evidence-expanded, AVAS AO/atom-fragment projection,
and merged evidence. AVAS preserves the PySCF orbital matrix as an approved
restricted/ROHF CAS initial guess; unrestricted AVAS and chemically named
fragment definitions remain future work.

Study-level resolution preserves explicit manual selections and their orbital
matrices ahead of automatic alternatives. Conflicting manual choices retain
their original contracts for review and block multireference execution.

## Model-Hamiltonian Strong-Correlation Flow

1. Model studies use a static full-grid scan with one explicitly selected
   solver contract. They do not expose molecular initial-scan or active-space
   controls.
2. Each report retains the original `case_id`, variables, solver, and
   diagnostics. Cross-case analysis may recommend a reviewed follow-up, but
   does not silently route cases through ActiveSpaceAudit.
3. Mean-field-based model solvers default to unrestricted references; the model
   spec always preserves sites, bonds, `nelec`, boundary, and units when a scan
   changes only selected parameters.
4. Hubbard diagnostics separate physical density evidence, solver stress, and
   parameter/state context. Physical scoring uses spin-channel nonidempotency
   and local double-occupancy suppression relative to matched spin populations.
   Spatial SOMOs, `U/t`, gaps, degeneracy and magnetic/charge order do not add
   physical score. T2/perturbative stress, invalid densities and nonconvergence
   affect solver warnings. Missing physical evidence is unknown, not weak.
   DMET scores covered fragment pairs with their local 1RDM baseline; assembled
   lattice occupations and impurity-SCF smearing remain informational. Scores
   are heuristic evidence summaries, not method-error or phase classifiers.
5. The primary comparison table favors final solver, final energy, correlation
   energy, filling, gap, diagnostic scalars when available, and total status.
   `screening_energy` is not a physical observable and is not a primary column.
6. Model FCI and block2 DMRG accept `nroots`. A low-energy manifold is classified
   within the requested particle/spin sector, and result analysis can track
   block2 root identity across ordered cases before checkpoint reuse.

The Builder now distinguishes `finite_cluster` and `bloch` representations.
Finite two-dimensional templates expose `Lx`/`Ly` down to one cell, preserve
periodic wrap hopping even when a direction contains one cell, and display the
outer cluster boundary plus primitive-cell subdivisions. Bloch templates use
primitive lattice vectors and integer intercell bond offsets instead of a
finite supercell graph.

These diagnostics characterize finite clusters. They must not be presented as
thermodynamic-limit phase boundaries without size, boundary, filling, and solver
trend checks. FCI diagnostics use full dense diagonalization only when the
Hilbert-space dimension is at most 5000; larger spaces are explicitly blocked
rather than silently switching to a partial eigensolver.

## Periodic Electronic-Structure Flow

1. A periodic task carries POSCAR or CIF structure text, a PBC Gaussian basis,
   GTH pseudopotential, HF/DFT method, and explicit numerical controls.
2. ASE expands CIF symmetry to an ordered P1 structure; the backend builds a
   three-dimensional `pyscf.pbc.gto.Cell` and validates basis/pseudopotential
   coverage for every element.
3. Gamma calculations use RHF/UHF or RKS/UKS; regular meshes use their k-point
   counterparts. FFTDF, GDF, MDF, and AFTDF are explicit choices.
4. Structured results distinguish total energy per cell, mean-field Fermi
   level, direct/fundamental gap, VBM/CBM, orbital energies/occupations, and
   smearing entropy/free/zero-temperature energies where applicable.
5. An optional ASE automatic, SeeK-path standardized primitive-cell, custom
   standard-point, or explicit-coordinate high-symmetry path evaluates
   mean-field bands with `get_bands`, separately from the SCF sampling mesh,
   and stores both absolute Hartree and Fermi-referenced eV values. SeeK-path
   calculations retain input/effective structures and transformation metadata,
   and report energy per standardized primitive cell.
6. The optional fcDMFT provider adds restricted periodic G0W0, approval-gated
   HF+DMFT, and staged GW+DMFT as single-task workflows. Native self-energy,
   continuation, localized-subspace, log, and checkpoint files remain registered
   artifacts. DOS, forces/stress, periodic MP2/CC, and Planner scans remain
   outside the executable ab-initio periodic boundary.

## Bloch Model-Hamiltonian Flow

1. A Bloch model spec contains a primitive basis, lattice vectors, integer bond
   `cell_offset` values, electrons per cell, k mesh, path, and DOS settings.
2. The backend constructs `h_ij(k) = sum_R t_ij(R) exp(2 pi i k.R)`, adds the
   requested Hermitian conjugates, diagonalizes each k point, and applies
   zero-temperature spin-degenerate filling.
3. Results include band energy per cell, Fermi energy, VBM/CBM, direct and
   indirect gaps, bandwidth, high-symmetry bands, Gaussian DOS, k-mesh
   occupations, and sampled complex Hamiltonians with plotting-ready artifacts.
4. This is a one-body solver. Nonzero `U` and `V` are retained and explicitly
   reported as unapplied; it is not a correlated Hubbard calculation and is not
   yet an adaptive Planner route.

## Implemented Capability Boundary

| Domain | Executable now | Important constraint |
| --- | --- | --- |
| Molecular methods | HF, DFT, MP2, CCSD, CCSD(T), FCI, CASCI, CASSCF, targeted low roots and state-averaged CASSCF | Post-HF methods require a valid mean-field reference; multi-root state averaging remains in one spin/symmetry sector. |
| Post-CAS | SC-NEVPT2 | Spin-adapted CAS and `root=0` only. |
| Molecular extras | 51 registered common basis sets, SCF density fitting, explicit restricted/ROHF DF-CASSCF with FCI or block2, Boys/Pipek-Mezey processing, 3D structure preview | Unrestricted DF-CASSCF is unsupported; fitted SCF does not make block2 CASCI active integrals density-fitted. Localization supplies initial active orbitals. |
| Periodic electronic structure | 3D POSCAR/CIF HF/DFT at Gamma or regular k mesh, optional high-symmetry bands, and optional fcDMFT G0W0/HF+DMFT/GW+DMFT single tasks | SeeK-path uses the standardized primitive cell. Correlated provider routes are restricted and not Planner-enabled; no correlated total energy is invented. |
| Model Hamiltonian | Finite Hubbard MP2, CCSD, CCSD(T), FCI, optional block2 DMRG; primitive-cell Bloch tight binding | FCI/block2 support targeted roots; block2 adds RDMs, entanglement, state tracking, bounded recovery, and explicit MPS continuation. Bloch `U/V` interactions are not applied and Holstein-Hubbard is not exposed. |
| Planner | static scan, adaptive scan, case design, recovery workflow, result merge, plots, restart-safe checkpoints, recoverable remote execution | Ready independent cases can use Slurm arrays; refined/dependent tasks remain one-at-a-time. Slurm owns remote limits; supervised local execution supports an optional per-task wall-time limit. |
| Workflow composition | Registered task/study modules, dependency compiler, runtime adapters, typed configuration, deterministic provenance, and execution traces | Task modules execute independently; study modules currently declare ownership by the established orchestrator. |
| Quality control | Compilation, input, resource, approval, execution-quality, artifact-integrity, continuation, and path-consistency gates | Gate thresholds remain heuristic and require benchmark calibration for production scientific claims. |
| Postprocessing | line, scatter, bar, heatmap; Hamiltonian dataset generation, collection, and energy-stratified frame selection actions | Plots retain PNG/spec/TSV/JSON artifacts. Dataset generation runs on the execution target; collection transfers an already generated dataset. Frame selection reads an accessible accepted index and exports geometries and provenance. |

## Runtime Registry And Artifact Contracts

The runtime capability registry is the executable source of truth for backend,
Planner, UI, and assistant prompts. In addition to methods, observables, basis
sets, pseudopotentials, and model operations, it now registers:

- study modes (`static`, `adaptive`), with adaptive routing limited to molecular Planner studies;
- molecular-dynamics jobs, trajectory observables, and separate dataset generation/collection actions;
- molecular initial-scan strategies (`auto`, `mp2`, `fci`), each selecting one
  method and one full-grid plan, with obsolete multi-stage, solver,
  screening-named, and point-limit fields rejected explicitly;
- task/study module contracts and task/study gate contracts;
- workflow options for cost review, restart-safe checkpoints, case recovery,
  Slurm independent-task batches, recoverable submitted execution, and merged
  postprocessing;
- block2 low-energy-root controls, shared result/recovery contracts, and
  cross-case state-tracking analysis;
- hidden adaptive-routing and scan-path-continuity policies consumed by the
  study orchestrator;
- DMET solver options, defaults, and automatic execution/fragment resolution
  policies, with no parallel module-level scientific configuration;
- artifact families, kinds, filename conventions, and payload schemas.

The registry is declarative: it states what can be selected and combined. The
workflow compiler turns those declarations plus `TaskSpec`/`StudySpec` into an
executable configuration. Runtime registries bind the compiled runtime or
evaluator IDs to trusted Python handlers. This separation lets one registered
capability be reused by the Calculation Assistant, Planner, local executor,
and Slurm workers without asking the LLM to generate execution code.

Every persisted artifact reference follows one machine-readable contract with
`kind`, `path`, `size_bytes`, `mime_type`, and `description`. Large provider
artifacts are registered without repeatedly reading them to compute a content
digest. JSON
payloads carry a versioned schema where a structured contract exists. Registered
contracts cover core task inputs/results, molecular diagnostics and
`ActiveSpaceAudit`, finite-model and Bloch outputs, periodic SCF/band outputs,
study/adaptive reports and decisions, cost estimates, checkpoints, comparison
tables, execution receipts, workflow/gate configuration and provenance,
module/gate traces, and plot specifications/raw data. Dynamic
per-observable artifact kinds
are covered by an explicit registry pattern instead of being undocumented
exceptions.

## Documentation Drift Corrected In This Update

The 2026-09-04 update adds the merged MD/dataset and runtime-identity boundary,
refreshes module/test counts, distinguishes cancellable Web child processes
from current-process execution, and records incomplete study-report/state
migration. It removes stale claims that every artifact reference has a content
digest. Historical numerical and deployment evidence retains its original date.
The following earlier corrections remain part of the maintained baseline.

- Replaced pre-modularization filenames such as `pyscf_backend_execution.py` and
  `adaptive_executor.py` with the current package layout.
- Replaced `static sweep`/`screening` wording with `static scan`/`initial scan`
  where this refers to planner workflow rather than physical screening.
- Corrected old claims that density fitting, CASSCF, and active-space audit were
  roadmap-only; their current restricted execution boundaries are now explicit.
- Added the adaptive workflow state machine, case-by-case failure recovery,
  plan-scoped ActiveSpaceAudit approval, subset-result merge, and postprocessing
  raw-data artifacts to maintained documentation.
- Added atomic per-study checkpoints with case fingerprints, bounded automatic
  retry, explicit unresolved-case reruns, and adaptive-stage reuse.
- Added protocol-neutral application services, local/direct-Slurm/SSH-Slurm
  executors, server-owned resource configuration, and job-array submission for
  independent cases.
- Added versioned execution receipts plus Planner inspect/collect actions so
  interrupted remote studies resume from existing scheduler handles instead of
  resubmitting completed tasks.
- Added CCSD T1/D1 diagnostics to molecular solver stress and routing while
  keeping the composite risk scores internal.
- Added ab-initio periodic `Cell` execution, resource-coverage validation,
  structured periodic artifacts, and explicit mean-field Fermi/gap semantics.
- Added optional fcDMFT G0W0, HF+DMFT, and GW+DMFT single-task workflows with
  approval-gated correlated subspaces, native provider artifacts, and explicit
  unavailable-total-energy semantics.
- Updated the Registry audit baseline to 298 typed entries, aligned provider
  schema constants with Registry artifact contracts, and removed the obsolete
  artifact-reference `sha256` requirement.
- Added optional automatic and validated custom high-symmetry mean-field bands
  using the same ASE/PySCF `bandpath`/`get_bands`/`get_fermi` convention as the
  PySCF 2.13 example, with structure-preview special points and dedicated
  JSON/TSV artifacts.
- Added explicit molecular/model targeted-root controls, low-energy degeneracy
  classification, and one common block2 result contract that records sector,
  convergence, resource, root-signature, and result-scope metadata.
- Added explicit cross-task DMRG state tracking from root 1RDM/natural-
  occupation evidence, state-trusted MPS continuation anchors, bounded
  bond-dimension/sweep recovery, and Slurm resource-failure recommendations.
- Added primitive-cell Bloch model construction, reciprocal-space
  diagonalization, plotting artifacts, and finite-cluster unit-cell rendering.
- Extended the optional block2 provider with registered composable modules for
  orbital entanglement, state-averaged excited roots, explicit MPS continuation,
  and spin/particle symmetry analysis, including structured artifacts and
  numerical regression tests.
- Added explicit Boys/Pipek-Mezey localization of approved molecular active
  blocks and canonical/Fiedler/manual block2 orbital ordering. Active-space,
  DMRG-result, and MPS-manifest provenance distinguish requested and executed
  transformations.
- Added a PySCF FCI-solver adapter for single-state and same-spin/symmetry-
  sector state-averaged DMRG-CASSCF. The adapter
  returns block2 1/2-RDMs to the orbital optimizer, continues from the preceding
  internal MPS, runs requested expensive diagnostics only on the final optimized
  active Hamiltonian, and records optimized orbitals plus macroiteration and
  solver traces. State-average weights determine the orbital objective while
  root energies and root RDMs remain structured outputs. Compatible cross-task
  runs can restore the optimized orbitals and MPS jointly with source-case and
  checkpoint provenance.
- Added bounded automatic DMRG schedule escalation, discarded-weight or final-
  sweep energy-error estimates, and entanglement/occupation-driven
  ActiveSpaceAudit expansion. Expanded spaces require approval, restore only
  optimized orbitals, and initialize a fresh MPS.
- Added exact fixed-`(Nalpha, Nbeta)` Schmidt-rank bond-dimension planning and
  registered its plan as a versioned artifact. The executor trims impossible
  or redundant schedule values before invoking block2.
- Replaced separate molecular Planner and browser-side active-space probe
  builders with one Registry-backed `Auto`/`MP2`/`FCI` role and versioned probe contract.
  Temporary unrestricted probe requirements remain separate from the approved
  CASCI/CASSCF method, block2/FCI target solver, and target solver options.
- Changed molecular `Auto` to an HF/SCF-first registered workflow with
  conditional MP2 refinement, and added a deterministic study-wide chemical
  active-space policy that preserves case-specific orbital mappings. Fixed
  frontier CAS defaults are not part of the routing path.
- Added named execution targets and server-owned resource-profile selection to
  both Web surfaces. One private `server-slurm.ini` now namespaces multiple
  servers and their resource profiles, and SSH RPC explicitly selects and
  verifies the intended cluster profile.
- Added automatic large-active-space routing to block2 DMRG-CASSCF,
  DMRG-specific resource estimates in ActiveSpaceAudit review, and
  an explicit study-level MPS continuation approval that preserves source-case,
  executor, quantum-sector, checkpoint, and Hamiltonian provenance.
- Added Assistant visual summaries and Planner/postprocessing scalar fields for
  DMRG roots, orbital entropy, mutual information, bipartite entanglement,
  symmetry, convergence, and restart provenance.
- Replaced the old claim that primitive vectors were future Builder work; they
  are now executable for Bloch inputs, while explicit per-site cell indices and
  twist metadata remain future work.
- Added canonical Planner initial-scan IDs and explicit rejection of removed
  strategy/solver fields, so backend, UI, prompts, and saved plans share one
  unambiguous vocabulary.
- Added explicit molecular/model physics-versus-solver-stress levels and made
  nonconvergence a strong overall-risk signal without relabeling the inferred
  physical regime.
- Added general variable-selector case revisions, batch ActiveSpaceAudit review
  per executable plan, explicitly requested path-continuity analysis and local
  refinement, and run-directory locking with a separately retained execution
  root. Adaptive execution itself preserves independent case calculations.
- Added registry-backed artifact contracts, versioned JSON schemas, MIME types,
  descriptions, and dedicated strong-correlation, periodic, Bloch,
  adaptive-decision, cost, comparison, and postprocessing data artifacts.
- Postprocessing now merges recovered fields into the complete study table,
  plots successful scientific rows only, excludes execution metadata from plot
  selectors, presents `final_energy` as `Energy`, and avoids duplicate heatmap
  title/colorbar labels while retaining explicit custom labels.
- Made FCI strong-correlation diagnostics an explicitly bounded full dense
  diagonalization contract (`dimension <= 5000`).
- Added first-class task/study module contracts with typed ports, compatibility,
  configuration, dependency/conflict rules, deterministic compilation, runtime
  adapters, and complete workflow provenance/execution traces.
- Added first-class task/study gate contracts for compilation, readiness,
  resources, approvals, execution quality, artifact integrity, continuation
  state, and scan-path consistency.
- Consolidated method support under the capability registry. Any unregistered
  method now follows the same generic pre-execution validation failure.
- Limited SCF stability analysis to the lowest response root. This preserves
  the stability decision while avoiding invalid multi-root requests for tiny
  orbital spaces.

## Upstream PySCF Example Baselines

The implementation was checked against a local PySCF 2.13.0 source snapshot.
The maintained relative example mapping is in `docs/pyscf_example_baselines.md`.
The primary references are:

| Contract | Upstream examples |
| --- | --- |
| Cell, Gamma, and k-mesh SCF/DFT | `examples/pbc/00-input_cell.py`, `10-gamma_point_scf.py`, `20-k_points_scf.py` |
| Smearing and PBC density fitting | `examples/pbc/23-smearing.py`, `35-gaussian_density_fit.py` |
| Mean-field band path | `examples/pbc/09-band_ase.py` |
| Future GW | `examples/pbc/22-k_points_gw.py` |
| Periodic post-HF | `examples/pbc/12-gamma_point_post_hf.py`, `22-k_points_mp2.py` |
| MP2 natural orbitals and CAS references | `examples/mp/12-dfump2-natorbs.py`, `examples/mcscf/11-casscf_with_uhf_uks.py`, `60-uhf_based_ucasscf.py` |
| AVAS, DMET-style CAS, and DMRG | `examples/mcscf/43-avas.py`, `43-dmet_cas.py`, `50-casscf_then_dmrgscf.py` |
| Finite-Hamiltonian FCI and density matrices | `examples/fci/01-given_h1e_h2e.py`, `14-density_matrix.py` |

These examples define API and physical-output conventions. Routine regression
tests should use reduced deterministic fixtures rather than run expensive or
optional-dependency examples verbatim.

## Formal Scientific Benchmarks

The versioned benchmark CLI now provides eight maintained default definitions;
the Si G0W0 definition is always reported but only executes after explicit opt-in:

| Benchmark | Execution | Reference and checks |
| --- | --- | --- |
| `n2-dissociation-sto3g` | Archived 37-point adaptive curve | Weaving et al., *npj Quantum Information* 11, 25 (2025), DOI `10.1038/s41534-024-00952-4`; CCSD(T)/CCSD overlap errors and one expected method-boundary continuity anomaly. |
| `hubbard-dimer-ed` | Direct finite-model FCI/ED at `U=0,2,4,8`, `t=-1` | Analytic half-filled two-site singlet energy and mean double occupancy at `1e-10`. |
| `periodic-he-gamma-rhf` | Direct public-backend periodic PySCF run | PySCF 2.13 GTH-SZV/GTH-Pade Gamma-point He baseline; convergence, energy per cell, Fermi/VBM energy, and artifacts at `1e-8 Ha`. |
| `fcdmft-si-g0w0` | Explicit optional-provider public-backend run | fcDMFT `examples/Si/si_gw.py` two-atom Si cell, GTH-DZVP/GTH-PBE, PBE reference, `4x4x4` k mesh, Pade continuation, full self-energy, finite-size correction, quasiparticle-gap sanity range, no-fabricated-total-energy semantics, and required GW artifacts. |
| `block2-h4-casci` | Optional direct molecular DMRG-CASCI run | Linear H4/STO-3G CAS(4,4) block2 energy against the identical PySCF direct-FCI active Hamiltonian at `1e-8 Ha`. |
| `block2-h2-casscf` | Optional direct molecular DMRG-CASSCF run | Stretched H2/6-31G CAS(2,2) orbital-optimized block2 energy and convergence against PySCF FCI-CASSCF at `1e-8 Ha`. |
| `block2-hubbard-ring` | Optional direct finite-model block2 run | Four-site half-filled Hubbard ring ground/first-excited energies, RDM observables, entanglement, symmetry, and changed-Hamiltonian MPS continuation against full diagonalization/cold restart. |
| `block2-adaptive-workflows` | Optional direct workflow benchmark | Joint H2 DMRG-CASSCF orbital/MPS continuation, entanglement-driven unapproved ActiveSpaceAudit generation, and adaptive eight-site Hubbard-ring bond-dimension escalation against exact diagonalization. |

The archived 2026-08-11 formal run passed the five benchmarks available at that
time. N2 maximum overlap
errors were `0.193828 mHa` for CCSD(T) and `0.015448 mHa` for CCSD; Hubbard
energy and double-occupancy errors were below `7.2e-15`; periodic energy and
Fermi-energy deviations from the frozen local PySCF 2.13 baseline were zero at
reported precision. The JSON record is
`reports/benchmarks/formal-benchmark-2026-08-11.json`. H4 DMRG-CASCI agreed
with PySCF FCI within `9e-12 Ha`; the Hubbard
ring energy and first excitation agreed within `9e-13 Ha`, RDM-observable error
was below `1e-7`, and warm/cold `U=6` restart energies agreed within `3e-13 Ha`.
The current benchmark also verifies two requested roots at both Hubbard points,
the common molecular/model block2 result contract, and non-empty cross-case
state mapping from root 1RDM/natural-occupation signatures.

The 2026-08-12 adaptive-workflow report passed separately. Warm/cold H2
DMRG-CASSCF energies differed by `3.418e-10 Ha`; adaptive eight-site Hubbard
DMRG differed from exact diagonalization by `1.296e-9 Ha` after increasing the
bond dimension from 16 to 128. The ActiveSpaceAudit check verified an
unapproved, costed orbital-only expansion request. The record is
`reports/benchmarks/block2-adaptive-workflows-2026-08-12.json`.

The N2 fixture is intentionally a reference-data regression rather than a
routine full recomputation. No production threshold, routing rule, or
active-space choice is keyed to N2. New external solvers must add direct
numerical benchmarks before registry promotion.

Two opt-in `tier=scaling` campaigns extend the block2 evidence without slowing
the default suite. Boys-localized, Fiedler-ordered H6/H8/H10 STO-6G chains
agree with PySCF FCI within `2.014e-9 Ha`; the H20 M=64-to-128 change is
`2.627e-9 Ha`. For Hubbard rings, the eight-site DMRG/ED error is
`1.182e-10 Ha`; 12/16-site runs converge below a maximum discarded weight of
`1.203e-4`, and increasing the 16-site bond dimension lowers the variational
energy by `0.018004 Ha`. The machine-readable record is
`reports/benchmarks/block2-scaling-2026-08-11.json`.

## Known Risks And Recommended Next Work

### Priority 0: Freeze And Validate The Current Platform

1. **Study report authority**: preserve Study -> Task -> Run ownership and
   extension-safe report updates. Run/Collect read current task checkpoints and
   retain the full plan/report across subset retries. Assess specialized stage
   writers independently; a report-revision engine is not required for retries.
2. **Executable study stages and runtime ports**: migrate orchestrator-owned
   study bindings to explicit typed stage adapters. Task handlers currently
   record declared outputs without validating realized port values; add a
   port-to-state resolver and required-output schema checks.
3. **Release and deployment evidence**: retain additive-v1 rules and report
   migration fixtures. Version 0.2.0 wheel/MCP checks and source-matched remote
   workflow campaigns are recorded below and in the current consolidation
   report. Extend this evidence to target wheelhouses and supported Python
   versions. No Git release tag or GitHub Actions workflow is added here.

### Priority 1: Scientific Validation And Reliability

4. **Dataset acceptance**: run and archive a complete MD campaign with accepted,
   rejected, and pending counts, trajectory energy-drift checks, verified AO
   matrix conventions, portable generation/collection, and reproducible splits.
   Keep QH9 compatibility and numerical-reproduction claims separate.
5. **Benchmark expansion**: extend the existing block2 hydrogen-chain, Hubbard,
   and adaptive-workflow evidence with recomputed H2/N2, frustrated clusters,
   periodic k-mesh/smearing/band comparisons, and converged GW/DMFT references.
6. **Cost and timeout calibration**: calibrate FCI/CAS/DMRG dimensions, memory
   proxies, MD trajectory work, and wall time against local and Slurm hardware.
   `LocalProcessExecutor` now persists its optional wall-time limit and enforces
   it independently of the Web process. The CLI flag selects that executor for
   local synchronous work; the direct in-process executor remains unbounded.
7. **Active-space and path quality**: improve chemical fragments, orbital
   visualization and candidate comparison, cross-geometry mappings, occupation
   and entropy trends, and transition-RDM/wavefunction state overlaps. Preserve
   the separation between physical correlation, solver stress, root identity,
   and cross-case curve continuity.

### Priority 2: Expand The Scientific Boundary

8. **Periodic studies**: promote periodic HF/DFT into Planner scans after
   cell/k-mesh identity, normalization, band comparability, resource limits,
   and remote artifacts have dedicated study tests.
9. **External solvers**: extend block2/libDMET to scheduler-scale restart
   campaigns and broader finite-graph normalization/convergence benchmarks.
   fcDMFT needs converged physical and restart evidence before adaptive routing;
   retain the explicit absence of a correlated total-energy estimator.
10. **External API distribution**: task and Study stdio MCP tools, nonblocking
    submission, reviewed subsets and report/wiki resources are implemented.
    Static and adaptive execution reuse one background application call.
    Review decisions are now persisted in StudyReport; scientific analysis,
    reviewed continuation and MD dataset preparation use the existing services.
    The initial thirteen-tool surface has source and installed-wheel acceptance below;
    newer workbench/Study/Task view tools have their own dated evidence.
    Generic remote submission reconciliation remains separate work.

## Verification Record

### Source Consolidation: 2026-09-21

The full local suite ran 1,228 tests: 1,165 passed and 63 skipped, with no
failures or errors. A 25-test follow-up covers verification and Wiki
curation/retrieval. Thirteen changed JavaScript/CJS files pass syntax checks;
the current Registry has 318 entries and zero validation issues.

A fresh wheel imports outside the checkout and passes resource, entry-point,
44-page Wiki and two scientific smoke checks. The temporary installation reuses
the selected environment's scientific dependencies; an isolated offline
wheelhouse is not certified. Wiki lint reports zero errors and three citation
warnings. Native optional-provider and Codex host-rendering boundaries remain
explicit. No new remote deployment or scientific job was performed during
this consolidation.

See the development report (author archive: `reports/development-status-2026-09-21.md`),
verification record (author archive: `reports/verification/consolidation-2026-09-21.json`)
and clean-wheel evidence (author archive: `reports/verification/consolidation-clean-wheel-2026-09-21.json`).

### Saved MCP Workflows: 2026-09-07

Reviews now survive client restart and are executed by Study ID. Scientific
analysis and optional plots do not need LLM credentials. MD dataset preparation
and manifest resources use the existing dataset planner and Study finalizer.
Changing retry inputs now refreshes the cost estimate before approval, fixing
an approval loop caused by the previous plan fingerprint.

The Python 3.12 focused suite passed 189 tests. An isolated local campaign
completed ten real H2 Task runs: mixed convergence, one selected retry with
attempts `[2, 1]`, adaptive execution, a two-frame MD dataset and probe-to-CASSCF
approval. Every MCP call reconnects through a fresh protocol server. The final
installed-wheel campaign passed the same ten runs outside the checkout. The
source-matched Amarel campaign passed 17 checks across ten Task runs, including
collection after coordinator termination, with no duplicate initial submission.
The Python 3.9 source suite ran 1,015 tests: 993 passed and 22 skipped; Python
3.12 supplies the supported-version installation evidence. Wiki lint retains
zero errors and three existing warnings. Exact evidence and the isolated remote
release are in the
[workflow implementation record](../development/2026-09-07-mcp-study-workflows.md).

### Thin MCP Agent Interface: 2026-09-07

The per-task nonblocking submission branch has been removed. One MCP start
invokes the existing static/adaptive runner in a detached application process.
Direct-CAS preparation and probe-to-review conversion now belong to the same
application service used by Web. Adaptive option normalization was made
idempotent to preserve the prepared Study ID on resume.

The Python 3.12 focused suite passed 192 tests. Real stdio and installed-wheel
checks cover one-start static/adaptive H2 execution after client exit, a reviewed
subset retry, and probe-to-approved-CASSCF execution. Wiki lint has zero errors
and the same three existing warnings. The Python 3.9 source suite ran 1,010
tests with 990 passes and 20 skips; supported-version installation evidence is
from Python 3.12. No cluster jobs were submitted. See the
[implementation record](../development/2026-09-07-mcp-agent-interface.md) and
`reports/verification/mcp-agent-2026-09-07-*` for exact evidence and boundaries.

### Study MCP Interface: 2026-09-06

The Study follow-up passed 108 focused tests on Python 3.12.14 with MCP 2.1.1
and PySCF 2.13.1. Real stdio tests cover partial and full collection across
server restarts and a reviewed subset retry: two H2 tasks retain three runs
with attempts `[2, 1]`, and the unselected result is unchanged. A newly built
wheel passed that workflow through its installed command outside the source
checkout, exposing eleven tools and 43 wiki pages. Wiki lint remains at zero
errors and three existing warnings. No new cluster acceptance was performed.

The existing Python 3.9.6 source environment ran 1,009 tests with 991 passes
and 18 skips. It remains source-regression evidence rather than supported-version
installation verification; the Python 3.12 full-suite limitations recorded below
have not been reclassified as passing.

See [the Study MCP record](../development/2026-09-06-mcp-study-interface.md) and
`reports/verification/mcp-study-2026-09-06-*` for evidence and remaining scope.

### Single-Task MCP Interface: 2026-09-06

The final focused Python 3.12.14 suite passed 87 tests, including 16 new MCP
and structured-preparation tests. A real stdio client submitted H2/STO-3G HF,
restarted the server, and collected energy `-1.1167593073964255` Hartree without
another execution. Worker cancellation after a server restart is also covered.
A built wheel passed the same H2/reconnect acceptance through its installed
console command outside the source checkout, with MCP 2.1.1 and PySCF 2.13.1.

The existing Python 3.9.6 source environment passed 979 of 992 tests, with
13 skips; this is not supported-version installation evidence. The exploratory
Python 3.12 full suite exposed 13 missing-libDMET fixture errors and one
block2 root-count failure; representative failures reproduce on Git baseline
`48a0478`. An additional root-cache archive exclusion failure was fixed and
passes focused distribution checks. The supported-environment full suite is
therefore not claimed as passing. Wiki lint reports 0 errors and the existing
3 citation warnings. No new Amarel acceptance was performed.

See [the implementation record](../development/2026-09-06-mcp-single-task-interface.md)
and `reports/verification/mcp-2026-09-06-*` for the exact scope and evidence.

### Study, Task, And Run Organization: 2026-09-06

The current local correction removes independent review child Study IDs,
registration, and the separate integration API. Review subsets retain the
original Study identity; only selected tasks acquire new runs. The saved plan,
checkpoint, full report, and comparison TSV retain every task. Collect reloads
the saved plan by Study ID and cannot submit calculations. Completed legacy
results can enter the original checkpoint without moving run artifacts.

The focused suite passed 107 tests; the full suite ran 976 tests in 105.228
seconds with 975 passes and 1 skip on Python 3.9.6. Wiki curation generated 43 pages with
0 lint errors and the same 3 citation warnings; JavaScript syntax and diff
whitespace checks passed. All 35 checked local documentation links resolve.
Affected production files have 123 fewer lines than at the start of this task,
including earlier local changes in that comparison baseline. Verification
results are recorded in the [implementation record](../development/2026-09-06-study-task-run-organization.md)
and `reports/verification/study-task-run-2026-09-06-*` outputs. The earlier two
records below describe superseded local implementations and keep their original
verification counts.

### Study Task-Management Simplification: 2026-09-06 (Superseded)

The next local follow-up removes full-report merge arguments, cached-child-run
comparisons, and the second checkpoint traversal. Remote and direct results
share complete-batch acceptance before checkpoint updates. Regression coverage
includes restored Web collection without a browser parent, contradictory cached
child-report values, preserved report evidence, and malformed batches that must
not partially update state. The implementation retains child Study containers,
execution receipts, parent case references, and existing submission/cancellation
controls. See the [simplification record](../development/2026-09-06-study-task-management-simplification.md)
and `reports/verification/study-task-simplification-2026-09-06-*` outputs.

Verification: 101 focused tests passed; the full source-tree suite ran 983 tests
with 982 passes and 1 skip on Python 3.9.6. Wiki generation produced 43 pages;
lint reported 0 errors and 3 existing citation warnings. JavaScript syntax,
diff whitespace, and 31 local documentation links passed checks. The five
affected production files have 57 fewer lines than at the start of this local
follow-up. No new supported-version package or remote-cluster acceptance is
claimed by that historical check; those changes were still uncommitted and
unpushed at its September 6 inspection.

### Study Child-Task References: 2026-09-06 (Superseded)

The next local follow-up registers review children under parent cases and reads
their current TaskReports from the existing execution checkpoints. Regression
coverage includes stale attempts, replaced children, pending retries, restart
collection, malformed case identities, independent subset updates, report-path
separation, and partial sequential collection. The focused suite passed 84 tests.
The final full suite ran 977 tests: 976 passed and one opt-in benchmark was
skipped. Wiki lint reports zero errors and the same three citation warnings.
See the [implementation record](../development/2026-09-06-study-child-task-references.md)
and `reports/verification/study-task-references-2026-09-06-*` outputs for the
full verification boundary.

### DMET Option And Quality-Evidence Validation: 2026-09-06

The local follow-up to `48a0478` rejects unknown top-level DMET solver options
using Registry names and validates the operands required by each result-quality
operator. Missing equality evidence, boolean/number confusion, non-finite
values, and numeric conversion overflow cannot produce a passing required
constraint. Valid option normalization, optional checks, and legacy results
retain their existing behavior. The full suite passed 958 tests with one opt-in
skip (959 total); the focused suite passed 85 tests. Wiki lint reports zero
errors and the same three citation warnings. See the
[validation record](../development/2026-09-06-dmet-option-and-quality-validation.md)
for the compatibility boundary and archived output paths.

### DMET Quality PR Integration: 2026-09-06

PR #1 adds occupation-aware DMET baths and projected impurity-sector selection,
requires energy, consecutive mean-field 1RDM, and density-fit convergence, and
adds provider-owned DMET/block2 quality evidence evaluated by task gates.
Study rows preserve `quality_status` and `publication_eligible`; built-in plots
exclude ineligible rows while retaining the raw results for review. The changes
merge with the retry/process-control implementation without textual conflicts.
The 4x4 Hubbard protocol (author archive: `reports/dmet-4x4-article-case.md`) records the
remaining scientific boundaries: per-run eligibility does not establish a
stability window, bidirectional-scan agreement, or fragment/finite-size
convergence. The integrated suite ran 949 tests in 97.741 seconds: 948 passed
and one opt-in benchmark was skipped. After synchronizing the curated DMET and
plotting guidance, all 28 knowledge tests passed. Registry validation reports
317 entries and zero issues; wiki lint reports zero errors and the same three
existing citation warnings. Integration checks use the
`reports/verification/pr1-integration-2026-09-06-*` prefix. These are source-tree
checks on Python 3.9.6, not a new supported-version wheel or live-cluster run.

### Local Process Control And Retry Recovery: 2026-09-06

The current implementation persists verified POSIX process identities, controls
calculations after submitter exit, enforces an optional wall-time limit, and
returns a failed TaskReport to synchronous Study callers after a timeout. SSH
submission reconciliation verifies server evidence without submitting work.
A real local H2/HF/STO-3G task succeeded at -1.1167593073964255 hartree.
The full suite passed 931 tests with one opt-in skip (932 total). The real
Slurm campaign passed 11 final checks across 10 small tasks on isolated
source-matched Amarel releases. It exposed and fixed duplicate concurrent Run
submission and premature failure synthesis before TaskReport visibility. No
recorded job remains queued or running. The compact result is
`reports/verification/retry-live-2026-09-06-summary.json`.
See the [implementation and acceptance record](../development/2026-09-06-local-process-and-retry-acceptance.md)
for final regression counts and precise evidence boundaries.

### Local StudyReport Conversion Fix: 2026-09-05

The report compatibility fix adds six contract regressions and strengthens the
existing numerical adaptive report test. Coverage includes both schemas,
schema-less reports, additive fields, optional null/empty values, independent
copies, malformed input, and the actual persisted adaptive report. Verification
records use `reports/verification/study-report-2026-09-05-*` filenames.
The full suite ran 916 tests in 64.194 seconds: 915 passed and one opt-in
numerical benchmark was skipped. Wiki validation reports zero errors and the
same three existing citation warnings; Registry validation remains clean.
The source-tree Python support-floor and live-remote limitations still apply.

### Local Retry And Collection Fixes: 2026-09-04

The updated working tree ran 910 tests in 64.476 seconds: 909 passed and the
opt-in numerical benchmark was skipped. This includes 14 new retry/collection
regressions and the eight preceding merge/dataset/run-id regression tests.
Output: `reports/verification/study-retry-2026-09-04-unittest.log`.

The read-only Registry audit reports 317 entries and zero issues; the JSON record
is `reports/verification/study-retry-2026-09-04-registry.json`. Wiki regeneration
produced 43 pages, with zero lint errors and three existing citation warnings;
output is `reports/verification/study-retry-2026-09-04-wiki.log`.
`git diff --check` passed. The Python support-floor and live-remote evidence
limitations described below still apply.

### Current Source Baseline: 2026-09-04

For source commit `33c7610`, the repository inspection ran:

```bash
python3 -B -m unittest discover -s tests -t . -p 'test*.py'
# Ran 888 tests in 225.165s; OK (skipped=1)
# 887 passed; opt-in 18-site libDMET numerical benchmark skipped
```

The original output is retained in
`reports/verification/unittest-2026-09-04.log`. The environment was Python
3.9.6, PySCF 2.13.0, and block2 0.5.3. The package declares Python >=3.10;
this source-tree run does not replace supported-version wheel acceptance.
Matplotlib/PyParsing, LibreSSL, LangGraph, and libDMET resource warnings were
emitted without failures. The suite combines contract, controlled-transport,
and real numerical tests; a green suite does not establish every optional
scientific benchmark or a full MD dataset campaign.

A separate read-only check found 29 task modules, 13 study modules, five task
gates, seven study gates, and no Registry consistency issues. The study report
extension-loss probe reproduced dropped `adaptive` and `postprocessing` fields
at that baseline. The September 5 local conversion fix below resolves it. Cr2/def2-TZVP eleven-point
planning and synthetic CAS(12,12) mapping tests remain planning-contract
checks, not a numerical Cr2 curve.

### Archived Evidence

- `reports/verification/install.json`: successful 0.1.0 wheel verification on
  2026-08-11, using existing system scientific dependencies. It does not verify
  current 0.2.0 or the isolated offline installation path.
- `reports/benchmarks/formal-benchmark-2026-08-11.json`: five passing
  N2/Hubbard/periodic/block2 definitions available at that time. The N2 portion
  is archived-reference regression; see the benchmark section for numerical
  tolerances and direct solver comparisons.
- `reports/benchmarks/block2-scaling-2026-08-11.json` and
  `block2-adaptive-workflows-2026-08-12.json`: passing scaling and adaptive
  numerical campaigns, with their original metrics preserved.
- `reports/verification/remote-batch-2026-08-11.json`: two independent H2/HF
  tasks in one Slurm batch, exact mapping, scientific reports, and scheduler
  artifacts verified in 63.386 seconds.
- `reports/verification/amarel-2026-08-16.json`: a later successful remote
  capability/contract verification record. The older `remote.json` records a
  failed hostname lookup and is not a current remote-health signal.

These archived files were not regenerated or relabeled during this update.
No live remote deployment, new converged GW/DMFT benchmark, or complete MD
campaign was performed for the documentation refresh.

### Documentation And Wiki Refresh: 2026-09-04

The maintained source pages and `curated_structure.json` were updated before
running `npm run wiki:curate`. The generator produced 43 curated pages, the
packaged runtime JSON, and local Markdown/HTML review exports. No generated
page or runtime JSON was edited by hand, and no external LLM extraction was
needed for this source-based refresh.

`npm run wiki:lint` completed with zero errors and three inferred-paragraph
citation warnings on existing density-fitting/local-orbital, multireference,
and scientific-benchmark pages. The runtime export has no broken internal page
links. Retrieval checks put the new MD dataset page first for a QH9/Fock/overlap
dataset query, the remote verification page first for a deployment-identity
query, and Public Data Contracts first for a StudyReport round-trip query.

The focused refresh check passed all 37 tests in 6.545 seconds:

```bash
python3 -B -m unittest \
  tests.computational_study_agent.test_llm_wiki \
  tests.pyscf_agent.test_public_api_boundaries \
  tests.pyscf_agent.test_registry_migration_audit
```

Its output is retained in `reports/verification/wiki-refresh-2026-09-04.log`.
`git diff --check` also passed. Numerical Python implementation files were not
changed; the full 888-test result above belongs to the preceding source
inspection, while these focused tests verify the refreshed packaged Wiki.
