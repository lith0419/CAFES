# CAFES Design Philosophy

## Purpose

CAFES turns scientific requests into reviewable, reproducible
calculations without allowing a language model, browser session, or transport
adapter to become the authority for numerical science. The architecture is
organized around explicit contracts, executable capabilities, controlled
state transitions, and persisted evidence.

## One Sentence

The LLM interprets intent, application services coordinate use cases, typed
contracts carry data, the registry defines what can execute, state machines
control when it may execute, providers perform the calculation, and artifacts
preserve the evidence.

## Core Principles

### 1. Public APIs Express Scientific Concepts

The package-root API exposes stable concepts, not workflow implementation
steps. The calculation root exports `CalculationApplicationService`,
`TaskSpec`, `TaskReport`, and the optional registry entry points. The study
root exports `StudyApplicationService`, `StudySpec`, `StudyPlan`, `StudyCase`,
and `StudyReport`.

Executors, request parsers, workflow nodes, retry helpers, registry builders,
and provider adapters remain in their owning namespaces. Removing an internal
step must not require application clients to rewrite their scientific code.

### 2. The System Has Two Orchestration Levels

The task level owns one complete scientific calculation. It validates one
`TaskSpec`, executes the required modules and gates, and returns one
`TaskReport`.

The study level owns relationships among complete tasks. It expands a
`StudySpec` into a `StudyPlan`, schedules cases, compares results, requests
review, performs approved refinement, and produces one merged `StudyReport`.
Cross-case continuity, recovery, and method promotion belong here rather than
inside an individual task.

Open-ended research planning above a study is deliberately outside this
package. External agents such as Codex may compose studies through the public
application API without becoming part of the numerical runtime.

### 3. Application Services Are the Inbound Boundary

Web UI, CLI, local Python, remote RPC, and the MCP adapter call the same
application services. Adapters may translate transport fields and present
errors, but they must not assemble an alternate scientific execution path.
MCP is an interface through which Codex calls the project agent. Static task
sequencing and adaptive diagnostics, routing, refinement, recovery and review
remain application/runner responsibilities. A detached application call may
outlive its client; its process metadata cannot become another task state store.

`CalculationApplicationService` owns one-task use cases.
`StudyApplicationService` owns multi-task use cases. Execution targets and
resource profiles are injected infrastructure, not scientific fields hidden
inside `TaskSpec`.

### 4. Contracts Are Protocol-Neutral

`pyscf_agent.contracts` owns the task dataclasses and task serialization
helpers. `pyscf_agent.schema_contracts` owns public schema identifiers,
version compatibility, and envelope validation. Study contracts live in
`computational_study_agent.schema`.

The same `TaskSpec` and `TaskReport` cross local, Slurm, SSH-RPC, Web, and
MCP boundaries. Scheduler status is not scientific status, and large
process-local numerical objects are never public payloads.

### 5. Registry, Wiki, and UI Have Different Authority

The `PlatformRegistry` is the executable fact source. It defines capabilities,
parameters, observables, modules, gates, providers, bindings, artifacts,
option sets, and non-limiting workflow templates.

The LLM Wiki explains scientific reasoning, workflow policy, limitations, and
failure patterns. It cannot enable a method. The UI owns labels, layout, and
interaction. It cannot make an unsupported capability executable.

These boundaries prevent prose, presentation defaults, and runtime behavior
from becoming competing capability catalogs.

### 6. Registry Relationships Are Hierarchical

A module owns a reusable workflow step. A capability is a requestable
scientific operation owned by a module. A provider supplies an implementation
boundary, and a binding connects a module to its runtime adapter. Parameters
and option values configure these entries; they are not modules themselves.

Workflow construction proceeds top-down from the requested capability to the
required modules, gates, and artifacts. Runtime validation proceeds bottom-up
by confirming that every module has compatible inputs, a provider binding,
and declared outputs.

Providers may be fine-grained or composite. If an upstream library owns one
inseparable self-consistent loop, the registry describes its operations but
binds one composite executable module instead of inventing artificial
intermediate APIs.

### 7. State Machines Own Lifecycle, Not Scientific Data

State machines make preparation, review, approval, execution, recovery,
analysis, and completion explicit. Gates block invalid, unaffordable, or
unapproved transitions. Browser state and conversation history are projections
of that lifecycle, not the merge authority.

Workflow state remains compact. It carries status, decisions, review context,
and artifact references. Numerical arrays, generated scripts, logs,
checkpoints, and detailed results belong in artifacts and reports.

### 8. LLM Decisions Are Bounded by Deterministic Checks

The LLM may interpret natural-language requests, draft structured parameters,
explain validation failures, and summarize structured results. It must not
invent energies, silently enable unavailable methods, bypass approval, or
replace deterministic validation.

Scientific method selection combines registered capability limits, numerical
diagnostics, solver evidence, and explicit human review for consequential
operations such as active-space approval, expensive execution, and cross-case
continuation.

### 9. Tasks Start Independent; Studies Add Relationships Explicitly

Initial cases in a scan are independent unless their contract declares a data
dependency. Cross-case 1RDM or MPS continuation, continuity analysis, active-
space expansion, and local method promotion occur only in explicit study
analysis or refinement stages. Approved subset results merge back into the
authoritative parent `StudyReport`.

This preserves reproducibility while still allowing the agent to use the
physical relationships among a series of calculations.

### 10. Reproducibility Is Persisted Evidence

Every execution produces a structured report and registered artifacts. The
run directory preserves generated PySCF scripts, normalized inputs, logs,
numerical arrays, checkpoints, and plot data when applicable. Reports contain
compact scientific summaries and artifact references rather than duplicating
large files.

The same normalized request must produce the same execution path independent
of whether it entered through the Web UI, CLI, local API, or remote executor.

### 11. General Rules Beat Benchmark-Specific Fixes

Diagnostics, routing, active-space selection, continuation, and recovery must
be expressed through system-independent contracts and registered scientific
evidence. A benchmark such as a bond dissociation curve may test a rule, but
must not introduce molecule-specific thresholds or hidden branches into the
runtime.

### 12. There Is One Authority Per Concern

New work must extend the existing owner rather than add a compatibility facade
or parallel catalog:

| Concern | Authority |
| --- | --- |
| One calculation request/result | `TaskSpec` / `TaskReport` |
| Multi-task request/result | `StudySpec` / `StudyReport` |
| Use-case coordination | Application services |
| Executable facts and bindings | `PlatformRegistry` |
| Lifecycle and approval | Workflow state machines and gates |
| Numerical implementation | Bound providers and executors |
| Large evidence and provenance | Registered artifacts |
| Scientific guidance | LLM Wiki source pages |
| Presentation | UI adapters and assets |

## Dependency Direction

The intended dependency direction is:

`public API -> application services -> contracts and registry -> workflow runtime -> providers and executors`

Inbound adapters depend on the public application boundary. Providers may
depend on external scientific libraries. Contracts must not import backend
workflow implementations, and UI code must not become a hidden runtime
registry.

## Change Checklist

Before adding a feature:

1. Identify whether it is a task capability, study capability, provider
   implementation, gate, parameter, observable, or artifact.
2. Extend the authoritative contract and registry entry instead of adding a
   side table.
3. Bind one runtime implementation and reject unsupported physics before
   execution.
4. Add structured outputs and registered artifacts.
5. Add approval only when the action has scientific or resource consequences.
6. Test the public path through an application service and the focused owner
   module.
7. Update the Wiki with the executable boundary and remaining limitations.

## Related Pages

- [[CAFES Architecture]]
- [[Public Data Contracts]]
- [[Capability Registry Runtime Boundary]]
- [[Study Orchestration And Report Authority]]
- [[Workflow State Design Principles]]
- [[Predefined Workflow Architecture]]

## Recovery Must Preserve Scientific Intent

Automatic recovery uses structured failure evidence and records every changed
parameter. Increasing iterations or continuing a compatible checkpoint can
preserve the requested method. Reference changes and precision relaxation must
not happen as exception-handler defaults. Missing diagnostics remain unknown
and cannot justify weak-correlation routing. Optional artifact failures preserve
completed numerical results and expose the missing evidence with its reason.
Review actions originate from the backend workflow, including when a client
reopens an existing report.

## Input And Evidence Owner Follow-Up

Preparation never authorizes a basis or geometry choice from keywords in user
history or generated summaries. The LLM returns explicit values and structure
proposals; a natural-language edit without an interpreter is unavailable rather
than ready with unchanged values. Questions follow missing-field records,
without identifying their ownership from words in their prose.

Adaptive normalization is idempotent and preserves sparse explicit provider
options. Invalid values cannot be replaced by defaults or truncated. Model
reference runtime comes from TaskSpec, and Study recovery updates canonical
runtime fields after consuming shorthand aliases. Bounded multi-start keeps
its attempt failures in the report.

Unknown classifications block routing. Missing UNO evidence is not a zero
fractionality score; nonphysical natural occupations retain raw values and an
out_of_bounds state. Spin-free entropy fallback requires a verified singlet.
Root matching retains an ambiguity margin at every supported root count.
Optional density, occupation and amplitude failures retain reasons alongside
completed energies. DMRG cost proxies use provider presets and state which
calculation stages they exclude.
