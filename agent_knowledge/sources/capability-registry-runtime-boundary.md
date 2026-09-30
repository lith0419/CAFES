# Unified Registry Runtime Boundary

## Scope

This document defines what belongs in the runtime capability registry and what
must remain in UI, wiki, or implementation-specific layers.

## Current Design

The runtime migration to `UnifiedRegistryIndex` is complete. Registry audit
reports no legacy runtime family consumers, compatibility call sites, metadata
relationships awaiting promotion, or legacy public payload keys. Source-family
builders remain an internal authoring convenience and are converted into typed
contracts before runtime consumers can query them; they are not a second
runtime fact store. Remaining cleanup may simplify that construction path but
must not recreate a compatibility facade.

The runtime registry is the shared fact source for the backend, planner, Web
APIs, UI option adapters, workflow compilers, and generated summaries. It uses
one typed index for capabilities, parameters, observables, option sets, workflow templates, modules,
quality gates, providers, provider bindings, and artifact contracts. Runtime
callers use `PlatformRegistry`; the former `CapabilityRegistry` facade and
module have been removed.

Entry IDs are hierarchical. Examples include `embedding.method.dmet`,
`task.module.solver.block2.dmrg`, `study.template.adaptive`, and
`provider.block2`. Hierarchy organizes entries; explicit `requires`,
`implements`, template references, and provider bindings express reuse.

Finite choice families such as basis sets, XC functionals, pseudopotentials,
and builder templates are represented by one lightweight `OptionSet` per family.
Each `OptionValueContract` owns only a canonical string, aliases, and bounded
scientific constraints; individual values are not registry entries and do not
carry module, provider, or workflow contracts.

Model variables such as `U`, `t`, `V`, electron count, and boundary conditions
are typed `ParameterContract` entries. Generated quantities are typed
`ObservableContract` entries with result fields, shape, units, and plot support.

## Entry Kinds And Ownership

| Entry kind | Owns | Must not own |
| --- | --- | --- |
| Capability | A requestable scientific operation or method and its execution status | UI copy or a full provider configuration |
| Parameter | One typed input, its scope, unit, choices, and sweepability | A workflow or solver implementation |
| Observable | One structured result field, shape, unit, and plot support | A method-selection rule |
| Option set | A finite family of lightweight canonical values, aliases, and bounded constraints | One registry entry per basis, functional, or preset value |
| Module | One reusable runtime step, its data ports, activation, ordering, compatibility, and stage-local configuration actually consumed by its handler | Solver scientific options already owned by a capability contract, or long scientific guidance |
| Provider | One external implementation boundary and optional dependency | Duplicated method entries for every workflow that reuses it |
| Binding | The provider and runtime adapter selected for a module | Scientific capability semantics |
| Artifact | Versioned artifact kinds, payload schemas, integrity rules, and continuation boundaries | Transient in-memory state or arbitrary metadata |
| Template | A non-limiting composition example | The complete set of valid workflows |

The source catalog may use the standardized `module_id` and `provider_id`
fields while constructing capabilities. `PlatformRegistry` must promote those
links into typed `requires` and `implements` references and reject unresolved
targets. Runtime consumers use the typed registry entries, not the source
metadata as an independent truth table.

## Composite Provider Rule

Register a scientific stage as a separate module only when it can be selected,
replaced, tested, or exchanged through an artifact boundary independently.
Otherwise register its scientific operations as granular capabilities and bind
them to one composite module. For example, libDMET internally builds a Schmidt
bath, solves an impurity, fits a correlation potential, and iterates to self-
consistency. Those operations are discoverable capabilities, but the current
provider exposes one executable `embedding.libdmet.dmet` module because the
upstream public workflow owns their shared mutable state.

## Registry Responsibilities

- Capability family and canonical id.
- Capability status such as `executable`, `design_only`, `planned`, or
  `reference_only`.
- Whether a capability is allowed for planner generation.
- Whether a capability is allowed for backend execution.
- Runtime limitations that affect validation or planning.
- Flow semantics needed by validation, such as model-Hamiltonian parameter
  scopes: `site`, `bond`, `global`, and `sweep`.
- Canonical study modes and adaptive initial-scan strategy ids, solver stages,
  aliases, and defaults.
- Hidden study policies whose numerical thresholds or routing maps are consumed
  by deterministic orchestration, including adaptive method/resource routing
  and scan-path continuity checks.
- Registered workflow modules and roles, including active-space probes that
  prepare evidence without activating the target solver.
- Study workflow options, versioned workflow/result schemas, artifact kinds,
  artifact payload schemas, and resource limits that cross module boundaries.
- Execution-target capabilities and public resource-profile ids. Raw SSH
  credentials and Slurm resource values remain private configuration rather
  than registry data.
- Submitted-task lifecycle schemas, scheduler states, operations, and the
  boundary between scheduler completion and scientific task status.
- Executable postprocessing metrics, including their source diagnostic fields,
  comparison-table scalar names, supported systems, and plot priority.
- Typed module and gate contracts, reusable non-limiting workflow templates, and explicit
  provider-to-runtime bindings.
- Cross-entry reference validation for templates, module dependencies, capability
  entry points, gates, providers, and artifacts.

## Non-Responsibilities

- UI display labels, button text, layout, colors, or helper copy.
- Long scientific explanations that belong in wiki source pages.
- Generated capability tables that can be recreated from code.
- Test-only fixtures or examples that are not capabilities.
- Browser-specific UI state.

## Rules

- Backend constants and planner capability snapshots should derive from the
  registry instead of keeping separate truth tables.
- Runtime code must use `default_registry()`, semantic queries, and typed
  entries. Do not recreate the deleted compatibility facade or parallel
  capability catalogs.
- A user-facing capability states what can be requested. A workflow template
  suggests seed modules, optional modules, and gates without limiting valid
  compositions. A module owns one reusable step. A provider
  binding resolves that step to a runtime implementation. An artifact contract
  defines the data exchanged between steps.
- Critical relationships must use typed references. Do not put `module_id`,
  provider selection, or workflow topology only in free-form metadata.
- Web APIs may expose registry snapshots for UI and wiki tooling.
- UI may use registry ids to decide which options exist, but UI adapters own
  labels and presentation.
- Wiki pages may explain the rationale for capabilities and roadmap status, but
  runtime code must not depend on compiled wiki output to decide executability.
- Platform-registry changes require audit tests that check backend constants,
  planner operation names, postprocessing tools, model-Hamiltonian parameter
  scopes, scan strategy ids, artifact contracts, UI option ids, and generated
  capability exports.
- Canonical adaptive initial-scan strategy ids are `auto`, `mp2`, and `fci`.
  These strategies belong to molecular adaptive studies only. Each strategy
  creates one full-grid initial plan. `auto` resolves its HF/SCF execution
  method and conditional MP2 refinement from Registry metadata; explicit
  `mp2` and `fci` select those probe methods directly. Additional target
  methods enter only through case-specific refinement or recovery.
  Obsolete strategy names, solver fields, and scan-point limits are rejected
  explicitly rather than normalized into a different workflow.
- Calculation Assistant active-space probes and Planner initial scans must
  query those same entries and consume their execution/refinement metadata; a
  second UI or planner strategy table is not allowed. The Calculation Assistant
  always requests the canonical `auto` entry and must not expose study initial-
  scan choices. Only the Planner may let users choose among `auto`, `mp2`, and
  `fci` for an adaptive molecular study. Model-Hamiltonian studies expose only
  `study_mode=static` and require an explicit solver in the task contract;
  optional `grid_refinement` adapts parameter sampling. Their correlation
  diagnostics are outputs rather than a molecular initial-scan stage.
- Method aliases must preserve scientific meaning. Reference labels such as
  `RHF` and `UHF` normalize to method `hf` plus a restricted/unrestricted
  reference constraint. Broad words such as `coupled cluster`, `band`, and
  `bands` do not imply a specific registered method or solver. Conventional
  unambiguous spellings such as `CCSDT` may normalize at the input boundary.
- Every artifact reference must carry `kind`, `path`, `size_bytes`,
  `mime_type`, and `description`. Dynamic per-observable artifacts use the
  registered `observable-*` pattern. Do not repeatedly read large artifacts to
  compute content digests unless an explicit integrity or transport boundary
  requires one.
- Registry metadata should stay runtime-oriented. Do not add `ui_label` or other
  presentation metadata to capability entries.
- A probe module must identify its probe method and preserve the target method,
  solver, and solver options in a structured contract. Probe requirements must
  not leak into validation or display of the approved target calculation.
- Scientific solver options and their defaults belong to the owning capability
  contract. Provider normalizers consume that contract, validate aliases and
  ranges, and resolve declared automatic policies; they must not maintain a
  second defaults dictionary. Expose only options that change scientific
  behavior or bounded execution, not every keyword accepted upstream.
- Module `optional_configuration` is reserved for stage-local behavior that the
  bound module handler actually consumes. It must not duplicate fields already
  carried by `TaskSpec` method, solver, runtime, or provider options. A module
  with no local configuration declares an empty schema and rejects nonempty
  invocation configuration rather than accepting ignored values.
- Composite-provider modules may expose granular implemented capabilities while
  retaining one invocation boundary. For example,
  `embedding.libdmet.dmet` exposes its bath, impurity, fitting, and iteration
  capabilities, but all DMET scientific settings are supplied once through
  `solver.options` under `model_hamiltonian.solver.dmet`.
- Recipes and workflow templates are examples. They may suggest modules and
  gates, but must not restrict other registry-valid compositions.

## Capability Promotion Checklist

Before changing a capability to `executable`, add all applicable pieces:

1. A normalized request contract and explicit scientific limitations.
2. Backend validation that rejects unsupported physics before execution.
3. A module contract, provider contract, and runtime binding without a second
   capability catalog.
4. Versioned structured results and registered artifact schemas.
5. A UI surface only when users need to configure or approve the capability.
6. Bounded resource and convergence behavior.
7. Contract tests, runtime routing tests, and a numerical benchmark where the
   result is scientifically meaningful.
8. Wiki source that states the executable boundary and remaining roadmap.

## Rationale

The registry should be small enough for runtime validation and planner prompts
but explicit enough to prevent hallucinated execution. Keeping UI and wiki
details outside the registry prevents it from becoming a general configuration
dump.

## Failure Cases

- A roadmap item appears in wiki text and the planner treats it as executable.
- A UI option exists for a solver that backend validation rejects.
- A backend-supported operation is missing from planner capability snapshots.
- A runtime writes an artifact kind or JSON schema that no artifact contract
  declares.
- A capability points to an unregistered module or a module has no provider
  binding.
- A provider-specific feature is copied into several method namespaces instead
  of being registered once and referenced by several workflows.
- A solver option exists both in `TaskSpec.solver.options` and a module
  configuration, or a module accepts configuration its handler never reads.
- UI display labels are stored in the registry, making presentation changes look
  like capability changes.

## Related Pages

- [[Registry Migration Baseline]]
- [[Planner Workflow Validation]]
- [[PySCF Reference Capability Boundary]]
- [[Predefined Workflow Architecture]]
