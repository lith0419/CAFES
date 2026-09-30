# CAFES Knowledge Sources

## Scope

`agent_knowledge/sources/` is the maintained, hand-edited source for the
curated runtime wiki. The directory is intentionally flat because the current
compiler scans Markdown files directly under `sources/`; filename prefixes group
topics instead of physical subdirectories.

## Filename Families

- `architecture-*`: platform layers, LangGraph, and state design.
- `workflow-rules-*`: task contracts, execution state, and artifact rules.
- `planner-agent-*`: planning, approvals, adaptive state, conversation, and
  postprocessing.
- `model-hamiltonian-*`: model input contracts, operations, and electron count.
- `embedding-*`: shared embedding contracts and provider-specific executable boundaries.
- `pyscf-methods-*`: executable method and reference support.
- `pyscf-reference-*` / `pyscf-roadmap-*`: reference knowledge and explicit
  non-executable boundaries.
- `ui-design-*`: navigation and user-facing layout rules.
- `error-book-*`: observed failures with permanent prevention rules.

## Compilation Goals

The curated wiki must answer architecture, capability, validation, approval,
artifact, recovery, and roadmap questions without confusing PySCF reference
coverage with implemented behavior. It must also reflect current explicit state
machines rather than only conversation conventions.

## Non-Goals

The wiki is not a test suite, a log archive, a raw PySCF output store, or a
runtime authorization layer. Large run data belongs in ignored `runs/` paths;
executability comes from the capability registry and backend validation.
