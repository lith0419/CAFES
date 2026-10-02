# Architecture

This is the human-facing entry point for the maintained architecture. The
implementation description lives in the [current architecture source](../agent_knowledge/sources/architecture-current-agent-architecture.md),
which also feeds the agent's curated knowledge. Update that source when the
implementation changes; do not maintain a second copy of the same description.

## Current execution and data ownership

```text
Web / MCP / CLI adapters
        |
        +--> CalculationApplicationService --> task workflow --> solver/provider
        |
        +--> StudyApplicationService --> task executor --> task workflow
                    |
                    +--> saved plans, review decisions, result integration

ArtifactRepository --> persisted reports, checkpoints, numerical artifacts
Browser             --> a view of those records, not execution authority
```

The two Python packages ship in one distribution. `pyscf_agent` currently holds
both task infrastructure and cross-application entry points; consequently those
entry points import `computational_study_agent`. Study also uses task contracts,
registry, executors, and some backend helpers. The existing package names are
not a guarantee of a strict dependency boundary.

Useful ownership references:

- [Public contracts](../agent_knowledge/sources/architecture-public-data-contracts.md)
- [Study orchestration and report authority](../agent_knowledge/sources/architecture-study-orchestration-and-report-authority.md)
- [Python API boundaries](guides/python-api.md)
- [Persistence, verification and distribution](guides/verification-and-distribution.md)
- [Configuration paths](../config/README.md)

## Enforced boundaries and compatibility

The [September 25 architecture review](architecture-review.md) establishes
one-way imports from application adapters to Study to task infrastructure.
Applications may also call task services directly. Shared contracts and storage
primitives must not import Study or presentation code. The shared platform
registry contains declarative Study contracts; those declarations do not import
the executable Study package. They remain shared metadata in this distribution.

Local architecture tests reject Study or inbound-adapter imports from task
infrastructure. Web implementations now live under `pyscf_agent.web`, and Study
grid and Hamiltonian dataset implementations under `computational_study_agent.grid`
and `computational_study_agent.datasets.hamiltonian`. Old module names alias
the same module objects, preserving callers and stored runtime binding identifiers.
Console aliases remain supported.

`StudyApplicationService` composes explicit use-case functions from preparation,
saved-study, execution, dataset, postprocessing, analysis, path-recovery, and MPS
continuation modules. Constructor injection and public signatures are kept.
Use cases share the facade's existing context without keeping competing copies
of execution or report state.

Both browser surfaces consume the frozen `AgentUI` helper namespace. Escaping,
IDs, session selection/labels and table markup are shared; scientific/review state
stays with its page. Classic page scripts remain an incremental migration boundary.
JS/CSS URLs use hashes of packaged contents, with no manually maintained versions.

See [local verification and migration instructions](guides/architecture-maintenance.md).
