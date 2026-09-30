# Registry Migration Baseline

## Purpose

Registry migration is sensitive because backend validation, Planner choices,
Web UI options, workflow composition, and generated reports all consume the
same runtime facts. The read-only audit remains the guard against reintroducing
a second catalog after the completed migration.

The baseline audit is available through:

```bash
python -m pyscf_agent.registry.audit --source-root .
```

It reports the typed entry inventory, validation state, legacy family
consumers, relationship fields still stored in metadata, legacy public payload
keys, and contract shapes that remain transitional. The audit does not register
entries, mutate the default Registry, or participate in task execution.

## Current Baseline

The current default Registry is valid and contains ten entry kinds:
capability, parameter, observable, option set, module, gate, provider, binding,
artifact, and template. The current audit baseline contains 317 typed entries
with no validation issues, runtime family consumers, compatibility call sites,
metadata relationships awaiting promotion, legacy public payload keys, or
catalog-shaped artifact/option contracts.

Domain source-family builders still feed the typed index during construction.
They are not queried by runtime consumers and therefore do not constitute a
parallel Registry. They may be replaced gradually by direct typed declarations
when that reduces conversion code, without changing public entry ids.

## Migration Safety Rules

1. Each scientific or runtime concept has one declaration owner.
2. No migration step changes a public payload and its source declaration in
   separate, untested changes.
3. A domain moves as one vertical slice: definitions, consumers, tests, and
   old declarations are changed together.
4. Unmigrated domains may continue using the compatibility surface, but a
   migrated concept must not retain a shadow declaration there.
5. Registry validation, backend constants, Planner options, and UI option IDs
   must remain aligned after every slice.
6. Audit observations become enforced errors only after the corresponding
   legacy pattern has been completely removed.

## Completed Migration Guarantees

- Runtime consumers resolve entries through `PlatformRegistry` and the unified
  typed index.
- Artifacts and option values use their dedicated typed contracts.
- Provider, module, gate, template, capability, parameter, and observable
  relationships are validated before execution.
- Audit tests reject new legacy family consumers and compatibility call sites.
- Current entry ids, kinds, and validation status are covered by regression
  tests.
- Registry artifact schemas are checked against provider-owned result constants.
  The Registry remains the public declaration owner; provider constants are
  implementation assertions, not a second capability catalog.

## Completed Slice: Artifact Contracts

All artifact definitions now originate as explicit `ArtifactContract` values.
Internal registration checks, schema lookup, kind validation, binary-container
metadata, and continuation boundaries consume those typed contracts. The old
artifact catalog is no longer a declaration source.

The unified index now rejects `CatalogItemContract` for artifact entries. This
prevents a migrated artifact from acquiring a second declaration path.

## Remaining Cleanup Boundary

Construction helpers may eventually emit typed contracts directly instead of
passing source families through a conversion step. This is internal cleanup,
not a new migration surface. It must preserve the sole runtime index, semantic
queries, ids, aliases, tests, and generated capability exports.

Runtime helpers may be shared across providers when they implement the same
mechanical contract, such as artifact-reference registration, NPZ
serialization, retry filenames, or module observations. Scientific option
normalization and numerical execution remain provider-owned and must not be
collapsed into a generic helper merely because their payloads look similar.

## Related Pages

- [[Unified Registry Runtime Boundary]]
- [[Current Agent Architecture]]
