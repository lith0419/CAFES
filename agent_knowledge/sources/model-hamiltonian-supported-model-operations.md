# Supported Model Hamiltonian Operations

## Scope

This document lists the model Hamiltonian operations the Calculation Assistant
and Planner are allowed to generate.

## Current Design

The Calculation Assistant edits one accepted Builder input and the Planner
expands multi-case studies through one controlled operation vocabulary. The
operation names and parameter scopes come from the capability Registry. The LLM
may translate user intent into these operations, but it must not mutate the
model directly or invent operations outside the contract.

## Supported Operations

- `set_global_parameter`
- `set_site_parameter`
- `shift_site_parameter`
- `scale_site_parameter`
- `add_site_defect`
- `set_bond_parameter`
- `shift_bond_parameter`
- `scale_bond_parameter`
- `add_bond_defect`
- `change_nelec`
- `change_boundary`
- `change_solver`

## Supported Site Parameters

- `epsilon`
- `U`

## Supported Bond Parameters

- `t`
- `V`
- `effective_t`
- `effective_V`

## Supported Global Parameters

- `epsilon`
- `U`
- `t`
- `V`

## Rules

- Use `case_design` for rich model Hamiltonian studies.
- Use `sweep` only for simple parameter scans.
- Single-task edits and study transformations must call the same deterministic
  operation engine; do not maintain a second Assistant or Planner implementation.
- Preserve every field not named by an operation. A single-task edit writes a
  new validated Builder input plus a structured before/after audit and returns
  it for review before execution.
- Validate all operation names before building a plan.
- If the user asks for an unsupported operation, ask for clarification or reject
  the plan.
- Prefer small case counts unless the user explicitly requests a large scan.
- `effective_t` and `effective_V` are internal bond fields. They may be used by
  transformation code where appropriate but should not be exposed as planner
  sweep parameters unless the capability registry explicitly marks them as such.
- Model-Hamiltonian parameter scope metadata in the capability registry should
  define whether a parameter is valid for site updates, bond updates, global
  updates, or sweep variables.
- Site selectors may address explicit ids, all sites, boundary sites, nearest
  sites, or sites within a radius. Bond selectors may address explicit ids,
  endpoint pairs, or all bonds. A selector that matches nothing is an error.
- Use `site: 3` for one site ID and `sites: [0, 2]` for a group. A group is
  applied natively by the shared operation engine; it does not need to be split
  into separate operations. Template references must resolve to the appropriate
  scalar or list before application. A list in `site` is an invalid input, not
  permission to select its first element or flatten it.
- Use `bond: 7` for one bond ID, `bonds: [2, 7]` for bond IDs, and
  `bond: [0, 2]` for a bond's endpoint site IDs. These namespaces are distinct.
- ID values and spin-resolved electron counts require lossless integers.
  Malformed resolved operations report the case/operation location and field;
  the Planner retains the invalid draft for correction without producing a
  partial executable plan.

## Examples

Set a global Hubbard U:

```json
{
  "op": "set_global_parameter",
  "parameter": "U",
  "value": "$U_value"
}
```

Add a local site defect:

```json
{
  "op": "add_site_defect",
  "site": "$defect_site",
  "epsilon_shift": "$defect_strength"
}
```

Change electron count:

```json
{
  "op": "change_nelec",
  "nelec": "$nelec"
}
```

## Rationale

The operation vocabulary is the boundary between free-form scientific language
and executable workflows. Keeping interpretation separate from deterministic
application makes edits auditable and prevents tool hallucination.
