# Model Hamiltonian Input Contract

## Scope

This document defines the contract between the Model Hamiltonian builder, the
single-calculation agent, and the planner agent.

## Current Design

The builder creates a local PySCF input file containing the model Hamiltonian
structure and parameters. The downstream agent reads that file from the selected
working directory.

The builder should not choose the downstream solver for planner studies. Solver
selection belongs to the agent or planner.

The saved model spec should expose the same structure in two compatible views:

- `sites` and `bonds`: the execution-facing physics parameters used by the
  backend.
- `graph.nodes` and `graph.edges`: the planner-facing site-bond graph used to
  resolve site ids, bond ids, and endpoint pairs.

## Required Model Fields

A model Hamiltonian spec should include:

- schema identifier
- model name
- dimension
- preset
- boundary condition
- energy unit
- electron count
- sites
- bonds

## Site Fields

Each site should include:

- `id`
- `x`
- `y`
- `epsilon`
- `U`

## Bond Fields

Each bond should include:

- `id`
- `source`
- `target`
- `t`
- `V`
- `effective_t`
- `effective_V`

## Graph Fields

`graph.nodes` should mirror site ids and positions. `graph.edges` should mirror
bond ids and endpoints:

- `id`
- `source`
- `target`
- `endpoints`
- `periodic`
- `offset`
- `kind`

The `graph` view is metadata for planning and inspection. It must not replace
the execution-facing `sites` and `bonds` lists.

## Rules

- Builder output should be saved directly into the active work directory.
- Agent and planner should not rely on browser downloads for this path.
- Model structure and solver choice should be separated.
- Holstein-Hubbard is not exposed by the current Builder and remains rejected by
  backend validation until an executable phonon solver is registered.
- Energy units must be explicit; use `a.u.` for atomic units.
- Energy-like model parameters should display the selected `energy_unit`
  wherever users inspect values, including builder labels, preview text,
  distance-coupled canvas labels, planner comparison tables, TSV exports, and
  plot axis labels.
- Electron counts, site ids, geometry sizes, and dimensionless distance-function
  exponents remain unitless.
- Bond ids and site ids are separate namespaces. If a user means the bond
  connecting site `i` and site `j`, planner operations should use `bond: [i, j]`
  or `selector: {kind: "site_pair", sites: [i, j]}`. Use `bonds: [id1, id2]`
  only when the user explicitly asks for bond indices or bond ids.
- When `graph.edges` is available, the planner should use it to verify endpoint
  pairs before producing a plan.

## Rationale

The builder is an editor for model Hamiltonian inputs. The agent and planner
are execution and study layers. Keeping the boundary clean prevents hidden
method choices from being baked into input files.
