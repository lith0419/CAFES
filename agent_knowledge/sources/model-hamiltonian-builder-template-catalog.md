# Model Hamiltonian Builder Template Catalog

## Scope

This document records builder-side lattice templates and preview rules. It is a
builder contract, not an execution capability promise.

## One-Dimensional Templates

- `chain`: open or periodic one-dimensional Hubbard chain.
- `ring`: circular visual layout for a periodic one-dimensional chain.
- `zigzag`: one-dimensional chain with alternating transverse displacement.
- `quasi_1d`: quasi-one-dimensional two-leg ladder.

## Two-Dimensional Templates

- `square`: square lattice.
- `lieb`: square lattice with edge-center sites.
- `honeycomb`: hexagonal honeycomb network.
- `triangular`: triangular lattice with equilateral local connectivity.
- `kagome`: corner-sharing triangular network.
- `dice`: T3 / dice lattice.
- `square_octagon`: square-octagon lattice.

## Rules

- The template name is metadata for reproducibility and UI preview.
- Execution must use the exported sites, bonds, boundary condition, offsets, and
  parameters rather than regenerate the lattice from the template name.
- Periodic bonds must preserve image offsets so the agent preview can draw
  boundary half-bonds or ghost markers instead of misleading direct in-cell
  bonds.
- Small periodic cells must preserve wrap bonds even when the translated
  neighbor maps onto a site pair that is already directly bonded. For example,
  2x2 square and triangular cells need distinct periodic image bonds instead of
  suppressing them as duplicate direct bonds.
- Self-image bonds should still be omitted unless the backend explicitly
  supports them.
- Periodic lattice auto-generation should preserve each template's allowed
  sublattice connectivity. Distance-only periodic completion is not sufficient
  for decorated lattices such as dice.
- Builder templates should generate bonds through a shared offset-distance
  routine where possible. Templates provide site positions, cell vectors,
  nearest-neighbor distance, optional sublattice `pairFilter`, and optional
  `indexHint` for ordering or modulation.
- Do not expose Holstein-Hubbard controls until a dedicated executable phonon
  solver and its validation/artifact contract are registered.

## Failure Cases

- A periodic honeycomb or triangular model is previewed as direct in-cell bonds;
  this hides the boundary condition and can imply the wrong graph.
- Tiny periodic cells collapse neighboring images onto the same displayed site
  pair. Filtering only by source-target pair loses valid neighbors because the
  bond offset is part of the physical identity.
- Refactoring manual bond loops breaks SSH strong/weak alternation because bond
  ordering metadata was not preserved.
- A template is treated as a solver capability; this incorrectly couples the UI
  builder to the execution backend.
