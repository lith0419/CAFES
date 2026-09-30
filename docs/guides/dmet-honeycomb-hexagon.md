# Honeycomb six-site impurities

In either the Assistant or Study Planner, select **DMET**, then **Fragment
Definition → Honeycomb hexagon (6 sites)**. Set Execution Mode to **Automatic**
or **Finite graph**. In the Hamiltonian Builder use a periodic honeycomb with
compatible primitive-cell repetitions.

| Builder size | Sites | Disjoint hexagons |
| --- | ---: | ---: |
| 3 × 3 | 18 | 3 |
| 3 × 6 | 36 | 6 |
| 6 × 6 | 72 | 12 |

A 4 × 4 honeycomb has 32 sites and cannot be covered by six-site fragments.
Divisibility by six is necessary but not sufficient: the rings must also fit
across both periodic seams (4 × 6 does not). Incompatible geometries fail
validation rather than dropping sites or mixing in smaller fragments.

For a task or saved Study, use:

```json
{
  "solver": {
    "name": "dmet",
    "options": {
      "fragment_definition": "honeycomb_hexagon",
      "execution_mode": "finite_graph",
      "impurity_solver": "ccsd",
      "impurity_solver_options": {"beta": 1000}
    }
  }
}
```

FCI and block2 DMRG use the same selector with their own impurity solver options.
Do not also supply `impurity_shape`, `impurity_size`, `impurity_site_ids`, or
`fragments`. A 3 × 1 cell block is not a hexagon. The CCSD beta policy is
unchanged; this feature adds no smearing options to FCI or DMRG.

The selector identifies actual elementary six-cycles using Builder bonds and
cell membership, excludes loops winding around the torus, and selects a
deterministic disjoint cover. Every original site appears exactly once. Site
numbering, input list order and positions in the drawing do not define the
partition. Onsite U/epsilon, hopping and intersite V are preserved, including
nonuniform values on the same nearest-neighbor honeycomb topology. Missing or
extra bonds, open boundaries and missing cell metadata are rejected by this
automatic selector; the existing explicit-fragment interface remains separate.

This uses the existing finite-graph DMET implementation and solves all fragments
(12 for 6 × 6). It does not claim a one-representative-hexagon translation
optimization. The existing finite-graph restriction to zero total spin
projection remains. Fragment site IDs and six-site sizes are recorded in the
resolved configuration and results; DMRG cost estimates use six impurity plus
up to six bath orbitals.

Tests cover actual Builder geometry, periodic seams, reordered site IDs,
conflicting selectors, both form round trips, saved Studies and DMRG cost
dimensions. The optional scientific regression compares a gapped
noninteracting honeycomb against exact one-body diagonalization, with 6 × 6
CCSD under both restricted and unrestricted references, and 3 × 3 FCI. The gap isolates partition correctness from the separate
fractional-occupation bath approximation at Dirac-point degeneracy. Interacting
calculations retain the existing energy, density and fitting convergence checks;
choosing hexagons does not guarantee convergence.
