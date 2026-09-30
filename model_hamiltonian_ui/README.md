# Model Hamiltonian Builder

Static browser UI for drafting finite-cluster and primitive-cell model
Hamiltonians and exporting CAFES input files.

The builder focuses on deterministic Hamiltonian construction. Execution and
method routing remain backend responsibilities. It supports:

- Finite-cluster and Bloch-lattice representations
- One-dimensional templates: linear chain, ring, zigzag chain, and quasi-1D ladder
- Two-dimensional templates: square, Lieb, honeycomb, triangular, kagome, dice, and square-octagon
- Editable primitive-cell vectors and integer intercell bond offsets for Bloch inputs
- Finite two-dimensional cluster boundaries with primitive-cell subdivisions
- Hubbard and Holstein-Hubbard parameterization
- Optional one-dimensional SSH bond modulation
- Drag editing of site positions
- Site-local `epsilon_i`, `U_i`, phonon frequency `omega_i`, and electron-phonon coupling `g_i`
- Bond-local hopping `t_ij` and intersite `V_ij`
- Optional geometry-coupled hopping preview
- Global energy-unit metadata for all `epsilon`, `t`, `U`, `V`, and phonon energy parameters
- JSON spec export
- CAFES input export containing only the `model_spec` structure
- Reciprocal-space k meshes, high-symmetry paths, and DOS sampling controls for
  Bloch tight binding

Open `index.html` directly in a browser.

## Hamiltonian

The UI represents:

```text
H = sum_i epsilon_i n_i
  + sum_<ij>,sigma t_ij c^dag_i,sigma c_j,sigma + h.c.
  + sum_i U_i n_i,alpha n_i,beta
  + sum_<ij> V_ij n_i n_j
  + sum_i omega_i b^dag_i b_i
  + sum_i g_i n_i (b^dag_i + b_i)
```

The phonon terms are enabled when the model type is Holstein-Hubbard. Each site can carry one local phonon mode.

For one-dimensional SSH bond modulation:

```text
t_strong = t * (1 + delta)
t_weak   = t * (1 - delta)
```

If geometry-coupled hopping is enabled:

```text
t_ij = f_t(t0, r_ij, r0, a_t)
V_ij = f_V(V0, r_ij, r0, a_V)
```

Supported distance-dependent forms:

```text
hopping:
  constant:    t(r) = t0
  exponential: t(r) = t0 * exp[-a(r-r0)]
  power:       t(r) = t0 * (r0/r)^a
  linear:      t(r) = t0 * [1-a(r-r0)]

interaction:
  constant:    V(r) = V0
  inverse:     V(r) = V0 * r0/r
  exponential: V(r) = V0 * exp[-b(r-r0)]
  power:       V(r) = V0 * (r0/r)^b
  linear:      V(r) = V0 * [1-b(r-r0)]
```

## Output

The exported Python file is intentionally structure-only:

```python
model_spec = {...}
```

For `finite_cluster`, the CAFES Web UI reads this `model_spec`, previews
the structure, and lets the user choose MP2, CCSD, CCSD(T), or FCI on the agent
side. The density-density `V_ij` integral convention should be validated
against the target PySCF solver before production use.

For `bloch`, the exported structure contains primitive lattice vectors,
integer `cell_offset` values, an electron count per cell, and reciprocal-space
sampling. The backend selects `tight_binding`, constructs and diagonalizes the
one-body `h(k)`, and writes band, DOS, k-mesh, occupation, and sampled
Hamiltonian artifacts. Nonzero `U` and `V` are retained as metadata but are not
applied by this solver.

Reciprocal-space requests are bounded before execution. The backend permits up
to 50,000 mesh k-points, 64 custom path vertices, 20,000 interpolated path
points, 4,001 DOS points, and 200,000,000 combined k-point/orbital/DOS-grid
evaluations. The builder adjusts the available DOS resolution to the current
k-mesh and primitive-cell orbital count.

The `energy_unit` field is metadata only. The builder and backend do not convert
numeric values between Hartree, eV, meV, or model units.

For Holstein-Hubbard exports, `phonon_terms` records local `omega_i`, `g_i`, and a per-mode cutoff. The current CAFES backend still blocks Holstein-Hubbard execution until a dedicated electron-phonon Hilbert-space solver is added.

## Future Work

- JSON import
- twist boundary conditions
- Planner static scans for Bloch-model parameters
- HDF5 integral export
- Interacting periodic model solvers such as DMET and DMFT

## Template Notes

The lattice templates are graph builders for model Hamiltonians. They generate
editable site positions, sublattice labels, and nearest-neighbor bonds. Finite
two-dimensional templates retain a supercell outline and primitive-cell grid;
Bloch templates retain explicit primitive vectors and intercell bond offsets.
They are practical Hubbard-model starting points rather than a complete
crystallographic representation. Explicit per-site unit-cell indices and twist
boundary metadata remain future work.

## Reference Boundary

The Bloch model solver is a model-Hamiltonian implementation and must not be
confused with `pyscf.pbc.gto.Cell`. Ab-initio periodic behavior is validated
separately against the upstream PySCF examples under `examples/pbc/`, while
custom finite Hamiltonian construction follows the integral conventions in
`examples/fci/01-given_h1e_h2e.py`.
