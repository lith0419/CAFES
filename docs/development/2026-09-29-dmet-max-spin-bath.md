# Restore max-dimension spin bath completion

The runtime used by Study `20260928-212724-0d0ddbd9` contained a fix for
unequal spin bath dimensions that was missing from the consolidated source.
Its `spin_bath.py` algorithm is restored, together with option validation,
execution wiring, result diagnostics, and regression tests.

Set these fields in `solver.options`:

```json
{
  "execution_mode": "finite_graph",
  "interacting_bath": false,
  "bath_spin_dimension_policy": "max"
}
```

The default policy is `native`. The `max` policy applies to the eigenvalue bath
used for fractional mean-field occupations. SVD baths retain native behavior.
Requests for `max` with translational execution or an interacting bath are
rejected. The helper requires a single-cell, all-valence fragment embedding.

For each spin, diagonalize the environment density and retain the native
selection: eigenvalues farther than `1e-9` from both zero and one. Keep all
selected vectors, then complete the smaller spin channel to the larger count
using omitted environment eigenvectors, prioritizing those closest to fractional
occupation. Eigenvalue order breaks ties. Check the full embedding basis for
orthonormality. Nonfinite, non-Hermitian, or unphysical environment densities
are rejected. The algorithm does not rescale density or electron counts.

Each finite-graph iteration records `bath_spin_dimension_diagnostics` per
fragment: original counts, common count, added eigenvector indices and
eigenvalues, and maximum orthogonality error. The result's
`bath.spin_dimension_policy` records the requested policy; per-fragment
diagnostics identify whether `max` or the native SVD path actually ran.

Tests cover unequal channels and preservation of their selected subspaces,
equal and empty baths, restricted and complex densities, invalid inputs,
request scope, and dispatch. Optional libDMET tests compare the native bath
projector and execute a four-site noninteracting ring under both references,
checking the exact energy and persisted diagnostics.

This restores bath construction only. DMET convergence still uses energy and
consecutive mean-field 1RDM changes. Damping remains independently configurable.

Validation: the restored helper is AST-identical to the old runtime's helper.
All 10 dedicated tests pass with the local libDMET preview source enabled,
including both full ring calculations. The default local suite reports
1,466 tests, zero failures, and 76 optional skips. Ruff and the configured mypy
checks pass.
