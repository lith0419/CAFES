# Outer DMET damping

DMET now accepts two solver options:

```json
{
  "correlation_potential_mixing": 0.2,
  "diis_enabled": false
}
```

This setting keeps 80% of the current correlation potential and 20% of the
fitted potential. The mixing fraction must be finite and in `(0, 1]`.
Defaults remain `1.0` and `true`, preserving existing calculations.

Both finite-graph and translational execution use the same update. Trace fixing
is applied first, optional outer DIIS second, and damping last:

`u_next = u_old + mixing * (u_proposed - u_old)`.

`diis_enabled` controls only outer DMET DIIS. Impurity HF DIIS retains its
independent `impurity_scf_diis` setting. When outer DIIS is enabled, its existing
start and history settings still apply.

Every iteration records the mixing fraction, whether DIIS was applied, the
raw fitted-potential change, the proposed change after DIIS, and the actual
mixed change. Energy, consecutive mean-field density, and density-fit convergence
thresholds remain unchanged. A small damped step alone does not establish
physical convergence; inspect all residuals and final observables.

For the oscillating honeycomb AFM/FCI points, start with mixing 0.2 and outer
DIIS disabled. Compare convergence and observables before using this across
the scan. Unit tests establish update behavior, not convergence of these
physical points. Already-running jobs and the isolated cluster runtime are
unchanged; applying these options requires a runtime containing this change
and a new Run.
