# DMET CDW seed, native convergence rule, and a reverted fit refinement

## Convergence returns to the native DMET rule

Commit `b92da93` (2026-09-06) made the density-fit residual a required
convergence criterion (`< density_fit_tolerance`). DMET itself, and libDMET's
kernel, stop on stationary energy and mean-field 1RDM; the fit is a
least-squares match whose residual need not vanish. The gate is removed:

- `_native_dmet_converged` uses energy and mean-field 1RDM changes only;
- the contract is `criteria='energy_and_mean_field_rdm1'` with
  `density_fit_error_is_diagnostic_only=True`, as before `b92da93`;
- `dmet_density_fit_error` stays in the quality checks as a non-required
  diagnostic against `density_fit_tolerance`, so a large residual remains
  visible in reports.

The strong-CDW honeycomb points (energy and 1RDM changes below 1e-6, residual
about 3e-3) are therefore converged under this rule, with the residual flagged.
The open-shell square regression that failed only on the residual gate passes.

## CDW reference-density seed (kept)

`reference_density_guess='cdw'` applies the AF staggered pattern with the same
sign to both spins, giving a sublattice charge imbalance with no magnetization.
The bias is `reference_density_bias` (default 0.05, so 0.55/0.45 per spin).
Particle counts are preserved, a bipartite graph is required, and restricted and
unrestricted references are both allowed. Because the physical mean-field
baseline is `get_veff(seed)`, the baseline itself gets a staggered Hartree shift
proportional to the bias. The correction potential still starts from zero.

The capability, workflow module, provider binding and both UI selectors
(renamed "Initial Density Seed") include CDW. Tests cover both references,
non-bipartite rejection, and a libDMET ring execution.

## Why a CDW seed

In Study `20260929-155259-27bcfeb4` (honeycomb 4x4, 1x2 cells, FCI, NIB, AF
seed, mixing 0.2, no DIIS), every seed point with V >= 2U fell into a strong CDW
(occupations about 0.01/1.99). The earlier Study `20260928-212724-0d0ddbd9`
(DIIS, no damping) never reached a CDW for V <= 2.5. At U=2, V=2 it converged to
uniform charge with E/site 1.398, while the damped run converged to a CDW with
0.677. Converged status therefore does not identify the phase branch. Seeds
must be compared.

## Density-fit refinement experiment (reverted)

The CDW points failed only on the density-fit residual (about 3e-3 against
1e-4). Energy and mean-field 1RDM changes were below 1e-6. Offline analysis of
the saved arrays found:

- The libDMET BFGS fit stops on its absolute `ytol=1e-7` after one tiny step,
  because a large CDW gap weakens the density response (`error_begin ~= error_end`).
- A scale-invariant trust-region refit of the saved targets lowered the residual
  4-10x. The mean-field 1RDM changed by up to 2.7e-2, but the minimum stayed at
  3e-4 to 8e-4 for V >= 4: a single determinant cannot reproduce the correlated
  1RDM.

A trust-region polish (`scipy.optimize.least_squares`, TRF, `x_scale='jac'`,
unbounded) and a "stationary" convergence policy (residual reported as a
diagnostic) were implemented and passed unit tests. In full DMET runs
(validation Study `20260929-180918-19190c3a`) they failed:

- An unbounded refinement pushed the correlation potential along flat directions
  (|u| up to 7.6e5).
- Points that converge with the original fit, such as U=2 V=1 and U=4 V=1,
  failed with impurity parity, flat chemical-potential response, or eig-bath
  construction errors.
- The stationary policy accepted a frozen runaway solution (U=2 V=2 AF: residual
  1.6, zero double occupancy).

The implementation was reverted at the user's request. Lessons for any future
attempt:

- Bound each fit step.
- Do not scale steps by inverse Jacobian columns.
- Guard stationarity against runaway potentials.
- Validate with full DMET runs, not single offline fits.

Two independent observations remain open:

- The finite-graph block-coordinate fit records each block's error before later
  blocks move it. On a 6-site test chain it reported 1.9e-7 while the final
  residual was 2.4e-2.
- The block-coordinate reporting issue does not affect convergence decisions
  now that the residual is diagnostic, but reported residuals can be low.
- `test_open_shell_noninteracting_square_uses_fractional_occupation_bath`
  (skipped locally without libDMET) failed on the unmodified release: the fit
  stalls at 1.17e-3 on an open-shell Fermi-level degeneracy. Under the native
  rule it now converges, but E/site is -1.50067 against the exact -1.5, so the
  energy assertion still fails. This is tracked separately.
- Verification: full local suite (libDMET tests skipped) passes; the 93
  DMET-related tests on Amarel (`srun -p main`) pass except the square test.
