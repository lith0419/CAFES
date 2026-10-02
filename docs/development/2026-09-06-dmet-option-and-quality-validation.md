# DMET Option And Quality-Evidence Validation

Inspection date: 2026-09-06. Local changes on top of `48a0478`.

## Problem And Result

The DMET normalizer previously retained unknown `solver.options` keys. For
example, `density_fit_tolerence=1e-8` left the effective
`density_fit_tolerance` at its default `1e-4`. It now rejects unknown top-level
keys using the existing Registry option names, including options that have an
automatic resolution policy instead of a default. Existing `nroots` and
`correlation_potential_tolerance` migration errors remain specific.
Normalization still preserves valid aliases, explicit values, and round trips.

The shared result-quality evaluator previously accepted an `equal` check with
both operands missing because Python compared two `None` values. It now
requires non-null operands and compatible scalar types before comparing them.
Boolean values cannot satisfy numeric equality. Finite integers and floats
remain comparable without converting equality operands to floats, so large
distinct integers do not become equal through rounding.

Numeric operators require finite numbers, with `limit` required for threshold
comparisons. Nulls, strings, booleans, containers, non-finite values, and numeric
conversion overflow become invalid evidence. The task gate independently
evaluates this evidence even if a provider supplies a cached `status=passed`.

## Compatibility And Boundaries

- Registry remains the authority for DMET option names; no parallel allowlist
  or new execution state is introduced. Nested impurity options continue to
  use the selected provider's existing validation.
- Old requests containing ignored keys must be corrected explicitly. There is
  no ambiguous automatic conversion from `conv_tol` to a scientific tolerance.
  The Study review fixture now uses the valid `energy_tolerance` option.
- The quality schema and downstream task/Study/plotting contracts are unchanged.
  Required invalid constraints block eligibility; invalid convergence evidence
  requires review. Optional checks remain advisory and retain their evidence.
- Legacy results without declared quality checks preserve their existing
  behavior. This change does not introduce observable coverage gates or alter
  solver convergence thresholds.

## Verification

Regression tests first reproduced the unknown-key acceptance and malformed
equality passing through the real task gate. Added cases cover missing/null
operands, boolean/number confusion, containers, non-finite/overflowing values,
valid scalar comparisons, comparison boundaries, optional evidence, explicit
DMET settings, task validation, and normalization round trips.

The focused suite passed all 85 tests. The full suite ran 959 tests in 97.989
seconds: 958 passed and one opt-in numerical benchmark was skipped. It includes
the DMET/block2 numerical paths, Study recovery, and Registry checks. Wiki
curation generated 43 pages; lint reports zero errors and the three existing
citation warnings. `git diff --check` and documentation-link checks passed.
Verification outputs use the
`reports/verification/contract-validation-2026-09-06-*` prefix. The source-tree
runtime is Python 3.9.6; these checks do not replace supported-version wheel
acceptance or a new live-cluster campaign.
