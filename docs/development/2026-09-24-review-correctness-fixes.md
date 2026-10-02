# Diagnostic and input correctness fixes

This change addresses four findings from the September 23 review. It does not
change AVAS compaction, model-method policy, resource estimation, or PES repair.

## Molecular diagnostics

Missing spatial natural occupations no longer leave an uninitialized variable
in the broken-symmetry promotion path. Their score remains null and contributes
no evidence; an unavailable UNO spectrum is not a zero-valued measurement.

For open shells, routing compares the sorted spatial occupations with the
spin-adapted determinant spectrum at the same electron count: doubly occupied
orbitals, `abs(spin)` singly occupied orbitals, and empty orbitals. The maximum
absolute deviation, capped at one, is the routing score. The electron count is
inferred from the spectrum, allowing paired inactive orbitals to be omitted in
CAS-only or fractional-occupation inputs. Incompatible sectors are unscored.
Every occupation participates; no SOMO is deleted. The original NOONs and
2/0-fractionality metrics remain available for active-space selection, alongside
the new `routing` evidence. Closed-shell occupation scoring is unchanged.

Open-shell count is informational, so it is removed from the weighted physics
score, including the denominator. The other component weights and 0.50/0.80
occupation promotion thresholds are unchanged. Nonphysical densities,
nonconvergence, spin contamination, T1/D1 and broken-symmetry singlet evidence
retain their existing roles. This corrects a determinant-limit false positive;
it does not establish broad diagnostic calibration or active-space completeness.

## Study active spaces

Explicit manual and already approved input contracts constrain family and CAS
dimension resolution. Their alternatives cannot replace their local orbital
indices or initial matrices, even when automatic candidates have broader case
support. Audit/provenance identity distinguishes an automatic candidate rendered
as `selection_method=manual` from an actual manual selection.

Conflicting fixed choices produce a review-required conflict. Original choices
are preserved in `case_active_space_contract`; no executable replacement is
fabricated. Ordinary automatic candidates retain the existing ranking policy.

## Initial-scan Registry consistency

The review's claim that molecular `auto` runs MP2 everywhere was based on stale
`study.adaptive_initial_scan` metadata. The actual molecular execution path
already resolves the shared HF-first probe and conditional MP2 refinement.
Both Registry namespaces now use the same strategy builder, including refinement
triggers, and the Planner WebUI exports that metadata. Explicit MP2 and FCI
strategies retain their meaning. The internal model helper retains its MP2
probe through the model-method policy; public model studies remain static-only.
This metadata correction is independent of the open-shell scoring bug above.

## Generated model input

The script uses one-pass interpolation and Python string-literal serialization
of the JSON input. Later placeholder substitutions can no longer alter model
labels or runtime/options content. The existing AST input reader accepts the
generated form. Regression tests both parse the artifact and execute it with
the numerical solver mocked, checking the exact delivered arguments.

## Validation

184 focused tests passed with local Python 3.12.14 and PySCF 2.13.1. The same
184 tests also passed with Python 3.9.6 and PySCF 2.13.0. Coverage
includes diagnostics, active-space audit and Study resolution, adaptive
execution/recovery, UNO/stability, input boundaries, public contracts, generated
model inputs and model numerical execution. The older Python run is an extra
compatibility check, below the project's declared minimum.

55 Wiki curation/retrieval/planner tests passed. The 44-page compiled Wiki was
regenerated; lint reported zero errors and the same three existing citation
warning pages. `git diff --check` passed.

New checks cover ROHF/UHF-like doublet/triplet and negative-spin determinant
limits, retained additional fractional pairs, unavailable UNO, raw occupation
preservation, manual/approved precedence, conflicting choices, and script labels
containing placeholders, Unicode, triple quotes, newlines and backslashes.

Numerical smoke checks use planar CH3/STO-3G (doublet, UHF/UMP2),
O2 at 1.21 Angstrom/STO-3G (triplet, UHF/UMP2), and H2 at 3 Angstrom/STO-3G
(singlet, RHF/FCI density). The open-shell cases no longer receive a strong route
solely from SOMOs, while the stretched singlet retains a strong diagnostic.
These are fixed-input regressions, not molecule-specific production rules.

One existing adaptive-planning test omitted the already established
`continuation_policy=project_all` default. Its failure was reproduced against
the unmodified HEAD archive; the expected contract was updated.

After the Registry alignment, 104 Registry, adaptive-planning, and molecular
correlation tests passed with Python 3.12.14 and PySCF 2.13.1. They verify that
both strategy namespaces, the molecular method resolver, and Planner UI metadata
agree, while the internal model helper retains MP2 and explicit FCI behavior.

No remote calculations, service restarts, commits or deployment were performed.
