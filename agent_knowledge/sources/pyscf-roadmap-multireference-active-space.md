# PySCF Multireference And Active-Space Roadmap

## Executable CAS Surface

CASCI and CASSCF are executable only with an explicit approved active-space
contract. Automatic CAS execution starts from a UHF mean-field/orbital reference,
converts that evidence to a shared spatial-orbital basis, and then uses a
spin-adapted RHF-style closed-shell or ROHF-style open-shell CAS route.
Unrestricted CAS remains an advanced explicit route with shared or spin-resolved
alpha/beta orbital indices.

`post_cas.sc_nevpt2` is executable for `root=0` after an approved spin-adapted
CASCI/CASSCF result. The correction is blocked for unrestricted CAS and
excited-state roots.

## ActiveSpaceAudit Requirements

- Record orbital numbering, energy, occupation, AO/atom contribution,
  atom-as-fragment contribution, localization method, selection reasons, and
  `ncas`/`nelecas` consistency.
- Offer manual, genuine UHF natural-occupation-window, chemical-valence baseline,
  evidence-expanded, AVAS AO/fragment projection, and merged candidate records.
  AVAS accepts AO labels or `atom:`/`fragment:` selectors, records the resolved
  targets and projection evidence, and retains the PySCF AVAS orbital matrix as
  a CAS initial guess after approval.
- For a UHF occupation-window request without correlated occupations, recommend
  the genuine UNO candidate for review. Diagonalize the spin-summed AO 1-RDM in
  the overlap metric; adding alpha/beta canonical occupations at the same index
  is not a UNO calculation. Preserve the complete core-active-virtual orbital
  matrix and actual UNO indices/occupations. Canonical mappings are display
  evidence only. This path needs no correlated 2-RDM or SCF stability analysis.
- For automatic selection, recommend the smallest valid chemically complete
  candidate first and order reviewable candidates by increasing `ncas`. Record
  high-confidence occupation/T2 evidence outside the baseline as the next
  expansion candidate; do not jump directly to the largest evidence union.
  This policy is independent of molecular identity and fixed CAS dimensions.
- The direct AVAS initial guess may be proposed from the UHF probe, but its
  approved orbital matrix is converted to the shared spatial basis required by
  the spin-adapted RHF/ROHF CAS calculation. Unrestricted AVAS remains a future
  integration.
- Use `0.02 < n < 1.98` as the default fractional-occupation window.
- Include natural-occupation and max-T2 evidence when the correlated solver
  exposes it; say explicitly when only mean-field evidence exists.
- Require an approval record after any user edit. Do not silently change a
  chemically ambiguous active space.

For a related molecular study, resolve one chemically supported candidate
family and one CAS dimension before approval. Preserve each case's own orbital
indices and initial orbital matrix; do not reuse indices across geometries.
Explicit manual or already approved selections constrain the shared family and
dimension; automatic alternatives cannot replace their orbital indices or
initial matrices. Provenance distinguishes a user selection from an automatic
candidate merely materialized with `selection_method=manual`. Conflicting
explicit selections are retained for review and block execution. Cases without
a compatible automatic mapping return to the common probe path. A fixed
frontier-orbital fallback and a largest-union default are both prohibited.

## Planner Recovery Rule

When a molecular single-reference refinement fails or remains unconverged, the
planner can propose case-local CASSCF recovery with an unapproved audit. When a
CAS case fails, offer active-space expansion or a bounded runtime adjustment for
that one case, then merge the new result into the parent scan.

## Current Resource Gate

Every study plan records a dimension-based cost estimate. FCI/CAS use
alpha/beta determinant dimensions while MP2/CCSD use tensor-scaling proxies.
Plans that cross the determinant, peak-memory, or total-work thresholds require
explicit approval before execution and persist the estimate as a study artifact.
For a proposed CAS, show this estimate in the same ActiveSpaceAudit review; the
user approves or edits the scientific active space once rather than navigating
to a second generic cost decision.

The `auto` active-space solver keeps approved spaces up to 14 orbitals on the
PySCF FCI route and sends larger or explicitly DMRG-recommended spaces to
optional block2 DMRG-CASSCF. The DMRG estimate uses active orbitals, bond
dimension, sweep count, and root count.

An approved block2 active block may be localized with Boys or Pipek-Mezey and
ordered canonically, by Fiedler ordering, or by a complete manual permutation.
For CASSCF, localization supplies initial active orbitals before PySCF orbital
optimization. The audit and solver artifacts preserve transformation and
ordering provenance.
Task orbital ordering is resolved before provider defaults. An explicit solver
option overrides the corresponding task setting, and an explicit workflow-module
option overrides the solver option. Provider configuration is retained even when
block2 is unavailable locally; the availability gate still blocks execution.

For DMRG-CASSCF, `solver.options.bond_dimensions` sets the orbital-optimization
DMRG schedule. Optional `final_bond_dimension` requests a separate fixed-orbital
solve at that M after orbital optimization, reusing its MPS and orbital order.
This explicit final solve does not stop at a smaller adaptive stage. Both phases
use `sweeps` as their sweep limit and share the configured convergence thresholds.
Without this option, the existing adaptive final schedule is unchanged.
`runtime.max_cycle` controls the reference SCF and CASSCF macro-iteration limits;
it does not control DMRG sweeps. A larger final M does not reoptimize the orbitals.
Each CASSCF solver-call trace records its effective bond-dimension schedule and
sweep limit alongside the role, actual ordering and convergence diagnostics.
Small-space exact-sector planning can cap the effective M below the request;
native block2 can stop before the sweep limit after energy convergence. Neither
an M ceiling nor a successful scheduler exit establishes discarded-weight convergence.

A smaller active space reduces the active solver's problem but leaves the AO
basis and most inactive/external orbital rotations in place. DF-CASSCF orbital
response can dominate wall time even when DMRG sweeps finish quickly. Use small
molecules for inexpensive workflow regression; do not promise that a smaller
CAS on a large molecule will finish faster. Inspect macroiterations, JK work,
solver-call timing and both orbital/DMRG convergence separately. A cancelled
orbital optimization has no final fixed-orbital result, even if intermediate
DMRG calls converged.

Single-state and same-spin/symmetry-sector state-averaged DMRG-CASSCF write a
joint optimized-orbital/MPS checkpoint for compatible same-CAS continuation.
Entanglement and natural-occupation boundary saturation may instead propose an
expanded ActiveSpaceAudit. That proposal is costed and unapproved, reuses the
optimized orbitals, and starts a fresh MPS because the active-space Hilbert
space changed.

## Upstream Example Baselines

- `examples/mp/12-dfump2-natorbs.py` is the reference for unrestricted MP2
  natural occupations and natural orbitals.
- `examples/mcscf/11-casscf_with_uhf_uks.py` demonstrates that UHF/UKS orbitals
  may seed spin-adapted CASSCF while the solver still imposes alpha/beta core
  degeneracy.
- `examples/mcscf/60-uhf_based_ucasscf.py` defines explicit UCASSCF as a
  separate advanced route whose spin contamination must be reported.
- `examples/mcscf/43-avas.py` is the reference for AVAS AO-label targets,
  projection weights, `ncas`, `nelecas`, and the returned orbital matrix.
- `examples/mcscf/43-dmet_cas.py` is the reference for AO/fragment-driven
  density-matrix active-space guesses; it is not the external libDMET workflow.
- `examples/mcscf/50-casscf_then_dmrgscf.py` and
  `50-dmrgscf_with_block.py` establish the FCI-solver protocol used by the
  optional block2 integration for fixed-orbital CASCI and single-state CASSCF.

These paths are relative to the `examples/` directory of a PySCF 2.13.0 source
snapshot. See `docs/pyscf_example_baselines.md` for the complete maintained
mapping.

## Roadmap

- Different-spin/unrestricted DMRG-CASSCF, molecular/ab initio DMET, AFQMC, and
  other external multireference solver adapters.
- Excited-state-specific orbital optimization workflows and excited-state SC-NEVPT2.
- Rich orbital visualization and Molden/Cube artifacts.
- Calibrated wall-time estimates and scheduler-aware resource reservations.

## Related Pages

- [[Strong Correlation Roadmap Rules]]
- [[Adaptive Scan Workflow State Machine]]
- [[PySCF Reference Capability Boundary]]
