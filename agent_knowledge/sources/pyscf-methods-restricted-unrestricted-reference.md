# Restricted And Unrestricted Reference Rules

## Scope

This document defines how the agent should treat restricted and unrestricted
references.

## Current Design

The molecular UI supports charge, spin, and reference type as advanced inputs.
The reference type may be automatic, restricted, or unrestricted.

## Rules

- Closed-shell systems may use restricted references by default.
- Open-shell systems should consider unrestricted references.
- If the user explicitly chooses restricted or unrestricted, preserve that
  choice in the task spec.
- If reference type is automatic, backend logic may select based on charge and
  spin.
- The selected reference should be reflected in generated PySCF code and result
  summaries where relevant.
- CASCI and CASSCF support unrestricted execution. Unrestricted active-space
  orbital indices may use one shared canonical index list for both spins or a
  spin-resolved contract: `{"alpha": [...], "beta": [...]}`.
- For final CASCI/CASSCF strong-correlation calculations, automatic reference
  selection should prefer a spin-adapted restricted/ROHF-style reference. Use
  unrestricted CAS only when the user explicitly requests it or when the active
  orbital contract is spin-resolved.
- MP2 active-space initial scan may use an unrestricted reference to avoid hiding
  broken-symmetry or near-degeneracy warnings before the final spin-adapted CAS
  calculation.
- SC-NEVPT2 is only registered for the spin-adapted restricted/ROHF-style CAS
  route in the current backend; unrestricted CAS may run, but its post-CAS
  perturbative correction is blocked. Only the ground-state correction
  (`root=0`) is executable; excited-state roots require a separate multi-state
  CAS workflow.

## Model Hamiltonian Rule

- Model Hamiltonian calculations default to unrestricted references for
  mean-field-based solvers.
- Half-filled Hubbard-like systems can make a closed-shell reference
  qualitatively misleading, especially during strong-correlation initial scans.
- For Full CI model Hamiltonian calculations, exact diagonalization is the final
  solver, but reference and correlation-energy diagnostics should still identify
  the unrestricted mean-field reference used for comparison.

## Rationale

Post-Hartree-Fock methods depend on the reference. Incorrect reference choices
can produce invalid or misleading results.

## Open Questions

- Whether to expose ROHF separately.
- Whether model Hamiltonian spin sectors should share the same reference UI or
  remain separate through `nelec`.
