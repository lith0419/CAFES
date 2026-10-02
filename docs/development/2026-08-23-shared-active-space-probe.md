# Shared Molecular Active-Space Probe

## Implemented

- Added one backend builder for molecular active-space probe requests and the
  versioned `pyscf-agent.active-space-probe.v1` contract.
- Made Calculation Assistant and Computational Study Planner resolve the same
  `molecular.active_space_probe_strategy` Registry entries: `auto`, `mp2`, and
  `fci`.
- `auto` now starts with an HF/SCF chemical-valence/AVAS proposal. A registered
  refinement module runs MP2 only when the preliminary audit is unresolved or
  ambiguous. Explicit `mp2` fixes that choice; explicit `fci` remains available
  for small systems where exact diagnostics pass resource review.
- Preserved the intended CASCI/CASSCF method, target FCI/block2 solver, solver
  options, occupation window, and AVAS intent as structured probe metadata.
- Removed the browser-side MP2 request rewrite. The Assistant now requests a
  compiled probe from the application service and executes that request through
  the normal single-task kernel.
- Reused the same workflow module configuration in Planner cases, request
  preparation, API responses, runtime observations, and approval provenance.
- Added regression coverage for MP2/FCI strategies, AVAS preservation, target
  solver isolation, API validation, generated Web assets, workflow runtime, and
  Planner/Assistant contract agreement.

## Scientific Boundary

The initial probe collects evidence for active-space selection; it is not the
formal correlated calculation. Probe-only unrestricted references, disabled
post-CAS correction, and diagnostic outputs must not overwrite the approved
CAS reference or solver configuration. Every generated active space still
requires explicit `ActiveSpaceAudit` approval before CASCI/CASSCF execution.

`Auto` is a stable Registry policy rather than browser-specific behavior. Its
execution and conditional refinement methods are declared once in the Registry
and consumed by both Web surfaces.
