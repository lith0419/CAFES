# fcDMFT Periodic GW and GW+DMFT

## Implemented

- Added executable `gw` and `gw_dmft` periodic solver contracts backed by the
  native fcDMFT periodic GW, local GW double-counting, and DMFT interfaces.
- Added a staged workflow with explicit data ports and Registry ownership for
  lattice GW, correlated-subspace preparation, local double counting, and the
  terminal GW+DMFT loop.
- Registered periodic GW and GW+DMFT methods, modules, templates, provider
  bindings, artifact schemas, and Calculation Assistant options.
- Preserved native HDF5 artifacts so a completed lattice GW or local
  double-counting stage can be reused by an approved follow-up without
  recomputation.
- Added strict preflight checks for restricted closed-shell references,
  k-resolved gamma-centered zero-shift meshes, GDF, integer occupations, Pade
  continuation, full matrix self-energy, and approved contiguous subspaces.
- Kept DFT and HF roles explicit: DFT defines the lattice GW reference and
  localized basis; the localized HF effective potential evaluated on the DFT
  density supplies the DMFT one-body interaction. The finite-size exchange
  correction follows the same setting in both stages.
- Added structured result rendering for GW completion, quasiparticle gap,
  reference SCF convergence, GW+DMFT convergence, correlated window, bath,
  resources, and unavailable-total-energy semantics.
- Added provider, TaskSpec round-trip, workflow-order, Registry, generated UI,
  and artifact-reuse tests.
- Added the opt-in `fcdmft-si-g0w0` public-backend benchmark using the native
  silicon example configuration. It validates scientific input, provider
  settings, artifact completeness, and no-fabricated-total-energy semantics;
  it is intentionally not claimed as a frozen physical baseline.
- Reused common runtime helpers for artifact-reference registration, compressed
  NPZ payloads, retry filenames, and module observations. Provider-specific
  validation and numerical execution remain local to fcDMFT.

## Scientific Boundary

The implementation is an executable provider integration, not yet a validated
materials benchmark. Neither standalone GW nor GW+DMFT is reported with a
correlated total energy. The native periodic GW interface is recorded as
completed rather than assigned an invented convergence flag. GW+DMFT terminal
success depends on the DMFT self-consistency result.

Spin-polarized references, nonzero k shifts, smearing, alternate analytic
continuation, cluster extensions, force/stress, Planner routing, and automatic
recommendations remain outside the executable contract.

## Maintenance Audit

The Registry remains the public source of capability and artifact declarations;
provider schema constants are checked against it in tests rather than forming a
second catalog. The workflow module catalog describes composition, while the
runtime registry binds trusted handlers, so those layers are intentionally
separate. Remaining maintainability risks are the large periodic solver,
libDMET provider, adaptive refinement, and study-application modules; they need
behavior-preserving decomposition, not another orchestration layer.
