# fcDMFT Periodic GW and GW+DMFT Workflow

## Executable Boundary

The optional fcDMFT provider exposes two related periodic workflows. Standalone
`solver.name=gw` runs a reusable k-resolved `G0W0` stage from a converged
periodic HF or DFT reference. `solver.name=gw_dmft` uses a restricted periodic
DFT reference, executes the same lattice GW stage, prepares an approved
localized correlated subspace, evaluates the local GW double counting, and
then runs the shared fcDMFT self-consistency loop.

Both routes currently require integer closed-shell occupations, more than one
k point, a gamma-centered mesh with zero shift, GDF density fitting, no
smearing, and Pade analytic continuation. GW+DMFT additionally requires a full
matrix self-energy, an approved contiguous correlated window `[ncore, nval)`,
and IAO or IAO+PAO localized artifacts. Unsupported requests are rejected
before the numerical provider is called.

## Workflow Composition

The compiled GW+DMFT task executes these modules in order:

1. `core.execution` builds and converges the periodic DFT reference.
2. `embedding.fcdmft.periodic_gw` computes or reuses the lattice GW artifacts.
3. `embedding.fcdmft.prepare_subspace` creates an IAO/IAO+PAO proposal and
   stops for explicit correlated-subspace approval when needed.
4. `embedding.fcdmft.gw_double_counting` computes or reuses the local GW
   double-counting artifacts in the approved basis.
5. `embedding.fcdmft.gw_dmft` runs the shared fcDMFT loop and owns terminal
   convergence.

The DFT density and orbitals define the lattice GW and localized basis. The
localized DMFT one-body interaction uses the restricted HF effective potential
evaluated on that DFT density. The same fcDMFT finite-size exchange correction
setting is applied to the lattice GW and the localized HF potential so the two
parts use a consistent reference convention.

## Reusable Artifacts

The lattice GW stage registers its compact result, frequency/orbital arrays,
provider log, analytic-continuation coefficients, reference-potential matrices,
and imaginary-axis self-energy. The local double-counting stage registers its
analytic-continuation coefficients, imaginary-axis self-energy, local orbital
transform, and provider log. Approved retries can consume these registered
artifacts without recomputing the completed GW stages.

The exact native filenames expected by fcDMFT are copied into an isolated
solver scratch directory only when the final GW+DMFT loop starts. Provider
HDF5 files are treated as numerical artifacts, not reimplemented or parsed into
an invented self-energy model by the agent.

## Result Semantics

Standalone GW reports provider completion, reference method, selected k points
and orbitals, frequency settings, Fermi energy, and available quasiparticle
HOMO, LUMO, and gap values. It does not expose a GW total energy and does not
invent a GW convergence flag when the native quasiparticle solver does not
provide one. Reference SCF convergence is reported separately.

GW+DMFT reports DMFT convergence, chemical potential, target occupancy,
correlated window, bath discretization, impurity solver, hybridization,
self-energy, frequency arrays, resources, logs, and checkpoint availability.
The current adapter does not expose a GW+DMFT total-energy estimator. The DFT
mean-field energy is retained only as reference provenance and is never labeled
as the correlated total energy.

## Current Limits

Unrestricted or spin-polarized GW+DMFT, shifted meshes, smearing, non-GDF
references, contour-deformation continuation, cluster DMFT, excited roots,
forces, a total-energy estimator, Planner scans, and adaptive method routing are
not executable. A maintained physical GW and GW+DMFT benchmark is required
before the method can participate in automatic recommendations.

## Verification Boundary

The registered `fcdmft-si-g0w0` benchmark freezes the scientific input from
fcDMFT `examples/Si/si_gw.py`: the two-atom primitive silicon cell,
GTH-DZVP/GTH-PBE, PBE, a gamma-centered `4x4x4` mesh, GDF, Pade continuation,
full self-energy, and finite-size correction. It executes only when explicitly
enabled because this is a server-scale calculation. The evaluator verifies the
public `TaskSpec`, native GW settings, artifact contract, unavailable-total-
energy semantics, and a broad quasiparticle-gap sanity range. It is not yet a
frozen numerical baseline.

Mechanical runtime behavior is shared with other providers only through common
helpers for artifact references, compressed numerical arrays, retry filenames,
and module observations. fcDMFT option validation, native calls, convergence,
and scientific result semantics remain inside the provider boundary.

## Related Pages

- [[fcDMFT HF+DMFT Workflow]]
- [[Unified Registry Runtime Boundary]]
- [[PySCF Method Capability Matrix]]
