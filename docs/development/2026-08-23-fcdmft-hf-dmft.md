# fcDMFT HF+DMFT Provider Boundary

## Implemented

- Added a side-effect-free fcDMFT availability probe and a lazy numerical
  adapter for periodic HF+DMFT.
- Registered `provider.fcdmft`, the HF+DMFT method, impurity-solver options,
  preparation/execution modules, workflow template, and result artifacts.
- Added strict contracts for approved embedding references, localized Fock and
  density matrices, localized two-electron integrals, spin-pair ordering, the
  contiguous `[ncore, nval)` correlated window, bath controls, and resources.
- Added automatic periodic HF preparation from POSCAR/CIF input. Gamma-point
  and regular gamma-centered references are converted to IAO or IAO+PAO
  orbitals, localized one-/two-body tensors, and a reviewable correlated-
  subspace audit.
- Added a reusable correlated-subspace approval view to the Calculation
  Assistant. The first run stops after successful HF preparation; the approved
  follow-up consumes the recorded artifacts and executes fcDMFT.
- The periodic HF reference remains open until the terminal DMFT module
  finishes. DMFT convergence therefore owns the final task status.
- Structured results preserve the HF reference energy but deliberately leave
  the DMFT total energy empty because the connected fcDMFT interface does not
  expose a total-energy estimator.
- Hybridization, self-energy, frequency grids, provider logs, and the fcDMFT
  checkpoint are stored under registered artifact contracts.
- Provider exceptions retain structured failure metadata, complete captured
  output, and traceback context. A failed DMFT stage never exposes the HF
  reference energy as a final correlated energy.
- Solver-aware recovery increases `solver.options.max_iterations`; it does not
  rerun an unchanged DMFT request after changing only the HF cycle limit.
  Per-attempt JSON, NPZ, and log artifacts carry retry suffixes.
- Result analysis distinguishes the periodic HF reference from terminal DMFT
  convergence and never labels the retained HF energy as a DMFT total energy.
- Slurm execution bounds explicit thread and memory options by the allocation;
  absent overrides, the adapter inherits allocated CPUs and uses 75% of node
  memory. A calibrated HF+DMFT working-set estimator is not yet available, so
  planning retains this as a provider request/allocation contract rather than
  inventing a numerical cost proxy.
- Generated HF+DMFT scripts replay the compiled agent workflow so they do not
  silently stop after periodic HF.

## Current Scientific Boundary

The executable provider can construct an IAO or IAO+PAO candidate and localized
periodic ERIs from an ordinary POSCAR/CIF HF task. Explicit orbital indices may
restrict that candidate, but the selected fcDMFT window must remain contiguous.
The generated audit must be approved before solver execution. Automatic
preparation currently requires integer occupations, a gamma-centered k mesh,
and zero k-point shift; manually prepared approved artifacts remain the
extension point for other localization conventions.

The executable preflight requires nonzero projected lattice hybridization. A
full-cell Gamma-only calculation is an isolated finite problem and is rejected
instead of being represented as a meaningful DMFT bath. The `opt`, `direct`,
and `log` discretizations require `nbath >= 2` because of the upstream fcDMFT
integration grid.

HF+DMFT is supported here; GW+DMFT is implemented through a separate staged
provider contract. Cluster DMFT, excited roots, a DMFT total-energy estimator,
and automatic study routing are not. Restricted references use CC
or FCI impurity solvers. Unrestricted references currently require UCC and
spin-resolved one- and two-body artifacts.

## Verification

Provider tests cover option normalization, artifact schemas and shapes,
restricted-reference inference, spin-layout agreement, Slurm resource bounds,
fcDMFT call mapping, terminal state transitions, generated-script routing,
solver-aware retry, reference/DMFT result analysis, and registry/module
compilation. A real two-k-point Si/GTH
smoke calculation verifies the PySCF-to-libDMET IAO/ERI preparation boundary,
enters the native fcDMFT CC kernel, and preserves separate one- and two-iteration
attempts when deliberately constrained below convergence. This is execution
evidence, not a physical benchmark; a converged result against literature
remains required before adaptive method selection is enabled.
