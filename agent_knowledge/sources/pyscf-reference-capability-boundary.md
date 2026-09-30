# PySCF Reference Capability Boundary

## Rule

PySCF documentation and source examples expand the agent's scientific context,
but they do not expand what it can execute. `PlatformRegistry` plus backend
validation are the only execution authority.

## Implemented Agent Surface

- Molecular HF, DFT, MP2, CCSD, CCSD(T), FCI, CASCI, CASSCF, and
  ground-state SC-NEVPT2 under its CAS restrictions.
- Molecular correlation diagnostics: frontier gap/degeneracy, natural
  occupations when available, T2 and CCSD T1/D1 evidence, opt-in SCF stability, and
  an auditable active-space proposal.
- Molecular SCF density fitting and restricted/ROHF DF-CASSCF with FCI or
  block2 (`scf_and_casscf`), Boys/Pipek-Mezey localization summaries,
  fixed-block localization plus canonical/Fiedler/manual orbital ordering for
  approved block2 CASCI/CASSCF, molecular structure preview, and a curated common
  basis-set list.
- Finite Hubbard model MP2, CCSD, CCSD(T), FCI, unrestricted-reference
  diagnostics, and finite-cluster FCI observables; plus primitive-cell one-body
  Bloch tight binding with band, DOS, filling, and Hamiltonian artifacts.
- Optional libDMET ground-state DMET for Builder-defined single-band Hubbard
  finite clusters, including translated and explicit finite-graph fragments,
  restricted or broken-symmetry UHF (`Sz=0`) references, FCI/CCSD/block2 DMRG
  impurity solvers, bounded self-consistency, and structured result/history/numerical
  artifacts.
- Static molecular/model scans and adaptive molecular scans, plan-scoped active-space approval, case-local
  recovery, result merge, restart-safe case checkpoints, plot-data artifacts,
  and explicit resource-review gates for expensive FCI/CAS or tensor-scaled
  plans.
- Molecular FCI/CASCI/CASSCF targeted roots within the configured sector,
  including same-sector state-averaged CASSCF. Targeted roots are not a complete
  spectrum and do not enable excited-root SC-NEVPT2.
- Molecular `job=molecular_dynamics` with the registered `qh9` (SCF 1e-13)
  or `qh9_relaxed_scf` (1e-8) profile: restricted B3LYP/def2-SVP NVE trajectories for neutral closed-shell
  H/C/N/O/F molecules. One trajectory is one Task; sampled coordinates,
  energies, Fock and overlap arrays are artifacts. This is not a general MD
  engine, optimizer, or mid-trajectory continuation interface.
- AVAS AO/atom-fragment projection with retained spin-adapted CAS initial
  orbitals after active-space approval.
- Optional block2 fixed-orbital DMRG-CASCI and single-state or same-spin/symmetry-
  sector state-averaged DMRG-CASSCF for
  restricted/ROHF molecular active spaces, plus finite-model targeted-root DMRG
  with structured sweep, RDM, orbital-ordering, and checkpoint-manifest
  artifacts. Molecular and model runs share a result/recovery contract.
  DMRG-CASSCF uses PySCF macroiterations with internal MPS
  continuation and preserves optimized orbitals plus solver traces. Compatible
  cross-task DMRG-CASSCF runs can restore optimized orbitals and MPS jointly.
  Exact fixed-particle-number Schmidt-rank caps, bounded bond-dimension
  escalation, and heuristic error estimates are
  available. Entanglement/occupation evidence may propose an unapproved
  expanded ActiveSpaceAudit that restores only optimized orbitals and starts a
  fresh MPS. Explicit
  result analysis tracks targeted roots from root 1RDM/natural-occupation
  signatures and may propose an approved same-executor retry from a compatible
  state-trusted successful MPS; ordinary initial cases remain independent.
- Single periodic HF/DFT tasks for 3D POSCAR or CIF cells. ASE expands CIF
  symmetry to an explicit P1 cell. The contract includes gamma-centered or
  Monkhorst-Pack meshes, shifts, FFT/auxiliary cutoffs, PBC density fitting,
  exact-exchange divergence, occupation smearing, and structured
  structure/numerics/orbital artifacts. Optional automatic or validated custom
  high-symmetry mean-field bands use analysis k points separate from the SCF
  sampling mesh and write Fermi-referenced JSON/TSV artifacts. SeeK-path mode
  additionally standardizes to the HPKOT primitive cell, executes on that
  effective cell, and preserves input/effective structures plus a full
  transformation audit. Charged, spin-polarized, or unrestricted-reference
  supercell reductions are blocked because input-cell electron counts and
  broken-symmetry magnetic periodicity cannot be inferred safely.
- Optional restricted periodic G0W0, approval-gated HF+DMFT, and staged
  GW+DMFT through the fcDMFT provider. These are single-task workflows with
  registered native artifacts; GW/DMFT total energies are not fabricated when
  the provider does not expose them.

## Roadmap Or Reference Only

- Different-spin state averaging, unrestricted DMRG-CASSCF, transition-RDM or
  direct-wavefunction root tracking, ab initio DMET including DMRG impurities, AFQMC,
  SHCI, MCPDFT,
  MRPT, and other external
  solver interfaces.
- Periodic Planner scans, ab-initio DOS, forces/stress, cell optimization, and
  periodic MP2/CC or unrestricted GW/DMFT.
- Interacting Bloch-model solvers and Bloch adaptive routing; the current
  tight-binding solver records but does not apply `U/V`.
- Different-spin state averaging and state-specific excited-state orbital
  optimization outside the registered targeted-root/state-average routes,
  and excited-root SC-NEVPT2.
- Standalone gradient/Hessian tasks, geometry optimization, solvent, QM/MM,
  general dynamics beyond the registered QH9 trajectory profile,
  electron-phonon solvers, GPU workflows, and forge extensions.

## Promotion Requirement

A roadmap capability becomes executable only after it has a structured contract,
backend adapter, registry entry, validation, artifact schema, UI surface where
needed, bounded cost behavior, and regression tests.

## Must Not

- Do not construct runnable plans from a PySCF API name alone.
- Do not call a design-only builder field an executable backend capability.
- Do not infer that a basis listed in PySCF documentation has been individually
  curated for UI selection; the backend accepts safe names separately.

## Related Pages

- [[Capability Registry]]
- [[Strong Correlation Roadmap Rules]]
- [[PySCF Roadmap Multireference Active Space]]
