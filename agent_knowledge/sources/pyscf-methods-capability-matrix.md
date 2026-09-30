# PySCF Method Capability Matrix

## Molecular Methods

| Method | Canonical value | Current execution boundary |
| --- | --- | --- |
| Hartree-Fock | `hf` | Reference selected as RHF/UHF/ROHF-compatible by charge, spin, and `restricted`. No `xc`. |
| DFT | `dft` | RKS/UKS-compatible reference. Requires a non-empty `xc`. |
| MP2 | `mp2` | Valid converged HF reference; restricted and unrestricted variants are supported. |
| CCSD | `ccsd` | Valid converged HF reference; refined molecular diagnostic route. |
| CCSD(T) | `ccsd_t` | CCSD plus perturbative triples; use only where single-reference diagnostics support it. |
| Full CI | `fci` | Small molecular systems only in practice; recovery path has conservative size limits. |
| CASCI | `casci` | Requires approved `ncas`, `nelecas`, and active-orbital contract. |
| CASSCF | `casscf` | Requires approved `ncas`, `nelecas`, and active-orbital contract. |
| DMRG-CASCI / DMRG-CASSCF | `solver.name=block2_dmrg` | Optional block2 solver for restricted/ROHF CASCI and single-state or same-spin/symmetry-sector state-averaged CASSCF, with targeted root energies/RDMs, exact-sector bond-dimension caps, adaptive recovery, localization, orbital ordering, shared result contracts, state-tracked joint orbital/MPS continuation, and entanglement-driven active-space review; no unrestricted active space or SC-NEVPT2 coupling yet. |
| SC-NEVPT2 | `post_cas.sc_nevpt2` | Ground-state correction after spin-adapted CASCI/CASSCF; unavailable for unrestricted CAS. |

## Model-Hamiltonian Solvers

| Solver | Canonical value | Current execution boundary |
| --- | --- | --- |
| MP2 | `mp2` | Unrestricted model mean-field reference. |
| CCSD | `ccsd` | Unrestricted model mean-field reference. |
| CCSD(T) | `ccsd_t` | CCSD reference plus triples correction. |
| Full CI | `fci` | Exact finite-model benchmark and many-body observables when feasible. |
| block2 DMRG | `block2_dmrg` | Optional finite-cluster solver with targeted low roots, exact-sector bond-dimension planning, sweep/resource recovery, RDMs, entanglement diagnostics, symmetry analysis, cross-case state tracking, and explicit MPS continuation. |
| libDMET DMET | `dmet` | Optional ground-state composite workflow for Builder-defined single-band Hubbard finite clusters, with translated or explicit fragments, restricted or broken-symmetry UHF (`Sz=0`) references, FCI/CCSD/block2 DMRG impurities, and structured convergence/embedding artifacts. |
| Bloch tight binding | `tight_binding` | One-body primitive-cell `h(k)`, bands, DOS, filling, and k-mesh occupations; backend-only and does not apply nonzero `U/V`. |

## Periodic Methods

| Method | Canonical value | Current execution boundary |
| --- | --- | --- |
| Hartree-Fock | `hf` | 3D gamma-point RHF/UHF or regular-k-mesh KRHF/KUHF from POSCAR or symmetry-expanded CIF. |
| DFT | `dft` | 3D gamma-point RKS/UKS or regular-k-mesh KRKS/KUKS; requires a registered XC functional. |
| Periodic G0W0 | `solver.name=gw` | Optional fcDMFT backend for restricted closed-shell, k-resolved HF/DFT references with GDF. Reports quasiparticle energies and self-energy artifacts, not a GW total energy. |
| HF+DMFT | `solver.name=hf_dmft` | Optional fcDMFT backend with automatic HF-to-IAO/IAO+PAO subspace and localized-ERI preparation followed by explicit correlated-subspace approval. It supports CC/UCC/FCI impurity solvers and reports convergence, chemical potential, hybridization, and self-energy; the current adapter does not provide a DMFT total energy. |
| GW+DMFT | `solver.name=gw_dmft` | Optional fcDMFT backend for restricted closed-shell DFT references. Executes reusable lattice GW, approval-gated IAO/IAO+PAO preparation, local GW double counting, and DMFT; no total-energy estimator is currently exposed. |

Periodic input uses a dedicated `periodic` task contract with GTH basis,
pseudopotential, k-point, numerical-grid, density-fitting, exact-exchange, and
smearing fields. Current outputs are energy per cell, fundamental/direct
mean-field gaps, VBM/CBM records, PySCF Fermi level, and spin/k-point-resolved
orbital energies and occupations. Integer-occupation insulators use PySCF's
VBM convention; smeared calculations report the mean-field chemical potential.

All 11 pseudopotential names and all 21 named GTH basis aliases shipped by the
installed PySCF runtime are registered. `GTH-LDA` aliases `GTH-PADE`, so the 11
pseudopotential names represent 10 unique data files. `GTH-PADE`, `GTH-LDA`,
`GTH-PBE`, and `GTH-HF-REV` cover H through Rn; other families have narrower
elemental coverage. Validation loads the selected basis and pseudopotential for
every species before execution.

## Shared Rules

- The runtime registry, not this table, is the executable source of truth.
- Any method absent from the runtime registry is rejected by generic capability
  validation before execution.
- Model FCI spectrum/strong-correlation diagnostics use complete dense
  diagonalization up to determinant dimension 5,000; this is distinct from an
  energy-only iterative FCI calculation.
- Model mean-field references default to unrestricted because half-filled and
  strongly correlated clusters can make closed-shell references misleading.
- Molecular CAS defaults to a spin-adapted restricted/ROHF-style route;
  unrestricted CAS is an explicit advanced route with shared or spin-resolved
  alpha/beta orbital indices.
- Analysis-scope Boys/Pipek-Mezey localization does not change solver orbitals.
  Active-space scope changes an approved block2 CASCI active block or the
  initial active orbitals for DMRG-CASSCF and records transformation/order provenance.
- Molecular and periodic density fitting use separate contracts. Periodic
  HF/DFT supports FFTDF, GDF, MDF, and AFTDF.
- Current molecular basis registry includes common minimal, Pople, Dunning,
  core-valence, def2, ANO, and LANL sets; arbitrary safe PySCF basis strings are
  not constrained to the curated UI list.

## Roadmap Boundary

Different-spin state averaging, transition-RDM/direct-wavefunction root
tracking, ab initio DMET including DMRG impurities, AFQMC, unrestricted AVAS,
Wannier-based periodic subspaces, periodic Planner scans,
state-specific orbital optimization, and excited-state SC-NEVPT2 remain non-executable until
they gain task contracts, backend adapters, registry entries, validation,
artifacts, and regression tests.

## Related Pages

- [[Molecular Method Selection Rules]]
- [[Model Hamiltonian Solver Support]]
- [[PySCF Reference Capability Boundary]]
