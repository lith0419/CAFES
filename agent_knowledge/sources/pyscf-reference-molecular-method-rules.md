# PySCF Reference Molecular Method Rules

## Implemented Surface

| Family | Agent value | Current pattern |
| --- | --- | --- |
| Hartree-Fock | `hf` | Reference-selected RHF/UHF/ROHF-compatible SCF. |
| DFT | `dft` | RKS/UKS-style SCF with required `xc`. |
| MP2 | `mp2` | Post-HF calculation after a valid HF reference. |
| CCSD | `ccsd` | Coupled-cluster calculation after a compatible HF reference. |
| CCSD(T) | `ccsd_t` | CCSD plus perturbative triples correction. |
| Full CI | `fci` | Small-system FCI path from the mean-field setup. |
| CASCI/CASSCF | `casci` / `casscf` | Approved canonical-orbital active space or an approved AVAS spin-adapted orbital guess, with explicit unrestricted advanced route. |
| DMRG-CASCI / DMRG-CASSCF | `solver.name=block2_dmrg` | Optional block2 solver for an approved active space with a restricted or ROHF reference; supports fixed-orbital CASCI and single-state or same-spin/symmetry-sector state-averaged CASSCF orbital optimization. |
| SC-NEVPT2 | `post_cas.sc_nevpt2` | Ground-state post-CAS correction for the spin-adapted CAS path. |

## Rules

- HF must omit `xc`; DFT must provide a non-empty `xc`; all other molecular
  methods normalize `xc` to `null`.
- Validate a mean-field reference before MP2, CCSD, CCSD(T), FCI, or CAS.
- Reference choice must respect charge, spin, and explicit `restricted` value.
- Treat CCSD(T) as a weak-single-reference refinement, not a general fallback
  for stretched or strongly correlated systems.
- FCI is an exact small-system benchmark, not a safe automatic recovery for
  arbitrary molecular size.
- CASCI/CASSCF require an approved `ActiveSpaceAudit` contract with `ncas` and
  `nelecas`; zero-based canonical orbital indices are used when supplied. AVAS
  may retain its approved orbital matrix as the initial guess for a restricted
  or ROHF spin-adapted CAS calculation.
- SC-NEVPT2 appears in `post_cas_results.sc_nevpt2`; `final_energy` becomes the
  corrected comparison energy when that correction succeeds.
- block2 DMRG requires `method=casci` or `method=casscf`, an approved active
  space, and a restricted or ROHF reference. DMRG convergence requires both the final energy
  change and discarded weight to satisfy their configured tolerances.
- Boys/Pipek-Mezey localization may transform the approved active block before
  block2 CASCI or provide the initial active orbitals for DMRG-CASSCF. PySCF then
  optimizes CASSCF orbitals from block2 1- and 2-RDMs. Solver orbitals may use
  canonical, Fiedler, or an explicit complete manual permutation.
- DMRG-CASSCF records orbital convergence, macro iterations, DMRG solver calls,
  optimized orbitals, and final RDMs. MPS continuation is automatic between its
  internal orbital iterations. Compatible cross-task runs restore optimized
  orbitals and MPS jointly; mismatched active spaces cannot reuse that MPS.
- Multi-root DMRG-CASSCF uses `solver.options.nroots` and normalized
  `state_average_weights`. The orbital objective and natural occupations use
  the state-average density; root energies and root-specific RDMs are retained.
  Current entanglement diagnostics are reported for root 0.
- Molecular FCI, CASCI, and CASSCF use the same `nroots` control. The returned
  states belong to one configured particle/spin/symmetry sector and are targeted
  low-energy roots, not a complete molecular spectrum.
- DMRG may increase bond dimension and sweeps within configured limits. A
  registered bond-dimension planning module computes the exact fixed-
  `(Nalpha, Nbeta)` Schmidt-rank cap across every orbital cut and trims an
  oversized requested schedule before execution. The cap is conservative for
  SU(2); multi-root calculations use a conservative multiple of the exact
  single-state cap. Energy change and discarded weight still determine early convergence.
  A
  discarded-weight extrapolation or final-sweep change is a heuristic error
  estimate, never a rigorous error bound or a substitute for benchmark checks.
- Entanglement plus fractional boundary occupations may propose a larger
  ActiveSpaceAudit. The proposal remains unapproved, carries a cost estimate,
  restores optimized orbitals, and starts a fresh MPS after approval.
- Optional block2 outputs include orbital entanglement, targeted low roots, and
  spin/particle symmetry checks. Explicit study result analysis tracks roots
  from root 1RDM or natural-occupation signatures before a compatible saved MPS
  manifest is proposed as the initial state for a later DMRG task.

## Roadmap Boundary

Selected CI, different-spin state averaging, transition-RDM/direct-wavefunction
root tracking, molecular/ab initio DMET, AFQMC, unrestricted AVAS execution, excited-root
SC-NEVPT2, ADC/GW/AGF2,
TDDFT, MCPDFT, and MRPT are
not current planner methods without their own registered integration.

## Related Pages

- [[Molecular Method Selection Rules]]
- [[Restricted vs Unrestricted Reference]]
- [[PySCF Roadmap Multireference Active Space]]
