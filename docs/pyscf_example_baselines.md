# PySCF Example Baselines

This document maps CAFES contracts to paths under the `examples/`
directory of an upstream PySCF 2.13.0 source snapshot. The installed PySCF
version used by the project is also 2.13.0.

The examples are API and scientific-convention references. Agent regression
tests should use small deterministic fixtures derived from these workflows,
not execute every upstream example verbatim. Many upstream inputs are too
expensive for routine CI or require optional external programs.

## Periodic Electronic Structure

| Agent contract | Upstream example | Current alignment | Baseline behavior |
| --- | --- | --- | --- |
| Periodic cell construction | `pbc/00-input_cell.py` | Implemented | Build `pyscf.pbc.gto.Cell` from atoms, row-wise lattice vectors, basis, pseudopotential, and coordinate convention. |
| Periodic basis and pseudopotential input | `pbc/04-input_basis.py`, `pbc/05-input_pp.py` | Implemented | Load PBC Gaussian basis and GTH pseudopotentials through the PBC loaders. |
| Plane-wave auxiliary cutoff | `pbc/06-input_ke_cutoff.py` | Implemented | Treat kinetic-energy cutoff and FFT mesh as numerical convergence controls. |
| Gamma HF/DFT | `pbc/10-gamma_point_scf.py` | Implemented | Use RHF/UHF or RKS/UKS objects and report energy per unit cell. |
| Regular k-mesh HF/DFT | `pbc/20-k_points_scf.py` | Implemented with an equivalent mesh constructor | Generate k points with `cell.make_kpts` and use KRHF/KUHF or KRKS/KUKS. |
| Occupation smearing | `pbc/23-smearing.py` | Implemented | Record entropy, free energy, and zero-temperature extrapolated energy; sigma is in Hartree. |
| Gaussian density fitting | `pbc/35-gaussian_density_fit.py` | Partially implemented | Use PBC GDF and preserve auxiliary-basis and three-index-integral controls. The agent exposes FFTDF/GDF/MDF/AFTDF and named auxiliary bases, but not custom ETB or reusable `cderi` controls. |
| Band path | `pbc/09-band_ase.py` | Implemented and extended with validated custom and SeeK-path routes | Separate SCF sampling k points from analysis-path k points and reference energies to `get_fermi()`. Automatic mode follows the example directly; custom mode uses the same ASE `Cell.bandpath` representation. SeeK-path is an agent extension that standardizes the primitive cell with the HPKOT recipe, executes SCF on that effective cell, and stores the transformation/path audit. |
| Gamma post-HF | `pbc/12-gamma_point_post_hf.py` | Deliberately deferred | A real gamma-point reference can seed molecular CC/response implementations, subject to explicit capability and cost gates. |
| K-point MP2 | `pbc/22-k_points_mp2.py` | Deliberately deferred | Use KMP2 for regular meshes and report correlated energy per unit cell. |
| K-point GW | `pbc/22-k_points_gw.py` | Deliberately deferred | Build KRGW from a converged k-point mean-field reference and retain quasiparticle energies separately from mean-field bands. |

## Molecular Correlation And Active Spaces

| Agent contract | Upstream example | Current alignment | Baseline behavior |
| --- | --- | --- | --- |
| Unrestricted MP2 natural orbitals | `mp/12-dfump2-natorbs.py` | Partially implemented | The agent derives correlated natural occupations from the post-HF density matrix and mean-field UNO evidence, but does not yet expose the complete `make_natorbs()` orbital matrix/export route. |
| UHF/UKS seed for spin-adapted CASSCF | `mcscf/11-casscf_with_uhf_uks.py` | Partially implemented | ROHF/restricted spin-adapted CAS routing and retained AVAS orbitals exist; a general UHF/UKS-to-spin-adapted orbital conversion contract does not. |
| Explicit unrestricted CASSCF | `mcscf/60-uhf_based_ucasscf.py` | Partially implemented | UCASCI/UCASSCF execution exists, but CAS-level spin diagnostics are not yet a first-class result contract. |
| Restricted versus unrestricted CAS scan | `mcscf/61-rcas_vs_ucas/o2-scan.py` | Partially implemented | Both reference routes can be requested, but a paired reference-comparison scan is not yet a dedicated workflow. |
| AVAS proposal | `mcscf/43-avas.py` | Implemented for spin-adapted CAS | Preserve AO-label targets, projection weights, `ncas`, `nelecas`, and the returned orbital matrix. |
| DMET-style CAS proposal | `mcscf/43-dmet_cas.py` | Not implemented | Use AO/fragment targets and density-matrix entanglement to produce localized active-space guesses. |
| DMRG as CAS solver | `mcscf/50-casscf_then_dmrgscf.py`, `mcscf/50-dmrgscf_with_block.py` | Not implemented | Replace the FCI solver behind a CAS contract and retain solver-specific convergence controls. |
| Custom finite Hamiltonian | `fci/01-given_h1e_h2e.py` | Partially implemented | Finite Hubbard tasks build and validate internal `h1e`/`h2e` tensors, but arbitrary user-supplied integral tensors are not a public task contract. |

## Finite Model-Hamiltonian Exact Diagnostics

| Agent contract | Upstream example | Current alignment | Baseline behavior |
| --- | --- | --- | --- |
| FCI energy and density matrices | `fci/01-given_h1e_h2e.py`, `fci/14-density_matrix.py` | Implemented for registered finite Hubbard models | Build the spin-resolved determinant space from the validated model integrals, then retain energy, one-body natural occupations, double occupancy, spin correlation, and connected charge correlation as structured outputs. |
| Complete many-body spectrum | PySCF FCI contraction APIs, with agent-owned dense assembly | Implemented with an explicit resource bound | Construct every Hamiltonian column with `direct_spin1.contract_2e` and use full dense diagonalization only when the determinant dimension is at most 5,000. Reject larger diagnostic-spectrum requests rather than presenting a partial eigenspectrum as exact. |

## Regression Policy

1. Keep at least one lightweight fixture for each executable contract.
2. Record energies with explicit units and whether they are per cell, total, or
   correlation-only quantities.
3. Test gamma and k-point result shapes separately.
4. For smearing, distinguish total energy, free energy, and zero-temperature
   extrapolation.
5. For CASSCF/DMRG, persist the active-space definition and orbital provenance,
   not only the final energy.
6. Mark examples requiring optional executables or large k meshes as manual or
   scheduled benchmarks rather than ordinary unit tests.
