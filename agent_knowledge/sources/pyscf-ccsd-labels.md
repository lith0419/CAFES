# Paired HF and CCSD machine-learning labels

Request `analysis.outputs = ["energy", "ccsd_labels"]` for a molecular restricted CCSD/def2-SVP single point. The initial implementation supports real all-electron RHF references for H/C/N/O/F, without density fitting.

One task executes HF, CCSD and Lambda, then saves `result-ccsd-labels.npz` and its JSON manifest. Arrays are `F_HF`, `D_CCSD`, `S`, `E_HF`, `E_CCSD`, atomic numbers, coordinates in Angstrom, and the explicit PySCF-to-QH9 AO permutation and phases. Matrices are float64. Both energies include nuclear repulsion. The HF Fock is not a CCSD effective Hamiltonian.

`D_CCSD` is the spin-summed unrelaxed CCSD 1-RDM including the reference contribution, with no orbital-response contribution. It is constructed by `make_rdm1(ao_repr=True, with_mf=True, with_frozen=True)` after convergence of T and Lambda. It is distinct from the optional reference-density restart artifact.

Required checks are convergence, finite values, matrix shapes, Hermiticity (max residual 1e-8), and `Tr(D_CCSD S) = Ne` (absolute residual 1e-6). Residuals and tolerances are saved; the exporter does not repair matrices. Lambda failure or missing required labels prevents successful label completion.

Set `workflow.module_config["molecular.correlation_diagnostics"] = {"enabled": false, "scf_stability": false}` to skip all optional correlation diagnostics and reference stability calculations. Lambda is part of calculating the requested density, not an optional diagnostic. Optional `workflow.module_config["core.execution"].label_provenance` records dataset join keys in the archive. The manifest records Run ID, runtime release and archive SHA256.

Schema `pyscf-agent.ccsd-labels.v2` additionally requires `t1` (nocc × nvir), `C_HF` (nao × nmo), `epsilon_HF`, `mo_occ`, and occupied/virtual MO index arrays. These are the actual closed-shell spatial RHF reference orbitals used by CCSD. Only the AO rows of C_HF are mapped to QH9; MO columns and t1 retain their original ordering and phases. No re-diagonalization, MO rephasing or orbital rotation is performed. Save these together: t1 alone does not define the orbital gauge. Orbital energies are HF orbital energies in Hartree, not CCSD excitation energies. The t1 array is not the scalar T1 diagnostic. The exporter also checks MO shapes, occupations, finiteness, and C_HF.T @ S @ C_HF = I (1e-8). Schema v1 historical files do not contain these required v2 fields and must not be counted as v2 results.

DFT Fock matrices and total energies remain in the paired source AIMD dataset. Join them using the frozen sample/molecule/frame and source Run identity, verify exact atom order and coordinates, and preserve the source AO convention and artifact references. Requesting these labels does not rerun AIMD or add optional reference diagnostics.
