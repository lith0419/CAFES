# PySCF Reference Core API Rules

## Scope

This document summarizes core PySCF API rules from the local PySCF source wiki
and user guide wiki. It is a reference page for code generation and future
backend work; it does not expand current agent execution capability.

## Source Coverage

- Source wiki pages: `StreamObject Protocol`, `Mole Input Contract`, `Cell Input
  Contract`, `Method Factory Selection`, `Kernel Return and State Pitfalls`.
- User guide chunks: `usage-model`, `molecule-gto`, `crystal-cell`, `scf`.

## Rules

- PySCF inputs are usually Python scripts or Python objects, not a separate
  input-file language.
- Molecular workflows start from `pyscf.gto.Mole`, `gto.M(...)`, or
  `pyscf.M(...)`.
- Periodic workflows start from `pyscf.pbc.gto.Cell`, `pyscf.pbc.gto.M(...)`, or
  compatible `pyscf.M(a=...)` patterns.
- Use module-level factory functions such as `scf.RHF(mol)`, `dft.RKS(mol)`,
  `mp.MP2(mf)`, `cc.CCSD(mf)`, and `mcscf.CASSCF(mf, ncas, nelecas)`.
- Existing PySCF objects may expose convenience methods through dynamic
  dispatch, such as `mol.RHF()`, `mol.KS(xc='pbe')`, `mol.CCSD()`, and
  `mol.CASSCF(6, 6)`.
- `.kernel()` returns the method's main result and is not type-uniform across
  modules.
- `.run()` calls `.kernel()` and returns the object, which is safer when later
  code must read attributes such as `e_tot`, `e_corr`, `mo_coeff`, `mo_occ`, or
  `converged`.

## Implemented Agent Boundary

- The current molecular backend may generate and execute molecular HF, DFT,
  MP2, CCSD, CCSD(T), FCI, CASCI, and CASSCF through its own task-spec and
  execution layer.
- The periodic backend executes 3D gamma-point or regular-k-mesh HF/DFT from a
  dedicated periodic task contract. It does not reuse molecular `Mole` input.
- The current model-Hamiltonian backend does not execute arbitrary PySCF
  scripts from the reference wiki; it consumes normalized model specifications.
- Periodic occupation smearing and ASE automatic, SeeK-path standardized, or
  validated custom mean-field band paths are executable through dedicated
  structured contracts. Optional fcDMFT contracts execute restricted periodic
  G0W0, approval-gated HF+DMFT, and staged GW+DMFT without exposing arbitrary
  PySCF post-HF calls. Periodic MP2/CC, force/stress, and cell optimization
  remain reference or roadmap material.

## Must Not

- Do not assume a PySCF object method is available just because a dynamic method
  name looks plausible.
- Do not mix molecular `Mole` and periodic `Cell` APIs unless a source-backed
  conversion helper is intentionally used.
- Do not expose `.kernel()` return values as stable structured results without
  normalizing them through backend code.

## Related Pages

- [[PySCF Reference Capability Boundary]]
- [[Task Specification Contract]]
- [[Workflow State Design Principles]]
