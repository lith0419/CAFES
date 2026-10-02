# Optional SCF Stability And Genuine UNO Candidates

Inspection date: 2026-09-10.

## Problem And Result

The molecular diagnostics module automatically ran internal and external SCF
stability analysis. Lutein job `61369295_0` had converged its restarted DF-UHF
reference in about 6.5 minutes, but then spent hours in Hessian response work.
SCF stability is now opt-in through the existing module configuration:
`workflow.module_config["molecular.correlation_diagnostics"].scf_stability=true`.
The default retains an explicit `not_requested` result with `stable=null`,
which contributes no stability score or instability warning. Module execution,
the active-space probe and the inline runner share this policy.

The old UNO candidate used alpha and beta canonical occupation counts at the
same index. Those indices label different spatial orbitals in UHF, so this
missed genuine fractional natural occupations. Candidate generation now uses
the existing overlap-metric diagonalization of the spin-summed AO 1-RDM.
It needs no correlated 2-RDM. A UHF occupation-window request without correlated
occupations selects this UNO candidate for review, retaining chemical-valence
and AVAS alternatives.

The audit records UNO indices, occupations, contributions, and an alpha
canonical maximum-overlap mapping for display. Its full `initial_mo_coeff`
retains core, active and virtual UNO columns in execution order. The active
electron count is the total molecular count minus two per inactive core
orbital. CAS execution uses the saved matrix instead of sorting mapped
canonical indices. Invalid electron/spin sectors are not executable candidates.
The shared projection reference labels energies from existing MO eigenvalues;
it no longer builds an extra Fock matrix just for candidate generation.

## Design Boundary

This extends existing diagnostics configuration and candidate artifacts. It adds
no public task type, MCP calculation logic, checkpoint protocol or solver.
SCF convergence does not establish stability, and UNO occupations alone do not
establish chemical completeness. Candidates remain unapproved until reviewed.
The default occupation threshold is unchanged; the lutein request keeps its
explicit `[0.01, 1.99]` window.

## Verification

Numerical regression tests check default and explicit stability behavior through
the public calculation service, the inline runner, and the absence of a false
stability score. A synthetic UHF H4 density has known fractional eigenvalues
although its canonical occupation counts are integers. Tests recover those
eigenvalues, verify metric orthonormality, preserve the active projector under
beta-orbital permutation, and show that it differs from the canonical display
subset. A real CASCI agrees with native PySCF using the retained UNO matrix.

Local Python 3.9 / PySCF 2.13.0 passed 190 focused regression tests covering
SCF retry, the new policy, workflow execution/configuration/gates, DF-CASSCF,
strong-correlation diagnostics and adaptive planning. Python 3.12 / PySCF
2.13.1 passed all four new numerical tests. The regenerated wiki has no lint
errors and retains three existing citation warnings.

## Amarel Follow-Up

Before cancelling job `61369295_0`, its converged UHF checkpoint and portable
one-particle artifact were saved under
`/home/your_username/.pyscf-agent/lutein-active-space-20260910/`. The reference energy was
`-1695.6864632017664` Hartree; the unrestricted reference had
`<S^2> = 2.6370081`. This is a broken-symmetry reference, not a spin-pure singlet.
Slurm confirmed cancellation after 4:41:22 elapsed.

The replacement campaign uses that converged density, standard DF-UHF,
geometry 12, def2-SVPD, 16 cores and 64 GB on `your_node`. Stability is explicitly
disabled. Only diagnostics and active-space candidates are requested; no CAS
dimension or CASSCF/DMRG calculation is approved. Deployment identity, request,
handle and live status are retained in the private campaign directory
`.pyscf-agent/lutein-active-space-20260910/`.
