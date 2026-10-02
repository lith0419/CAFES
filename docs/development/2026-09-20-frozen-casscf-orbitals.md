# Frozen CASSCF orbitals

The structured input accepts `orbital_processing.frozen_orbital_indices`, a list
of unique, nonnegative, zero-based MO indices. It defaults to an empty list and
constrains restricted/ROHF CASSCF with either the FCI or block2 solver.
The indices address the full CASSCF MO matrix after
active-space selection/checkpoint loading and initial orbital processing; they
are not AO indices or Fiedler-reordered DMRG sites. The matrix dimensions are
checked before optimization.

The result records the list, count and index convention under
`cas_result.orbital_optimization`; DMRG-CASSCF also retains it in its optimizer
summary. Freezing constrains orbital rotations without deleting basis functions,
changing the CAS electron count, or removing the frozen orbitals' interactions.
Inactive double occupancy alone does not freeze an orbital's coefficients.

Explicitly frozen external virtual columns are now omitted from the internal
MO integral/optimization workspace. All inactive occupied columns, including
frozen core orbitals, and all active columns remain. PySCF receives the retained
rectangular MO matrix and the remaining frozen occupied/active indices. This
is the same constrained variational problem as a full matrix with the same
frozen list; it does not select a new energy cutoff or reduce the AO basis.
`orbital_optimization.integral_space` records the full and working MO counts,
excluded full-space indices and working-to-full mapping.

Each native macroiteration checkpoint and the final MO result restore the full
column layout, including unchanged frozen external columns. Final orbital
energies include the complete AO Fock potential, rather than the projected
potential outside its valid subspace. Full orbitals are attached to block2
continuation manifests and analysis arrays. See the
integral-space report (author archive: `reports/casscf-integral-space-2026-09-20.md`).

No element-specific or energy-window selection is introduced in the backend.
The caller supplies a reviewed list. In the lutein campaign, projection onto the
MINAO C/O 1s span identifies 42 orbitals (indices 0–41), each with greater than
0.99992 core-subspace weight. This is campaign evidence, not a default rule.

Verification compares frozen occupied and virtual orbitals against native
density-fitted PySCF CASSCF, compares FCI/block2 energies, checks preservation of
the frozen MO coefficients, exercises the full public workflow, and rejects
invalid indices and unsupported methods.

The new Amarel campaign increases Slurm memory to 128 GiB and PySCF's budget to
80000 MB, retains 16 cores and a 16 GiB block2 stack, and freezes the 42 reviewed
core orbitals. It uses the same starting MO/density artifacts as job 61719688,
not its newly converged orbitals, so the initial state remains comparable.
Both memory and the orbital constraint change; the timing difference alone
cannot isolate the contribution of either change.
