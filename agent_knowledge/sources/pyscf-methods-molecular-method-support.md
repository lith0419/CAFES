# Molecular Method Support

## Current Design

The assistant and planner derive molecular method choices from the runtime
registry. Current canonical values are `hf`, `dft`, `mp2`, `ccsd`, `ccsd_t`,
`fci`, `casci`, and `casscf`.

Energy, HOMO-LUMO, and dipole are default molecular outputs. Correlation
diagnostics and ActiveSpaceAudit are generated workflow outputs when their
prerequisites are available; they are not ordinary checkbox outputs. SCF
stability is opt-in through the correlation-diagnostics module. The default
`not_requested`/`stable=null` is missing evidence, not a stable-reference result.

## Rules

- DFT requires `xc`; HF, MP2, CCSD, CCSD(T), FCI, and CAS methods must
  pass `xc=null` rather than an empty string.
- Post-HF methods require a valid mean-field reference and charge/spin-consistent
  reference selection.
- Assistant active-space selection and Planner adaptive initial scans use the
  same Registry strategy and probe builder. `Auto` begins with HF/SCF chemical
  evidence and invokes the registered MP2 refinement when the preliminary
  active space is unresolved or ambiguous, or when SCF diagnostics show
  instability, frontier degeneracy, or a routing-boundary score. Explicit `MP2` and small-system
  `FCI` probes remain user-controlled alternatives. The final calculation is
  routed and approved independently.
- `MolecularCorrelationRisk` reports separate physical-evidence and
  solver-stress levels. SCF or post-HF nonconvergence promotes the overall
  diagnostic level to strong solver stress without relabeling the physical score.
- The diagnostic level records the strongest warning, while the independent
  routing level chooses the next method. Solver stress without strong physical
  evidence routes through CCSD instead of creating an immediate CAS approval.
  The correlated occupation routing score is moderate from 0.50 and strong
  from 0.80. Closed-shell routing uses 2/0 fractionality; open-shell routing
  uses the maximum deviation from the matching spin-adapted 2/1/0 determinant
  spectrum, retaining the raw occupations for CAS selection. Normal SOMOs and
  the open-shell electron count alone do not promote a multireference route.
  CCSD T1 >= 0.05 or D1 >= 0.10 is a direct multireference promotion.
  These are implementation policy thresholds, not universal physical cutoffs.
  The next-method recommendation is not a guarantee of improved energy accuracy.
- Adaptive recovery retries a failed CCSD(T) calculation with CCSD before it
  proposes CASSCF/CASCI active-space treatment.
- CASSCF/CASCI require explicit, approved `active_space.ncas`,
  `active_space.nelecas`, and orbital indices when supplied.
- A related molecular study must resolve one chemical target and CAS dimension
  across its cases while preserving case-specific orbital mappings. A missing
  automatic mapping is reprobed; it is not assigned a fixed frontier space.
  Explicit manual choices constrain that resolution. Conflicting manual choices
  are preserved for review rather than replaced with automatic alternatives.
- Molecular FCI, CASCI, and CASSCF accept `solver.options.nroots` for targeted
  low-energy roots in the selected particle, spin, and configured spatial-
  symmetry sector. Requesting `excited_states` without an explicit value
  defaults to two roots.
- Multi-root CASSCF uses state-averaged orbital optimization. Optional
  `state_average_weights` must be non-negative, match `nroots`, and normalize to
  one. block2 DMRG-CASSCF supports roots in one common spin/symmetry sector;
  different-spin state averaging is not executable.
- Root energies and root 1RDM/natural-occupation signatures feed explicit
  study result analysis. They do not turn a targeted-root calculation into a
  complete spectrum, and an ambiguous cross-case assignment requires review.
- SC-NEVPT2 is a post-CAS result, not a standalone molecular method or an MP2
  option. It supports `root=0` for spin-adapted CAS only.
- UI changes and conversation edits must update the canonical task spec before
  execution; do not let stale DFT defaults survive a change to MP2 or HF.

## block2 MPO Construction

`solver.options.mpo_algorithm` defaults to `conventional` for the supported
electronic Hamiltonians; `fast_bipartite` remains an explicit option. Existing
saved requests retain an explicitly recorded algorithm. Two active orbitals
use the Conventional NC variant, as recommended by block2.
The result records both the requested and effective algorithm. Orbital
optimization and the final fixed-orbital solve use the same selection.

`integral_cutoff` screens Hamiltonian integrals and is distinct from `cutoff`,
which controls DMRG density-matrix truncation. Omission resolves to `1e-12` for
Conventional, matching block2main, and `1e-20` for FastBipartite. An explicit
nonnegative value overrides this choice. The effective threshold is recorded
in the result configuration. Integral arrays are copied before block2 screening
so the original PySCF Hamiltonian remains available for subsequent calls.

Results and CASSCF solver-call traces record MPO construction, DMRG sweeps,
and RDM timings. The H6 fixed-orbital comparison with official dmrgscf/block2main
checks numerical agreement. The fixed-orbital lutein M=600 Amarel comparison
observed about 55% lower custom solver-process peak RSS with Conventional,
but only 2.3% shorter median solve-plus-RDM time over two repeats. All paths
remained unconverged after 30 sweeps. This is not evidence of a general speedup
or CASSCF orbital convergence.

Active-space Boys/PM localization optionally accepts
`orbital_processing.localization_occupation_thresholds`. The default empty
list localizes the whole active block. `[1.0]` separates low- and high-occupation
subspaces; `[0.1, 1.9]` also retains a separate intermediate group. The runtime
projects the spin-summed SCF density into the chosen active block, diagonalizes
it, and localizes each occupation interval independently. These are reference
occupations, not correlated NOONs. No 2-RDM is required. Boundaries must be
explicit, strictly increasing and inside (0,2); equality belongs to the lower
interval. Active orbitals and electrons are not added or removed. Group sizes,
reference occupations and subspace/orthonormality checks are recorded in
`cas_result.orbital_provenance`. Fiedler ordering remains a separate request.
This is initial orbital processing; it does not constrain subsequent CASSCF
rotations or silently relocalize a compatible orbital/MPS restart.

The lutein fixed-CAS(20e,20o), M=600 pilot found that whole PM plus Fiedler
and split PM plus Fiedler both met the configured DMRG criteria, whereas
whole PM in its output order did not. Whole PM plus Fiedler was slightly
lower in energy and faster in this single comparison. This supports explicit
ordering for that input; it does not justify making split localization a
universal default or treating the previously optimized CASSCF orbitals as
converged with the improved solver. See the September 20 split-localization
comparison report for controls and independent verification.

## Validated Lutein Boundary (September 21)

The September 20–21 lutein follow-up achieved converged M=600 orbital optimization with PM+Fiedler and Conventional MPO. A separate fixed-orbital M=2000 solve does not establish M=2000-optimized CASSCF. With 42 frozen core orbitals, the full external-space result shifted by 0.0248 mHa from the preceding unfrozen result. Initial +3/+5 eV external virtual windows failed the agreed 0.1 mHa energy and 0.01 NOON-difference targets despite inner/orbital convergence. Compressed and original MO workspaces agreed within 8e-10 Ha at the same frozen constraints. These are case-specific observations, not automatic core/window defaults. AO DF storage remains full-size, so fewer working MOs do not imply proportionally lower peak RSS. See the September 21 overnight-results report.

## Reference Rule

Restricted/ROHF CASSCF with FCI or block2 accepts
`orbital_processing.frozen_orbital_indices` (default `[]`). These are explicit,
unique, zero-based indices in the complete CASSCF MO matrix after active-space
preparation, not AO indices or DMRG site indices. They are recorded in
`cas_result.orbital_optimization`. Frozen orbital
coefficients remain fixed; basis size, CAS electron count and their interactions
are retained. Double occupation of inactive orbitals alone does not freeze them.
The caller selects and reviews the indices; the backend does not infer a
chemical-core count or a virtual-energy cutoff. Explicitly frozen external
virtuals are omitted from the internal MO integral workspace; all occupied
and active orbitals remain, with their requested PySCF frozen constraints.
This preserves the same frozen-space variational problem. The backend restores
the complete MO layout in checkpoints, final arrays and continuation manifests.
The result's `orbital_optimization.integral_space` records the working-to-full
mapping. AO integrals and AO J/K operations still use the full AO basis.

Automatic CAS selection and execution use a two-stage reference policy. The
initial mean-field calculation is UHF so that broken-symmetry and open-shell
orbital evidence is available. Its orbitals are converted to a shared spatial
basis before the final spin-adapted CAS calculation, which uses RHF-style
closed-shell or ROHF-style open-shell behavior. Structured results report
`initial_reference` and `cas_reference` separately. Unrestricted CAS is allowed
only as an explicit advanced request or a spin-resolved orbital contract. See
[[Restricted vs Unrestricted Reference]].

## Failure Cases

- An MP2 request retains `xc=""` and fails validation even though no functional
  is scientifically relevant.
- A stretched/open-shell system is silently forced into a closed-shell
  single-reference path.
- A CAS calculation starts before the user has reviewed its active-space audit.
- A failed/unconverged result is shown only as a generic error, rather than a
  case-local CASSCF/active-space/runtime recovery choice.
