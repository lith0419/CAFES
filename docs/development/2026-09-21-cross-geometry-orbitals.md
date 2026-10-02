# General orbital projection and calculation continuation

## Separation of responsibilities

Orbital continuation must not depend on a molecule name, a particular scan
anchor, active-space size, solver bond dimension, or cluster resource profile.
The implementation separates the mathematical operation from method policy;
workflow dependency resolution remains in the existing Study executor.

### Shared projection kernel (implemented)

`pyscf_agent/backend/orbital_projection.py::project_orbital_blocks` accepts a
source molecule, target molecule, real AO-by-MO coefficients, and ordered named
blocks of column indices. The blocks cover the supplied columns exactly once;
they may be noncontiguous or empty. The kernel has no checkpoint or block2
dependency and no knowledge of occupations, frozen cores, or active spaces.

Projection solves `S_target C_projected = S_target,source C_source`, using
PySCF `project_mo_nr2nr`. Each block is orthogonalized against earlier blocks
in the target AO metric, then symmetrically orthonormalized internally. Block
order therefore expresses a caller-selected priority, not a universal physical
rule. Mixing within a block is allowed; boundaries and dimensions are retained.

Rectangular coefficient matrices are supported. For example, an RHF caller can
transfer only occupied orbitals and form an initial density from the result.
Different scalar molecular AO bases are also allowed if the supplied orbital
rank survives. A partial set remains partial: this operation does not invent
missing virtual orbitals or establish compatibility with a downstream method.

Diagnostics include block columns and dimensions, residual metric eigenvalues,
cross-geometry overlap singular values, orthonormality errors, and numerical
tolerances. Rank loss and invalid source/output orthonormality fail explicitly.
Overlap acceptance thresholds belong to the calculation policy; an orthonormal
result alone does not establish continuity of the electronic state.

### CASSCF and checkpoint adapters (implemented)

`pyscf_agent/backend/correlation/orbital_projection.py` derives frozen inactive,
remaining inactive, active, and external blocks from calculation inputs. This
priority retains the validated CASSCF behavior. No atom counts, orbital counts,
geometry numbers, or bond dimensions are fixed in the adapter. Its admission
contract remains stricter than the kernel: full real MO space, matching atom/AO
layout, charge/spin, CAS definition, and frozen inactive partition.

The block2 checkpoint reader delegates projection to that adapter. Checkpoint
metadata describes source geometry and partitions; the provider retains its
own MPS compatibility checks. Other solver adapters can reuse the kernel without
importing block2. This does not yet introduce a solver-neutral persisted orbital
artifact; Study continuation uses the existing solver checkpoint adapter.

### Study dependency resolution

A Study case may set `solver.options.orbital_restart_manifest` to
`{"source_case_id": "case-0001", "artifact_kind": "block2_mps_manifest"}`.
The existing executor resolves this deferred reference to the successful parent's
saved artifact path just before execution. It records parent case, Run, and
artifact provenance in the resolved request and its resume fingerprint. A source
explicitly reported unconverged is rejected. Density continuation remains a
separate `initial_state.mode=projected_1rdm` input and may reference the same parent.

Parents must precede dependent cases in the plan. Orbital continuations are
excluded from independent batches. Missing/failed dependencies block children;
existing Study receipts, artifact fingerprints, collection, and selected-case
retry own execution and recovery. Orbital-only continuation does not transfer MPS.
A middle anchor and two outward branches are ordinary plan configuration, not
a molecule-specific scheduler. Electronic-state continuity still requires
scientific review; successful projection is not root following.

## Checkpoint safety

The old joint-checkpoint reader compared an AO-layout signature that excluded
coordinates, then symmetrically orthonormalized all MO columns in the target
metric. This neither projects between geometries nor preserves the partition
boundaries, and an orbital rotation cannot be silently combined with MPS reuse.

New joint checkpoints save the source PySCF molecule and frozen indices. A joint
`restart_manifest` defaults to identical geometry and returns the saved orbitals
unchanged after an orthonormality check. Missing legacy geometry metadata fails
explicitly; source geometry must be independently verified before migration.

`orbital_restart_manifest` transfers orbitals only. At a changed geometry it
uses PySCF `project_mo_nr2nr` (target-overlap inverse times cross overlap), then
metric Gram-Schmidt with symmetric orthonormalization inside each block: frozen
core, remaining inactive, active, external. Rank loss, changed atom/basis layout,
changed CAS or frozen partition, and frozen active/external constraints fail.
No canonical replacement orbital or MPS conversion is introduced. Whole-active
PM localization can then run on the transferred guess; a new MPS and its first
Fiedler order belong to that new calculation.

The projection kernel transfers an orbital guess, not an MPS or an electronic
state. It does not perform atom mapping, rigid-body alignment, root following,
or automatic active-space replacement. Coordinate-frame correspondence is a
caller responsibility. Periodic, complex/spinor, and unrestricted-pair transfer
are outside this interface's tested contract and require explicit adapters.

Reference: [PySCF orbital projection source](https://pyscf.org/_modules/pyscf/scf/addons.html#project_mo_nr2nr).

## Opt-in cross-geometry MPS initial guesses

Restricted/ROHF molecular DMRG-CASSCF may set
`solver.options.restart_geometry_policy="transport"` with a required joint
`restart_manifest`. The default `same_geometry` remains unchanged. Transport
requires the same atoms/AO layout, charge, spin, CAS and inactive counts, and
frozen partition. Existing block2 checks also require matching root count,
weights, symmetry and MPS site order. `target_scf_core`, orbital-only input and
optional fallback (`restart_required=false`) cannot be combined with transport.

The CASSCF adapter projects frozen inactive columns, then active, remaining
inactive and external columns. Partitionwise orthogonal Procrustes alignment
maximizes source/target overlap without changing orbital labels or partition
boundaries. The smallest active cross-overlap singular value must exceed
`restart_min_active_overlap` (default 0.9, configurable in (0,1]). Rank loss or
insufficient overlap fails explicitly; there is no silent cold restart.
Coordinate frames and atom correspondence are the caller's responsibility.

Saved MPS occupation coefficients are interpreted in these transported orbitals
as an **approximate initial guess**. This is not an exact real-space wavefunction
projection or a tensor-network orbital-rotation algorithm. No relocalization or
new Fiedler ordering is performed. The saved permutation is pinned even when the
manifest is supplied by filename. Target integrals are rebuilt and all roots
are reoptimized. Energy agreement after convergence, rather than the starting
energy, is the acceptance criterion. Root identities are not automatically
tracked through crossings, and speedup is not guaranteed.

`orbital_projection` records alignment, partition overlaps and orthonormality;
`casscf.mps_initial_guess_transport_applied` additionally requires the first
solver call to report successful MPS loading. Source files are copied into a
new scratch directory and retained unchanged. A Study `restart_manifest` may
use the same deferred `source_case_id`/`artifact_kind` reference as orbital-only
continuation, with parent Run provenance and sequential dependency handling.

Reference: [Hu and Chan, excited-state geometry optimization with DMRG](https://arxiv.org/html/1502.07731v2#S5).

On 2026-09-24, Slurm test 61824800 passed all 86 targeted tests with real block2
and no skips. H4/STO-3G single-root and equal-weight SA4 transports from 1.00 to
1.03 Angstrom spacing preserve source checkpoint hashes and site permutations;
all converged target energies agree with independent cold starts within 1e-7
Hartree. Contract tests cover gauge covariance, overlap rejection, opt-in
admission, partition mismatch, and deferred Study MPS dependencies.

## Validation

Local tests use LiH, H2O, and H2, not the production molecule. They cover identity,
cross-overlap projection, noncontiguous blocks, column permutations, intrablock
rotations, four nonempty CASSCF partitions, empty partitions, rectangular
occupied-only guesses, an actual RHF restart, changed AO bases, and rejection of
invalid inputs/rank loss. Checkpoint tests distinguish same-basis MPS reuse from
orbital-only guesses. A saved four-block H2O projection is bitwise unchanged by
the shared-kernel refactor.

The portable remote release was tested with 90 real-block2 regressions (no
skips) and a three-geometry H2O acceptance with four nonempty partitions on
multiple Amarel nodes. All 22 DMRG calls converged; projected orthonormality error
was at most 1.11e-15, frozen-orbital change during optimization was zero, and
fixed-orbital FCI energy agreed to floating-point precision. Shared logical
scratch paths remain readable across nodes. This validates orbital transfer;
the new Study dependency adapter additionally requires a sequential three-point
acceptance before launching a production chain.

Study adapter tests cover three-point parent resolution, exclusion from
independent batches, completed-case reuse, combined density/orbital references,
changed parent Run provenance, and blocked failed/unconverged or invalid sources.

## Target-SCF frozen core policy

`orbital_processing.continuation_policy="target_scf_core"` is an explicit
orbital-only CASSCF policy. Frozen columns come from the current geometry's
converged shared-spatial SCF reference, at the configured frozen indices.
Projected active columns take the next priority, followed by other inactive
columns. PySCF's public `mcscf.addons.project_init_guess` performs the SVD
orthogonalization and completes the entire external space from the target
reference basis. External MOs are not truncated or individually tracked.
No molecule name, atom count, CAS size, or M occurs in this implementation.

The adapter checks source/reference orthonormality and residual rank of every
supplied block before invoking the SVD routine, and verifies the target frozen
columns and final orthonormality. It records the policy and frozen reference;
initial frozen columns are retained as a binary result array and compared to
the final optimized columns. MPS reuse is forbidden with this policy, including
at unchanged geometry. Existing `project_all` remains the default for old inputs.
The current public checkpoint route is block2; the mathematical adapter calls
PySCF and is independent of block2.

The low cross-geometry overlap of a tightly localized core orbital alone does
not diagnose a bad chemical state: it also reflects nuclear displacement.
This policy deliberately selects the core in the *target* geometry instead of
freezing a transported source-core shape. Active-space/state continuity must
still be reviewed. A translated small-molecule test exercises this distinction;
a CASSCF initial guess can reach a different orbital minimum even with valid
core and full-space orthonormality.
