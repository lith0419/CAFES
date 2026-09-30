# PySCF Density Fitting And Local Orbitals

## Current Executable Scope

Molecular density fitting is an explicit task-spec option for the SCF/reference
stage: `density_fitting={enabled: true, auxbasis: ..., apply_to: "scf"}`.
Restricted/ROHF CASSCF supports `apply_to: "scf_and_casscf"` with either FCI
or block2. The shared backend explicitly configures PySCF DFCASSCF; block2
receives the fitted active-space integrals from that optimizer. Legacy `scf`
requests on restricted CASSCF are normalized to the combined scope, preserving
the DF inheritance that PySCF previously applied implicitly. Enabling DF in the
existing form is sufficient; no separate MCP implementation is needed.

Explicitly frozen external virtual MOs are removed from the CASSCF integral
workspace, while all occupied and active MOs remain. PySCF's existing AO2MO
and optimizer operate on the retained rectangular coefficient matrix. Results
and checkpoints restore the full MO layout; the mapping is recorded in
`cas_result.orbital_optimization.integral_space`. AO integrals and AO J/K
operations still use the full AO basis. This reuses the requested frozen list
and adds no automatic energy-window rule or scientific approximation beyond
that orbital constraint.

Results record requested `apply_to`, actual `applied_to`, `resolved_auxbasis`,
and `casscf_implementation`. An automatic auxiliary basis remains `null` in
the request and its resolved PySCF value appears in the result. Unrestricted
CASSCF fitting is unsupported. block2 CASCI currently uses conventional active
two-electron integrals even when its SCF reference uses fitting; its SCF flag
must not be described as DF-CASCI. `def2-svpd` accepts `def2-svp-jkfit`.
Validation normalizes known auxbasis aliases, accepts the registered auxbasis
sets, and rejects unsupported scopes. It is not an implicit performance default.

Boys and Pipek-Mezey localization are executable orbital-processing options.
With `localization_scope=analysis`, they generate orbital summaries without
changing the wavefunction used by the solver. With
`localization_scope=active_space`, an approved restricted/ROHF block2 CASCI or
CASSCF task localizes only the initial active orbital block, preserves the
inactive/active partition, checks overlap-metric orthonormality, and records the
input basis, output basis, active range, and numerical error in
`cas_result.orbital_provenance`. CASSCF subsequently optimizes those orbitals.

For active-space localization, optional `localization_occupation_thresholds`
partition the projected spin-summed SCF reference density's natural orbitals.
An empty list keeps whole-block localization; `[1.0]` separates virtual-like
and occupied-like groups. Each group is localized independently, preserving
the complete active subspace. Reference occupations and group membership are
recorded; they are not a new correlated-density diagnostic. The setting does
not apply to analysis-only localization and does not choose the active space.

block2 tasks may then use canonical, Fiedler, or explicit manual orbital
ordering. The requested and executed permutations are stored in the DMRG result
and MPS checkpoint manifest. Fixed-orbital continuation requests must match
both the ordering request and the executed permutation. DMRG-CASSCF keeps the
chosen permutation fixed across its internal solver calls. Compatible
cross-task DMRG-CASSCF continuation restores the optimized full-space orbitals
and MPS jointly. An expanded active space restores only the optimized orbitals
and constructs a fresh MPS after approval.

Periodic HF/DFT has a separate executable PBC density-fitting contract with
FFTDF, GDF, MDF, and AFTDF. GDF/MDF may carry a periodic auxiliary basis. These
options are validated under the periodic task and must not be translated into
the molecular SCF density-fitting fields.

## Numerical Artifact Storage

`initial_mo_coeff` accepts an inline matrix or a
`pyscf-agent.numeric-array.v1` reference containing path, shape and dtype.
Full MO guesses above 4096 elements are stored as lossless NPY files before
workflow history, new Study execution, and report copies are created. CAS loads
the selected array on demand with pickle disabled. Small inline inputs remain
supported. UNO/AVAS proposals and approvals preserve the same reference.
SSH stages local numerical references and rewrites their paths on the server;
references already pointing to remote files remain remote.

Existing DMRG/DMET RDM and one-particle-state NPZ artifacts retain their formats.
This storage change does not implement cross-geometry orbital projection,
transform an MPS, or change active-space selection.

## Roadmap Boundary

- Additional molecular post-HF and SGX density-fitting scopes; periodic post-HF density
  fitting beyond the current mean-field PBC path.
- General transformed two-electron-integral/AO2MO export. The registered
  molecular dataset already writes QH9-ordered Fock and overlap arrays; those
  one-electron matrices do not constitute general two-electron-integral export.
- IAO/IBO, rich orbital visualization, Molden/Cube export, and chemically named
  fragment definitions.
- Unrestricted DMRG-CASSCF (restricted state averaging is executable).

## Rules

- Do not add density fitting or localization silently to a plan; both are
  approximation/processing choices that belong in the visible task spec.
- Prefer size-aware artifacts for transformed integrals and orbital coefficients.
  Preserve transformation provenance. The existing `initial_mo_coeff` input
  remains supported alongside file-backed orbital references.
- If localized orbitals inform active-space selection, record the method and
  selection rationale in the audit.
- Active-space localization must be explicit and is currently limited to
  approved block2 CASCI/CASSCF. CASCI remains fixed-orbital; CASSCF performs
  PySCF orbital optimization with block2 as its active-space solver.
- Do not call AO2MO or IAO/IBO executable until separately registered and
  validated. PBC mean-field density fitting is executable only through its
  dedicated periodic registry entries and validation.

## Related Pages

- [[PySCF Reference Capability Boundary]]
- [[Strong Correlation Roadmap Rules]]
- [[Run Directory Artifact Storage]]
