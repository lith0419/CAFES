# Model Hamiltonian Solver Support

## Scope

This document defines solver support for model Hamiltonian calculations.

## Current Design

The model Hamiltonian backend supports solver choices such as:

- MP2
- CCSD
- CCSD(T)
- Full CI
- block2 DMRG (`block2_dmrg`, optional dependency)
- libDMET DMET (`dmet`, optional dependency)
- Bloch tight binding (`tight_binding`)

Solver support must be checked through the backend constants and capability
registry.

Mean-field-based model Hamiltonian solvers use an unrestricted reference by
default. This applies to MP2, CCSD, and CCSD(T), and to the mean-field reference
used when reporting Full CI reference and correlation-energy diagnostics.

The representation determines the solver family:

- `finite_cluster` uses the real-space many-body path and requires `nelec` as
  `[nalpha, nbeta]`.
- `bloch` uses a primitive cell, lattice vectors, integer bond `cell_offset`
  values, electrons per cell, and reciprocal-space sampling. Its current
  executable solver is `tight_binding` only.

The Bloch solver constructs
`h_ij(k) = sum_R t_ij(R) exp(2 pi i k.R)` in reduced reciprocal coordinates,
adds the declared Hermitian conjugates, diagonalizes each k point, and applies
zero-temperature spin-degenerate filling. It reports band energy per cell,
Fermi energy, VBM/CBM, direct and indirect gaps, bandwidth, a high-symmetry
band path, Gaussian DOS, and k-mesh occupations. Nonzero U and V terms are
preserved in the input and explicitly reported as not applied.

The executable Bloch contract is resource bounded before array allocation:
at most 50,000 mesh k-points, 64 custom path vertices, 20,000 interpolated
path points, 4,001 DOS grid points, and 200,000,000 combined
`k-point x orbital x DOS-grid` evaluations. DOS broadening is accumulated in
bounded chunks. The builder dynamically reduces its DOS-point limit when the
current mesh and primitive-cell orbital count would exceed this budget.

## Rules

- For finite clusters, the builder should not hard-code the correlated solver.
  A Bloch input declares `tight_binding` because that solver is part of the
  representation contract.
- The single-calculation agent should choose a solver for one model task.
- The planner may vary solver across cases.
- Unsupported solvers must be rejected before execution.
- FCI and block2 DMRG accept `solver.options.nroots` for targeted low-energy
  roots in one particle/spin sector. Requesting `excited_states` without an
  explicit value defaults to two roots. These roots are not a claim that the
  complete many-body spectrum was computed.
- FCI requests that explicitly need a complete spectrum or dense
  strong-correlation observables use full dense diagonalization. Reject
  determinant dimensions above 5,000 before matrix construction and show the
  cost estimate; targeted-root and energy-only FCI requests may use the
  ordinary eigensolver path.
- block2 DMRG is a finite-cluster solver. It can produce 1RDM/2RDM artifacts,
  density, double occupancy, spin/charge correlations, orbital entanglement,
  targeted low-lying roots, and spin/particle symmetry checks. The common
  molecular/model result contract records the requested and computed roots,
  sector, root signatures, convergence, bond dimension, error estimate,
  checkpoint availability, and which roots supplied RDMs and diagnostics.
- Explicit result analysis tracks compatible DMRG roots across ordered cases.
  It compares per-root 1RDMs when available, falls back to natural occupations,
  and treats excitation energy only as secondary evidence. Ambiguous mappings
  are reported for review and are excluded from continuation anchors.
- A saved block2 MPS manifest can explicitly initialize a compatible later
  task. Validate orbital count, electron/spin sector, symmetry backend, and root
  count plus the requested and executed orbital ordering before staging it;
  record whether the Hamiltonian fingerprint changed.
- Accept block2 DMRG convergence only when both energy change and discarded
  weight meet their configured tolerances. A same-method recovery may extend
  sweeps or increase bond dimension through a compatible checkpoint up to the
  Registry automatic-recovery limit (currently `M=1024`). This policy budget
  does not limit an explicitly approved larger-M calculation.
- DMET is a finite-cluster model solver backed by one composite libDMET module.
  It consumes Builder sites and bonds directly. Translated execution requires
  primitive-cell translation equivalence and no intersite V; native uniform
  chain/ring and square lattices are also supported. Open, translation-breaking,
  extended-Hubbard, and arbitrary finite graphs use an explicit partition.
  Set `solver.options.execution_mode=finite_graph` to preserve intersite V;
  explicit `translational` requests that fail the audit are rejected. FCI, CCSD,
  and block2 DMRG impurity solvers support restricted or broken-symmetry UHF
  references; `auto`
  defaults to UHF so local-moment solutions are not excluded by construction.
- Unrestricted DMET keeps separate alpha/beta density, Fock, embedding, and
  correlation-potential blocks. Finite-graph partitions require `nalpha=nbeta`;
  translated representative fragments may use a validated nonzero spin sector.
  Reject nonzero fixed spin projection in finite-graph mode, excited roots, incomplete or
  overlapping fragment partitions, and translated impurity shapes that do not
  tile the lattice. Do not silently alter the Builder Hamiltonian.
- Treat DMET convergence as an outer self-consistency condition on energy
  change, consecutive mean-field 1RDM change, and density-fit residual;
  correlation-potential change is diagnostic only. Persist the iteration
  history and numerical embedding arrays under their registered artifact
  contracts. Outer convergence does not establish that every intermediate
  impurity SCF converged; inspect that evidence separately.
- DMET correlation-potential fitting must use the same Hamiltonian as lattice
  mean field across translational and finite-graph routes, for every impurity
  solver. Preserve the physical Fock for interacting-bath construction even
  when the density fitter uses hcore. For cell-local interactions, construct
  the physical HF potential from the k-averaged density and apply it at every
  k point. This is density-matrix fitting, separate from molecular auxiliary-
  basis density fitting.
- Energy spectra should be saved for the current method when available, not only
  for Full CI.
- Do not offer or execute methods absent from the capability registry.
- Model-Hamiltonian Planner studies use static full-grid scans with an explicit
  solver. Strong-correlation diagnostics are calculated for each case and may
  guide a later user-reviewed study revision; they do not trigger a molecular
  initial-scan or active-space workflow. Full CI is a small-system benchmark,
  not a silent default.
- Do not interpret a Bloch tight-binding result as a correlated Hubbard result
  when U or V is nonzero.
- Bloch tight binding is backend-executable but is not yet a Planner adaptive
  routing option.
- Reject reciprocal-space requests that exceed the registered resource limits
  before constructing k-mesh, band-path, occupation, or DOS arrays.
- Persist model strong-correlation diagnostics, full spectra, Bloch bands, DOS,
  k-mesh occupations, and sampled Hamiltonians under their registered JSON/TSV
  artifact kinds.

## Units

Model Hamiltonian energy inputs and outputs should carry an explicit unit. Use
`a.u.` for atomic units.

## Rationale

Model Hamiltonian calculations often compare solvers or parameters. Keeping the
solver separate from the structure makes those comparisons possible.
