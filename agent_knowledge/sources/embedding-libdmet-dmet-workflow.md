# libDMET Hubbard Workflow

## Executable Boundary

The optional libDMET provider executes ground-state density matrix embedding
theory for single-band Hubbard models built by the model-Hamiltonian interface.
The current adapter supports:

- translated representative fragments when the Builder primitive-cell graph
  passes translation-equivalence checks, or the native uniform periodic
  chain/ring or square-lattice contract applies;
- explicit finite-graph fragment partitions for open boundaries, arbitrary
  finite graphs, translation-breaking site/bond parameters, nonzero intersite
  V, or an explicitly requested complete partition;
- on-site `U`, site energy, arbitrary hopping, and intersite `V` from the
  Builder ModelSpec without reconstructing a different lattice;
- restricted references and fixed-spin UHF references with independently
  optimized alpha/beta densities. In the libDMET API, `Sz` means
  `Nalpha-Nbeta=2S_z`;
- FCI, CCSD, or block2 DMRG impurity solvers;
- occupation-aware embedding baths (SVD for idempotent references and the
  libDMET eigenvalue construction for fractional occupations), interacting-bath
  projection when `V` is present, correlation-potential fitting, DIIS, and
  bounded self-consistency iterations.

It rejects excited roots and ab initio molecular/periodic Hamiltonians. The
block2 route is a ground-state impurity solver inside the DMET loop, not a
replacement for the top-level `dmet` solver. Fixed nonzero `S_z` is supported only by the translated
representative-fragment path; fragment-specific spin-sector allocation for a
finite-graph partition is not yet implemented. These boundaries are not
silently approximated.

Both execution modes are DMET. The translated mode exploits an explicit
equivalence assumption and solves one representative impurity. The finite-graph
mode instead preserves the complete Builder Hamiltonian and solves every
non-overlapping fragment in a partition. A local perturbation to `U`, site
energy, hopping, or `V` therefore invalidates the translated assumption and
automatically selects finite-graph execution.

Use `solver.options.execution_mode=finite_graph` for intersite V. Explicit
`translational` requests must pass all translation checks, including
`no_intersite_interaction`; a uniform nonzero V still fails that check. When the
mode is omitted, validation resolves it from the supplied graph. A multi-site
primitive cell alone does not prohibit translated execution if its graph passes
the current audit. Preserve the supplied sites, bonds, U, V, and hopping.

## Request Contract

Use `task_type=model_hamiltonian`, `representation=finite_cluster`, and
`solver.name=dmet`. `solver.options` may contain:

- `fragments`: canonical explicit fragment partition. Each fragment contains a
  stable `fragment_id` plus Builder `site_ids` or correlated
  `orbital_indices`. Finite-graph DMET requires non-overlapping fragments that
  cover every site;
- `impurity_shape`: shape in primitive-cell units. It is a geometric
  convenience for a regular translated lattice and a complete-cell grouping
  rule for a finite graph;
- `impurity_size`: sites per fragment for automatic finite-graph partitioning;
- `fragment_definition=honeycomb_hexagon`: disjoint elementary six-site rings
  on a periodic honeycomb. Use automatic or finite-graph execution, without any
  other fragment selector. A 6x6 primitive-cell lattice has 72 sites and 12
  hexagons; 3x3 and 3x6 are also compatible. A 4x4 lattice has 32 sites and is
  rejected, as are incompatible periodic seams (for example 4x6). This solves
  all fragments and preserves the full graph, not one translated representative;
- `impurity_site_ids`: convenience form for one explicit impurity. It is most
  useful as the shared input boundary for a future single-impurity DMFT route;
- `impurity_solver`: `fci`, `ccsd`, or `block2_dmrg`;
- `impurity_solver_options`: CCSD `beta` (default 1000) shared by preliminary
  impurity SCF, lattice mean-field and density fitting, or block2 controls such as
  preset, orbital ordering, sweep schedule, thread count, and memory. DMET
  obtains the required impurity 1RDM and 2RDM and reuses compatible MPS state
  between consecutive self-consistency iterations;
- `reference`: `auto`, `restricted`, or `unrestricted`. `auto` remains the
  Hubbard-aware default and resolves to UHF. Nonzero spin is taken from the
  ModelSpec `nelec=[nalpha, nbeta]` and requires a translated representative
  fragment; choose `restricted` explicitly only when `nalpha=nbeta`;
- `max_iterations`, `energy_tolerance`, `density_tolerance`, and
  `density_fit_tolerance`. Their defaults are 50 cycles, `1e-6` for the DMET
  energy change, `1e-4` for the maximum consecutive mean-field 1RDM change,
  and `1e-4` for the correlated-to-auxiliary density residual. Convergence
  follows the native DMET rule (energy and mean-field 1RDM stationary); the
  density-fit residual is a reported, non-required quality diagnostic, because
  a single determinant need not reproduce a correlated 1RDM (for example in a
  strong charge-density wave);
- `reference_density_guess`: `pm`, `af`, `fm`, or `cdw`. `cdw` adds the
  staggered bias to both spins (sublattice charge, no magnetization) and needs
  a bipartite graph; it works with restricted or unrestricted references;
- `reference_density_source`: optional finite-graph density warm start from
  a saved converged neighboring Run. Use `{"source_case_id": "saved-neighbor-id"}`
  in a Study retry, or a fixed absolute `path` with `sha256`. The Study resolves
  the source Run and its `dmet_mean_field_state` artifact. The provider checks
  site basis, fragments, spin/reference and solver compatibility, physical
  density bounds, and source convergence/quality. Only U/V may change. The
  full final mean-field 1RDM overrides the analytic seed; the target rebuilds
  its physical baseline and starts its auxiliary correction at zero. Old Runs
  without this artifact cannot supply their initial seed as a substitute;
- bounded impurity-solver cycle, tolerance, and memory controls.

The Registry entry `model_hamiltonian.solver.dmet` is the sole scientific
option contract. It owns defaults, ranges, choices, and automatic-resolution
policies; the provider normalizer reads that contract rather than maintaining a
second constants table. Omitting `execution_mode` lets the validated Builder
graph choose translated execution when its primitive-cell translations remain
equivalent and finite-graph execution otherwise. Omitting
`fragment_definition` uses an explicit selector when present and the Builder
primitive cell otherwise. The Web UI's `Automatic` choice leaves these fields
absent instead of serializing a forced mode.

Top-level `solver.options` keys must appear in that Registry contract. Unknown
keys, including misspelled tolerance names and the unregistered `conv_tol`,
are rejected before execution. They are not retained or silently replaced
by defaults. Nested `impurity_solver_options` still follow the selected
impurity provider's existing contract.

Only one fragment selector may be supplied. Every selector is normalized to the
same explicit fragment records before execution. The Builder's site ids are
mapped to the sorted site-orbital basis and preserved in the result, preventing
an accepted graph from being silently replaced by a regular lattice.

For DMFT, impurity size is derived from the selected correlated orbitals rather
than treated as an independent physical quantity. The first fcDMFT adapter
should accept exactly one inequivalent fragment; `impurity_size` is a
convenience, while `fragments[0].orbital_indices` remains the canonical input.
Cluster DMFT can later relax the one-fragment restriction without changing the
shared contract.

The libDMET 0.5 unrestricted impurity-SCF path passes two equivalent spin
overlap blocks while current PySCF expects one common overlap matrix. The
adapter verifies that the blocks agree numerically before replacing them with
their common matrix. Distinct alpha/beta density, Fock, correlation-potential,
and embedding blocks remain spin resolved. A provider result with genuinely
different alpha/beta overlaps is rejected rather than silently collapsed.

After projecting the mean-field density into the embedding basis, an
idempotent reference takes its impurity electron count from the integer
projected trace. This is essential when the bath collapses, as in the atomic
limit. For a fractionally occupied ensemble, the eigenvalue bath contains
additional partially occupied environment modes and does not itself define one
integer solver sector; the adapter therefore retains the validated fixed
valence-sector electron count. The selected rule, projected traces, orbital
capacity, and final spin sector are recorded in the result.

## Runtime Composition

`embedding.libdmet.dmet` is one composite task module. It implements the
registered capabilities `build_dmet_bath`, `solve_dmet_impurity`,
`fit_dmet_correlation_potential`, `iterate_dmet_self_consistency`, and
`embedding.method.dmet`. The module runs before input generation and prepares
the libDMET provider contract consumed by the normal model-Hamiltonian executor.

The composite module has no independent module configuration. A nonempty
`workflow.module_config["embedding.libdmet.dmet"]` is rejected because DMET
settings already belong to `TaskSpec.solver.options`. This prevents a request
from carrying two values for the same physical choice or accepting a value that
the runtime silently ignores.

The stages remain one runtime module because the current libDMET public workflow
passes mutable lattice, basis, chemical-potential, density, impurity-Hamiltonian,
and correlation-potential objects directly between them. They are not presented
as independently executable tools. Shared localization and embedding artifact
contracts remain separate so a later DMFT provider can reuse stable inputs
without duplicating DMET execution.

## Results And Artifacts

The structured result reports:

- DMET energy per site and the corresponding total finite-lattice energy;
- the selected translated or finite-graph execution mode plus the individual
  eligibility checks that led to it;
- fragment ids, Builder site membership, size, energy contribution, and
  electron count;
- impurity solver, reference type, bath construction, and chemical potentials;
- the bath-basis policy and per-iteration impurity electron-count selection;
- requested alpha/beta counts plus fragment alpha/beta populations and local
  `nalpha-nbeta` magnetization for unrestricted calculations;
- outer-loop convergence and every iteration's energy change, maximum
  mean-field 1RDM change, density-fit error, and correlation-potential change.

The convergence decision requires the energy change and consecutive mean-field
1RDM change to satisfy their registered tolerances. The correlated-to-auxiliary
density-fit residual is a non-required diagnostic. For translated execution the energy criterion is
applied to the representative fragment/cell energy; for finite-graph execution
it is applied to the summed fragment energy. Correlation-potential changes are
recorded for diagnosis but do not define convergence.

DMET also emits a compact `quality_checks` contract for the existing
`task.execution_quality` gate. The provider reports observations, operators,
limits, and sources; the gate computes pass/fail independently. Required
convergence checks cover the final energy and mean-field 1RDM changes; the
density-fit residual check is optional. Required constraints cover finite/normalized energy, aggregate
impurity-orbital capacity, spin-sector consistency, and projected electron-
sector consistency. When a block2 impurity summary exposes its own convergence
flag, that observation is included as well. A failed convergence check requests
review, while a failed or malformed constraint blocks publication eligibility.
Older methods without `quality_checks` remain displayable and are explicitly
marked as legacy evidence. Their `publication_eligible` is null (unknown),
not verified true; a failed execution remains ineligible.

Every declared check requires a non-null `observed` value. `equal` also requires
non-null `expected`: both values must be booleans, strings, or finite numbers of
compatible types. Boolean-to-number equality is invalid; integer/float numeric
equality remains supported. `finite` requires a finite number, and threshold
operators require finite numeric `observed` and `limit` values. Missing,
malformed, or overflowing evidence is recorded as invalid rather than passing
or raising an unhandled conversion error. Optional checks remain advisory.

Computational Study rows preserve every raw result and add `quality_status` and
`publication_eligible`. The study gate treats an ineligible row as unresolved,
and built-in postprocessing excludes it from plots. The `U=0` and `t=0` limits
remain regression tests rather than per-run convergence criteria.

Before the stopping test, the adapter assembles the correlated lattice 1RDM and
uses libDMET's full-lattice correlation-potential fit with the native
finite-temperature BFGS setup (`beta=1000`, at most 300 fit iterations). This
keeps the numerical fitting semantics in libDMET; the agent layer only supplies
the translated representative or finite-graph fragment orchestration.

Four registered artifact families are written:

- `dmet_result`: compact scientific summary;
- `dmet_iteration_history`: convergence trace;
- `dmet_numerical_arrays`: NPZ archive containing the embedding basis, density
  matrices, correlation potential, and impurity one-/two-body Hamiltonian
  components;
- `dmet_output_log`: complete libDMET and impurity-solver stdout/stderr captured
  in `solver-dmet/log-libdmet-output.log`. It is retained for both successful
  and failed executions without embedding the verbose text in structured JSON.

The outer DMET convergence flag is distinct from lattice and impurity SCF
convergence. For CCSD, `impurity_solver_details` records every preliminary
impurity SCF call, including its iteration count, convergence flag, and
unshifted orbital-gradient norm. It also reports the final CCSD amplitude and
lambda convergence flags. These are not lattice-SCF convergence evidence.

CCSD preliminary RHF/UHF uses native PySCF ADIIS/CDIIS blending by default.
The controller uses the RMS orbital commutator in an orthonormal basis:
ADIIS above `1e-2`, CDIIS below `1e-4`, and a linear blend between them.
Both native extrapolators keep the same input history, with libDMET's native
12-vector default. Each impurity solve starts a fresh history. CCSD is stopped
before amplitude solving if preliminary SCF is unconverged, nonfinite, or
fails its unshifted gradient tolerance. This policy applies to both restricted
and unrestricted CCSD references; it uses ordinary SCF, without Newton,
level-shift retries, or relaxed tolerances. FCI retains its native SCF policy.

CCSD owns `solver.options.impurity_solver_options.beta`, default `1000.0`.
The same positive finite value controls preliminary impurity RHF/UHF, DMET
lattice RHF/UHF and correlation-potential density fitting in both translational
and finite-graph execution. Fermi width is `1/beta` in model-energy units.
CCSD and its correlated RDMs retain integer electron counts; this is not
finite-temperature CCSD. Every impurity chemical-potential trial uses the same
beta. FCI and block2 retain their existing policies: zero-temperature lattice
mean-field and density-fit beta 1000; no new smearing setting is added to them.

The Calculation Assistant and Study Planner expose **CCSD / DMET β** only for
CCSD-DMET. Blank input resolves to 1000. Saved explicit beta values are retained
and now also control lattice SCF and density fitting. Switching to FCI or block2
excludes the CCSD field. Invalid explicit values are rejected, never defaulted.

Results record effective beta and width in `dmet_result.smearing`, fitting beta
in the iteration-history contract and measured beta in impurity SCF call records.
Legacy reports without this evidence are not assigned a new effective value.
Smearing is not a convergence guarantee: the documented six-site [4,2] CCSD
reference retains a degeneracy and diverges at beta=1000. Such a failure remains
failed; no automatic zero-temperature retry or solver substitution is applied.
The former optional zero-temperature CCSD default is superseded. Historical
runs and the September 24 comparison remain unchanged evidence of that policy.
See `docs/development/2026-09-25-ccsd-dmet-shared-beta.md` for the current contract.

Recovery must identify the failing loop. `runtime.max_cycle` does not control
DMET; `max_iterations` controls its outer loop, while `solver_max_cycle` is
passed to the correlated impurity solver. A traceback identifying a linear
algebra failure in preliminary impurity SCF DIIS supports a reviewed retry
with `solver.options.impurity_scf_diis=false` (default true). This registered
option controls only the native FCI/CCSD impurity SCF instances, including
chemical-potential fitting calls. It does not change lattice or outer-loop
DIIS, the Hamiltonian, or the convergence tolerances. The block2 impurity bridge
does not run this preliminary SCF and rejects disabling this option.
For CCSD, disabling DIIS disables both components of the hybrid and still
requires converged preliminary SCF.

The Study gate supplies the proposal, reason and affected case; accepting it
prepares a subset plan, and Run Plan executes it through the existing agent.
Do not offer the same change again when it is already in effect or classify
every LinAlgError as a DIIS error. Disabling DIIS avoids that extrapolation
but does not guarantee SCF or DMET convergence. Full CI can converge with
unconverged preliminary HF orbitals; verify the FCI and DMET results separately
rather than reporting all intermediate SCFs as converged.

## Verification

Fast contract tests cover a periodic Hubbard ring, an open nonuniform finite
graph, and an extended-Hubbard graph with intersite `V`. The optional numerical
tests also cover the fractionally occupied `4x4`, half-filled, `U=0` square
lattice and the bath-collapsed `t=0` atomic limit. The optional numerical
benchmark fixes an 18-site, two-site-fragment, half-filled one-dimensional
Hubbard calculation at `U/t=4` to the agent's interacting-bath, zero-correlation-
potential protocol. It is a product-protocol regression benchmark rather than
the noninteracting-bath protocol used by the upstream libDMET example. Enable
it with `PYSCF_AGENT_RUN_LIBDMET_BENCHMARKS=1` when libDMET is installed.

## Related Pages

- [[Unified Registry Runtime Boundary]]
- [[Model Hamiltonian Solver Support]]
- [[Strong Correlation Rules And Roadmap]]
- [[DMET Density Continuation And Branch Analysis]]
- [[Model Grid Convergence Barriers]]

## Spin Bath Dimension Policy

`bath_spin_dimension_policy` defaults to `native`. The explicit `max` option
completes the smaller spin eigenvalue bath to the larger dimension with omitted
orthonormal environment eigenvectors, retaining every originally selected mode.
It requires finite-graph execution and a noninteracting bath. SVD baths retain
native behavior. Each fragment records the original and completed dimensions,
added eigenvectors, and orthogonality error. This option changes bath construction,
not density normalization, electron counts, or convergence tolerances.

### Comparing and continuing DMET branches

Use `analyze_dmet_branches` to compare all indexed independent and continuation
Runs by final correlated sublattice order parameters and qualified energy per
site. Initial seed labels do not identify the final branch. The report records
thresholds, mixed/uncertain states, candidate provenance, minima, possible
hysteresis, and adjacent AFM/CDW energy-crossing brackets.

`prepare_dmet_continuation` previews a bounded sweep and cost;
`start_dmet_continuation` starts the saved action through the Study retry runner.
Default sweeps propagate AFM toward increasing V and CDW toward decreasing V at
fixed U, choosing a fresh nearest compatible qualified donor at each step.
`mode="nearest"` selects neighbors in U/V. Defaults are mixing 0.2, DIIS off,
`bath_spin_dimension_policy="max"`, and **density only, initial u=0**. Missing
donors skip a step. Previous numerical Runs stay in candidate history and are
not overwritten by later retries. A failed continuation does not establish a
physical branch endpoint. After interruption, collect existing Runs before
preparing another sweep; repeated starts of the same action do not resubmit.
