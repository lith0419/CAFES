# Task Spec Rules

## Scope

This document defines rules for task specifications used by the single
calculation agent and planner agent.

## Current Design

Task specs convert user intent into structured, reviewable input. They are not
raw prompts and should not contain ambiguous natural language when execution is
about to begin.

## Rules

- A task spec must identify the task family.
- Molecular task specs should contain molecular structure, method, basis,
  charge, spin, reference type, job type, and requested outputs when applicable.
- Model Hamiltonian task specs should contain a model input file or normalized
  model spec, plus a supported solver.
- Periodic task specs should contain POSCAR/CIF structure text plus registered
  periodic basis, pseudopotential, method, k-point, numerical, and optional
  band-path settings.
- A task spec must be validated before execution.
- Missing mandatory fields should cause clarification or a blocked state, not
  silent execution.
- User-visible UI controls should reflect task spec updates in real time.
- Execution target and resource-profile selection are transport metadata and
  must not be serialized as scientific method or solver fields in `TaskSpec`.

## Molecular Fields

Expected molecular fields include:

- `task_type`
- `atom`
- `basis`
- `method`
- `xc` for DFT
- `charge`
- `spin`
- `restricted`
- `job`
- `outputs`

An automatic active-space request first executes a registered HF/SCF probe. A
separate registered refinement module may add MP2 evidence when the preliminary
chemical-valence/AVAS proposal is not review-ready or SCF diagnostics show
instability, frontier degeneracy, or a routing-boundary score. The structured probe
contract records the eventual CASCI/CASSCF target method, target solver, and
target solver options. Approval restores those target fields; probe methods and
their temporary requirements must not become the formal calculation method.

Reference convergence and target-solver convergence are separate evidence. For
CASSCF, the mean-field calculation supplies an initial orbital guess and the
variational orbital optimizer supplies the target result. An unconverged initial
reference is therefore retained as a warning but does not invalidate a
converged CASSCF result. CASCI and single-reference methods still require their
reference to converge. An automatic reference retry preserves the requested SCF
algorithm and doubles `runtime.max_cycle`. Newton is an explicit
`runtime.scf_algorithm=newton` choice, not an automatic fallback. Approval must
preserve those runtime controls instead of rebuilding defaults.

An automatic molecular SCF retry uses the latest attempt's saved
`one_particle_state` as `initial_state.mode=projected_1rdm` when the reference
policy is unchanged. PySCF receives that AO density through its native `dm0`
argument. The artifact records `scf_converged=false` for an unfinished SCF;
the restart report retains `initial_state.source_scf_converged`. This is a
starting guess, not accepted scientific evidence. Recovery never selects an
older artifact just because its filename is present in the run directory.
Changing the restricted/unrestricted policy starts with a fresh compatible
guess. Density restart does not restore DIIS history, Newton inner iterations,
or DF integral caches.

For CASCI/CASSCF, `method.restricted` selects the CAS formulation, while the
initial reference remains UHF. One-particle-state metadata and restart
compatibility follow the actual initial SCF policy, so a converged UHF probe
density can seed spin-adapted CAS together with its approved orbital matrix.

SCF stability is an optional diagnostic. It runs only when
`workflow.module_config["molecular.correlation_diagnostics"].scf_stability=true`.
The default reports `status=not_requested` and `stable=null`; neither convergence
nor an omitted check establishes stability. UNO candidates need the converged
UHF spin-summed AO 1-RDM and overlap, not a correlated 2-RDM or a stability check.
Their saved `initial_mo_coeff` defines the core-active-virtual orbital basis;
canonical indices in the audit are display mappings, not replacement orbitals.

## Model Hamiltonian Fields

Expected model Hamiltonian fields include:

- `task_type`
- `model_hamiltonian_input_file` or normalized `model_hamiltonian.spec`
- `solver`

## Periodic Fields

Expected periodic fields include:

- `task_type: periodic`
- `periodic.structure_text` and `periodic.structure_format`
- `periodic.basis` and `periodic.pseudo`
- `method`, plus `xc` only for DFT
- `charge`, `spin`, and `restricted`
- `periodic.kmesh`, `kpoint_scheme`, and optional shift
- precision/mesh or auxiliary cutoff, PBC density fitting, `exxdiv`, and
  optional smearing settings
- optional high-symmetry band-path mode and its mode-specific parameters

Periodic tasks are executable in the Calculation Assistant but remain outside
the current Computational Study Planner contract.

## Rationale

Task specs make LLM interpretation auditable. They also make it possible to
separate conversation from execution.

## Failure Cases

- User asks for Hartree-Fock but the task spec remains DFT/B3LYP.
- User updates basis in conversation but the UI does not update the control.
- Model Hamiltonian input exists but the solver is not selected.

All of these should be caught before execution.
