# Development Note: block2 Low-Energy Study Workflows

Date: 2026-08-16

## Scope

This update completes the current block2 boundary for molecular active spaces
and finite model Hamiltonians. It adds explicit low-energy-root semantics,
cross-task state continuity, a shared result contract, checkpoint-aware
recovery, scheduler resource guidance, and benchmark coverage. The behavior is
defined by task and study contracts rather than molecule- or lattice-specific
rules.

## Low-Energy Roots

- Molecular FCI, CASCI, CASSCF, and model FCI/block2 DMRG accept
  `solver.options.nroots`; requesting `excited_states` without a value defaults
  to two roots.
- Multi-root CASSCF uses normalized state-average weights. block2 state
  averaging is restricted to one particle/spin/symmetry sector.
- Low-energy diagnostics distinguish numerical degeneracy, near degeneracy,
  and an isolated ground state within the computed sector. If every requested
  root lies in the near-degenerate window, the report recommends increasing
  `nroots` instead of claiming a complete manifold.
- Root 1RDMs may be generated for all targeted roots. 2RDM and entanglement
  diagnostics remain ground-state outputs by default unless the orbital
  optimizer requires per-root RDMs.

## State Continuity And MPS Reuse

- Explicit Planner result analysis writes
  `adaptive-dmrg-state-tracking.json` with schema
  `pyscf-agent.dmrg-state-tracking.v1`.
- State assignment prefers root 1RDM similarity, falls back to natural-
  occupation similarity, and uses excitation energy only as secondary evidence.
  Root reordering is recorded; ambiguous cases require review and cannot seed a
  continuation retry.
- MPS continuation keeps up to four compatible source candidates, selects a
  state-trusted source on the same execution target, preserves source/checkpoint
  and Hamiltonian fingerprints, and requires approval before rerunning the
  affected subset.

## Shared Result And Recovery Contracts

- Molecular and finite-model block2 runs emit
  `pyscf-agent.block2-result-contract.v1`. It records requested/computed roots,
  state sector, root signatures, result scope, final bond dimension, discarded
  weight, energy change, sweeps, error estimate, and checkpoint availability.
- Unconverged DMRG emits
  `pyscf-agent.block2-recovery-recommendation.v1`. The bounded same-method path
  extends sweeps or increases bond dimension through a compatible checkpoint,
  with automatic bond-dimension escalation capped at `M=1024`.
- Slurm OOM and timeout outcomes request larger memory or wall-time profiles.
  Node failures and preemption may retry the same profile. A terminal scheduler
  failure without a result file becomes a synthetic failed per-case
  `TaskReport`, preserving successful sibling cases.

## Benchmarks And Registry

- The optional Hubbard-ring benchmark now checks two block2 roots against FCI,
  the shared result contract, warm/cold continuation, observables, and root
  tracking from `U=4` to `U=6`.
- The capability registry declares molecular/model root controls, result scope,
  state tracking, recovery schemas, and the new artifact kind. LLM instructions
  distinguish requestable outputs from generated fields and targeted roots from
  a complete spectrum.

## Remaining Boundaries

- Transition RDMs and direct MPS wavefunction overlaps are not available, so a
  weak or ambiguous cross-case state assignment remains reviewable evidence.
- Different-spin and unrestricted DMRG-CASSCF, block2 SC-NEVPT2, DMET, and DMFT
  remain outside the executable boundary.
- The automatic bond-dimension limit and scheduler profiles are operational
  policies, not universal accuracy guarantees; larger calculations need
  hardware-specific calibration.

## Verification

- Full Python regression suite: 575 tests passed on 2026-08-16.
- Targeted block2 contract/recovery/state-tracking tests passed.
- Deterministic Wiki curation wrote 40 runtime pages and synchronized both JSON
  exports; Wiki lint reported 0 errors and 8 non-blocking warnings.
- `git diff --check` passed after documentation consolidation.
