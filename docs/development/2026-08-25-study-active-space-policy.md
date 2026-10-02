# Conditional Active-Space Probe And Study Policy

## Scope

This change strengthens the shared molecular active-space workflow without
adding a second executor or molecule-specific routing rules.

## Implemented

- Changed the registered `auto` probe from unconditional MP2 to an HF/SCF-first
  workflow. Complete and consistent chemical-valence or AVAS candidates proceed
  to review; ambiguous proposals invoke a registered MP2 refinement module.
- Preserved the requested CASCI/CASSCF method, FCI or block2 target solver, and
  target solver options in the versioned probe contract. Probe-only method,
  reference, localization, and ordering requirements do not leak into the
  formal calculation.
- Added a deterministic study active-space resolver. It selects one supported
  chemical target and CAS dimension for related cases while preserving each
  case's canonical-orbital indices and initial orbital matrix.
- Removed fixed frontier-space behavior from the study decision path. Missing
  or conflicting mappings return to the shared numerical probe instead of being
  copied from another case or replaced by an arbitrary CAS size.
- Added one bounded LLM contract-repair turn. The repair receives structured
  `StudySpec` validation issues and must pass the same deterministic validator;
  it cannot bypass Registry, resource, or approval gates.
- Kept active-space review plan-scoped and generated only after numerical probe
  reports exist.

## Design Audit

The implementation follows the current architecture:

- Registry entries define the probe strategy and conditional refinement.
- Workflow-module identity defines probe role; method names do not.
- The single-task kernel remains the only molecular numerical executor.
- The study layer consumes complete task reports and applies cross-case policy.
- `ActiveSpaceAudit` remains the approval artifact and execution boundary.
- Benchmark molecules appear only in regression tests, never in runtime rules.

## Model-study boundary

Finite model-Hamiltonian studies do not use this active-space policy. The
Planner fixes them to a static full-grid scan, requires an explicit solver, and
records strong-correlation diagnostics as case outputs. The `Auto`/`MP2`/`FCI`
initial-scan controls and `ActiveSpaceAudit` are molecular workflow concepts.

## Verification Target

The maintained planner regression includes a Cr2 scan from 1.2 to 3.2 Angstrom
in 0.2 Angstrom steps with def2-TZVP and a block2 DMRG-CASSCF target. Plan
construction must produce eleven HF/SCF probe cases, preserve the block2 target,
leave CAS dimensions unset before numerical evidence exists, and pass study
validation. A separate synthetic audit regression verifies a shared CAS(12,12)
chemical manifold with case-specific orbital mappings.

These are contract tests, not claims that the full Cr2 curve was recomputed or
scientifically benchmarked in this change.
