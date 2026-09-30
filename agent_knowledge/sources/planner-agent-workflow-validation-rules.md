# Planner Workflow Validation Rules

## Scope

This document defines the validation and inheritance rules for study plans and
adaptive subsets.

## Required Checks

- Model Study `observables` accepts `mean_double_occupancy` as a derived metric
  request. Compile it to `strong_correlation_diagnostics` in executable task
  outputs and deduplicate the bundle. The comparison table and postprocessing
  retain the scalar `mean_double_occupancy` when computed; never fabricate a
  missing value. This Study shorthand does not make diagnostic internals into
  standalone backend outputs.
- Validate `StudySpec` before building a plan and validate the expanded
  `StudyPlan` before execution.
- Reject unsupported molecular methods, model solvers, observables, operations,
  and system types through the capability registry.
- Molecular DFT requires a non-empty `xc`; non-DFT methods must normalize empty
  `xc` to `null` rather than passing an empty string to validation.
- Model-Hamiltonian cases require a model spec/input artifact and an explicit
  registered solver under `study_mode=static`, with optional parameter-grid
  sampling refinement. The adaptive initial-scan and ActiveSpaceAudit workflow
  applies to molecules only. See [[Model Grid Convergence Barriers]].
- Normalize safe aliases and total electron-count inputs only when their
  spin-resolved meaning is unambiguous; otherwise return a user-facing issue.

## Scan Inheritance Rule

Both static and adaptive molecular studies retain their full case set. When a
static model scan varies selected parameters, every untouched field inherits from the
base input, including sites, bonds, `nelec`, boundary, energy unit, and global
metadata. The same rule applies when a molecular scan mutates only geometry or
a named request field.

## Model Parameter Grid Refinement

Static model studies can opt into `grid_refinement` while retaining explicit,
fixed solver settings within each parameter group. This controls sampling;
the molecular `study_mode=adaptive` method-routing workflow is separate.
Choose one or two continuous axes from a sweep (`U`, `V`, `t`, `epsilon`) or
named grid-template variables that only change those coefficient values.
Case-specific overrides and explicit-case lists cannot be interpolated.

Hamiltonian coefficients belong in `sweep` or explicit `operations`, not an
invented `request_updates.parameters` field. A U/V grid template uses
`set_global_parameter` operations with `parameter: U, value: "$U"` and
`parameter: V, value: "$V"`; site/bond operations express nonuniform edits.
The draft contract rejects misplaced parameters before grid validation and
uses the existing single structured-repair attempt. Unrepaired drafts remain
blocked. Explicit fixed solver options, including impurity SCF beta, stay in
`base_task.solver` and are inherited by every seed and refinement point.

The shared runner probes interval midpoints in one dimension and compares
measured quantities with interpolation using per-quantity absolute/relative
tolerances. New two-dimensional Studies use local rectangular cells: probe a
cell center against its four corners, then add missing edge midpoints and split
only cells with excessive residuals or widths. Adjacent cells share calculated
points; no complete rows or columns are added. Existing edge probes also enter
the residual checks. `max_new_points` counts distinct added calculations;
`max_rounds` limits 1D rounds or 2D probe levels (coarse cells are level 1).
Missing required quantities and incompatible local coverage are unresolved
evidence, not zero error. Correlation classifications do not trigger sampling.
Existing saved states without a local strategy retain legacy alternating-axis
refinement on resume. New-point IDs use a separate `grid-0001` sequence.

A frozen source specification, stable case identifiers and the persisted
`grid-refinement-state` artifact make inserts recoverable without resubmitting
completed tasks. Collect only reads existing execution; Run resumes sampling.
Changing a saved sampling policy or computational contract requires a new
Study. The report separates satisfied sampling criteria, exhausted budget,
round/spacing limits and insufficient evidence. These are local sampling
criteria, not calibrated solver-error bounds or phase-transition claims.

## Adaptive Subset Rule

- A recovery or approved-CAS subset must preserve the same Study identifier and
  the original case identifiers.
- It may mutate only the reviewed case's request, method/solver, runtime, or
  approved active-space contract.
- After the subset completes, merge final status, result fields, artifacts, and
  decision provenance into the parent report before presenting comparisons.
- Do not rerun completed initial-scan cases merely because a recovery subset was
  created.

## Registry Boundary

Registry entries control canonical ids, status, backend/planner permission, and
runtime limitations. UI adapters own labels and presentation. Generated
capability documentation must be regenerated from the registry, never edited by
hand.

## Failure Cases

- A `U`/`t` scan drops `sites` and `nelec`, producing blocked model cases.
- A molecular MP2 request carries `xc=""`, producing an avoidable DFT-field
  validation failure.
- A single recovery result overwrites the full adaptive table and makes trend
  analysis impossible.
- An adaptive initial-scan strategy must create one full-grid plan with one
  registered method. The resulting report preserves the original `case_id` and
  variables for routing. CCSD and other higher-level methods are selected later
  for individual refined or recovery cases; they are not hidden supplementary
  stages of the initial scan.
- Unknown or removed initial-scan strategy ids and legacy solver fields are
  validation errors. The planner must not silently reinterpret them as `auto`.
- Molecular initial-scan cases must be compiled by the shared active-space
  probe builder used by the Calculation Assistant. The probe contract must
  retain requested strategy, executed probe method, target CAS method/solver,
  solver options, selection method, and occupation window.
- Workflow-module identity, not a transient method label, determines whether a
  task is an active-space probe. Validation must apply probe requirements to the
  executed probe and defer target-only solver options, localization, and orbital
  ordering until the approved CAS calculation.
- Related molecular cases must pass the deterministic study active-space
  resolver before approval. The resolver selects one evidence-backed chemical
  target and CAS dimension, retains case-specific orbital mappings, and returns
  missing or conflicting cases to the probe workflow.
- An LLM draft that fails only the deterministic `StudySpec` contract may
  receive one structured repair request containing the validation issues and
  prior payload. The repaired payload must pass the same validator; free-form
  retries cannot bypass registry or approval rules.

## Related Pages

- [[Adaptive Scan Workflow State Machine]]
- [[Planner Agent Approval Boundary]]
- [[Task Specification Contract]]
