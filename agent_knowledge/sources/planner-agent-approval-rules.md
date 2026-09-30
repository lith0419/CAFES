# Planner Agent Approval Rules

## Scope

This page defines the human decision boundary for ordinary, adaptive, and
recovery planner runs.

## Approval Gates

| Gate | User action | What becomes allowed |
| --- | --- | --- |
| Plan review | Build/rebuild plan | Inspect a deterministic case table; no calculation runs. |
| Run plan | Run static or initial scan | Execute the validated cases in that plan. |
| Active-space review | Approve or edit all `ActiveSpaceAudit` candidates in one plan | Build and run that approved CASCI/CASSCF subset. |
| Recovery review | Choose one suggested action for the current failed case | Build and run only the affected recovery subset. |
| Result analysis | Select `Analyze Results` | Compare completed cases and allow scan-path diagnostics to propose continuation or refinement reviews. |
| MPS continuation review | Approve the displayed compatible block2 checkpoint sources, or skip | Approval runs same-method retries for unresolved block2 cases and merges them into the parent report. |
| Continuation review | Approve the displayed left/right SCF-reference 1RDM propagation chains, or skip | Approval unlocks Run Plan; Run Plan retries the unchanged target methods across one local window. Approval is required for endpoint validation and cross-method/reference seeds. |
| Path-continuity review | Approve or edit the local ActiveSpaceAudit batch | Recompute both sides of an overlap region that remains non-smooth after bidirectional same-method continuation checks. |

## Rules

- Chat messages may update planning state but must not execute calculations.
- A plan must be validated and shown as a case table before run approval.
- Running an adaptive plan may use case-local diagnostics and recovery, but it
  must not inspect path continuity or transfer an AO 1RDM between cases.
  Cross-case operations begin only after an explicit result-analysis request.
- A task-changing edit makes the relevant plan stale and requires validation and
  approval again.
- Active-space approval is a scientific approval, not a generic acknowledgement.
  Show `ncas`, `nelecas`, orbital identifiers, selection rationale, and audit
  evidence before the user confirms it. If the candidate crosses a cost-review
  threshold, show its deterministic estimate in this same approval surface.
- When adaptive recovery has several unresolved cases, present one failed case
  at a time. Different geometries or parameters can need different recovery
  choices.
- Once a recovery/refined/path plan has produced its ActiveSpaceAudit
  candidates, present every candidate in that plan together. The user can
  compare CAS sizes, electron counts, orbitals, and cost before one approval;
  candidates from other plan kinds remain separate.
- A path-continuity flag is not a license to overwrite the curve. First retry
  compatible lower-level points with a tracked neighboring AO 1RDM; only a
  remaining anomaly opens a plan-scoped local ActiveSpaceAudit batch for the
  affected overlap region.
- Endpoint validation and cross-method/reference 1RDM reuse are scientific
  review actions. Display the complete window, unchanged target method, branch
  order, concrete seed artifacts, deferred adjacent-task sources, source
  methods, and mean-field references. Approval applies only to that
  fingerprinted continuation plan. Approval prepares the plan but does not
  execute it; the user launches the approved retry with Run Plan. Skipping
  proceeds to method review without running a continuation calculation.
- An MPS continuation proposal is also a scientific review action. Show each
  target case, source case, checkpoint or manifest, normalized parameter
  distance, executor identity, orbital ordering, and quantum-sector
  compatibility. Approval is fingerprinted to those sources. Approved MPS
  retries run sequentially and merge automatically because they resolve failed
  results rather than replacing already successful scan points. Skipping leaves
  ordinary path and method review available.
- An approved subset may run without rerunning the initial scan, but its results
  must merge into the parent full report afterward.

## Must Not

- Do not run a calculation from a conversation send action.
- Do not treat an acknowledgement as approval for a method, active space, or
  expensive recovery action.
- Do not mix active-space candidates from unrelated executable plans.
- Do not replace the complete adaptive report with a single recovery result.
- Do not transfer an MPS across execution targets or incompatible orbital,
  electron, spin, symmetry, root, or ordering sectors.

## Failure Pattern

Treating an active-space approval as an arbitrary batch can mix different
recovery plans. Batch only the candidates already assembled into the same
validated plan, display every case identifier and audit, and keep failed-case
method recovery itself case-local.

## Related Pages

- [[Adaptive Scan Workflow State Machine]]
- [[Planner Workflow Validation]]
- [[User-facing Planning Summaries]]
