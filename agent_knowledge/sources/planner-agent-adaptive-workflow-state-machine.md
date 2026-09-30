# Adaptive Scan Workflow State Machine

## Scope

The adaptive planner uses the versioned state schema
`pyscf-agent.adaptive-workflow.v1` to make multi-round recovery deterministic
and user-reviewable.

This initial-scan/refinement workflow applies to molecular studies. Model-
Hamiltonian studies retain an explicit solver under `study_mode=static`;
optional `grid_refinement` adapts parameter sampling with separate convergence
barriers. Result diagnostics do not create a molecular probe or active-space stage.
See [[Model Grid Convergence Barriers]].

## States And Transitions

```text
initial_plan_ready
  |-> cost_review_required -> initial_plan_ready
  |-> run_initial_scan
       |-> initial_scan_blocked
       |-> cost_review_required
       |-> active_space_review_required
       |-> refined/recovery execution
             |-> recovery_review_required
             |-> active_space_review_required
                    |-> completed | completed_with_issues
                    |-> Analyze Results
                          |-> DMRG root-state tracking
                          |-> mps_continuation_review_required
                          |-> approved same-method MPS retry
                          |-> continuation_review_required
                          |-> bidirectional continuation-window retry
                          |-> path_refinement_review_required
                          |-> analysis completed
```

`initial_scan_blocked` is entered when a planned case cannot produce valid
diagnostics. `active_space_review_required` is entered when a refined or
recovery CASCI/CASSCF request has an enabled but unapproved active space.
`recovery_review_required` is entered when a refined/recovery result is not
`succeeded` and an action must be chosen.
`path_refinement_review_required` is entered after a completed one-dimensional
scan is explicitly analyzed and a non-smooth method transition cannot be resolved by a
bidirectional same-method continuation-window retry; it presents the local
CASSCF ActiveSpaceAudit candidates for both sides of the overlap region.
The continuation retry preserves each target method and changes only its SCF
initial density. It uses two registered AO 1RDM anchors outside the
suspect window. Same-method/reference anchors may run automatically; a
cross-method/reference anchor enters `continuation_review_required` first. The
parent result changes only if both branch results pass the registered
consistency checks.
Normal adaptive execution does not enter either path state. It stores
independent case results and a deferred path-analysis marker until the user
selects `Analyze Results`.
For block2 cases, result analysis first attaches
`pyscf-agent.dmrg-state-tracking.v1`, then checks MPS continuation before
scan-path continuation. A compatible state-trusted source creates
`mps_continuation_review_required`; approval runs the retry sequentially on the
same executor and merges it into the parent report. Skipping leaves the later
path and method-review states available. Initial cases never share MPS state.
An ambiguous state assignment excludes that case as a continuation anchor.
`cost_review_required` is entered before an initial, refined, or recovery plan
crosses the registered determinant/work/memory review policy. For an
active-space candidate, the estimate is shown in the same scientific approval
surface; cost approval does not replace active-space approval.

## Required State Fields

- `schema`: workflow schema version.
- `stage`: current state from the state list above.
- `active_plan_kind`: `initial`, `refined`, `recovery`, `path_restart`,
  `path_refinement`, `mps_continuation`, or the direct CASSCF review overlay used
  by the Planner UI.
- `scope`: `full` or `subset`.
- `pending_case_ids`, `completed_case_ids`, and `failed_case_ids`.
- `allowed_actions`: stable action ids, labels, and the case ids they affect.
- For unresolved-case queues: `review_mode=case_queue`, `current_case_id`,
  `queue_position`, and `queue_total`.
- For an ActiveSpaceAudit batch: `review_mode=batch`, all candidate case ids in
  `pending_case_ids`, and actions scoped to that one executable plan.
- Result analysis may attach `dmrg_state_tracking` with ordered root links,
  confidence, reordered roots, and ambiguous case ids. It is evidence for later
  actions, not an execution state by itself.

## Case-Queue Rule

For unresolved or blocked cases, actions apply only to `current_case_id`.
Active-space candidates are different: candidates from one refined, recovery,
or path-refinement plan are approved together, while candidates from a
different plan kind are never mixed into that batch. After a successful subset
calculation, rebuild workflow state from the merged parent report; it should
advance to the next unresolved case or to completion. Never replay an old
approval message after the case is resolved.

## Permitted Actions

- `run_initial_scan`
- `approve_active_space`
- `approve_path_restart`
- `skip_path_restart`
- `approve_mps_continuation`
- `skip_mps_continuation`
- `expand_active_space`
- `promote_casscf_recovery`
- `increase_recovery_max_cycle`
- `try_fci_recovery` only when conservative molecular FCI limits permit it
- `show_case_guidance`
- `suggest_plots`
- `acknowledge`

`approve_cost_estimate` is available only for a deterministic estimate bound to
the current plan fingerprint. Approval cannot be inferred or set by the LLM.
`approve_path_restart` is likewise bound to the displayed local window and
anchor fingerprint.
`approve_mps_continuation` is bound to the displayed source checkpoint,
compatibility checks, executor identity, and continuation-plan fingerprint.

Actions that change method, active space, or runtime construct a validated
subset plan. They do not mutate already succeeded cases.

## Failure Cases

- Repeatedly showing the same completed CASSCF approval because chat history,
  rather than merged workflow state, was used as the source of truth.
- Combining candidates from different recovery/path/refinement plans into one
  executable approval subset.
- Returning a one-case recovery report as the study result, losing full-scan
  comparison and postprocessing context.
- Promoting a path anomaly before attempting a compatible two-anchor restart,
  or treating the restart densities as untracked hidden inputs.
- Letting one branch overwrite a completed point before its opposite-anchor
  result has been compared.

## Related Pages

- [[Planner Agent Approval Boundary]]
- [[Planner Workflow Validation]]
- [[Workflow State Design Principles]]
