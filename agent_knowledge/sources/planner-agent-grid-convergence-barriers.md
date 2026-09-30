# Model Grid Convergence Barriers

## Convergence Barrier Contract

- With `grid_refinement.require_converged`, every case must succeed, explicitly
  converge and pass required quality checks before more sampling. Optional
  `convergence_retry` allows one fallback per case; a failed fallback leaves
  `awaiting_convergence`. A shared `grid_round_target` advances only when all
  grouped Studies pass the current generation. These barriers do not change
  the solver or transfer a donor density.

## Sampling And Solver Choice

- Model studies use explicit solver settings under `study_mode=static`; optional
  `grid_refinement` adapts parameter sampling. This does not enable molecular
  initial-scan method routing or active-space selection.
- `grid_refinement.require_converged` requires every current case to have
  succeeded, explicitly converged, and passed required provider quality checks
  before generating further sampling points. Missing quality evidence blocks
  the barrier. Optional density-fit diagnostics do not independently block it.

## Bounded Convergence Retry

- Optional `convergence_retry` applies one numerical fallback per case, using
  the validated mixing, iteration limit and outer-DIIS settings. The recorded
  honeycomb policy uses mixing 0.1, 400 iterations and DIIS off; those study
  settings are not universal provider defaults.
- Persist fallback overrides before submission and use them during collection
  and interrupted-receipt recovery. Preserve the canonical sampling plan,
  analytic seed, case identity, actual execution requests and previous Runs.
- This fallback transfers neither donor density nor correlation potential.
  Density continuation is a separate planned action. An unsuccessful fallback
  leaves `awaiting_convergence`; do not generate later points or retry forever.

## Shared Generation Barrier

- The application `grid_group` coordinator uses normal saved Study services.
  An absolute `grid_round_target=0` permits only seed points; each additional
  generation permits one insertion batch. Restarting a generation must not
  resubmit completed tasks.
- Grant generation N+1 only when every member has passed generation N. A member
  that reaches its sampling/budget limit remains completed while others finish.
  Errors, interrupted workers and cancellation hold the group for inspection.
- Each Study retains its own coordinator, reports, receipts and candidate
  history; the group records barriers and runs no numerical calculations.
  Different studies may choose different refined coordinates after a barrier.
- The grid group is an application coordinator, not an independently exposed
  MCP numerical solver. Preserve the execution target and deployed runtime.

## Related Pages

- [[Planner Workflow Validation]]
- [[Adaptive Scan Workflow State Machine]]
- [[DMET Density Continuation And Branch Analysis]]
