# Two Studies with a shared convergence barrier

The AFM and CDW honeycomb scans are separate saved Studies. Each starts with
U,V in {1,2,4,8}, uses 4 CPUs per task on Amarel `main`, and has an array limit
of 16. Their combined numerical concurrency is therefore at most 32.

`grid_refinement.require_converged` requires every current case to report
successful execution, affirmative convergence, and passing required quality
checks before refinement. Optional `convergence_retry` applies one DMET
fallback per case: mixing 0.1, 400 outer iterations, DIIS disabled. Initial
runs use mixing 0.2 and 200 iterations. Both use bath spin dimension `max`.
Retries retain the same seed; they transfer neither density nor correlation
potential. Density continuation remains a separate, explicitly planned action.

Fallback settings are persisted before submission. The sampling source and
canonical plan remain unchanged; actual execution requests and prior Runs are
retained in the Study checkpoint. Collection and interrupted receipt recovery
apply the persisted numerical settings. If the fallback still fails, the
Study stops at `awaiting_convergence` and no later points are generated.

The application `grid_group` coordinator invokes the ordinary saved Study
service. An absolute `grid_round_target` makes restarting a generation safe:
0 executes only the initial points, 1 permits one insertion batch, and so on.
The group grants generation N+1 only after every member has passed generation
N. A member that reaches its sampling or budget limit remains completed while
other members finish. Errors, cancellation, or an interrupted Study worker
hold the group for inspection. No numerical work runs in the group process.

The group and each Study coordinator run on the server, independently of the
SSH client. Each Study has its own Planner report, stable case identities,
execution receipts and history. The group JSON records passed barriers.

Tests cover failed fallback, required-quality failure, preservation of seed
and Run history, collecting and replaying a generation without resubmission,
and withholding the next generation while the other Study is still running.
