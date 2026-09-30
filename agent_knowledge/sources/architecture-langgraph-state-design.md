# LangGraph And Workflow State Design

## Single-Task State

The single-calculation workflow carries compact request, validation, execution,
result, artifact, and message fields between deterministic stages. Long PySCF
stdout, generated scripts, spectra, and structured results are persisted as
artifacts rather than embedded in LLM-visible state.

The final payload for one `TaskSpec` is a versioned
`pyscf-agent.task-report.v1` `TaskReport`. A study executor combines the
per-case `TaskReport` payloads into a `StudyReport`; it does not create a
second competing single-task result format.

## Planner State

The planner keeps a hidden `StudySpec`, a validated `StudyPlan`, a report, task
session state, and a separate adaptive workflow object. The adaptive object is
versioned as `pyscf-agent.adaptive-workflow.v1` and is the source of truth for
multi-round approval/recovery, not the conversation transcript.

## Rules

- Each state transition must consume and produce explicit fields.
- State schemas should be versioned when new task families or report formats are
  introduced.
- Store compact summaries and artifact paths in state; persist large evidence.
- Record initial-reference status separately from target-solver status so a
  probe or orbital guess cannot overwrite the final method outcome.
- Subset recovery must retain parent case identity and merge its result into the
  parent report before calculating the next state.
- Do not replay an active-space approval after the state shows that case as
  succeeded.

## Study Checkpoints

Each `Study` keeps a versioned `pyscf-agent.study-state.v1` checkpoint under
its work directory. The state stores each case's computational-contract
fingerprint, reusable `TaskReport`, status, and study-level attempt count. Checkpoint
writes are atomic. Fingerprint-matched successful cases resume without a new
PySCF run; blocked cases wait for an input change or an explicit rerun, and
failed/unconverged cases stop automatic retries at the bounded attempt count.

Adaptive studies resume by retaining the parent study ID, so their initial,
refined, and recovery studies reuse the same case checkpoints.

The checkpoint adds a current `execution` association: `run_id`, receipt path,
pending flag, and outer attempt number. Status and Collect never create an
attempt. Run can start a fresh attempt; resuming existing handles does not.
A submission-intent receipt is persisted before transport, and an unknown
acknowledgement blocks resubmission until reconciled. This bookkeeping does not
replace the existing lifecycle or create another retry decision engine.

Collect can leave unexecuted cases or a prepared adaptive stage for a separate
Run action. It does not invoke automatic recovery. Invalid state is an explicit
error, not an empty checkpoint. Compatible old receipt/checkpoint pairs can be
adopted without recalculation; new records mark their association format so
missing associations are not mistaken for legacy data.

## Open Work

- State-size budget and truncation warning.
- Hard per-case process timeouts and scheduler-aware cancellation.
- Bounded parallel execution with resource-aware scheduling.

## Related Pages

- [[Adaptive Scan Workflow State Machine]]
- [[Run Directory Artifact Storage]]
- [[Predefined Workflow Architecture]]
