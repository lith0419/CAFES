# Error Book: Repeated Runs Produced Different Model Hamiltonian Energies

## Error

The same model Hamiltonian calculation produced different energies when
submitted twice. The second result was correct.

## Symptoms

The UI showed two successful FCI runs with different energies for the same
visible request.

## Likely Root Cause

Some default model or task field was applied differently between the first and
second submission. Hidden state, cached input files, or default work-directory
state may have diverged from visible UI state.

## Rules

- Visible UI state and executable task state must stay synchronized.
- Preparing a request should not mutate defaults in a way that changes the next
  run invisibly.
- Model Hamiltonian input files should be explicit and stable.
- Re-running the same task should use the same normalized spec unless the user
  changes a field.
- Regression tests should compare first-run and second-run behavior.

## Rationale

Scientific reproducibility requires identical inputs to produce identical
workflows and comparable outputs.

## Open Questions

- Whether every run should save the fully normalized model spec used for
  execution.
- Whether the UI should expose a normalized-spec preview for debugging.

