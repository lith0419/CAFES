# Builder Agent Interface

## Scope

This document defines how the Model Hamiltonian builder communicates with the
single-calculation agent and planner agent.

## Current Design

The builder can be opened from either the single-calculation page or the planner
page. It receives a target identifier and work directory. When the user saves
the input, the builder writes a local file and posts a message back to the target
page.

## Rules

- The builder must not rely on browser downloads for the primary workflow.
- The builder should save input files into the active work directory.
- The builder should include the target page in the postMessage payload.
- The target page should ignore messages intended for another target.
- Returning to the target page should not refresh and lose state.
- The builder should expose navigation back to both Agent and Planner pages.

## Expected Message

The builder should send a message with:

```json
{
  "type": "pyscf-agent:model-hamiltonian-input-saved",
  "target": "agent or study",
  "path": "/path/to/pyscf_model_hamiltonian_input.py",
  "work_dir": "/path/to/run/dir",
  "run_id": "..."
}
```

## Rationale

The builder is a separate editing surface. Explicit messages avoid hidden global
browser state and make the interface testable.

## Failure Cases

- Back button refreshes the agent page and loses state.
- Builder saves to a path the agent cannot identify.
- Planner receives a builder message intended for the single-calculation agent.

