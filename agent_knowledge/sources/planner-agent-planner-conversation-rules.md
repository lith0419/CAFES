# Planner Conversation Rules

## Rule

The planner conversation exposes scientific decisions and next actions, not raw
planner JSON or hidden reasoning.

## Required Messages

- After a draft, summarize objective, system type, base task, changed variables,
  methods/solvers, observables, and any material limitations.
- For model systems without a declared solver, ask for one; recommend FCI for a
  small exact benchmark, CCSD/CCSD(T) for controlled comparison, and MP2 for a
  low-cost initial scan.
- For adaptive scan, explain that **every requested case** first receives an
  initial diagnostic calculation and then receives a method-specific refinement.
- State why a point is classified weak, moderate, or strong using interpretable
  evidence, not internal numeric scoring variables.
- On blocked/unconverged output, name the current case, current method, status,
  attempted recovery, and the next approval action. Present actions as buttons
  or an approval card rather than asking the user to synthesize raw JSON.
- After a subset recovery, state that its result was merged into the parent plan
  and either advance to the next case review or present full-study analysis.

## Terminology

Use **initial scan** for the first adaptive-calculation stage. Avoid calling it
`screening` in user-facing messages, where it can be confused with physical
screening. Do not present `screening_energy` as a physical conclusion.

## Must Not

- Do not show private chain-of-thought.
- Do not tell users merely to inspect a table when the system has a concrete
  approved recovery action to offer.
- Do not repeat an already-resolved active-space approval request.

## Good Progress Message

```text
Initial scan completed for all 15 geometries. Five stretched points show
near-degeneracy or single-reference stress, so CASSCF candidates are ready for
plan-scoped active-space review.
```
