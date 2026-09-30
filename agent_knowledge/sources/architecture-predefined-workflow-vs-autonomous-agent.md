# Predefined Workflow Versus Fully Autonomous Agent

## Scope

This document explains why CAFES uses predefined workflows
instead of allowing the LLM to autonomously select arbitrary tools, memory, and
execution steps.

## Current Design

The agent uses predefined workflow stages for request preparation, validation,
execution, artifact creation, and result analysis.

The LLM helps interpret user intent and draft structured plans, but the workflow
controls what can actually run.

## Rules

- The LLM may propose a task or plan.
- The backend must validate the task or plan before execution.
- Tool selection should be constrained by capability registries.
- Memory or prior context should not be trusted as executable truth without
  validation.
- Expensive or risky steps require explicit user approval.

## Rationale

PySCF is flexible and powerful, which also means invalid combinations are easy
to generate:

- unsupported methods
- wrong spin or electron-count format
- missing basis or functional
- unsupported model Hamiltonian operation
- expensive full-CI calculations on too-large systems

A fully autonomous agent may hallucinate PySCF APIs or silently choose an
inappropriate method. A predefined workflow reduces this risk by giving the LLM
a narrow, reviewable interface.

## Advantages For PySCF

- Reproducibility: every run goes through the same state and artifact path.
- Auditability: task specs and generated input files are saved.
- Safety: execution is separated from planning.
- Debuggability: validation and retry points are explicit.
- Extensibility: new solvers or operations can be added to capability registries.

## Example

The user asks:

> Compare total electron counts from 1 to 15 on this Hubbard lattice.

The LLM may identify this as an electron-count scan, but the workflow must
convert each total count to spin-resolved `nelec=[nalpha, nbeta]`, validate the
case list, show a plan table, and wait for `Run Plan`.

## Failure Mode

A fully autonomous agent might generate:

```json
{"op": "change_nelec", "nelec": 7}
```

This is not directly valid for PySCF. The workflow must normalize or reject it.

