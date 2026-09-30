# Agent Builder Planner Navigation

## Scope

This document defines navigation rules between the three local web pages.

## Pages

- Single Calculation Agent
- Model Hamiltonian Builder
- Planner Agent

## Rules

- The single agent should link to the planner agent.
- The single agent should open the model Hamiltonian builder when model input is
  needed.
- The planner should open the model Hamiltonian builder when a study starts from
  a lattice model.
- The builder should allow returning to both the single agent and the planner.
- Returning from the builder should preserve target-page state.
- Builder save events should include a target identifier.

## Rationale

The builder serves both the single-calculation agent and the planner. Navigation
must make this shared role clear without creating accidental page refreshes.

## Failure Case

The builder back button initially refreshed the web UI and lost state. The fix
is to reuse named windows and postMessage rather than relying on blind navigation.

