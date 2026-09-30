# Error Book: Builder Return Refreshed Target Page

## Error

Returning from the Model Hamiltonian builder refreshed the target web UI and
lost state.

## Context

The builder can be opened from both the single-calculation agent and planner.
After saving an input file, the user wanted to return without losing the
prepared conversation, work directory, or hidden planning state.

## Root Cause

Navigation used a page reload instead of reusing a named window and preserving
state through postMessage.

## Rules

- Use named windows for the agent, planner, and builder pages.
- The builder should post structured save messages to the target window.
- The target page should update state from the message without refreshing.
- A return button should focus or navigate to the existing target window when
  possible.

## Rationale

The builder is part of a multi-page workflow. State loss breaks the connection
between model construction and calculation planning.

