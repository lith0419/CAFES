# Error Book: change_nelec Requires Two-Item nelec List

## Error

```text
Study LLM draft failed: change_nelec requires a two-item nelec list
```

## Context

The user asked the planner:

```text
I want to check the behavior when increasing the total electron from 1 to 15
```

The planner generated an invalid model Hamiltonian operation that treated total
electron count as a scalar final `nelec`.

## Root Cause

PySCF expects spin-resolved electron count:

```json
"nelec": [nalpha, nbeta]
```

The LLM did not decompose total electron count into alpha and beta sectors.

## Permanent Rule

- For even total electron count `N`, use `[N/2, N/2]`.
- For odd total electron count `N`, use `[(N+1)/2, (N-1)/2]`.
- The planner should display this decomposition to the user.
- The backend should normalize recoverable scalar electron counts.

## Fix

The backend now supports normalizing total electron counts for `change_nelec`
and simple `nelec` sweeps.

## Regression Test

A test should verify:

```text
1 -> [1, 0]
2 -> [1, 1]
3 -> [2, 1]
4 -> [2, 2]
```

