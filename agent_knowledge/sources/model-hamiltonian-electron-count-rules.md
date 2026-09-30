# Model Hamiltonian Electron Count Rules

## Scope

This document defines how model Hamiltonian electron counts should be represented
and normalized.

## Current Design

PySCF model Hamiltonian calculations expect spin-resolved electron counts:

```json
"nelec": [nalpha, nbeta]
```

Users often describe a total electron count. The planner and backend must
normalize total electron counts before execution.

## Rules

- Never pass a scalar total electron count directly as final `nelec`.
- If the user gives total electron count `N`, convert it to spin-resolved form.
- For even `N`, use `[N/2, N/2]`.
- For odd `N`, use `[(N+1)/2, (N-1)/2]`.
- The planner may accept language such as "increase total electron from 1 to 15".
- The planner should show the decomposition rule to the user.
- Validation should reject or normalize invalid electron-count operations before
  running PySCF.

## Examples

```text
N = 1  ->  [1, 0]
N = 2  ->  [1, 1]
N = 3  ->  [2, 1]
N = 4  ->  [2, 2]
N = 15 ->  [8, 7]
```

## Valid Operations

These are acceptable after normalization:

```json
{"op": "change_nelec", "nelec": [1, 0]}
```

```json
{"op": "change_nelec", "total_electrons": 3}
```

The second form must be normalized before planning or during operation
application.

## Failure Case

The following error happened during planner development:

```text
change_nelec requires a two-item nelec list
```

This happened because the LLM generated a scalar electron count where the
backend expected `[nalpha, nbeta]`.

## Rationale

PySCF solvers need spin-resolved electron counts. Explicit normalization keeps
natural language convenient while maintaining a valid execution contract.

