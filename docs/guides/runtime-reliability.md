# Runtime reliability boundaries

## JSON and saved fingerprints

`pyscf_agent.serialization.json_default` accepts NumPy booleans, integers,
floating scalars and arrays as native JSON values. `Path` and Python dates and
datetimes retain their historical string representations. Complex numbers,
bytes, sets and unknown objects raise `TypeError`; NaN and infinity are rejected
by general JSON writers, input contracts and fingerprints. Scientific code must
explicitly represent complex components or zero-temperature limits in its contract.

Result JSON artifacts and the final TaskReport have an output-only recovery
boundary: NaN and positive/negative infinity become `null`. TaskReport `warnings`
records `code=nonfinite_result`, the original value label, and an RFC 6901 JSON
Pointer in `path`. Artifact warnings also carry `artifact_path`; their pointer is
relative to that artifact, while other pointers are relative to the TaskReport.
Existing warnings and finite results are preserved. This applies before early
result artifact writes as well as final reporting, so a missing DMRG diagnostic
does not prevent the completed energy and report from being saved. Unknown types
still fail explicitly. Numerical in-memory results and binary NPY/NPZ arrays
(including intentional NaN padding) are unchanged; convergence decisions are not
rewritten by serialization. Logs use the same cleanup on their output copy.

Artifact writes remain atomic by default. Failed encoding leaves an existing
file intact and removes the partial temporary file. This does not repair already
stringified historical data or alter the explicit `atomic=False` behavior.

Study, receipt and retry fingerprints share canonical JSON. Grid fingerprints
retain their existing ASCII escaping and integer/float normalization. Existing
JSON-compatible inputs therefore retain their hashes; previously unsupported
NumPy values now hash like their native JSON equivalents. Other historical
fingerprint formats are not silently migrated.

## Remote commands and HTTP errors

Scheduler and SSH control commands have a 60-second limit. Deployment has a
300-second limit; dataset generation and SFTP downloads have a separate
1,800-second limit. These are subprocess limits, not scientific Slurm job wall
times. SSH/SFTP send keepalives every 15 seconds with a maximum of three missed
replies. The bounds are centralized in `pyscf_agent.executors.commands`.

A timeout means the remote outcome is unknown. Receipt inspection marks the
status temporarily unreadable and preserves handles for later collection; it
does not declare the job failed or resubmit it. In particular, a submission
timeout requires inspection/recovery before another submission. Injectable
command runners must implement their own execution deadline; `TimeoutExpired`
from such runners is translated at the adapter boundary.

Both Web APIs log unexpected failures with their traceback and return a generic
500 response. Recognized invalid requests return 400, Study ownership conflicts
409, missing resources 404, and timeouts 503. Explicit validation branches keep
their existing structured responses. Python `ValueError` is still used by many
legacy validators; a future typed validation migration would further distinguish
internal `ValueError` bugs from invalid input.

Scientific FCI configuration no longer ignores a rejected attribute assignment.
It fails explicitly with the field and requested value. Narrow optional-data
fallbacks remain, as do broad catches at worker/provider boundaries that record
failures or preserve recoverable state.

## Study ownership

Foreground execution, adaptive execution and background workers share
`.study-invocation.lock` using `fcntl.flock`. Ownership is reentrant within the
same thread, exclusive across threads and processes, and released when the last
owning descriptor closes. Background launch transfers the descriptor to its
worker, which adopts it before entering execution.

Duplicate Run requests fail immediately. Collect waits up to 30 seconds to
serialize overlapping collection requests. Status inspection briefly takes a
shared lock; writers distinguish these readers from a running owner and wait
for readers without reporting a false running conflict. A continuously held
reader is bounded by a five-second timeout.

These are advisory locks: every writer must participate. They coordinate Web
and MCP processes sharing a filesystem with working `flock` semantics, not a
distributed lock service. Do not remove or replace an active lock file.

## Provider compatibility

`PYSCF_AGENT_LLM_STRUCTURED_OUTPUT` accepts `auto` (default), `true`, or `false`.
Set `false` for a known incompatible provider to avoid a speculative structured
request; `true` requires structured output and disables fallback. Auto mode
caches support per base URL/model and retries only for a recognized unsupported
feature response (HTTP 400/422). Structured error codes take precedence; a narrow
message fallback remains for providers that omit codes. Gateways returning the
generic `invalid_request_error` with `param=null` also permit fallback for the
specific `response_format type is unavailable` message. Schema-validation errors
and unrelated availability errors still propagate. The Study Planner honors the
same explicit policy and capability cache as the calculation request builder.
There is no universal
provider capability-discovery endpoint. Network errors, internal errors and
ordinary schema validation failures do not trigger capability fallback.

CIF partial occupancy validation no longer matches its own exception message.
The libDMET adapter retains its old SciPy flat-regression translation and also
rejects non-finite chemical-potential shifts before updating the DMET iteration.
This matters because SciPy 1.18.1 returns NaN for constant regression inputs
instead of the historical exception. A regression test calls the installed
SciPy implementation and verifies unrelated errors still propagate. Dependency
bounds are SciPy `>=1.11,<1.19`, ASE `>=3.22,<3.30` and the existing PySCF
`>=2.13,<2.14`; this revision was checked with SciPy 1.18.1, ASE 3.29.0 and
PySCF 2.13.1. Lower supported versions are not a claim of a complete version
matrix test. The optional source-installed libDMET dependency is not silently
replaced by an unrelated package pin.

## Runtime dependencies and remaining refactors

Consumers obtain the cached artifact repository at call time. The repository
resolves its default registry lazily unless an explicit registry was injected.
`default_registry()` is cached too; runtime consumers no longer capture a
registry instance in a module-global handle. Static compatibility catalog
projections still exist. Tests replacing factories must clear the relevant
getter cache before and after replacement.

Shared helpers cover plan case IDs, diagnostic thresholds, executor paths and
collectable status, benchmark fixtures, and DMRG benchmark execution. The two
old DMRG functions were nested in separate benchmarks, not overwriting module
definitions. Molecular preview uses the shared browser HTML encoder.

Domain-specific JSON wrappers remain where they supply artifact kind,
description or return shape. Two whole-decision copies were removed from
adaptive refinement after checking that those decisions are read-only. Copies
of outgoing requests, mutable tags/reasons and scientific arrays remain. A
wholesale immutable report/plan/state conversion and broader copy elimination
need profiling and explicit ownership contracts; they are separate migrations.

## Verification

On the development Python 3.12 environment, the complete unittest suite ran
1,399 tests successfully, with 67 existing optional-provider/environment skips.
Ruff passes; mypy passes for the five modules declared in `pyproject.toml`.
Three additional tests with the optional source-installed libDMET pass: shared
beta propagation through both DMET execution modes, structured FCI DMET output,
and CCSD smearing/fragment diagnostics. This is not a full optional-provider
matrix. No cluster job was submitted to test timeout handling; stalled commands
and unreadable receipts were injected, while file-lock exclusion and worker
termination were tested with real local processes.
