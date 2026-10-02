# Scientific Input And Recovery Boundaries

Implementation date: 2026-09-17.

## Design

Missing parameters may receive documented defaults. Explicit invalid values
must remain errors: normalization cannot turn a typo, fractional integer,
non-finite number, or malformed object into a different calculation.
`input_validation.py` supplies lossless scalar conversion and dataclass-owned
field checks. The existing task parser owns shorthand aliases. Execution
requests reject unknown fields; report readers retain additive-field support.
No new orchestration or validation framework was introduced.

Provider validation runs before submission. Block2 validates its typed options
and unknown keys for molecular and model tasks, including nested DMET impurity
options. Its presets and automatic recovery limits are read from the existing
capability registry. String boolean aliases are normalized explicitly; zero
sweeps and non-finite thresholds are rejected.

The LLM interprets free text. The Web adapter accepts its complete StudySpec
and does not override methods, promote an impurity solver, reconstruct a scan
from its seed, infer an operation's parameter identity, or change adaptive
mode to static. Invalid drafts remain reviewable errors. Geometry aliases and
explicit numeric ranges use the shared StudySpec normalization boundary;
grouped site selectors use the existing native operation contract.

## Evidence And Recovery

- Missing initial-scan diagnostics block method routing; they are not weak
  correlation evidence. An empty decision set cannot generate a refined plan.
- Legacy results remain displayable. Their `publication_eligible` is null
  when quality evidence is absent; explicitly failed results remain false.
  Existing plots can display legacy rows without claiming verified quality.
- Optional binary artifact failures preserve completed energies and record
  `artifact_errors` plus a warning log with the missing kind and reason.
- Retries use structured convergence state, preserve the reference choice,
  and record the previous and retry TaskSpec. DMET outer-loop failures no
  longer trigger an unrelated generic SCF-cycle retry. Existing explicit
  Study recovery actions remain available.
- GW Cholesky failure propagates with its stage and provider log. The runtime
  does not silently relax the factorization threshold; successful summaries
  include the applied tolerance.
- Orbital projection uses the required SciPy assignment implementation;
  implementation errors do not silently select a different greedy mapping.
- The browser displays backend-provided review actions. Missing actions in
  an old report do not cause it to invent executable recovery choices.

## Compatibility

Existing QH9 dataset planning now explicitly sets
`molecular_dynamics.profile="qh9_relaxed_scf"`. Standalone MD requests and old
saved MD requests must include that profile before a new execution. Reading
old reports does not require migration. The available MD implementation is
still the restricted QH9-compatible trajectory slice; this change does not
claim general MD method support.

Clients submitting misspelled fields or malformed numeric values must correct
them. Clients reading legacy results must allow null publication eligibility.
Consumers requiring scientific quality evidence should distinguish null from
true rather than treating execution success as certification.

LLM fixture tests now provide structured scientific choices explicitly.
Regression tests separately verify that negated method names, impurity solver
names, and mixed local/global revisions cannot be overwritten by transport
heuristics.

## Verification

The new boundary regression suite covers invalid input, provider options,
explicit MD profile selection, solver ownership, full scan revisions, unknown
diagnostics, bounded same-reference retry, and optional artifact write failure.
Full-suite results and environment details are recorded in
`reports/verification/input-recovery-boundaries-2026-09-17.json`.

The final suite ran 1,105 tests: 1,048 passed and 57 were skipped, with no
failures or errors. Python 3.9 compatibility passed 70 focused tests. The two
edited review scripts passed JavaScript syntax checks. Wiki curation exported
43 pages; lint reported zero errors and three citation warnings in unchanged
roadmap/benchmark pages. Block2 and libDMET were unavailable locally, and
opt-in numerical MCP acceptance was not enabled; this verification does not
replace remote numerical acceptance.

No remote jobs were submitted and no production service was restarted. Python
source changes require restarting the existing Web/MCP processes in their
configured environment. Changes are local and have not been pushed.
