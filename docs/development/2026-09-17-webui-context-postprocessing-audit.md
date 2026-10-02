# WebUI Context And Postprocessing Audit

Audit date: 2026-09-17. This follows the input-boundary and remaining-fallback
cleanup. Runtime implementation was not changed during this audit. The new
audit utility and evidence describe the current uncommitted working tree.

Implementation follow-up: [postprocessing contract cleanup](2026-09-17-postprocessing-contract-cleanup.md).
The findings and token counts below describe the pre-fix snapshot.

## Scope And Evidence

The inventory covers 220 Python files and 18 browser JavaScript files under
`pyscf_agent/` and `computational_study_agent/`. It finds 460 exception handlers,
including 100 broad handlers. These are search counts, not defect counts.
Call-site review focused on planning, context composition, result analysis,
scientific postprocessing and optional-evidence failures.

The earlier failing 16-case request and provider response were not retained.
Its documented scientific context is a 32-site Builder model and a 4-by-4 DMET
U/V scan. This audit reconstructs that class of request using an explicitly
synthetic 4-by-8 open square graph: 32 sites, 52 bonds, 32 electrons, FCI as
the DMET impurity solver, U=[1,2,3,4] and V=[0.1,0.2,0.3,0.4]. The goal is:

> Keep all 16 cases and change the DMET execution_mode to finite_graph.

The fixture already has `finite_graph`, as in the existing repair regressions.
This measures context retention and contract repair, not the LLM's ability to
choose or change the execution mode. It is not a reconstruction of the exact
historical lattice, request text, output or bill.

`tools/audit_webui_context.py` executes the real `draftWithLlm`,
`plannerRequestContext`, preparation helpers and `buildPlan` in a Node VM with
a minimal DOM fixture. It captures both WebUI request bodies, passes them to
the real Python handlers and planning/validation services, and replaces only
the model transport with deterministic replies. Plan persistence is exercised
in a temporary directory. No model inference or remote numerical job is run.

Evidence lives in `reports/verification/webui-context-audit-2026-09-17/`:

- `audit.json`: input, counts, replay outcomes, counterexamples and source hashes.
- `planner-request.json`: the exact model-bound initial Planner request.
- `analysis-request.json`: the exact model-bound synthetic 16-case analysis request.

No API keys or headers are included. The tokenizer was installed in a temporary
directory, without adding a project dependency.

## Submission Flow And Token Counts

The static Study route is:

1. WebUI reads the current structured StudySpec and the latest message.
2. `/api/study-llm-draft` builds a scientific Wiki query and retrieves evidence.
3. The model receives the goal, full seed, Builder site context, Wiki excerpts
   and StudySpec schema. Old assistant chat messages are not appended.
4. The generated complete StudySpec is validated and expanded into 16 cases.
5. Build Plan calls `/api/study-prepare` and persists that plan without an LLM.
6. Static execution/collection is handled by the executor. Language analysis is
   a separate requested operation; plotting and dataset actions are deterministic.

The following are exact local tokenizations of **message contents**, not
provider usage. Provider-specific chat framing, schema handling, reasoning,
cache accounting and actual generated output are not measured.

| Route | Transport attempts | o200k_base input | cl100k_base input | Outcome |
| --- | ---: | ---: | ---: | --- |
| Initial valid draft | 1 | 8,799 | 8,805 | HTTP 200, 16 cases, no validation issues |
| Short invalid reply, then contract repair | 2 | 17,683 | 17,695 | HTTP 200, 16 cases |
| Unsupported response format, plain reply invalid, then repair | 3 | 26,482 | 26,500 | HTTP 200, 16 cases |
| Both replies invalid | 2 | 17,683 | 17,695 | HTTP 400, no replacement spec or plan |
| Build Plan / save | 0 | 0 | 0 | HTTP 201, 16 persisted cases |
| Plot suggestion / rendering / dataset generation and collection | 0 | 0 | 0 | Deterministic source paths |
| Analyze synthetic 16-case results | 1 | 13,735 | 13,746 | HTTP 200 via real analysis handler |

The rejected-format attempt is counted as transmitted input, not necessarily
billed model input. Repair counts depend on the invalid reply: a large invalid
StudySpec will be repeated and cost more than the short fixture reply.

The initial `response_format` object separately occupies 207/206 tokens when
serialized. Adding that representation gives 9,006/9,011 initial input tokens,
before provider framing. It must not be confused with the 9,556/9,600 tokens
obtained by tokenizing the entire HTTP JSON envelope, which includes escaped
strings and transport keys the model does not receive as ordinary text.

Initial Planner components, measured separately with o200k_base:

| Component | Tokens |
| --- | ---: |
| System instructions | 1,130 |
| Current goal | 16 |
| Complete seed StudySpec | 3,653 |
| Builder site context | 616 |
| Retrieved Wiki evidence | 2,798 |
| Schema inside the user message | 175 |

Independent component counts are not additive because JSON escaping, field
names, delimiters and token boundaries change the full-message count. The full
message contents total 28,035 characters. The measured fixture is different
from earlier synthetic character-count audits; do not interpret their sizes
as a before/after regression on an identical request.

Returning the unchanged complete seed in the fixture is another 3,653/3,655
output tokens when serialized. This illustrates output size, not actual model
generation. A normal draft plus one 16-case analysis therefore sends about
22.5k message-content input tokens (about 22.7k including the separately
serialized Planner schema), plus the actual outputs and provider overhead.

The current Planner failure diagnostic retains provider `usage` only for the
two-invalid-reply failure path. Successful drafts, successful repairs and
result analysis do not preserve aggregate usage. Consequently the application
cannot yet provide a reliable per-Study historical token ledger.

## Confirmed Findings

### B01 — Result analysis still injects every output contract

Priority: P1 for context composition. Locations:
`pyscf_agent/request_builder/llm.py:317`, `:726`;
`pyscf_agent/registry/platform.py:985`, `:1062`.

This is not the entire capability registry previously removed from Planner.
It is the complete **75-field output-contract catalog**, regardless of which
solver, observables or results occur in the report. The same catalog is sent
as generated system prose and structured user data:

- System contract prose: 4,016 o200k tokens.
- User contract catalog: 8,162 tokens.
- Combined: 12,178 tokens, approximately 88.7% of the 16-case analysis input.
- Actual prepared Study analysis data in that request: 1,097 tokens.

The fixed catalog dominates even a tiny report. A real saved two-case report
(`runs/20260916-144523-9cf9e952/study-report.json`) was replayed through the same
handler without modifying the saved run. Its roughly 1.05-million-character
report is correctly projected, but the model still receives 13,120 tokens,
of which only 521 are the prepared Study analysis data.

The fix belongs in the shared analysis-context builder: select contracts from
the executed method and actual result fields, render each once, and retrieve
Wiki explanations only when relevant. Keep runtime contract validation intact.

### B02 — Study analysis has projection but no explicit scale budget

Priority: P2. Location:
`computational_study_agent/application/service.py:1136`, `:1992`.

The service does not send every raw TaskReport. It selects the comparison
table, path diagnostics and state tracking. That is a useful boundary.
However, it sends all rows and all selected diagnostics without an input
budget; multi-root tracking can also add substantial matrices and mappings.

| Synthetic DMET rows | o200k_base analysis input |
| ---: | ---: |
| 1 | 12,879 |
| 16 | 13,735 |
| 64 | 16,471 |
| 256 | 27,415 |

These fixtures contain only small scalar rows. A general solution should
retain full data in artifacts, send global statistics and relevant anomalies,
and allow scoped follow-up analysis. Do not replace scientific evidence with
arbitrary character truncation. Small tables can remain complete.

Planner also repeats sites in the complete seed and `model_site_context`, and
requires a complete output. This is a smaller issue here (616 input tokens),
not a reason to return partial StudySpecs again. Use existing model-file
references and a task-appropriate structural view when improving this boundary.

### B03 — Plot units are inferred rather than validated

Priority: P1. Locations:
`computational_study_agent/postprocessing.py:202`, `:214`, `:333`, `:420`.

A column containing `-1 Ha` and `-27.211386 eV` is accepted as numeric values
-1 and -27.211386. The y-axis is labelled Ha, taken from the first row; no unit
conversion or consistency check occurs. Plain molecular `bond` values are
labelled Angstrom even when the attached case request specifies Bohr.

This can misrepresent scientific data without a runtime failure. Units should
come from canonical quantity metadata. Mixed units require an explicit
conversion or rejection; an unknown unit should remain unknown.

### B04 — Heatmaps silently overwrite duplicate coordinates

Priority: P1. Location:
`computational_study_agent/postprocessing.py:660`.

The actual rendering function was intercepted at `Axes.imshow`. Given two rows
at (0,0), with values 0 and 999, its rendered grid contains 999 with no error.
This can occur when U and V are plotted while solver, root or another sweep
dimension is left unspecified. Row order then determines the figure.

The plot contract should require unique coordinates within each selected
group, or an explicitly requested aggregation/facet. Do not silently choose a
row or introduce an implicit averaging policy.

### B05 — WebUI and backend disagree about eligible rows

Priority: P2; P1 impact for affected legacy/external reports. Locations:
`computational_study_agent/postprocessing.py:81`;
`computational_study_agent/web_assets/planner-postprocessing.js:71`.

The backend only removes rows whose `publication_eligible` is exactly false.
An explicit `status=failed` row with no eligibility field remains accepted.
The WebUI instead filters successful status strings and retains
`status=succeeded, publication_eligible=false` in its preview selection.
The backend subsequently removes that second row, so this is not a claim that
normal current reports necessarily publish failed data. It demonstrates an
inconsistent contract at the direct/legacy API boundary and preview mismatch.

Eligibility should be resolved once in the existing backend context and sent
to the UI with exclusion reasons. Legacy missing flags must not override an
explicit failed status.

### B06 — PlotSpec still silently coerces invalid explicit input

Priority: P2. Locations:
`computational_study_agent/postprocessing.py:333`, `:420`;
`computational_study_agent/web_assets/planner-postprocessing.js:27`.

Confirmed examples:

- `sort_by_x="false"` becomes true.
- `dpi=0` becomes 300.
- An unknown `group` column is accepted; grouping can silently collapse.
- The frontend parses `1.2 eV extra` as 1.2 and `1e999` as Infinity, while the
  backend rejects both as numeric cells.
- Rows missing usable numeric data are omitted from plot data without a
  per-row exclusion reason. This is separate from publication eligibility.

Validate the plot request in its existing owner: missing values may receive
documented defaults, but invalid explicit values should produce field errors.
Make usable/excluded row counts visible instead of adding repair heuristics.

### B07 — Optional execution feedback still contains language-specific logic

Priority: P2. Locations:
`pyscf_agent/request_builder/llm.py:635`, `:693`;
`pyscf_agent/application/calculation_service.py:373`.

The same failed-task fixture calls the feedback model with 8,706 o200k input
tokens, including the 8,162-token full output catalog. This is a separate
single-task feedback route; it is not multiplied by the 16 Study cases above.

The model reply `无需调整。` is suppressed while `No adjustment is needed.` is
returned. Suppression depends on hardcoded Chinese substrings even for an
English request. The application also silently catches feedback exceptions.
Optional commentary should not invalidate a numerical result, but its failure
should be distinguishable from no feedback being needed. Keep that decision
structured or in the existing deterministic eligibility rule, not language
substring lists.

### B08 — Some optional evidence failures still lose their reasons

Priority: P2 for observability. Examples:
`adaptive/state_tracking.py:59`, `:129`; `adaptive/refinement.py:91`;
`pyscf_agent/backend/correlation/active_space.py:255`.

Unreadable root-RDM artifacts become None; molecular-size probing returns
unknown dimensions after any exception; AO contribution enrichment returns
its partial result without a failure reason. State tracking does expose which
similarity evidence it ultimately uses and marks energy-only matching
ambiguous, so this is not evidence of automatic acceptance based on energy
alone. Preserve these optional boundaries while recording why evidence was
unavailable, following the completed NOON/T2/1-RDM changes.

## Reasonable Defaults And Boundaries To Keep

- One bounded contract repair, without a locally invented scientific plan.
- Compatibility retry for a provider lacking structured-output support. Its
  current string-based exception classifier could be narrowed to structured
  provider error information when transport supports it.
- Runtime solver/capability validation, cost approval and execution receipts.
- Optional enrichment failures that preserve completed energies and explicitly
  report unavailable evidence.
- Numerical convergence thresholds, declared algorithmic heuristics and
  user-selected resource profiles. These are not all generic input fallbacks.
- The user's `your_node` setting; it should not be removed as an accidental
  scientific default.

Dataset generation already rejects unsuccessful receipts; collection either
returns success after checks or raises. The unconditional outer success value
is not currently evidence of swallowed child failures on these two paths.
Generation and collection remain separate, deterministic operations.

The previously discussed block2 MPO optimization is still separate work:
`providers/block2/driver.py:747` does not explicitly pass `algo_type`. This audit
does not claim a installed-provider default or a performance improvement; no
block2 numerical benchmark was run.

## Recommended Order

1. Complete the existing PostprocessContext/PlotSpec contract: eligibility,
   units, unique plot coordinates, valid fields and explicit exclusions.
   Let the UI consume that resolved context instead of maintaining another
   interpretation of successful rows and numbers.
2. Complete the shared LLM context boundary: relevant output contracts once,
   relevant Wiki guidance, scoped scientific evidence and a measured input
   budget. Keep complete stored data and complete validated Study semantics.
3. Preserve provider usage/model/finish reason for every attempt and aggregate
   by user action; remove language-based feedback filtering and record optional
   evidence failures. These fit existing result/diagnostic structures and do
   not require a new recovery framework.

## Verification

The replay asserts 16 cases on every successful drafting route, no replacement
draft after two invalid replies, and 16 persisted cases with zero Planner model
calls during Build Plan. Counterexamples use real normalization/rendering code
and repository JavaScript functions; numerical fixture values are synthetic.

130 existing focused tests passed with no skips or failures, covering Planner
Wiki/repair, plotting, dataset postprocessing, request building and the previous
fallback cleanup. The suite took 5.334 seconds locally. The complete test suite
and remote solver benchmarks were not rerun for this audit-only change.

Reproduction (make tiktoken available in the selected Python environment):

```sh
PYTHONPATH=/tmp/pyscf-token-audit-deps \
TIKTOKEN_CACHE_DIR=/tmp/pyscf-token-audit-cache \
MPLCONFIGDIR=/tmp/pyscf-audit-matplotlib \
XDG_CACHE_HOME=/tmp/pyscf-audit-cache \
.venv/bin/python tools/audit_webui_context.py \
  --output reports/verification/webui-context-audit-2026-09-17
```
