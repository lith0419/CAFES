# Wiki Implementation And Scientific Evidence Alignment

## Problem

The runtime export contained a mixture of maintained source bodies and explicit
`pageGuidance` summaries. Updating a source did not update its editorial
summary. In addition, historical compiler candidates could override a newer
maintained source during curation. The result was internally inconsistent
guidance about model adaptive scans, diagnostic levels, and supported methods.

## Changes

Maintained source notes now take precedence over historical candidates and
exports for matching source/title routes. Explicit `pageGuidance` remains an
editorial body replacement and is updated alongside its source. Curation can
also build a source-only knowledge tree without previously generated state.
No LLM compilation or additional retrieval service was introduced.

The runtime Wiki now contains 44 pages. Eleven existing pages changed and one
new **Correlation Diagnostic Interpretation** page explains current heuristic
scores, evidence-count confidence, missing diagnostics, orbital/density
provenance, and the distinct meanings of orbital convergence, inner-solver
convergence, and active-space adequacy. Proposed scientific validation remains
explicitly unperformed work.

Corrected guidance includes:

- Static full-grid model studies with an explicit solver; molecular adaptive
  stages are not model execution stages.
- DMET `finite_graph` for intersite V, complete fragment coverage, and the
  translated-mode translation/spin restrictions verified against the provider.
- Independent physical evidence, solver stress, overall warning, and routing
  levels; a failed solve is not proof of physical strong correlation.
- Explicit DF-CASSCF, targeted roots/state averaging, registered periodic
  fcDMFT routes, and the restricted QH9 molecular-dynamics profile.
- Study/Task/Run ownership and retained full reports during selected retries.
- The Registry M=1024 automatic-recovery budget, distinguished from supported
  explicit larger-M calculations and from scientific convergence targets.

The prompt wrapper now identifies Wiki excerpts as planning/interpretation
guidance and preserves Registry, schema, and backend validation authority. It
no longer says that a Wiki rule unconditionally wins over a draft.

## Retrieval And Verification

Ranking and query construction remain unchanged: English scientific context,
local keyword scoring, five primary pages, linked expansion up to eight pages,
and at most 1,800 body characters per page. Seven recorded queries cover the
user's Honeycomb U/V DMET input, SCF diagnostic interpretation, final-M DMRG,
QH9 MD, DF-CASSCF, periodic fcDMFT, and impurity DIIS recovery.

The Honeycomb regression initially exposed missing execution-mode guidance in
the published solver summary. Adding the current mode and static-study rules
to that summary makes those constraints appear in the actual bounded excerpt.
This corrects the knowledge content without adding a molecule-specific query
rule or changing ranking weights.

The 68-test focused suite covers curation precedence, source-only generation,
editorial overrides, packaged retrieval, Planner contract repair, and
postprocessing context. Node syntax and whitespace checks pass. Wiki lint has
zero errors and the same three inferred-paragraph citation warnings on density
fitting, multireference active-space, and benchmark pages. The curated format
retains source provenance while stripping old compiler inline citations.

The exported runtime bodies match the generated Markdown. Detailed retrieval
evidence and interpreter results are recorded in
`wiki-alignment-2026-09-18.json` (author archive: `reports/verification/wiki-alignment-2026-09-18.json`).
No live LLM calls or numerical jobs were run. Broad keyword/link matches and
bounded excerpt omissions remain limitations; these checks do not establish
scientific diagnostic calibration.

## Loading The Update

Wiki bodies are read from the packaged JSON on each retrieval, so the updated
content is available to services importing this checkout without a restart.
An already running Python service must reload to obtain the revised prompt
wrapper. The WebUI process was left running; no deployment or GitHub push was
performed as part of this Wiki update.
