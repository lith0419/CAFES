# Remaining Fallback Cleanup

Implementation date: 2026-09-17. This follows the
[remaining-fallback audit](2026-09-17-remaining-fallback-audit.md).
The audit is a historical record; its counterexamples describe the preceding
working tree.

## Design

Complete the existing owners instead of adding another recovery framework.
Scientific inputs distinguish missing values from invalid values. The request
builder receives explicit scientific choices from the LLM; diagnostic consumers
check evidence validity and applicability before using a value. Recovery changes
canonical runtime fields, and provider configuration supplies numerical defaults.

## Changes Against The Audit

| Finding | Implemented behavior |
| --- | --- |
| A01: unknown diagnostic classification | Initial-scan routing accepts weak/moderate/strong only, including a molecular routing-level override. Blocked decisions retain the original diagnostic payload. |
| A02: keyword default authorization | Removed text-based basis/geometry generation, molecule-name matching and its unused coordinate table. The LLM supplies explicit basis/coordinate proposals; new inferred coordinates still require approval. Question filtering uses missing-field records, preserving unassociated questions when only some fields are resolved. |
| A03: unapplied natural-language revision | With no configured LLM, a natural-language revision returns unavailable with the unchanged seed and no executable request. A structured seed without messages remains preparable. |
| A04: adaptive defaults | Invalid options, method-policy keys, resource integers, localization and ordering are rejected. Explicit natural-orbital settings survive. Sparse provider options remain sparse and repeated normalization preserves saved Study identity. |
| A05: browser coercion | Filled invalid numbers, fractional iteration counts, malformed impurity dimensions and split seeds produce form errors. An empty optional numeric control alone uses its documented default. |
| A06: owner normalization | Task orbital indices use lossless integer conversion. Block2 rejects malformed weights/order and negative seed/verbosity; valid single-root [1.0] weights remain equivalent to a state-specific solve. fcDMFT and dataset booleans normalize explicitly. Preparation normalizes spin before choosing a reference. |
| A07: DF aliases | Accepted DF aliases and parsing loops share declarations: density_fit, df, auxbasis, df_auxbasis and density_fitting_auxbasis work again. |
| A08: model runtime | Finite-model reference construction receives RuntimeSpec through task dispatch and generated scripts. Iterative post-HF solvers use applicable cycle/tolerance controls. Every attempted UHF guess records success, nonconvergence or failure; exhausted guesses raise with all reasons, without an unrecorded duplicate attempt. |
| A09: recovery override | Study recovery reads canonical runtime values and removes their top-level aliases before updating. A max_cycle=800 request becomes an effective 1600 rather than a hidden nested 200. |
| A10: occupation evidence | Missing UNO construction stays unavailable instead of adding unrelated alpha/beta MO indices. Its fractionality component is not scored as zero. Model NOONs retain raw eigenvalues; values outside [0,2] beyond 1e-6 roundoff tolerance are marked out_of_bounds and excluded from occupation scoring. |
| A11: root matching | All root counts use optimal assignment and an exact best/runner-up margin. Identical signatures at nine or more roots remain ambiguous. |
| A12: entropy applicability | Spin-free RDM entropy fallback is limited to a verified SU(2) singlet. Other cases retain the NPDM failure and report entropy unavailable, preserving the completed solve. |
| A13: cost presets | Costing uses normalized block2 provider presets. Reports explicitly scope the proxy to the initial DMRG schedule and exclude adaptive extensions, final solves and CASSCF macroiterations. |
| A14: optional evidence | Failed reference-density capture, CAS natural occupations and T2 importance expose status and reasons while keeping completed energies. Unavailable amplitudes and failed amplitude processing are distinct. |

Block2 orbital-only restart manifest/provenance fields are now included in the
owned options contract so validation, costing and saved adaptive preparation
preserve the existing continuation workflow. QH9 relaxed SCF runtime defaults
are read from the Registry instead of another dataset table. MD guidance no
longer automatically turns an arbitrary MD request into a QH9 campaign; the
currently executable MD profile restriction remains unchanged.

## Compatibility And Limits

- Finite-model references now follow TaskSpec defaults: max_cycle=50, and an
  absent conv_tol leaves the upstream PySCF default. Previously they silently
  used 200 and 1e-10. Set explicit runtime values to reproduce that reference
  policy. DMET's provider-specific reference controls remain separate.
- Existing bounded UHF multi-start heuristics and their site-index guesses are
  retained. They are not a general graph-sublattice algorithm. Failure reasons
  are visible; this change does not introduce a new reference-search engine.
- Fixed capability bounds, review gates, the user's your_node node selection,
  and valid compatibility aliases are retained.
- No claim is made that every remaining heuristic or broad exception handler
  is removed. Optional evidence boundaries intentionally catch failures and
  return explicit evidence status.
- MPO construction, split localization, larger-M convergence and server
  performance comparisons need separate numerical benchmarks. No remote job,
  deployment, production restart or GitHub push was performed here.

## Verification

`tests/pyscf_agent/test_remaining_fallbacks.py` supplies 20 focused regression
tests, including browser-number behavior, provider/task contracts, optimal
root matching, model runtime dispatch/generated input, bounded reference
attempts and a real small H2 CASCI calculation with an injected optional
natural-occupation failure. Existing Study save/approval/resume and MCP tests
exercise stable request identity and orbital-only restart workflows.

Full regression counts, Python 3.9 compatibility, JavaScript checks and wiki
lint results are recorded in
`reports/verification/remaining-fallback-cleanup-2026-09-17.json`.
Block2/libDMET are not installed locally, so provider-specific numerical runs
remain skipped; no lutein-scale numerical acceptance is implied.

The complete suite ran 1,125 tests: 1,068 passed and 57 skipped, with no
failures or errors. Python 3.9 passed 90 focused tests. After the final small
question/evidence cleanup, 105 affected-module tests passed. Both edited browser
scripts passed syntax checks. Wiki export produced 43 pages; lint found zero
errors and three pre-existing citation warnings.
