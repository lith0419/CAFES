# Wiki Query Relevance And Passage Selection

Inspection date: 2026-09-17. Follows the removal of full-registry input in
[Planner draft contract and failure semantics](2026-09-17-planner-draft-contract.md).

## Problem

The retrieval query still serialized the complete seed StudySpec. JSON field
names, study labels, geometry, and paths could therefore influence document
ranking independently of the scientific request. The tokenizer dropped all
single-letter terms, including the Hubbard parameters U, V, and t, and could
retain sentence punctuation on identifiers such as `finite_graph.`.

Page ranking examined the complete body, but evidence generation kept only
its first 1,800 characters. A page could rank well because of a rule near its
end while that rule was absent from the actual LLM input.

## Design

`build_study_wiki_query` projects the goal and StudySpec into scientific search
context. It preserves the objective, system/model kind, observables, method,
basis, reference, solver/options, enabled active-space/DF/diagnostic settings,
scan parameter names, and scientific choices in case overrides. It omits
administrative metadata and paths. This is a retrieval projection; the full
seed StudySpec still reaches the LLM and retains the original cases and values.
The projection does not validate or change a scientific request.

Tokenization retains U/V/t as complete tokens, keeps identifiers such as
`execution_mode`, `DMRG-CASSCF`, and `def2-SVPD`, and removes terminal sentence
punctuation. An English contraction does not introduce a false hopping term t.
Retrieval remains deterministic local keyword scoring with the existing page
weights; there is no new model, embedding service, or registry export.

Long-page evidence now ranks paragraphs and list items against query terms,
giving less common passage terms greater weight. Selected text is returned
in document order with its section-heading context. An oversized paragraph
contributes a window around matching terms. Omitted text is explicitly marked;
headings and markers count toward the existing 1,800-character body budget.
Short pages remain complete. The existing limit of five primary pages and
linked-page expansion up to eight pages remains unchanged.

## Verification

All constructed queries and fixtures use English. The focused 153-test suite
passed on both Python 3.12.14 and Python 3.9.6. LLM transport was stubbed; no
numerical jobs were submitted. Regressions cover:

- Invariant ranking when only study names, paths, or administrative metadata
  change, while scientific method/solver/scan revisions remain searchable.
- Independent U/V/t matches and scientific identifiers with punctuation.
- Disabled options and nested solver paths excluded from retrieval context.
- Partially edited scan inputs remaining usable for drafting.
- Rules beyond a long introduction, at the end of a long list, and inside an
  oversized paragraph appearing in the returned evidence.
- Multiple relevant passages preserving source order and headings, and strict
  character limits including separators and truncation markers.
- Existing LLM fallback/repair, seed preservation, and backend plan validation.

For the English example `Keep all 16 cases and change the DMET execution_mode
to finite_graph.`, with a DMET U/V scan seed, the query is 131 characters. The
first three results are Model Hamiltonian Solver Support, Model Hamiltonian
Input Contract, and Model Hamiltonian Operations. This is a constructed local
retrieval check, not a captured user request or a real-provider LLM run.

The remaining keyword ranking and linked-page expansion can still retrieve
broad or unrelated pages. A selected passage is bounded evidence, not a claim
that every relevant rule in the knowledge base has been included. Runtime
Registry validation remains authoritative for executable plans.

Python source changes require restarting the existing Web service in its
configured shell; refreshing the page alone does not reload the backend.
