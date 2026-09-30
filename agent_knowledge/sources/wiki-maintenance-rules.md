# Wiki Maintenance Rules

## Scope

This document defines how conversation-derived rules enter the maintained
CAFES wiki.

## Current Design

`agent_knowledge/conversation_rules/` is an inbox for rules discovered during
agent/user work sessions. `agent_knowledge/sources/` contains reviewed source
material that can be compiled or curated into the main wiki.

## Rules

- Capture conversation-derived rules in `conversation_rules/` first.
- Periodically merge stable inbox notes into `sources/`.
- Keep rough, contradictory, or unreviewed notes out of `sources/`.
- After source promotion, recompile or curate the wiki and run lint before
  trusting generated pages.
- Runtime code should not read `conversation_rules/`.
- The planner should use compiled wiki evidence only as explanatory or planning
  context; capability execution still depends on backend validation and the
  runtime capability registry.
- Generated capability markdown is a documentation snapshot from code. Regenerate
  it from the registry instead of editing generated capability tables by hand.
- Maintained source notes outrank historical compiler candidates and exports
  during curation. Explicit `pageGuidance` replaces a page body; update that
  guidance together with the reviewed source when its behavior changes.
- Curate deterministically, lint links, then inspect retrieved excerpts from the
  packaged export. Page generation alone does not prove that relevant guidance
  reaches the planner within its passage budget.
- Use English for maintained guidance and constructed retrieval context. Keep
  runtime Registry facts separate from explanatory heuristics, proposals, and
  archived validation evidence. Do not claim proposed benchmarks have passed.

## Promotion Checklist

- Decide whether the rule belongs in an existing source or a new source file.
- Preserve the original rationale or failure case when it explains why the rule
  exists.
- Add or update related-page links.
- If a source changes a runtime behavior rule, make sure code and tests enforce
  it.
- Run the wiki curate/lint flow after source updates.

## Failure Cases

- Conversation notes are assumed to be available to the planner even though they
  were never compiled into the wiki.
- Generated wiki pages are edited directly and overwritten by the next curate
  run.
- Capability tables in wiki text drift away from the runtime registry.

## Related Pages

- [[Wiki Compilation Goals]]
- [[Capability Registry]]
