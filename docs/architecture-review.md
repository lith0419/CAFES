# Architecture review — September 25, 2026

This review checks the current working tree, including the recent scan and
retry changes. It distinguishes defects from structural improvements and from
changes that would add migration cost without fixing the underlying problem.
The findings below record the initial review. The implementation update at the end records the subsequent authorized work and its enforcement follow-up.

## Findings and decisions

| Area | Verified finding | Recommended treatment |
| --- | --- | --- |
| Runtime directories | `runs/` is now approximately 5.0 GiB; `.pyscf-agent/`, `dist/`, and `tmp/` are approximately 1.4 GiB, 724 MiB, and 466 MiB. These are ignored local files, not evidence of Git history size. | Move the default for **new** runs outside the checkout in a dedicated migration. Keep explicit `--work-dir` and environment overrides. Preserve access to existing directories; do not move active runs or rewrite paths in place. |
| Existing path support | `backend/artifacts.py` already supports `PYSCF_AGENT_RUNS_DIR`. The plugin configurator already defaults to a user data directory outside the repository. | Reuse and unify this policy across CLI, Web, MCP, and remote execution instead of inventing another configuration system. Keep local configuration, remote release cache, run data, and build outputs as separate categories. |
| Versioned data | Git tracks 1,123 files under `datasets/` and 291 under `reports/`; their current working-tree sizes are about 19 MiB and 11 MiB. | Classify by purpose: keep small regression fixtures, manifests, identifiers, hashes, and selected scientific evidence; relocate bulk/reproducible exports through a manifest with acquisition and verification instructions. A file count alone is not grounds for removal. |
| Scripts | Ten tracked Python scripts are under `reports/`; two selection scripts live in dataset directories. | Put reusable data preparation in an explicit tool/package entry point. Keep report-specific reproduction recipes next to their evidence or provide stable links when moving them. Build scripts and Wiki curation can remain with their owning components. |
| Seed selection | The 40-molecule script selects named motifs; the 500-molecule script uses quotas, screening, and diversity selection. | Share parsing, checksums, source provenance, and validation; retain distinct selection policies and freeze current selected IDs as regression fixtures. Their large diff does not mean one policy should replace the other. |
| Configuration | The root configurator forwards to `pyscf_agent.configure`; the plugin configurator generates a machine-local MCP launcher. Tests still import the root shim and patch its exports. | Keep one runtime configuration authority, explicit entry-point wrappers, and shared path resolution. Replace `import *` only after migrating those callers. The plugin entry point has a different responsibility and is not simply a third copy. |
| Package dependency | Eight `pyscf_agent` files import Study, including function-local imports. They are Web/MCP/RPC entry points, benchmarks, and verification. Study declarations also live in the task registry/catalog. | Prefer logical core → Study → apps layering (apps import Study/core; Study imports core). Move composition and Study registration ownership before considering physical package renaming. Combining everything under `pyscf_agent.study` alone would leave the dependency problem intact. |
| Module names | Task and Study runtimes share names; the orbital projection modules implement a general projection kernel and CASSCF admission/partition policy. | Name runtimes for their actual responsibility when touched. Preserve the projection kernel/policy distinction. Identical basenames in separate namespaces are not inherently a defect. |
| Artifacts and contracts | Persistence is already centralized in `ArtifactRepository`; artifact-kind declarations, scientific extraction, array storage, and workflow evidence are separate responsibilities. Contract files belong to distinct domains. | Consolidate actual duplicate mechanisms, not all files with the same word. Keep domain contracts close to their owners and expose small stable APIs between domains. |
| Module layout | Grid refinement files are at Study's root; four dataset files use a common filename prefix. | Introduce `study/grid/` and `study/datasets/hamiltonian/` at the logical Study layer. Grid refinement is also used by **static** model scans and should not be hidden under molecular `adaptive/`. Move in isolated changes with compatibility for public imports and persisted runtime identifiers. |
| Web naming and aliases | Web entry modules repeat the package prefix. Two console commands target the same server. | A `web/` adapter package is sensible during the app-layer move. Retain existing CLI/module aliases during a documented deprecation period; an alias is cheap and deleting it can break launchers. The empty, untracked `capabilities/` directory has no architectural impact. |
| Resource lookup | Both `web_assets/template.py` fallbacks incorrectly appended `web_assets` twice. The preview module's fallback was correct because that module is one directory higher. | Fixed in this review: three callers now use `resources.text.read_text_resource` with an explicit containing directory. Package resources remain the first choice. Regression tests exercise both missing-resource exception paths. |
| Small helpers | Several JSON helpers already delegate to the same repository; the checkpoint helper additionally transforms checkpoint payloads, while postprocessing selects a reference root. UTC timestamp helpers are small duplicates. Both Web adapters define `JsonResponse`, but only the Study adapter defines `_json_response`. | Reuse storage primitives without erasing domain semantics. Consolidate transport response encoding in the app layer. Treat timestamp deduplication as opportunistic, not a major migration. |
| Schema literals | An AST scan finds 279 `pyscf-agent.*.v*` literal occurrences, including canonical definitions and non-public schemas. Existing public constants are used inconsistently. | Replace repeated references to **existing** constants first. Define owners for other schemas before adding constants. Keep literal-value assertions in contract tests so a wrongly edited constant cannot make both implementation and test silently agree. |
| Frontend duplication | Eighteen function names overlap across the two asset directories. The session state and rendering semantics are not all identical. Classic scripts rely on load order, and asset version query strings are inconsistent. | Extract pure escaping/ID helpers and explicit shared components first; pass state/callbacks into shared session behavior. Move to module-scoped dependencies incrementally. Generate asset versions from content or a build manifest instead of hand-edited dates. |
| Service size | `application/service.py` is over 2,000 lines; its main service has 43 public methods, and path restart handling exceeds 350 lines. | Keep the public facade while extracting cohesive use cases with injected dependencies: preparation, saved-study/review operations, execution/collection, result diagnostics/continuation, and postprocessing/artifact access. Preserve transaction and report-authority semantics. |
| Long functions and catalogs | Forty functions exceed 200 lines. Catalog builders are large, but CAS execution, active-space selection, parsing, and strong-correlation diagnostics contain substantial branching logic. | Split logic at invariant-preserving stage boundaries with numerical regressions. Break declarations into typed domain factories first. Convert only proven pure/static data to JSON when useful; do not convert all catalogs to YAML or invent a new configuration language merely to reduce function length. |
| Private imports | An AST scan counts 189 imported underscore-prefixed names in production packages, including imports internal to the same subsystem. | Prioritize private imports crossing ownership boundaries and private helpers embedded in exported scripts. Promote genuinely reusable contracts; do not just remove underscores or ban all internal use/test access. |
| Generated input | Molecular generated scripts call `_run_pyscf_task` and private formatting helpers. Other paths call the workflow, and model generation has a separate implementation. | Describe these as Agent replay scripts where appropriate. First provide a stable replay API plus normalized input, version/source identity, provider settings, and referenced files. A standalone native PySCF exporter is a separate feature and must not duplicate the solver pipeline casually. |
| Exceptions and compatibility | There are 101 `except Exception` handlers. Transport/process boundaries need some broad catches; scientific fallback sites need closer review. | Audit by semantics: preserve structured failure evidence and original exceptions; narrow catches that mask scientific errors. Give compatibility adapters an owner, replacement, deprecation version, and removal criterion including old persisted studies. A calendar deadline alone is insufficient. |
| HTTP routing | One hand-written server hosts multiple adapters and a long route dispatch chain. | Replace the chain with explicit route registration, consistent validation/errors, and endpoint tests before changing frameworks. A framework migration is justified by concrete needs such as streaming, authentication, or concurrency, not the number of branches alone. |
| Engineering checks | No committed CI workflow or Ruff/mypy/pytest configuration was found. The repository already has a documented `unittest` command, 95 tracked `test*.py` files, additional new tests, Node-driven UI tests, benchmarks, and wheel verification. | Add CI around existing checks immediately, then introduce lint and scoped typing incrementally. Missing pytest is not missing tests. Organize tests by domain as related code moves; do not combine a test-directory move with every other migration. |
| Documentation | There are 55 dated development notes **and** a maintained architecture source under `agent_knowledge/sources/`. | Add a visible human entry point and link the existing authority rather than duplicate it. `docs/architecture.md` and the documentation indexes now provide that entry point. |

## Execution order and acceptance criteria

1. **Fix known defects and establish automated evidence.** Start with resource
   loading and saved-study retry/report/plot regressions. Put the existing test
   command and Node UI checks in CI, then add installed-wheel resource/entry-point
   smoke checks. Use the declared supported Python baseline (currently >=3.10).
   Acceptance: fresh checkout and installed wheel checks pass; checks do not
   require personal runs, credentials, or remote clusters.
2. **Introduce incremental engineering rules and explicit dependency ownership.**
   Enable high-signal Ruff correctness checks first, then imports/style in bounded
   batches. Start mypy at typed contracts and newly extracted modules, with a
   visible checked scope. Enforce import direction separately with import-graph
   tests or a dependency checker; ordinary lint does not enforce architecture.
   Keep numerical and optional-provider suites separate from fast tests.
3. **Separate runtime defaults from source and make common infrastructure stable.**
   Agree on one new-run path policy, retaining explicit overrides and discovery of
   old studies. Test reopen, retry, checkpoint resume, artifact reads, and plotting
   against an old directory. Classify dataset/report assets before moving them.
   Consolidate actual duplicate resource/transport/schema helpers with behavior
   tests; keep storage atomicity, relative paths, and failure semantics intact.
4. **Enforce layers and make cohesive package moves.** Extract app composition
   from task infrastructure, compose Study catalogs at that boundary, then move
   grid, dataset, and Web modules in separate changes. Split the service behind
   its existing facade as those use cases become clear. Acceptance: no new
   forbidden imports; existing entry points and saved runtime identifiers work.
5. **Refactor scientific stages and frontend ownership.** Use frozen small-case
   numerical evidence to split CAS/active-space logic. Extract shared browser
   components with tests for stale replies, session switching, retry completion,
   and asynchronous plotting. Acceptance: scientific settings/results and
   task/review behavior remain equivalent within the established tolerances.
   Data-only catalog conversion and removal of obsolete aliases come last.

Ruff supports project configuration in `pyproject.toml`; mypy documents starting
with a selected portion of an existing codebase. Pytest can run the existing
`unittest.TestCase` suite if adopting it becomes useful; it does not require a
test rewrite. See the official [Ruff configuration guide](https://docs.astral.sh/ruff/configuration/),
[mypy adoption guide](https://mypy.readthedocs.io/en/stable/existing_code.html), and
[pytest unittest support](https://docs.pytest.org/en/stable/how-to/unittest.html).

## Changes completed with this review

- Consolidated the three text-resource readers and fixed the two invalid source
  fallback paths; permission and other unexpected errors still propagate.
- Added regression tests for actual template/preview fallback paths and packaged
  resource precedence. The resource/Web test selection passes 29 tests.
- Added the human architecture entry point and this decision/acceptance roadmap.

## Authorized P0–P2 implementation update

The initial migration deferred CI; the subsequent enforcement commit adds it. Implemented changes include:

- A shared user-data run-root policy, explicit overrides, and old-checkout Study
  discovery that preserves original paths for reopening, collecting and retrying.
- Local import-boundary checks and compatibility module-identity checks.
- A Study facade composed from eight use-case modules, retaining injection,
  signatures and existing scientific/report behavior.
- Canonical Web, grid and Hamiltonian dataset packages, a concise CLI module,
  explicit POST route registration, and forwarding aliases for old commands.
- Shared paths, run IDs, UTC timestamps, JSON responses and public schema
  constants. Existing checkpoint-specific serialization remains domain-owned.
- Shared browser escaping, IDs, session selectors and table markup; content-hash
  asset URLs. Page-level classic scripts remain for incremental migration.
- Local Ruff correctness checks and strict mypy checks on five infrastructure
  modules. CI now runs Ruff, the bounded mypy scope, and unittest with Node available.
- Shared QM9 preparation mechanisms with separate preserved pilot/diversity
  policies and unchanged data/output defaults; original generation scripts and hashes remain frozen in each dataset.

No saved run or scientific dataset was relocated or regenerated. Plain Study
catalog declarations remain in the shared platform registry; they do not import
executable Study code. Whole-catalog YAML conversion, numerical algorithm
refactoring, full native-JS-module conversion, and removal of compatibility aliases
remain separate changes, as described in the findings above.

See [migration and local checks](guides/architecture-maintenance.md).
