# Architecture maintenance and migration

Runtime serialization, timeout, Study ownership and provider-error policies are
documented in [Runtime reliability boundaries](runtime-reliability.md).

## Local checks

The same checks run locally and in GitHub Actions:

```sh
python -m pip install -e '.[dev]'
ruff check .
mypy
python -m unittest discover -s tests -t . -p 'test*.py'
python -m pyscf_agent.configure verify-install --output /tmp/pyscf-agent-wheel-check.json
```

Ruff checks correctness, unused imports (F401), and misplaced imports (E402),
without imposing a repository-wide format change. Mypy strictly checks the five explicitly listed infrastructure
modules in `pyproject.toml`; other modules are not claimed as type-checked.
Architecture tests scan imports inside functions as well as at module scope.
Node must be installed to run browser-state tests. Process-control and HTTP tests
need a normal local environment that permits process inspection and loopback
listeners. Optional scientific providers keep their existing skip conditions.

## Run directories

New calculations default to `$XDG_DATA_HOME/pyscf-agent/runs`, or
`~/.local/share/pyscf-agent/runs` when `XDG_DATA_HOME` is unset. Precedence is:

1. An explicit work directory supplied by the caller.
2. `PYSCF_AGENT_RUNS_DIR`.
3. The user data default.

When browsing the ordinary default, saved-Study discovery also checks `./runs`
in the current checkout. Entries carry their actual work directory; opening,
collecting or retrying an old Study continues there. An explicit custom work
directory or environment override remains a confined search root. Duplicate IDs
in separate roots retain separate locations in the saved list; an ID-only open
prefers the new default. Pass `--work-dir /old/root` for an unambiguous old root.

Nothing migrates existing runs, remote release caches or private configuration
automatically. Use the existing explicit remote work-directory configuration on
clusters. Resource/path lookup itself does not create directories.

## Supported entry points

The nine forwarding modules and duplicate Web command were removed during
cleanup of the new repository. Use these entry points:

| Area | Module or command |
| --- | --- |
| Calculation CLI | `pyscf_agent.cli` / `pyscf-agent` |
| Web server | `pyscf_agent.web.server` / `pyscf-agent-web` |
| Web API and UI | `pyscf_agent.web.api`, `pyscf_agent.web.ui` |
| Hamiltonian datasets | `computational_study_agent.datasets.hamiltonian.{contracts,planner,finalize,postprocessing}` |
| Source configuration | `python -m pyscf_agent.configure` |
| Installed configuration | `pyscf-agent-configure` |

Repository launchers, tests, registry bindings, installation instructions, and
distribution scripts use these paths. This cleanup removes the old import and
command names. Persisted report readers, checkpoint validation, and scientific
provider compatibility remain separate concerns. Archived dataset selectors
remain generation evidence.

## Data and scientific boundaries

Scientific baselines and report evidence remain versioned. Selected QM9 geometries
and their provenance were archived on Amarel on 2026-10-01; see
[storage and restore locations](repository-storage.md).
Shared QM9 preparation mechanisms moved to `tools/datasets/qm9`;
the two selection policies and output defaults are preserved. Historical dataset
scripts and their original SHA256SUMS remain in that archive as generation snapshots. Source archives,
temporary caches and fresh calculations belong in external configured locations.
Report-specific reproduction scripts stay with their evidence.

This reorganization does not change CAS/DMET algorithms or solver parameters.
Large numerical functions and a wholesale catalog-format conversion require
separate scientific equivalence work. The platform catalog stays declarative
Python, and both top-level packages continue to ship in one distribution.

## Verification of this migration

- Full suite on Python 3.12: 1,354 tests, OK, 66 skipped under existing
  optional-provider/environment conditions.
- Ruff correctness checks: pass. Mypy strict checks: pass for the five declared
  infrastructure modules.
- Installed wheel: eight checks passed outside the source tree, using the
  interpreter's existing scientific dependencies.
- Existing Study `20260925-183246-e1e9d36d`: opens from the original checkout
  `runs/` with 80 planned cases and 80 eligible postprocessing rows.
- Dataset delivery checksums: all 94 pilot and 1,025 diverse-set entries match.
  The original selector snapshots and checksum entries are restored;
  scientific data was not regenerated.
- Shared QM9 numeric/checksum helpers passed smoke checks. Complete selection
  and optional-RDKit CLI smoke were not run because RDKit is absent from the
  development environment; no new source archive or calculation was requested.

## Follow-up migrations, each reviewed separately

The current four-commit series separates grid/scan behavior, architecture
organization, typed retries, and enforcement. It does not claim that extracting
functions with a shared `self` completed dependency inversion. Subsequent PRs
should have these independently testable scopes:

1. Replace shared-self use-case functions with cohesive classes. Constructors
   receive only the storage, execution, planning, and analysis interfaces they
   need. The facade delegates and is never injected back into a use case.
2. Extract a browser session coordinator with page-specific snapshot/restore
   callbacks. Preserve stale-response and postprocessing-context protections.
3. Replace remaining production schema literals directory by directory. Keep
   independent protocol assertions and historical fixture values in tests.
4. Move cross-package composition roots to an apps layer, migrate entry points,
   and then eliminate the explicit root-adapter exceptions in the boundary test.
5. Convert only static registry declarations to validated data files. Keep dynamic
   builders in code and compare complete catalog outputs before and after.

F401 compatibility exports are explicit. The few differently named private
aliases retain narrowly scoped suppressions with a compatibility explanation.
Root-module scanning covers contracts and infrastructure; `__main__`, CLI,
configuration, verification, workbench, and documented old entry points are the
remaining composition-root exceptions.

Asset hashes use bounded caching. Filesystem identity, size, modification time,
and change time invalidate the cache for editable installations. Non-filesystem
package resources are treated as immutable during the server process lifetime.

## Verification of the enforcement update

Each split commit was tested as its own complete checkout:

| Commit scope | Tests | Result |
| --- | ---: | --- |
| Grid refinement and scan contracts | 1,342 | Passed, 66 optional/environment skips |
| Architecture and shared infrastructure | 1,354 | Passed, 66 optional/environment skips |
| Typed retries and explicit execution scope | 1,373 | Passed, 66 optional/environment skips |
| Import checks, caching, and CI | 1,374 | Passed, 66 optional/environment skips |

Ruff passes including E402/F401; mypy passes for the five declared modules.
Installed-wheel verification passes all eight checks with the existing scientific
dependencies. All 1,119 original dataset checksum entries match. GitHub Actions
uses Python 3.12 and Node 22 for the same lint, typing, and unittest commands.
