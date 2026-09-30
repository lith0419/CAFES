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
python configure.py verify-install --output /tmp/pyscf-agent-wheel-check.json
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

## Compatibility ownership

| Compatibility surface | Current implementation | Removal criterion |
| --- | --- | --- |
| `pyscf_agent.pyscf_agent_web*` | `pyscf_agent.web.{server,api,ui}` | A documented breaking release after installed launchers and generated commands migrate. |
| `pyscf_agent.pyscf_agent_cli` | `pyscf_agent.cli` | Same release policy as command entry points. |
| `hamiltonian_dataset_*` | `computational_study_agent.datasets.hamiltonian.*` | Old callers and saved dataset workflow identifiers have a tested migration. |
| Root `configure.py` | `pyscf_agent.configure` | Source-install documentation and external launchers stop requiring the wrapper. |
| Dataset `select_seeds.py` scripts | Frozen generation snapshots | Retain with the published dataset; these are evidence, not compatibility adapters. |

Application maintainers own these adapters. They contain no alternative execution
logic. Keep runtime aliases through the current 0.2 series. Target removal is **0.3.0**,
conditional on migrating documented entry points, installed launchers, and saved
workflow identifiers and passing their migration tests. If a condition remains
unmet, document the deferred surface rather than deleting it automatically.
The new grid modules were never published under their flat names and need no aliases.

## Data and scientific boundaries

Shared QM9 preparation tools live in `tools/datasets/qm9`. Research datasets,
calculation outputs and campaign reports are maintained separately from this
public source repository.

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
