# Native Planner examples

Four separate saved Studies for the existing Computational Study Planner.

| Study ID | Results |
| --- | --- |
| `example-lutein-gs` | State-specific GS, 23 geometries |
| `example-lutein-sa4` | Four-state-averaged SA4, 23 geometries |
| `example-honeycomb-afm` | AFM-initialized DMET, 239 selected points |
| `example-honeycomb-cdw` | CDW-initialized DMET, 84 selected points |

## Open from the Web interface

Prerequisite: clone the source repository and follow the
[installation guide](../docs/guides/installation.md). Activate the Python
environment where CAFES is installed; its location does not need to be `.venv`.
The examples are included in the source repository, not the wheel.

Start the normal Web app from the checkout:

```sh
pyscf-agent-web
```

1. Open `http://127.0.0.1:8000/computational-study/`.
2. Under **Saved Studies → Report examples**, select one of the four Studies.
3. Click **Open Example**. Its plan, results and saved figures appear in the
   ordinary Planner. AFM and CDW have separate entries.

The first click copies only the selected example into the local Work Directory
and expands its shared templates into native Study files, then resolves report
paths. Existing imported Studies are reopened without
replacing user changes. Loading an example does not submit calculations or
require cluster access. No preparation or plotting command is needed.

Optional source verification:

```sh
python examples/open_planner.py --verify-sources
```

The existing `python examples/open_planner.py` launcher remains an optional
shortcut that prepares all four examples and starts Planner on port 8767.

## Evidence and scope

These are imported result snapshots for inspection, not complete restart
bundles. All `sources[].path` entries in `manifest.json` are relative to
`examples/` and resolve to bundled JSON files. Their `sha256` values verify
those public copies with `--verify-sources`. Workspace and cluster path prefixes
have been replaced with `source://` identifiers; numerical data are retained.
Each `original_archive` records the author's original path and original file hash
separately, and does not claim that the original archive is downloadable.
The original local report directories are preserved. `source://` paths identify archived sources, not accessible
restart files. One imported checkpoint per case does not represent the
original number of execution attempts. Unavailable diagnostics are left absent.

- **Lutein:** CAS(20e,20o), M=2000, 23 geometries per Study. GS freezes 42 core
  orbitals; SA4 freezes none. Their energy difference is not solely a
  state-averaging effect. SA4 roots are energy ordered, without character
  tracking. The source geometry citation is absent from the supplied metadata.
  Curves use the existing Planner postprocessor and the archived energies.
- **Honeycomb:** AFM and CDW remain separate. Original Voronoi panels and
  precise selected observables are retained as source files. Planner displays
  native adaptive-cell heatmaps. The reports preserve `grid_refinement` with
  the archived leaf-cell bounds and case references (147 AFM cells, 48 CDW
  cells), so regenerating plots uses the same geometry. A cell uses its
  computed center value when present, otherwise its four-corner mean;
  missing results remain masked. AFM contains 239 points; CDW
  contains only the 84-point second-round cohort. Opposite-seed and externally
  accepted results are included. Per-result solver settings and continuation
  artifacts are incomplete; the saved plan is a view projection of the known
  model and solver, not a reconstruction of every original execution setting.
  These scans do not establish a ground-state phase diagram.

Use the existing Planner tables and plotting controls to inspect the results.
Full reruns require the original inputs, solver environments and continuation
artifacts.

## Storage format

Repeated case data use `*.template.json` files: one `row_template` plus the
per-case changes in `changes`. Paths with the same changing values share one
entry, so the fixed lattice stays in the shared template and U/V variations
occupy only a few values per case.
The same format removes repeated metadata in reports and source records.
All fields, numerical values and case order are preserved.

The Web importer restores standard `study-plan.json` and `study-report.json`
files in the Work Directory and verifies their `expanded_sha256` against the
original content. Planner and execution code continue to use the native schemas.
The distribution uses plain JSON, with no gzip or download step. Source hashes
in `manifest.json` check the distributed files; template hashes check their
expanded content. The shared reader is
`computational_study_agent/example_storage.py`.

## What is specific to these examples

Geometry, CAS size, frozen orbitals, state averaging, DMET settings, scan axes,
table columns and saved plots belong to the Study files. Planner does not
branch on these four Study IDs, their names, point counts or provenance markers.
Saved plots are restored by artifact kind; the existing previewer displays PNGs.
When execution receipts are unavailable, any Study with saved TaskReports gets
the same status explanation, including reports containing failed tasks.

`example_provenance` and `workflow_provenance.report_example` document the import;
they do not select execution or rendering behavior. All four bundles mark
`execution_inputs_complete=false`: Lutein retains calculation settings but
references external restart files, while Honeycomb retains a display projection
of the model. These bundles demonstrate result inspection, not complete reruns.
The launcher reads the Study list from `manifest.json`; its port and working
directory are configurable, and adding another Study requires no Planner changes.

Publish `examples/` together with the Web catalog/import API, its
`application/example_studies.py` and `example_storage.py` implementations, the Planner HTML and JavaScript
changes, and `test_example_studies.py` / `test_saved_report_rendering.py`.
The Web entry point and generic saved-plot restoration both require these code
changes. The earlier frontend restores only plots under its old `dmet-phases/`
path convention.
