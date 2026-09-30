# Planner Agent Postprocessing

## Built-In Tools

- `line_plot`
- `scatter_plot`
- `bar_plot`
- `heatmap`

Dataset generation and collection are separate registered
`postprocessing.action` capabilities, not plot tools.
`generate_hamiltonian_dataset` is available when a completed StudyReport
contains a Hamiltonian dataset manifest; `collect_hamiltonian_dataset` requires
the successful generation receipt produced by the first action.

## Energy-Stratified Trajectory Selection

`select_energy_stratified_frames` is a registered `postprocessing.action` owned
by `study.postprocessing`. It reads the accepted Hamiltonian sample index,
without loading Fock/overlap arrays or submitting calculations. Use
`count_per_molecule` (required), `time_start_fs` (default 0), and `time_end_fs`
(default null, meaning trajectory end). All options belong to the action object.

For each molecule, retain frames in the inclusive time window and sort by
potential energy, then frame index and sample ID. Divide them into K groups
whose populations differ by at most one (larger groups first), and choose the
lower median rank of each group. These are energy quantile bins, not equal-width
energy intervals. Identical energies remain deterministic; no random seed is
needed. Molecules with fewer than K candidates are reported without filling
their quota with duplicates. Source rejections and incomplete coverage remain
visible. Missing required data or malformed accepted records produce errors.

The source `HamiltonianSample.total_energy_hartree` is electronic total energy,
including nuclear repulsion: the MD potential, not nuclear kinetic plus
potential energy. Use recorded `provenance.time_au`, or the source dataset's
fixed timestep times `frame_index` for older indexes. Never use array offset as
physical time. The time window is user-specified; do not assume a universal
200 fs equilibration period or a constant 300 K NVE trajectory.

Export a registered selection manifest, selected-frame JSONL with coordinates,
energies, bins, split and source Study/Run/frame identities, and multi-frame XYZ
in Angstrom. Preserve original data and atom ordering. The operation requires
the accepted index to be accessible where postprocessing runs; remote matrix
artifacts do not need downloading. The WebUI exposes count and time-window
inputs through the existing postprocessing endpoint. No LLM is involved.

This is a reproducible labeling subset, not proof of independent structures or
equilibrium sampling. The action does not launch CCSD or change dataset splits.

## Rules

- Plot choices are data-driven: x, y, color, and group fields come from the
  actual comparison table, never from hardcoded `U`, `t`, or `energy` names.
- Plot generation uses successful calculation rows only. Execution metadata
  such as status, attempt count, run directory, and execution source remains
  available for review but is not offered as a scientific plot variable.
- PostprocessContext owns eligible rows and scientific/numeric column choices.
  WebUI reads that backend projection rather than parsing numeric prefixes or
  maintaining its own success-status filter. Explicit failed task/row statuses
  exclude legacy rows even when publication eligibility is absent.
- Built-in plotting also excludes rows with `publication_eligible=false` and
  records `excluded_case_ids`. Raw Study rows and quality evidence remain
  available for review. `quality_status` and `publication_eligible` are internal
  metadata, not scientific axes. An all-ineligible report skips plotting;
  explicit dataset actions retain their separate eligibility checks.
- Total/final energy must remain an eligible molecular and model observable;
  dissociation curves should label a recognized coordinate such as bond,
  `bond_length`, or `bond_distance` as **Bond length** with its available unit.
  The UI presents the internal `final_energy` field simply as **Energy**.
- Model-Hamiltonian plots may use final energy, filling,
  gaps, natural-occupation summaries, amplitude summaries, mean double
  occupancy, nearest-neighbor spin correlation, nearest-neighbor charge
  correlation, and other numeric fields actually present in the report.
- The three correlation postprocessing metrics are scalar summaries of the
  executed strong-correlation diagnostics. They never replace the full
  site-resolved double-occupancy vector or pair-correlation matrices retained
  in the underlying diagnostic payload.
- Every generated plot must emit a PNG plus exact `.data.tsv` and `.data.json`
  artifacts under the same `plot-` prefix. This data is part of the study result,
  not an internal implementation detail.
- Plot specs are runtime capabilities and are validated against available data.
  The Web UI may de-emphasize spec JSON but must expose plot/data artifacts.
- Unit labels use units in cells or actual case TaskSpecs; a bare molecular
  coordinate with no unit evidence stays unlabeled. Mixed known units, or
  known and unspecified units in one column, require correction before plotting.
  Plotting does not silently convert or relabel values.
- Invalid explicit plot parameters produce field errors. Group columns must
  exist, DPI must be a positive integer, and figure dimensions and heatmap
  center must be finite. All selected plots are checked before rendering and
  their output names must be distinct.
- Image rendering and data export use the same resolved numeric rows. Missing
  numeric/group values are excluded with case ids and reasons in the data
  JSON and response. Raw Study rows remain unchanged.
- Default plot titles identify the plotted observable and its unit. Heatmaps
  place that description in the title and leave the colorbar unlabeled to avoid
  duplicate text; an explicitly requested colorbar label is still honored.
- Plot artifact references use the common kind/path/size/MIME/description contract;
  plot spec and JSON data carry their registered versioned schemas.
- Hamiltonian dataset finalization first creates lightweight sample and
  rejection indexes. It does not implicitly download large executor-owned NPZ
  files. The explicit `generate_hamiltonian_dataset` action materializes the
  complete portable dataset on the selected execution target, validates file
  presence, array keys, indexes, and matrix shapes, and rewrites Fock and
  overlap paths relative to that dataset root. For SSH/Slurm execution it
  returns only a lightweight local generation receipt; it must not transfer
  trajectory arrays to the client. The separate `collect_hamiltonian_dataset`
  action uses that receipt to download the already generated dataset only when
  the user requests a local copy. Checksum validation is not required by either
  action.
- Dataset status is derived from the existing Study plan, execution receipts,
  final manifest, generation receipt, and collected manifest. It reports
  terminal, successful, and failed trajectories; available, failed, and pending
  structures; and generated/collected state separately. It must not introduce
  a second scheduler or duplicate task lifecycle state.
- `Run Plan` leaves completed scan cases independent and records path analysis
  as deferred. An explicit `Analyze Results` request runs the deterministic
  method-continuity check before the final curve is trusted. A flagged method
  stitch emits `scan-path-diagnostics`, offers retries from tracked left and
  right AO 1RDM anchors, and accepts a replacement only after the two branches
  agree. Endpoint validation and cross-method/reference anchors require an
  explicit continuation review before execution. The analysis opens a
  shared-CAS local CASSCF refinement approval when no unique continuation
  result is established. It never alters a plot by silently substituting
  lower-level data.

## Heatmap Rules

- Require two varying numeric case variables and one numeric observable.
- Automatic recommendations additionally require observed crossed sampling:
  at least one complete 2×2 rectangle in eligible rows with finite values for
  the selected observable. Two changing columns, correlation, or a one-to-one
  coordinate mapping alone do not establish a two-dimensional grid. Incomplete
  grids may be suggested when a rectangle exists; missing cells stay missing.
- Check every pair of numeric scan variables, not just the first two columns.
  Reject automatic heatmaps with duplicate coordinates; do not silently merge
  replicates or slices of a higher-dimensional scan.
- A single declared numeric scan coordinate defaults to a line plot. Multiple
  changing coordinates without supported repeated series default to scatter
  plots, since the sampling order cannot be inferred reliably. A grouping
  variable is recommended only if it supplies at least two groups and every
  group has at least two distinct x values for the selected observable.
- Keep identifiers in case IDs/provenance instead of declaring them independent
  scan variables. Do not infer the physical coordinate from a special column
  name. Explicit custom plot specs remain available when the user knows the
  intended scan path or grid.
- Use x/y as plane coordinates and color as the observable.
- Use a red-blue diverging colormap with zero as the white midpoint.
- Require unique numeric x/y coordinates for the selected series. Repeated
  coordinates must be resolved by explicit series selection or upstream
  aggregation; never choose the last row or average implicitly. Grouped heatmap
  requests are rejected until facets are supported.

## Language Analysis

- Plotting and dataset actions do not call an LLM. Optional result analysis
  sends the scientific comparison data plus only the result/artifact contracts
  relevant to the actual report, scoped to its system type when known.
- Send selected contracts once in the user payload. Do not append the entire
  Registry output catalog to both the system prompt and the user payload.
- Preserve scientific rows; reducing contract overhead does not authorize
  sampling or truncating a Study comparison table. Large-result scoping remains
  a separate interface decision.
- Optional execution feedback uses the same context selection. Do not filter
  returned advice through language-specific substring lists. A feedback failure
  is recorded separately and does not change a completed calculation's result.

## Failure Cases

- Selecting a fixed legacy field leaves new molecular/adaptive result columns
  unavailable for plotting.
- Returning a pretty figure without its underlying selected rows prevents
  scientific review and reuse.
- Plotting only a last recovery subset instead of the merged full study creates a
  false trend.
- Treating a single-point correlation score as evidence that a mixed-method
  curve is smooth misses discontinuities that exist only along the scan path.

## Related Pages

- [[Run Directory Artifact Storage]]
- [[Adaptive Scan Workflow State Machine]]
- [[Capability Registry]]
- [[Molecular Hamiltonian Dataset Workflow]]
