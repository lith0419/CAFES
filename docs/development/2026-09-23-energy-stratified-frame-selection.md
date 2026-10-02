# Energy-stratified trajectory frame selection

CAFES now registers `select_energy_stratified_frames` under
`postprocessing.action`, owned by `study.postprocessing`. The implementation
uses the existing Study application service and `/api/study-postprocess`
endpoint. The WebUI exposes **Select by Energy** for datasets with accepted
samples. There is no second planner, scheduler, LLM call, or CCSD submission.

## Example

```python
result = service.run_postprocessing(report, actions=[{
    "action": "select_energy_stratified_frames",
    "count_per_molecule": 20,
    "time_start_fs": 200,
    "time_end_fs": 1000,
}])
```

Only `study_id`, `work_dir`, `system_type` and `dataset_manifest` are needed in
the report for this action. It reads the existing manifest and streams the
accepted JSONL index; trajectory NPZ files, wavefunctions and the full Study
comparison table are unnecessary. For remote calculations, run beside the
saved index or use an already available local index. This change does not add
a remote RPC or deploy a new Amarel runtime.

## Selection rule

1. Filter each molecule's accepted frames by the inclusive time window.
2. Sort by MD potential energy, then original frame index and sample ID.
3. Partition into K equal-population bins, with remainder rows assigned to the
   first bins. Take the lower middle row of each bin.

The required K is `count_per_molecule`; the default time window is the entire
trajectory. The example's 20 and 200–1000 fs are campaign choices, not global
scientific defaults. For fewer than K candidates, select none for that molecule
and record the shortage. Partial source campaigns produce partial selections.
Equal energies are resolved deterministically without random sampling.

`HamiltonianSample.total_energy_hartree` contains the electronic BO total
energy, i.e. MD potential energy. Time comes from provenance or the declared
fixed MD timestep multiplied by the original frame index. Older sample indexes
therefore remain usable without loading arrays or rewriting source artifacts.

## Outputs

Under `postprocessing/energy-frame-selection/`:

- `selection-manifest.json`: options, algorithm, source paths and per-molecule
  counts, including shortages and the source rejection count.
- `selected-frames.jsonl`: selected geometries, original atom order, charge,
  spin, molecule split, source method, Study/Run/frame identity, time, energy
  and energy-bin membership.
- `selected-frames.xyz`: coordinates in Angstrom and identifying comments.

Outputs use registered artifact kinds and JSON schemas. Original trajectory,
sample index and scientific reports are unchanged. The subset is for labeling;
it does not establish statistical independence or a canonical ensemble.

## Validation

Unit/integration tests cover uneven bins, energy ties, shuffled inputs,
per-molecule grouping, time boundaries, frame versus array index, insufficient
samples, invalid options, invalid/duplicate samples, artifact exports, Registry
ownership, the application and HTTP paths, and the compact browser request.

The same selector is also checked read-only against the accepted pilot40
Amarel index; see `reports/qm9_energy_selection_20260923/REPORT.md` for the
selected geometries and counts. No new electronic-structure jobs are submitted.
