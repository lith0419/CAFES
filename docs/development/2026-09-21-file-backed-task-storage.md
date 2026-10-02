# File-backed task storage — 2026-09-21

## Problem and behavior

The lutein controller exceeded its 256 MiB allocation while decoding a roughly
367 MB TaskReport. Full MO guesses appeared in the request, message history,
approval payload and report copies. Studies then embedded entire reports in
their checkpoint. Increasing controller memory would preserve this scaling.

Full `initial_mo_coeff` matrices now become lossless NPY artifacts before
workflow history and new Study execution copies. The existing field accepts
either an inline matrix or a `pyscf-agent.numeric-array.v1` reference with
`path`, `shape`, `dtype`, and standard artifact metadata. The array content
identity (`array_sha256`, covering dtype, shape and bytes, not the NPY header)
deduplicates repeated copies. Matrices of at most 4096 elements may stay inline.
Malformed inputs remain the responsibility of numerical input validation.

CAS explicitly loads a referenced array through
`pyscf_agent.artifacts.arrays.read_array`, using a read-only memory map and
`allow_pickle=False`. Missing data or inconsistent dimensions are errors.
UNO/AVAS proposals and approvals pass the same reference. Existing DMRG/DMET
RDM and one-particle-state NPZ artifacts keep their existing readers and formats.
This does not implement cross-geometry MO projection or excited-state MPS reuse.

Local workers and Slurm manifests preserve file-backed requests. SSH attaches
local referenced arrays, materializes them on the remote host and rewrites
paths. Existing server paths stay remote. Numerical attachments have a 64 MiB
aggregate server limit; existing 5 MiB text-input limits remain unchanged.
No numerical files are loaded during ordinary report viewing.

## Persistence and readers

- All Study checkpoint writes use one persistence boundary, including the
  initial, intermediate and final lifecycle writes.
- Collected TaskReports are stored in `case-reports/`. Each checkpoint case
  contains `task_report_ref` and `task_summary` (run ID and scientific status),
  alongside the existing task/run/receipt association and attempt count.
- Write order is numerical/text artifacts, TaskReport, then atomic checkpoint.
  The public StudyReport still contains the full case view, with compact
  TaskReports; the execution checkpoint no longer duplicates those reports.
- `load_checkpoint(..., include_reports=False)` reads status only. Collection
  explicitly reads report JSON; arrays remain references until numerical work
  requires them. No lazy dictionary or additional state engine is introduced.
- Presentation text exceeding 12000 characters is retained in a separate text
  artifact with a bounded preview. Repeated saves are idempotent. Numerical
  contracts and scientific result values are not truncated.
- Postprocessing takes comparison rows, case variables and coordinate units,
  instead of deep-copying task wavefunctions, inputs and logs for plotting.
- The shared JSON writer streams into a temporary file and atomically replaces
  the destination. It no longer deep-copies and serializes a second full payload
  in memory. A failed write preserves the previous destination.

## Compatibility and rollout

Legacy inline orbital guesses and checkpoints remain readable. Existing inline
Studies preserve their submitted input fingerprints and pending-run identity;
historical files are not automatically bulk-migrated. New Studies record
`orbital_storage=artifacts` and keep file-backed inputs through retries.
Persisted numerical files must remain available for subsequent numerical work.

Registry checks also exposed an earlier frozen-orbital registration omission.
The capability is now owned by the orbital-processing module. Explicit Registry
metadata distinguishes localization methods, so neither the planner nor the UI
offers `frozen_orbitals` as a localization method.

Changes are local until deployed. No Amarel calculation was submitted, and no
existing scientific results were modified during this work.

## Verification

Tests cover lossless 1294-by-1294 matrix storage and deduplication, lazy reads,
missing/corrupt metadata, 23-case compact checkpoints, legacy loading, selected
retries, annotation preservation, atomic write failure, idempotent previews,
remote attachment transfer, and plot contexts that cannot copy numerical data.
Real H2 CASCI and complete DF-CASSCF workflows compare file-backed and inline
initial guesses to 12 decimal places.

The synthetic 23-case storage benchmark is recorded in
`reports/verification/file-backed-storage-2026-09-21.json`. It is a storage test,
not a 23-point scientific calculation or an Amarel resource measurement.
