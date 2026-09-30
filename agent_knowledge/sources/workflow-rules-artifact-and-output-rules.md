# Artifact And Output Rules

## Rule

Run directories store complete evidence; UI/workflow state stores compact
summaries and artifact references. `runs/` is local runtime data and must remain
outside version control.

## Required Artifacts

- Generated molecular or model input. Periodic runs additionally preserve the
  normalized structured task, original POSCAR/CIF, and symmetry-expanded P1
  POSCAR.
- Long stdout/stderr logs when produced.
- Structured result JSON and compact result summary.
- Molecular MD manifest and frame-array NPZ with explicit geometry/energy/time
  units, AO convention, sampled frame indexes, and Fock/overlap metadata.
  Dataset finalization adds sample/rejection JSONL indexes and a dataset
  manifest. Explicit generation and collection retain their own receipts and
  manifests; finalization does not implicitly transfer remote arrays.
- Parsed periodic cell, effective periodic numerics/k-points, full orbital
  energies and occupations by spin/k-point, and periodic SCF summary when
  applicable.
- Model spectra or observable tables when a solver produces them.
- Model strong-correlation diagnostics as their own JSON artifact; Bloch band,
  DOS, and k-mesh outputs each have JSON plus plotting-ready TSV where defined.
- ActiveSpaceAudit, CAS, SCF-stability, and SC-NEVPT2 artifacts when applicable.
- block2 runs preserve the normalized DMRG result, exact-sector bond-dimension
  plan, requested and executed schedules, root energies/RDMs when requested,
  checkpoint/MPS provenance, common result contract, recovery recommendation,
  and optional entanglement arrays. Large arrays remain binary artifacts rather
  than inline report fields. Root 1RDMs may cover every requested root, while
  2RDM and entanglement diagnostics remain ground-state outputs by default.
- Molecular SCF attempts may store their last finite AO reference 1RDM as a
  binary `one_particle_state`, including when SCF has not converged. Metadata
  records `scf_converged` separately; availability as an initial guess does not
  establish a successful result. `one_particle_state.data_artifact` identifies
  that attempt's binary state for compatible continuation or automatic retry.
  The matrix itself remains in the run directory and is not copied into
  `TaskReport` context, planner prompts, or LLM messages.
- `study-state.json` checkpoint for every multi-case study, including case
  fingerprints, status, attempt count, and reusable `TaskReport` payloads.
- `execution-receipt.json` for submitted scheduler work, including plan/task
  fingerprints, executor identity, and reusable batch/job handles. It must be
  written as intent before submission; acknowledged handles are added before
  polling or result collection. An unknown acknowledgement blocks resubmission.
- Local supervised jobs retain `job-process.json`,
  `job-calculation-process.json`, and `job-cancel.json` beside `job-state.json`.
  These are executor control records; they do not replace TaskReport authority.
- SSH batch submissions retain server-owned `submission.json` evidence with
  original request fingerprints and acknowledged handles. Explicit recovery may
  associate matching pending attempts without submitting calculations.
- Adaptive initial/refined/recovery plans, decision log, and final adaptive
  study report. Normal execution stores a deferred `scan-path-diagnostics`
  marker without inspecting neighboring tasks. After an explicit result
  analysis, a scan-path continuation retry additionally stores paired
  `adaptive-path-window-restart-plan` branches, their explicit validation
  result, and both anchor provenances in the decision log. block2 result
  analysis additionally stores `adaptive-dmrg-state-tracking.json` before any
  reviewable MPS continuation plan.

## Registry Contract

Artifact types are typed `ArtifactContract` entries; this Wiki describes them
but does not authorize them. The registry groups single-task,
molecular-correlation, finite-model, Bloch-model, periodic, study, adaptive,
and postprocessing artifacts.

Every artifact reference must include:

```text
kind, path, size_bytes, mime_type, description
```

JSON payloads use their registered versioned schema when one exists. The
single-task execution contract is `pyscf-agent.task-report.v1` (`TaskReport`).
Current study artifact schemas include `pyscf-agent.study-plan.v1`,
`study-state.v1`, `study-report.v1`, `cost-estimate.v1`,
`study-execution-receipt.v1`, `adaptive-decision-log.v1`, and
`adaptive-study-report.v1`. Domain payloads include versioned
ActiveSpaceAudit, molecular/model correlation diagnostics, Bloch, and periodic
schemas. block2 payloads include `pyscf-agent.block2-dmrg-result.v1`,
`pyscf-agent.block2-bond-dimension-plan.v1`,
`pyscf-agent.block2-dmrg-casscf.v1`, and
`pyscf-agent.block2-mps-manifest.v1`. The shared summary and adaptive schemas are
`pyscf-agent.block2-result-contract.v1`,
`pyscf-agent.block2-recovery-recommendation.v1`, and
`pyscf-agent.dmrg-state-tracking.v1`. Dynamic per-case observable files match
the registered `observable-*` artifact pattern.

The writer must never emit an unregistered kind silently. Registry audit tests
compare static writers and dynamic patterns with the artifact contracts.

## Plot Artifact Contract

Every planner plot uses the common `plot-` filename prefix and writes four
co-located artifacts:

```text
plot-<name>.png
plot-<name>.spec.json
plot-<name>.data.tsv
plot-<name>.data.json
```

The PNG is the visual result. The TSV and JSON are the exact filtered plotted
data and should be visible/downloadable alongside the plot. The spec is useful
for reproducibility but need not be the primary UI artifact.

## Filename Rules

- Prefer semantic filenames such as `input-generated-pyscf.py`,
  `artifact-model-hamiltonian-spec.json`, `result-structured.json`,
  `result-model-energy-levels.json`, and `log-stdout.log`.
- Use `a.u.` or the declared input unit for model energy values.
- Avoid an `attempt` suffix unless several attempts must be compared.
- A recovery artifact must retain its parent case identifier rather than using a
  filename that hides which point changed.

## Must Not

- Do not put full stdout, generated source, or large spectra only in state.
- Do not expose an internal plot-spec file as the sole user-facing result.
- Do not overwrite full-study evidence with a one-case recovery result.
- Do not treat a path-only record as a complete artifact reference; preserve
  content length, MIME type, and description. A content digest is not required
  by the common artifact reference contract.
- Do not pass a numerical initial state by embedding a dense matrix in a plan;
  use the registered artifact reference and runtime compatibility checks.

## Related Pages

- [[Planner Agent Postprocessing]]
- [[Adaptive Scan Workflow State Machine]]
- [[Workflow State Design Principles]]
