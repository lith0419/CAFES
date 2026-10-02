# Development Note: block2 And Continuation Workflows

Date: 2026-08-11
Baseline: changes after commit `ff75415`

## Scope

This milestone extends the existing molecular, periodic, and model-Hamiltonian
platform without changing its central execution rule: numerical work is run by
registered backends, while the LLM prepares structured requests and explains
reviewable decisions.

## Implemented

- Added an optional block2 provider for fixed-orbital molecular DMRG-CASCI and
  finite-cluster model DMRG.
- Added an internal electronic-Hamiltonian boundary shared by molecular and
  model providers.
- Added Boys/Pipek-Mezey active-block localization and canonical, Fiedler, or
  explicit manual solver-orbital ordering with recorded provenance.
- Added structured block2 sweep, root, RDM, entanglement, symmetry, checkpoint,
  and MPS-manifest results.
- Added explicit MPS continuation. During user-requested result analysis, an
  unresolved block2 case may be retried from the nearest compatible successful
  MPS on the same execution target after a fingerprinted approval.
- Kept initial study cases independent. Neither AO-reference 1RDM propagation
  nor MPS continuation runs during the ordinary initial/refined calculation;
  cross-task reuse begins only from `Analyze Results` and its review gates.
- Added full-diagonalization model observables and registered scalar
  postprocessing fields for double occupancy, spin correlation, and connected
  charge correlation.
- Extended formal benchmarks with H4 DMRG-CASCI and a four-site Hubbard ring;
  added opt-in hydrogen-chain and Hubbard-ring scaling campaigns.
- Extended SSH/Slurm batch verification and bounded report-propagation handling.
  The maintained 2026-08-11 verification artifact records a successful remote
  two-task batch. Remote connectivity was not re-tested during this cleanup.
- Temporarily fixed the Calculation Assistant to English while retaining the
  translation resources and disabled language-switch control for later reuse.

## Scientific Boundaries

- The block2 molecular route is fixed-orbital DMRG-CASCI, not DMRG-SCF.
- State-averaged low roots are not tracked across scan points.
- MPS reuse requires compatible orbital/electron/spin/symmetry/root sectors and
  the same executor identity; changed Hamiltonian integrals are recorded.
- Finite-cluster trends and diagnostics do not establish thermodynamic phase
  boundaries.
- DMET, DMFT, AFQMC, periodic post-HF/GW, and unrestricted AVAS remain outside
  the executable capability registry.

## Maintained Evidence

- `reports/benchmarks/formal-benchmark-2026-08-11.json`
- `reports/benchmarks/block2-scaling-2026-08-11.json`
- `reports/verification/remote-batch-2026-08-11.json`

## Local Verification

- `python3 -B -m unittest discover`: 487 tests passed.
- Default formal benchmark suite: 5 of 5 benchmarks passed.
- Source wheel build: succeeded and included the optional provider modules.
- Python compile, changed-JavaScript syntax, and `git diff --check`: passed.
- Deterministic wiki curation: 40 pages and packaged runtime exports generated.
- Wiki lint: zero errors and eight warnings. Five warnings are citation-density
  guidance and three are stale llmwiki candidate hashes. A full llmwiki compile
  was not run because no wiki LLM provider was configured; the deterministic
  curated pages and runtime JSON are current.
- Remote SSH/Slurm verification: deliberately not repeated in this pass.

## Next Work

1. Validate realized module output ports after every runtime handler.
2. Add local hard timeout/cancellation and remote interrupted-resume campaigns.
3. Calibrate DMRG cost estimates and convergence defaults on target hardware.
4. Add DMRG-SCF or orbital-optimization support before presenting block2 as an
   orbital-optimized molecular solver.
5. Add libDMET and fcDMFT only through the same provider, registry, artifact,
   benchmark, and approval boundaries.
