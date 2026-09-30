# DMET Density Continuation And Branch Analysis

## Compare Saved States

- `analyze_dmet_branches(study_id, policy=None)` analyzes saved candidates without
  submitting calculations. Use final correlated sublattice charge and magnetic
  order, never the analytic seed name, to classify each converged state.
- Default order thresholds are 0.001 on absolute sublattice half-differences.
  A 20% threshold band is ambiguous; both orders give `mixed`, neither gives
  `near_unordered`, and unavailable observations give `unknown`. Record the
  configured policy with the analysis.
- Energy comparison requires successful execution, explicit convergence,
  passed required quality checks, and finite energy per site. Missing legacy
  quality evidence is unqualified; the optional density-fit residual is not a
  veto. Compare all indexed `candidate_runs`, including historical successful
  Runs after a failed retry. Current task rows and candidate minima differ.
- Energy ties default to 1e-6 per site. Order-magnitude differences above
  `state_tolerance=0.01` mark possible hysteresis; a sign reversal alone does not.
  AFM/CDW crossings require both branches at adjacent sampled V values. Do not
  bridge missing points or label numerical failure as a physical spinodal.

## Prepare And Start A Bounded Sweep

- `prepare_dmet_continuation` saves a preview and cost estimate;
  `start_dmet_continuation` starts that saved action through the existing Study
  retry lifecycle. Preserve cost-review requirements and the saved Study revision.
- Default `mode=bidirectional` follows AFM toward increasing V and CDW toward
  decreasing V at fixed U. `mode=nearest` uses scaled U/V distance and also
  supports mixed and near-unordered branches. Automatic selection requires
  uniform U/V, inline model specs, and compatible finite-graph DMET cases.
- Select the nearest qualified compatible donor again before each step,
  including newly completed and historical Runs. Each selected physical point
  and branch receives at most one new Run per action; equivalent seed cases are
  deduplicated. Missing donors cause a recorded skip, not an analytic fallback.
- Defaults are mixing 0.2, 200 outer iterations, outer DIIS off, bath policy
  `max`, and initial auxiliary correction zero. Repeated starts of one action
  do not resubmit. After interruption, collect existing Runs before preparing
  another action. A new action can revisit points while retaining candidates.
- FCI/CCSD DMET work/memory estimates are currently unavailable; missing
  estimates are not resource upper bounds.

## Density Transfer Contract

- `solver.options.reference_density_source` names a saved `source_case_id`,
  optionally an exact `source_run_id`, or a fixed absolute artifact path and
  SHA-256. The resolved donor is pinned to its Run and `dmet_mean_field_state`.
- Transfer the full final spin-resolved mean-field 1RDM, including off-diagonal
  entries. The target rebuilds its baseline with its own Hamiltonian and starts
  auxiliary u at zero. Correlation potential, chemical potential, DIIS history,
  and impurity wavefunction are not resumed.
- Check hash, source convergence/required quality, site basis, graph, fragments,
  spin/reference, bath and impurity settings. Only U/V may differ. Require real,
  finite Hermitian density of shape `(2,nsite,nsite)`, correct spin traces, and
  occupations in [0,1] within validation tolerance; never rescale the density.
- Old Runs without the final-density artifact cannot substitute initial seeds
  or correlated embedding densities. SSH stages local artifacts with checksum
  verification; cross-server paths must already be accessible to the receiver.

## Related Pages

- [[Model Hamiltonian Solver Support]]
- [[Model Grid Convergence Barriers]]
- [[Correlation Diagnostic Interpretation]]
