# Correlation Diagnostic Interpretation

## What The Current Labels Mean

- `confidence` describes available evidence, not a probability of correctness.
  `MolecularCorrelationRisk` is a routing heuristic built from existing
  indicators and fixed implementation thresholds/weights. It is not a new
  validated physical observable, a learned classifier, or a calibrated energy
  error estimate.
- Keep `physics_level`, `solver_stress_level`, overall `level`, and
  `routing_level` distinct. SCF/post-HF nonconvergence can set overall
  `level=strong` while physical evidence is weaker and the next-method route
  remains moderate. A failed calculation alone does not prove strong correlation.
  Here `confidence` counts available evidence, not a probability of correctness.
- Missing diagnostics and SCF stability `not_requested`/`stable=null` are
  unavailable evidence. Do not convert them into a weak-correlation or stable
  reference claim. A blocked input is not a physical diagnostic.

## Fixed Scoring Policy And System Scope

- Thresholds and weights are manually specified, not data-fitted. Learning and
  calibration are future work. This molecular policy was checked against
  `pyscf_agent/backend/correlation/molecular.py` at commit `1ca69488`.
- Molecular physical weights are gap 0.30, frontier degeneracy 0.16, low-energy
  roots 0.30, mean-field/UNO occupations 0.18, and correlated occupations 0.34.
  Stress weights are SCF convergence 0.20, spin contamination 0.22, broken
  symmetry 0.28, stability 0.28, post-HF convergence 0.16, T1/D1 0.26,
  doubles amplitudes 0.22, correlation energy 0.12, and invalid density 0.30.
- Each category averages available components with its own weight sum.
  Empty categories return zero; the total remains 0.65*physics + 0.35*stress.
  Initial levels use 0.35 and 0.67. Missing evidence is not a physical weak
  correlation claim. Explicit warnings override labels, not numerical scores.
- Ordered molecular overrides are nonconvergence, invalid correlated density,
  numerically degenerate roots, T1/D1 promotion, correlated occupation scores
  at least 0.80 or 0.50, and broken symmetry with UNO score at least 0.50.
  Only the first matching rule applies. T1 >= 0.05 or D1 >= 0.10 promotes;
  the lower singles warning thresholds do not independently promote a weak route.
- Model physical scoring instead uses spin nonidempotency and matched local
  double-occupancy suppression, weighted 0.16 and 0.26. Model stress uses a
  maximum and there is no molecular weighted numeric total. No physical evidence
  gives a null score and unknown level. Model recommendations and explicit Study
  execution policies are separate from the molecular CC/CAS route.
- Periodic checks cover SCF convergence, sampled band gaps/occupations,
  metallicity and supported embedding-provider results. They do not apply the
  molecular score or T1/D1 method-promotion policy. Finite-model libDMET support
  does not establish first-principles crystalline DMET support.

## Interpreting Orbital Evidence

- Name the reference and density source when reporting NOONs: spin-summed UHF
  UNO, MP2/CC density, CAS density, or a state-average density. These are not
  interchangeable measurements. For spin-summed spatial occupations, values
  outside [0, 2] signal an invalid approximate density for active-space
  selection; clipping them would hide the problem.
- A spatial SOMO with occupation near one is normal for an open-shell
  determinant and remains relevant to CAS selection. Molecular routing compares
  open-shell occupations with a sorted 2/1/0 determinant spectrum having the
  same electron count and `abs(spin)` singly occupied orbitals. The maximum
  deviation is recorded separately from raw 2/0 fractionality; occupations are
  not removed. Open-shell electron count alone contributes no risk score.
  This is a screening baseline, not a completeness or accuracy certificate.
- UNO construction needs the spin-summed 1-RDM in the AO overlap metric, not a
  correlated 2-RDM or mandatory SCF stability analysis. Inspect spin symmetry
  and occupation evidence together; spin contamination alone is not a
  quantitative multireference error estimate.
- An ActiveSpaceAudit records a candidate and its evidence. Matching CAS
  dimensions across geometries does not prove matching orbital character or
  active-space completeness. In-space entanglement does not measure excluded
  orbitals. Compare orbital character and target-property changes when expanding
  the space; CASSCF is not a universal energy-accuracy upgrade over CCSD(T).
- Study resolution preserves explicit manual choices and their orbital matrices.
  Automatic alternatives cannot override them. Conflicting explicit choices
  require review; the original contracts remain in the diagnostic decision.

## Interpreting Solver Results

- Hubbard diagnostics distinguish measured correlation evidence, solver stress,
  and state/parameter context. Spatial NOONs remain raw data; the occupation
  score uses mean `4*n_sigma*(1-n_sigma)` over spin natural occupations. Each
  spin density must be physical and match its electron count. Open-shell and
  antiferromagnetic UHF determinants are not fractional-correlation evidence.
- Local double occupancy is compared with `n_i_alpha*n_i_beta` on the same
  covered sites and from the same density source. Zero baselines are unscored.
  DMET uses fragment 1RDMs with fragment 2RDMs, reports partial coverage, and
  keeps assembled-lattice occupation nonidempotency informational. Preliminary
  impurity-SCF beta smearing is not the correlated density or thermal CCSD.
  CCSD-DMET materializes final fragment 2RDMs only when local observables or
  strong-correlation diagnostics are requested, after the self-consistency loop.
- Hubbard `U/t`, gaps, degeneracy, spin/charge correlations and magnetic order
  describe context; they do not enter the physical score. Large amplitudes,
  perturbative energy stress, invalid densities and nonconvergence enter solver
  stress. Overall `level` is a warning, while `physics_level` describes the
  available density evidence; absent evidence gives `unknown` and null scores.
  A low score from an approximate density does not establish accuracy or rule
  out correlations missed by the selected reference/solver. The two retained
  physical components and level thresholds remain heuristic, not calibrated.

- Distinguish CASSCF orbital convergence, DMRG convergence within the active
  space, and adequacy of the active space. A final fixed-orbital solve at larger
  M improves the active-space solve but does not reoptimize the orbitals at that M.
- Inspect energy changes and discarded weights against the configured targets.
  A small sweep energy change, successful scheduler exit, or heuristic DMRG
  energy-error estimate is not a bound on the full electronic-structure error.
- DMET outer convergence does not establish convergence of every impurity
  mean-field/solver call. `publication_eligible` covers registered per-run checks,
  not fragment-size, finite-size, or scan-direction validation. Finite-cluster
  gaps and correlations alone do not establish a thermodynamic phase.

## Evidence And Next Actions

- Explain the observed failure and the registered next action separately.
  Numerical recovery, method comparison, and active-space expansion address
  different uncertainties. Model studies retain their explicit solver and use
  static solver policies with optional parameter-grid sampling refinement;
  diagnostics do not authorize automatic solver replacement.
- To evaluate diagnostic usefulness, compare with independent same-Hamiltonian
  references and specified property-error targets. Keep reference answers out
  of the tested decision input, count diagnostic cost, and evaluate missed
  unreliable results, unnecessary escalation, and curve continuity. Existing
  interface/regression benchmarks do not establish diagnostic calibration.

## Implementation Evidence

- `pyscf_agent/backend/correlation/molecular.py`: risk levels, weights, evidence
  count confidence, and method recommendations.
- `pyscf_agent/backend/correlation/active_space.py` and
  `pyscf_agent/backend/correlation/orbital_diagnostics.py`: candidate and density evidence.
- `pyscf_agent/providers/block2/recovery.py`: convergence-based recovery policy.
- `computational_study_agent/normalization.py`: static model-study boundary.

## Related Pages

- [[Strong Correlation Roadmap Rules]]
- [[PySCF Roadmap Multireference Active Space]]
- [[Scientific Benchmark Protocol]]
