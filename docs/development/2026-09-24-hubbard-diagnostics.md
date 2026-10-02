# Hubbard diagnostic evidence and reference baselines

Hubbard diagnostics now distinguish physical density evidence, solver
reliability, and parameter/state context. This corrects open-shell and magnetic
determinant false positives without interpreting nonconvergence as a physical
correlation measurement. The explicitly selected model solver is unchanged.

## Density evidence

The raw spatial natural occupations and their 2/0 fractionality remain in
results. A separate spin-resolved summary uses the mean of `4*n*(1-n)` over
alpha and beta natural occupations. Idempotent spin densities contribute zero,
including open-shell and antiferromagnetic UHF determinants with different
alpha/beta orbitals. Complex Hermitian densities retain their imaginary parts.
Invalid bounds, Hermiticity or electron counts are quality warnings, not clipped
correlation evidence. Spin-summed-only input remains unscored. FCI, DMRG and
post-HF sources retain their spin information.

Double occupancy is compared with the same-site product of alpha and beta
populations. The suppression score is `1 - sum(D_i)/sum(n_ia*n_ib)` on matched
covered sites with nonzero baselines. A zero denominator remains undefined.
Enhancement is reported as signed connected opposite-spin density and is not
assigned a weak score by this repulsive-suppression metric.

The two physical components retain their previous relative weights, 0.16 and
0.26, and the 0.35/0.67 classification boundaries. Other former ingredients
no longer enter this physical average. These are heuristic evidence summaries,
not calibrated probabilities or energy-error bounds; an approximate density
can miss correlations. Degenerate states and mixed densities require their
provenance, and these checks do not determine thermodynamic phases.

## Context, solver stress, and output

Interaction scales, gaps, degeneracy, spin/charge correlations and order
parameters remain visible as context. Large T2 amplitudes, perturbative energy
stress, nonphysical densities and failed convergence enter solver stress.
Missing physical evidence gives `physics_level=unknown` and null scores.
Overall `level` remains a warning for existing clients and includes a reason.
Assistant displays the groups and separate levels; Planner's correlation column
uses physical evidence and displays solver risk separately in its diagnostics.

## DMET

Each fragment's 2RDM is paired with its own 1RDM populations, before averaging
over covered sites. The assembled global density cannot substitute for that
baseline, and its nonidempotency remains informational. Coverage is retained;
uncomputed sites/bonds are not zeros. On a diagnostics/local-observable request,
CCSD generates its final 2RDM once per fragment after self-consistency, in both
translated and finite-graph execution. Energy-only execution does not request
this additional tensor. It has the usual fourth-order density-tensor storage
cost. Impurity SCF beta is separate context, not a temperature assigned to CCSD
or a source of correlated-density occupations.

The spin conventions follow the installed adapters and PySCF's
[FCI spin RDM interface](https://pyscf.org/_modules/pyscf/fci/direct_spin1.html)
and [UCCSD density implementation](https://pyscf.org/_modules/pyscf/cc/uccsd_rdm.html).

## Numerical checks

The comparison artifact (author archive: `reports/verification/hubbard-diagnostics-2026-09-24.json`)
evaluates the old and new diagnostics on identical freshly computed FCI data.
It records the baseline commit and PySCF version. The old and new scores have
different definitions and are not interchangeable.

| Finite open chain | Old physical label / score | New physical label / score |
| --- | --- | --- |
| 4 sites, (3,0), U/t=0 | moderate / 0.354127 | weak / 0 |
| 4 sites, (4,0), U/t=16 | strong / 0.700000 | weak / 0 |
| Half-filled dimer, U/t=8 | strong / 0.708717 | strong / 0.858455 |
| Half-filled dimer, U/t=16 | strong / 0.740865 | strong / 0.959108 |

Dimer energies, occupations and double occupancy agree with their analytic
expressions at U/t=0,2,4,8,16; the new physical score increases monotonically.
Additional tests cover doped/noninteracting chains, negative U, AF determinants,
missing and invalid evidence, partial DMET coverage, CCSD-DMET with default
beta and beta=1000, both fragment execution modes, and DMRG/FCI agreement.
The CCSD-DMET tests use one outer iteration to validate data transport and
diagnostic scope; they do not claim converged DMET energies.

Validation: 281 main regression tests passed on Python 3.12.14 / PySCF 2.13.1,
including Registry, both WebUIs, numerical modules, adaptive compatibility,
public inputs and Wiki retrieval. The installed optional-provider environment
(Python 3.9.6 / PySCF 2.13.0) passed 140 libDMET/block2/Hubbard tests, with one
opt-in long benchmark skipped; 14 focused tests also passed after the final
edge-case and raw-observable changes. The older runtime is additional provider
validation, below the project's declared Python minimum. Wiki lint reported
zero errors and the same three existing citation-warning pages. JavaScript
syntax checks and `git diff --check` passed. No service restart or deployment
was performed.
