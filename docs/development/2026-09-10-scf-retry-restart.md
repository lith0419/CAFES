# SCF Retry From The Last Reference Density

Inspection date: 2026-09-10.

## Problem And Result

Commit `17a50ae` (2026-08-27) introduced automatic molecular reference recovery:
an unconverged SCF switches to Newton and doubles the cycle limit. The runner
then discarded the previous solver object and created a new one. Because
reference-density artifacts were written only for converged SCF, the retry
usually restarted from the default atomic guess.

The existing one-particle-state path now also captures the last finite density
of an unconverged molecular SCF. Its metadata retains `scf_converged=false`.
The compact result identifies its binary artifact through
`one_particle_state.data_artifact`. Automatic SCF recovery sets the existing
`initial_state=projected_1rdm` contract to that artifact; the existing runner
loads it and calls PySCF's native `kernel(dm0=...)` interface, including Newton.
No new task type, public restart mode, MCP tool, or numerical solver is added.

The subsequent lutein inspection found that automatic Newton recovery was also
an unsuitable default. Its first standard DF-UHF attempt completed 50 cycles in
about 36 minutes; restarting Newton from an atomic guess then took more than
18 hours to complete 21 outer cycles. Automatic reference recovery now preserves
the requested SCF algorithm and doubles the cycle limit. Newton remains an
explicit `runtime.scf_algorithm=newton` option, which is also preserved on retry.
This policy change does not relax the convergence thresholds.

The retry log records the effective initial state. The resulting report records
the source artifact and `initial_state.source_scf_converged`. A usable density
does not convert an unconverged attempt into a successful scientific result.

## Boundaries

- Recovery uses only the density referenced by the latest attempt's result,
  even when earlier attempts or an external study case have artifacts nearby.
- A missing latest artifact leaves the compatible original initial-state
  policy in effect. Recovery does not scan old files for a substitute.
- If the existing recovery rule changes a restricted reference to unrestricted,
  it clears incompatible initial-state data and starts with a fresh guess.
- Non-finite or incorrectly sized density matrices are not saved as restart
  evidence. Existing molecular compatibility checks still apply when loading.
- This restores an AO density guess after a numerical attempt returns. It does
  not restore DIIS/Newton iteration history, reuse DF integral caches, or add
  interruption-time checkpoint capture for killed processes.

## Verification

The full lutein CAS follow-up exposed a reference-policy mismatch: CAS tasks
use UHF for the initial SCF even when `method.restricted=true` selects a
spin-adapted CAS optimizer. Portable SCF-density metadata and restart checks
now use that actual initial-reference policy. A saved UHF probe density can
therefore initialize restricted/ROHF CAS without being rejected as RHF/UHF
mixing. A true UHF-to-RHF single-reference restart remains incompatible.
The CAS orbital provenance also labels a generic audit-provided matrix as
`active_space.initial_mo_coeff`; it does not call every projected matrix AVAS.
The new numerical regression compares restarted DF-CASSCF with the FCI and
block2 solvers where block2 is installed, and checks the native `dm0` input.

The public CalculationApplicationService with LocalExecutor runs real
DF-RHF and DF-UHF water calculations with a deliberately short first SCF. Both
perform one automatic standard-SCF retry and converge. The test rejects any
implicit Newton call and compares the native retry input with the first
attempt's persisted density,
checks the unconverged source metadata, and serializes the full report.

Additional tests cover latest-attempt selection, removal of an external case's
identity, missing artifacts, reference-policy changes, invalid density data,
and preservation of an explicitly selected Newton algorithm. Existing numerical
workflow tests continue to exercise explicit Newton execution.

- Python 3.9 / PySCF 2.13.0: 155 focused regression tests passed, including
  calculation workflows, gates, numerical modules, DF-CASSCF, and adaptive study
  planning.
- Python 3.12 / PySCF 2.13.1: all 6 restart regression tests passed.
- Earlier density-reuse evidence:
  `reports/verification/scf-retry-restart-2026-09-10-*`.
- Algorithm-preserving recovery and restart evidence:
  `.pyscf-agent/lutein-df-restart-20260910/`.

## Amarel Boundary

The earlier lutein job `61337604_0` used immutable release
`lutein-df-20260909-20260909-190953z-2c203ec9aa`. It was cancelled on the user's
request after saving its latest complete Newton checkpoint. The original
standard-SCF density was no longer available. The saved Newton density is an
unconverged starting guess for the replacement standard DF-UHF calculation;
it is not a final energy or a converged scientific result.

The replacement request retains geometry 12, def2-SVPD, a neutral singlet UHF
reference, density fitting, correlation diagnostics and active-space audit.
It uses 100 standard SCF cycles with the original thresholds, the existing
portable density-restart contract, and a separate Run. Resource settings remain
16 cores, 64 GB and 24 hours, with the requested `your_node` node. No active space
is approved and no CASSCF task is included. The existing UNO audit caveat from
the density-fitting development record still applies.

Deployment identity, checkpoint provenance, validation, request and the new
job handle are recorded privately under
`.pyscf-agent/lutein-df-restart-20260910/`. The old source release and job history
remain available separately.
