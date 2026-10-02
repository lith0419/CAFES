# Explicit Density Fitting For CASSCF

Inspection date: 2026-09-09.

Historical record: the job snapshots and remaining UNO defect below describe
September 9, not the current runtime. The
[September 10 correction](2026-09-10-uno-stability-policy.md) implements genuine
UNO candidates and makes stability analysis optional. The
September 14 outcome (author archive: `reports/lutein-workflow-status-2026-09-14.md`)
records the subsequent completed CAS(20,20) runs and cancelled CAS(8,14) run.

## Problem And Result

Enabling molecular density fitting previously wrapped the reference SCF object.
PySCF then implicitly inherited fitting in restricted CASSCF, including the
block2 optimizer path, but validation accepted only `apply_to=scf` and reports
described the calculation as SCF-only fitting.

The existing DensityFittingSpec now supports `apply_to=scf_and_casscf` for
restricted/ROHF CASSCF. The shared CAS backend explicitly configures
`mc.density_fit(auxbasis=...)`; both FCI and block2 consume that optimizer's
integrals. No workflow, MCP tool, or alternate solver implementation is added.
The existing form's DF toggle works through the same normalization.

Legacy enabled SCF-scope requests on restricted CASSCF normalize to the combined
scope, preserving their previous numerical behavior. DF remains opt-in.
Reference-only probe requests convert the combined scope back to SCF. Registry
defines supported scopes for validation and the LLM schema, and def2-SVPD now
accepts the matching def2-SVP-JKFIT auxiliary basis.

Reports retain requested enabled/auxbasis/apply_to and add actual applied_to,
resolved_auxbasis, and casscf_implementation. A DF report is based on the live
optimizer and reference objects. Unrestricted DF-CASSCF is unsupported; block2
CASCI still uses conventional active two-electron integrals and must not be
presented as DF-CASCI. Fitting is an approximation separate from DMRG truncation.

## Verification

The focused regression exercises actual PySCF DF ERIs, ground-state block2/FCI
energy agreement at 1e-8 Ha, disabled fitting, legacy normalization, unsupported
method/reference combinations, automatic auxiliary basis reporting, reference
probes, and the registry-derived schema. Verification logs are recorded under
`reports/verification/df-casscf-2026-09-09-*`.

- Python 3.9 / PySCF 2.13.0 / block2 0.5.3: 198 focused tests passed.
- Python 3.12 / PySCF 2.13.1: 39 numerical/strong-correlation tests, 38 passed
  and the block2 comparison skipped because that interpreter lacks block2.
- Wiki: 43 curated pages, zero lint errors and three existing warnings.
- The initial wider Python 3.12 run hit a process-fixture startup timeout under
  the restricted sandbox. The same real MCP restart/cancellation test passed
  with process-query permission. Both outcomes are retained; no MCP code was
  changed for this environment-specific failure.

## Amarel Lutein Diagnostic-First Workflow

The initial frontier-space acceptance plan was rejected by the user. Selecting
canonical orbitals 146–165 manually was not supported by a preceding numerical
active-space audit. Those are frontier orbitals rather than the first twenty
core orbitals, but that distinction does not make the selection valid. The
existing correlation-diagnostics module would have run only after that CASSCF
calculation. The diagnostic task must precede the decision about its space.

At this inspection the job was `61337604_0`, Run `lutein-geom12-df-uhf-diagnostics`.
A compute-node inspection confirmed the running job has the diagnostic-then-audit
module order and a roughly 26-GiB DF temporary file. Its TaskReport is not yet
available; no CAS dimension has been accepted.
It uses the existing CalculationApplicationService and these registered steps:

1. DF-UHF for BLA_scan_lut_geom_12.xyz, neutral singlet, def2-SVPD.
2. `molecular.correlation_diagnostics`: SCF stability, spin contamination,
   frontier degeneracy, spin-summed mean-field/UNO occupations and routing risk.
3. `molecular.active_space_audit`: occupation/chemical-valence/AVAS candidate
   evidence and a reviewable ActiveSpaceAudit.

The request has no ncas, nelecas, orbital indices, or approved active space.
The occupation window is 0.01–1.99. The later target remains ground-state
block2 CASSCF, but no correlated task will be started from a fixed frontier
space. A candidate's existence is not proof that it is the intended conjugated
pi manifold; the audit must be examined for orbital character, mapping,
occupation evidence and SCF stability before selecting the space.

This is an explicit HF diagnostic Task using existing modules, not the Auto
probe's conditional full-space MP2 route. Current `_run_mp2_probe` retains T2;
for 156 occupied and 1138 virtual orbitals in each spin channel its unrestricted
T2 storage is approximately 704.44 GiB, exceeding the 64-GB allocation. The
initial task therefore gathers SCF evidence first. It must not present absent
MP2 natural occupations or amplitudes as computed diagnostics, or treat
unstable HF evidence as a validated final active space.

The correlation diagnostic builds genuine UHF natural occupations by
diagonalizing the spin-summed AO density in the overlap metric. However,
`propose_active_space` currently builds its separate `uno` candidate from
canonical alpha/beta occupation numbers added by index. These are different
quantities when the spin orbitals differ. An empty legacy candidate is therefore
not evidence that the genuine UNO occupation window is empty.

A real broken-symmetry H2 calculation at 3 angstrom / STO-3G reproduces the
discrepancy: converged UHF has S^2=0.998590725, natural occupations
1.037540312 and 0.962459688, but the audit helper selects no orbitals from
the 0.01–1.99 window. The reproduction is saved in
`reports/verification/lutein-diagnostics-2026-09-09-uno-audit-reproduction.json`.
This is a confirmed remaining audit limitation, not a fixed runtime path or a
lutein result. Review must use `correlation_diagnostics`' genuine occupations
and the projection reference evidence. UNO indices must not be substituted for
canonical MO indices without retaining their coefficient transformation.

At 20:05 UTC the lutein checkpoint was still updating and no TaskReport was
available. Its energy was approximately -1695.625738532 Ha; this is an interim
SCF value, not a converged single-point result. The running Slurm step had
reported MaxRSS=14487728 KiB (about 13.82 GiB), which is only the peak observed
so far. Neither memory use nor the change in checkpoint energy establishes SCF
stability or an acceptable active space.

The original directory contains 23 coordinate files with a comment but no XYZ
atom-count line. A normalized copy of geometry 12 preserves all 98 atoms
(C40H56O2), 312 electrons, and BLA 0.0559242852 angstrom; def2-SVPD has 1294
spherical basis functions. The source files are unchanged. Geometry 12 is a
scan point, not an established equilibrium structure.

The diagnostic allocation uses 16 cores, 64 GB, a 24-hour wall limit and PySCF
max_memory=30000 MB. Temporary files go explicitly under the environment's
job-specific scratch directory. The target's block2 M<=600 and 8-GiB stack
budget remain future settings, not a solver invoked by the diagnostic job.

The immutable release is `lutein-df-20260909-20260909-190953z-2c203ec9aa`,
source fingerprint
`blake2b-256:2c203ec9aa94a55dea46b8cc2122f6410c3d2a0ed8a3b8af3a12898ce073f126`.
It runs PySCF 2.13.1 and block2 0.5.3. This correction changes the task request
and execution order using already deployed capabilities; no second diagnostic
implementation or new approval mechanism was added.

A local real H2 DF-UHF check exercised the same diagnostic request structure.
It produced correlation_diagnostics, scf_stability and ActiveSpaceAudit, did not
invoke CAS, retained unset CAS dimensions and approved=false, and reported
fitting only at SCF. Its unstable HF result is diagnostic evidence, not a claim
of scientific acceptance. Evidence is under
`reports/verification/lutein-diagnostics-2026-09-09-*`.

## Earlier Attempts And Evidence

| Slurm task | Outcome | Evidence boundary |
| --- | --- | --- |
| 61337172_0 | Stopped during startup to correct the temporary path | SLURM_SUBMIT_DIR resolves to home; its empty temporary directory was removed after cancellation. |
| 61337196_0 | Cancelled following the user's rejection of manual frontier selection | Generated a 27,897,557,136-byte DF file with 4162 auxiliary rows; this verifies DF reference-integral activity, not CASSCF convergence or valid active-space selection. |
| 61337604_0 | Current diagnostic task | Await completed TaskReport and ActiveSpaceAudit; no final CASSCF result exists. |

The earlier running-job memory sample of 13.85 GiB was partial, not final
MaxRSS. A read-only compute-node inspection verified the configured memory,
PYTHONPATH and scratch settings. Opened files resolve through Amarel's
`/scache/scratch/` cache, so a login-node directory listing alone may not show
them. Reports and handles from cancelled attempts are preserved separately.
