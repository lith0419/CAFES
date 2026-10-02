# Model scan parameter placement

A live 32-site Hubbard U/V grid draft failed before execution with
`Refined axes cannot control request or solver settings`. The planner had emitted
`case_design.template.request_updates.parameters = {U: "$U", V: "$V"}` with
no model operations. That field is not a supported Hamiltonian update. Relaxing
the refinement solver guard would not make it an executable coefficient scan.

The shared case-design contract now rejects this field in templates, explicit
cases and overrides, with a message naming the field and showing the supported
`sweep` / `set_global_parameter` alternatives. Study validation and plan building
apply the check even when grid refinement is disabled. LLM draft validation
applies it before accepting a candidate, so the existing bounded contract-repair
request can correct the response once. A second invalid response remains an
error; no silent conversion or solver change is introduced. The prompt and Wiki
now show U/V operation templates and retain fixed solver/refinement settings.
Successful Build Plan also clears stale draft-validation text.

The live draft was corrected to two global coefficient operations, keeping its
16 initial U/V points, 32 sites, 48 bonds, half filling, t=-1, finite-graph DMET,
primitive-cell fragments, AF reference, CCSD impurity solver and beta=1000.
The browser saved Study `20260925-135734-072fdd7c` successfully. Every seed point
and a constructed midpoint (U=1.5, V=0.75) were inspected for site U, bond V and
effective V, hopping, electron sector and full solver-contract preservation.
No calculation was submitted during this debugging session.

Validation: 116 related tests and 20 Wiki tests passed in the main Python 3.12
environment. The four new contract tests also passed under the Python 3.9
interpreter used by the user's existing WebUI (additional compatibility evidence,
below the supported Python minimum). JavaScript syntax and `git diff --check`
passed. LLM repair was tested with controlled provider responses, not a live LLM
regeneration. The saved plan works in the existing WebUI; new server-side
validation/prompt changes require a WebUI restart to load.

Evidence: `reports/verification/scan-parameter-plan-2026-09-25.json`.
