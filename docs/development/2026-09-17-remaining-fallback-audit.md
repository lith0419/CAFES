# Remaining Fallbacks And Specific Policies

Audit date: 2026-09-17. This reviews the current working tree after the input
and recovery cleanup, including uncommitted changes. No runtime source was
changed during this audit.

Implementation follow-up: [remaining fallback cleanup](2026-09-17-remaining-fallback-cleanup.md).
The findings below describe the pre-fix snapshot.

## Scope And Interpretation

The scan covered 220 Python files and 18 JavaScript files in `pyscf_agent/`
and `computational_study_agent/`, followed by call-site inspection and small
local counterexamples. Templates, Registry policies, and the preceding
implementation report were also inspected. An AST inventory found 469
exception handlers, including 100 broad handlers; 22 broad handlers directly
return an empty value, pass, or continue. These counts are search inventory,
not defect counts: many API adapters correctly turn failures into structured
errors.

The previous changes removed several transport-side scientific overrides,
but did not complete normalization or evidence handling across every path.
In particular, a nonempty unknown diagnostic level still routes as weak,
the new task-field allowlist omits existing DF aliases, and adaptive option
normalization still replaces explicit invalid values with defaults.

Evidence is recorded in
`reports/verification/remaining-fallback-audit-2026-09-17.json`. Counterexamples
use real local contract functions, mocked absence of an LLM, small NumPy
arrays, or the existing JavaScript functions in a minimal DOM stub. They do
not submit calculations, contact an LLM, or establish provider numerical
performance. The previous full regression suite was not rerun because this
turn changes only audit artifacts.

## Confirmed Findings

### A01 — Unknown diagnostic levels still select the weak method

Locations: `computational_study_agent/adaptive/executor.py:488`, `:321`, `:341`.

The new entry check rejects an absent/empty `level`, but the two routing
functions treat every other unrecognized level as weak. Successful molecular
rows with `level="unknown"`, `"unavailable"`, or `"nonsense"` all receive
`ccsd_t`; they are not tagged `initial_scan_blocked`.

Priority: high. Require a recognized, usable diagnostic classification before
routing. Missing evidence and invalid evidence must not become weak evidence.

### A02 — Free-text default authorization remains a keyword rule

Locations: `pyscf_agent/request_builder/utils.py:170`, `:194`, `:203`, `:212`;
`pyscf_agent/request_builder/prepare.py:194`, `:255`, `:295`.

`Do not use default settings.` sets parameter authorization to true and fills
a missing basis with `sto-3g`. `Do not generate the structure.` sets structure
authorization to true. The examined text also includes prior user messages,
seed summaries, and LLM-generated summaries, so the current user's choice is
not the only source of this decision. An eight-molecule lookup table supplies
fixed geometries using substring matching. Generated geometry has a later
approval step; that limits execution risk but does not correct the intent
misinterpretation or default basis change.

Priority: high. Represent default/structure choices explicitly in preparation
state. Keep the geometry table as an explicitly selected example library.
Question suppression also uses field-name substrings (`utils.py:238`) and
should follow structured missing-field records instead of rewriting prose.

### A03 — No-LLM fallback reports an unapplied revision as ready

Location: `pyscf_agent/request_builder/prepare.py:460`.

With an HF seed and no configured LLM, `Change the method to MP2.` returns
`status="ready"`, `source="fallback"`, and `method="hf"`. The instruction is
stored as `request` text; it is not interpreted as a structured edit. The
backend parser treats this as metadata alongside the seed.

Priority: high. A structured seed can be prepared without an LLM. A new
uninterpreted natural-language revision needs an explicit unavailable or
needs-input result rather than a ready result suggesting that it was applied.

### A04 — Adaptive option normalization remains lossy

Location: `computational_study_agent/adaptive/initial_scan.py:83`, `:114`.

Confirmed examples: `active_space_solver="blcok2"` becomes `auto`;
`max_cas_orbitals="invalid"` becomes 14; 3.9 becomes 3. Invalid localization
and ordering become `none` and `canonical`. Active-space orbital processing
also unconditionally sets `use_natural_orbitals=false`, and invalid orbital
indices may be dropped. Unknown options are not generally rejected; only a
small list of obsolete option names is explicitly blocked.

Priority: high. Apply the same absent-versus-invalid rule used for tasks at
the Study adaptive contract. Reuse owned option normalization before making
scientific decisions.

### A05 — Browser controls can hide invalid inputs from the backend

Locations: `computational_study_agent/web_assets/planner-spec.js:94`, `:100`,
`:107`; `planner-dataset.js:100`.

The real control functions convert an invalid or negative iteration value to
50 and truncate 3.9 to 3. Impurity shape `2.8x2bad` becomes `[2,2]` through
`parseInt`. Dataset split seeds also use `parseInt` before checking integer
type. Consequently stricter backend validation sees already-altered values.

Priority: high. Preserve an invalid field as an error in the form. Defaults
should apply only to fields that are actually absent; canonical scientific
normalization belongs to the shared contract.

### A06 — Scalar/list normalization is still inconsistent across owners

Locations: `pyscf_agent/backend/parsing.py:345`;
`pyscf_agent/providers/block2/config.py:188`, `:212`, `:274`;
`pyscf_agent/providers/fcdmft/contracts.py:99`, `:163`, `:215`;
`computational_study_agent/hamiltonian_dataset_contracts.py:102`;
`pyscf_agent/request_builder/prepare.py:271`.

Confirmed counterexamples:

- A public CAS request with manual orbital order `[0.4,1.7]` validates as
  `[0,1]`. The new dataclass checker covers scalar fields but not list entries;
  this parser branch still uses `int`.
- Block2 `orbital_order="invalid"` becomes `[]`, negative `seed` becomes zero,
  and malformed single-root weights disappear. Some inactive options may
  legitimately be irrelevant, but silently replacing malformed values is
  inconsistent with the new contract.
- GW `gw_finite_size_correction="false"` becomes true;
  `gw_broadening=true` becomes 1.0. A non-object direct HF+DMFT options argument
  becomes the default options. The outer task boundary does reject malformed
  options containers; the provider's direct normalizer still does not.
- Dataset `restricted="false"` becomes true.
- A string spin `"0"` is accepted as a lossless task alias, but preparation
  compares it with integer zero before normalization and infers unrestricted.

Priority: high for fields changing a calculation, lower for print verbosity.
Normalize once before inference. A second universal validation framework is
unnecessary; finish the existing owner contracts and shared scalar helpers.

### A07 — The new allowlist blocks aliases the parser still supports

Locations: `pyscf_agent/backend/parsing.py:92`, `:178`, `:366`.

`density_fit`, `df`, `auxbasis`, `df_auxbasis`, and
`density_fitting_auxbasis` are consumed by existing alias loops but absent
from `_TASK_INPUT_FIELDS`. Requests using them now fail as unknown fields.
This is a regression introduced by the preceding cleanup, not a desirable
rejection of misspelled options.

Priority: high compatibility fix. Derive accepted aliases and their
canonicalization from one declaration, or deliberately remove/version an
alias. Avoid an independently maintained allowlist alongside parser logic.

### A08 — Model reference runtime is still fixed in the implementation

Locations: `pyscf_agent/backend/execution.py:280`;
`pyscf_agent/backend/model_hamiltonian/solver.py:152`, `:570`.

The model execution dispatch passes solver options but no task runtime.
Its reference builder hardcodes `conv_tol=1e-10` and `max_cycle=200`.
Finite-model MP2/CCSD/FCI/block2 reference calculations therefore do not
consume the TaskSpec's corresponding runtime values. DMET has its own
separate provider controls and is not covered by this statement.

The UHF builder tries default, even/odd, edge-separated, and uniform density
guesses, then selects the lowest converged result. Failed guesses are silently
omitted. Even/odd site numbering is a particular initial-guess heuristic, not
a graph-independent antiferromagnetic sublattice construction.

Priority: high for ignored runtime settings. Preserve bounded multi-start as
an explicit reference policy, record every attempt/error, and connect its
controls to the model task/provider contract.

### A09 — Study cycle recovery can be ineffective for shorthand requests

Locations: `computational_study_agent/adaptive/refinement.py:256`;
`pyscf_agent/backend/parsing.py:244`.

Recovery reads only nested `runtime.max_cycle`, substitutes 50 on errors,
and raises it to at least 200. With top-level `max_cycle=800`, it produces
both the unchanged top-level 800 and nested 200. The task parser then gives
the top-level field precedence, leaving the effective limit at 800.

Priority: high. Recover from the canonical TaskSpec and update its sole
runtime field. The hardcoded 200 floor is a policy choice that should be
explicit, not an implicit repair of invalid input.

### A10 — Some diagnostic substitutions still conceal evidence quality

Locations: `pyscf_agent/backend/correlation/molecular.py:167`, `:1011`;
`pyscf_agent/backend/model_hamiltonian/reference_diagnostics.py:224`.

When AO-density UNO construction fails, molecular diagnostics substitute the
sum of alpha/beta occupations at equal MO indices. The summary correctly says
`approximate`, but fractional-occupation scoring consumes those values without
checking that status. Independent alpha/beta orbital indices need not describe
the same spatial orbital.

The model natural-occupation summary clips all eigenvalues into `[0,2]` and
retains no raw occupations or out-of-bounds status. A constructed density with
occupations `[2.4,-0.4]` becomes `[2,0]` with zero fractionality. This erases an
actual diagnostic of nonphysical density.

Priority: high for evidence used in routing. Preserve raw values, numerical
validity, and applicability. Small rounding tolerance is different from
unbounded clipping; approximate occupations must not stand in for valid NOONs.

### A11 — More than eight roots lose ambiguity detection

Locations: `computational_study_agent/adaptive/state_tracking.py:183`, `:309`.

Up to eight roots use exhaustive assignment with a best/second-best margin.
Larger root sets use greedy assignment and return no margin. Two cases with
identical root signatures require review at two roots, but are marked tracked
with approximately unit confidence at nine roots. The missing margin is not
treated as missing ambiguity evidence.

Priority: high for state tracking/continuation. Use one scalable assignment
method and a separately defined ambiguity test for every supported root count.
This did not involve a real multi-root DMRG run.

### A12 — Entropy fallback assumes equal alpha/beta local probabilities

Locations: `pyscf_agent/providers/block2/driver.py:427`, `:562`.

The spin-free RDM fallback sets both single-occupation probabilities to
`n/2 - d`. It has no spin/applicability check. For a fully spin-polarized
single-orbital product state, the supplied spin-free data `n=1,d=0` produce
entropy `ln(2)`, whereas the product state's local density has entropy zero.
Spin-free occupations alone cannot distinguish that state from an equal-spin
mixture. The fallback's origin and errors are recorded, but its scientific
scope is not constrained.

Priority: high when non-singlet/SZ calculations use this fallback. Use
spin-resolved evidence or restrict the formula to a verified applicable
spin-symmetric case. This counterexample does not establish an error in the
previous singlet lutein calculation.

### A13 — Costing keeps a second block2 preset table

Location: `computational_study_agent/costing.py:357`.

Presets moved to the Registry, but costing still has a separate
`screening:200, balanced:500, high_accuracy:1000` map and defaults to ten
sweeps. Confirmed: the normalized screening solve uses eight sweeps and
high_accuracy uses twelve; both cost estimates use ten. Adaptive/final solve
budgets also need explicit scope in an estimate.

Priority: medium. Use the normalized provider configuration for the inputs to
the existing transparent cost proxy. This does not require a new estimator.

### A14 — Optional evidence generation still drops error reasons

Locations: `pyscf_agent/backend/one_particle_state.py:65`;
`pyscf_agent/backend/correlation/cas_execution.py:732`;
`pyscf_agent/backend/correlation/orbital_diagnostics.py:277`.

The previous change records artifact-write failures. Earlier evidence
generation can still return `None` or `{}` without a reason: capturing a
reference 1RDM, obtaining CAS natural occupations, or deriving amplitude
importance. Keeping a completed energy is appropriate. Losing the distinction
between not requested, unsupported, malformed, and failed is not.

Priority: medium. Reuse explicit unavailable/failed evidence records and keep
optional diagnostic failure separate from the energy result.

## Specific Settings Requiring Clear Ownership

| Setting | Current location/behavior | Assessment |
| --- | --- | --- |
| QH9 MD | Registry profile, dataset contracts, Planner, browser, and request-builder prompt each contain settings. `request_builder/llm.py:220` says to use QH9 for MD, while `constants.py:228` says to clarify generic MD. | The restricted QH9 implementation is a valid capability boundary. The duplicated defaults and contradictory prompt need consolidation. |
| Adaptive method/resource policy | Registry `defaults/study.py:80` maps levels to methods; limits include FCI sites 8, molecular FCI orbitals/electrons 6, CAS orbitals 14. Block2 automatic recovery cap is M=1024. | These are explicit default policy choices, not universal physical limits. Keep configurable scope and derive resource decisions from dimensions and the approved resource budget. Moving constants alone does not make a policy general. |
| Correlation and state-tracking thresholds | `correlation/molecular.py:17`, `:40`, `:646`; `adaptive/state_tracking.py:166`, `:309`. T1/D1, gap, weighted-score and confidence cutoffs are embedded in code. | Several are useful diagnostics; their applicability, units, provenance, and combined decision rules should be explicit. They should not be represented as universal scientific truth. |
| Explicit range size 1000 | `computational_study_agent/normalization.py:158` rejects a range longer than 1000 values. | This is a format-dependent expansion guard, not a total Study resource limit. A literal list or product of multiple shorter axes has a different path. Use a consistent plan-size/resource policy if this is meant to bound execution. |
| DMRG MPO algorithm | Both `driver.get_qc_mpo` calls (`block2/driver.py:495`, `:741`) omit `algo_type`. | The previously discussed Conventional algorithm has not been implemented here; execution still depends on the installed library default. No current-library default or performance was measured in this audit because block2 is unavailable locally. |
| Amarel environment | `resources/templates/server-slurm.ini:24`, `:35`; deployment defaults use the profile name `amarel`. | `your_node` was explicitly requested by the user and should be retained for their profile. For distribution, separate the Amarel example from a generic template; do not silently change the user's working configuration. |
| Finite implementation limits | Complete FCI diagonalization dimension 5000; registered Bloch mesh/path limits; unsupported DMET translated graphs; CAS/DF/root/NPDM capability restrictions. | Retain honest implementation limits and explicit rejection. Generality does not require pretending every solver or geometry is supported. |

The scan found no new lutein-named runtime routing branch. Named molecules
still appear in examples, benchmarks, and the preparation geometry library.

## Fallbacks That Should Usually Remain

- Bounded same-method retries preserve requested scientific choices and store
  the previous/retry task. Compatible checkpoint reuse is not a method change.
- Remote collection preserves receipts and prevents resubmission after an
  interruption. Its blanket `connection_interrupted` classification
  (`executor.py:630`) could distinguish data/merge failures from transport
  failures more precisely, but receipt preservation should remain.
- JSON-schema transport fallback and one contract-repair attempt preserve
  the same scientific request and still validate the result. The transport
  support test currently matches error text (`request_builder/llm.py:353`);
  explicit provider error classification would avoid caching an unrelated
  failure as lack of structured-output support.
- PySCF-supported aliases, serialized geometry formats, and old report
  extensions are compatibility behavior when they preserve meaning.
- AVAS/UNO candidate construction can fail without destroying a valid
  completed energy, provided it returns failed/unavailable candidate evidence
  and cannot invent an executable frontier active space.
- Periodic VBM display fallback explicitly records its source. A display
  reference should remain distinguishable from a computed chemical potential.
- Resource probing can leave preparation available while execution revalidates
  the selected profile. Unknown resource limits should remain visible.
- Optional narrative generation failure should not destroy numerical results;
  a short unavailable-analysis record would improve observability.

## Suggested Order

1. Fix A01, A02, A03 and A07: unknown evidence routing, default authorization,
   unapplied revisions, and the newly introduced alias regression.
2. Finish canonical input normalization across task preparation, adaptive
   options, browser controls, and provider-owned lists/booleans; fix A08/A09
   so accepted runtime settings reach the calculation and recovery path.
3. Fix diagnostic validity and applicability (A10–A12), then unify the cost
   inputs and optional-evidence failure records (A13/A14).
4. Consolidate QH9 prompt/profile definitions, document/version the explicit
   scientific policy, and separately benchmark the MPO algorithm choice.

None of these recommendations require another task manager, validation
framework, or transport-specific scientific planner.
