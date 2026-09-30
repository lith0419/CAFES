from __future__ import annotations

from typing import Any, Callable, Dict

from ..registry import default_registry


PreparedRequest = Dict[str, Any]
HttpPost = Callable[[str, Dict[str, Any], Dict[str, str], float], Dict[str, Any]]

LLM_BASE_URL_ENV = 'PYSCF_AGENT_LLM_BASE_URL'
LLM_API_KEY_ENV = 'PYSCF_AGENT_LLM_API_KEY'
LLM_MODEL_ENV = 'PYSCF_AGENT_LLM_MODEL'
LLM_TIMEOUT_ENV = 'PYSCF_AGENT_LLM_TIMEOUT'

_STRUCTURED_OUTPUT_SUPPORT: Dict[str, bool] = {}

SUPPORTED_REQUEST_FIELDS = (
    'atom',
    'basis',
    'method',
    'solver',
    'xc',
    'job',
    'unit',
    'charge',
    'spin',
    'symmetry',
    'restricted',
    'max_cycle',
    'conv_tol',
    'conv_tol_grad',
    'grid_level',
    'diis_space',
    'verbose',
    'outputs',
    'molecular_dynamics',
    'orbital_processing',
    'density_fitting',
    'active_space',
    'post_cas',
    'workflow',
    'request_summary',
    'request',
)

SYSTEM_PROMPT = '''
You normalize PySCF task requests.

Schema:
{
  "atom": string | null,
  "basis": string | null,
  "method": supported_methods item | null,
  "solver": {
    "name": supported_active_space_solvers item,
    "options": {
      "nroots": integer | null,
      "state_average_weights": number[] | null
    } | null
  } | null,
  "xc": string | null,
  "job": supported_jobs item | null,
  "unit": string | null,
  "charge": integer | null,
  "spin": integer | null,
  "symmetry": boolean | null,
  "restricted": boolean | null,
  "max_cycle": integer | null,
  "conv_tol": number | null,
  "conv_tol_grad": number | null,
  "grid_level": integer | null,
  "diis_space": integer | null,
  "verbose": integer | null,
  "outputs": string[] | null,
  "molecular_dynamics": {
    "profile": "qh9" | "qh9_relaxed_scf",
    "ensemble": "nve",
    "temperature_kelvin": number,
    "time_step_au": number,
    "steps": integer,
    "sample_stride": integer,
    "sample_offset": integer,
    "velocity_seed": integer,
    "ao_convention": "qh9",
    "store_velocities": boolean,
    "store_fock": boolean,
    "store_overlap": boolean
  } | null,
  "orbital_processing": {
    "enabled": boolean,
    "localization_method": supported_orbital_processing_methods item,
    "use_natural_orbitals": boolean
  } | null,
  "density_fitting": {
    "enabled": boolean,
    "auxbasis": one_of_supported_auxbasis | null,
    "apply_to": "scf"
  } | null,
  "active_space": {
    "enabled": boolean,
    "selection_method": supported_active_space_selection_methods item,
    "ncas": integer | null,
    "nelecas": integer | integer[] | string | null,
    "orbital_indices": integer[] | {"alpha": integer[], "beta": integer[]} | string | null,
    "approved": boolean
  } | null,
  "post_cas": {
    "sc_nevpt2": {
      "enabled": boolean,
      "root": 0,
      "density_fit": boolean
    }
  } | null,
  "workflow": {
    "modules": supported_workflow_modules item[],
    "dependency_policy": "auto" | "strict"
  } | null,
  "request_summary": string | null,
  "request": string | null,
  "missing_fields": string[],
  "clarification_questions": string[]
}

Rules:
- Interpret default-setting or geometry-generation permission from the user's intent, including negation. Return the chosen basis and any proposed coordinates explicitly; the application will not infer them from keywords or summaries.
- Clarification questions must be concise English shown directly to the user.
- Do not ask separately about low-risk defaults such as unit, charge, spin, symmetry, restricted, max_cycle, conv_tol, or verbose when they can use ordinary defaults.
- For MP2 active-space screening, unrestricted references are acceptable and often preferred for strong-correlation diagnostics; for final CASCI/CASSCF requests, leave restricted null/auto or set restricted=true unless the user explicitly requests unrestricted CAS or provides spin-resolved alpha/beta active orbitals.
- Treat DMRG-CASCI and DMRG-CASSCF as composite requests: method is casci/casscf and solver.name is block2_dmrg. Never silently replace block2_dmrg with the default FCI active-space solver.
- For molecular FCI, CASCI, or CASSCF with either the FCI or block2 active-space solver, put the requested number of low-energy roots in solver.options.nroots. For multi-root CASSCF, optional normalized solver.options.state_average_weights control state-averaged orbital optimization. All roots share the selected particle, spin, and configured spatial-symmetry sector.
- For Model Hamiltonian FCI or block2 DMRG, put the requested number of low-lying states in solver.options.nroots. These are targeted roots in one particle/spin sector, not a complete spectrum.
- If a DMRG-CASCI/CASSCF request does not include a user-approved active-space contract, leave approval false. The deterministic request builder will prepare an active-space probe before the final block2 calculation.
- Only set post_cas.sc_nevpt2.enabled=true for CASCI/CASSCF requests with an explicit approved active space; do not attach SC-NEVPT2 to MP2 screening. The current backend supports SC-NEVPT2 root=0 only.
- If the user explicitly authorizes default, standard, or draft structure generation and the molecule identity is clear enough, you may return a draft atom geometry using standard bond lengths/angles instead of asking for coordinates again.
- If information is missing, put only the truly blocking fields in missing_fields and ask at most 1-2 concise clarification_questions.
- Capabilities and allowed enum values are provided in the user payload; do not use values outside those lists.
- Select workflow modules only when the user explicitly requests one; automatic modules are added by the deterministic compiler.
- Follow capability_rules in the user payload for method, workflow-option, auxiliary-basis, and output constraints.
- Molecular dynamics requires an explicit profile: "qh9" uses the reference SCF tolerance 1e-13; "qh9_relaxed_scf" uses 1e-8. Match the user's requested accuracy. Clarify generic MD requests instead of assuming a QH9 profile. Sampling steps, stride and offset are explicit choices independent of the SCF profile.
- outputs must be selected from supported_outputs in the user payload.
- messages may contain only the most recent clarification turn, not the full chat history.
- Prefer the provided task_spec_seed as the authoritative accumulated state when it already contains structured values.
- request_summary should be a short cumulative summary of the user's goal and constraints across turns.
- request should keep the user's intent or extra preferences in concise form.
'''.strip()

EXECUTION_FEEDBACK_PROMPT = '''
You are reviewing the result of a PySCF agent run for the user.
Reply in English, concise and actionable.

Rules:
- Only reply when the user should change the next prompt or structured request.
- If the task succeeded and no user action is needed, return an empty response.
- If the task failed, was blocked, unconverged, or missed key information, say what to change in the next prompt.
- If an ActiveSpaceAudit candidate requires approval, tell the user to review the Active Space approval panel instead of asking them to manually set approved=true in chat.
- For an unavailable periodic basis or pseudopotential, recommend only replacements explicitly listed by the backend as verified compatible for all structure elements. Never infer alternatives from basis-set names or size.
- Distinguish requested low-energy FCI or DMRG roots from a complete many-body spectrum. Do not call a calculation incomplete when all requested roots were returned.
- `state_energies` and `excitation_energies` are result fields; the corresponding requestable output is `excited_states`.
- `dmrg_spin_square` is a derived postprocessing metric; the requestable numerical output is `symmetry_analysis`.
- Treat task_spec.solver as the executed solver. Do not report a conflicting solver copied from nested Model Hamiltonian input metadata.
- Keep active-space probes separate from the approved CAS calculation. A probe's SCF status is evidence for ActiveSpaceAudit and is not the final CAS task status.
- For CASSCF, `reference_converged=false` describes only the initial mean-field orbital guess when `reference_status.affects_task_status=false`. If CASSCF/DMRG and all targeted roots converged, treat the task as succeeded and mention the reference issue only as a limitation; do not ask for a rerun merely to clear that flag.
- Ground your advice in the execution result provided. Do not invent chemistry facts.
- Keep the answer within 3 short sentences.
'''.strip()

RESULT_ANALYSIS_PROMPT = '''
You are writing a formal result analysis for a PySCF agent run.
Reply in English, professional, concise, and grounded only in the provided execution result.

Output requirements:
- Use the following section titles exactly once when applicable:
    1. Calculation Overview
    2. Key Results
    3. Interpretation
    4. Reliability and Limitations
    5. Next Steps
- Use complete sentences, not chatty dialogue.
- If the task failed or was blocked, clearly state what prevented a valid chemistry conclusion.
- For an unavailable periodic basis or pseudopotential, mention only replacements explicitly listed by the backend as verified compatible for all structure elements. If none are listed, do not invent one.
- Do not invent chemistry facts not supported by the execution result.
- Respect registered output contracts appended by the runtime; registered result fields and artifact kinds are generated outputs.
- Distinguish requested low-energy FCI or DMRG roots from a complete many-body spectrum, and do not call a completed target-root calculation incomplete.
- Refer to requestable capabilities rather than generated result fields. In particular, use `excited_states` and `symmetry_analysis`, not `state_energies`, `excitation_energies`, or `dmrg_spin_square`.
- Keep active-space probe results separate from the approved CAS calculation. For CASSCF, an unconverged initial mean-field orbital guess is a limitation rather than a failed final result when `reference_status.affects_task_status=false` and the CASSCF/DMRG solver converged.
- Keep the full answer within 8 short paragraphs or bullet-like lines.
'''.strip()

LLM_RESPONSE_JSON_SCHEMA = {
    'name': 'pyscf_prepared_request',
    'strict': True,
    'schema': {
        'type': 'object',
        'additionalProperties': False,
        'properties': {
            'atom': {'type': ['string', 'null']},
            'basis': {'type': ['string', 'null']},
            'method': {'type': ['string', 'null']},
            'solver': {
                'type': ['object', 'null'],
                'additionalProperties': False,
                'properties': {
                    'name': {'type': 'string'},
                    'options': {
                        'type': ['object', 'null'],
                        'additionalProperties': False,
                        'properties': {
                            'nroots': {'type': ['integer', 'null'], 'minimum': 1},
                            'state_average_weights': {
                                'type': ['array', 'null'],
                                'items': {'type': 'number', 'minimum': 0},
                            },
                        },
                    },
                },
                'required': ['name', 'options'],
            },
            'xc': {'type': ['string', 'null']},
            'job': {'type': ['string', 'null']},
            'unit': {'type': ['string', 'null']},
            'charge': {'type': ['integer', 'null']},
            'spin': {'type': ['integer', 'null']},
            'symmetry': {'type': ['boolean', 'null']},
            'restricted': {'type': ['boolean', 'null']},
            'max_cycle': {'type': ['integer', 'null']},
            'conv_tol': {'type': ['number', 'null']},
            'conv_tol_grad': {'type': ['number', 'null']},
            'grid_level': {'type': ['integer', 'null']},
            'diis_space': {'type': ['integer', 'null']},
            'verbose': {'type': ['integer', 'null']},
            'outputs': {
                'type': ['array', 'null'],
                'items': {'type': 'string'},
            },
            'molecular_dynamics': {
                'type': ['object', 'null'],
                'additionalProperties': False,
                'properties': {
                    'profile': {'type': 'string', 'enum': list(default_registry().capability(
                        'molecular_dynamics', namespace='molecular.job',
                    ).metadata['profiles'])},
                    'ensemble': {'type': 'string', 'enum': ['nve']},
                    'temperature_kelvin': {'type': 'number', 'exclusiveMinimum': 0},
                    'time_step_au': {'type': 'number', 'exclusiveMinimum': 0},
                    'steps': {'type': 'integer', 'minimum': 1},
                    'sample_stride': {'type': 'integer', 'minimum': 1},
                    'sample_offset': {'type': 'integer', 'minimum': 0},
                    'velocity_seed': {'type': 'integer', 'minimum': 0},
                    'ao_convention': {'type': 'string', 'enum': ['qh9']},
                    'store_velocities': {'type': 'boolean'},
                    'store_fock': {'type': 'boolean'},
                    'store_overlap': {'type': 'boolean'},
                },
                'required': [
                    'ensemble', 'temperature_kelvin', 'time_step_au', 'steps',
                    'sample_stride', 'sample_offset', 'velocity_seed',
                    'ao_convention', 'store_velocities', 'store_fock',
                    'store_overlap',
                ],
            },
            'orbital_processing': {
                'type': ['object', 'null'],
                'additionalProperties': False,
                'properties': {
                    'enabled': {'type': 'boolean'},
                    'localization_method': {'type': 'string'},
                    'use_natural_orbitals': {'type': 'boolean'},
                },
                'required': ['enabled', 'localization_method', 'use_natural_orbitals'],
            },
            'density_fitting': {
                'type': ['object', 'null'],
                'additionalProperties': False,
                'properties': {
                    'enabled': {'type': 'boolean'},
                    'auxbasis': {'type': ['string', 'null']},
                    'apply_to': {'type': 'string'},
                },
                'required': ['enabled', 'auxbasis', 'apply_to'],
            },
            'active_space': {
                'type': ['object', 'null'],
                'additionalProperties': False,
                'properties': {
                    'enabled': {'type': 'boolean'},
                    'selection_method': {'type': 'string'},
                    'ncas': {'type': ['integer', 'null']},
                    'nelecas': {
                        'anyOf': [
                            {'type': 'integer'},
                            {'type': 'array', 'items': {'type': 'integer'}, 'minItems': 2, 'maxItems': 2},
                            {'type': 'string'},
                            {'type': 'null'},
                        ],
                    },
                    'orbital_indices': {
                        'anyOf': [
                            {'type': 'array', 'items': {'type': 'integer'}},
                            {
                                'type': 'object',
                                'additionalProperties': False,
                                'properties': {
                                    'alpha': {'type': 'array', 'items': {'type': 'integer'}},
                                    'beta': {'type': 'array', 'items': {'type': 'integer'}},
                                },
                                'required': ['alpha', 'beta'],
                            },
                            {'type': 'string'},
                            {'type': 'null'},
                        ],
                    },
                    'approved': {'type': 'boolean'},
                },
                'required': [
                    'enabled', 'selection_method', 'ncas', 'nelecas',
                    'orbital_indices', 'approved',
                ],
            },
            'post_cas': {
                'type': ['object', 'null'],
                'additionalProperties': False,
                'properties': {
                    'sc_nevpt2': {
                        'type': 'object',
                        'additionalProperties': False,
                        'properties': {
                            'enabled': {'type': 'boolean'},
                            'root': {'type': 'integer', 'minimum': 0, 'maximum': 0},
                            'density_fit': {'type': 'boolean'},
                        },
                        'required': ['enabled', 'root', 'density_fit'],
                    },
                },
                'required': ['sc_nevpt2'],
            },
            'workflow': {
                'type': ['object', 'null'],
                'additionalProperties': False,
                'properties': {
                    'modules': {
                        'type': 'array',
                        'items': {'type': 'string'},
                    },
                    'dependency_policy': {
                        'type': 'string',
                        'enum': ['auto', 'strict'],
                    },
                },
                'required': ['modules', 'dependency_policy'],
            },
            'request_summary': {'type': ['string', 'null']},
            'request': {'type': ['string', 'null']},
            'missing_fields': {
                'type': 'array',
                'items': {'type': 'string'},
            },
            'clarification_questions': {
                'type': 'array',
                'items': {'type': 'string'},
            },
        },
        'required': ['missing_fields', 'clarification_questions'],
    },
}
