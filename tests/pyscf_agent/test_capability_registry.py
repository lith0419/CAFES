from __future__ import annotations

import ast
import importlib.util
import re
import tempfile
import unittest
from pathlib import Path

import computational_study_agent.postprocessing as postprocessing
import computational_study_agent.web_ui as study_web_ui
from computational_study_agent.adaptive.initial_scan import initial_scan_method_for_options
import pyscf_agent.pyscf_agent_cli as pyscf_agent_cli
import pyscf_agent.pyscf_agent_web_ui as agent_web_ui
from pyscf_agent.backend import artifacts as backend_artifacts
from pyscf_agent.backend.model_hamiltonian import operations as model_operations
from pyscf_agent.registry import platform as registry_platform
from pyscf_agent.providers.fcdmft.contracts import (
    GW_DMFT_ARRAYS_SCHEMA,
    GW_DMFT_RESULT_SCHEMA,
    HF_DMFT_ARRAYS_SCHEMA,
    HF_DMFT_RESULT_SCHEMA,
    PERIODIC_GW_ARRAYS_SCHEMA,
    PERIODIC_GW_RESULT_SCHEMA,
)
from pyscf_agent.registry import default_registry
from pyscf_agent.registry.platform import (
    artifact_kind_is_registered,
    artifact_payload_schema,
    capabilities_markdown,
    density_fitting_auxbasis_recommendations,
    public_registry_payload,
    result_analysis_output_contracts,
    registered_artifact_kinds,
    write_capabilities_markdown,
)


REPO_ROOT = Path(__file__).resolve().parents[2]


def _select_options(markup: str, select_id: str) -> list[str]:
    match = re.search(
        r'<select[^>]*id="{0}"[^>]*>(.*?)</select>'.format(re.escape(select_id)),
        markup,
        re.DOTALL,
    )
    if match is None:
        raise AssertionError('Missing select #{0}'.format(select_id))
    return re.findall(r'<option[^>]*value="([^"]+)"', match.group(1))


def _capability_ids(registry, namespace: str, **filters) -> list[str]:
    return [
        item.id
        for item in registry.capabilities(namespace=namespace, **filters)
    ]


def _option_ids(registry, option_set_id: str, **filters) -> list[str]:
    return [
        item.id
        for item in registry.option_values(option_set_id, **filters)
    ]


def _observable_ids(registry, namespace: str, **filters) -> list[str]:
    return [
        item.id
        for item in registry.observables(namespace=namespace, **filters)
    ]


class PlatformRegistryAuditTests(unittest.TestCase):
    def test_frozen_orbitals_are_owned_but_are_not_a_localization_method(self):
        from pyscf_agent.request_builder.llm import _llm_capability_payload
        registry = default_registry()
        owner = registry.module_for_capability('molecular.orbital_processing.frozen_orbitals')
        self.assertEqual(owner.module_id, 'molecular.orbital_processing')
        self.assertEqual(_llm_capability_payload()['supported_orbital_processing_methods'],
                         ['none', 'boys', 'pipek_mezey'])

    def setUp(self):
        self.registry = default_registry()

    def test_unified_registry_resolves_all_typed_contracts(self):
        self.assertEqual(self.registry.registry_issues(), ())
        expected_kinds = {
            'artifact',
            'binding',
            'capability',
            'gate',
            'module',
            'observable',
            'option_set',
            'parameter',
            'provider',
            'template',
        }
        self.assertEqual(
            expected_kinds,
            {entry.kind for entry in self.registry.entries_for()},
        )
        self.assertEqual(
            'embedding.method.dmet',
            self.registry.entry('embedding.method.dmet').entry_id,
        )
        self.assertEqual(
            'provider.block2',
            self.registry.entry('provider.block2', kind='provider').entry_id,
        )
        self.assertEqual(
            ('provider.libdmet',),
            self.registry.entry('embedding.backend.libdmet').contract.required_provider_ids,
        )
        self.assertEqual(default_registry().registry_issues(), ())
        self.assertEqual(public_registry_payload(self.registry)['schema'], 'pyscf-agent.registry.v3')
        self.assertIsNone(importlib.util.find_spec('pyscf_agent.capability_registry'))

    def test_options_parameters_and_observables_have_distinct_contracts(self):
        option_entries = self.registry.entries_for(kind='option_set')
        self.assertEqual(len(option_entries), 12)
        self.assertFalse(self.registry.entries_for(kind='value'))

        molecular_basis = self.registry.option_set('molecular_basis_sets')
        self.assertEqual(molecular_basis.option_set_id, 'options.molecular.basis')
        self.assertEqual(
            molecular_basis.values,
            tuple(item.id for item in self.registry.option_values('options.molecular.basis')),
        )
        self.assertNotIn('values', molecular_basis.to_dict())
        self.assertTrue(molecular_basis.contains('CC-PVDZ'))
        self.assertEqual(molecular_basis.canonical_id('CC-PVDZ'), 'cc-pvdz')

        periodic_basis = self.registry.option_set('options.periodic.basis')
        self.assertIn('gth-dzvp', periodic_basis.values)

        payload = public_registry_payload(self.registry)
        self.assertEqual(
            len([entry for entry in payload['entries'] if entry['kind'] == 'option_set']),
            12,
        )
        hopping = self.registry.entry('model_hamiltonian.parameter.t', kind='parameter')
        self.assertEqual(hopping.contract.value_type, 'number')
        self.assertEqual(hopping.contract.unit, 'energy')
        self.assertTrue(hopping.contract.sweepable)
        self.assertNotIn('scopes', hopping.contract.metadata)
        self.assertNotIn('values', hopping.contract.metadata)
        energy = self.registry.entry('molecular.observable.energy', kind='observable')
        self.assertEqual(energy.contract.result_fields, ('energy',))
        self.assertEqual(energy.contract.unit, 'Hartree')
        self.assertTrue(energy.contract.requestable_output)
        for duplicate_key in (
            'result_field',
            'result_fields',
            'result_shape',
            'unit',
            'requestable_output',
            'supports_plot',
        ):
            self.assertNotIn(duplicate_key, energy.contract.metadata)

    def test_capability_module_and_provider_relations_are_explicit(self):
        capability = self.registry.entry('study.workflow_option.block2_state_tracking')
        module = self.registry.entry('analysis.block2.state_tracking', kind='module')
        self.assertIn(capability.entry_id, module.contract.capability_ids)
        self.assertEqual(
            self.registry.module_for_capability(capability.entry_id).module_id,
            'analysis.block2.state_tracking',
        )

        dmrg_module = self.registry.entry('solver.block2.dmrg', kind='module')
        bindings = [
            entry.contract
            for entry in self.registry.entries_for(kind='binding')
            if entry.contract.module_id == dmrg_module.entry_id
        ]
        self.assertEqual(len(bindings), 1)
        self.assertEqual(bindings[0].provider_id, 'provider.block2')

        reference_guesses = [
            item.id
            for item in self.registry.capabilities(
                namespace='embedding.reference_density', backend_allowed=True
            )
        ]
        self.assertEqual(reference_guesses, ['pm', 'af', 'fm', 'cdw'])
        for strategy in reference_guesses:
            capability = self.registry.capability(
                strategy, namespace='embedding.reference_density'
            )
            self.assertEqual(
                capability.required_provider_ids,
                ('provider.libdmet',),
            )
            self.assertEqual(
                self.registry.module_for_capability(
                    capability.domain + '.' + capability.id
                ).module_id,
                'embedding.libdmet.reference_density.{0}'.format(strategy),
            )
            self.assertEqual(capability.metadata['initial_correlation_potential'], 'zero')

    def test_workflow_templates_are_non_limiting_registry_hints(self):
        adaptive = self.registry.entry('study.template.adaptive', kind='template')
        self.assertIsNotNone(adaptive)
        self.assertIn(
            'study.module.study.correlation_routing',
            adaptive.contract.required_module_ids,
        )
        self.assertIn(
            'study.module.analysis.block2.state_tracking',
            adaptive.contract.optional_module_ids,
        )
        self.assertIn('non_limiting', adaptive.tags)
        payload = self.registry.as_dict()
        self.assertEqual(payload['registry']['schema'], 'pyscf-agent.registry.v3')
        self.assertEqual(payload['registry']['validation']['status'], 'valid')
        self.assertIn('hierarchy', payload['registry'])

    def test_single_calculation_ui_options_are_registry_derived(self):
        self.assertEqual(
            tuple(_capability_ids(self.registry, 'task_type', backend_allowed=True)),
            tuple(value for value, _label, _label_key in agent_web_ui.TASK_FAMILY_OPTIONS),
        )
        self.assertEqual(
            tuple(_option_ids(self.registry, 'options.molecular.basis', backend_allowed=True)),
            agent_web_ui.BASIS_OPTIONS,
        )
        basis_options = tuple(_option_ids(
            self.registry, 'options.molecular.basis', backend_allowed=True
        ))
        for basis in (
            '6-311++g**',
            'cc-pvtz',
            'aug-cc-pvtz',
            'cc-pcvdz',
            'def2-tzvp',
            'def2-qzvppd',
            'ano-rcc',
            'lanl2dz',
        ):
            self.assertIn(basis, basis_options)
        self.assertEqual(
            tuple(_option_ids(self.registry, 'options.molecular.xc', backend_allowed=True)),
            agent_web_ui.XC_OPTIONS,
        )
        self.assertEqual(
            ('__off__', '') + tuple(_option_ids(
                self.registry, 'options.molecular.auxbasis', backend_allowed=True
            )),
            tuple(value for value, _label in agent_web_ui.AUXBASIS_OPTIONS),
        )
        self.assertEqual(
            tuple(_capability_ids(self.registry, 'molecular.method', backend_allowed=True)),
            tuple(value for value, _label in agent_web_ui.METHOD_OPTIONS),
        )
        self.assertIn('ccsd_t', tuple(value for value, _label in agent_web_ui.METHOD_OPTIONS))
        self.assertEqual(
            tuple(_capability_ids(self.registry, 'molecular.job', backend_allowed=True)),
            tuple(value for value, _label, _label_key in agent_web_ui.JOB_OPTIONS),
        )
        self.assertIn(
            'density_fitting',
            _capability_ids(
                self.registry, 'molecular.workflow_option', backend_allowed=True
            ),
        )
        self.assertEqual(
            density_fitting_auxbasis_recommendations(self.registry),
            agent_web_ui.DENSITY_FITTING_AUXBASIS_RECOMMENDATIONS,
        )
        self.assertEqual(
            tuple(_capability_ids(
                self.registry,
                'model_hamiltonian.solver',
                backend_allowed=True,
                ui_visible=True,
            )),
            tuple(value for value, _label in agent_web_ui.MODEL_SOLVER_OPTIONS),
        )
        self.assertIn('ccsd_t', tuple(value for value, _label in agent_web_ui.MODEL_SOLVER_OPTIONS))
        self.assertEqual(
            tuple(_capability_ids(
                self.registry, 'molecular.active_space_solver', backend_allowed=True
            )),
            tuple(value for value, _label in agent_web_ui.ACTIVE_SPACE_SOLVER_OPTIONS),
        )
        self.assertIn('block2_dmrg', tuple(value for value, _label in agent_web_ui.ACTIVE_SPACE_SOLVER_OPTIONS))
        self.assertEqual(
            ('none',) + tuple(
                item.id
                for item in self.registry.capabilities(
                    namespace='molecular.orbital_processing', backend_allowed=True
                )
                if item.metadata.get('localization_method') is True
            ),
            tuple(value for value, _label, _label_key in agent_web_ui.LOCALIZATION_METHOD_OPTIONS),
        )
        self.assertEqual(
            tuple(_capability_ids(
                self.registry, 'molecular.active_space', backend_allowed=True
            )),
            tuple(value for value, _label, _label_key in agent_web_ui.ACTIVE_SPACE_METHOD_OPTIONS),
        )
        requestable_outputs = tuple(
            item.id
            for item in self.registry.observables(
                namespace='molecular.observable',
                requestable=True,
                backend_allowed=True,
            )
        )
        self.assertEqual(requestable_outputs, registry_platform.SUPPORTED_ANALYSIS)
        self.assertEqual(registry_platform.DEFAULT_ANALYSIS, ('energy', 'homo_lumo', 'dipole'))
        self.assertNotIn('reference_energy', requestable_outputs)
        molecular_ui_outputs = tuple(
            item.id
            for item in self.registry.observables(
                namespace='molecular.observable',
                requestable=True,
                backend_allowed=True,
            )
            if item.metadata.get('assistant_ui_output', True)
        )
        self.assertEqual(molecular_ui_outputs, tuple(value for value, _label in agent_web_ui.OUTPUT_OPTIONS))
        self.assertNotIn('energy', molecular_ui_outputs)
        self.assertNotIn('homo_lumo', molecular_ui_outputs)
        self.assertNotIn('dipole', molecular_ui_outputs)
        model_ui_outputs = tuple(
            item.id
            for item in self.registry.observables(
                namespace='model_hamiltonian.observable',
                requestable=True,
                backend_allowed=True,
            )
            if item.metadata.get('assistant_ui_output', True)
        )
        self.assertEqual(model_ui_outputs, tuple(value for value, _label in agent_web_ui.MODEL_OUTPUT_OPTIONS))
        self.assertEqual(
            (
                'entanglement_diagnostics',
                'excited_states',
                'symmetry_analysis',
                'strong_correlation_diagnostics',
            ),
            model_ui_outputs,
        )
        self.assertNotIn('energy', model_ui_outputs)
        self.assertNotIn('gap', model_ui_outputs)

    def test_registry_metadata_stays_runtime_oriented(self):
        for entry in self.registry.entries_for():
            payload = entry.contract.to_dict()
            metadata = payload.get('metadata') or payload.get('constraints') or {}
            with self.subTest(entry=entry.entry_id):
                self.assertNotIn('ui_label', metadata)

    def test_periodic_capabilities_drive_backend_and_assistant_options(self):
        self.assertIn('periodic', _capability_ids(
            self.registry, 'task_type', backend_allowed=True
        ))
        self.assertNotIn('periodic', _capability_ids(
            self.registry, 'task_type', planner_allowed=True
        ))
        self.assertEqual(
            tuple(_capability_ids(self.registry, 'periodic.method', backend_allowed=True)),
            tuple(value for value, _label in agent_web_ui.PERIODIC_METHOD_OPTIONS),
        )
        self.assertEqual(
            tuple(_option_ids(self.registry, 'options.periodic.basis', backend_allowed=True)),
            tuple(value for value, _label in agent_web_ui.PERIODIC_BASIS_OPTIONS),
        )
        from pyscf.pbc.gto import basis as pbc_basis

        pyscf_periodic_basis_names = {
            Path(filename).stem
            for filename in pbc_basis.ALIAS.values()
        }
        self.assertEqual(
            set(_option_ids(self.registry, 'options.periodic.basis', backend_allowed=True)),
            pyscf_periodic_basis_names,
        )
        self.assertEqual(
            tuple(_option_ids(
                self.registry, 'options.periodic.pseudopotential', backend_allowed=True
            )),
            tuple(value for value, _label in agent_web_ui.PERIODIC_PSEUDO_OPTIONS),
        )
        from pyscf.pbc.gto import pseudo as pbc_pseudo

        pyscf_periodic_pseudopotential_names = {
            Path(filename).stem
            for filename in pbc_pseudo.ALIAS.values()
        } | {'gth-lda'}
        self.assertEqual(
            set(_option_ids(
                self.registry, 'options.periodic.pseudopotential', backend_allowed=True
            )),
            pyscf_periodic_pseudopotential_names,
        )
        self.assertEqual(
            tuple(_option_ids(self.registry, 'options.periodic.xc', backend_allowed=True)),
            tuple(value for value, _label in agent_web_ui.PERIODIC_XC_OPTIONS),
        )
        self.assertEqual(
            tuple(_option_ids(
                self.registry, 'options.periodic.density_fitting', backend_allowed=True
            )),
            tuple(value for value, _label in agent_web_ui.PERIODIC_DENSITY_FITTING_OPTIONS),
        )
        self.assertEqual(
            tuple(_option_ids(
                self.registry, 'options.periodic.kpoint_scheme', backend_allowed=True
            )),
            tuple(value for value, _label in agent_web_ui.PERIODIC_KPOINT_SCHEME_OPTIONS),
        )
        self.assertEqual(
            tuple(_option_ids(
                self.registry, 'options.periodic.band_path_mode', backend_allowed=True
            )),
            tuple(value for value, _label, _label_key in agent_web_ui.PERIODIC_BAND_PATH_MODE_OPTIONS),
        )
        self.assertEqual(
            tuple(_option_ids(
                self.registry, 'options.periodic.smearing', backend_allowed=True
            )),
            tuple(value for value, _label in agent_web_ui.PERIODIC_SMEARING_OPTIONS),
        )
        self.assertEqual(
            tuple(_option_ids(
                self.registry, 'options.periodic.exxdiv', backend_allowed=True
            )),
            tuple(value for value, _label in agent_web_ui.PERIODIC_EXXDIV_OPTIONS),
        )
        for pseudo_name in ('gth-pade', 'gth-pbe'):
            pseudo = self.registry.option_value(
                'options.periodic.pseudopotential', pseudo_name
            )
            self.assertTrue(pseudo.constraints.get('general'))
            self.assertEqual(pseudo.constraints.get('element_range'), 'H-Rn')
        self.assertEqual(
            ('energy', 'band_gap', 'fermi_energy', 'band_structure'),
            tuple(_observable_ids(
                self.registry, 'periodic.observable', backend_allowed=True
            )[:4]),
        )
        self.assertEqual(
            registry_platform.SUPPORTED_PERIODIC_ANALYSIS,
            ('energy', 'band_gap', 'fermi_energy', 'band_structure'),
        )
        self.assertNotIn('periodic_structure', registry_platform.SUPPORTED_PERIODIC_ANALYSIS)
        contracts = result_analysis_output_contracts(self.registry)
        self.assertIn('fermi_energy', contracts)
        self.assertIn('periodic_structure', contracts)
        self.assertIn('periodic_task_input', contracts['periodic_structure']['artifact_kinds'])
        self.assertIn('periodic_band_structure', contracts)
        self.assertFalse(contracts['periodic_band_structure']['compact_result'])
        self.assertFalse(contracts['periodic_orbitals']['compact_result'])
        self.assertEqual(_capability_ids(
            self.registry, 'periodic', planner_allowed=True
        ), [])
        for option_set_id in (
            'options.periodic.basis',
            'options.periodic.pseudopotential',
            'options.periodic.xc',
            'options.periodic.density_fitting',
            'options.periodic.kpoint_scheme',
            'options.periodic.band_path_mode',
            'options.periodic.smearing',
            'options.periodic.exxdiv',
        ):
            with self.subTest(option_set=option_set_id):
                self.assertEqual(self.registry.option_values(
                    option_set_id, planner_allowed=True
                ), [])

    def test_ccsd_t_aliases_are_registry_supported(self):
        self.assertTrue(self.registry.capability_is_allowed(
            'CCSD(T)', namespace='molecular.method', action='backend'
        ))
        self.assertTrue(self.registry.capability_is_allowed(
            'ccsd-t', namespace='molecular.method', action='planner'
        ))
        self.assertTrue(self.registry.capability_is_allowed(
            'CCSD(T)', namespace='model_hamiltonian.solver', action='backend'
        ))
        self.assertTrue(self.registry.capability_is_allowed(
            'ccsdt', namespace='model_hamiltonian.solver', action='planner'
        ))
        self.assertEqual(self.registry.canonical_capability_id(
            'CCSD(T)', namespace='molecular.method'
        ), 'ccsd_t')
        self.assertEqual(self.registry.canonical_capability_id(
            'ccsdt', namespace='model_hamiltonian.solver'
        ), 'ccsd_t')
        contracts = result_analysis_output_contracts(self.registry)
        self.assertIn('triples_correction', contracts)
        self.assertIn('ccsd_t', contracts['triples_correction']['source_capabilities'])

    def test_model_fci_aliases_are_registry_supported(self):
        self.assertTrue(self.registry.capability_is_allowed(
            'full ci', namespace='model_hamiltonian.solver', action='backend'
        ))
        self.assertTrue(self.registry.capability_is_allowed(
            'exact diagonalization', namespace='model_hamiltonian.solver', action='planner'
        ))
        self.assertEqual(self.registry.canonical_capability_id(
            'ED', namespace='model_hamiltonian.solver'
        ), 'fci')

    def test_ambiguous_method_and_solver_words_are_not_registry_aliases(self):
        for value in ('rhf', 'uhf', 'cc', 'coupled cluster', 'coupled-cluster'):
            with self.subTest(value=value):
                self.assertIsNone(self.registry.canonical_capability_id(
                    value, namespace='molecular.method'
                ))
        for value in ('band', 'bands'):
            with self.subTest(value=value):
                self.assertIsNone(self.registry.canonical_capability_id(
                    value, namespace='model_hamiltonian.solver'
                ))

    def test_registry_declares_reference_and_post_cas_request_contracts(self):
        casscf = self.registry.capability('casscf', namespace='molecular.method')
        policy = casscf.metadata['reference_policy']
        self.assertEqual(policy['closed_shell_default'], 'rhf')
        self.assertEqual(policy['open_shell_default'], 'rohf')
        self.assertEqual(policy['unrestricted'], 'uhf')
        self.assertEqual(
            casscf.metadata['request_contract']['requires_active_space'],
            ['ncas', 'nelecas', 'approved'],
        )

        sc_nevpt2 = self.registry.capability(
            'sc_nevpt2', namespace='molecular.workflow_option'
        )
        contract = sc_nevpt2.metadata['request_contract']
        self.assertEqual(contract['requires_methods'], ['casci', 'casscf'])
        self.assertTrue(contract['requires_active_space_approval'])
        self.assertEqual(contract['allowed_references'], ['rhf', 'rohf'])
        self.assertEqual(contract['allowed_roots'], [0])

    def test_orbital_processing_outputs_are_registry_declared(self):
        boys = self.registry.capability(
            'boys', namespace='molecular.orbital_processing'
        )
        self.assertEqual(boys.metadata['result_field'], 'orbital_processing')
        self.assertIn('localization_status', boys.metadata['summary_fields'])
        self.assertIn('localized_orbital_shape', boys.metadata['summary_fields'])
        self.assertEqual(boys.metadata['preview_tables']['orbital_table']['limit'], 8)
        self.assertEqual(boys.metadata['artifact_kinds'], ['orbital_processing', 'orbital_summary_table'])
        self.assertEqual(boys.metadata['supported_scopes'], ['analysis', 'active_space'])
        self.assertEqual(
            boys.metadata['active_space_execution']['solvers'],
            ['block2_dmrg'],
        )

        block2 = self.registry.capability(
            'block2_dmrg', namespace='molecular.active_space_solver'
        )
        self.assertEqual(block2.metadata['orbital_ordering'], ['canonical', 'fiedler', 'manual'])
        self.assertEqual(block2.metadata['active_orbital_localization'], ['boys', 'pipek_mezey'])
        block2_module = self.registry.module('solver.block2.dmrg')
        self.assertEqual(
            block2_module.optional_configuration['properties']['orbital_ordering']['enum'],
            ['canonical', 'fiedler', 'manual'],
        )

        contracts = result_analysis_output_contracts(self.registry)
        self.assertIn('orbital_processing', contracts)
        self.assertIn('boys', contracts['orbital_processing']['source_capabilities'])
        self.assertIn('orbital_processing', contracts['orbital_processing']['artifact_kinds'])
        self.assertIn('orbital_summary_table', contracts['orbital_processing']['artifact_kinds'])
        self.assertIn('localized_orbital_shape', contracts['orbital_processing']['summary_fields'])

    def test_molecular_observable_outputs_are_registry_declared(self):
        energy = self.registry.observable('energy', namespace='molecular.observable')
        dipole = self.registry.observable('dipole', namespace='molecular.observable')
        homo_lumo = self.registry.observable('homo_lumo', namespace='molecular.observable')

        self.assertEqual(energy.result_fields, ('energy',))
        self.assertEqual(energy.unit, 'Hartree')
        self.assertTrue(energy.requestable_output)
        self.assertFalse(energy.metadata['assistant_ui_output'])
        self.assertEqual(dipole.result_fields, ('dipole',))
        self.assertEqual(dipole.result_shape, 'vector3')
        self.assertEqual(homo_lumo.result_fields, ('homo', 'lumo', 'gap'))

        contracts = result_analysis_output_contracts(self.registry)
        self.assertIn('dipole', contracts)
        self.assertIn('homo_lumo', contracts['homo']['source_capabilities'])
        self.assertIn('homo_lumo', contracts['lumo']['source_capabilities'])
        self.assertIn('homo_lumo', contracts['gap']['source_capabilities'])
        self.assertIn('Debye', contracts['dipole']['units'])
        self.assertIn('Hartree', contracts['energy']['units'])

    def test_molecular_workflow_outputs_are_registry_declared(self):
        contracts = result_analysis_output_contracts(self.registry)

        self.assertIn('active_space', contracts)
        self.assertIn('active_space', contracts['active_space']['source_capabilities'])
        self.assertIn('active_space_summary_table', contracts['active_space']['artifact_kinds'])
        self.assertIn('active_space_audit', contracts['active_space']['artifact_kinds'])
        self.assertIn('active_space_audit_table', contracts['active_space']['artifact_kinds'])
        self.assertIn('cas_result', contracts)
        self.assertIn('casci', contracts['cas_result']['source_capabilities'])
        self.assertIn('casscf', contracts['cas_result']['source_capabilities'])
        self.assertIn('post_cas_results', contracts)
        self.assertIn('sc_nevpt2', contracts['post_cas_results']['source_capabilities'])
        self.assertIn('sc_nevpt2', contracts['post_cas_results']['artifact_kinds'])
        sc_nevpt2 = self.registry.capability(
            'sc_nevpt2', namespace='molecular.workflow_option'
        )
        self.assertTrue(any('root must be 0' in limitation for limitation in sc_nevpt2.limitations))
        self.assertIn('correlation_diagnostics', contracts)
        self.assertIn('correlation_diagnostics', contracts['correlation_diagnostics']['artifact_kinds'])
        self.assertIn('molecular_correlation_risk', contracts['correlation_diagnostics']['summary_fields'])
        self.assertNotIn('physics_score', contracts['correlation_diagnostics']['summary_fields'])
        self.assertNotIn('solver_stress_score', contracts['correlation_diagnostics']['summary_fields'])
        self.assertIn('scf_stability', contracts)
        self.assertIn('scf_stability', contracts['scf_stability']['artifact_kinds'])

    def test_runtime_compact_result_keys_are_registry_derived(self):
        keys = backend_artifacts.compact_result_keys()

        self.assertIn('dipole', keys)
        self.assertIn('active_space', keys)
        self.assertIn('cas_result', keys)
        self.assertIn('post_cas_results', keys)
        self.assertIn('final_energy', keys)
        self.assertIn('final_method', keys)
        self.assertIn('quality_checks', keys)
        self.assertIn('correlation_diagnostics', keys)
        self.assertIn('scf_stability', keys)
        self.assertIn('strong_correlation_diagnostics', keys)
        self.assertIn('periodic_numerics', keys)
        self.assertIn('band_path', keys)
        self.assertIn('band_structure_status', keys)
        self.assertIn('dmft_result', keys)
        self.assertIn('mean_field_energy', keys)
        self.assertNotIn('periodic_band_structure', keys)
        self.assertNotIn('periodic_orbitals', keys)
        self.assertNotIn('density_mean', keys)
        self.assertNotIn('raw_scf_output', keys)

    def test_cli_result_listing_uses_registry_contracts(self):
        text = pyscf_agent_cli.format_human_report({
            'execution_status': 'succeeded',
            'structured_results': {
                'converged': True,
                'correlation_diagnostics': {'warnings': ['small_homo_lumo_gap']},
                'active_space': {'ncas': 2, 'nelecas': 2},
            },
        })

        self.assertIn('correlation_diagnostics:', text)
        self.assertIn('active_space:', text)

    def test_model_hamiltonian_observable_boundary_is_registry_visible(self):
        planner_observables = {
            item.id for item in self.registry.observables(
                namespace='model_hamiltonian.observable', planner_allowed=True
            )
        }
        backend_observables = {
            item.id for item in self.registry.observables(
                namespace='model_hamiltonian.observable', backend_allowed=True
            )
        }

        self.assertTrue({
            'energy',
            'strong_correlation_diagnostics',
        }.issubset(planner_observables))
        self.assertTrue({
            'energy',
            'strong_correlation_diagnostics',
        }.issubset(backend_observables))
        for internal_observable in (
            'gap',
            'density',
            'double_occupancy',
            'spin_correlation',
            'charge_correlation',
            'mean_field_gap',
            'frontier_orbital_degeneracy',
            'natural_occupations',
            'max_double_excitation_amplitude',
        ):
            with self.subTest(internal_observable=internal_observable):
                self.assertIsNone(self.registry.observable(
                    internal_observable, namespace='model_hamiltonian.observable'
                ))

        energy = self.registry.observable('energy', namespace='model_hamiltonian.observable')
        self.assertTrue(energy.requestable_output)
        self.assertFalse(energy.metadata['assistant_ui_output'])
        energy_per_site = self.registry.observable(
            'energy_per_site', namespace='model_hamiltonian.observable'
        )
        self.assertEqual(energy_per_site.requires_solvers, ('dmet',))
        self.assertIn('line_plot', energy_per_site.supports_plot)

        strong_correlation = self.registry.observable(
            'strong_correlation_diagnostics', namespace='model_hamiltonian.observable'
        )
        self.assertEqual(strong_correlation.result_fields, ('strong_correlation_diagnostics',))
        self.assertEqual(strong_correlation.result_shape, 'diagnostic_summary')
        self.assertEqual(
            strong_correlation.requires_solvers,
            ('mp2', 'ccsd', 'ccsd_t', 'fci', 'block2_dmrg', 'dmet'),
        )
        self.assertTrue(strong_correlation.requestable_output)
        self.assertIn('energy_per_site', strong_correlation.metadata['comparison_summary_fields'])
        self.assertNotIn('correlation_energy', strong_correlation.metadata['comparison_summary_fields'])
        self.assertIn('mean_field_gap', strong_correlation.metadata['comparison_summary_fields'])
        self.assertIn('natural_occupation_fractionality', strong_correlation.metadata['comparison_summary_fields'])
        self.assertIn('max_double_excitation_amplitude', strong_correlation.metadata['comparison_summary_fields'])
        self.assertIn('mean_double_occupancy', strong_correlation.metadata['comparison_summary_fields'])
        self.assertIn('nearest_neighbor_spin_correlation', strong_correlation.metadata['comparison_summary_fields'])
        self.assertIn('nearest_neighbor_charge_correlation', strong_correlation.metadata['comparison_summary_fields'])
        self.assertEqual(
            set(strong_correlation.metadata['internal_quantities']),
            {
                'gap',
                'density',
                'double_occupancy',
                'spin_correlation',
                'charge_correlation',
                'mean_field_gap',
                'frontier_orbital_degeneracy',
                'natural_occupations',
                'max_double_excitation_amplitude',
                'local_magnetization',
                'sublattice_order_parameters',
                'nearest_neighbor_one_body_coherence',
            },
        )

        contracts = result_analysis_output_contracts(self.registry)
        self.assertIn('strong_correlation_diagnostics', contracts)
        comparison_fields = contracts['strong_correlation_diagnostics']['comparison_summary_fields']
        self.assertIn('final_energy', comparison_fields)
        self.assertNotIn('correlation_energy', comparison_fields)
        self.assertIn('mean_field_gap', comparison_fields)
        self.assertIn('natural_occupation_fractionality', comparison_fields)
        self.assertIn('max_double_excitation_amplitude', comparison_fields)
        self.assertIn('mean_double_occupancy', comparison_fields)
        self.assertIn('nearest_neighbor_spin_correlation', comparison_fields)
        self.assertIn('nearest_neighbor_charge_correlation', comparison_fields)

        postprocessing_metrics = self.registry.observables(
            namespace='postprocessing.metric', backend_allowed=True
        )
        self.assertEqual(
            [item.id for item in postprocessing_metrics],
            [
                'mean_double_occupancy',
                'nearest_neighbor_spin_correlation',
                'nearest_neighbor_charge_correlation',
                'sublattice_charge_imbalance',
                'staggered_magnetization',
                'mean_nearest_neighbor_one_body_coherence',
                'energy_over_abs_t',
                'energy_per_site_over_abs_t',
                'first_excitation_energy',
                'max_single_orbital_entropy',
                'mean_single_orbital_entropy',
                'max_bipartite_entanglement',
                'dmrg_spin_square',
            ],
        )
        for metric in postprocessing_metrics:
            with self.subTest(metric=metric.id):
                self.assertTrue(metric.comparison_field)
                if metric.source_fields == ('strong_correlation_diagnostics',):
                    self.assertEqual(metric.source_fields, ('strong_correlation_diagnostics',))
                    self.assertTrue(metric.metadata['diagnostic_name'])
                    self.assertIn('model_hamiltonian', metric.metadata['system_types'])
                elif metric.id in ('energy_over_abs_t', 'energy_per_site_over_abs_t'):
                    self.assertEqual(metric.unit, 'dimensionless')
                    self.assertEqual(metric.metadata['system_types'], ['model_hamiltonian'])
                else:
                    self.assertIn('molecular', metric.metadata['system_types'])
                    self.assertIn('model_hamiltonian', metric.metadata['system_types'])

        for observable in ('structure_factor',):
            with self.subTest(observable=observable):
                capability = self.registry.observable(
                    observable, namespace='model_hamiltonian.observable'
                )
                self.assertEqual(capability.status, 'planned')
                self.assertFalse(capability.planner_allowed)
                self.assertFalse(capability.backend_allowed)

        self.assertEqual(
            artifact_payload_schema('adaptive-mps-continuation-plan', self.registry),
            'pyscf-agent.mps-continuation-review.v1',
        )

    def test_backend_constants_are_registry_derived(self):
        self.assertEqual(
            tuple(_capability_ids(self.registry, 'task_type', backend_allowed=True)),
            registry_platform.SUPPORTED_TASK_TYPES,
        )
        self.assertEqual(
            tuple(_capability_ids(self.registry, 'molecular.method', backend_allowed=True)),
            registry_platform.SUPPORTED_METHODS,
        )
        self.assertEqual(
            tuple(_capability_ids(self.registry, 'molecular.job', backend_allowed=True)),
            registry_platform.SUPPORTED_JOBS,
        )
        self.assertEqual(
            tuple(_capability_ids(
                self.registry, 'model_hamiltonian.model', backend_allowed=True
            )),
            registry_platform.SUPPORTED_MODEL_HAMILTONIANS,
        )
        self.assertEqual(
            tuple(_capability_ids(
                self.registry, 'model_hamiltonian.solver', backend_allowed=True
            )),
            registry_platform.SUPPORTED_MODEL_SOLVERS,
        )

    def test_bloch_tight_binding_capabilities_are_backend_only_for_now(self):
        solver = self.registry.capability(
            'tight_binding', namespace='model_hamiltonian.solver'
        )

        self.assertIsNotNone(solver)
        self.assertTrue(solver.backend_allowed)
        self.assertFalse(solver.planner_allowed)
        self.assertEqual(solver.ui_visibility, 'hidden')
        self.assertEqual(solver.metadata['requires_representation'], 'bloch')
        self.assertEqual(solver.metadata['resource_limits']['max_kpoints'], 50000)
        self.assertEqual(solver.metadata['resource_limits']['max_dos_work_items'], 200000000)
        self.assertIn('U and V', ' '.join(solver.limitations))
        self.assertIn('tight_binding', registry_platform.SUPPORTED_MODEL_SOLVERS)
        self.assertNotIn(
            'tight_binding',
            _capability_ids(
                self.registry, 'model_hamiltonian.solver', planner_allowed=True
            ),
        )
        self.assertNotIn(
            'tight_binding',
            _capability_ids(
                self.registry, 'model_hamiltonian.solver', ui_visible=True
            ),
        )
        contracts = result_analysis_output_contracts(self.registry)
        self.assertFalse(contracts['bloch_band_structure']['compact_result'])
        self.assertFalse(contracts['bloch_dos']['compact_result'])
        self.assertFalse(contracts['bloch_kmesh']['compact_result'])

    def test_model_hamiltonian_builder_options_match_registry(self):
        builder_html = (REPO_ROOT / 'model_hamiltonian_ui' / 'index.html').read_text(encoding='utf-8')
        self.assertEqual(
            [
                item.id
                for item in self.registry.capabilities(
                    namespace='model_hamiltonian.model'
                )
                if item.ui_visibility != 'hidden'
            ],
            _select_options(builder_html, 'model-type'),
        )
        self.assertEqual(
            _option_ids(self.registry, 'options.model_hamiltonian.builder_template'),
            _select_options(builder_html, 'template-1d') + _select_options(builder_html, 'template-2d'),
        )
        self.assertNotIn('value="holstein_hubbard"', builder_html)
        self.assertFalse(self.registry.capability_is_allowed(
            'holstein_hubbard', namespace='model_hamiltonian.model', action='ui'
        ))

    def test_planner_postprocessing_tools_match_registry(self):
        self.assertEqual(
            _capability_ids(self.registry, 'postprocessing.tool', backend_allowed=True),
            _select_options(study_web_ui.build_study_index_html(), 'custom-plot-tool'),
        )
        self.assertEqual(
            tuple(_capability_ids(
                self.registry, 'postprocessing.tool', backend_allowed=True
            )),
            postprocessing.SUPPORTED_PLOT_TOOLS,
        )
        dataset_collection = self.registry.capability(
            'collect_hamiltonian_dataset',
            namespace='postprocessing.action',
        )
        self.assertIsNotNone(dataset_collection)
        self.assertIn(
            'hamiltonian_collected_dataset_manifest',
            dataset_collection.metadata['artifact_kinds'],
        )
        dataset_generation = self.registry.capability(
            'generate_hamiltonian_dataset',
            namespace='postprocessing.action',
        )
        self.assertIsNotNone(dataset_generation)
        self.assertIn(
            'hamiltonian_dataset_generation_receipt',
            dataset_generation.metadata['artifact_kinds'],
        )
        self.assertEqual(
            tuple(_capability_ids(
                self.registry,
                'postprocessing.action',
                backend_allowed=True,
            )),
            postprocessing.SUPPORTED_POSTPROCESSING_ACTIONS,
        )

    def test_planner_scan_modes_and_initial_strategies_match_registry(self):
        html = study_web_ui.build_study_index_html()
        self.assertEqual(
            _capability_ids(self.registry, 'study.mode', backend_allowed=True),
            _select_options(html, 'study-mode'),
        )
        self.assertEqual(
            _capability_ids(
                self.registry, 'study.adaptive_initial_scan', backend_allowed=True
            ),
            _select_options(html, 'adaptive-initial-scan-strategy'),
        )
        static_mode = self.registry.capability('static', namespace='study.mode')
        adaptive_mode = self.registry.capability('adaptive', namespace='study.mode')
        self.assertEqual(
            static_mode.metadata['supported_system_types'],
            ['molecular', 'model_hamiltonian'],
        )
        self.assertEqual(adaptive_mode.metadata['supported_system_types'], ['molecular'])
        self.assertEqual(
            self.registry.canonical_capability_id(
                'cheap_diagnostics_mp2', namespace='study.adaptive_initial_scan'
            ),
            None,
        )
        self.assertEqual(
            self.registry.canonical_capability_id(
                'ccsd_spotcheck', namespace='study.adaptive_initial_scan'
            ),
            None,
        )
        self.assertEqual(
            self.registry.canonical_capability_id(
                'mp2+ccsd', namespace='study.adaptive_initial_scan'
            ),
            None,
        )
        for strategy in self.registry.capabilities(
            namespace='study.adaptive_initial_scan', backend_allowed=True
        ):
            with self.subTest(strategy=strategy.id):
                self.assertIsInstance(strategy.metadata.get('execution_method'), str)
                self.assertEqual(strategy.metadata.get('supported_system_types'), ['molecular'])
                self.assertNotIn('solver', strategy.metadata)
                self.assertNotIn('solvers', strategy.metadata)
                probe_strategy = self.registry.capability(
                    strategy.id, namespace='molecular.active_space_probe_strategy'
                )
                self.assertEqual(strategy.metadata, probe_strategy.metadata)
                self.assertEqual(
                    initial_scan_method_for_options(
                        {'initial_scan_strategy': strategy.id}, system_type='molecular'
                    ),
                    strategy.metadata['execution_method'],
                )
                ui_strategy = study_web_ui.ADAPTIVE_INITIAL_SCAN_STRATEGIES[strategy.id]
                self.assertEqual(
                    ui_strategy['execution_method'], strategy.metadata['execution_method']
                )
                self.assertEqual(
                    ui_strategy['refinement_method'], strategy.metadata.get('refinement_method', '')
                )
                self.assertEqual(
                    ui_strategy['refinement_triggers'], strategy.metadata.get('refinement_triggers', [])
                )
        continuation = self.registry.capability(
            'continuation_restart', namespace='study.workflow_option'
        )
        self.assertIsNotNone(continuation)
        self.assertEqual(continuation.metadata['activation'], 'explicit_analyze_results')
        self.assertEqual(
            continuation.metadata['adaptive_execution_policy'],
            'independent_cases_no_cross_case_1rdm_reuse',
        )
        job_lifecycle = self.registry.capability(
            'task_job_lifecycle', namespace='study.workflow_option'
        )
        self.assertEqual(job_lifecycle.metadata['local_submission_mode'], 'immediate')
        self.assertEqual(
            job_lifecycle.metadata['status_boundary']['task_report_execution_status'],
            'scientific_task_outcome',
        )

    def test_assistant_and_study_auto_strategy_is_scf_first(self):
        strategy = self.registry.capability(
            'auto', namespace='molecular.active_space_probe_strategy'
        )

        self.assertIsNotNone(strategy)
        self.assertEqual(strategy.metadata['execution_method'], 'hf')
        self.assertEqual(strategy.metadata['refinement_method'], 'mp2')
        self.assertEqual(
            strategy.metadata['refinement_triggers'],
            [
                'active_space_candidate_not_review_ready',
                'scf_instability',
                'frontier_orbital_degeneracy',
                'near_correlation_decision_boundary',
            ],
        )
        self.assertEqual(
            self.registry.capability(
                'auto', namespace='study.adaptive_initial_scan'
            ).metadata['execution_method'],
            'hf',
        )

    def test_adaptive_policies_are_registry_owned(self):
        routing = self.registry.capability('adaptive_routing', namespace='study.policy')
        continuity = self.registry.capability('scan_path_continuity', namespace='study.policy')

        self.assertIsNotNone(routing)
        self.assertIsNotNone(continuity)
        self.assertEqual(routing.metadata['resource_limits']['max_fci_sites'], 8)
        self.assertEqual(routing.metadata['molecular_method_policy']['moderate'], 'ccsd')
        self.assertEqual(continuity.metadata['energy_residual_hartree'], 0.005)
        self.assertEqual(continuity.metadata['molecular_method_levels']['casscf'], 4)

    def test_dmet_registry_owns_defaults_and_resolution_policies(self):
        dmet = self.registry.capability('dmet', namespace='model_hamiltonian.solver')
        options = dmet.metadata['solver_options']

        self.assertTrue(options['interacting_bath']['default'])
        self.assertEqual(options['max_iterations']['default'], 50)
        self.assertEqual(
            options['execution_mode']['resolution_policy'],
            'translational_if_eligible_else_finite_graph',
        )
        self.assertNotIn('default', options['execution_mode'])
        self.assertNotIn('default', options['fragment_definition'])

    def test_artifact_contracts_cover_core_and_generated_artifacts(self):
        registered = set(registered_artifact_kinds(self.registry))
        for kind in (
            'structured_results',
            'job-state',
            'task-report',
            'workflow-configuration',
            'workflow-provenance',
            'workflow-execution-trace',
            'gate-configuration',
            'gate-provenance',
            'gate-execution-trace',
            'execution-receipt',
            'correlation_diagnostics',
            'one_particle_state',
            'active_space_audit',
            'sc_nevpt2',
            'strong_correlation_diagnostics',
            'model_bloch_kmesh_tsv',
            'periodic_scf_summary',
            'cost-estimate',
            'study-state',
            'adaptive-decision-log',
            'postprocess-plot-data-json',
        ):
            with self.subTest(kind=kind):
                self.assertIn(kind, registered)
                self.assertTrue(artifact_kind_is_registered(kind, self.registry))
        self.assertTrue(artifact_kind_is_registered('observable-energy', self.registry))
        self.assertFalse(artifact_kind_is_registered('unregistered-output', self.registry))
        self.assertEqual(
            artifact_payload_schema('active_space_audit', self.registry),
            'pyscf-agent.active-space-audit.v1',
        )
        self.assertEqual(
            artifact_payload_schema('adaptive-study-report', self.registry),
            'pyscf-agent.adaptive-study-report.v1',
        )
        self.assertEqual(
            artifact_payload_schema('execution-receipt', self.registry),
            'pyscf-agent.study-execution-receipt.v1',
        )
        self.assertEqual(
            artifact_payload_schema('workflow-configuration', self.registry),
            'pyscf-agent.workflow-configuration.v1',
        )
        self.assertEqual(
            artifact_payload_schema('workflow-execution-trace', self.registry),
            'pyscf-agent.workflow-execution-trace.v1',
        )
        self.assertEqual(
            artifact_payload_schema('gate-configuration', self.registry),
            'pyscf-agent.gate-configuration.v1',
        )
        self.assertEqual(
            artifact_payload_schema('gate-execution-trace', self.registry),
            'pyscf-agent.gate-execution-trace.v1',
        )
        for kind, schema in (
            ('hf_dmft_result', HF_DMFT_RESULT_SCHEMA),
            ('hf_dmft_arrays', HF_DMFT_ARRAYS_SCHEMA),
            ('periodic_gw_result', PERIODIC_GW_RESULT_SCHEMA),
            ('periodic_gw_arrays', PERIODIC_GW_ARRAYS_SCHEMA),
            ('gw_dmft_result', GW_DMFT_RESULT_SCHEMA),
            ('gw_dmft_arrays', GW_DMFT_ARRAYS_SCHEMA),
        ):
            with self.subTest(provider_artifact=kind):
                self.assertEqual(artifact_payload_schema(kind, self.registry), schema)
        typed_reference = self.registry.artifact_contract('artifact_reference')
        self.assertEqual(
            typed_reference.required_reference_fields,
            ('kind', 'path', 'size_bytes', 'mime_type', 'description'),
        )
        self.assertEqual(len(self.registry.artifact_contracts()), 22)
        self.assertEqual(
            self.registry.artifact_contract('block2_dmrg').binary_artifact_kinds,
            ('block2_dmrg_arrays',),
        )

        capability_artifact_kinds = {
            artifact_kind
            for entry in self.registry.entries_for()
            if hasattr(entry.contract, 'metadata')
            for artifact_kind in entry.contract.metadata.get('artifact_kinds', [])
        }
        self.assertEqual(capability_artifact_kinds - registered, set())

    def test_static_artifact_writer_kinds_are_registered(self):
        writer_paths = (
            REPO_ROOT / 'pyscf_agent' / 'backend' / 'execution.py',
            REPO_ROOT / 'pyscf_agent' / 'backend' / 'result_artifacts.py',
            REPO_ROOT / 'computational_study_agent' / 'executor.py',
            REPO_ROOT / 'computational_study_agent' / 'adaptive' / 'executor.py',
            REPO_ROOT / 'computational_study_agent' / 'postprocessing.py',
        )
        positional_kind_writers = {'write_binary_artifact', 'write_json_artifact',
                                   'write_result_json_artifact', 'write_text_artifact'}
        keyword_kind_writers = {'_write_json', '_write_text', '_artifact_ref'}
        emitted_kinds = set()

        for path in writer_paths:
            tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                function_name = (
                    node.func.id
                    if isinstance(node.func, ast.Name)
                    else node.func.attr
                    if isinstance(node.func, ast.Attribute)
                    else ''
                )
                kind_node = None
                if function_name in positional_kind_writers and len(node.args) >= 2:
                    kind_node = node.args[1]
                elif function_name in keyword_kind_writers:
                    kind_node = next(
                        (keyword.value for keyword in node.keywords if keyword.arg == 'kind'),
                        None,
                    )
                if isinstance(kind_node, ast.Constant) and isinstance(kind_node.value, str):
                    emitted_kinds.add(kind_node.value)

        self.assertGreater(len(emitted_kinds), 40)
        unregistered = sorted(
            kind for kind in emitted_kinds
            if not artifact_kind_is_registered(kind, self.registry)
        )
        self.assertEqual(unregistered, [])

    def test_planner_operation_names_match_registry(self):
        self.assertEqual(
            set(_capability_ids(
                self.registry, 'model_hamiltonian.operation', backend_allowed=True
            )),
            model_operations.MODEL_OPERATION_NAMES,
        )

    def test_model_parameter_scopes_match_operation_contract(self):
        def scoped(scope):
            return {item.id for item in self.registry.parameters(scope=scope)}

        self.assertEqual(scoped('site'), model_operations.SITE_PARAMETERS)
        self.assertEqual(scoped('bond'), model_operations.BOND_PARAMETERS)
        self.assertEqual(scoped('global'), model_operations.GLOBAL_PARAMETERS)
        self.assertEqual(
            {
                item.id for item in self.registry.parameters(
                    namespace='model_hamiltonian.parameter', planner_allowed=True
                )
            },
            {
                item.id for item in self.registry.parameters(
                    namespace='model_hamiltonian.parameter', sweepable=True
                )
            },
        )

    def test_capability_markdown_export_uses_runtime_registry(self):
        markdown = capabilities_markdown(self.registry)

        self.assertIn('# PySCF Agent Capability Registry', markdown)
        self.assertIn('| hubbard | executable | yes | yes |', markdown)
        self.assertIn('| holstein_hubbard | design_only | no | no |', markdown)
        self.assertIn('<!-- Generated from pyscf_agent.registry.', markdown)

        with tempfile.TemporaryDirectory() as tmpdir:
            output_path = Path(tmpdir) / 'capabilities.md'
            result = write_capabilities_markdown(str(output_path), self.registry)

            self.assertEqual(result, str(output_path))
            self.assertEqual(output_path.read_text(encoding='utf-8'), markdown)


if __name__ == '__main__':
    unittest.main()
