from __future__ import annotations

from typing import Any, Callable, Dict, List


def build_embedding_families(
    item_factory: Callable[..., Any],
    *,
    capability_design_only: str,
    capability_planned: str,
) -> Dict[str, List[Any]]:
    _capability = item_factory
    CAPABILITY_PLANNED = capability_planned

    return {
        'embedding_providers': [
            _capability(
                'libdmet',
                'libDMET embedding provider',
                'embedding.backend',
                description='Provides localization/operator transforms and a Builder-backed single-band Hubbard DMET loop.',
                limitations=[
                    'Requires the optional libDMET, PySCF, SciPy, NumPy, and h5py dependencies.',
                    'Restricted and fixed-spin unrestricted references are supported; libDMET Sz means Nalpha-Nbeta.',
                    'The executable loop is ground-state only.',
                ],
                required_provider_ids=('provider.libdmet',),
                metadata={
                    'optional_dependency': 'libdmet',
                    'supported_stages': [
                        'reference',
                        'reference_density_initialization',
                        'localization',
                        'subspace_audit',
                        'operator_transform',
                        'dmet_bath',
                        'impurity_solution',
                        'correlation_potential_fit',
                        'dmet_self_consistency',
                    ],
                    'artifact_kinds': [
                        'embedding_reference',
                        'localized_subspace',
                        'localized_hamiltonian',
                        'correlated_subspace_audit',
                        'dmet_result',
                        'dmet_iteration_history',
                        'dmet_numerical_arrays',
                    ],
                },
            ),
            _capability(
                'fcdmft',
                'fcDMFT embedding provider',
                'embedding.backend',
                description=(
                    'Runs periodic GW and builds reviewable IAO-localized periodic '
                    'Hamiltonians for registered HF+DMFT and GW+DMFT workflows.'
                ),
                limitations=[
                    'Requires the optional fcDMFT, mpi4py, PySCF, NumPy, SciPy, and h5py dependencies.',
                    'Periodic GW and GW+DMFT currently require a restricted closed-shell DFT reference, GDF, and a zero-shift gamma-centered k mesh without smearing.',
                    'The current fcDMFT adapter does not expose GW or DMFT total-energy estimators.',
                ],
                required_provider_ids=('provider.fcdmft',),
                metadata={
                    'optional_dependency': 'fcdmft',
                    'supported_stages': [
                        'hf_reference',
                        'dft_reference',
                        'periodic_gw',
                        'localization',
                        'subspace_audit',
                        'operator_transform',
                        'interaction_transform',
                        'gw_double_counting',
                        'dmft_bath',
                        'impurity_solution',
                    ],
                    'artifact_kinds': [
                        'embedding_reference',
                        'localized_subspace',
                        'localized_hamiltonian',
                        'correlated_subspace_audit',
                        'hf_dmft_result',
                        'hf_dmft_arrays',
                        'hf_dmft_output_log',
                        'hf_dmft_checkpoint',
                        'periodic_gw_result',
                        'periodic_gw_arrays',
                        'periodic_gw_analytic_continuation',
                        'periodic_gw_imaginary_self_energy',
                        'gw_dmft_result',
                        'gw_dmft_arrays',
                        'gw_dmft_checkpoint',
                    ],
                },
            ),
        ],
        'embedding_reference_density_guesses': [
            _capability(
                'pm',
                '0 (Paramagnetic)',
                'embedding.reference_density',
                description='Use zero initial spin polarization in the requested particle sector.',
                required_provider_ids=('provider.libdmet',),
                metadata={
                    'initial_correlation_potential': 'zero',
                    'bias_kind': 'none',
                },
            ),
            _capability(
                'af',
                'Antiferromagnetic',
                'embedding.reference_density',
                description='Add a small staggered bias only to the initial UHF spin-density seed.',
                limitations=[
                    'Requires an unrestricted reference and a bipartite graph.',
                    'The bias preserves the requested Nalpha and Nbeta counts.',
                ],
                required_provider_ids=('provider.libdmet',),
                metadata={
                    'initial_correlation_potential': 'zero',
                    'bias_kind': 'staggered_spin_density',
                    'default_bias': 0.05,
                },
            ),
            _capability(
                'fm',
                'Ferromagnetic',
                'embedding.reference_density',
                description='Initialize UHF with the uniform spin polarization fixed by the requested Nalpha-Nbeta sector.',
                limitations=[
                    'Requires an unrestricted reference and nonzero Nalpha-Nbeta.',
                    'No independent uniform bias is added because that would change the requested particle sector.',
                ],
                required_provider_ids=('provider.libdmet',),
                metadata={
                    'initial_correlation_potential': 'zero',
                    'bias_kind': 'fixed_spin_sector',
                },
            ),
            _capability(
                'cdw',
                'Charge density wave',
                'embedding.reference_density',
                description='Add a small staggered sublattice charge bias, equal in both spins, only to the initial reference density.',
                limitations=[
                    'Requires a bipartite graph with both sublattices in the reference cell.',
                    'The centered bias preserves the requested Nalpha and Nbeta counts and adds no magnetization.',
                ],
                required_provider_ids=('provider.libdmet',),
                metadata={
                    'initial_correlation_potential': 'zero',
                    'bias_kind': 'staggered_charge_density',
                    'default_bias': 0.05,
                },
            ),
        ],
        'embedding_localization_methods': [
            _capability(
                'manual',
                'Manual localized subspace',
                'embedding.localization',
                description='Use user-supplied localized-orbital coefficients and correlated orbital indices.',
            ),
            _capability(
                'identity_sites',
                'Model site basis',
                'embedding.localization',
                description='Treat finite-cluster model sites or orbitals as the local basis without rotation.',
            ),
            _capability(
                'iao',
                'Intrinsic atomic orbitals',
                'embedding.localization',
                limitations=['Requires the optional libDMET provider.'],
                metadata={'minimal_basis_default': 'minao', 'include_pao': False},
            ),
            _capability(
                'iao_pao',
                'IAO plus projected atomic orbitals',
                'embedding.localization',
                limitations=['Requires the optional libDMET provider.'],
                metadata={'minimal_basis_default': 'minao', 'include_pao': True},
            ),
            _capability(
                'lowdin',
                'Lowdin localized orbitals',
                'embedding.localization',
                limitations=['Requires the optional libDMET provider.'],
            ),
            _capability(
                'wannier',
                'Wannier localized orbitals',
                'embedding.localization',
                CAPABILITY_PLANNED,
                limitations=['No Wannier backend is connected in the current workflow.'],
            ),
        ],
        'embedding_operations': [
            _capability(
                'define_correlated_subspace',
                'Define correlated subspace',
                'embedding.operation',
                metadata={
                    'approval_schema': 'pyscf-agent.correlated-subspace-audit.v1',
                    'fragment_contract': {
                        'canonical_membership': (
                            'fragments[].site_ids or fragments[].orbital_indices'
                        ),
                        'derived_size': 'length of canonical fragment membership',
                        'convenience_selectors': [
                            'impurity_size',
                            'impurity_shape',
                            'impurity_site_ids',
                        ],
                        'selector_rule': 'Choose primitive-cell membership, a cell block, or a finite-graph site partition.',
                        'impurity_shape_unit': 'primitive cells',
                        'impurity_size_unit': 'sites for graphs without multi-site cell metadata',
                    },
                    'approval_requires': [
                        'approved_by',
                        'approved_at',
                        'audit_artifact.path',
                        'localized_subspace_artifact.path',
                        'localized_hamiltonian_artifact.path',
                    ],
                },
            ),
            _capability(
                'transform_one_particle_operators',
                'Transform one-particle operators',
                'embedding.operation',
                description='Transform one-body operators and one-particle density matrices into a local basis.',
            ),
            _capability(
                'serialize_embedding_artifacts',
                'Serialize embedding artifacts',
                'embedding.operation',
                description='Write registered HDF5 numerical artifacts and JSON review records.',
            ),
            _capability(
                'build_interaction_tensor',
                'Build localized interaction tensor',
                'embedding.operation',
                limitations=[
                    'The executable implementation currently covers periodic density-fitted '
                    'unit-cell ERIs for the fcDMFT HF+DMFT workflow.',
                    'General molecular and arbitrary embedding-basis ERI transformations remain unconnected.',
                ],
                required_provider_ids=('provider.fcdmft', 'provider.libdmet'),
            ),
            _capability(
                'audit_builder_translation',
                'Audit Builder translation equivalence',
                'embedding.operation',
                limitations=[
                    'Requires primitive_cell, cell_index, and basis_index metadata from Model Builder.',
                    'A local site or bond edit invalidates representative-impurity reuse and routes DMET to finite-graph execution.',
                ],
            ),
            _capability(
                'build_dmet_bath',
                'Build DMET Schmidt bath',
                'embedding.operation',
                limitations=['Executable inside the registered libDMET Hubbard DMET composite module.'],
            ),
            _capability(
                'solve_dmet_impurity',
                'Solve DMET impurity problem',
                'embedding.operation',
                limitations=['FCI, CCSD, and block2 DMRG impurity solvers are registered.'],
            ),
            _capability(
                'fit_dmet_correlation_potential',
                'Fit DMET correlation potential',
                'embedding.operation',
            ),
            _capability(
                'iterate_dmet_self_consistency',
                'Iterate DMET self-consistency',
                'embedding.operation',
            ),
            _capability(
                'build_dmft_bath',
                'Build DMFT dynamical bath',
                'embedding.operation',
                limitations=[
                    'Executable only inside the registered fcDMFT HF+DMFT composite module.',
                ],
                required_provider_ids=('provider.fcdmft',),
            ),
            _capability(
                'gw_double_counting',
                'Build local GW double counting',
                'embedding.operation',
                description=(
                    'Construct the local GW self-energy and MO-to-local transform '
                    'consumed by the fcDMFT GW+DMFT loop.'
                ),
                required_provider_ids=('provider.fcdmft',),
            ),
        ],
        'embedding_methods': [
            _capability(
                'dmet',
                'Density matrix embedding theory',
                'embedding.method',
                limitations=[
                    'Restricted and fixed-spin unrestricted references are supported; nonzero spin currently uses one translated representative fragment.',
                    'Model Hamiltonians consume Builder sites and bonds directly; ab initio DMET remains design-only.',
                    'Translation-audited periodic Builder cells reuse one representative impurity; other graphs use an explicit finite-graph partition.',
                    'The Schmidt bath retains projected two-body interactions by default; a noninteracting-bath approximation remains available explicitly.',
                    'The result is ground-state only and reports both per-site energy and a lattice-total estimate.',
                ],
                required_provider_ids=('provider.libdmet',),
                metadata={
                    'impurity_solvers': ['fci', 'ccsd', 'block2_dmrg'],
                    'bath_modes': ['interacting', 'noninteracting'],
                    'default_bath_mode': 'interacting',
                    'initial_correlation_potential': 'zero',
                    'reference_density_guesses': ['pm', 'af', 'fm', 'cdw'],
                    'default_reference_density_guess': 'pm',
                    'fragment_contract': {
                        'canonical': 'fragments[].site_ids or fragments[].orbital_indices',
                        'convenience': ['impurity_size', 'impurity_shape', 'impurity_site_ids'],
                        'default': 'complete Builder primitive-cell membership',
                        'dmft_reuse': 'DMFT initially accepts one inequivalent impurity fragment.',
                    },
                    'artifact_kinds': [
                        'dmet_result',
                        'dmet_iteration_history',
                        'dmet_numerical_arrays',
                    ],
                },
            ),
            _capability(
                'gw',
                'Periodic G0W0',
                'embedding.method',
                limitations=[
                    'Requires a converged restricted closed-shell periodic DFT reference with GDF and more than one k point.',
                    'The executable contract currently supports Pade analytic continuation and integer occupations without smearing.',
                    'Reports quasiparticle energies and registered self-energy artifacts, not a total energy.',
                ],
                required_provider_ids=('provider.fcdmft',),
                metadata={
                    'reference_method': 'periodic_dft',
                    'result_schema': 'pyscf-agent.periodic-gw-result.v1',
                    'energy_available': False,
                    'artifact_kinds': [
                        'periodic_gw_result',
                        'periodic_gw_arrays',
                        'periodic_gw_output_log',
                        'periodic_gw_analytic_continuation',
                        'periodic_gw_mean_field_potential',
                        'periodic_gw_imaginary_self_energy',
                    ],
                },
            ),
            _capability(
                'hf_dmft',
                'Hartree-Fock plus DMFT',
                'embedding.method',
                limitations=[
                    'Requires a converged periodic HF reference.',
                    'IAO and IAO+PAO proposals can be generated automatically; execution still requires explicit approval.',
                    'Automatic preparation currently requires a zero-shift gamma-centered mesh without smearing.',
                    'Correlated orbitals currently form one contiguous fcDMFT [ncore, nval) window.',
                    'The current result contract reports convergence, chemical potential, hybridization, and self-energy, but not a DMFT total energy.',
                ],
                required_provider_ids=('provider.fcdmft',),
                metadata={
                    'fragment_contract': {
                        'inequivalent_fragment_count': 1,
                        'canonical_membership': (
                            'fragments[0].site_ids or fragments[0].orbital_indices'
                        ),
                        'impurity_size': 'derived from canonical membership',
                    },
                    'reference_method': 'periodic_hf',
                    'result_schema': 'pyscf-agent.hf-dmft-result.v1',
                    'energy_available': False,
                    'result_fields': [
                        'embedding_preparation',
                        'dmft_result',
                        'reference_energy',
                        'mean_field_energy',
                        'final_method',
                        'energy_kind',
                    ],
                    'artifact_kinds': [
                        'hf_dmft_result',
                        'hf_dmft_arrays',
                        'hf_dmft_output_log',
                        'hf_dmft_checkpoint',
                    ],
                },
            ),
            _capability(
                'gw_dmft',
                'GW plus DMFT',
                'embedding.method',
                limitations=[
                    'Requires an approved IAO or IAO+PAO correlated subspace from a restricted closed-shell periodic DFT reference.',
                    'Requires registered lattice GW and local GW double-counting artifacts before DMFT execution.',
                    'Correlated orbitals must form one contiguous fcDMFT [ncore, nval) window.',
                    'Reports convergence, chemical potential, hybridization, and self-energy, but not a GW+DMFT total energy.',
                ],
                required_provider_ids=('provider.fcdmft',),
                metadata={
                    'reference_method': 'periodic_dft_gw',
                    'result_schema': 'pyscf-agent.gw-dmft-result.v1',
                    'energy_available': False,
                    'impurity_solvers': ['cc', 'fci'],
                    'artifact_kinds': [
                        'periodic_gw_result',
                        'periodic_gw_analytic_continuation',
                        'periodic_gw_imaginary_self_energy',
                        'gw_dmft_local_analytic_continuation',
                        'gw_dmft_local_imaginary_self_energy',
                        'gw_dmft_local_orbital_transform',
                        'gw_dmft_result',
                        'gw_dmft_arrays',
                        'gw_dmft_output_log',
                        'gw_dmft_checkpoint',
                    ],
                },
            ),
        ],
        'embedding_impurity_solvers': [
            _capability(
                'fci',
                'FCI impurity solver',
                'embedding.impurity_solver',
                required_provider_ids=('provider.libdmet',),
                metadata={
                    'solver_options': {
                        'beta': {
                            'type': 'number',
                            'exclusiveMinimum': 0,
                            'default': 1000.0,
                            'meaning': 'Opt-in shared inverse-temperature scale for lattice SCF, density fitting and preliminary impurity SCF. Empty FCI options preserve legacy defaults. The FCI solve remains ground-state.',
                        },
                        'smearing': {
                            'type': 'boolean',
                            'default': True,
                            'meaning': 'With explicit FCI options, false disables smearing in lattice SCF, density fitting and preliminary impurity SCF.',
                        },
                    },
                },
            ),
            _capability(
                'ccsd',
                'CCSD impurity solver',
                'embedding.impurity_solver',
                required_provider_ids=('provider.libdmet',),
                limitations=[
                    'Single-reference impurity solver; convergence and applicability must be reviewed.',
                    'CCSD beta also controls DMET lattice SCF and density fitting; CCSD retains integer electron counts, not finite-temperature CCSD. A degenerate smeared reference can make CCSD diverge.',
                ],
                metadata={
                    'solver_options': {
                        'smearing': {
                            'type': 'boolean',
                            'default': True,
                            'meaning': 'Set false for zero-temperature lattice SCF, density fitting and preliminary impurity SCF; beta is ignored when disabled.',
                        },
                        'beta': {
                            'type': 'number',
                            'exclusiveMinimum': 0,
                            'default': 1000.0,
                            'meaning': 'Finite positive inverse model-energy scale shared by CCSD preliminary impurity SCF, DMET lattice SCF and density fitting (Fermi width 1/beta). Applies only to CCSD-DMET.',
                        },
                    },
                },
            ),
            _capability(
                'block2_dmrg',
                'block2 DMRG impurity solver',
                'embedding.impurity_solver',
                required_provider_ids=('provider.libdmet', 'provider.block2'),
                limitations=[
                    'Ground-state impurity solving only; DMET controls the required 1- and 2-RDM evaluation.',
                    'MPS continuation is internal to consecutive impurity iterations and cannot be supplied manually.',
                ],
                metadata={
                    'result_schema': 'pyscf-agent.dmet-block2-impurity.v1',
                    'supported_references': ['restricted', 'unrestricted'],
                    'unrestricted_integral_order': ['alpha_alpha', 'alpha_beta', 'beta_beta'],
                    'required_outputs': ['energy', '1rdm', '2rdm'],
                },
            ),
        ],
        'embedding_dmft_impurity_solvers': [
            _capability(
                'cc',
                'Coupled-cluster DMFT impurity solver',
                'embedding.dmft_impurity_solver',
                required_provider_ids=('provider.fcdmft',),
                limitations=['Requires a restricted HF reference.'],
            ),
            _capability(
                'ucc',
                'Unrestricted coupled-cluster DMFT impurity solver',
                'embedding.dmft_impurity_solver',
                required_provider_ids=('provider.fcdmft',),
                limitations=['Requires an unrestricted HF reference.'],
            ),
            _capability(
                'fci',
                'FCI DMFT impurity solver',
                'embedding.dmft_impurity_solver',
                required_provider_ids=('provider.fcdmft',),
                limitations=['Practical only for small correlated impurity spaces.'],
            ),
        ],
    }


__all__ = ['build_embedding_families']
