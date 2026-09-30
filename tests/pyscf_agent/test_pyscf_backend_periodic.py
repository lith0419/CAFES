from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

from pyscf_agent.contracts import (
    AnalysisSpec,
    MethodSpec,
    PeriodicSystemSpec,
    TaskSpec,
)
from pyscf_agent.backend.parsing import task_spec_from_partial
from pyscf_agent.backend.workflow import execute_request
from pyscf_agent.backend.periodic.solver import (
    _periodic_kpoints,
    compatible_periodic_basis_sets,
    compatible_periodic_pseudopotentials,
    normalized_periodic_poscar,
    parse_periodic_structure,
    periodic_structure_summary,
    run_periodic_task,
    validate_periodic_task,
)


HE_POSCAR = """Helium primitive cell
1.0
5.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
He
1
Selective dynamics
Direct
0.0 0.0 0.0 T T T
"""

HE_SUPERCELL_POSCAR = """Helium 2x1x1 supercell
1.0
10.0 0.0 0.0
0.0 5.0 0.0
0.0 0.0 5.0
He
2
Direct
0.0 0.0 0.0
0.5 0.0 0.0
"""

HE_CIF = """data_helium
_symmetry_space_group_name_H-M 'P 1'
_symmetry_Int_Tables_number 1
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
He1 He 0.0 0.0 0.0
"""

INVERSION_CIF = """data_carbon
_space_group_name_H-M_alt 'P -1'
_space_group_IT_number 2
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_space_group_symop_operation_xyz
'x,y,z'
'-x,-y,-z'
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
C1 C 0.1 0.2 0.3
"""

PARTIAL_OCCUPANCY_CIF = """data_carbon
_space_group_name_H-M_alt 'P 1'
_space_group_IT_number 1
_cell_length_a 5.0
_cell_length_b 5.0
_cell_length_c 5.0
_cell_angle_alpha 90
_cell_angle_beta 90
_cell_angle_gamma 90
loop_
_atom_site_label
_atom_site_type_symbol
_atom_site_fract_x
_atom_site_fract_y
_atom_site_fract_z
_atom_site_occupancy
C1 C 0.0 0.0 0.0 0.5
"""

SI_POSCAR = """Silicon primitive cell
1.0
5.43 0.0 0.0
0.0 5.43 0.0
0.0 0.0 5.43
Si
1
Direct
0.0 0.0 0.0
"""

ZNSE_POSCAR = """Zinc selenide cell
1.0
5.67 0.0 0.0
0.0 5.67 0.0
0.0 0.0 5.67
Zn Se
1 1
Direct
0.0 0.0 0.0
0.25 0.25 0.25
"""


class PeriodicBackendTests(unittest.TestCase):
    def test_periodic_default_uses_broad_coverage_molopt_basis(self):
        self.assertEqual(PeriodicSystemSpec().basis, 'gth-szv-molopt-sr')

    def test_periodic_kpoint_schemes_match_pyscf_make_kpts(self):
        from pyscf.pbc import gto

        cell = gto.Cell()
        cell.atom = 'He 0 0 0'
        cell.a = np.eye(3) * 5.0
        cell.unit = 'Angstrom'
        cell.basis = 'gth-szv'
        cell.pseudo = 'gth-pade'
        cell.verbose = 0
        cell.build()

        for scheme, with_gamma_point in (
            ('gamma_centered', True),
            ('monkhorst_pack', False),
        ):
            with self.subTest(scheme=scheme):
                task_spec = TaskSpec(
                    task_type='periodic',
                    periodic=PeriodicSystemSpec(
                        kmesh=(3, 2, 1),
                        kpoint_scheme=scheme,
                    ),
                    method=MethodSpec(name='hf', restricted=True),
                    analysis=AnalysisSpec(outputs=['energy']),
                )

                actual, _, _, _ = _periodic_kpoints(cell, task_spec)
                expected = cell.make_kpts(
                    [3, 2, 1],
                    wrap_around=True,
                    with_gamma_point=with_gamma_point,
                )

                np.testing.assert_allclose(actual, expected, atol=1e-12)

    def test_poscar_parser_supports_selective_direct_coordinates(self):
        lattice, atoms = parse_periodic_structure(HE_POSCAR, 'poscar')
        summary = periodic_structure_summary(HE_POSCAR, 'vasp')

        self.assertEqual(lattice.shape, (3, 3))
        self.assertEqual(atoms[0][0], 'He')
        self.assertEqual(summary['source_format'], 'poscar')
        self.assertEqual(summary['formula'], 'He')
        self.assertEqual(summary['atom_count'], 1)
        self.assertAlmostEqual(summary['cell_volume_angstrom3'], 125.0)
        self.assertEqual(summary['atoms'][0]['fractional'], [0.0, 0.0, 0.0])
        self.assertEqual(summary['band_path_preview']['status'], 'available')
        self.assertEqual(summary['band_path_preview']['automatic_path'], 'GXMGRX,MR')
        self.assertEqual(
            set(summary['band_path_preview']['special_points_scaled']),
            {'G', 'M', 'R', 'X'},
        )
        seekpath_preview = summary['seekpath_preview']
        self.assertEqual(seekpath_preview['status'], 'available')
        self.assertEqual(seekpath_preview['spacegroup_number'], 221)
        self.assertEqual(seekpath_preview['spacegroup_international'], 'Pm-3m')
        self.assertEqual(seekpath_preview['primitive_atom_count'], 1)
        self.assertEqual(seekpath_preview['volume_original_wrt_primitive'], 1.0)
        self.assertIn('GAMMA-X', seekpath_preview['path'])
        self.assertIn('GAMMA', seekpath_preview['point_coords'])
        self.assertEqual(seekpath_preview['warnings'], [])

    def test_poscar_parser_rejects_nonfinite_coordinates(self):
        invalid_poscar = HE_POSCAR.replace('0.0 0.0 0.0 T T T', 'nan 0.0 0.0 T T T')

        with self.assertRaisesRegex(ValueError, 'finite numeric values'):
            parse_periodic_structure(invalid_poscar, 'poscar')

    def test_cif_parser_accepts_p1_and_expands_symmetry_to_p1(self):
        summary = periodic_structure_summary(HE_CIF, 'cif')

        self.assertEqual(summary['formula'], 'He')
        self.assertEqual(summary['atom_count'], 1)
        self.assertAlmostEqual(summary['cell_volume_angstrom3'], 125.0)
        self.assertFalse(summary['symmetry_expanded'])

        expanded = periodic_structure_summary(INVERSION_CIF, 'cif')

        self.assertEqual(expanded['source_site_count'], 1)
        self.assertEqual(expanded['atom_count'], 2)
        self.assertTrue(expanded['symmetry_expanded'])
        self.assertEqual(expanded['space_group_number'], 2)
        self.assertEqual(expanded['space_group_symbol'], 'P -1')
        normalized = normalized_periodic_poscar(expanded)
        self.assertIn('C2 normalized by PySCF Agent', normalized)
        self.assertIn('\n  C\n  2\nDirect\n', normalized)

    def test_cif_parser_rejects_partial_occupancy(self):
        with self.assertRaisesRegex(ValueError, 'partial site occupancies'):
            periodic_structure_summary(PARTIAL_OCCUPANCY_CIF, 'cif')

    def test_periodic_parser_rejects_left_handed_lattice(self):
        left_handed = HE_POSCAR.replace('0.0 0.0 5.0', '0.0 0.0 -5.0')

        with self.assertRaisesRegex(ValueError, 'right-handed'):
            parse_periodic_structure(left_handed, 'poscar')

    def test_periodic_partial_request_preserves_nested_contract(self):
        task_spec = task_spec_from_partial({
            'task_type': 'periodic',
            'periodic': {
                'structure_format': 'poscar',
                'structure_text': HE_POSCAR,
                'basis': 'gth-szv',
                'pseudo': 'gth-pbe',
                'kmesh': [2, 1, 1],
                'kpoint_scheme': 'monkhorst_pack',
                'kpoint_shift': [0.25, 0.0, 0.0],
                'band_path_npoints': 64,
                'band_path_reference_distance': 0.04,
                'band_path_symprec': 2e-5,
                'dimension': 3,
                'precision': 1e-10,
                'ke_cutoff': 80.0,
                'density_fitting_method': 'gdf',
                'density_fitting_auxbasis': 'weigend',
                'exxdiv': 'ewald',
                'smearing_method': 'fermi',
                'smearing_sigma': 0.01,
                'smearing_fix_spin': True,
                'band_path_mode': 'explicit',
                'band_path': 'G-Q-G',
                'band_path_special_points': {
                    'G': [0.0, 0.0, 0.0],
                    'Q': [0.25, 0.25, 0.25],
                },
            },
            'method': 'dft',
            'xc': 'pbe',
            'charge': 0,
            'spin': 0,
        })

        self.assertEqual(task_spec.task_type, 'periodic')
        self.assertEqual(task_spec.periodic.kmesh, (2, 1, 1))
        self.assertEqual(task_spec.periodic.kpoint_scheme, 'monkhorst_pack')
        self.assertEqual(task_spec.periodic.kpoint_shift, (0.25, 0.0, 0.0))
        self.assertEqual(task_spec.periodic.band_path_npoints, 64)
        self.assertEqual(task_spec.periodic.band_path_reference_distance, 0.04)
        self.assertEqual(task_spec.periodic.band_path_symprec, 2e-5)
        self.assertEqual(task_spec.periodic.precision, 1e-10)
        self.assertEqual(task_spec.periodic.ke_cutoff, 80.0)
        self.assertEqual(task_spec.periodic.density_fitting_method, 'gdf')
        self.assertEqual(task_spec.periodic.density_fitting_auxbasis, 'weigend')
        self.assertEqual(task_spec.periodic.smearing_method, 'fermi')
        self.assertEqual(task_spec.periodic.smearing_sigma, 0.01)
        self.assertTrue(task_spec.periodic.smearing_fix_spin)
        self.assertEqual(task_spec.periodic.band_path_mode, 'explicit')
        self.assertEqual(task_spec.periodic.band_path, 'G-Q-G')
        self.assertEqual(task_spec.periodic.band_path_special_points['Q'], (0.25, 0.25, 0.25))
        self.assertEqual(task_spec.periodic.pseudo, 'gth-pbe')
        self.assertEqual(task_spec.method.name, 'dft')
        self.assertEqual(task_spec.method.xc, 'pbe')
        self.assertEqual(task_spec.analysis.outputs, ['energy', 'band_gap', 'fermi_energy'])

    def test_periodic_validation_blocks_missing_structure_and_invalid_kmesh(self):
        task_spec = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(kmesh=(0, 1, 1)),
            method=MethodSpec(name='hf', restricted=True),
            analysis=AnalysisSpec(outputs=['energy']),
        )

        errors = validate_periodic_task(task_spec)

        self.assertTrue(any('structure text is required' in error.lower() for error in errors))
        self.assertTrue(any('three positive integers' in error for error in errors))

    def test_malformed_structured_numerics_are_not_silently_defaulted(self):
        for field, value in {'kmesh': ['bad'], 'kpoint_shift': [0.0, 'bad', 0.0],
                             'precision': 'bad', 'ke_cutoff': 'bad', 'fft_mesh': [20, 20]}.items():
            with self.subTest(field=field), self.assertRaises(ValueError):
                task_spec_from_partial({'task_type': 'periodic', 'method': 'hf',
                                        'periodic': {'structure_text': HE_POSCAR, field: value}})

    def test_periodic_hf_discards_dft_functional(self):
        task_spec = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(structure_text=HE_POSCAR),
            method=MethodSpec(name='hf', restricted=True, xc='pbe'),
            analysis=AnalysisSpec(outputs=['energy']),
        )

        errors = validate_periodic_task(task_spec)

        self.assertEqual(errors, [])
        self.assertIsNone(task_spec.method.xc)

    def test_periodic_validation_checks_numerical_control_contract(self):
        task_spec = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=HE_POSCAR,
                ke_cutoff=80.0,
                fft_mesh=(20, 20, 20),
                density_fitting_method='fft',
                density_fitting_auxbasis='weigend',
                smearing_method='fermi',
                smearing_sigma=None,
                band_path_npoints=1,
            ),
            method=MethodSpec(name='dft', restricted=True, xc='pbe'),
            analysis=AnalysisSpec(outputs=['energy', 'band_structure']),
        )

        errors = validate_periodic_task(task_spec)

        self.assertTrue(any('either periodic ke_cutoff or fft_mesh' in error for error in errors))
        self.assertTrue(any('only valid with GDF or MDF' in error for error in errors))
        self.assertTrue(any('smearing requires' in error for error in errors))
        self.assertTrue(any('band-path sampling requires' in error for error in errors))

    def test_custom_and_explicit_band_paths_are_validated_against_the_cell(self):
        custom_task = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=HE_POSCAR,
                band_path_mode='custom',
                band_path='G-X-M-G',
                band_path_npoints=24,
            ),
            method=MethodSpec(name='hf', restricted=True),
            analysis=AnalysisSpec(outputs=['energy', 'band_structure']),
        )

        self.assertEqual(validate_periodic_task(custom_task), [])
        self.assertEqual(custom_task.periodic.band_path, 'GXMG')

        unknown_label_task = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=HE_POSCAR,
                band_path_mode='custom',
                band_path='GQ',
            ),
            method=MethodSpec(name='hf', restricted=True),
            analysis=AnalysisSpec(outputs=['band_structure']),
        )
        unknown_errors = validate_periodic_task(unknown_label_task)
        self.assertTrue(any('Unknown standard special-point label(s): Q' in error for error in unknown_errors))

        explicit_task = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=HE_POSCAR,
                band_path_mode='explicit',
                band_path='G-Q-G',
                band_path_special_points={
                    'G': (0.0, 0.0, 0.0),
                    'Q': (0.25, 0.25, 0.25),
                },
            ),
            method=MethodSpec(name='hf', restricted=True),
            analysis=AnalysisSpec(outputs=['band_structure']),
        )

        self.assertEqual(validate_periodic_task(explicit_task), [])
        self.assertEqual(explicit_task.periodic.band_path, 'GQG')

        missing_coordinate_task = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=HE_POSCAR,
                band_path_mode='explicit',
                band_path='GQ',
                band_path_special_points={'G': (0.0, 0.0, 0.0)},
            ),
            method=MethodSpec(name='hf', restricted=True),
            analysis=AnalysisSpec(outputs=['band_structure']),
        )
        missing_errors = validate_periodic_task(missing_coordinate_task)
        self.assertTrue(any('missing coordinates for label(s): Q' in error for error in missing_errors))

    def test_seekpath_validation_standardizes_supercells_and_enforces_safety_limits(self):
        seekpath_task = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=HE_SUPERCELL_POSCAR,
                band_path_mode='seekpath',
                band_path_reference_distance=0.2,
                band_path_symprec=1e-5,
            ),
            method=MethodSpec(name='hf', restricted=True),
            analysis=AnalysisSpec(outputs=['energy', 'band_structure']),
        )

        self.assertEqual(validate_periodic_task(seekpath_task), [])
        self.assertIsNone(seekpath_task.periodic.band_path)
        self.assertEqual(seekpath_task.periodic.band_path_special_points, {})

        charged_supercell_task = task_spec_from_partial({
            'task_type': 'periodic',
            'periodic': {
                'structure_text': HE_SUPERCELL_POSCAR,
                'band_path_mode': 'seekpath',
                'band_path_reference_distance': 0.2,
            },
            'method': 'hf',
            'charge': 1,
            'outputs': ['band_structure'],
        })
        charged_errors = validate_periodic_task(charged_supercell_task)
        self.assertTrue(any('nonzero total charge/spin' in error for error in charged_errors))

        unrestricted_supercell_task = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=HE_SUPERCELL_POSCAR,
                band_path_mode='seekpath',
                band_path_reference_distance=0.2,
            ),
            method=MethodSpec(name='hf', restricted=False),
            analysis=AnalysisSpec(outputs=['band_structure']),
        )
        unrestricted_errors = validate_periodic_task(unrestricted_supercell_task)
        self.assertTrue(any('unrestricted reference' in error for error in unrestricted_errors))

        dense_path_task = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=HE_POSCAR,
                band_path_mode='seekpath',
                band_path_reference_distance=0.005,
            ),
            method=MethodSpec(name='hf', restricted=True),
            analysis=AnalysisSpec(outputs=['band_structure']),
        )
        dense_path_errors = validate_periodic_task(dense_path_task)
        self.assertTrue(any('exceeding the 400-point safety limit' in error for error in dense_path_errors))

        invalid_numerics_task = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=HE_POSCAR,
                band_path_mode='seekpath',
                band_path_reference_distance=0.001,
                band_path_symprec=1.0,
            ),
            method=MethodSpec(name='hf', restricted=True),
            analysis=AnalysisSpec(outputs=['band_structure']),
        )
        invalid_numerics_errors = validate_periodic_task(invalid_numerics_task)
        self.assertTrue(any('reference distance' in error for error in invalid_numerics_errors))
        self.assertTrue(any('symmetry tolerance' in error for error in invalid_numerics_errors))

    def test_general_periodic_pseudopotentials_cover_h_through_rn(self):
        from pyscf.data.elements import ELEMENTS
        from pyscf.pbc.gto import pseudo as pbc_pseudo

        for pseudo_name in ('gth-pade', 'gth-lda', 'gth-pbe', 'gth-hf-rev'):
            missing = []
            for symbol in ELEMENTS[1:87]:
                try:
                    pbc_pseudo.load(pseudo_name, symbol)
                except Exception:
                    missing.append(symbol)
            self.assertEqual(missing, [], msg='{0} lacks: {1}'.format(pseudo_name, ', '.join(missing)))

    def test_periodic_pseudopotential_recommendations_are_verified_for_every_element(self):
        self.assertEqual(
            compatible_periodic_pseudopotentials(['Zn', 'Se']),
            ['gth-pade', 'gth-lda', 'gth-pbe', 'gth-blyp', 'gth-bp', 'gth-hf-rev'],
        )

    def test_periodic_validation_checks_element_basis_pseudo_coverage(self):
        task_spec = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=SI_POSCAR,
                basis='gth-szv',
                pseudo='gth-pbesol',
            ),
            method=MethodSpec(name='dft', restricted=True, xc='pbe'),
            analysis=AnalysisSpec(outputs=['energy']),
        )

        errors = validate_periodic_task(task_spec)

        pseudo_error = next(error for error in errors if error.startswith('Periodic pseudopotential'))
        self.assertIn('Periodic pseudopotential gth-pbesol is unavailable for element(s): Si.', pseudo_error)
        self.assertIn('Verified compatible registered pseudopotential replacements', pseudo_error)

    def test_periodic_basis_recommendations_are_verified_for_every_element(self):
        self.assertEqual(
            compatible_periodic_basis_sets(['Zn', 'Se']),
            ['gth-szv-molopt-sr', 'gth-dzvp-molopt-sr'],
        )
        task_spec = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=ZNSE_POSCAR,
                basis='gth-szv',
                pseudo='gth-pbe',
            ),
            method=MethodSpec(name='dft', restricted=True, xc='pbe'),
            analysis=AnalysisSpec(outputs=['energy']),
        )

        errors = validate_periodic_task(task_spec)

        self.assertEqual(len(errors), 1)
        self.assertEqual(
            errors[0],
            'Periodic basis gth-szv is unavailable for element(s): Zn, Se. '
            'Verified compatible registered basis replacements for all structure elements: '
            'gth-szv-molopt-sr, gth-dzvp-molopt-sr.',
        )

    def test_non_gamma_hf_selects_kpoint_aware_reference(self):
        task_spec = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=HE_POSCAR,
                basis='gth-szv',
                pseudo='gth-pade',
                kmesh=(2, 1, 1),
            ),
            method=MethodSpec(name='hf', restricted=True),
            analysis=AnalysisSpec(outputs=['energy', 'band_structure']),
        )
        cell = mock.Mock()
        cell.get_abs_kpts.side_effect = lambda scaled: np.asarray(scaled, dtype=float) * 0.5
        cell.nao_nr.return_value = 1
        cell.nelectron = 2
        cell.dimension = 3
        cell.precision = 1e-8
        cell.mesh = np.asarray([9, 9, 9])
        mean_field = SimpleNamespace(
            stdout=None,
            max_cycle=0,
            conv_tol=1e-9,
            converged=True,
            mo_energy=np.asarray([[-0.5, 0.2], [-0.4, 0.3]]),
            mo_occ=np.asarray([[2.0, 0.0], [2.0, 0.0]]),
            with_df=object(),
            get_fermi=lambda: -0.4,
            kernel=lambda: -1.25,
            get_bands=mock.Mock(return_value=(
                np.asarray([[-0.5, 0.2], [-0.45, 0.25], [-0.4, 0.3]]),
                None,
            )),
        )
        constructor = mock.Mock(return_value=mean_field)
        band_path = SimpleNamespace(
            path='GX',
            kpts=np.asarray([[0.0, 0.0, 0.0], [0.25, 0.0, 0.0], [0.5, 0.0, 0.0]]),
            special_points={
                'G': np.asarray([0.0, 0.0, 0.0]),
                'X': np.asarray([0.5, 0.0, 0.0]),
            },
            get_linear_kpoint_axis=lambda: (
                np.asarray([0.0, 0.5, 1.0]),
                np.asarray([0.0, 1.0]),
                ['G', 'X'],
            ),
        )

        with mock.patch(
            'pyscf_agent.backend.periodic.solver.build_periodic_cell',
            return_value=(
                cell,
                {'formula': 'He', 'cell_role': 'user_input'},
                {
                    'input_structure': None,
                    'seekpath_standardization': None,
                    'path_definition': None,
                },
            ),
        ), mock.patch(
            'pyscf_agent.backend.periodic.solver._configure_periodic_density_fitting',
            side_effect=lambda mean_field, _cell, _kpts, _task_spec: mean_field,
        ), mock.patch(
            'pyscf.pbc.tools.pyscf_ase.bandpath',
            return_value=band_path,
        ) as bandpath_mock, mock.patch('pyscf.pbc.scf.KRHF', constructor):
            result = run_periodic_task(task_spec)

        expected_kpoints = np.asarray([[0.0, 0.0, 0.0], [-0.25, 0.0, 0.0]])
        np.testing.assert_allclose(constructor.call_args.kwargs['kpts'], expected_kpoints)
        self.assertEqual(result['reference'], 'krhf')
        self.assertEqual(result['kpoint_count'], 2)
        self.assertFalse(result['gamma_point'])
        self.assertAlmostEqual(result['band_gap'], 0.6)
        self.assertEqual(result['fermi_energy'], -0.4)
        self.assertEqual(result['fermi_energy_source'], 'pyscf_get_fermi_vbm_convention')
        self.assertEqual(result['band_edges']['energy_unit'], 'Hartree')
        self.assertEqual(result['band_edges']['vbm']['kpoint_scaled'], [-0.5, 0.0, 0.0])
        self.assertEqual(result['band_edges']['cbm']['kpoint_scaled'], [0.0, 0.0, 0.0])
        self.assertEqual(result['periodic_numerics']['fft_mesh'], [9, 9, 9])
        self.assertEqual(len(result['periodic_orbitals']['channels']), 1)
        bandpath_mock.assert_called_once_with(cell, npoints=80)
        expected_band_kpoints = np.asarray([
            [0.0, 0.0, 0.0],
            [0.125, 0.0, 0.0],
            [0.25, 0.0, 0.0],
        ])
        np.testing.assert_allclose(mean_field.get_bands.call_args.args[0], expected_band_kpoints)
        self.assertEqual(result['band_path'], 'GX')
        self.assertEqual(result['band_path_point_count'], 3)
        self.assertEqual(result['band_path_labels'], ['G', 'X'])
        band_result = result['periodic_band_structure']
        self.assertEqual(band_result['energy_reference'], 'mean_field_fermi_energy')
        self.assertAlmostEqual(
            band_result['channels'][0]['energies_relative_to_fermi_ev'][0][0],
            -0.1 * 27.21138602,
        )

    def test_smearing_reports_pyscf_chemical_potential_and_thermal_energies(self):
        task_spec = TaskSpec(
            task_type='periodic',
            periodic=PeriodicSystemSpec(
                structure_text=HE_POSCAR,
                smearing_method='fermi',
                smearing_sigma=0.02,
                exxdiv='none',
            ),
            method=MethodSpec(name='dft', restricted=True, xc='pbe'),
            analysis=AnalysisSpec(outputs=['energy', 'fermi_energy']),
        )
        cell = mock.Mock()
        cell.get_abs_kpts.side_effect = lambda scaled: np.asarray(scaled, dtype=float) * 0.5
        cell.nao_nr.return_value = 2
        cell.nelectron = 2
        cell.dimension = 3
        cell.precision = 1e-8
        cell.mesh = np.asarray([9, 9, 9])
        smearing_calls = []
        mean_field = SimpleNamespace(
            stdout=None,
            max_cycle=0,
            conv_tol=1e-9,
            converged=True,
            mo_energy=np.asarray([-0.5, 0.2]),
            mo_occ=np.asarray([1.8, 0.2]),
            with_df=object(),
            get_fermi=lambda: -0.1,
            kernel=lambda: -1.1,
        )

        def apply_smearing(**kwargs):
            smearing_calls.append(kwargs)
            mean_field.smearing_method = kwargs['method']
            mean_field.entropy = 0.3
            mean_field.e_free = -1.11
            mean_field.e_zero = -1.105
            return mean_field

        mean_field.smearing = apply_smearing

        with mock.patch(
            'pyscf_agent.backend.periodic.solver.build_periodic_cell',
            return_value=(
                cell,
                {'formula': 'He', 'cell_role': 'user_input'},
                {
                    'input_structure': None,
                    'seekpath_standardization': None,
                    'path_definition': None,
                },
            ),
        ), mock.patch(
            'pyscf_agent.backend.periodic.solver._configure_periodic_density_fitting',
            side_effect=lambda current, _cell, _kpts, _task_spec: current,
        ), mock.patch('pyscf.pbc.dft.RKS', return_value=mean_field):
            result = run_periodic_task(task_spec)

        self.assertEqual(smearing_calls, [{
            'sigma': 0.02,
            'method': 'fermi',
            'fix_spin': False,
        }])
        self.assertIsNone(mean_field.exxdiv)
        self.assertTrue(result['is_metal'])
        self.assertEqual(result['fermi_energy'], -0.1)
        self.assertEqual(result['fermi_energy_source'], 'pyscf_get_fermi_smearing_chemical_potential')
        self.assertEqual(result['smearing']['free_energy'], -1.11)
        self.assertEqual(result['smearing']['zero_temperature_energy'], -1.105)

    def test_gamma_hf_executes_and_writes_periodic_artifacts(self):
        request = {
            'task_type': 'periodic',
            'periodic': {
                'structure_format': 'poscar',
                'structure_text': HE_POSCAR,
                'basis': 'gth-szv',
                'pseudo': 'gth-pade',
                'kmesh': [1, 1, 1],
                'dimension': 3,
                'band_path_mode': 'custom',
                'band_path': 'GXMG',
                'band_path_npoints': 12,
            },
            'method': 'hf',
            'job': 'single_point',
            'charge': 0,
            'spin': 0,
            'outputs': ['energy', 'band_gap', 'fermi_energy', 'band_structure'],
            'max_cycle': 30,
            'verbose': 0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            state = execute_request(
                json.dumps(request),
                channel='test',
                locale='en',
                work_dir=tmpdir,
                run_id='periodic-gamma-hf',
            )
            report = state['task_report']
            artifact_paths = [Path(item['path']) for item in report['artifacts']]

            self.assertEqual(report['execution_status'], 'succeeded')
            self.assertEqual(report['task_spec']['task_type'], 'periodic')
            self.assertEqual(report['structured_results']['task_type'], 'periodic')
            self.assertEqual(report['structured_results']['reference'], 'rhf')
            self.assertTrue(report['structured_results']['converged'])
            self.assertEqual(report['structured_results']['energy_unit'], 'Ha/cell')
            self.assertEqual(report['structured_results']['kmesh'], [1, 1, 1])
            self.assertEqual(report['structured_results']['band_structure_status'], 'completed')
            self.assertEqual(report['structured_results']['band_path_mode'], 'custom')
            self.assertEqual(report['structured_results']['band_path'], 'GXMG')
            self.assertEqual(report['structured_results']['band_path_point_count'], 12)
            self.assertEqual(
                report['structured_results']['fermi_energy'],
                report['structured_results']['valence_band_max'],
            )
            self.assertEqual(
                report['structured_results']['fermi_energy_source'],
                'pyscf_get_fermi_vbm_convention',
            )
            self.assertIn('periodic_numerics', report['structured_results'])
            self.assertNotIn('periodic_orbitals', report['structured_results'])
            self.assertTrue(any(path.name == 'input-periodic-structure.vasp' for path in artifact_paths))
            self.assertTrue(any(path.name == 'input-periodic-task.json' for path in artifact_paths))
            self.assertTrue(any(path.name == 'artifact-periodic-cell.json' for path in artifact_paths))
            self.assertTrue(any(path.name == 'artifact-periodic-structure-normalized.vasp' for path in artifact_paths))
            self.assertTrue(any(path.name == 'artifact-periodic-numerics.json' for path in artifact_paths))
            self.assertTrue(any(path.name == 'result-periodic-orbitals.json' for path in artifact_paths))
            self.assertTrue(any(path.name == 'result-periodic-band-structure.json' for path in artifact_paths))
            self.assertTrue(any(path.name == 'result-periodic-band-structure.tsv' for path in artifact_paths))
            self.assertTrue(any(path.name == 'result-periodic-scf.json' for path in artifact_paths))
            self.assertTrue(all(path.exists() for path in artifact_paths))
            completed_modules = {
                item.get('module_id')
                for item in report['module_runtime_observations']
                if item.get('status') == 'completed'
            }
            self.assertIn('periodic.band_analysis', completed_modules)

    def test_seekpath_supercell_executes_on_primitive_and_writes_audit_artifacts(self):
        request = {
            'task_type': 'periodic',
            'periodic': {
                'structure_format': 'poscar',
                'structure_text': HE_SUPERCELL_POSCAR,
                'basis': 'gth-szv',
                'pseudo': 'gth-pade',
                'kmesh': [1, 1, 1],
                'band_path_mode': 'seekpath',
                'band_path_reference_distance': 0.2,
                'band_path_symprec': 1e-5,
            },
            'method': 'hf',
            'charge': 0,
            'spin': 0,
            'outputs': ['energy', 'band_structure'],
            'max_cycle': 30,
            'verbose': 0,
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            state = execute_request(
                json.dumps(request),
                channel='test',
                locale='en',
                work_dir=tmpdir,
                run_id='periodic-seekpath-hf',
            )
            report = state['task_report']
            results = report['structured_results']
            artifact_paths = {
                Path(item['path']).name: Path(item['path'])
                for item in report['artifacts']
            }
            input_structure = json.loads(
                artifact_paths['artifact-periodic-input-cell.json'].read_text(encoding='utf-8')
            )
            seekpath_audit = json.loads(
                artifact_paths['artifact-periodic-seekpath-standardization.json'].read_text(encoding='utf-8')
            )

            self.assertEqual(report['execution_status'], 'succeeded')
            self.assertEqual(results['energy_unit'], 'Ha/standardized primitive cell')
            self.assertEqual(input_structure['atom_count'], 2)
            self.assertEqual(input_structure['cell_role'], 'user_input')
            self.assertEqual(results['periodic_structure']['atom_count'], 1)
            self.assertEqual(results['periodic_structure']['cell_role'], 'seekpath_standardized_primitive')
            self.assertEqual(results['periodic_standardization']['input_atom_count'], 2)
            self.assertEqual(results['periodic_standardization']['primitive_atom_count'], 1)
            self.assertEqual(seekpath_audit['volume_original_wrt_primitive'], 2.0)
            self.assertEqual(seekpath_audit['spacegroup_number'], 221)
            self.assertEqual(results['band_path_mode'], 'seekpath')
            self.assertIn('GAMMA-X', results['band_path'])
            self.assertLessEqual(results['band_path_point_count'], 400)
            self.assertEqual(
                results['periodic_numerics']['band_path']['kmesh_cell_role'],
                'seekpath_standardized_primitive',
            )
            self.assertIsNone(results['periodic_numerics']['band_path']['requested_point_count'])
            self.assertEqual(
                results['periodic_numerics']['band_path']['generated_point_count'],
                results['band_path_point_count'],
            )
            self.assertIn('artifact-periodic-input-structure-normalized.vasp', artifact_paths)
            self.assertIn('artifact-periodic-cell.json', artifact_paths)
            self.assertIn('artifact-periodic-structure-normalized.vasp', artifact_paths)
            self.assertIn('result-periodic-band-structure.tsv', artifact_paths)


if __name__ == '__main__':
    unittest.main()
