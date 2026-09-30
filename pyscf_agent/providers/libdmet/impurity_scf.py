"""Instance-owned SCF policy for the native libDMET CCSD impurity solver."""

from __future__ import annotations

import inspect
from typing import Any


def build_ccsd_impurity_scf(*, use_diis: bool) -> Any:
    # Keep the optional numerical dependencies out of Registry/planning imports.
    import numpy as np
    from libdmet.solver.scf import SCF
    from pyscf import lib
    from pyscf.scf import diis

    class HybridDIIS(lib.diis.DIIS):
        """Blend native ADIIS/CDIIS using the orthonormal commutator RMS."""

        def __init__(self, mf: Any, filename: Any = None):
            super().__init__(mf, filename)
            self.adiis = diis.ADIIS(mf)
            self.cdiis = diis.CDIIS(mf)
            self.adiis.space = self.cdiis.space = mf.diis_space
            # Batched eigh also handles libDMET's spin-resolved overlap.
            values, vectors = np.linalg.eigh(mf.get_ovlp())
            if np.any(values <= 0):
                raise ValueError('CCSD impurity SCF requires a positive-definite overlap')
            self.orthogonalizer = (vectors / np.sqrt(values)[..., None, :]) @ (
                vectors.conj().swapaxes(-1, -2)
            )
            self.cdiis.Corth = self.orthogonalizer

        def update(self, s, d, f, mf, h1e, vhf, *args, **kwargs):
            residual = diis.get_err_vec(s, d, f, self.orthogonalizer)
            rms = float(np.linalg.norm(residual) / np.sqrt(residual.size))
            # The validated policy uses ADIIS above 1e-2, CDIIS below 1e-4,
            # and a linear blend between them. Both retain the same history.
            weight = float(np.clip((1e-2 - rms) / (1e-2 - 1e-4), 0.0, 1.0))
            fa = self.adiis.update(s, d, f, mf, h1e, vhf, *args, **kwargs)
            fc = self.cdiis.update(s, d, f, mf, h1e, vhf, *args, **kwargs)
            return (1.0 - weight) * fa + weight * fc

    class CCSDImpuritySCF(SCF):
        def __init__(self):
            self.history = []
            super().__init__(newton_ah=False)

        @property
        def mf(self):
            return self._mf

        @mf.setter
        def mf(self, value):
            self._mf = value
            if value is not None and use_diis:
                # Native HF creates a new RHF/UHF object on every call. Set its
                # factory here; PySCF instantiates it after overlap/integral setup.
                # No provider classes or other SCF instances are modified.
                value.DIIS = HybridDIIS

        def HF(self, *args, **kwargs):
            native = super().HF
            parameters = inspect.signature(native).bind(*args, **kwargs)
            if not use_diis:
                parameters.arguments['do_diis'] = False
            energy, density = native(*parameters.args, **parameters.kwargs)
            mf = self.mf
            # Use the physical (spin-summed for RHF) density, not libDMET's
            # returned per-spin RHF density, to check the unshifted gradient.
            fock = mf.get_fock(dm=mf.make_rdm1(), diis=None, cycle=-1, level_shift_factor=0)
            gradient = float(np.linalg.norm(mf.get_grad(mf.mo_coeff, mf.mo_occ, fock)))
            gradient_tolerance = mf.conv_tol_grad
            if gradient_tolerance is None:
                gradient_tolerance = float(np.sqrt(mf.conv_tol))
            accepted = bool(
                mf.converged and np.isfinite(energy) and np.all(np.isfinite(density))
                and np.isfinite(gradient) and gradient < gradient_tolerance
            )
            record = {
                # JSON null denotes the native zero-temperature (infinite-beta) limit.
                'beta': (float(parameters.arguments['beta'])
                         if np.isfinite(parameters.arguments['beta']) else None),
                'converged': bool(mf.converged),
                'accepted': accepted,
                'diis': 'adiis_cdiis' if mf.diis else 'disabled',
                'cycles': int(mf.cycles),
                'energy': float(energy),
                'gradient_norm': gradient,
                'gradient_tolerance': float(gradient_tolerance),
            }
            self.history.append(record)
            if not accepted:
                raise RuntimeError(
                    'CCSD impurity SCF did not converge to the required accuracy '
                    f'after {record["cycles"]} cycles (converged={mf.converged}, '
                    f'unshifted gradient={gradient:.6g}, tolerance={gradient_tolerance:.6g}). '
                    'CCSD was not started for this impurity solve.'
                )
            return energy, density

    return CCSDImpuritySCF()


def configure_ccsd_impurity_scf(solver: Any, *, use_diis: bool) -> Any:
    """Retain native CCSD/RDM execution and expose its preliminary SCF evidence."""
    solver.scfsolver = build_ccsd_impurity_scf(use_diis=use_diis)

    def summary():
        cc = getattr(solver, 'cisolver', None)
        cc_converged = bool(getattr(cc, 'converged', False))
        lambda_converged = bool(getattr(cc, 'converged_lambda', False))
        history = [dict(record) for record in solver.scfsolver.history]
        return {
            'solver': 'ccsd',
            'converged': bool(history and all(row['accepted'] for row in history)
                              and cc_converged and lambda_converged),
            'scf_calls': history,
            'ccsd_converged': cc_converged,
            'lambda_converged': lambda_converged,
        }

    solver.summary = summary
    return solver
