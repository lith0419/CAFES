import contextlib
import io
import math
import tempfile
import unittest
from unittest import mock
from pyscf_agent.providers.libdmet import libdmet_availability, normalize_dmet_options
from pyscf_agent.providers.libdmet import dmet as adapter
from pyscf_agent.backend.model_hamiltonian.solver import run_model_hamiltonian_solver
from tests.pyscf_agent.test_noninteracting_v_bath import model

class FciSmearingContractTests(unittest.TestCase):
    def test_explicit_fci_smearing_preserves_legacy_default(self):
        legacy=normalize_dmet_options({'impurity_solver':'fci'})
        self.assertEqual(adapter._lattice_scf_beta(legacy),math.inf)
        self.assertEqual(adapter._density_fit_beta(legacy),1000.)
        self.assertEqual(adapter._ccsd_smearing_metadata(legacy),{})
        for smearing in (True,False):
            options=normalize_dmet_options({'impurity_solver':'fci','impurity_solver_options':{'beta':250,'smearing':smearing}})
            self.assertEqual(normalize_dmet_options(options),options)
            expected=250. if smearing else math.inf
            self.assertEqual(adapter._lattice_scf_beta(options),expected)
            self.assertEqual(adapter._density_fit_beta(options),expected)
        for beta in (None,True,0,-1,float('inf'),float('nan'),'bad'):
            with self.subTest(beta=beta), self.assertRaises(ValueError):
                normalize_dmet_options({'impurity_solver':'fci','impurity_solver_options':{'beta':beta}})
        with self.assertRaises(ValueError):
            normalize_dmet_options({'impurity_solver':'fci','impurity_solver_options':{'smearing':'sometimes'}})

@unittest.skipUnless(libdmet_availability()['available'],'libDMET unavailable')
class FciSmearingExecutionTests(unittest.TestCase):
    def test_beta_reaches_three_stages_with_nonzero_v(self):
        from libdmet.dmet import Hubbard as dmet
        from libdmet.routine import slater
        for reference in ('restricted','unrestricted'):
            for bath in (False,True):
                with self.subTest(reference=reference,bath=bath), tempfile.TemporaryDirectory() as scratch, contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                    original=adapter._solver; solvers=[]
                    def capture(*args,**kwargs):
                        solver=original(*args,**kwargs);solvers.append(solver);return solver
                    name='RHartreeFock' if reference=='restricted' else 'HartreeFock'
                    with mock.patch.object(dmet,name,wraps=getattr(dmet,name)) as mf, mock.patch.object(slater,'FitVcorFull',wraps=slater.FitVcorFull) as fit, mock.patch.object(adapter,'_solver',side_effect=capture):
                        r=run_model_hamiltonian_solver(model(),solver_name='dmet',outputs=['energy'],scratch_directory=scratch,solver_options={
                            'impurity_solver':'fci','impurity_solver_options':{'beta':250.,'smearing':True},
                            'reference':reference,'interacting_bath':bath,'max_iterations':1,
                            'solver_max_memory_mb':1000,'fragments':[{'fragment_id':'a','site_ids':[0,1]},{'fragment_id':'b','site_ids':[2,3]}]})['dmet_result']
                    self.assertTrue(mf.call_args_list and fit.call_args_list and solvers)
                    self.assertTrue(all(c.kwargs['beta']==250. for c in mf.call_args_list))
                    self.assertTrue(all(c.args[4]==250. for c in fit.call_args_list))
                    self.assertTrue(all(s.beta==250. for s in solvers))
                    self.assertTrue(math.isfinite(r['energy']))
                    self.assertEqual(r['smearing']['lattice_scf_beta'],250.)
                    self.assertEqual(r['smearing']['density_fit_beta'],250.)
                    self.assertEqual(r['smearing']['impurity_scf_beta'],250.)
                    self.assertEqual(r['smearing']['correlated_solver_temperature'],'ground_state')
