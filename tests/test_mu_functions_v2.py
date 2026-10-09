"""Run on AutoDL: density constraints, encoding adapter, paired bootstrap."""
import json
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from build_mu_functions_v2 import density_setup,tilted_mu
from jedc_distribution_v1 import make_edges,midpoint_grid,rebin
from mu_evaluation_v2 import FrozenGridPolicy
from summarize_mu_functions_v2 import bootstrap
from networks import PolicyNetwork
from networks_mu_v2 import MuPolicyNetwork


class FunctionContracts(unittest.TestCase):
    def test_draw_tilt_retains_atoms_groups_and_targets(self):
        fit=json.loads((ROOT/'runs/jedc_mu_momentfit_20260929/report.json').read_text())
        theta=np.asarray([r['theta'] for r in fit['records']]);atoms=np.array([.003575,.000297917])
        edges=make_edges();setup=density_setup(edges)
        for z in [0,1]:
            population=np.array([.1,.9] if z==0 else [.04,.96])
            for target in [29.4,36.668577,44.]:
                mu,lam,K=tilted_mu(theta,target,population,atoms,edges,setup)
                np.testing.assert_allclose(mu.sum(1),population,atol=1e-12)
                np.testing.assert_array_equal(mu[:,0],population*atoms)
                self.assertAlmostEqual(float((mu*midpoint_grid(edges)).sum()),target,places=8)
                self.assertTrue(np.all(mu>=0));self.assertLess(abs(K-target),.01)

    def test_grid_adapter_rebins_continuation_mu(self):
        class Capture(torch.nn.Module):
            def forward(self,mu,z,a,e):self.mu=mu;return torch.zeros_like(a)
        source=np.arange(9)*.5;target=np.arange(5)*1.
        mu=np.array([[.01,.01,.02,.01,.01,.01,.01,.01,.01],[.09,.1,.1,.1,.1,.1,.1,.1,.1]])
        net=Capture();adapter=FrozenGridPolicy(net,torch.tensor(source),torch.tensor(target))
        adapter(torch.tensor(mu)[None],torch.tensor([0]),torch.zeros(1,2,3),torch.zeros(1,2,3))
        np.testing.assert_allclose(net.mu[0],rebin(mu,source,target),atol=1e-14)
        mu2=mu.copy();mu2[0,1]+=.001;mu2[0,2]-=.001
        adapter(torch.tensor(mu2)[None],torch.tensor([1]),torch.zeros(1,2,3),torch.zeros(1,2,3))
        np.testing.assert_allclose(net.mu[0],rebin(mu2,source,target),atol=1e-14)

    def test_paired_bootstrap_constant_difference(self):
        result=bootstrap(np.ones((3,12))*-.002,np.random.default_rng(42))
        self.assertAlmostEqual(result['ci_low'],-.002);self.assertAlmostEqual(result['ci_high'],-.002)

    def test_factorized_mlp_outputs_and_gradients(self):
        torch.manual_seed(11);grid=torch.arange(8,dtype=torch.float64)
        for mode in ['capital','capital_only']:
            old=PolicyNetwork('mlp',8,500.,7,4,macro_features=mode,grid=grid).double()
            new=MuPolicyNetwork('mlp',8,500.,7,4,macro_features=mode,grid=grid).double()
            new.load_state_dict(old.state_dict())
            mu=torch.rand(2,2,8,dtype=torch.float64);mu/=mu.sum((1,2),keepdim=True)
            z=torch.tensor([0,1]);assets=torch.rand(2,2,8,dtype=torch.float64)*100
            employment=torch.arange(2)[None,:,None].expand(2,2,8)
            a=old(mu,z,assets,employment);b=new(mu,z,assets,employment)
            torch.testing.assert_close(a,b,atol=1e-12,rtol=1e-12)
            a.square().sum().backward();b.square().sum().backward()
            for p,q in zip(old.parameters(),new.parameters()):torch.testing.assert_close(p.grad,q.grad,atol=1e-12,rtol=1e-12)


if __name__=='__main__':unittest.main()
