"""Meaningful representation and gradient invariants for the new transport."""
import sys
import unittest
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from jedc_distribution_v1 import push_assets_numpy,push_assets_torch,midpoint_grid,rebin,transition_numpy
from build_jedc_mu_v1 import perturb

class IntervalContract(unittest.TestCase):
    def test_identity_and_atom(self):
        edges=np.array([0.,1.,2.,3.]);mass=np.array([.2,.3,.1,.4])
        np.testing.assert_allclose(push_assets_numpy(mass,edges,edges),mass,atol=1e-14)
        mu=np.stack([mass*.1,mass*.9])
        out=rebin(mu,edges,np.arange(7)*.5)
        np.testing.assert_array_equal(out[:,0],mu[:,0])
        np.testing.assert_allclose((out*midpoint_grid(np.arange(7)*.5)).sum(),(mu*midpoint_grid(edges)).sum())

    def test_positive_small_saving_never_atom(self):
        edges=np.array([0.,1.,2.]);mass=np.array([.2,.4,.4])
        out=push_assets_numpy(mass,np.array([.01,.02,.03]),edges)
        np.testing.assert_allclose(out,[0.,1.,0.],atol=1e-14)
        out=push_assets_numpy(mass,np.array([0.,0.,.3]),edges)
        np.testing.assert_allclose(out,[.6,.4,0.],atol=1e-14)

    def test_torch_numpy_nonmonotone_and_gradient(self):
        edges=np.array([0.,1.,2.,3.]);mass=np.array([.1,.2,.3,.4]);saving=np.array([.2,2.7,.4,2.2])
        m=torch.tensor(mass,dtype=torch.float64,requires_grad=True)
        s=torch.tensor(saving,dtype=torch.float64,requires_grad=True)
        out=push_assets_torch(m,s,torch.tensor(edges))
        np.testing.assert_allclose(out.detach(),push_assets_numpy(mass,saving,edges),atol=1e-14)
        self.assertTrue(torch.autograd.gradcheck(lambda x:push_assets_torch(m,x,torch.tensor(edges)),(s,),eps=1e-6,atol=1e-5))
        (out*torch.arange(4)).sum().backward();self.assertTrue(torch.isfinite(s.grad).all())

    def test_shape_constraints(self):
        edges=np.arange(201)*.5;grid=midpoint_grid(edges)
        raw=np.exp(-((grid-35)/15)**2);raw[0]=.003;raw/=raw.sum()
        mu=np.stack([.1*raw,.9*raw])
        for same,cap in [(True,.01),(False,.03)]:
            out,audit=perturb(mu,grid,np.random.default_rng(42),cap,same)
            np.testing.assert_array_equal(out[:,0],mu[:,0]);np.testing.assert_allclose(out.sum(1),mu.sum(1),atol=1e-12)
            self.assertLessEqual(audit['conditional_cdf_change'],cap+1e-12)
            self.assertGreater(np.max(abs(out-mu)),1e-6)
            if same:self.assertAlmostEqual(float((out*grid).sum()),float((mu*grid).sum()),places=10)

if __name__=='__main__':unittest.main()
