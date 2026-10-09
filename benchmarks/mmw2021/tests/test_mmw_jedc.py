"""Meaningful numerical checks, execute only on the cloud; no pilot gate."""
import json
import sys
import tempfile
import unittest
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mmw_jedc import MMWNetwork, MMWEconomy, complementarity


@unittest.skipUnless(torch.cuda.is_available(), 'cloud CUDA needed')
class MMWChecks(unittest.TestCase):
    def test_kkt_sign(self):
        d=torch.device('cuda')
        s=torch.tensor([0.,1.,0.,1.],device=d)
        c=torch.ones_like(s);h=torch.tensor([.8,1.,1.2,.8],device=d)
        fb=complementarity(s,c,h)
        self.assertTrue(torch.allclose(fb[:2],torch.zeros(2,device=d),atol=1e-6))
        self.assertGreater(float(fb[2].abs()),.1)
        self.assertGreater(float(fb[3].abs()),.1)

    def test_paper_wealth_normalization(self):
        # JME (2021), Eq. (44): the first FB argument is 1-c/w=s/(s+c).
        s=torch.tensor([1.],device='cuda')
        c=torch.tensor([1.],device='cuda')
        h=torch.tensor([.8],device='cuda')
        expected=(.5**2+.2**2)**.5-.5-.2
        self.assertAlmostEqual(float(complementarity(s,c,h)),expected,places=6)

    def test_double_draw_unbiased_identity(self):
        # Sum over all pairs of independent joint outcomes; gradients also match.
        q=torch.tensor([.2,.7,1.1,1.7],device='cuda',requires_grad=True)
        p=torch.tensor([.525,.35,.03125,.09375],device='cuda')
        h=torch.tensor(.9,device='cuda',requires_grad=True)
        product=(p[:,None]*p[None,:]*(q-h)[:,None]*(q-h)[None,:]).sum()
        exact=((p*q).sum()-h).square()
        self.assertTrue(torch.allclose(product,exact,atol=1e-6))
        gp=torch.autograd.grad(product,(q,h),retain_graph=True)
        ge=torch.autograd.grad(exact,(q,h))
        self.assertTrue(all(torch.allclose(x,y,atol=1e-6) for x,y in zip(gp,ge)))

    def test_continuation_gradients_and_resume_stream(self):
        d=torch.device('cuda');edges=torch.tensor([0.,1.,5.,20.,100.,500.],device=d)
        grid=torch.cat((edges.new_zeros(1),(edges[:-1]+edges[1:])/2))
        econ=MMWEconomy(ROOT/'data/jedc2010_calibration.json',edges,d)
        net=MMWNetwork(grid,500.,16,-3.).to(d)
        mu=torch.zeros((2,2,len(grid)),device=d)
        mu[:,0,3]=.1;mu[:,1,3]=.9;z=torch.zeros(2,dtype=torch.long,device=d)
        g=torch.Generator(device=d).manual_seed(42);state=g.get_state()
        loss,_=econ.objective(net,mu,z,g);loss.backward()
        self.assertTrue(all(p.grad is not None and bool(torch.isfinite(p.grad).all()) for p in net.parameters()))
        self.assertGreater(float(net.output.weight.grad.abs().sum()),0.)
        g.set_state(state);again,_=econ.objective(net,mu,z,g)
        self.assertTrue(torch.equal(loss.detach(),again.detach()))
        nm,nz=econ.advance(net,mu,z,g)
        self.assertTrue(torch.allclose(nm.sum((1,2)),torch.ones(2,device=d),atol=1e-6))
        expected=torch.where(nz==0,.1,.04)
        # Model B's published transition entries are rounded to six decimals;
        # bad->good implies .0399976 rather than exactly .04.
        self.assertTrue(torch.allclose(nm[:,0].sum(1),expected,atol=3e-6))


if __name__=='__main__':unittest.main()
