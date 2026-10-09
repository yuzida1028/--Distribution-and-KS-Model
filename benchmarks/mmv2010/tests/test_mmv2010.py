"""Numerical verification to execute on AutoDL only, never local Windows."""
import sys,unittest
if sys.platform!='linux':raise RuntimeError('numerical tests belong on AutoDL')
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'src'))
import numpy as np
from mmv2010_jedc import PolicyTable,fit_alm,generate_shocks
import json
CAL=json.loads((Path(__file__).resolve().parents[1]/'data/jedc2010_calibration.json').read_text())
class Verification(unittest.TestCase):
    def policy(self):
        a=np.linspace(0,500,100)**1 # exact affine saving has no cubic interpolation error
        K=np.linspace(27,50,4)
        t=np.broadcast_to(.8*a,(2,2,4,100)).copy()
        return PolicyTable(CAL,a,K,t,np.array([[0,1],[0,1]]))
    def test_affine_interpolation_and_budget(self):
        p=self.policy();a=np.array([0.,1.,10.,100.,400.]);s=p.query(36,0,a,1)
        np.testing.assert_allclose(s,.8*a,atol=1e-10)
    def test_out_of_domain_is_preserved_failure(self):
        with self.assertRaises(ValueError):self.policy().query(26.,0,10.,1)
    def test_invalid_state_rejected(self):
        with self.assertRaises(ValueError):self.policy().query(36.,2,10.,1)
    def test_prices_match_joint_population_core(self):
        import torch
        from jedc_distribution_v1 import IntervalEconomy,make_edges
        from mmv2010_jedc import prices
        econ=IntervalEconomy(Path(__file__).resolve().parents[1]/'data/jedc2010_calibration.json',torch.tensor(make_edges(),dtype=torch.float64),torch.device('cpu'))
        for z,pi in [(0,[.1,.9]),(1,[.04,.96])]:
            mu=torch.zeros((1,2,len(econ.grid)),dtype=torch.float64)
            mu[0,:,1300]=torch.tensor(pi,dtype=torch.float64)
            K,w,r,tax=econ.prices(mu,torch.tensor([z]))
            got=prices(CAL,float(K),z)
            np.testing.assert_allclose([float(w),float(r),float(tax)],got,atol=1e-12)
    def test_regression_and_fixed_random_stream(self):
        z,e=generate_shocks(CAL,9711,1100,100)
        zz,ee=generate_shocks(CAL,9711,1100,100)
        np.testing.assert_array_equal(z,zz);np.testing.assert_array_equal(e,ee)
        B=np.array([[.12,.966],[.14,.963]]);K=np.empty(len(z));K[0]=36
        for i in range(len(z)-1):K[i+1]=np.exp(B[z[i],0]+B[z[i],1]*np.log(K[i]))
        got,_=fit_alm(K,z,100);np.testing.assert_allclose(got,B,atol=1e-10)
if __name__=='__main__':unittest.main()
