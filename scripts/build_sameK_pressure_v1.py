"""Original five fixed-K shape families, separately labelled pressure states.

No Young, labels, or solver scores. Exact atom/group-mass constraints; positive
interval density tilt sets the same overall K. Not equilibrium observations.
"""
import argparse
import sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from jedc_distribution_v1 import make_edges,midpoint_grid,sha,write_json
from build_jedc_mu_v1 import official_initial,save_states

def tilt_positive(density,weights,x,target):
    log=np.log(np.maximum(density,1e-300))
    def evaluate(lam):
        s=log+lam*x;s-=s.max();p=np.exp(s)*weights;p/=p.sum();return p
    lo,hi=-10.,10.
    for _ in range(100):
        m=(lo+hi)/2
        if evaluate(m)@x<target:lo=m
        else:hi=m
    return evaluate((lo+hi)/2)

def main():
    p=argparse.ArgumentParser();p.add_argument('--output',required=True);args=p.parse_args();out=ROOT/args.output;out.mkdir(parents=True,exist_ok=False)
    edges=make_edges();grid=midpoint_grid(edges);initial,_=official_initial(edges)
    rho=initial[:,0]/initial.sum(1);K=float((initial*grid).sum());x=grid[1:];width=np.diff(edges)
    states=[];zs=[];ids=[];families=[]
    for scale in [.75,1.,1.25]:
        shapes=dict(narrow=np.exp(-.5*((x-K)/(4*scale))**2),broad=np.where(x<100*scale,1.,1e-300),
            polarized=np.exp(-.5*(x/(3*scale))**2)+np.exp(-.5*((x-100)/(3*scale))**2),
            low_spike_tail=np.exp(-.5*(x/(5*scale))**2)+.1*np.exp(-.5*((x-85)/(9*scale))**2),
            high_spike_tail=np.exp(-.5*((x-85)/(8*scale))**2)+.2*np.exp(-.5*((x-10)/(9*scale))**2))
        for family,density in shapes.items():
            for z in [0,1]:
                population=np.array([[.1,.9],[.04,.96]][z]);positive_mass=float((population*(1-rho)).sum())
                positive=tilt_positive(density,width,x,K/positive_mass)
                mu=np.stack([np.r_[population[e]*rho[e],population[e]*(1-rho[e])*positive] for e in [0,1]])
                if abs((mu*grid).sum()-K)>1e-10:raise ValueError('K mismatch')
                states.append(mu);zs.append(z);ids.append('%s_scale%s_z%d'%(family,scale,z));families.append(family)
    save_states(out/'pressure_sameK.npz',states,zs,ids,ids,families,grid,edges)
    write_json(out/'manifest.json',dict(status='supplementary_off_equilibrium_shape_pressure_not_primary_test',
        count=len(states),families=list(shapes),scales=[.75,1.,1.25],target_K=K,zero_conditionals=rho.tolist(),
        population='Model B by z',policy_labels_used=False,young_required=False,
        file_sha256=sha(out/'pressure_sameK.npz'),source_sha256=sha(Path(__file__))))
    print(out,len(states),'states, K=',K)

if __name__=='__main__':main()
