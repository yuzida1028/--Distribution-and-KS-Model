"""Data integrity/conversion audit, never a model-performance eligibility test."""
import argparse
import json
import sys
import time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from jedc_distribution_v1 import make_edges,midpoint_grid,rebin,sha,write_json


def audit_bank(path, expected, parents_seen):
    with np.load(path) as d:
        mu=d['mu'].astype(float);z=d['z'];edges=d['edges'].astype(float);grid=d['grid'].astype(float)
        parents=d['parent_state_id'];variants=d['variant'];ids=d['state_id']
        if len(mu)!=expected or len(set(ids))!=expected:raise ValueError('state count/ID uniqueness '+str(path))
        if not np.all(np.isfinite(mu)) or mu.min()<0:raise ValueError('invalid mu')
        pop=np.where(z[:,None]==0,np.array([.1,.9]),np.array([.04,.96]))
        mass_error=float(np.max(abs(mu.sum((1,2))-1)))
        marginal_error=float(np.max(abs(mu.sum(2)-pop)))
        if mass_error>2e-6 or marginal_error>2e-6:raise ValueError('float32 mass/marginal tolerance')
        if not np.allclose(grid,midpoint_grid(edges),atol=5e-5,rtol=0):raise ValueError('interval grid')
        if np.max(abs(d['K']-(mu*grid).sum((1,2))))>2e-5:raise ValueError('stored K inconsistency')
        if not np.allclose(d['cdf'],np.cumsum(mu,2),atol=2e-6,rtol=0):raise ValueError('CDF encoding')
        if not np.array_equal(d['zero_atom'],d['mu'][:,:,0]):raise ValueError('atom encoding')
        if not np.array_equal(d['positive_interval_mass'],d['mu'][:,:,1:]):raise ValueError('interval mass encoding')
        current=set(parents)
        if current & parents_seen:raise ValueError('cross-split parent leakage')
        parents_seen.update(current)
        raw={str(parents[i]):i for i in range(len(mu)) if variants[i]=='raw'}
        maximum_cdf=maximum_sameK=maximum_localK=maximum_atom=0.
        for i,v in enumerate(variants):
            if v=='raw':continue
            j=raw[str(parents[i])];delta=mu[i]-mu[j]
            cdf=float(np.max(abs(np.cumsum(delta,1))/pop[i,:,None]))
            dk=abs(float((delta*grid).sum()));relative=dk/float((mu[j]*grid).sum())
            atom=float(np.max(abs(delta[:,0])));maximum_cdf=max(maximum_cdf,cdf);maximum_atom=max(maximum_atom,atom)
            if atom!=0:raise ValueError('perturbation changed atom')
            if v=='sameK':
                maximum_sameK=max(maximum_sameK,dk)
                if dk>2e-5 or cdf>.010002:raise ValueError('sameK/CDF constraint after float32')
            elif v=='localK':
                maximum_localK=max(maximum_localK,relative)
                if relative>.020002 or cdf>.030002:raise ValueError('localK/CDF constraint')
            else:raise ValueError('unknown variant')
        probes=[]
        for i in np.linspace(0,len(mu)-1,12,dtype=int):
            for label,target in [('fine',make_edges(True)),('upper1000',make_edges(False,1000))]:
                converted=rebin(mu[i],edges,target);back=rebin(converted,target,edges)
                probes.append(dict(state_id=str(ids[i]),grid=label,mass_error=float(abs(converted.sum()-mu[i].sum())),
                    atom_error=float(np.max(abs(converted[:,0]-mu[i,:,0]))),
                    K_error=float((converted*midpoint_grid(target)).sum()-(mu[i]*grid).sum()),
                    roundtrip_CDF_error=float(np.max(abs(np.cumsum(back-mu[i],1))))))
        return dict(states=len(mu),parents=len(current),mass_error=mass_error,employment_error=marginal_error,
            maximum_perturb_CDF=maximum_cdf,maximum_sameK_error=maximum_sameK,
            maximum_localK_relative_change=maximum_localK,maximum_atom_change=maximum_atom,conversion_probes=probes)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--output',required=True)
    p.add_argument('--fine-output',required=True);args=p.parse_args()
    started=time.perf_counter()
    dataset=ROOT/args.dataset;out=ROOT/args.output;fine=ROOT/args.fine_output
    split=json.loads((dataset/'split_manifest.json').read_text(encoding='utf-8'))
    if split['version']!='mu_functions_v2' or split['status']!='complete':raise ValueError('complete function bank required')
    for name,digest in split['sha256'].items():
        if sha(dataset/name)!=digest:raise ValueError('bank changed '+name)
    seen=set();results={}
    for name,n in [('train',21600),('dev',5400)]:results[name]=audit_bank(dataset/(name+'.npz'),n,seen)
    out.mkdir(parents=True,exist_ok=False);fine.mkdir(parents=True,exist_ok=False)
    target=make_edges(True);target_grid=midpoint_grid(target)
    # A numerical-grid ablation of the SAME uniform-interval functions, not new
    # density draws. Test states will be converted by the frozen evaluator.
    for name in ['train','dev']:
        with np.load(dataset/(name+'.npz')) as d:
            arrays={k:d[k].copy() for k in d.files};source=d['edges'].astype(float)
            converted=np.asarray([rebin(m.astype(float),source,target) for m in d['mu']],dtype=np.float32)
        arrays.update(mu=converted,grid=target_grid.astype(np.float32),edges=target.astype(np.float32),
            K=(converted.astype(float)*target_grid).sum((1,2)),cdf=np.cumsum(converted,2),
            zero_atom=converted[:,:,0],positive_interval_mass=converted[:,:,1:])
        np.savez_compressed(fine/(name+'.npz'),**arrays)
    write_json(fine/'manifest.json',dict(classification='fine-grid rebin of identical parent and perturbed functions',
        parent_directory=args.dataset,transport_sha256=sha(ROOT/'src/jedc_distribution_v1.py'),
        sha256={name+'.npz':sha(fine/(name+'.npz')) for name in ['train','dev']}))
    report=dict(status='data_integrity_complete_no_policy_scores',partition_unit='function_group',banks=results,
        train_sha256=sha(dataset/'train.npz'),dev_sha256=sha(dataset/'dev.npz'),
        transport_sha256=sha(ROOT/'src/jedc_distribution_v1.py'),fine_directory=args.fine_output,
        fine_manifest_sha256=sha(fine/'manifest.json'),test_scores_read=False,
        seconds=time.perf_counter()-started,
        validity_tolerances=dict(float32_mass=2e-6,float32_K=2e-5),
        checks_are='encoding and numerical conversion only; no reference convergence or pilot accuracy threshold')
    write_json(out/'audit.json',report);print('conversion audit complete',out,flush=True)


if __name__=='__main__':main()
