"""42 independent paths, whole-path split, raw interval mu and local shifts.

No network scores are read. Final data are kept in sealed/, and no final policy
evaluation occurs here. Failure manifests and partial paths are retained.
"""
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import numpy as np
import torch
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from jedc_distribution_v1 import (VERSION, IntervalEconomy,
    sha, write_json, make_edges, midpoint_grid, rebin, transition_numpy)
from neural_state_source_v1 import NeuralStateSource

def official_initial(edges):
    with np.load(ROOT/'data/jedc2010/initial_distribution.npz') as d:
        mu = d['mu'].astype(float)
        source_grid = d['grid'].astype(float)
    source_edges = np.arange(1001)*.1
    if not np.allclose(source_grid,midpoint_grid(source_edges)): raise ValueError('official grid mapping')
    # Correct only the documented float32 population rounding, separately by e.
    population = np.array([.1,.9])
    correction = population/mu.sum(1)
    mu *= correction[:,None]
    return rebin(mu,source_edges,edges), {'group_rounding_factors':correction.tolist(),
        'official_zero_joint':mu[:,0].tolist(),'K_before_rebin':float((mu*source_grid).sum())}

def stats(mu, grid, edges, z, transition=None):
    p = mu.sum(1)
    K = float((mu*grid).sum())
    return dict(K=K, mass_error=float(abs(mu.sum()-1)), unemployment=float(p[0]),
        unemployment_error=float(abs(p[0]-[.1,.04][z])),
        zero_unemployed=float(mu[0,0]),zero_employed=float(mu[1,0]),
        low_asset_mass=float(mu[:,grid<=5].sum()),
        tail_mass=float(mu[:,grid>=edges[-1]-10].sum()),
        tail_capital=float((mu[:,grid>=edges[-1]-10]*grid[grid>=edges[-1]-10]).sum()),
        second_moment=float((mu*grid**2).sum()),third_moment=float((mu*grid**3).sum()),
        fourth_moment=float((mu*grid**4).sum()),min_mass=float(mu.min()))

def perturb(mu, grid, rng, cap, same_k):
    """Smooth mixtures of local positive-interval transfers; exact constraints.

    Redistribute a random fraction of each donor interval to two surrounding
    positive intervals, weighted to preserve its midpoint mean. For varying K,
    use a one-sided transfer. The atom is untouched in both variants.
    """
    candidate = mu.copy()
    for e in range(2):
        n = len(grid)
        donor = np.arange(2,n-1)
        reach = rng.integers(2,80,size=len(donor))
        left = np.maximum(1,donor-reach)
        right = np.minimum(n-1,donor+reach)
        amount = mu[e,donor]*rng.uniform(.05,.5,len(donor))
        d = np.zeros(n)
        np.add.at(d,donor,-amount)
        if same_k:
            wr = (grid[donor]-grid[left])/(grid[right]-grid[left])
            np.add.at(d,left,amount*(1-wr));np.add.at(d,right,amount*wr)
        else:
            target = right if rng.random()<.5 else left
            np.add.at(d,target,amount)
        cdf = np.max(abs(np.cumsum(d)))/mu[e].sum()
        scale = min(1., cap/max(cdf,1e-30))
        candidate[e] += scale*d
    delta = candidate-mu
    K = float((mu*grid).sum())
    dk = float((delta*grid).sum())
    if not same_k and abs(dk) > .02*K:
        candidate = mu+delta*(.02*K/abs(dk))
    cdf_error = np.max(abs(np.cumsum(candidate-mu,axis=1)),axis=1)/mu.sum(1)
    if candidate.min() < -1e-14 or not np.allclose(candidate.sum(1),mu.sum(1),atol=1e-12,rtol=0): raise ValueError('perturb mass')
    if not np.array_equal(candidate[:,0],mu[:,0]): raise ValueError('perturb atom')
    if cdf_error.max()>cap+1e-12 or (same_k and abs((candidate*grid).sum()-K)>1e-10): raise ValueError('perturb shape/moment')
    return candidate, dict(source='positive_interval_mass_transfer',conditional_cdf_change=float(cdf_error.max()),
        relative_K_change=float(((candidate-mu)*grid).sum()/K))

def save_states(path, states, zs, ids, parents, variants, grid, edges):
    mu = np.asarray(states,dtype=np.float32)
    np.savez_compressed(path,mu=mu,z=np.asarray(zs,dtype=np.int64),grid=grid.astype(np.float32),
        edges=edges.astype(np.float32),K=(mu.astype(float)*grid).sum((1,2)),
        cdf=np.cumsum(mu,axis=2),zero_atom=mu[:,:,0],positive_interval_mass=mu[:,:,1:],
        state_id=np.asarray(ids),parent_state_id=np.asarray(parents),variant=np.asarray(variants))

def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output',required=True)
    p.add_argument('--source-config',required=True)
    p.add_argument('--source-checkpoint',required=True)
    p.add_argument('--source-kind',choices=['deeponet','mlp'],default='mlp')
    p.add_argument('--device',default='cpu')
    p.add_argument('--paths',type=int,default=42)
    p.add_argument('--periods',type=int,default=4000)
    p.add_argument('--burn',type=int,default=1000)
    p.add_argument('--stride',type=int,default=10)
    p.add_argument('--seed',type=int,default=202609299700)
    p.add_argument('--smoke',action='store_true',help='isolated integrity fixture, never a training eligibility pilot')
    args = p.parse_args()
    if not args.smoke and (args.paths,args.periods,args.burn,args.stride)!=(42,4000,1000,10): raise ValueError('formal dimensions fixed by protocol')
    if args.periods<=args.burn or args.paths<3: raise ValueError('invalid dimensions')
    out = ROOT/args.output
    out.mkdir(parents=True,exist_ok=False)
    for sub in ['train_paths','dev_paths','sealed']: (out/sub).mkdir()
    started = time.perf_counter()
    torch.set_num_threads(1)
    calibration_path = ROOT/'data/jedc2010_calibration.json'
    cal = json.loads(calibration_path.read_text(encoding='utf-8'))
    transition = np.array(cal['joint_transition'],dtype=float)
    edges = make_edges();grid = midpoint_grid(edges)
    ref=NeuralStateSource(ROOT/args.source_config,ROOT/args.source_checkpoint,args.source_kind,calibration_path,edges,args.device)
    initial, initial_audit = official_initial(edges)
    ref_manifest = dict(**ref.provenance,
        calibration=cal,calibration_sha256=sha(calibration_path),
        initial_sha256=sha(ROOT/'data/jedc2010/initial_distribution.npz'),
        transport_sha256=sha(ROOT/'src/jedc_distribution_v1.py'),
        K_range=ref.capitals.tolist(),asset_range=[float(ref.assets[0]),float(ref.assets[-1])],
        convergence_evidence='Approximate frozen neural policy; no precise equilibrium or representativeness claim.',
        young_required=False,policy_labels_used=False,
        initial_conversion=initial_audit,representation=VERSION)
    write_json(out/'state_source_manifest.json',ref_manifest)
    # Compatibility metadata filename. It denotes a STATE SOURCE, never a
    # requirement for a traditional reference or supervised policy labels.
    write_json(out/'reference_manifest.json',ref_manifest)
    master = np.random.SeedSequence(args.seed)
    streams = master.spawn(args.paths)
    records=[];diagnostics=[];banks={s:dict(states=[],zs=[],ids=[],parents=[],variants=[]) for s in ['train','dev','test','shape']}
    dates = list(range(args.burn,args.periods,args.stride))
    split = ['train']*24+['dev']*6+['test']*12 if not args.smoke else ['train','dev']+['test']*(args.paths-2)
    protocol = dict(version='jedc_mu_v1',formal=not args.smoke,period_definition='4000 states t=0..3999, 3999 realized transitions',
        root_seed=args.seed,paths=args.paths,periods=args.periods,burn=args.burn,stride=args.stride,
        sampling_dates=dates,transient_dates=list(range(args.burn)),
        shape_dates=[dates[i] for i in np.linspace(0,len(dates)-1,min(10,len(dates)),dtype=int)],
        test_score_access=False,employment_transition='joint block divided by P(z_next|z), source saving push then employment mixing',
        input='joint mass [zero atom, positive uniform interval masses] per employment, full mu plus computed K and z',
        generator_sha256=sha(Path(__file__)),transport_sha256=sha(ROOT/'src/jedc_distribution_v1.py'))
    write_json(out/'split_manifest.json',{**protocol,'status':'building','paths':[]})
    try:
        for pid,stream in enumerate(streams):
            shock_stream,shift_stream = stream.spawn(2)
            rng = np.random.default_rng(shock_stream);shift_rng=np.random.default_rng(shift_stream)
            shocks = np.zeros(args.periods,dtype=np.int8)
            for t in range(1,args.periods):
                current=int(shocks[t-1]);p_bad=transition[current*2,:2].sum()
                shocks[t]=int(rng.random()>=p_bad)
            subgroup=split[pid];bank=banks[subgroup]
            directory='sealed' if subgroup=='test' else subgroup+'_paths'
            pathfile=out/directory/('path_%02d.npz'%pid)
            mu=initial.copy();transient=[];sampled=[];path_z=[]
            max_push=0.;max_mass=0.;minK=1e100;maxK=-1e100;max_overflow=0.;max_overflow_K=0.
            for t,zraw in enumerate(shocks):
                z=int(zraw);K=float((mu*grid).sum());minK=min(minK,K);maxK=max(maxK,K)
                if not ref.capitals[0]<=K<=ref.capitals[-1]: raise ValueError('path %d t=%d: K=%g outside reference'%(pid,t,K))
                if t<args.burn: transient.append(mu.astype(np.float32))
                if t in dates:
                    sid='p%02d_t%04d'%(pid,t);sampled.append(mu.astype(np.float32));path_z.append(z)
                    for variant,cap,same in [('raw',0,True)]+([] if subgroup=='test' else [('sameK',.01,True),('localK',.03,False)]):
                        m=mu if variant=='raw' else perturb(mu,grid,shift_rng,cap,same)[0]
                        bank['states'].append(m.astype(np.float32));bank['zs'].append(z);bank['ids'].append(sid+'_'+variant);bank['parents'].append(sid);bank['variants'].append(variant)
                        diagnostics.append(dict(path_id=pid,split=subgroup,date=t,variant=variant,
                            **stats(m,grid,edges,z),conditional_cdf_change=float(np.max(abs(np.cumsum(m-mu,axis=1))/mu.sum(1)[:,None])),
                            relative_K_change=float(((m-mu)*grid).sum()/K)))
                    if subgroup=='test' and t in protocol['shape_dates']:
                        m,audit=perturb(mu,grid,shift_rng,.03,True)
                        shape=banks['shape'];shape['states'].extend([mu.astype(np.float32),m.astype(np.float32)])
                        shape['zs'].extend([z,z]);shape['ids'].extend([sid+'_raw',sid+'_sameK']);shape['parents'].extend([sid,sid]);shape['variants'].extend(['raw','sameK'])
                if t==args.periods-1: break
                ref.mu=mu
                choices=ref.choices(edges,K,z)
                overflow=(np.maximum(choices[:,:-1],choices[:,1:])>edges[-1])
                overflow_mass=float(mu[:,1:][overflow].sum())
                overflow_K=float((mu[:,1:]*np.maximum(0,np.maximum(choices[:,:-1],choices[:,1:])-edges[-1])).sum())
                max_overflow=max(max_overflow,overflow_mass);max_overflow_K=max(max_overflow_K,overflow_K)
                if overflow_mass>1e-12 or overflow_K>1e-9: raise ValueError('reference asset bound exceeded: mass=%g K upper loss=%g'%(overflow_mass,overflow_K))
                # Explicit numerical-tail tolerance, never an unreported clip.
                # This also handles arbitrary policies on empty upper bins.
                choices=np.minimum(choices,edges[-1])
                nz=int(shocks[t+1])
                next_mu=transition_numpy(mu,z,nz,choices,edges,transition)
                # Affine saving mean within each original uniform interval.
                expectedK=float((mu[:,0]*choices[:,0]).sum()+(mu[:,1:]*(choices[:,:-1]+choices[:,1:])/2).sum())
                error=float((next_mu*grid).sum()-expectedK);max_push=max(max_push,abs(error));max_mass=max(max_mass,abs(next_mu.sum()-1))
                if next_mu.min()<-1e-14 or max_mass>1e-8: raise ValueError('transport conservation')
                mu=next_mu
            np.savez_compressed(pathfile,shocks=shocks,transient_mu=np.asarray(transient),raw_mu=np.asarray(sampled),raw_z=np.asarray(path_z),dates=np.asarray(dates),grid=grid,edges=edges)
            record=dict(path_id=pid,split=subgroup,shock_spawn_key=list(shock_stream.spawn_key),perturb_spawn_key=list(shift_stream.spawn_key),
                file=str(pathfile.relative_to(out)),sha256=sha(pathfile),max_capital_push_error=max_push,max_mass_error=max_mass,
                K_min=minK,K_max=maxK,states=len(sampled),max_asset_overflow_mass=max_overflow,
                max_asset_overflow_capital_upper_loss=max_overflow_K)
            records.append(record)
            write_json(out/'split_manifest.json',{**protocol,'status':'building','paths':records})
            print('path %02d %s complete: K %.6f..%.6f push error %.3g'%(pid,subgroup,minK,maxK,max_push),flush=True)
        for name in banks:
            b=banks[name];target=out/('sealed/'+name+'.npz' if name in ['test','shape'] else name+'.npz')
            save_states(target,b['states'],b['zs'],b['ids'],b['parents'],b['variants'],grid,edges)
        with (out/'distribution_diagnostics.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=diagnostics[0].keys());w.writeheader();w.writerows(diagnostics)
        hashes={str(x.relative_to(out)):sha(x) for x in [out/'train.npz',out/'dev.npz',out/'sealed/test.npz',out/'sealed/shape.npz',out/'distribution_diagnostics.csv']}
        write_json(out/'split_manifest.json',{**protocol,'status':'complete','paths':records,'sha256':hashes,
            'counts':{k:len(v['states']) for k,v in banks.items()},'seconds':time.perf_counter()-started})
        (out/'sealed/DO_NOT_EVALUATE.txt').write_text('No model scores before experiment/evaluation protocol freeze. All final paths and shape pairs are preselected.\n')
    except Exception as exc:
        write_json(out/'failure.json',dict(status='failed_preserved',error=str(exc),completed_paths=records,
            elapsed_seconds=time.perf_counter()-started,last_path_id=pid,last_date=t,last_K=K))
        raise

if __name__=='__main__': main()
