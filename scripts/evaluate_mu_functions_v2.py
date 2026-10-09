"""Separate common-function, same-K and own-policy dynamics; guarded final access."""
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from jedc_distribution_v1 import IntervalEconomy,make_edges,rebin,sha,write_json
from networks_mu_v2 import MuPolicyNetwork as PolicyNetwork
from mu_evaluation_v2 import metrics,quantiles,FrozenGridPolicy
from build_jedc_mu_v1 import official_initial


def save_csv(path,rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='',encoding='utf-8') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)


def verify_plan(planpath):
    plan=json.loads(planpath.read_text(encoding='utf-8'))
    if plan.get('version')!='mu_functions_v2':raise ValueError('wrong plan version')
    for name,h in plan['sha256'].items():
        if sha(ROOT/name)!=h:raise ValueError('changed frozen file '+name)
    for item in plan['configs']:
        if sha(ROOT/item['config'])!=item['sha256']:raise ValueError('changed config')
    if sha(planpath.parent/'evaluation_protocol.json')!=plan['evaluation_protocol_sha256']:raise ValueError('changed protocol')
    return plan


def verify_registry(planpath,registry_path,run):
    registry=json.loads(registry_path.read_text(encoding='utf-8'))
    if registry['plan_sha256']!=sha(planpath):raise ValueError('final registry belongs to different plan')
    entry=next((r for r in registry['runs'] if (ROOT/r['run_dir']).resolve()==run.resolve()),None)
    if entry is None or sha(run/'policy.pt')!=entry['policy_sha256']:raise ValueError('policy not in frozen final registry')
    return registry,entry


def main():
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--config',required=True)
    p.add_argument('--run-dir',required=True);p.add_argument('--split',choices=['dev','test'],default='dev')
    p.add_argument('--grid',choices=['native','fine','upper1000'],default='native')
    p.add_argument('--registry');p.add_argument('--static-only',action='store_true');p.add_argument('--stress-only',action='store_true')
    p.add_argument('--pressure',default='data/jedc_sameK_pressure_cloud_v2');args=p.parse_args()
    planpath=ROOT/args.plan;plan=verify_plan(planpath);run=ROOT/args.run_dir
    item=next((r for r in plan['configs'] if r['config']==args.config),None)
    if item is None:raise ValueError('unknown frozen configuration')
    config=json.loads((ROOT/args.config).read_text());manifest=json.loads((run/'manifest.json').read_text())
    if manifest['sha256']['config']!=item['sha256'] or manifest['model']!=item['kind']:raise ValueError('run/config mismatch')
    dataset=ROOT/plan['data_directory']
    if args.split=='test':
        if not args.registry:raise ValueError('final registry required before test access')
        verify_registry(planpath,ROOT/args.registry,run)
        for f,h in plan['sealed_sha256'].items():
            if sha(dataset/f)!=h:raise ValueError('sealed bank changed')
        for path in plan['sealed_paths']:
            if sha(dataset/path['file'])!=path['sha256']:raise ValueError('sealed shock stream changed')
    suffix=('stress' if args.stress_only else args.split)+'_'+args.grid
    out=run/('evaluation_v2_'+suffix);out.mkdir(exist_ok=False)
    started=time.perf_counter();torch.set_num_threads(1)
    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    with np.load(ROOT/config['train_data_file']) as d:
        input_edges=d['edges'].copy();input_grid=d['grid'].copy()
    eval_edges=input_edges.astype(float) if args.grid=='native' else make_edges(args.grid=='fine',1000 if args.grid=='upper1000' else 500)
    edges=torch.tensor(eval_edges,dtype=torch.float32,device=device)
    econ=IntervalEconomy(ROOT/config['calibration_file'],edges,device,policy_mode=config['policy_mode'])
    # Changing numerical domain does not silently change the trained policy cap.
    econ.asset_max=config['asset_max']
    hidden=config['hidden'] if item['kind']=='deeponet' else config['mlp_hidden']
    net=PolicyNetwork(item['kind'],len(input_grid),config['asset_max'],hidden,config['basis'],
        macro_features=config['macro_features'],grid=torch.tensor(input_grid,device=device)).to(device)
    net.load_state_dict(torch.load(run/'policy.pt',map_location=device,weights_only=True));net.eval()
    adapter=FrozenGridPolicy(net,edges,torch.tensor(input_edges,device=device)).to(device)
    if args.stress_only:
        pressure=ROOT/args.pressure;stress_manifest=json.loads((pressure/'manifest.json').read_text())
        bankfile=pressure/'pressure_sameK.npz'
        if sha(bankfile)!=stress_manifest['file_sha256']:raise ValueError('pressure hash mismatch')
    else:bankfile=dataset/('sealed/test.npz' if args.split=='test' else 'dev.npz')
    with np.load(bankfile) as d:
        source_edges=d['edges'].astype(float);mu=d['mu'].copy();zs=d['z'].copy()
        ids=d['state_id'].copy() if 'state_id' in d else np.asarray(['pressure_%d'%i for i in range(len(mu))])
        parents=d['parent_state_id'].copy() if 'parent_state_id' in d else ids
        variants=d['variant'].copy() if 'variant' in d else np.asarray(['raw']*len(mu))
    if args.stress_only:variants=np.asarray(['raw']*len(mu))
    rows=[];responses=[];failures=[];raw_by_parent={};dynamic_summary=[]
    for i in range(len(mu)):
        group=int(str(parents[i])[1:3]) if not args.stress_only else i
        try:
            current=rebin(mu[i].astype(float),source_edges,eval_edges)
            m=torch.tensor(current[None],dtype=torch.float32,device=device);z=torch.tensor([int(zs[i])],device=device)
            row=dict(state_id=str(ids[i]),parent=str(parents[i]),group_id=group,variant=str(variants[i]),**metrics(econ,adapter,m,z))
            rows.append(row)
            with torch.no_grad():a,e=econ.queries(1);s,_=econ.policy(adapter,m,z,a,e);s=s.cpu().numpy()[0]
            if variants[i]=='raw':raw_by_parent[str(parents[i])]=(s,current,row)
            elif variants[i]=='sameK':
                s0,m0,r0=raw_by_parent[str(parents[i])]
                responses.append(dict(parent=str(parents[i]),group_id=group,K_difference=row['K']-r0['K'],
                    policy_response_mass_MAE=float((m0*abs(s-s0)).sum()),policy_response_max=float(abs(s-s0).max()),
                    KKT_raw=r0['KKT'],KKT_sameK=row['KKT']))
        except Exception as exc:
            failures.append(dict(stage='common',state_id=str(ids[i]),group_id=group,error=str(exc)))
        if i%300==0:print('common %s %s %d/%d'%(args.split,args.grid,i,len(mu)),flush=True)
    save_csv(out/'common_states.csv',rows)
    if args.split=='test' and not args.stress_only:
        with np.load(dataset/'sealed/shape.npz') as d:
            pair_mu=d['mu'].copy();pair_z=d['z'].copy();pair_ids=d['parent_state_id'].copy();pair_edges=d['edges'].astype(float)
        for i in range(0,len(pair_mu),2):
            try:
                choices=[];scored=[];masses=[]
                for j in [i,i+1]:
                    current=rebin(pair_mu[j].astype(float),pair_edges,eval_edges);masses.append(current)
                    m=torch.tensor(current[None],dtype=torch.float32,device=device);z=torch.tensor([int(pair_z[j])],device=device)
                    scored.append(metrics(econ,adapter,m,z))
                    with torch.no_grad():a,e=econ.queries(1);s,_=econ.policy(adapter,m,z,a,e);choices.append(s.cpu().numpy()[0])
                responses.append(dict(parent=str(pair_ids[i]),group_id=int(str(pair_ids[i])[1:3]),
                    K_difference=scored[1]['K']-scored[0]['K'],policy_response_mass_MAE=float((masses[0]*abs(choices[1]-choices[0])).sum()),
                    policy_response_max=float(abs(choices[1]-choices[0]).max()),KKT_raw=scored[0]['KKT'],KKT_sameK=scored[1]['KKT']))
            except Exception as exc:failures.append(dict(stage='shape',parent=str(pair_ids[i]),error=str(exc)))
    save_csv(out/'sameK_shape.csv',responses)
    summaries=[]
    for group in sorted({r['group_id'] for r in rows}):
        original=[r for r in rows if r['group_id']==group and r['variant']=='raw']
        if original:summaries.append(dict(group_id=group,n=len(original),**{k:float(np.mean([r[k] for r in original])) for k in ['KKT','P90','P99','max_populated_KKT']}))
    save_csv(out/'common_original_group_summary.csv',summaries)
    if not args.static_only and not args.stress_only:
        split_manifest=json.loads((dataset/'split_manifest.json').read_text())
        for path in [r for r in split_manifest['paths'] if r['split']==args.split]:
            with np.load(dataset/path['file']) as d:shocks=d['shocks'].copy()
            initial,_=official_initial(eval_edges);m=torch.tensor(initial[None],dtype=torch.float32,device=device)
            dyn=[];sample_mu=[];sample_dates=[];failed=False;pid=path['path_id'];reference_cdf=np.cumsum(initial,1)
            try:
                for t,zv in enumerate(shocks):
                    z=torch.tensor([int(zv)],device=device);current=m.cpu().numpy()[0];grid=econ.grid.cpu().numpy()
                    scored=metrics(econ,adapter,m,z);qs=quantiles(current,eval_edges)
                    row=dict(path_id=pid,t=t,phase='transition' if t<1000 else 'post_burn',z=int(zv),**scored,
                        mass_error=abs(float(current.sum())-1),employment_error=abs(float(current[0].sum())-[.1,.04][int(zv)]),
                        zero_atom=float(current[:,0].sum()),tail_mass=float(current[:,grid>=490].sum()),tail_capital=float((current[:,grid>=490]*grid[grid>=490]).sum()),
                        moment2=float((current*grid**2).sum()),moment3=float((current*grid**3).sum()),moment4=float((current*grid**4).sum()),
                        q10=qs[0],q50=qs[1],q90=qs[2],q99=qs[3],CDF_distance_own_initial=float(np.max(abs(np.cumsum(current,1)-reference_cdf))))
                    dyn.append(row)
                    if t%10==0 or t==len(shocks)-1:
                        sample_mu.append(current.copy());sample_dates.append(t)
                    if row['mass_error']>1e-4 or row['employment_error']>1e-4:raise ValueError('dynamic mass/employment drift')
                    if t==len(shocks)-1:break
                    with torch.no_grad():
                        a,e=econ.edge_queries(1);s,_=econ.policy(adapter,m,z,a,e);m=econ.push_distribution(m,z,s,int(shocks[t+1]))
                    if t%500==0:print('dynamic %s path %d date %d'%(args.split,pid,t),flush=True)
            except Exception as exc:
                failed=True;failures.append(dict(stage='dynamic',path_id=pid,date=len(dyn),error=str(exc)))
            save_csv(out/('path_%02d_own_dynamic.csv'%pid),dyn)
            np.savez_compressed(out/('path_%02d_distribution_samples.npz'%pid),mu=np.asarray(sample_mu),dates=np.asarray(sample_dates),edges=eval_edges)
            for phase in ['transition','post_burn']:
                subset=[r for r in dyn if r['phase']==phase]
                dynamic_summary.append(dict(path_id=pid,phase=phase,failed=failed,dates=len(subset),
                    KKT=float(np.mean([r['KKT'] for r in subset])) if subset else None,
                    P99=float(np.mean([r['P99'] for r in subset])) if subset else None,K_final=dyn[-1]['K'] if dyn else None))
    save_csv(out/'dynamic_path_summary.csv',dynamic_summary);write_json(out/'failures.json',failures)
    if torch.cuda.is_available():torch.cuda.synchronize()
    write_json(out/'report.json',dict(status='complete' if not failures else 'complete_with_failures',
        split=args.split,grid=args.grid,stress_only=args.stress_only,label=item['label'],seed=item['seed'],phase=item['phase'],
        policy_sha256=sha(run/'policy.pt'),config_sha256=item['sha256'],bank_sha256=sha(bankfile),
        plan_sha256=sha(planpath),expected_states=len(mu),scored_states=len(rows),failure_count=len(failures),
        groups=summaries,sameK_pairs=len(responses),dynamic_paths=dynamic_summary,seconds=time.perf_counter()-started,
        final_data_read=args.split=='test',dynamic_KKT_sampling='every date',
        interpretation='Exogenous common functions, fixed-K pairs, stress and own-policy dynamic states are separate; no welfare/equilibrium truth claims.',
        budget_checks='tax/UI and midpoint resource identities are sanity checks; affine discrepancy is numerical quadrature error'))


if __name__=='__main__':main()
