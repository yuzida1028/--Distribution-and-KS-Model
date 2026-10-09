"""Independent common development and own-policy dynamics; sealed test prohibited."""
import argparse
import csv
import json
import platform
import sys
import time
import traceback
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'));sys.path.insert(0,str(ROOT/'scripts'))
from mmv2010_protocol import verify
from mmv2010_bridge import TorchMMVPolicy, MMVIntervalEconomy, perceived_kkt
from solve_mmv2010 import initial_joint as official_mu
from mu_evaluation_v2 import metrics, quantiles
from jedc_distribution_v1 import sha, write_json, rebin, make_edges, midpoint_grid


def save_csv(path, rows):
    if not rows:return
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with Path(path).open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields);writer.writeheader();writer.writerows(rows)


def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--run-dir',required=True)
    p.add_argument('--static-only',action='store_true');p.add_argument('--output')
    a=p.parse_args()
    if platform.system()!='Linux' or not torch.cuda.is_available(): raise RuntimeError('cloud GPU only')
    b=verify(a.binding);run=Path(a.run_dir);cfg=json.loads((run/'config.json').read_text())
    manifest=json.loads((run/'manifest.json').read_text())
    if sha(run/'config.json')!=manifest['config_sha256']:raise ValueError('run config changed')
    if manifest['binding_sha256']!=sha(a.binding): raise ValueError('run belongs to different binding')
    if not any(x['seed']==cfg['seed'] and x['profile']==cfg['profile'] and x['sha256']==manifest['config_sha256'] for x in b['configs']): raise ValueError('run config mismatch')
    report=json.loads((run/'report.json').read_text())
    if report['status']!='converged' or report['policy_sha256']!=sha(run/'policy_table.npz'): raise ValueError('policy changed or unconverged')
    out=Path(a.output) if a.output else run/('development_static' if a.static_only else 'development_with_dynamics')
    out.mkdir(parents=True,exist_ok=False);started=time.perf_counter();device=torch.device('cuda:0')
    with np.load(b['input_paths']['dev']) as d:
        edges=d['edges'].copy();grid=torch.tensor(d['grid'].copy(),device=device)
        masses=d['mu'].copy();zs=d['z'].copy();ids=d['state_id'].copy()
        parents=d['parent_state_id'].copy();variants=d['variant'].copy()
    if cfg['asset_max']!=500:
        source_edges=edges;edges=make_edges(upper=cfg['asset_max'])
        masses=np.stack([rebin(m,source_edges,edges) for m in masses]).astype(np.float32)
        grid=torch.tensor(midpoint_grid(edges),dtype=torch.float32,device=device)
    torch.set_num_threads(1)
    econ=MMVIntervalEconomy(b['input_paths']['calibration'],torch.tensor(edges,dtype=torch.float32,device=device),device)
    net=TorchMMVPolicy(run/'policy_table.npz').to(device);net.eval()
    rows=[];failures=[];pairs=[];raw={};dynamic_summary=[]
    for i in range(len(masses)):
        try:
            mu=torch.tensor(masses[i:i+1],device=device);z=torch.tensor([int(zs[i])],device=device)
            scored=metrics(econ,net,mu,z)
            scored['perceived_ALM_KKT_secondary']=perceived_kkt(econ,net,mu,z)
            forecast=float(net.policy_table.predict_K(scored['K'],int(zs[i])))
            scored['ALM_K_prediction']=forecast
            scored['ALM_actual_affine_K_error']=forecast-scored['one_step_affine_policy_K']
            scored['perceived_forecast_clipped']=int(forecast<cfg['capital_min'] or forecast>cfg['capital_max'])
            parent=str(parents[i]);group=int(parent[1:3])
            rows.append(dict(state_id=str(ids[i]),parent=parent,group_id=group,variant=str(variants[i]),**scored))
            with torch.no_grad():aa,ee=econ.queries(1);s,_=econ.policy(net,mu,z,aa,ee);choices=s.cpu().numpy()[0]
            if variants[i]=='raw':raw[parent]=(choices,masses[i],scored)
            elif variants[i]=='sameK':
                s0,m0,r0=raw[parent]
                pairs.append(dict(parent=parent,group_id=group,K_difference=scored['K']-r0['K'],
                    KKT_raw=r0['KKT'],KKT_sameK=scored['KKT'],policy_response_mass_MAE=float((m0*abs(choices-s0)).sum()),
                    policy_response_max=float(abs(choices-s0).max())))
        except Exception:
            failures.append(dict(stage='common',state_id=str(ids[i]),traceback=traceback.format_exc()))
        if i%300==0:print(f'MMV common development {i}/{len(masses)}',flush=True)
    save_csv(out/'common_states.csv',rows);save_csv(out/'sameK_shape.csv',pairs)
    groups=[]
    for group in sorted({r['group_id'] for r in rows}):
        selected=[r for r in rows if r['group_id']==group and r['variant']=='raw']
        if selected:groups.append(dict(group_id=group,n=len(selected),**{
            k:float(np.mean([r[k] for r in selected])) for k in ['KKT','P90','P99','max_populated_KKT']}))
    save_csv(out/'common_original_group_summary.csv',groups)
    if not a.static_only:
        split_path=Path(b['input_paths']['split']);split=json.loads(split_path.read_text())
        for entry in split['paths']:
            if entry['split']!='dev':continue
            shock_path=split_path.parent/entry['file']
            if sha(shock_path)!=entry['sha256']:raise ValueError('development shock hash changed')
            with np.load(shock_path) as d:shocks=d['shocks'].copy()
            mu=torch.tensor(official_mu(b['input_paths']['initial'],edges)[None],dtype=torch.float32,device=device)
            trajectory=[];failed=False;pid=entry['path_id']
            try:
                for t,zvalue in enumerate(shocks):
                    z=torch.tensor([int(zvalue)],device=device);current=mu.cpu().numpy()[0]
                    score=metrics(econ,net,mu,z);qs=quantiles(current,edges)
                    row=dict(path_id=pid,t=t,phase='transition' if t<1000 else 'post_burn',z=int(zvalue),**score,
                        mass_error=abs(float(current.sum())-1),employment_error=abs(float(current[0].sum())-[.1,.04][int(zvalue)]),
                        zero_atom=float(current[:,0].sum()),q10=qs[0],q50=qs[1],q90=qs[2],q99=qs[3],
                        tail_mass=float(current[:,grid.cpu().numpy()>=490].sum()))
                    trajectory.append(row)
                    if row['mass_error']>1e-4 or row['employment_error']>1e-4 or score['price_capital_floor_active']:
                        raise ValueError('invalid dynamic state/price floor')
                    if t+1<len(shocks):
                        with torch.no_grad():
                            aa,ee=econ.edge_queries(1);s,_=econ.policy(net,mu,z,aa,ee)
                            mu=econ.push_distribution(mu,z,s,int(shocks[t+1]))
                    if t%500==0:print(f'MMV development path {pid} date {t}',flush=True)
            except Exception:
                failed=True;failures.append(dict(stage='dynamic',path_id=pid,date=len(trajectory),traceback=traceback.format_exc()))
            save_csv(out/f'path_{pid:02d}_own_dynamic.csv',trajectory)
            for phase in ['transition','post_burn']:
                subset=[r for r in trajectory if r['phase']==phase]
                dynamic_summary.append(dict(path_id=pid,phase=phase,failed=failed,dates=len(subset),
                    KKT=float(np.mean([r['KKT'] for r in subset])) if subset else None,
                    P99=float(np.mean([r['P99'] for r in subset])) if subset else None))
    save_csv(out/'dynamic_path_summary.csv',dynamic_summary);write_json(out/'failures.json',failures)
    torch.cuda.synchronize()
    write_json(out/'report.json',dict(status='complete' if not failures else 'complete_with_failures',
        expected_states=len(masses),scored_states=len(rows),groups=groups,sameK_pairs=len(pairs),failure_count=len(failures),
        dynamic_paths=dynamic_summary,seconds=time.perf_counter()-started,final_data_read=False,
        python=platform.python_version(), torch=torch.__version__, torch_cuda=torch.version.cuda, numpy=np.__version__,
        profile=cfg['profile'], policy_domain=cfg['asset_max'], primary_comparable=cfg['profile']!='author_domain',
        interpolation_audit=net.policy_table.audit, solver_device='CPU', evaluator_peak_GPU_MiB=torch.cuda.max_memory_allocated()/2**20, evaluator_device=torch.cuda.get_device_name(0),
        binding_sha256=sha(a.binding),policy_sha256=sha(run/'policy_table.npz'),dev_sha256=sha(b['input_paths']['dev']),
        interpretation='partial evaluations are not successful primary means; common functions and own dynamics separate'))


if __name__=='__main__':main()
