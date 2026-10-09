"""Cloud-only MMW adaptation, exact RNG/online-state resume, development only."""
import argparse
import csv
import json
import os
import platform
import random
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'server'))
from run_mmw_jedc import verify_binding
from jedc_distribution_v1 import sha, write_json
from mmw_jedc import MMWNetwork, MMWEconomy, official_mu
from mu_evaluation_v2 import metrics


def atomic_save(value, path):
    temporary=path.with_suffix('.tmp')
    torch.save(value,temporary)
    os.replace(temporary,path)


def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--config',required=True)
    p.add_argument('--run-dir',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if platform.system()!='Linux' or not torch.cuda.is_available(): raise RuntimeError('Linux cloud CUDA required')
    binding_path=Path(a.binding);binding=verify_binding(binding_path.parent)
    cfgpath=Path(a.config);cfg=json.loads(cfgpath.read_text())
    if not any(x['path']==str(cfgpath.resolve()) and x['sha256']==sha(cfgpath) for x in binding['configs']): raise ValueError('unregistered configuration')
    seed=cfg['seed'];random.seed(seed);np.random.seed(seed);torch.manual_seed(seed);torch.cuda.manual_seed_all(seed)
    # This is a fresh training process, so peak allocation starts from zero.
    # The RiverMind CUDA build rejects reset_peak_memory_stats before the
    # first training allocation ("Invalid device argument"). Omit the reset;
    # max_memory_allocated below still records this process's actual peak.
    device=torch.device('cuda:0')
    inputs=binding['input_paths']
    with np.load(inputs['train']) as d:
        edges=d['edges'].copy();grid=torch.tensor(d['grid'].copy(),device=device)
        bank_mu=torch.tensor(d['mu'].copy(),device=device) if cfg['sampling']=='functions' else None
        bank_z=torch.tensor(d['z'].copy(),device=device).long() if bank_mu is not None else None
    with np.load(inputs['dev']) as d:
        # Fixed, balanced raw-state development subset; never chosen by error.
        indices=np.flatnonzero(d['variant']=='raw')
        selection=indices[np.linspace(0,len(indices)-1,cfg['development_raw_states'],dtype=int)]
        dev_mu=torch.tensor(d['mu'][selection].copy(),device=device)
        dev_z=torch.tensor(d['z'][selection].copy(),device=device).long()
    econ=MMWEconomy(inputs['calibration'],torch.tensor(edges,device=device),device)
    net=MMWNetwork(grid,cfg['asset_max'],cfg['hidden'],cfg['share_bias']).to(device)
    optimizer=torch.optim.Adam(net.parameters(),lr=cfg['learning_rate'])
    streams={k:torch.Generator(device=device).manual_seed(seed+offset) for k,offset in [('batch',100000),('objective',200000),('simulation',300000)]}
    initial=official_mu(inputs['initial'],edges)
    online_mu=torch.tensor(np.repeat(initial[None],cfg['batch_size'],axis=0),dtype=torch.float32,device=device)
    online_z=torch.zeros(cfg['batch_size'],dtype=torch.long,device=device)
    run=Path(a.run_dir);history=[];devhistory=[];start=1;elapsed=0.;simulation_dates=0;update_seconds=0.;development_seconds=0.
    max_online_mass_correction=0.;sum_online_mass_correction=0.
    environment=dict(gpu=torch.cuda.get_device_name(0),gpu_bytes=torch.cuda.get_device_properties(0).total_memory,
                     python=platform.python_version(),torch=torch.__version__,cuda=torch.version.cuda)
    current=dict(binding_sha256=sha(binding_path),config_sha256=sha(cfgpath),environment=environment)
    if a.resume:
        old=json.loads((run/'manifest.json').read_text())
        if any(old[k]!=current[k] for k in current): raise ValueError('resume hashes/hardware/software changed')
        saved=torch.load(run/'checkpoint.pt',map_location=device,weights_only=False)
        net.load_state_dict(saved['network']);optimizer.load_state_dict(saved['optimizer'])
        for k,g in streams.items(): g.set_state(saved['streams'][k].cpu())
        torch.set_rng_state(saved['torch_rng'].cpu());torch.cuda.set_rng_state_all([x.cpu() for x in saved['cuda_rng']])
        random.setstate(saved['python_rng']);np.random.set_state(saved['numpy_rng'])
        online_mu=saved['online_mu'];online_z=saved['online_z'];history=saved['history'];devhistory=saved['devhistory']
        start=saved['step']+1;elapsed=saved['effective_seconds'];simulation_dates=saved['simulation_dates']
        update_seconds=saved['update_seconds'];development_seconds=saved['development_seconds']
        max_online_mass_correction=saved['max_online_mass_correction']
        sum_online_mass_correction=saved['sum_online_mass_correction']
    else:
        run.mkdir(parents=True,exist_ok=False)
        write_json(run/'manifest.json',{**current,'created_utc':datetime.now(timezone.utc).isoformat(),
                   'method':'MMW-style histogram Model B adaptation','final_data_read':False})
        write_json(run/'config.json',cfg)
    with (run/'invocations.jsonl').open('a') as f:
        f.write(json.dumps(dict(resume=a.resume,start_step=start,utc=datetime.now(timezone.utc).isoformat(),**environment))+'\n')
    torch.cuda.synchronize();started=time.perf_counter()
    def checkpoint(step):
        torch.cuda.synchronize()
        atomic_save(dict(step=step,network=net.state_dict(),optimizer=optimizer.state_dict(),
            streams={k:g.get_state() for k,g in streams.items()},torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),python_rng=random.getstate(),numpy_rng=np.random.get_state(),
            online_mu=online_mu,online_z=online_z,history=history,devhistory=devhistory,
            effective_seconds=elapsed+time.perf_counter()-started,simulation_dates=simulation_dates,
            update_seconds=update_seconds,development_seconds=development_seconds,
            max_online_mass_correction=max_online_mass_correction,
            sum_online_mass_correction=sum_online_mass_correction),run/'checkpoint.pt')
    try:
        for step in range(start,cfg['steps']+1):
            update_started=time.perf_counter()
            if cfg['sampling']=='simulation': mu,z=online_mu,online_z
            elif cfg['sampling']=='functions':
                ix=torch.randint(len(bank_mu),(cfg['batch_size'],),generator=streams['batch'],device=device)
                mu,z=bank_mu[ix],bank_z[ix]
            else: raise ValueError('unknown sampler')
            optimizer.zero_grad(set_to_none=True)
            loss,diag=econ.objective(net,mu,z,streams['objective'],cfg['estimator'],cfg['mass_weight'],
                                     cfg['high_asset_multiplier'],cfg['high_asset_cutoff'])
            if not bool(torch.isfinite(loss)): raise FloatingPointError('nonfinite MMW loss')
            loss.backward()
            if any(p.grad is not None and not bool(torch.isfinite(p.grad).all()) for p in net.parameters()):
                raise FloatingPointError('nonfinite MMW gradient')
            torch.nn.utils.clip_grad_norm_(net.parameters(),cfg['gradient_clip']);optimizer.step()
            if cfg['sampling']=='simulation':
                for _ in range(cfg['simulation_steps_per_update']):
                    online_mu,online_z=econ.advance(net,online_mu,online_z,streams['simulation']);simulation_dates+=1
                    correction=econ.last_online_mass_correction
                    max_online_mass_correction=max(max_online_mass_correction,correction)
                    sum_online_mass_correction+=correction
                mass_error=float((online_mu.sum((1,2))-1).abs().max())
                capital=(online_mu*grid).sum((1,2));labor=online_mu[:,1].sum(1)*econ.hours
                if mass_error>1e-4 or bool((capital<.2).any()) or bool((labor<.05).any()):
                    raise FloatingPointError(f'online state invalid: mass_error={mass_error:.9g}, '
                        f'min_capital={float(capital.min()):.9g}, min_labor={float(labor.min()):.9g}; failure retained')
            row=dict(step=step,**{k:float(v.detach()) for k,v in diag.items()})
            # Scalar diagnostics synchronize CUDA work above. Exclude evaluation
            # and checkpoint I/O from this separately reported compute budget.
            update_seconds+=time.perf_counter()-update_started
            row['simulation_dates_per_chain']=simulation_dates
            row['update_and_simulation_seconds']=update_seconds
            row['max_online_mass_correction']=max_online_mass_correction
            row['sum_online_mass_correction']=sum_online_mass_correction
            if step==1 or step%cfg['print_every']==0 or step==cfg['steps']:
                torch.cuda.synchronize();row['effective_seconds']=elapsed+time.perf_counter()-started
                row['peak_allocated_bytes']=torch.cuda.max_memory_allocated(device)
                print(f"mmw step {step}/{cfg['steps']}: loss={row['loss']:.6g} FB={row['fb']:.6g} EulerEstimate={row['euler_estimate']:.6g} K={row['K']:.6g} seconds={row['effective_seconds']:.2f}",flush=True)
            history.append(row)
            if step%cfg['development_every']==0 or step==cfg['steps']:
                development_started=time.perf_counter()
                values=[metrics(econ,net,dev_mu[i:i+1],dev_z[i:i+1]) for i in range(len(dev_mu))]
                devrow=dict(step=step,states=len(values),KKT=float(np.mean([v['KKT'] for v in values])),
                            P99=float(np.mean([v['P99'] for v in values])),final_data_read=False,
                            update_and_simulation_seconds=update_seconds)
                development_seconds+=time.perf_counter()-development_started
                devhistory.append(devrow);write_json(run/'development_curve.json',devhistory)
                print('DEVELOPMENT',json.dumps(devrow),flush=True)
            if step%cfg['checkpoint_every']==0 or step==cfg['steps']: checkpoint(step)
        torch.cuda.synchronize();seconds=elapsed+time.perf_counter()-started
        atomic_save(net.state_dict(),run/'policy.pt')
        fields=list(dict.fromkeys(k for row in history for k in row))
        with (run/'history.csv').open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(history)
        write_json(run/'report.json',dict(status='training_complete',steps=cfg['steps'],seed=seed,
            parameters=sum(p.numel() for p in net.parameters()),effective_seconds=seconds,
            update_and_simulation_seconds=update_seconds,periodic_development_seconds=development_seconds,
            timing_scope='updates, online simulation, periodic development, checkpoint I/O; full development evaluation excluded',
            simulation_dates_per_chain=simulation_dates,peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
            peak_reserved_bytes=torch.cuda.max_memory_reserved(device),policy_sha256=sha(run/'policy.pt'),**environment,
            max_online_mass_correction=max_online_mass_correction,
            sum_online_mass_correction=sum_online_mass_correction,
            estimator=cfg['estimator'],sampling=cfg['sampling'],final_data_read=False,
            interpretation='independent common KKT and own dynamics required; negative double-draw loss is possible'))
    except BaseException:
        failure=dict(step=locals().get('step',start),traceback=traceback.format_exc(),
                     last_checkpoint_retained=True,
                     max_online_mass_correction=max_online_mass_correction,
                     sum_online_mass_correction=sum_online_mass_correction,
                     online_state_may_precede_failed_advance=True)
        try:
            failure.update(online_mass_error=float((online_mu.sum((1,2))-1).abs().max()),
                           online_min_capital=float((online_mu*grid).sum((1,2)).min()),
                           online_min_labor=float((online_mu[:,1].sum(1)*econ.hours).min()))
        except BaseException as diagnostic_error:
            failure['diagnostic_error']=repr(diagnostic_error)
        write_json(run/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),failure)
        raise


if __name__=='__main__': main()
