"""Cloud CPU MMV Algorithm1 solve with outer-checkpoint resume and failures."""
import argparse
import json
import os
import platform
import sys
import time
import traceback
import resource
from datetime import datetime,timezone
from pathlib import Path
import numpy as np
import scipy
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mmv2010_jedc import solve_individual,generate_shocks,sample_official_panel,simulate_panel,fit_alm,PolicyTable
from mmv2010_protocol import verify
from jedc_distribution_v1 import sha,write_json


def initial_joint(path,edges):
    from jedc_distribution_v1 import rebin,midpoint_grid
    with np.load(path) as d:
        mass=d['mu'].astype(float);original=d['grid'].copy()
    source=np.arange(1001)*.1
    if not np.allclose(original,midpoint_grid(source)):raise ValueError('official representation')
    mass*= (np.array([.1,.9])/mass.sum(1))[:,None]
    return rebin(mass,source,edges)


def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--config',required=True)
    p.add_argument('--run-dir',required=True);p.add_argument('--resume',action='store_true');a=p.parse_args()
    if platform.system()!='Linux':raise RuntimeError('cloud scientific work only')
    setup_started=time.perf_counter()
    b=verify(a.binding);cfg_path=Path(a.config).resolve();cfg=json.loads(cfg_path.read_text())
    if not any(x['path']==str(cfg_path) and x['sha256']==sha(cfg_path) for x in b['configs']):raise ValueError('unregistered config')
    cal=json.loads(Path(b['input_paths']['calibration']).read_text());run=Path(a.run_dir)
    B=np.array([[0.,1.],[0.,1.]]);table=None;history=[];start=1;previous=0.;phases=dict(individual_seconds=0.,panel_seconds=0.,regression_seconds=0.)
    env=dict(python=platform.python_version(),numpy=np.__version__,scipy=scipy.__version__,cpu=platform.processor(),
             logical_cpus=os.cpu_count(),solver_device='CPU',solver_dtype='float64')
    manifest=dict(binding_sha256=sha(a.binding),config_sha256=sha(cfg_path),environment=env)
    z,employment=generate_shocks(cal,cfg['seed'],cfg['simulation_periods'],cfg['panel_agents'])
    with np.load(b['input_paths']['train']) as d:edges=d['edges'].copy()
    official=initial_joint(b['input_paths']['initial'],edges)
    initial=sample_official_panel(official,edges,employment[0],cfg['seed'])
    if a.resume:
        old=json.loads((run/'manifest.json').read_text())
        if any(old[k]!=manifest[k] for k in manifest):raise ValueError('resume source/runtime changed')
        saved=np.load(run/'outer_checkpoint.npz')
        B=saved['coefficients'].copy();table=saved['table'].copy();initial=saved['initial_panel'].copy();start=int(saved['outer'])+1
        metadata=json.loads(str(saved['metadata_json']));history=metadata['history'];previous=metadata['seconds'];phases=metadata['phase_times']
        if not np.array_equal(z,saved['z']) or not np.array_equal(employment,saved['employment']):raise ValueError('random stream changed')
    else:
        run.mkdir(parents=True,exist_ok=False);write_json(run/'manifest.json',manifest);write_json(run/'config.json',cfg)
        np.savez_compressed(run/'fixed_training_shocks.npz',z=z,employment=employment)
    setup_seconds=time.perf_counter()-setup_started
    started=time.perf_counter();converged=False;last_fit=None
    try:
        if history and history[-1]['raw_coefficient_l2']<=cfg['alm_raw_l2_tolerance']:converged=True
        for outer in range(start,cfg['outer_max_iterations']+1):
            if converged:break
            tick=time.perf_counter()
            policy,individual=solve_individual(cal,cfg,B,table,
                progress=lambda it,err:print(f'MMV outer {outer} individual {it}: max_fixed_point_error={err:.9g}',flush=True))
            phases['individual_seconds']+=time.perf_counter()-tick;table=policy.table
            tick=time.perf_counter();K,terminal,panel=simulate_panel(policy,initial,z,employment);phases['panel_seconds']+=time.perf_counter()-tick
            tick=time.perf_counter();estimated,stats=fit_alm(K,z,cfg['regression_burn']);phases['regression_seconds']+=time.perf_counter()-tick
            difference=float(np.linalg.norm(B-estimated))
            if difference>cfg['terminal_initial_update_until_raw_l2']:initial=terminal.copy()
            B=cfg['alm_damping']*estimated+(1-cfg['alm_damping'])*B
            row=dict(outer=outer,raw_coefficient_l2=difference,coefficients=B.tolist(),regression=stats,
                individual=individual,panel=panel,K_min=float(K.min()),K_max=float(K.max()),
                K_initial=float(K[0]),panel_initial_unemployment=float(np.mean(employment[0]==0)),
                K_mean=float(K[cfg['regression_burn']:].mean()),seconds=previous+time.perf_counter()-started)
            history.append(row);write_json(run/'outer_history.json',history)
            metadata=dict(history=history,seconds=previous+time.perf_counter()-started,phase_times=phases)
            temporary=run/'outer_checkpoint.tmp.npz'
            np.savez_compressed(temporary,outer=outer,coefficients=B,table=table,initial_panel=initial,z=z,employment=employment,metadata_json=json.dumps(metadata))
            os.replace(temporary,run/'outer_checkpoint.npz')
            print('MMV OUTER',json.dumps({k:row[k] for k in ['outer','raw_coefficient_l2','coefficients','K_mean','seconds']}),flush=True)
            if difference<=cfg['alm_raw_l2_tolerance']:converged=True;break
        if not converged:raise RuntimeError('ALM iteration budget exhausted; not a converged solution')
        # Unlike the original MAIN save, align final policy with updated B.
        tick=time.perf_counter();policy,final_individual=solve_individual(cal,cfg,B,table)
        phases['individual_seconds']+=time.perf_counter()-tick
        policy.save(run/'policy_table.npz')
        write_json(run/'report.json',dict(status='converged',outer_iterations=len(history),coefficients=B.tolist(),
            solve_seconds_accumulated=previous+time.perf_counter()-started+sum(json.loads(f.read_text()).get('uncheckpointed_seconds',0.) for f in run.glob('failure_*.json')),
            successful_checkpoint_lineage_seconds=previous+time.perf_counter()-started, setup_seconds_this_invocation=setup_seconds,
            interruptions=[json.loads(f.read_text()) for f in run.glob('failure_*.json')],
            peak_process_RSS_MiB=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,phase_times=phases,final_individual=final_individual,
            panel_shocks_sha256=sha(run/'fixed_training_shocks.npz'),policy_sha256=sha(run/'policy_table.npz'),
            binding_sha256=sha(a.binding),profile=cfg['profile'],seed=cfg['seed'],environment=env,
            interpretation='MMV2010 Algorithm1 Python/SciPy matched-policy-domain adaptation, independent KKT still required',
            final_data_read=False))
    except BaseException:
        write_json(run/('failure_'+datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S%f')+'.json'),
            dict(outer=locals().get('outer',start),traceback=traceback.format_exc(),seconds=previous+time.perf_counter()-started,
                 uncheckpointed_seconds=max(0.,previous+time.perf_counter()-started-(history[-1]['seconds'] if history else 0.)),
                 coefficients=B.tolist(),checkpoint_preserved=True,final_data_read=False))
        raise


if __name__=='__main__':main()
