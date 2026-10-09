"""Freeze nine main runs, three fine-grid runs, and all final-test rules."""
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from networks import PolicyNetwork
from jedc_distribution_v1 import VERSION,sha,write_json


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',required=True);p.add_argument('--audit',required=True)
    p.add_argument('--fine-dataset',required=True);p.add_argument('--output',required=True);args=p.parse_args()
    data=ROOT/args.dataset;fine=ROOT/args.fine_dataset;out=ROOT/args.output
    split=json.loads((data/'split_manifest.json').read_text(encoding='utf-8'))
    audit=json.loads((ROOT/args.audit).read_text(encoding='utf-8'))
    if split['version']!='mu_functions_v2' or split['status']!='complete':raise ValueError('complete v2 bank required')
    if split['counts']!=dict(train=21600,dev=5400,test=3600,shape=240):raise ValueError('formal counts')
    if [sum(g['split']==s for g in split['groups']) for s in ['train','dev','test']]!=[24,6,12]:raise ValueError('group split')
    if [sum(g['split']==s for g in split['paths']) for s in ['train','dev','test']]!=[24,6,12]:raise ValueError('shock split')
    for f,h in split['sha256'].items():
        if sha(data/f)!=h:raise ValueError('bank changed '+f)
    for g in split['groups']+split['paths']:
        if sha(data/g['file'])!=g['sha256']:raise ValueError('group/path changed')
    if audit['status']!='data_integrity_complete_no_policy_scores' or audit['transport_sha256']!=sha(ROOT/'src/jedc_distribution_v1.py'):raise ValueError('audit mismatch')
    for s in ['train','dev']:
        if audit[s+'_sha256']!=sha(data/(s+'.npz')):raise ValueError('audit data mismatch')
    if audit['fine_manifest_sha256']!=sha(fine/'manifest.json'):raise ValueError('fine bank audit mismatch')
    fm=json.loads((fine/'manifest.json').read_text())
    for f,h in fm['sha256'].items():
        if sha(fine/f)!=h:raise ValueError('fine bank changed')
    # Metadata-only final checks are allowed before scores; no policy loaded.
    parent_sets=[]
    for filename in ['train.npz','dev.npz','sealed/test.npz']:
        with np.load(data/filename) as d:parent_sets.append(set(d['parent_state_id']))
    if any(parent_sets[i]&parent_sets[j] for i in range(3) for j in range(i)):raise ValueError('parent leakage')
    out.mkdir(parents=True,exist_ok=False);configs=[];capacities=[]
    for phase,bank,seeds in [('main',data,[9711,9712,9713]),('fine',fine,[9711])]:
        with np.load(bank/'train.npz') as d:grid=torch.from_numpy(d['grid'].copy())
        n=len(grid);target=sum(p.numel() for p in PolicyNetwork('deeponet',n,500.,64,32,macro_features='capital',grid=grid).parameters())
        widths={mode:min(range(16,2049),key=lambda h:abs(h*h+((2*n+8) if mode=='capital' else 8)*h+1-target)) for mode in ['capital','capital_only']}
        for label,kind,mode,h in [('fullmu_deeponet','deeponet','capital',64),('fullmu_mlp','mlp','capital',widths['capital']),('onlyK_mlp','mlp','capital_only',widths['capital_only'])]:
            params=sum(p.numel() for p in PolicyNetwork(kind,n,500.,h,32,macro_features=mode,grid=grid).parameters())
            capacities.append(dict(phase=phase,label=label,parameters=params,width=h,basis=32,grid_points=n,
                relative_capacity_difference=(params-target)/target,macro_inputs=(2*n if mode=='capital' else 0)+3,micro_inputs=2))
            for seed in seeds:
                config=dict(status='frozen_mu_functions_v2_equation_only',calibration_file='data/jedc2010_calibration.json',
                    train_data_file=str((bank/'train.npz').relative_to(ROOT)).replace('\\','/'),
                    valid_data_file=str((bank/'dev.npz').relative_to(ROOT)).replace('\\','/'),
                    seed=seed,asset_points=n,asset_max=500.,hidden=64,mlp_hidden=h,basis=32,
                    policy_mode='consumption_softplus',macro_features=mode,transport=VERSION,
                    steps=100000,batch_size=8,learning_rate=2e-5,checkpoint_every=5000,
                    mass_weight=.5,high_asset_cutoff=20.,high_asset_multiplier=4.,device='auto',
                    n_train=21600,n_valid=5400,experiment_phase=phase)
                file=out/('jedc_mu_functions_v2_%s_%s_s%d.json'%(phase,label,seed));write_json(file,config)
                configs.append(dict(phase=phase,label=label,kind=kind,seed=seed,config=str(file.relative_to(ROOT)).replace('\\','/'),sha256=sha(file),parameters=params))
    protocol=dict(version='mu_functions_v2',frozen_before_final_scores=True,
        primary='mass-weighted KKT abs(gap) at saving>1e-7; max(-gap,0) at lower bound; raw common test functions only',
        secondary=['weighted median/P90/P99 and populated maximum','employment x zero/(0,5]/(5,20]/>20 groups',
            'positive consumption, lower/upper saving constraints','mass/employment conservation',
            'midpoint vs affine-interval capital push','tax/UI and resource budget checks labelled algebraic sanity checks'],
        shape='120 frozen pairs preserving K, employment margins and each atom; report both KKT and policy response, no truth labels',
        dynamic=dict(paths=12,periods=4000,initial='public official initial, bad z',phases=[[0,999],[1000,3999]],
            KKT_dates='every date',outputs=['K','Y','wage','interest','moments2..4','q10/q50/q90/q99','atoms','tail',
            'mass/employment','KKT/P99','upper saturation','capital reprojection error','budget sanity']),
        robustness=dict(checkpoint_grids=['native','fine','upper1000'],fine_retrain_seeds=[9711],
            adapter='rebin evaluation mu to frozen input grid; hold network scaling and policy upper bound500 for checkpoint-only quadrature checks; positive mass above500 is an explicit adapter failure'),
        confidence=dict(unit_common='function_generation_group',unit_dynamic='shock_path',seed_axis='paired training seeds',
            draws=2000,seed=202609299799,three_seed_limitations=True),
        stress='five original shape families separately; never pooled with primary',
        failure_policy='retain failed states/paths, counts and coverage; incomplete primary evaluations do not silently become successful means',
        no_test_selection=True,no_reference_policy_required=True,no_equilibrium_or_welfare_claim=True)
    write_json(out/'evaluation_protocol.json',protocol);write_json(out/'capacity_table.json',capacities)
    files=['src/ks_model.py','src/networks.py','src/networks_mu_v2.py','src/jedc_distribution_v1.py','src/neural_state_source_v1.py',
        'src/train_jedc_mu_v1.py','src/mu_evaluation_v2.py','scripts/build_jedc_mu_v1.py',
        'scripts/build_mu_functions_v2.py','scripts/audit_mu_functions_v2.py','scripts/freeze_mu_functions_v2.py',
        'scripts/evaluate_mu_functions_v2.py','scripts/finalize_mu_functions_v2.py','scripts/summarize_mu_functions_v2.py',
        'scripts/plot_mu_functions_v2_data.py',
        'scripts/build_sameK_pressure_v1.py','server/run_jedc_mu_v1.py','server/prepare_mu_functions_v2.py',
        'config/mu_functions_v2_generation.json','data/jedc2010_calibration.json',
        'data/jedc2010/initial_distribution.npz','runs/jedc_mu_momentfit_20260929/report.json',args.audit]
    files += [str((bank/f).relative_to(ROOT)).replace('\\','/') for bank in [data,fine] for f in ['train.npz','dev.npz']]
    files += [str((data/f).relative_to(ROOT)).replace('\\','/') for f in ['state_source_manifest.json','split_manifest.json']]
    files += [str((data/g['file']).relative_to(ROOT)).replace('\\','/') for g in split['paths'] if g['split']!='test']
    write_json(out/'experiment_manifest.json',dict(version='mu_functions_v2',status='frozen_no_final_scores',
        data_directory=args.dataset,fine_directory=args.fine_dataset,configs=configs,capacities=capacities,
        evaluation_protocol_sha256=sha(out/'evaluation_protocol.json'),
        sha256={f:sha(ROOT/f) for f in files},sealed_sha256={k:v for k,v in split['sha256'].items() if k.startswith('sealed/')},
        sealed_paths=[g for g in split['paths'] if g['split']=='test'],
        young_required=False,policy_labels_used=False,test_scores_read=False,
        expected_device='prefer RTX 4080 SUPER; actual hardware and timing scope recorded'))
    print('Frozen 9 main + 3 fine-grid configurations',out,flush=True)


if __name__=='__main__':main()
