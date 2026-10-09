"""Freeze independent MMV benchmark against an existing v2 source manifest."""
import json
from pathlib import Path
from jedc_distribution_v1 import sha,write_json
ROOT=Path(__file__).resolve().parents[1]
SOURCES=['src/mmv2010_jedc.py','src/mmv2010_protocol.py','src/mmv2010_bridge.py',
         'src/ks_model.py','src/jedc_distribution_v1.py','src/mu_evaluation_v2.py',
         'src/networks.py','src/networks_mu_v2.py','scripts/solve_mmv2010.py',
         'scripts/evaluate_mmv2010.py','scripts/summarize_mmv2010.py','server/run_mmv2010.py','config/mmv2010_template.json',
         'tests/test_mmv2010.py','docs/mmv2010_author_source/provenance.json',
         'docs/mmv2010_author_source/MMV_JEDC_34_2010_original.zip',
         'docs/mmv2010_author_source/LICENSE_AGREEMENT.txt','docs/MMV2010_JEDC_README.md',
         'server/start_mmv2010.sh']


def freeze(reference_path,out):
    reference_path=Path(reference_path).resolve();base=reference_path.parents[2]
    plan=json.loads(reference_path.read_text(encoding='utf-8'))
    if plan.get('version')!='mu_functions_v2':raise ValueError('new v2 reference required')
    for name in ['src/ks_model.py','src/jedc_distribution_v1.py','src/mu_evaluation_v2.py','src/networks.py','src/networks_mu_v2.py']:
        if sha(base/name)!=plan['sha256'][name] or sha(ROOT/name)!=sha(base/name):raise ValueError('shared source changed '+name)
    entries=[r for r in plan['configs'] if r['label']=='fullmu_deeponet' and r['phase']=='main']
    for entry in entries:
        if sha(base/entry['config'])!=entry['sha256']:raise ValueError('reference config changed')
    cfg=json.loads((base/entries[0]['config']).read_text())
    data=base/plan['data_directory']
    inputs=dict(train=base/cfg['train_data_file'],dev=base/cfg['valid_data_file'],
                split=data/'split_manifest.json',calibration=base/cfg['calibration_file'],
                initial=base/'data/jedc2010/initial_distribution.npz')
    for file in inputs.values():
        if sha(file)!=plan['sha256'][file.relative_to(base).as_posix()]:raise ValueError('reference data changed')
    if sorted(x['seed'] for x in entries)!=[9711,9712,9713]:raise ValueError('three new DeepONet main seeds required')
    split=json.loads(inputs['split'].read_text())
    dev_paths=[dict(path=str((data/x['file']).resolve()),sha256=x['sha256']) for x in split['paths'] if x['split']=='dev']
    if len(dev_paths)!=6:raise ValueError('six development paths required')
    for x in dev_paths:
        if sha(x['path'])!=x['sha256']:raise ValueError('dev shock changed')
    provenance=json.loads((ROOT/'docs/mmv2010_author_source/provenance.json').read_text())
    if sha(ROOT/'docs/mmv2010_author_source/MMV_JEDC_34_2010_original.zip')!=provenance['archive_sha256']:raise ValueError('author archive changed')
    template=json.loads((ROOT/'config/mmv2010_template.json').read_text())
    import numpy as np
    with np.load(inputs['train']) as d:edges=d['edges'].copy();grid=d['grid'].copy()
    if edges[-1]!=500 or cfg['asset_max']!=500:raise ValueError('matched profile requires original500 policy domain')
    out.mkdir(parents=True,exist_ok=False);configs=[]
    for profile,parameters in template['profiles'].items():
        seeds=template['seeds'] if profile=='matched' else [9711]
        for seed in seeds:
            config={k:v for k,v in template.items() if k not in ['seeds','profiles']}
            config.update(parameters,profile=profile,seed=seed)
            file=out/f'{profile}_s{seed}.json';write_json(file,config)
            configs.append(dict(profile=profile,seed=seed,path=str(file.resolve()),sha256=sha(file)))
    protocol=dict(primary='joint-mass-weighted current-state KKT under each policy actual endogenous next mu; not perceived ALM residual',
        common='same existing dev5400 functions, raw group mean1800 + separate local/sameK; no model-score selection',
        dynamic='same six dev4000 shock paths, official initial; each policy evolves its own mu; transition1000/post3000',
        mmv_secondary='one-step ALM errors, perceived-ALM KKT; not comparable to DeepONet transport identities',
        final_access=False,failure='retain errors and partial coverage; actual K out-of-domain fails; perceived forecast clipping counted; no averaging only successes',
        numerical='matched100 vs fine300 grids; author1000/domain30-50 as separate sensitivity, not main mean',
        costs='solve CPU float64 best-suited implementation, same cloud host; independent evaluator uses same GPU float32; no fictitious100k optimizer updates',
        superiority='not predetermined; onlyK wins/failures retained')
    write_json(out/'evaluation_protocol.json',protocol)
    binding=dict(version='mmv2010_jedc_v1',reference_plan=str(reference_path),reference_plan_sha256=sha(reference_path),
        development_paths=dev_paths, reference_base=str(base),reference_configs=entries,reference_config=cfg,
        input_paths={k:str(p.resolve()) for k,p in inputs.items()},input_sha256={k:sha(p) for k,p in inputs.items()},
        source_sha256={k:sha(ROOT/k) for k in SOURCES},configs=configs,
        protocol_sha256=sha(out/'evaluation_protocol.json'),final_data_read=False,
        grid_points=len(grid),status='frozen_no_final_scores')
    write_json(out/'binding.json',binding)
    print('MMV frozen: 3 matched seeds + fine9711 + author-domain9711; no solving started',flush=True)
    return binding


def verify(binding_path):
    binding_path=Path(binding_path);b=json.loads(binding_path.read_text())
    if b.get('version')!='mmv2010_jedc_v1':raise ValueError('binding version')
    if sha(b['reference_plan'])!=b['reference_plan_sha256']:raise ValueError('reference plan changed')
    for k,h in b['input_sha256'].items():
        if sha(b['input_paths'][k])!=h:raise ValueError('reference input changed '+k)
    for x in b['development_paths']:
        if sha(x['path'])!=x['sha256']:raise ValueError('dev shocks changed')
    for x in b['reference_configs']:
        if sha(Path(b['reference_base'])/x['config'])!=x['sha256']:raise ValueError('DeepONet configuration changed')
    for k,h in b['source_sha256'].items():
        if sha(ROOT/k)!=h:raise ValueError('benchmark code changed '+k)
    for item in b['configs']:
        if sha(item['path'])!=item['sha256']:raise ValueError('benchmark config changed')
    if sha(binding_path.parent/'evaluation_protocol.json')!=b['protocol_sha256']:raise ValueError('protocol changed')
    return b
