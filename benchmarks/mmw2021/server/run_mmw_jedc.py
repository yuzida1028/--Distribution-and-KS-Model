"""Cloud-only isolated freeze and serial three-seed queue; no final evaluation."""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from jedc_distribution_v1 import sha, write_json

FILES = ['src/mmw_jedc.py', 'src/train_mmw_jedc.py', 'src/ks_model.py',
         'src/jedc_distribution_v1.py', 'src/mu_evaluation_v2.py',
         'scripts/evaluate_mmw_jedc.py', 'server/run_mmw_jedc.py',
         'config/mmw_jedc_template.json', 'tests/test_mmw_jedc.py']


def source_binding(plan_path):
    plan_path = plan_path.resolve()
    base = plan_path.parents[2]
    plan = json.loads(plan_path.read_text(encoding='utf-8'))
    if plan.get('version') != 'mu_functions_v2':
        raise ValueError('must bind to the new frozen function-library experiment')
    for name in ['src/ks_model.py', 'src/jedc_distribution_v1.py', 'src/mu_evaluation_v2.py']:
        if sha(base/name) != plan['sha256'][name] or sha(ROOT/name) != sha(base/name):
            raise ValueError('shared economic/evaluation source differs: '+name)
    item = next(x for x in plan['configs'] if x['label']=='fullmu_deeponet' and x['phase']=='main' and x['seed']==9711)
    cfg_path = base/item['config']
    if sha(cfg_path) != item['sha256']:
        raise ValueError('reference config hash differs')
    cfg = json.loads(cfg_path.read_text(encoding='utf-8'))
    paths = dict(train=base/cfg['train_data_file'], dev=base/cfg['valid_data_file'],
                 calibration=base/cfg['calibration_file'], initial=base/'data/jedc2010/initial_distribution.npz',
                 split=base/plan['data_directory']/'split_manifest.json')
    for path in paths.values():
        key = path.relative_to(base).as_posix()
        if sha(path) != plan['sha256'][key]:
            raise ValueError('frozen reference input changed: '+key)
    return plan, cfg, item, paths


def freeze(plan_path, directory, template):
    import math
    import numpy as np
    import torch
    from mmw_jedc import MMWNetwork
    plan, reference, item, paths = source_binding(plan_path)
    cfg = json.loads(template.read_text(encoding='utf-8'))
    with np.load(paths['train']) as d:
        grid = torch.from_numpy(d['grid'].copy())
        edges = d['edges'].copy()
    if reference['asset_max'] != float(edges[-1]):
        raise ValueError('reference policy bound differs from transport grid')
    calibration = json.loads(paths['calibration'].read_text(encoding='utf-8'))
    alpha, beta, delta = [calibration[k] for k in ['alpha', 'beta', 'delta']]
    c_over_k = (1/beta-1+delta)/alpha-delta
    initial_share = c_over_k/(1+c_over_k)
    if not 0 < initial_share < 1:
        raise ValueError('steady consumption share invalid')
    share_bias = math.log(initial_share/(1-initial_share))
    target = item['parameters']
    input_dim = 2*len(grid)+5
    hidden = min(range(16, 2049), key=lambda h: abs(h*h+(input_dim+4)*h+2-target))
    net = MMWNetwork(grid, reference['asset_max'], hidden, share_bias)
    actual = sum(p.numel() for p in net.parameters())
    if (cfg['steps'], cfg['batch_size'], cfg['learning_rate']) != (reference['steps'], reference['batch_size'], reference['learning_rate']):
        raise ValueError('template budget differs from reference; define a new comparison version')
    directory.mkdir(parents=True, exist_ok=False)
    binding = dict(reference_plan=str(plan_path.resolve()), reference_plan_sha256=sha(plan_path),
                   source_reference_config=item, reference_config=reference,
                   input_paths={k:str(p.resolve()) for k,p in paths.items()},
                   input_sha256={k:sha(p) for k,p in paths.items()},
                   source_sha256={f:sha(ROOT/f) for f in FILES},
                   development_protocol=dict(split='dev', common='all raw + perturbations, grouped by generation group',
                       sameK=True, dynamics='six frozen dev paths, 4000 dates; first1000 separate', final_access=False),
                   hidden=hidden, parameters=actual, reference_parameters=target,
                   relative_parameter_difference=(actual-target)/target, asset_max=reference['asset_max'],
                   share_bias=share_bias, grid_points=len(grid), configs=[],
                   created_utc=datetime.now(timezone.utc).isoformat(), status='frozen_no_final_scores')
    for seed in cfg['seeds']:
        frozen = {**cfg, 'seed':seed, 'hidden':hidden, 'share_bias':share_bias,
                  'asset_max':reference['asset_max'], 'grid_points':len(grid)}
        file = directory/f'seed_{seed}.json'
        write_json(file, frozen)
        binding['configs'].append(dict(path=str(file.resolve()), sha256=sha(file), seed=seed))
    write_json(directory/'binding.json', binding)
    print(f'Frozen MMW seeds={cfg["seeds"]}, hidden={hidden}, params={actual}, reference={target}', flush=True)


def verify_binding(directory):
    b = json.loads((directory/'binding.json').read_text(encoding='utf-8'))
    if sha(b['reference_plan']) != b['reference_plan_sha256']:
        raise ValueError('reference plan changed')
    for name, digest in b['input_sha256'].items():
        if sha(b['input_paths'][name]) != digest:
            raise ValueError('reference input changed: '+name)
    ref=b['source_reference_config']
    base=Path(b['reference_plan']).parents[2]
    if sha(base/ref['config'])!=ref['sha256']:
        raise ValueError('reference DeepONet configuration changed')
    for name, digest in b['source_sha256'].items():
        if sha(ROOT/name) != digest:
            raise ValueError('MMW source changed: '+name)
    for entry in b['configs']:
        if sha(entry['path']) != entry['sha256']:
            raise ValueError('MMW config changed')
    return b


def idle_gpu():
    # Do not match ancestor bash -c text (the earlier main-nine guard failure).
    own = {os.getpid()}; parent = os.getppid()
    while parent and parent not in own:
        own.add(parent)
        try: parent = int((Path('/proc')/str(parent)/'stat').read_text().rsplit(')', 1)[1].split()[1])
        except (OSError, ValueError, IndexError): break
    snapshot = subprocess.check_output(['ps', '-eo', 'pid=,args='], text=True)
    for line in snapshot.splitlines():
        parts = line.strip().split(None, 1)
        if len(parts)<2 or int(parts[0]) in own: continue
        if any(x in parts[1] for x in ['train_jedc_mu_v1.py','run_jedc_mu_v1.py','train_mmw_jedc.py','run_mmw_jedc.py','evaluate_mu_functions_v2.py','evaluate_mmw_jedc.py']):
            raise RuntimeError('active experiment task; preserve it and run MMW after completion: '+line)
    pids = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'], text=True)
    if any(p.strip().isdigit() for p in pids.splitlines()):
        raise RuntimeError('GPU occupied; no concurrent timed benchmark')
    return snapshot


def reference_environment(binding):
    base=Path(binding['reference_plan']).parents[2]
    entry=binding['source_reference_config'];stem=Path(entry['config']).stem
    matches=[]
    for run in (base/'runs').glob('deeponet_'+stem+'_*'):
        if not all((run/f).is_file() for f in ['manifest.json','policy.pt','report.json']):continue
        manifest=json.loads((run/'manifest.json').read_text())
        if manifest.get('sha256',{}).get('config')==entry['sha256']:matches.append(manifest)
    if len(matches)!=1:raise RuntimeError('one completed matching new DeepONet seed9711 run required for hardware/software comparison')
    return matches[0]


def main():
    import fcntl
    p=argparse.ArgumentParser()
    p.add_argument('--reference-plan', required=True)
    p.add_argument('--tag', default='mmw971_sim_20260929')
    p.add_argument('--prepare-only', action='store_true')
    p.add_argument('--resume', action='store_true')
    p.add_argument('--expected-gpu', default='NVIDIA GeForce RTX 2080 Ti')
    a=p.parse_args()
    if platform.system()!='Linux': raise RuntimeError('scientific work is cloud-only')
    if not a.tag.replace('_','').replace('-','').isalnum(): raise ValueError('invalid tag')
    os.chdir(ROOT)
    directory=ROOT/'config'/a.tag
    if directory.exists():
        if not a.resume: raise FileExistsError('existing freeze preserved; use --resume')
        b=verify_binding(directory)
        if Path(b['reference_plan'])!=Path(a.reference_plan).resolve(): raise ValueError('different reference plan')
    else:
        freeze(Path(a.reference_plan), directory, ROOT/'config/mmw_jedc_template.json')
        b=verify_binding(directory)
    if a.prepare_only: return
    with open('/tmp/empractical_mmw_gpu_queue.lock','a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX|fcntl.LOCK_NB)
        processes=idle_gpu()
        import torch
        if not torch.cuda.is_available(): raise RuntimeError('CUDA required')
        gpu=torch.cuda.get_device_name(0)
        if gpu!=a.expected_gpu: raise ValueError(f'GPU {gpu!r} differs from expected {a.expected_gpu!r}; set actual reference GPU explicitly')
        reference=reference_environment(b)
        for field,value in [('gpu',gpu),('python',platform.python_version()),('torch',str(torch.__version__)),('torch_cuda',torch.version.cuda)]:
            if reference.get(field)!=value:raise ValueError(f'reference environment differs for {field}; keep hardware/software fixed for this cost comparison')
        logs=ROOT/'logs';logs.mkdir(exist_ok=True)
        env=dict(gpu=gpu, python=platform.python_version(), torch=torch.__version__,cuda=torch.version.cuda,
                 process_snapshot=processes, time=datetime.now(timezone.utc).isoformat())
        with (logs/(a.tag+'_invocations.jsonl')).open('a') as f: f.write(json.dumps(env)+'\n')
        for entry in b['configs']:
            run=ROOT/'runs'/a.tag/f'seed_{entry["seed"]}'
            cmd=[sys.executable,'-u','src/train_mmw_jedc.py','--binding',str(directory/'binding.json'),
                 '--config',entry['path'],'--run-dir',str(run)]
            if run.exists():
                if (run/'policy.pt').exists() and (run/'report.json').exists():
                    report=json.loads((run/'report.json').read_text())
                    manifest=json.loads((run/'manifest.json').read_text())
                    if (report.get('status')!='training_complete' or report.get('steps')!=100000
                            or report.get('policy_sha256')!=sha(run/'policy.pt')
                            or manifest.get('binding_sha256')!=sha(directory/'binding.json')
                            or manifest.get('config_sha256')!=entry['sha256']):
                        raise ValueError('incomplete or altered existing MMW run')
                    print('Completed training retained', run, flush=True)
                    cmd=None
                elif a.resume: cmd.append('--resume')
                else: raise FileExistsError(run)
            if cmd:
                print('EXEC', cmd, 'training not verified until step log', flush=True)
                with (logs/f'{a.tag}_s{entry["seed"]}.log').open('a') as log:
                    subprocess.run(cmd, stdout=log, stderr=subprocess.STDOUT, check=True)
            out=run/'development_static'
            if not out.exists():
                subprocess.run([sys.executable,'-u','scripts/evaluate_mmw_jedc.py','--binding',str(directory/'binding.json'),
                                '--run-dir',str(run),'--static-only'], check=True)
            elif not (out/'report.json').exists():
                raise RuntimeError('partial development evaluation preserved; select a new --output for reevaluation')


if __name__=='__main__': main()
