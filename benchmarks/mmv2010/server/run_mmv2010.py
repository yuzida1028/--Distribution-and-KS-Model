"""Isolated cloud MMV queue; preserves existing DeepONet and sealed tests."""
import argparse,json,os,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from mmv2010_protocol import freeze,verify
from jedc_distribution_v1 import sha,write_json

def idle():
    ancestors={os.getpid()};pid=os.getppid()
    while pid>1:
        ancestors.add(pid)
        try:pid=int(Path(f'/proc/{pid}/stat').read_text().rsplit(')',1)[1].split()[1])
        except (OSError,ValueError):break
    patterns=['train_jedc','run_jedc','train_mmw','run_mmw','solve_mmv2010','evaluate_mu_functions','evaluate_mmv2010']
    for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()[1:]:
        number,command=line.strip().split(None,1)
        if int(number) not in ancestors and any(x in command for x in patterns):
            raise RuntimeError('active experiment; wait, do not kill: '+line)
    gpu=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory','--format=csv,noheader'],text=True)
    if any(line.split(',',1)[0].strip().isdigit() for line in gpu.splitlines()):
        raise RuntimeError('GPU has active compute process; preserve comparable host conditions: '+gpu)
    return subprocess.check_output(['nvidia-smi','--query-gpu=name,memory.total,driver_version','--format=csv'],text=True)

def reference_environment(binding):
    base=Path(binding['reference_base'])
    entry=next(x for x in binding['reference_configs'] if x['seed']==9711)
    stem=Path(entry['config']).stem
    matches=[]
    for run in (base/'runs').glob('deeponet_'+stem+'_*'):
        if not all((run/name).is_file() for name in ['manifest.json','policy.pt','report.json']):continue
        manifest=json.loads((run/'manifest.json').read_text())
        if manifest.get('sha256',{}).get('config')==entry['sha256']:
            matches.append(manifest)
    if len(matches)!=1:raise RuntimeError('expected one completed new DeepONet 9711 reference run')
    return matches[0]

def main():
    p=argparse.ArgumentParser();p.add_argument('--reference-plan',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--profile',choices=['matched','fine','author_domain'],default='matched')
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume',action='store_true');a=p.parse_args()
    if platform.system()!='Linux':raise RuntimeError('run scientific work on cloud only')
    if not a.tag or any(c not in 'abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-' for c in a.tag):raise ValueError('unsafe tag')
    os.chdir(ROOT);folder=ROOT/'config'/('mmv2010_'+a.tag);binding=folder/'binding.json'
    if binding.exists():
        if not a.resume:raise ValueError('tag exists; use --resume, never overwrite')
        b=verify(binding)
        if Path(b['reference_plan']).resolve()!=Path(a.reference_plan).resolve():raise ValueError('different reference')
    else:b=freeze(Path(a.reference_plan),folder)
    if a.prepare_only:return
    gpu=idle();reference=reference_environment(b)
    actual_name=gpu.splitlines()[1].split(',')[0].strip()
    if actual_name!=reference['gpu'] or platform.python_version()!=reference['python']:
        raise RuntimeError('MMV host differs from new DeepONet reference; cost scope must be separate')
    queue=ROOT/'runs'/('mmv2010_queue_'+a.tag);queue.mkdir(exist_ok=True,parents=True)
    write_json(queue/f'environment_{a.profile}.json',dict(gpu=gpu,platform=platform.platform(),
        cpu=Path('/proc/cpuinfo').read_text(),solver='CPU float64; GPU used only in independent evaluation',
        reference_deeponet_environment=reference,thread_limit=1,reference_plan=b['reference_plan'],final_data_read=False))
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1')
    for entry in b['configs']:
        if entry['profile']!=a.profile:continue
        name=f"{entry['profile']}_s{entry['seed']}";run=queue/name;report=run/'report.json'
        if report.exists():
            r=json.loads(report.read_text());m=json.loads((run/'manifest.json').read_text())
            if r['status']!='converged' or sha(run/'policy_table.npz')!=r['policy_sha256'] or m['binding_sha256']!=sha(binding) or m['config_sha256']!=entry['sha256']:raise ValueError('invalid completed run')
        else:
            cmd=[sys.executable,'-u','scripts/solve_mmv2010.py','--binding',str(binding),'--config',entry['path'],'--run-dir',str(run)]
            if run.exists():
                if not a.resume:raise ValueError('partial run; explicit --resume required')
                cmd.append('--resume')
            print('MMV dispatched; wait for actual individual/OUTER log:',name,flush=True)
            with (queue/(name+'.log')).open('a') as f:subprocess.run(cmd,check=True,env=env,stdout=f,stderr=subprocess.STDOUT)
        if not (run/'development_static'/'report.json').exists():
            # Failed evaluation directories are retained; a resume gets a fresh numbered attempt.
            output=run/'development_static';index=1
            while output.exists():output=run/f'development_static_attempt{index}';index+=1
            cmd=[sys.executable,'-u','scripts/evaluate_mmv2010.py','--binding',str(binding),'--run-dir',str(run),'--static-only','--output',str(output)]
            with (queue/(name+'_dev.log')).open('a') as f:subprocess.run(cmd,check=True,env=env,stdout=f,stderr=subprocess.STDOUT)
    print('MMV requested profile finished; final test remains sealed',flush=True)
if __name__=='__main__':main()
