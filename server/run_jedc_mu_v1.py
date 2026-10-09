"""Nine NEW fixed-budget equation-only runs. No legacy seeds or Young gate."""
import argparse
import csv
import json
import os
import platform
import subprocess
import sys
import traceback
from datetime import datetime,timezone
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from jedc_distribution_v1 import sha,write_json

def now():return datetime.now(timezone.utc).isoformat()

def complete(run):
    required=['history.csv','policy.pt','checkpoint.pt','report.json','manifest.json']
    if not all((run/x).is_file() for x in required):return False
    with (run/'history.csv').open(newline='') as f:steps=[int(r['step']) for r in csv.DictReader(f)]
    return steps==list(range(1,100001))

def launcher_ancestors():
    pids = {os.getpid()}
    pid = os.getppid()
    while pid > 0 and pid not in pids:
        pids.add(pid)
        try:
            pid = int((Path("/proc") / str(pid) / "stat").read_text().rsplit(")", 1)[1].split()[1])
        except (OSError, ValueError, IndexError):
            break
    return pids


def main():
    import fcntl
    p=argparse.ArgumentParser();p.add_argument('--plan',required=True);p.add_argument('--tag',required=True)
    p.add_argument('--resume-queue',action='store_true');p.add_argument('--allow-other-gpu',action='store_true')
    p.add_argument('--dev-audit',action='store_true',help='Post-training development metrics only; never final test')
    p.add_argument('--phase',choices=['main','fine','all'],default='all')
    args=p.parse_args();os.chdir(ROOT)
    if not args.tag.replace('_','').replace('-','').isalnum():raise ValueError('invalid tag')
    planpath=ROOT/args.plan;plan=json.loads(planpath.read_text(encoding='utf-8'))
    v2=plan.get('version')=='mu_functions_v2'
    if len(plan['configs'])!=(12 if v2 else 9) or {r['seed'] for r in plan['configs']}!={9711,9712,9713}:raise ValueError('unexpected new-run queue')
    items=[r for r in plan['configs'] if args.phase=='all' or r.get('phase','main')==args.phase]
    for path,digest in plan['sha256'].items():
        if sha(ROOT/path)!=digest:raise ValueError('frozen source/data changed: '+path)
    for item in plan['configs']:
        if sha(ROOT/item['config'])!=item['sha256']:raise ValueError('config changed')
    if not torch.cuda.is_available():raise RuntimeError('CUDA unavailable')
    gpu=torch.cuda.get_device_properties(0)
    if '4080 SUPER' not in gpu.name and not args.allow_other_gpu:raise RuntimeError('GPU '+gpu.name+' differs; use --allow-other-gpu to record a separate hardware scope')
    processes=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
    for line in processes.splitlines():
        tokens=line.strip().split(None,1)
        if len(tokens)>1 and int(tokens[0]) not in launcher_ancestors() and any(s in tokens[1] for s in ['src/train.py','src/train_jedc_mu_v1.py','resume_train_rng_cpu','run_onlyk_ablation','run_mlp_4080']):
            raise RuntimeError('active research task; refusing duplicate: '+line)
    gpu_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
    if any(p.strip().isdigit() and int(p.strip())!=os.getpid() for p in gpu_pids.splitlines()):raise RuntimeError('GPU has another active compute PID')
    batch=ROOT/'runs'/('jedc_mu_v1_queue_'+args.tag)
    if batch.exists() and not args.resume_queue:raise FileExistsError('queue exists; explicit --resume-queue required')
    batch.mkdir(parents=True,exist_ok=True)
    with (batch/'queue.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        provenance=batch/'provenance.json'
        if provenance.exists():
            old=json.loads(provenance.read_text());
            if old['plan_sha256']!=sha(planpath):raise ValueError('queue plan changed')
            if old['gpu']!=gpu.name or old['torch']!=torch.__version__ or old['torch_cuda']!=torch.version.cuda:
                raise ValueError('resume hardware/runtime changed; retain original queue and use a separately documented migration')
        else:
            if any((ROOT/'runs'/(item['kind']+'_'+Path(item['config']).stem+'_'+args.tag)).exists() for item in plan['configs']):raise RuntimeError('run destination already exists')
            write_json(provenance,dict(created_utc=now(),PID=os.getpid(),project_root=str(ROOT.resolve()),plan_sha256=sha(planpath),
                gpu=gpu.name,gpu_memory_bytes=gpu.total_memory,python=platform.python_version(),torch=torch.__version__,torch_cuda=torch.version.cuda,
                nvidia_smi=subprocess.check_output(['nvidia-smi'],text=True),process_snapshot=processes,
                policy_labels_used=False,young_required=False,final_scores_read=False))
        try:
            for item in items:
                name=item['kind']+'_'+Path(item['config']).stem+'_'+args.tag
                run=ROOT/'runs'/name
                if complete(run):print('verified complete, skipping',name,flush=True)
                else:
                    command=[sys.executable,'-u','src/train_jedc_mu_v1.py','--config',item['config'],'--model',item['kind'],'--run-id',args.tag]
                    if run.exists():
                        if not args.resume_queue or not (run/'checkpoint.pt').is_file():raise RuntimeError('incomplete directory without resumable checkpoint: '+str(run))
                        command.append('--resume')
                    write_json(batch/'status.json',dict(status='executing_training_wait_for_step_log',run=name,command=command,utc=now()))
                    with (batch/(name+'.log')).open('a',encoding='utf-8') as log:
                        log.write('\nINVOCATION '+now()+'\n');log.flush()
                        subprocess.run(command,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,check=True)
                    if not complete(run):raise RuntimeError('100k artifacts incomplete')
                devdir='evaluation_v2_dev_native' if v2 else 'dev_evaluation'
                if args.dev_audit and not (run/devdir).exists():
                    evaluator='scripts/evaluate_mu_functions_v2.py' if v2 else 'scripts/evaluate_jedc_mu_v1.py'
                    command=[sys.executable,'-u',evaluator,'--plan',args.plan,'--config',item['config'],'--run-dir',str(run.relative_to(ROOT))]
                    if v2:command.append('--static-only')
                    subprocess.run(command,cwd=ROOT,check=True)
            write_json(batch/'status.json',dict(status='selected_runs_complete_no_final_evaluation',phase=args.phase,runs=len(items),utc=now()))
        except Exception:
            write_json(batch/'failure.json',dict(utc=now(),traceback=traceback.format_exc(),status='failure_preserved'))
            raise

if __name__=='__main__':main()
