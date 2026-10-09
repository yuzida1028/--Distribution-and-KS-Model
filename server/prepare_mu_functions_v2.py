"""AutoDL preparation and launch. Never intended for a local Windows run."""
import argparse
import json
import os
import platform
import subprocess
import sys
from datetime import datetime,timezone
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from jedc_distribution_v1 import write_json,sha


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
    p=argparse.ArgumentParser();p.add_argument('--tag',required=True)
    p.add_argument('--prepare-only',action='store_true');p.add_argument('--resume-queue',action='store_true')
    args=p.parse_args();os.chdir(ROOT)
    if platform.system()!='Linux':
        raise RuntimeError('run all experiment computation in the user-selected Linux cloud GPU instance; never on local Windows')
    if not args.tag.replace('_','').replace('-','').isalnum():raise ValueError('invalid unique tag')
    workspace=ROOT/'runs'/('mu_functions_v2_prepare_'+args.tag)
    workspace.mkdir(parents=True,exist_ok=args.resume_queue)
    with (workspace/'prepare.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        processes=subprocess.check_output(['ps','-eo','pid=,args='],text=True)
        for line in processes.splitlines():
            pid,command=line.strip().split(None,1) if len(line.strip().split(None,1))==2 else ('0','')
            if int(pid) not in launcher_ancestors() and any(s in command for s in ['src/train.py','src/train_jedc_mu_v1.py','server/run_jedc_mu_v1.py','server/prepare_mu_functions_v2.py','run_onlyk_ablation','run_mlp_4080','build_mu_functions_v2.py','audit_mu_functions_v2.py']):
                raise RuntimeError('active experiment task: '+line)
        smi=subprocess.check_output(['nvidia-smi'],text=True)
        gpu_pids=subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],text=True)
        if any(p.strip().isdigit() for p in gpu_pids.splitlines()):raise RuntimeError('GPU already has active compute PID')
        data='data/mu_functions_v2_'+args.tag;fine=data+'_fine';audit='runs/mu_functions_v2_audit_'+args.tag
        frozen='config/mu_functions_v2_'+args.tag;plan=frozen+'/experiment_manifest.json'
        def call(script,*arguments):
            command=[sys.executable,'-u',script,*arguments];print('EXEC',command,flush=True)
            subprocess.run(command,cwd=ROOT,check=True)
        if args.resume_queue:
            if not (ROOT/plan).is_file():raise RuntimeError('preparation incomplete; preserve it and use a new tag')
        else:
            write_json(workspace/'environment.json',dict(root=str(ROOT.resolve()),utc=datetime.now(timezone.utc).isoformat(),
                PID=os.getpid(),python=platform.python_version(),nvidia_smi=smi,process_snapshot=processes,
                tag=args.tag,location='AutoDL',status='preparation_started_not_training',
                generation_spec_sha256=sha(ROOT/'config/mu_functions_v2_generation.json')))
            # This runs the formal generator directly. There is no pilot
            # performance qualification step or traditional solver dependency.
            call('scripts/build_mu_functions_v2.py','--output',data)
            call('scripts/audit_mu_functions_v2.py','--dataset',data,'--output',audit,'--fine-output',fine)
            call('scripts/freeze_mu_functions_v2.py','--dataset',data,'--audit',audit+'/audit.json','--fine-dataset',fine,'--output',frozen)
            call('scripts/plot_mu_functions_v2_data.py','--dataset',data,'--output','paper/mu_functions_v2_data_'+args.tag)
        if not args.prepare_only:
            command=[sys.executable,'-u','server/run_jedc_mu_v1.py','--plan',plan,'--tag',args.tag,'--allow-other-gpu','--dev-audit']
            if args.resume_queue:command.append('--resume-queue')
            subprocess.run(command,cwd=ROOT,check=True)


if __name__=='__main__':main()
