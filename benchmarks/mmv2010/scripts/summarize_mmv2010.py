"""Cloud-only, complete-coverage development comparison; no test access."""
import argparse,csv,json,platform,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'src'))
from mmv2010_protocol import verify
from jedc_distribution_v1 import sha,write_json

def read(path):
    with Path(path).open(newline='') as f:return list(csv.DictReader(f))
def save(path,rows):
    with Path(path).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
def main():
    p=argparse.ArgumentParser();p.add_argument('--binding',required=True);p.add_argument('--queue',required=True)
    p.add_argument('--deep-tag',required=True);p.add_argument('--output',required=True);a=p.parse_args()
    if platform.system()!='Linux':raise RuntimeError('scientific summaries belong on cloud')
    b=verify(a.binding);rows=[];costs=[];provenance=[]
    keys=['KKT','P90','P99','max_populated_KKT']
    for entry in b['reference_configs']:
        seed=entry['seed'];deep=Path(b['reference_base'])/'runs'/('deeponet_'+Path(entry['config']).stem+'_'+a.deep_tag)
        mmv=Path(a.queue)/f'matched_s{seed}'
        dm=json.loads((deep/'manifest.json').read_text());mr=json.loads((mmv/'report.json').read_text())
        if dm['sha256']['config']!=entry['sha256'] or mr['status']!='converged' or sha(mmv/'policy_table.npz')!=mr['policy_sha256']:raise ValueError('mismatched/unconverged run')
        if json.loads((mmv/'manifest.json').read_text())['binding_sha256']!=sha(a.binding):raise ValueError('MMV binding mismatch')
        dev=deep/'evaluation_v2_dev_native';mv=mmv/'development_static'
        for folder in [dev,mv]:
            rr=json.loads((folder/'report.json').read_text())
            if rr['status']!='complete':raise ValueError('incomplete coverage: retain failures, no success-only primary mean '+str(folder))
        de=json.loads((dev/'report.json').read_text());me=json.loads((mv/'report.json').read_text())
        if de['policy_sha256']!=sha(deep/'policy.pt') or me['policy_sha256']!=sha(mmv/'policy_table.npz'):raise ValueError('evaluation checkpoint mismatch')
        if de['bank_sha256']!=b['input_sha256']['dev'] or me['dev_sha256']!=b['input_sha256']['dev'] or de['plan_sha256']!=b['reference_plan_sha256']:raise ValueError('evaluation data/protocol mismatch')
        dr=read(dev/'common_states.csv');vr=read(mv/'common_states.csv')
        if len(dr)!=5400 or len(vr)!=5400:raise ValueError('expected all5400 dev states')
        d={r['state_id']:r for r in dr};v={r['state_id']:r for r in vr}
        if len(d)!=5400 or set(d)!=set(v):raise ValueError('duplicate/unpaired states')
        for group in range(24,30):
            ids=[k for k in d if int(d[k]['group_id'])==group and d[k]['variant']=='raw']
            if len(ids)!=300:raise ValueError('raw generation group missing')
            row=dict(seed=seed,group_id=group,parents=len(ids))
            for metric in keys:
                dv=sum(float(d[k][metric]) for k in ids)/len(ids);vv=sum(float(v[k][metric]) for k in ids)/len(ids)
                row['DeepONet_'+metric]=dv;row['MMV_'+metric]=vv;row['MMV_minus_DeepONet_'+metric]=vv-dv
            rows.append(row)
        training=json.loads((deep/'report.json').read_text())
        costs.append(dict(seed=seed,DeepONet_GPU=training['gpu'],DeepONet_train_seconds=training['train_seconds_accumulated'],
            MMV_CPU_solve_seconds=mr['solve_seconds_accumulated'],MMV_setup_seconds_current_invocation=mr['setup_seconds_this_invocation'],
            DeepONet_dev_seconds=json.loads((dev/'report.json').read_text())['seconds'],MMV_dev_seconds=json.loads((mv/'report.json').read_text())['seconds']))
        provenance.append(dict(seed=seed,DeepONet_policy_sha256=sha(deep/'policy.pt'),MMV_policy_sha256=sha(mmv/'policy_table.npz'),
            DeepONet_csv_sha256=sha(dev/'common_states.csv'),MMV_csv_sha256=sha(mv/'common_states.csv')))
    out=Path(a.output);out.mkdir(parents=True,exist_ok=False)
    save(out/'paired_raw_group_metrics.csv',rows);save(out/'costs_separate_devices.csv',costs)
    write_json(out/'report.json',dict(status='complete_development_only',binding_sha256=sha(a.binding),provenance=provenance,
        final_data_read=False,groups=6,seeds=3,parent_states_per_seed=1800,
        interpretation='generation group and solver/training seed are uncertainty units; no state-level significance; CPU/GPU solve costs separately labelled'))
if __name__=='__main__':main()
