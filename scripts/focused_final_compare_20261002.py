"""Freeze the 9+3+3 Model B comparison, then evaluate one sealed entry.

The freeze subcommand hashes sealed files but never opens their arrays or scores.
The evaluate subcommand is the only entry point that opens final states/paths.
All outputs are exclusive new directories; existing evidence is never replaced.
"""
import argparse
import csv
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'server'))
from jedc_distribution_v1 import IntervalEconomy, sha, write_json
from evaluate_mu_functions_v2 import verify_plan
from run_jedc_mu_v1 import complete as neural_complete

NEURAL_TAG = 'mu971_main9_20260929_fix1'
MMV_TAG = 'mmv971_matched_review_20260930'
MMW_TAG = 'mmw971_sim_numeric_fix2_20261001'
SEEDS = (9711, 9712, 9713)
LABELS = ('fullmu_deeponet', 'fullmu_mlp', 'onlyK_mlp')
VERSION = 'focused_modelb_9plus3plus3_final_20261002'


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def expect_hash(path, digest):
    path = Path(path)
    if not path.is_file() or sha(path) != digest:
        raise ValueError(f'missing or changed frozen file: {path}')


def verify_external_binding(root, binding_path, *, mmv):
    binding = read_json(binding_path)
    expect_hash(binding['reference_plan'], binding['reference_plan_sha256'])
    for name, digest in binding['input_sha256'].items():
        expect_hash(binding['input_paths'][name], digest)
    for name, digest in binding['source_sha256'].items():
        expect_hash(root / name, digest)
    for config in binding['configs']:
        expect_hash(config['path'], config['sha256'])
    if mmv:
        expect_hash(binding_path.parent / 'evaluation_protocol.json', binding['protocol_sha256'])
        for path in binding['development_paths']:
            expect_hash(path['path'], path['sha256'])
    return binding


def artifacts(run, names):
    return {name: sha(run / name) for name in names}


def completed_development(run, folder):
    report = read_json(run / folder / 'report.json')
    failures = read_json(run / folder / 'failures.json')
    if (report['status'] != 'complete' or report['scored_states'] != 5400
            or report['expected_states'] != 5400 or report['sameK_pairs'] != 1800
            or report['failure_count'] != 0 or failures):
        raise ValueError(f'incomplete development evaluation: {run / folder}')
    return report


def freeze(args):
    plan_path = Path(args.plan).resolve()
    plan = verify_plan(plan_path)
    mmv_root = Path(args.mmv_root).resolve()
    mmw_root = Path(args.mmw_root).resolve()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f'final registry exists; preserve it: {out}')
    if plan_path.parents[2] != ROOT:
        raise ValueError('install this script in the frozen main workspace')
    if plan['data_directory'].startswith('/'):
        raise ValueError('unexpected absolute dataset path')
    data = ROOT / plan['data_directory']
    split_path = data / 'split_manifest.json'
    split = read_json(split_path)
    if split['counts'] != dict(train=21600, dev=5400, test=3600, shape=240):
        raise ValueError('frozen data counts changed')
    if len([x for x in split['paths'] if x['split'] == 'test']) != 12:
        raise ValueError('expected 12 sealed shock paths')
    # Hash bytes only; no np.load on sealed states or paths in this phase.
    for name, digest in plan['sealed_sha256'].items():
        expect_hash(data / name, digest)
    for path in plan['sealed_paths']:
        expect_hash(data / path['file'], path['sha256'])
    mmv_binding_path = mmv_root / 'config' / ('mmv2010_' + MMV_TAG) / 'binding.json'
    mmw_binding_path = mmw_root / 'config' / MMW_TAG / 'binding.json'
    mmv_binding = verify_external_binding(mmv_root, mmv_binding_path, mmv=True)
    mmw_binding = verify_external_binding(mmw_root, mmw_binding_path, mmv=False)
    if (Path(mmv_binding['reference_plan']).resolve() != plan_path
            or Path(mmw_binding['reference_plan']).resolve() != plan_path):
        raise ValueError('benchmark bound to a different reference plan')
    for root, binding in ((mmv_root, mmv_binding), (mmw_root, mmw_binding)):
        for name in ('src/jedc_distribution_v1.py', 'src/mu_evaluation_v2.py'):
            if sha(root / name) != sha(ROOT / name):
                raise ValueError(f'shared evaluator differs in {root}: {name}')
    runs = []
    selected = [x for x in plan['configs'] if x['phase'] == 'main' and x['label'] in LABELS]
    if len(selected) != 9 or {(x['label'], x['seed']) for x in selected} != {
            (label, seed) for label in LABELS for seed in SEEDS}:
        raise ValueError('expected exactly nine frozen main neural configurations')
    for item in selected:
        run = ROOT / 'runs' / (item['kind'] + '_' + Path(item['config']).stem + '_' + NEURAL_TAG)
        manifest = read_json(run / 'manifest.json')
        if manifest['sha256']['config'] != item['sha256'] or not neural_complete(run):
            raise ValueError(f'neural run/config or step mismatch: {run}')
        if completed_development(run, 'evaluation_v2_dev_native')['policy_sha256'] != sha(run / 'policy.pt'):
            raise ValueError(f'neural static policy differs: {run}')
        dynamic = read_json(run / 'evaluation_v2_dev_own_dynamic_focus1' / 'report.json')
        if (dynamic.get('failure_count') != 0 or len(dynamic.get('dynamic_paths', [])) != 12
                or dynamic.get('policy_sha256') != sha(run / 'policy.pt')):
            raise ValueError(f'neural development dynamics incomplete: {run}')
        files = artifacts(run, ('policy.pt', 'checkpoint.pt', 'history.csv', 'report.json', 'manifest.json',
            'evaluation_v2_dev_native/report.json', 'evaluation_v2_dev_native/failures.json',
            'evaluation_v2_dev_own_dynamic_focus1/report.json',
            'evaluation_v2_dev_own_dynamic_focus1/failures.json'))
        runs.append(dict(id=f"{item['label']}_s{item['seed']}", method=item['label'], seed=item['seed'],
                         family='neural', run_dir=str(run), config_path=str(ROOT / item['config']),
                         config_sha256=item['sha256'], policy_sha256=files['policy.pt'], artifact_sha256=files))
    for seed in SEEDS:
        run = mmv_root / 'runs' / ('mmv2010_queue_' + MMV_TAG) / f'matched_s{seed}'
        report = read_json(run / 'report.json')
        manifest = read_json(run / 'manifest.json')
        config = next(x for x in mmv_binding['configs'] if x['profile'] == 'matched' and x['seed'] == seed)
        if (report['status'] != 'converged' or manifest['binding_sha256'] != sha(mmv_binding_path)
                or manifest['config_sha256'] != config['sha256']
                or report['policy_sha256'] != sha(run / 'policy_table.npz')):
            raise ValueError(f'MMV run invalid: {run}')
        static = completed_development(run, 'development_static')
        dynamic = completed_development(run, 'development_with_dynamics')
        if (len(dynamic['dynamic_paths']) != 12
                or static['policy_sha256'] != report['policy_sha256']
                or dynamic['policy_sha256'] != report['policy_sha256']):
            raise ValueError(f'MMV own paths incomplete: {run}')
        files = artifacts(run, ('policy_table.npz', 'report.json', 'manifest.json', 'config.json',
            'development_static/report.json', 'development_static/failures.json',
            'development_with_dynamics/report.json', 'development_with_dynamics/failures.json'))
        runs.append(dict(id=f'mmv_matched_s{seed}', method='mmv_matched', seed=seed, family='mmv',
                         run_dir=str(run), config_path=config['path'], config_sha256=config['sha256'],
                         binding_path=str(mmv_binding_path), binding_sha256=sha(mmv_binding_path),
                         policy_sha256=files['policy_table.npz'], artifact_sha256=files))
    for seed in SEEDS:
        run = mmw_root / 'runs' / MMW_TAG / f'seed_{seed}'
        report = read_json(run / 'report.json')
        manifest = read_json(run / 'manifest.json')
        config = next(x for x in mmw_binding['configs'] if x['seed'] == seed)
        if (report['status'] != 'training_complete' or report['steps'] != 100000
                or manifest['binding_sha256'] != sha(mmw_binding_path)
                or manifest['config_sha256'] != config['sha256']
                or report['policy_sha256'] != sha(run / 'policy.pt')):
            raise ValueError(f'MMW run invalid: {run}')
        static = completed_development(run, 'development_static')
        dynamic = completed_development(run, 'development_with_dynamics')
        if (len(dynamic['dynamic_paths']) != 12
                or static['policy_sha256'] != report['policy_sha256']
                or dynamic['policy_sha256'] != report['policy_sha256']):
            raise ValueError(f'MMW own paths incomplete: {run}')
        files = artifacts(run, ('policy.pt', 'report.json', 'manifest.json', 'config.json', 'history.csv',
            'development_static/report.json', 'development_static/failures.json',
            'development_with_dynamics/report.json', 'development_with_dynamics/failures.json'))
        runs.append(dict(id=f'mmw_s{seed}', method='mmw', seed=seed, family='mmw',
                         run_dir=str(run), config_path=config['path'], config_sha256=config['sha256'],
                         binding_path=str(mmw_binding_path), binding_sha256=sha(mmw_binding_path),
                         policy_sha256=files['policy.pt'], artifact_sha256=files))
    if len(runs) != 15 or len({x['id'] for x in runs}) != 15:
        raise ValueError('final registry must contain exactly 15 policies')
    protocol = dict(version=VERSION, main_neural_seeds=list(SEEDS), main_neural_tag=NEURAL_TAG,
        mmv_tag=MMV_TAG, mmw_tag=MMW_TAG, grid='native500',
        common='sealed raw original states, 12 groups x 300 parents; shared actual-next-mu Euler/KKT',
        shape='sealed 120 fixed-discrete-K pairs, both endpoint KKT and mass-weighted saving response',
        dynamics='12 own-policy 4000-date paths, first 1000 transition then 3000 post-burn',
        strict_failure='nonfinite/invalid metrics, mass or employment error >1e-4, either price floor, policy domain exit',
        partial='retain every failed state/path and report coverage; successful-only mean is conditional',
        cost='CPU MMV solve versus GPU neural training separately; final evaluator GPU time per policy',
        no_test_selection=True, no_extra_training=True)
    registry = dict(version=VERSION, created_utc=datetime.now(timezone.utc).isoformat(),
        final_scores_read_at_creation=False, plan_path=str(plan_path), plan_sha256=sha(plan_path),
        data_dir=str(data), split_sha256=sha(split_path), sealed_sha256=plan['sealed_sha256'],
        sealed_paths=plan['sealed_paths'], script_sha256=sha(Path(__file__)),
        mmv_root=str(mmv_root), mmw_root=str(mmw_root),
        mmv_binding_sha256=sha(mmv_binding_path), mmw_binding_sha256=sha(mmw_binding_path),
        protocol=protocol, runs=runs)
    out.mkdir(parents=True, exist_ok=False)
    write_json(out / 'focused_final_registry.json', registry)
    print(f'FROZEN {len(runs)} policies; no final array opened; registry SHA256={sha(out / "focused_final_registry.json")}', flush=True)


def save_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_registry(registry_path):
    registry = read_json(registry_path)
    if registry['version'] != VERSION:
        raise ValueError('wrong focused registry version')
    expect_hash(registry['plan_path'], registry['plan_sha256'])
    expect_hash(Path(registry['data_dir']) / 'split_manifest.json', registry['split_sha256'])
    expect_hash(Path(__file__), registry['script_sha256'])
    plan = verify_plan(Path(registry['plan_path']))
    mmv_root = Path(registry['mmv_root'])
    mmw_root = Path(registry['mmw_root'])
    mmv_binding_path = mmv_root / 'config' / ('mmv2010_' + MMV_TAG) / 'binding.json'
    mmw_binding_path = mmw_root / 'config' / MMW_TAG / 'binding.json'
    expect_hash(mmv_binding_path, registry['mmv_binding_sha256'])
    expect_hash(mmw_binding_path, registry['mmw_binding_sha256'])
    verify_external_binding(mmv_root, mmv_binding_path, mmv=True)
    verify_external_binding(mmw_root, mmw_binding_path, mmv=False)
    for name, digest in registry['sealed_sha256'].items():
        expect_hash(Path(registry['data_dir']) / name, digest)
    for path in registry['sealed_paths']:
        expect_hash(Path(registry['data_dir']) / path['file'], path['sha256'])
    if registry['sealed_sha256'] != plan['sealed_sha256']:
        raise ValueError('sealed manifest differs')
    for entry in registry['runs']:
        run = Path(entry['run_dir'])
        expect_hash(entry['config_path'], entry['config_sha256'])
        for name, digest in entry['artifact_sha256'].items():
            expect_hash(run / name, digest)
        if entry['family'] != 'neural':
            expect_hash(entry['binding_path'], entry['binding_sha256'])
    return registry


def load_policy(entry, registry, edges, device):
    import numpy as np
    import torch
    from mu_evaluation_v2 import FrozenGridPolicy
    cfg = read_json(entry['config_path'])
    calibration = ROOT / 'data/jedc2010_calibration.json'
    tensor_edges = torch.tensor(edges, dtype=torch.float32, device=device)
    if entry['family'] == 'neural':
        from networks_mu_v2 import MuPolicyNetwork
        with np.load(ROOT / cfg['train_data_file']) as bank:
            input_edges = bank['edges'].copy()
            grid = bank['grid'].copy()
        if not np.array_equal(input_edges, edges):
            raise ValueError('native evaluation grid differs from frozen neural input')
        kind = 'deeponet' if entry['method'] == 'fullmu_deeponet' else 'mlp'
        hidden = cfg['hidden'] if kind == 'deeponet' else cfg['mlp_hidden']
        net = MuPolicyNetwork(kind, len(grid), cfg['asset_max'], hidden, cfg['basis'],
            macro_features=cfg['macro_features'], grid=torch.tensor(grid, device=device)).to(device)
        net.load_state_dict(torch.load(Path(entry['run_dir']) / 'policy.pt', map_location=device, weights_only=True))
        net.eval()
        econ = IntervalEconomy(calibration, tensor_edges, device, policy_mode=cfg['policy_mode'])
        econ.asset_max = cfg['asset_max']
        return econ, FrozenGridPolicy(net, tensor_edges, torch.tensor(input_edges, dtype=torch.float32, device=device)).to(device)
    if entry['family'] == 'mmv':
        sys.path.insert(0, str(Path(registry['mmv_root']) / 'src'))
        from mmv2010_bridge import MMVIntervalEconomy, TorchMMVPolicy
        econ = MMVIntervalEconomy(calibration, tensor_edges, device)
        net = TorchMMVPolicy(Path(entry['run_dir']) / 'policy_table.npz').to(device)
        net.eval()
        return econ, net
    sys.path.insert(0, str(Path(registry['mmw_root']) / 'src'))
    from mmw_jedc import MMWEconomy, MMWNetwork
    binding = read_json(entry['binding_path'])
    with np.load(binding['input_paths']['train']) as bank:
        input_edges = bank['edges'].copy()
        grid = bank['grid'].copy()
    if not np.array_equal(input_edges, edges):
        raise ValueError('native evaluation grid differs from frozen MMW input')
    econ = MMWEconomy(calibration, tensor_edges, device)
    net = MMWNetwork(torch.tensor(grid, dtype=torch.float32, device=device), cfg['asset_max'],
                     cfg['hidden'], cfg['share_bias']).to(device)
    net.load_state_dict(torch.load(Path(entry['run_dir']) / 'policy.pt', map_location=device, weights_only=True))
    net.eval()
    return econ, net


def evaluate(args):
    import numpy as np
    import torch
    from jedc_distribution_v1 import rebin
    from mu_evaluation_v2 import metrics, quantiles
    from build_jedc_mu_v1 import official_initial

    if not torch.cuda.is_available():
        raise RuntimeError('focused final evaluation requires the RiverMind GPU')
    registry_path = Path(args.registry).resolve()
    registry = verify_registry(registry_path)
    entry = next((x for x in registry['runs'] if x['id'] == args.entry_id), None)
    if entry is None:
        raise ValueError('entry not in frozen 15-policy registry')
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(f'existing final output is preserved: {output}')
    data = Path(registry['data_dir'])
    split = read_json(data / 'split_manifest.json')
    paths = [x for x in split['paths'] if x['split'] == 'test']
    if len(paths) != 12 or {x['path_id'] for x in paths} != {
            x['path_id'] for x in registry['sealed_paths']}:
        raise ValueError('sealed shock path list changed')
    torch.set_num_threads(1)
    device = torch.device('cuda:0')
    started = time.perf_counter()
    with np.load(data / 'sealed/test.npz') as bank:
        edges = bank['edges'].astype(float)
        states = bank['mu'].copy()
        shocks = bank['z'].copy()
        state_ids = bank['state_id'].copy()
        parents = bank['parent_state_id'].copy()
    if len(states) != 3600 or edges[-1] != 500:
        raise ValueError('unexpected sealed native test grid or state count')
    econ, policy = load_policy(entry, registry, edges, device)
    output.mkdir(parents=True, exist_ok=False)
    failures = []
    common = []
    for i in range(len(states)):
        try:
            current = rebin(states[i].astype(float), edges, edges)
            mass = torch.tensor(current[None], dtype=torch.float32, device=device)
            z = torch.tensor([int(shocks[i])], device=device)
            scored = metrics(econ, policy, mass, z)
            parent = str(parents[i])
            common.append(dict(state_id=str(state_ids[i]), parent=parent,
                               group_id=int(parent[1:3]), variant='raw', **scored))
        except Exception:
            failures.append(dict(stage='common', state_id=str(state_ids[i]),
                                 traceback=traceback.format_exc()))
        if i % 300 == 0:
            print(f'{entry["id"]} common {i}/{len(states)}', flush=True)
    save_csv(output / 'common_states.csv', common)
    group_rows = []
    for group in sorted({x['group_id'] for x in common}):
        rows = [x for x in common if x['group_id'] == group]
        group_rows.append(dict(group_id=group, n=len(rows), **{
            key: float(np.mean([x[key] for x in rows]))
            for key in ('KKT', 'P90', 'P99', 'max_populated_KKT')}))
    save_csv(output / 'common_original_group_summary.csv', group_rows)

    with np.load(data / 'sealed/shape.npz') as bank:
        shape_edges = bank['edges'].astype(float)
        shape_mu = bank['mu'].copy()
        shape_z = bank['z'].copy()
        shape_parents = bank['parent_state_id'].copy()
    if len(shape_mu) != 240:
        raise ValueError('expected 120 frozen fixed-capital pairs')
    shape_rows = []
    for i in range(0, len(shape_mu), 2):
        try:
            scores, choices, masses = [], [], []
            for j in (i, i + 1):
                current = rebin(shape_mu[j].astype(float), shape_edges, edges)
                mass = torch.tensor(current[None], dtype=torch.float32, device=device)
                z = torch.tensor([int(shape_z[j])], device=device)
                scores.append(metrics(econ, policy, mass, z))
                with torch.no_grad():
                    assets, employment = econ.queries(1)
                    saving, _ = econ.policy(policy, mass, z, assets, employment)
                    choices.append(saving.cpu().numpy()[0])
                masses.append(current)
            parent = str(shape_parents[i])
            if parent != str(shape_parents[i + 1]):
                raise ValueError('shape pair parent mismatch')
            shape_rows.append(dict(parent=parent, group_id=int(parent[1:3]),
                K_difference=scores[1]['K'] - scores[0]['K'],
                policy_response_mass_MAE=float((masses[0] * abs(choices[1] - choices[0])).sum()),
                policy_response_max=float(abs(choices[1] - choices[0]).max()),
                KKT_raw=scores[0]['KKT'], KKT_sameK=scores[1]['KKT']))
        except Exception:
            failures.append(dict(stage='shape', parent=str(shape_parents[i]),
                                 traceback=traceback.format_exc()))
    save_csv(output / 'sameK_shape.csv', shape_rows)

    path_summary = []
    for path in paths:
        path_id = int(path['path_id'])
        expect_hash(data / path['file'], path['sha256'])
        with np.load(data / path['file']) as bank:
            shock_path = bank['shocks'].copy()
        if len(shock_path) != 4000:
            raise ValueError(f'sealed shock path {path_id} has wrong length')
        initial, _ = official_initial(edges)
        mass = torch.tensor(initial[None], dtype=torch.float32, device=device)
        initial_cdf = np.cumsum(initial, axis=1)
        trajectory, sample_mu, sample_dates = [], [], []
        failed = False
        for t, zvalue in enumerate(shock_path):
            try:
                z = torch.tensor([int(zvalue)], device=device)
                current = mass.cpu().numpy()[0]
                score = metrics(econ, policy, mass, z)
                grid = econ.grid.cpu().numpy()
                qs = quantiles(current, edges)
                row = dict(path_id=path_id, t=t,
                    phase='transition' if t < 1000 else 'post_burn', z=int(zvalue), **score,
                    mass_error=abs(float(current.sum()) - 1),
                    employment_error=abs(float(current[0].sum()) - [.1, .04][int(zvalue)]),
                    zero_atom=float(current[:, 0].sum()),
                    tail_mass=float(current[:, grid >= 490].sum()),
                    tail_capital=float((current[:, grid >= 490] * grid[grid >= 490]).sum()),
                    moment2=float((current * grid**2).sum()),
                    moment3=float((current * grid**3).sum()),
                    moment4=float((current * grid**4).sum()),
                    q10=qs[0], q50=qs[1], q90=qs[2], q99=qs[3],
                    CDF_distance_own_initial=float(np.max(abs(np.cumsum(current, axis=1) - initial_cdf))))
                trajectory.append(row)
                if t % 10 == 0 or t == len(shock_path) - 1:
                    sample_mu.append(current.copy())
                    sample_dates.append(t)
                if (row['mass_error'] > 1e-4 or row['employment_error'] > 1e-4
                        or score['price_capital_floor_active'] or score['price_labor_floor_active']):
                    raise ValueError('strict dynamic mass/employment/price-floor failure')
                if t + 1 < len(shock_path):
                    with torch.no_grad():
                        assets, employment = econ.edge_queries(1)
                        saving, _ = econ.policy(policy, mass, z, assets, employment)
                        mass = econ.push_distribution(mass, z, saving, int(shock_path[t + 1]))
                if t % 500 == 0:
                    print(f'{entry["id"]} path {path_id} date {t}', flush=True)
            except Exception:
                failed = True
                failures.append(dict(stage='dynamic', path_id=path_id, date=t,
                                     traceback=traceback.format_exc()))
                break
        save_csv(output / f'path_{path_id:02d}_own_dynamic.csv', trajectory)
        np.savez_compressed(output / f'path_{path_id:02d}_distribution_samples.npz',
                            mu=np.asarray(sample_mu), dates=np.asarray(sample_dates), edges=edges)
        for phase, expected in (('transition', 1000), ('post_burn', 3000)):
            rows = [x for x in trajectory if x['phase'] == phase]
            path_summary.append(dict(path_id=path_id, phase=phase, failed=failed,
                dates=len(rows), expected_dates=expected,
                KKT=float(np.mean([x['KKT'] for x in rows])) if rows else None,
                P99=float(np.mean([x['P99'] for x in rows])) if rows else None,
                K_final=trajectory[-1]['K'] if trajectory else None))
    save_csv(output / 'dynamic_path_summary.csv', path_summary)
    write_json(output / 'failures.json', failures)
    torch.cuda.synchronize()
    complete_paths = sum(
        all(x['dates'] == x['expected_dates'] and not x['failed']
            for x in path_summary if x['path_id'] == path['path_id'])
        for path in paths)
    write_json(output / 'report.json', dict(version=VERSION,
        status='complete' if not failures else 'complete_with_failures',
        entry_id=entry['id'], method=entry['method'], seed=entry['seed'],
        registry_sha256=sha(registry_path), policy_sha256=entry['policy_sha256'],
        bank_sha256=sha(data / 'sealed/test.npz'), shape_sha256=sha(data / 'sealed/shape.npz'),
        expected_states=3600, scored_states=len(common), expected_shape_pairs=120,
        scored_shape_pairs=len(shape_rows), expected_paths=12, complete_paths=complete_paths,
        failure_count=len(failures), groups=group_rows, dynamic_paths=path_summary,
        seconds=time.perf_counter() - started,
        peak_allocated_bytes=torch.cuda.max_memory_allocated(device),
        peak_reserved_bytes=torch.cuda.max_memory_reserved(device),
        gpu=torch.cuda.get_device_name(device), torch=torch.__version__, python=platform.python_version(),
        final_data_read=True, interpretation='common exogenous states and each policy own dynamic states reported separately'))
    print(f'FINAL {entry["id"]}: {len(common)}/3600 states, {len(shape_rows)}/120 pairs, '
          f'{complete_paths}/12 strict paths, failures={len(failures)}', flush=True)


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze')
    p.add_argument('--plan', required=True)
    p.add_argument('--mmv-root', required=True)
    p.add_argument('--mmw-root', required=True)
    p.add_argument('--output', required=True)
    q = sub.add_parser('evaluate')
    q.add_argument('--registry', required=True)
    q.add_argument('--entry-id', required=True)
    q.add_argument('--output', required=True)
    args = parser.parse_args()
    if platform.system() != 'Linux':
        raise RuntimeError('scientific freeze/evaluation only on RiverMind Linux')
    if args.command == 'freeze':
        freeze(args)
    else:
        evaluate(args)


if __name__ == '__main__':
    main()
