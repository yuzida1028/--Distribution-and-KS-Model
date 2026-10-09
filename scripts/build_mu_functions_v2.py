"""AutoDL-only formal function bank; no reference policy or checkpoint needed."""
import argparse
import csv
import json
import sys
import time
import traceback
from pathlib import Path
import numpy as np
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from jedc_distribution_v1 import make_edges, midpoint_grid, rebin, sha, write_json, VERSION
from build_jedc_mu_v1 import official_initial, perturb, save_states, stats


def density_setup(edges, order=8):
    """Quadrature integrates the continuous density within each positive bin."""
    lo, hi = edges[:-1], edges[1:]
    active = hi <= 100. + 1e-10
    u, w = np.polynomial.legendre.leggauss(order)
    x = (lo[active, None]+hi[active, None])/2 + (hi[active, None]-lo[active, None])*u/2
    weights = (hi[active, None]-lo[active, None])*w/2
    basis = np.polynomial.legendre.legvander(2*x/100-1, 4)[..., 1:]
    return x, np.log(weights), basis, active


def tilted_mu(theta, target, population, atoms, edges, setup):
    """Shared exponential tilt preserves independently drawn group shapes.

    Match the K actually read by the fixed-grid network. The continuous first
    moment is independently reported, not silently substituted for this K.
    """
    x, logw, basis, active = setup
    grid = midpoint_grid(edges)
    base = np.einsum('ijq,eq->eij', basis, theta) + logw[None]
    positive = population*(1-atoms)
    def at(lam):
        logits = base + lam*x[None]
        logits -= logits.max(axis=(1, 2), keepdims=True)
        weights = np.exp(logits)
        weights /= weights.sum(axis=(1, 2), keepdims=True)
        masses = weights.sum(2)
        K = float(np.sum(positive[:, None]*masses*grid[1:][active]))
        return K, masses, float(np.sum(positive[:, None, None]*weights*x[None]))
    lower, upper = -1., 1.
    if not at(lower)[0] < target < at(upper)[0]:
        raise ValueError('target not bracketed by fixed tilt [-1,1]')
    for _ in range(52):
        mid = (lower+upper)/2
        if at(mid)[0] < target: lower = mid
        else: upper = mid
    tilt = (lower+upper)/2
    K, masses, continuous_K = at(tilt)
    mu = np.zeros((2, len(edges)))
    mu[:, 0] = population*atoms
    mu[:, 1:][:, active] = positive[:, None]*masses
    if abs(K-target) > 1e-9 or mu.min() < 0 or np.max(abs(mu.sum(1)-population)) > 1e-12:
        raise ValueError('density mass/target constraint')
    return mu, tilt, continuous_K


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--output', required=True)
    p.add_argument('--spec', default='config/mu_functions_v2_generation.json')
    p.add_argument('--fit', default='runs/jedc_mu_momentfit_20260929/report.json')
    p.add_argument('--fine', action='store_true')
    args = p.parse_args()
    spec = json.loads((ROOT/args.spec).read_text(encoding='utf-8'))
    fit = json.loads((ROOT/args.fit).read_text(encoding='utf-8'))
    if spec['groups'] != dict(train=24, dev=6, test=12) or spec['parents_per_group'] != 300:
        raise ValueError('formal v2 dimensions are fixed')
    if sha(ROOT/'data/jedc2010/initial_distribution.npz') != fit['source_sha256']:
        raise ValueError('fit and official initial mismatch')
    out = ROOT/args.output
    out.mkdir(parents=True, exist_ok=False)
    for name in ['train_groups', 'dev_groups', 'train_paths', 'dev_paths', 'sealed']:
        (out/name).mkdir()
    started = time.perf_counter()
    edges = make_edges(args.fine); grid = midpoint_grid(edges)
    initial, conversion = official_initial(edges)
    conditional_atoms = initial[:, 0]/initial.sum(1)
    theta0 = np.asarray([r['theta'] for r in fit['records']])
    K0 = float(fit['mean_K_exact_bins'])
    setup = density_setup(edges, spec['density_quadrature_order'])
    streams = np.random.SeedSequence(spec['root_seed']).spawn(42)
    shock_streams = np.random.SeedSequence(spec['shock_seed']).spawn(42)
    cal = json.loads((ROOT/'data/jedc2010_calibration.json').read_text(encoding='utf-8'))
    transition = np.asarray(cal['joint_transition'])
    splits = ['train']*24 + ['dev']*6 + ['test']*12
    banks = {k:dict(states=[],zs=[],ids=[],parents=[],variants=[]) for k in ['train','dev','test','shape']}
    groups, paths, diagnostics, attempts = [], [], [], []
    source = dict(provider='exponential_polynomial_function_library', classification=spec['classification'],
        spec=spec, spec_sha256=sha(ROOT/args.spec), fit_sha256=sha(ROOT/args.fit),
        calibration_sha256=sha(ROOT/'data/jedc2010_calibration.json'), initial_conversion=conversion,
        initial_sha256=sha(ROOT/'data/jedc2010/initial_distribution.npz'), representation=VERSION,
        generator_sha256=sha(Path(__file__)), transport_sha256=sha(ROOT/'src/jedc_distribution_v1.py'),
        policy_labels_used=False, young_required=False, official_K=K0,
        coefficient_note='theta0 + independent N(0,0.15^2) by employment and degree; common asset tilt after draw',
        numerical_rules=dict(K_tolerance=1e-9, mass_tolerance=1e-12, tilt_bracket=[-1,1], tilt_bisections=52),
        asset_upper=500, fine=args.fine, score_filtering=False)
    write_json(out/'state_source_manifest.json', source)
    protocol = dict(version='mu_functions_v2', formal=True, partition_unit='independent_function_generation_group',
        group_count=42, parents_per_group=300, root_seed=spec['root_seed'], shock_seed=spec['shock_seed'],
        test_score_access=False, groups=[], paths=[], status='building')
    write_json(out/'split_manifest.json', protocol)
    gid = index = -1
    try:
        for gid, (stream, shock_stream, split) in enumerate(zip(streams, shock_streams, splits)):
            draw_stream, shift_stream = stream.spawn(2)
            rng = np.random.default_rng(draw_stream); shifts = np.random.default_rng(shift_stream)
            zs = np.tile([0,1],150); rng.shuffle(zs)
            group_mu, group_theta, group_tilt, group_target, group_continuous = [], [], [], [], []
            selected = set(np.linspace(0,299,10,dtype=int).tolist())
            for index, zraw in enumerate(zs):
                z = int(zraw); population = np.array([.1,.9] if z == 0 else [.04,.96])
                theta = theta0 + rng.normal(0,spec['coefficient_sigma'],(2,4))
                target = K0*rng.uniform(*spec['target_K_relative_range'])
                mu, tilt, continuous_K = tilted_mu(theta,target,population,conditional_atoms,edges,setup)
                sid = 'g%02d_f%03d'%(gid,index)
                attempts.append(dict(group_id=gid,index=index,status='accepted_numerically',target_K=target))
                group_mu.append(mu.astype(np.float32)); group_theta.append(theta);group_tilt.append(tilt)
                group_target.append(target);group_continuous.append(continuous_K)
                variants = [('raw',0,True)] if split=='test' else [('raw',0,True),('sameK',.01,True),('localK',.03,False)]
                for variant, cap, same in variants:
                    m = mu if variant=='raw' else perturb(mu,grid,shifts,cap,same)[0]
                    b = banks[split];b['states'].append(m.astype(np.float32));b['zs'].append(z)
                    b['ids'].append(sid+'_'+variant);b['parents'].append(sid);b['variants'].append(variant)
                    diagnostics.append(dict(group_id=gid,index=index,split=split,variant=variant,
                        **stats(m,grid,edges,z),conditional_cdf_change=float(np.max(abs(np.cumsum(m-mu,axis=1))/population[:,None])),
                        relative_K_change=float(((m-mu)*grid).sum()/target),target_K=target,continuous_parent_K=continuous_K))
                if split=='test' and index in selected:
                    m = perturb(mu,grid,shifts,.03,True)[0];b = banks['shape']
                    b['states'].extend([mu.astype(np.float32),m.astype(np.float32)]);b['zs'].extend([z,z])
                    b['ids'].extend([sid+'_raw',sid+'_sameK']);b['parents'].extend([sid,sid]);b['variants'].extend(['raw','sameK'])
            directory = 'sealed' if split=='test' else split+'_groups'
            file = out/directory/('group_%02d.npz'%gid)
            np.savez_compressed(file,mu=np.asarray(group_mu),z=zs,theta=np.asarray(group_theta),
                tilt=np.asarray(group_tilt),target_K=np.asarray(group_target),continuous_K=np.asarray(group_continuous),grid=grid,edges=edges)
            groups.append(dict(group_id=gid,split=split,states=300,draw_spawn_key=list(draw_stream.spawn_key),
                perturb_spawn_key=list(shift_stream.spawn_key),file=str(file.relative_to(out)),sha256=sha(file)))
            shock_rng = np.random.default_rng(shock_stream);shocks = np.zeros(4000,dtype=np.int8)
            for t in range(1,4000):
                shocks[t] = int(shock_rng.random() >= transition[2*int(shocks[t-1]),:2].sum())
            directory = 'sealed' if split=='test' else split+'_paths'
            file = out/directory/('path_%02d.npz'%gid)
            np.savez_compressed(file,shocks=shocks)
            paths.append(dict(path_id=gid,split=split,file=str(file.relative_to(out)),sha256=sha(file),spawn_key=list(shock_stream.spawn_key)))
            write_json(out/'split_manifest.json',{**protocol,'groups':groups,'paths':paths})
            print('function group %02d %s complete; independent shocks saved'%(gid,split),flush=True)
        for name,b in banks.items():
            save_states(out/('sealed/'+name+'.npz' if name in ['test','shape'] else name+'.npz'),
                b['states'],b['zs'],b['ids'],b['parents'],b['variants'],grid,edges)
        for file, rows in [('distribution_diagnostics.csv',diagnostics),('generation_attempts.csv',attempts)]:
            with (out/file).open('w',newline='',encoding='utf-8') as f:
                w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
        files=['train.npz','dev.npz','sealed/test.npz','sealed/shape.npz','distribution_diagnostics.csv','generation_attempts.csv','state_source_manifest.json']
        write_json(out/'split_manifest.json',{**protocol,'status':'complete','groups':groups,'paths':paths,
            'counts':{k:len(b['states']) for k,b in banks.items()},'seconds':time.perf_counter()-started,
            'sha256':{f:sha(out/f) for f in files},'failed_candidates':0})
        (out/'sealed/DO_NOT_EVALUATE.txt').write_text('Frozen protocol and run registry required before final scores.\n')
    except BaseException:
        write_json(out/'failure.json',dict(status='failed_preserved',group_id=gid,index=index,
            traceback=traceback.format_exc(),completed_groups=groups,seconds=time.perf_counter()-started))
        raise


if __name__=='__main__':main()
