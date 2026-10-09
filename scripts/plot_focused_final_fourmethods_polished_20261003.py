"""Make source-audited final figures on RiverMind from the frozen 15-run comparison.

Scientific summaries and plots must run on RiverMind. This script reads the
existing final files and never starts evaluation or overwrites an output.
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'scripts'), str(ROOT / 'src')]
from focused_final_compare_20261002 import verify_registry
from jedc_distribution_v1 import sha, write_json

EXPECTED_REGISTRY_SHA = '8abc0fba54a71ac6904cb1703b1e21c86594241eda12446ac34103ff50f906dc'
ORDER = ('fullmu_deeponet', 'mmw', 'fullmu_mlp', 'onlyK_mlp')
LABEL = {
    'fullmu_deeponet': 'DeepONet', 'fullmu_mlp': 'Full $\\mu$ MLP',
    'onlyK_mlp': 'Capital-only MLP',
    'mmw': 'MMW adapted',
}
COLOR = {
    'fullmu_deeponet': '#0072B2', 'fullmu_mlp': '#009E73',
    'onlyK_mlp': '#D55E00', 'mmw': '#CC79A7',
}
SEEDS = (9711, 9712, 9713)


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def save_csv(path, rows):
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def require(condition, message):
    if not condition:
        raise ValueError(message)


def finite_positive(value):
    return math.isfinite(float(value)) and float(value) > 0


def finite_nonnegative(value):
    return math.isfinite(float(value)) and float(value) >= 0


def figure_style():
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
        'font.size': 9,
        'axes.spines.top': False, 'axes.spines.right': False,
        'axes.labelsize': 9, 'axes.titlesize': 10,
        'axes.linewidth': .8, 'xtick.major.width': .7, 'ytick.major.width': .7,
        'legend.frameon': False,
        'figure.dpi': 140, 'savefig.dpi': 300,
        'svg.fonttype': 'none',
        'pdf.fonttype': 42,
    })


def export(fig, directory, stem):
    for ext in ('png', 'svg', 'pdf'):
        fig.savefig(directory / f'{stem}.{ext}', bbox_inches='tight')
    fig.savefig(directory / f'{stem}.tiff', dpi=600, bbox_inches='tight')
    plt.close(fig)


def load_checked(registry_path, attempt, audit):
    registry_path, attempt, audit = map(Path, (registry_path, attempt, audit))
    require(sha(registry_path) == EXPECTED_REGISTRY_SHA, 'unexpected frozen registry hash')
    registry = verify_registry(registry_path)
    require(len(registry['runs']) == 15, 'expected all 15 frozen policies')
    sources = [{'path': str(registry_path), 'sha256': sha(registry_path)}]
    audit_report = audit / 'audit_report.json'
    ar = read_json(audit_report)
    require(ar['registry_sha256'] == EXPECTED_REGISTRY_SHA and ar['entries_reported'] == 15
            and ar['entries_full_coverage'] == 15 and ar['issue_count'] == 0,
            'final raw-file audit is incomplete')
    sources.append({'path': str(audit_report), 'sha256': sha(audit_report)})
    audit_files = {}
    for name in ('entry_summary.csv', 'group_raw_KKT.csv', 'path_post_KKT.csv'):
        file = audit / name
        audit_files[name] = read_csv(file)
        sources.append({'path': str(file), 'sha256': sha(file)})
    summaries = audit_files['entry_summary.csv']
    groups = audit_files['group_raw_KKT.csv']
    paths = audit_files['path_post_KKT.csv']
    require(len(summaries) == 15 and len(groups) == 180 and len(paths) == 180,
            'audit CSV dimensions changed')
    ids = {entry['id'] for entry in registry['runs']}
    require({row['entry_id'] for row in summaries} == ids, 'audit summary IDs changed')
    require({row['entry_id'] for row in groups} == ids, 'audit group IDs changed')
    require({row['entry_id'] for row in paths} == ids, 'audit path IDs changed')
    shape_summary = []
    for entry in registry['runs']:
        key = entry['id']
        report_path = attempt / key / 'report.json'
        failure_path = attempt / key / 'failures.json'
        shape_path = attempt / key / 'sameK_shape.csv'
        report = read_json(report_path)
        failures = read_json(failure_path)
        shape = read_csv(shape_path)
        sources += [{'path': str(p), 'sha256': sha(p)}
                    for p in (report_path, failure_path, shape_path)]
        require(report['status'] == 'complete' and report['registry_sha256'] == EXPECTED_REGISTRY_SHA
                and report['failure_count'] == 0 and not failures
                and report['scored_states'] == 3600 and report['scored_shape_pairs'] == 120
                and report['complete_paths'] == 12 and len(shape) == 120,
                f'incomplete entry: {key}')
        require(len({row['parent'] for row in shape}) == 120, f'duplicate shape pair: {key}')
        summary = next(row for row in summaries if row['entry_id'] == key)
        require(summary['full_coverage'] == 'True' and int(summary['failures']) == 0,
                f'audit summary incomplete: {key}')
        require(all(finite_nonnegative(row['policy_response_mass_MAE']) for row in shape),
                f'invalid saving response: {key}')
        for measure in ('KKT_raw', 'KKT_sameK'):
            require(all(finite_nonnegative(row[measure]) for row in shape),
                    f'invalid shape measure {measure}: {key}')
        shape_summary.append({
            'entry_id': key, 'method': entry['method'], 'seed': entry['seed'],
            'pairs': 120,
            'response_MAE': statistics.fmean(float(row['policy_response_mass_MAE']) for row in shape),
            'KKT_raw': statistics.fmean(float(row['KKT_raw']) for row in shape),
            'KKT_sameK': statistics.fmean(float(row['KKT_sameK']) for row in shape),
            'max_abs_K_difference': max(abs(float(row['K_difference'])) for row in shape),
        })
    for method in ORDER:
        require({int(row['seed']) for row in summaries if row['method'] == method} == set(SEEDS),
                f'missing seeds: {method}')
        for seed in SEEDS:
            subset = [r for r in groups if r['method'] == method and int(r['seed']) == seed]
            require(len(subset) == 12 and all(int(r['n']) == 300 for r in subset),
                    f'incomplete original-state groups: {method}/{seed}')
            own = [r for r in paths if r['method'] == method and int(r['seed']) == seed]
            require(len(own) == 12 and all(r['strict_complete'] == 'True'
                    and int(r['dates']) == 4000 and finite_positive(r['post_burn_KKT']) for r in own),
                    f'incomplete own paths: {method}/{seed}')
    return registry, summaries, groups, paths, shape_summary, sources


ASSET_CELLS = tuple((f'e{e}_{band}', f'Employment {e}: {band}')
                    for e in (0, 1) for band in ('zero', 'low', 'mid', 'high'))


def load_detailed(registry, attempt, sources):
    """Stream existing final CSVs; retain only 100-date trace checkpoints."""
    attempt = Path(attempt)
    cells, phase_rows, traces = [], [], []
    path_ids = sorted(int(item['path_id']) for item in registry['sealed_paths'])
    require(len(path_ids) == 12 and len(set(path_ids)) == 12,
            'expected 12 distinct frozen shock paths')
    for entry in registry['runs']:
        key = entry['id']
        directory = attempt / key
        common_file = directory / 'common_states.csv'
        phase_file = directory / 'dynamic_path_summary.csv'
        acc = {cell: [0.0, 0.0] for cell, _ in ASSET_CELLS}
        state_count = 0
        with common_file.open(newline='', encoding='utf-8') as stream:
            for row in csv.DictReader(stream):
                state_count += 1
                for cell, _ in ASSET_CELLS:
                    mass = float(row[f'{cell}_mass'])
                    kkt = float(row[f'{cell}_KKT'])
                    require(finite_nonnegative(mass) and finite_nonnegative(kkt),
                            f'invalid asset cell: {key}/{cell}')
                    acc[cell][0] += mass * kkt
                    acc[cell][1] += mass
        require(state_count == 3600, f'common row count changed: {key}')
        sources.append({'path': str(common_file), 'sha256': sha(common_file)})
        for cell, label in ASSET_CELLS:
            numerator, mass = acc[cell]
            cells.append({'entry_id': key, 'method': entry['method'], 'seed': entry['seed'],
                          'cell': cell, 'label': label, 'mass_total': mass,
                          'mass_weighted_KKT': numerator / mass if mass > 0 else None})
        phases = read_csv(phase_file)
        require(len(phases) == 24, f'dynamic phase count changed: {key}')
        sources.append({'path': str(phase_file), 'sha256': sha(phase_file)})
        for phase, expected in (('transition', 1000), ('post_burn', 3000)):
            subset = [r for r in phases if r['phase'] == phase]
            require(len(subset) == 12 and all(int(r['dates']) == expected
                    and r['failed'] == 'False' and finite_positive(r['KKT']) for r in subset),
                    f'incomplete phase: {key}/{phase}')
            phase_rows.append({'entry_id': key, 'method': entry['method'],
                               'seed': entry['seed'], 'phase': phase,
                               'paths': 12,
                               'median_path_KKT': statistics.median(float(r['KKT']) for r in subset)})
        for path_id in path_ids:
            file = directory / f'path_{path_id:02d}_own_dynamic.csv'
            count = 0
            with file.open(newline='', encoding='utf-8') as stream:
                for row in csv.DictReader(stream):
                    t = int(row['t'])
                    require(t == count, f'date sequence changed: {key}/{path_id}/{t}')
                    count += 1
                    if t % 100 == 0:
                        sample = {'entry_id': key, 'method': entry['method'],
                                  'seed': entry['seed'], 'path_id': path_id, 't': t}
                        for measure in ('K', 'zero_atom', 'q90'):
                            value = float(row[measure])
                            require(finite_nonnegative(value),
                                    f'bad path diagnostic: {key}/{path_id}/{measure}')
                            sample[measure] = value
                        traces.append(sample)
            require(count == 4000, f'path date count changed: {key}/{path_id}')
            sources.append({'path': str(file), 'sha256': sha(file)})
    require(len(cells) == 120 and len(phase_rows) == 30 and len(traces) == 7200,
            'detailed figure source dimensions changed')
    return cells, phase_rows, traces


def plot_common(groups, out):
    fig, ax = plt.subplots(figsize=(7.7, 4.8))
    group_ids = sorted({int(r['group_id']) for r in groups})
    require(len(group_ids) == 12, 'expected 12 final function groups')
    for method in ORDER:
        values = [[float(r['KKT']) for r in groups if r['method'] == method
                   and int(r['group_id']) == group] for group in group_ids]
        require(all(len(v) == 3 and all(finite_positive(x) for x in v) for v in values),
                f'group values incomplete: {method}')
        ax.plot(group_ids, [statistics.median(v) for v in values], 'o-',
                lw=1.7, ms=3.2, color=COLOR[method], label=LABEL[method])
        ax.fill_between(group_ids, [min(v) for v in values], [max(v) for v in values],
                        color=COLOR[method], alpha=.12, linewidth=0)
    ax.set_yscale('log')
    ax.set_xticks(group_ids)
    ax.set_xlabel('Independent sealed function group')
    ax.set_ylabel('Mean mass-weighted Euler–KKT violation')
    ax.set_title('Common exogenous states (300 parents per group)')
    ax.grid(alpha=.22, which='both')
    fig.subplots_adjust(left=.14, right=.74, bottom=.15, top=.88)
    ax.legend(frameon=False, ncol=1, fontsize=8,
              loc='center left', bbox_to_anchor=(1.02, .5))
    export(fig, out, 'figure1_common_group_KKT')


def plot_tail(groups, out):
    fig, ax = plt.subplots(figsize=(7.7, 4.8))
    group_ids = sorted({int(r['group_id']) for r in groups})
    for method in ORDER:
        values = [[float(r['P99']) for r in groups if r['method'] == method
                   and int(r['group_id']) == group] for group in group_ids]
        require(all(len(v) == 3 and all(finite_positive(x) for x in v) for v in values),
                f'P99 groups incomplete: {method}')
        ax.plot(group_ids, [statistics.median(v) for v in values], 'o-',
                lw=1.7, ms=3.2, color=COLOR[method], label=LABEL[method])
        ax.fill_between(group_ids, [min(v) for v in values], [max(v) for v in values],
                        color=COLOR[method], alpha=.12, linewidth=0)
    ax.set_yscale('log')
    ax.set_xticks(group_ids)
    ax.set_xlabel('Independent sealed function group')
    ax.set_ylabel('Mean within-state mass-weighted P99 KKT')
    ax.set_title('Tail of household equation violations on common states')
    ax.grid(alpha=.22, which='both')
    fig.subplots_adjust(left=.14, right=.74, bottom=.15, top=.88)
    ax.legend(frameon=False, ncol=1, fontsize=8,
              loc='center left', bbox_to_anchor=(1.02, .5))
    export(fig, out, 'figure5_common_group_P99')


def plot_asset_cells(cells, out):
    from matplotlib.colors import LogNorm
    matrix = np.full((len(ORDER), len(ASSET_CELLS)), np.nan)
    for i, method in enumerate(ORDER):
        for j, (cell, _) in enumerate(ASSET_CELLS):
            rows = [r for r in cells if r['method'] == method and r['cell'] == cell]
            require(len(rows) == 3, f'missing asset cell seeds: {method}/{cell}')
            values = [r['mass_weighted_KKT'] for r in rows if r['mass_weighted_KKT'] is not None]
            if len(values) == 3:
                matrix[i, j] = statistics.median(values)
    positive = matrix[np.isfinite(matrix) & (matrix > 0)]
    require(len(positive) > 0, 'no positive populated asset-cell residuals')
    fig, ax = plt.subplots(figsize=(10.5, 4.2))
    masked = np.ma.masked_invalid(np.where(matrix > 0, matrix, np.nan))
    image = ax.imshow(masked, aspect='auto', cmap='viridis',
                      norm=LogNorm(vmin=float(positive.min()), vmax=float(positive.max())))
    ax.set_yticks(range(len(ORDER)), [LABEL[m] for m in ORDER])
    ax.set_xticks(range(len(ASSET_CELLS)),
                  [f'{"Unemp." if cell.startswith("e0") else "Empl."}\n{cell.split("_")[1]}'
                   for cell, _ in ASSET_CELLS])
    ax.set_title('Equation violations by employment and wealth bin')
    for i in range(len(ORDER)):
        for j in range(len(ASSET_CELLS)):
            if math.isfinite(matrix[i, j]) and matrix[i, j] == 0:
                # LogNorm cannot represent an exact zero. Give it an explicit
                # neutral tile and label instead of leaving an invisible "0".
                ax.add_patch(Rectangle((j - .5, i - .5), 1, 1,
                                       facecolor='#e8e8e8', edgecolor='none'))
                ax.text(j, i, '0', ha='center', va='center', color='black', fontsize=7)
            elif math.isfinite(matrix[i, j]):
                ax.text(j, i, f'{matrix[i,j]:.2g}', ha='center', va='center',
                        color='white' if matrix[i, j] < np.sqrt(positive.min() * positive.max()) else 'black',
                        fontsize=7)
            else:
                ax.text(j, i, 'n/a', ha='center', va='center', fontsize=7)
    fig.colorbar(image, ax=ax, label='Mass-weighted KKT, median over 3 seeds')
    fig.subplots_adjust(left=.18, right=.89, bottom=.23, top=.88)
    export(fig, out, 'figure6_employment_wealth_KKT')


def plot_shape(shape, out):
    fig, (ax_r, ax_k) = plt.subplots(1, 2, figsize=(10.8, 4.7))
    for pos, method in enumerate(ORDER):
        rows = sorted((r for r in shape if r['method'] == method), key=lambda r: r['seed'])
        require(len(rows) == 3, f'shape seed count: {method}')
        for j, row in enumerate(rows):
            offset = (j - 1) * .10
            ax_r.scatter(pos + offset, row['response_MAE'], color=COLOR[method], s=25,
                         marker=('o', 's', '^')[j], zorder=3)
            ax_k.plot([pos + offset - .025, pos + offset + .025],
                      [row['KKT_raw'], row['KKT_sameK']], color=COLOR[method], lw=1)
            ax_k.scatter(pos + offset - .025, row['KKT_raw'], facecolor='white',
                         edgecolor=COLOR[method], s=24, zorder=3)
            ax_k.scatter(pos + offset + .025, row['KKT_sameK'], color=COLOR[method], s=24, zorder=3)
    labels = [LABEL[m] for m in ORDER]
    for ax in (ax_r, ax_k):
        ax.set_xticks(range(len(ORDER)), labels, rotation=18, ha='right')
        ax.grid(axis='y', alpha=.22, which='both')
    ax_r.set_yscale('log')
    ax_r.set_ylabel('Mass-weighted absolute saving response')
    ax_r.set_title('Response to shape at nearly fixed $K$')
    ax_k.set_yscale('log')
    ax_k.set_ylabel('Mass-weighted Euler–KKT violation')
    ax_k.set_title('Both pair endpoints: open = original')
    fig.subplots_adjust(left=.08, right=.98, bottom=.27, top=.80, wspace=.27)
    fig.legend(handles=[Line2D([0], [0], marker=m, color='gray', linestyle='None',
                               label=f'Seed {s}') for m, s in zip(('o', 's', '^'), SEEDS)],
               loc='upper center', ncol=3, frameon=False, bbox_to_anchor=(.5, .98))
    export(fig, out, 'figure2_sameK_response_endpoint_KKT')


def plot_paths(paths, out):
    fig, ax = plt.subplots(figsize=(8.0, 4.7))
    rng = np.random.default_rng(20261003)
    for i, method in enumerate(ORDER):
        rows = [r for r in paths if r['method'] == method]
        values = [float(r['post_burn_KKT']) for r in rows]
        require(len(values) == 36, f'expected 3 seeds x 12 own paths: {method}')
        ax.scatter(i + rng.uniform(-.19, .19, len(values)), values,
                   color=COLOR[method], alpha=.48, s=13, linewidths=0)
        ax.scatter(i, statistics.median(values), color=COLOR[method],
                   edgecolor='black', marker='D', s=55, zorder=5)
    ax.set_yscale('log')
    ax.set_xticks(range(len(ORDER)), [LABEL[m] for m in ORDER], rotation=15, ha='right')
    ax.set_ylabel('Post-burn mean Euler–KKT violation')
    ax.set_title('Own-policy paths (diamond: median across 36 paths)')
    ax.grid(axis='y', alpha=.22, which='both')
    fig.subplots_adjust(left=.15, right=.97, bottom=.23, top=.86)
    export(fig, out, 'figure3_own_path_KKT')


def plot_cost(summary, out):
    fig, axes = plt.subplots(1, 2, figsize=(9.8, 4.5))
    subsets = (ORDER, ORDER)
    keys = ('solve_seconds', 'final_evaluation_gpu_seconds')
    titles = ('GPU training and online simulation', 'Full final GPU evaluation')
    for ax, methods, key, title in zip(axes, subsets, keys, titles):
        for i, method in enumerate(methods):
            rows = sorted((r for r in summary if r['method'] == method),
                          key=lambda r: int(r['seed']))
            values = [float(r[key]) / 3600 for r in rows]
            require(len(values) == 3 and all(finite_positive(v) for v in values),
                    f'cost values invalid: {method}/{key}')
            for j, value in enumerate(values):
                ax.scatter(i + (j - 1) * .10, value, color=COLOR[method],
                           marker=('o', 's', '^')[j], s=28)
        ax.set_xticks(range(len(methods)), [LABEL[m] for m in methods],
                      rotation=25, ha='right')
        ax.set_ylabel('Wall-clock hours')
        ax.set_title(title)
        ax.grid(axis='y', alpha=.22)
    fig.subplots_adjust(left=.09, right=.98, bottom=.29, top=.78, wspace=.31)
    fig.legend(handles=[Line2D([0], [0], marker=m, color='#555555',
                               linestyle='None', label=f'Seed {s}')
                        for m, s in zip(('o', 's', '^'), SEEDS)],
               loc='upper center', ncol=3, frameon=False,
               bbox_to_anchor=(.5, .98))
    export(fig, out, 'figure4_measured_cost')


def plot_phases(phases, out):
    fig, grid = plt.subplots(2, 2, figsize=(8.6, 5.8), sharey=True)
    axes = grid.ravel()
    for ax, method in zip(axes, ORDER):
        for j, seed in enumerate(SEEDS):
            rows = [r for r in phases if r['method'] == method and int(r['seed']) == seed]
            require(len(rows) == 2, f'missing phases: {method}/{seed}')
            by_phase = {r['phase']: float(r['median_path_KKT']) for r in rows}
            ax.plot((0, 1), (by_phase['transition'], by_phase['post_burn']),
                    marker=('o', 's', '^')[j], color=COLOR[method],
                    linewidth=1.2, markersize=4.5, alpha=.55 + .2*j)
        ax.set_yscale('log')
        ax.set_xticks((0, 1), ('0–999', '1000–3999'))
        ax.set_title(LABEL[method], fontsize=9)
        ax.grid(axis='y', alpha=.22, which='both')
    grid[0, 0].set_ylabel('Median path mean KKT')
    grid[1, 0].set_ylabel('Median path mean KKT')
    fig.suptitle('Within-policy change from transition to post-burn phase', y=.97)
    fig.subplots_adjust(left=.12, right=.98, bottom=.10, top=.84, wspace=.28, hspace=.42)
    export(fig, out, 'figure7_dynamic_transition_postburn')


def plot_state_traces(traces, out):
    fig, axes = plt.subplots(3, 1, figsize=(8.3, 8.1), sharex=True)
    for ax, measure, label in zip(axes, ('K', 'zero_atom', 'q90'),
                                  ('Aggregate capital $K$', 'Zero-asset mass',
                                   'Wealth 90th percentile')):
        for method in ORDER:
            xs = list(range(0, 4000, 100))
            ys = []
            for t in xs:
                values = [r[measure] for r in traces if r['method'] == method and r['t'] == t]
                require(len(values) == 36, f'trace sample incomplete: {method}/{t}')
                ys.append(statistics.median(values))
            ax.plot(xs, ys, color=COLOR[method], lw=1.25, label=LABEL[method])
        ax.axvline(1000, color='#888888', linestyle='--', lw=.9)
        ax.set_ylabel(label)
        ax.grid(alpha=.2)
    axes[-1].set_xlabel('Date in common aggregate-shock paths')
    fig.suptitle('Distribution diagnostics along each method’s own paths', y=.98)
    fig.subplots_adjust(left=.13, right=.97, bottom=.08, top=.83, hspace=.26)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=4, fontsize=8,
               loc='upper center', bbox_to_anchor=(.5, .94))
    export(fig, out, 'figure8_dynamic_distribution_traces')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--audit', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    out = Path(args.output).resolve()
    if out.exists():
        raise FileExistsError(f'preserve existing figures: {out}')
    figure_style()
    registry, summary, groups, paths, shape, sources = load_checked(
        args.registry, args.attempt, args.audit)
    cells, phases, traces = load_detailed(registry, args.attempt, sources)
    out.mkdir(parents=True, exist_ok=False)
    save_csv(out / 'shape_summary_from_120_pairs.csv', shape)
    save_csv(out / 'employment_wealth_KKT_source.csv', cells)
    save_csv(out / 'dynamic_phase_source.csv', phases)
    save_csv(out / 'dynamic_trace_checkpoints.csv', traces)
    plot_common(groups, out)
    plot_shape(shape, out)
    plot_paths(paths, out)
    plot_cost(summary, out)
    plot_tail(groups, out)
    plot_asset_cells(cells, out)
    plot_phases(phases, out)
    plot_state_traces(traces, out)
    outputs = sorted(p for p in out.iterdir() if p.is_file())
    write_json(out / 'figure_audit.json', {
        'status': 'complete_from_frozen_final',
        'registry_sha256': EXPECTED_REGISTRY_SHA,
        'script_sha256': sha(Path(__file__)),
        'source_files': sources,
        'figure_files': [{'name': p.name, 'sha256': sha(p)} for p in outputs],
        'displayed_methods': list(ORDER),
        'notes': ['Four displayed methods are a post-evaluation presentation selection from the unchanged 15-policy frozen registry.',
                  'Seed ranges are descriptive, not confidence intervals.',
                  'Shape response is accompanied by both endpoint KKT values.',
                  'Own-policy paths visit different distributions.',
                  'GPU training/online simulation and full GPU evaluation are shown separately.',
                  'Polished derivative exports PNG/SVG/PDF/TIFF; original figure files remain untouched.'],
    })
    print(f'POLISHED FIGURES: 8 figures x PNG/SVG/PDF/TIFF, 4 displayed methods / 12 policies; '
          f'15 archived policies audited; output={out}', flush=True)


if __name__ == '__main__':
    main()
