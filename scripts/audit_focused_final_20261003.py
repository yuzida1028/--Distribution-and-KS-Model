"""Read-only coverage, equation-error, own-path, and cost audit for frozen 15 runs.

Scientific summaries must be computed on RiverMind from the actual final files.
This script never launches or changes an evaluator and refuses to overwrite output.
"""
import argparse
import csv
import json
import math
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
sys.path.insert(0, str(ROOT / 'src'))
from focused_final_compare_20261002 import verify_registry
from jedc_distribution_v1 import sha, write_json


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(newline='', encoding='utf-8') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    if not rows:
        return
    names = list(dict.fromkeys(name for row in rows for name in row))
    with Path(path).open('x', newline='', encoding='utf-8') as stream:
        writer = csv.DictWriter(stream, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def finite(value):
    return value is not None and math.isfinite(float(value))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--registry', required=True)
    parser.add_argument('--attempt', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    registry_path = Path(args.registry).resolve()
    registry = verify_registry(registry_path)
    attempt = Path(args.attempt).resolve()
    output = Path(args.output).resolve()
    if output.exists():
        raise FileExistsError(output)
    if len(registry['runs']) != 15:
        raise ValueError('expected all 15 frozen entries')
    registry_hash = sha(registry_path)
    summary, groups, paths, issues = [], [], [], []
    for entry in registry['runs']:
        directory = attempt / entry['id']
        report_path = directory / 'report.json'
        base = dict(entry_id=entry['id'], method=entry['method'], seed=entry['seed'],
                    family=entry['family'])
        if not report_path.is_file():
            summary.append(dict(**base, status='missing_report'))
            issues.append(dict(**base, issue='final report missing', path=str(report_path)))
            continue
        report = read_json(report_path)
        failures_path = directory / 'failures.json'
        failures = read_json(failures_path) if failures_path.is_file() else None
        if (report.get('registry_sha256') != registry_hash
                or report.get('policy_sha256') != entry['policy_sha256']
                or report.get('entry_id') != entry['id'] or report.get('final_data_read') is not True):
            issues.append(dict(**base, issue='report provenance mismatch', path=str(report_path)))
        if failures is None or len(failures) != report.get('failure_count'):
            issues.append(dict(**base, issue='failure file/count mismatch', path=str(failures_path)))
        common_path = directory / 'common_states.csv'
        shape_path = directory / 'sameK_shape.csv'
        group_path = directory / 'common_original_group_summary.csv'
        common = read_csv(common_path) if common_path.is_file() else []
        shape = read_csv(shape_path) if shape_path.is_file() else []
        group_rows = read_csv(group_path) if group_path.is_file() else []
        if (len(common) != report.get('scored_states') or len(shape) != report.get('scored_shape_pairs')
                or len(group_rows) != len(report.get('groups', []))):
            issues.append(dict(**base, issue='raw CSV/report coverage mismatch', path=str(directory)))
        for row in group_rows:
            groups.append(dict(**base, group_id=int(row['group_id']), n=int(row['n']), KKT=float(row['KKT']),
                               P90=float(row['P90']), P99=float(row['P99'])))
        if len(group_rows) != 12 or any(int(row['n']) != 300 for row in group_rows):
            issues.append(dict(**base, issue='12 x 300 original group coverage missing', path=str(group_path)))
        dynamic_rows = read_csv(directory / 'dynamic_path_summary.csv') if (directory / 'dynamic_path_summary.csv').is_file() else []
        if len(dynamic_rows) != 24:
            issues.append(dict(**base, issue='expected 24 path-phase rows', path=str(directory)))
        post = []
        strict_paths = 0
        for path_id in sorted({int(row['path_id']) for row in dynamic_rows}):
            phases = {row['phase']: row for row in dynamic_rows if int(row['path_id']) == path_id}
            raw_path = directory / f'path_{path_id:02d}_own_dynamic.csv'
            dates = read_csv(raw_path) if raw_path.is_file() else []
            price_floor = any(float(row['price_capital_floor_active']) > 0
                              or float(row['price_labor_floor_active']) > 0 for row in dates)
            good = (len(dates) == 4000 and not price_floor and
                    set(phases) == {'transition', 'post_burn'} and
                    int(phases['transition']['dates']) == 1000 and
                    int(phases['post_burn']['dates']) == 3000 and
                    phases['transition']['failed'] == 'False' and phases['post_burn']['failed'] == 'False')
            strict_paths += int(good)
            value = float(phases['post_burn']['KKT']) if good and finite(phases['post_burn']['KKT']) else None
            if value is not None:
                post.append(value)
            paths.append(dict(**base, path_id=path_id, dates=len(dates), strict_complete=good,
                              price_floor=price_floor, post_burn_KKT=value))
        if strict_paths != report.get('complete_paths'):
            issues.append(dict(**base, issue='strict path count differs from report', path=str(directory)))
        training = read_json(Path(entry['run_dir']) / 'report.json')
        if entry['family'] == 'neural':
            solve_seconds = training.get('train_seconds_accumulated')
            solve_device = 'GPU training'
            peak = training.get('peak_gpu_allocated_bytes')
        elif entry['family'] == 'mmv':
            solve_seconds = training.get('solve_seconds_accumulated')
            solve_device = 'CPU solve'
            peak = None
        else:
            solve_seconds = training.get('effective_seconds')
            solve_device = 'GPU training and online simulation'
            peak = training.get('peak_allocated_bytes')
        complete = (len(common) == 3600 and len(shape) == 120 and strict_paths == 12
                    and report.get('failure_count') == 0 and report.get('status') == 'complete')
        summary.append(dict(**base, status=report.get('status'), full_coverage=complete,
            common_states=len(common), shape_pairs=len(shape), strict_paths=strict_paths,
            failures=report.get('failure_count'),
            common_raw_group_KKT=statistics.fmean(float(row['KKT']) for row in group_rows)
                if len(group_rows) == 12 else None,
            own_post_burn_KKT=statistics.fmean(post) if len(post) == 12 else None,
            solve_seconds=solve_seconds, solve_device=solve_device,
            solve_peak_gpu_bytes=peak, final_evaluation_gpu_seconds=report.get('seconds'),
            final_evaluation_peak_gpu_bytes=report.get('peak_allocated_bytes')))
    output.mkdir(parents=True, exist_ok=False)
    write_csv(output / 'entry_summary.csv', summary)
    write_csv(output / 'group_raw_KKT.csv', groups)
    write_csv(output / 'path_post_KKT.csv', paths)
    write_csv(output / 'issues.csv', issues)
    write_json(output / 'audit_report.json', dict(registry_sha256=registry_hash,
        entries_expected=15, entries_reported=sum(x['status'] != 'missing_report' for x in summary),
        entries_full_coverage=sum(bool(x.get('full_coverage')) for x in summary),
        issue_count=len(issues), final_scores_read=True,
        interpretation='common exogenous group KKT and own-policy path KKT are distinct domains; CPU and GPU solve costs are separate'))
    print(f"FINAL AUDIT: reports={sum(x['status'] != 'missing_report' for x in summary)}/15, "
          f"full={sum(bool(x.get('full_coverage')) for x in summary)}/15, issues={len(issues)}", flush=True)


if __name__ == '__main__':
    main()
