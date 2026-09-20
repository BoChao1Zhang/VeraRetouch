"""Read-only experiment audit; writes only a separate takeover report.

Does not launch evaluations, alter tables, or control existing processes.
The endpoint trainer already runs its own full-400 final evaluation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime
import hashlib
import json
import math
from pathlib import Path
import time

TRAIN = Path('/home/bc/data/runs/epr072_local_continuation_20260919/ENDPT2/full')
RUNS = Path('/home/bc/nfsvfs/bc/data/runs')


def read_json(path):
    return json.loads(path.read_text()) if path.is_file() else {}


def jsonl(path):
    if not path.is_file():
        return []
    lines = path.read_bytes().splitlines(keepends=True)
    rows = []
    for i, line in enumerate(lines):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            # A writer may currently be appending the final line. Never ignore
            # corrupt completed lines or an error in the middle of the file.
            if i != len(lines)-1 or line.endswith(b'\n'):
                raise
    return rows


def endpoint_status(root):
    run = root / 'run'
    config = read_json(root / 'config.json')
    rows = jsonl(run / 'steps.jsonl')
    evaluations = sorted((run / 'evaluations').glob('step*.json'))
    latest = read_json(evaluations[-1]) if evaluations else {}
    final = read_json(run / 'evaluations/step003200.json')
    full = read_json(run / 'global_full_final.json')
    matched = (config.get('arm') == 'ENDPT' and config.get('total_steps') == 3200
               and config.get('schedule_steps') == 6250)
    metrics = full.get('groups', {}).get('all', {}).get('top1', {})
    measured = all(metrics.get(k, {}).get('n') == 400 and
                   isinstance(metrics[k].get('mean'), (int, float)) and
                   math.isfinite(metrics[k]['mean'])
                   for k in ('l1', 'l2', 'psnr', 'de00'))
    local_ok = final.get('step') == 3200 and all(
        isinstance(final.get(k), (int, float)) and math.isfinite(final[k])
        for k in ('local_single_l1', 'local_rollout_l1'))
    ready = bool(matched and local_ok and full.get('n') == 400 and measured
                 and (run / 'final.pt').is_file() and rows
                 and rows[-1].get('step') == 3200)
    result = dict(ready=ready, matched_schedule=matched,
                  step=rows[-1].get('step') if rows else None,
                  latest_evaluation_step=latest.get('step'),
                  local_single_l1=latest.get('local_single_l1'),
                  local_rollout_l1=latest.get('local_rollout_l1'))
    if ready:
        # Final only: do not substitute best.pt, an intermediate evaluation,
        # or val50 for the fixed-budget full-400 control.
        result['final_table_cells'] = {
            'single_l1': final['local_single_l1'],
            'rollout_l1': final['local_rollout_l1'],
            'artedit_l1_x100': metrics['l1']['mean']}
        result['source_sha256'] = {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in
            (root / 'config.json', run / 'evaluations/step003200.json',
             run / 'global_full_final.json')}
    return result


def extension_status(root):
    result = {}
    for dataset in ('artedit100', 'mmart'):
        expected, observed, shards = set(), Counter(), []
        for folder in sorted(root.glob(dataset+'*')):
            manifest = read_json(folder / 'manifest.json')
            if not manifest:
                continue
            planned = set(manifest['sample_ids'])
            rows = jsonl(folder / 'rows.jsonl')
            keys = [r['key'] for r in rows]
            expected.update(planned)
            observed.update(keys)
            shards.append(dict(name=folder.name, planned=len(planned),
                               completed=len(set(keys)),
                               unexpected=sorted(set(keys)-planned)))
        required = ('closed', 'nilut500', 'cnilut500', 'vrmlp500', 'vrz500')
        coverage = {name: set() for name in required}
        for folder in sorted(root.glob(dataset+'*')):
            for row in jsonl(folder / 'rows.jsonl'):
                for name in required:
                    value = row.get(name+'_mae_baked')
                    if isinstance(value, (int, float)) and math.isfinite(value):
                        coverage[name].add(row['key'])
        result[dataset] = dict(
            planned_unique=len(expected), completed_unique=len(observed),
            duplicate_rows=sum(n-1 for n in observed.values()),
            ready=bool(expected) and all(coverage[k] == expected for k in required)
                  and not any(s['unexpected'] for s in shards),
            metric_coverage={k: len(v) for k, v in coverage.items()}, shards=shards,
            stability_note='Extension raw-parameter and LUT-space distances are not the original standardized-W delta_rel.')
    return result


def style_status(folder, methods, lut_ids, image_ids):
    rows = [r for p in sorted(folder.glob('rows_*.jsonl')) for r in jsonl(p)]
    result = {}
    for method in methods:
        chosen = [r for r in rows if r['method'] == method]
        keys = Counter((r['lut_id'], r['image_id']) for r in chosen)
        expected = {(lut, image) for lut in lut_ids for image in image_ids}
        valid = all(all(isinstance(r.get(k), (int, float)) and math.isfinite(r[k])
                        for k in ('psnr', 'ssim', 'de76', 'de00', 'lpips'))
                    for r in chosen)
        result[method] = dict(unique=len(keys), expected=len(expected),
                              duplicates=sum(n-1 for n in keys.values()),
                              ready=set(keys) == expected and valid
                                    and all(n == 1 for n in keys.values()))
    return result


def snapshot():
    lut_ids = ['LUT01', 'LUT02', 'LUT03', 'LUT04', 'LUT05', 'LUT08', 'LUT10']
    methods = ['ours_hald128', 'ours_hald128_r3', 'ours_hald33', 'lut33_roundtrip']
    return dict(time=datetime.now().astimezone().isoformat(),
                endpoint=endpoint_status(TRAIN),
                extension=extension_status(RUNS/'epr073_closedform_20260920/ext'),
                style_eval3=style_status(RUNS/'epr079_style_transfer_20260920/eval3',
                                        methods, lut_ids,
                                        [f'{i:03d}' for i in range(1, 101)]))


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--out', type=Path, default=RUNS/'experiment_takeover_20260920')
    ap.add_argument('--watch-hours', type=float, default=0)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    deadline = time.monotonic() + args.watch_hours * 3600
    previous = None
    while True:
        report = snapshot()
        temp = args.out/'status.json.partial'
        temp.write_text(json.dumps(report, indent=2)+'\n')
        temp.replace(args.out/'status.json')
        signature = {k: v for k, v in report.items() if k != 'time'}
        if signature != previous:
            print(json.dumps(report), flush=True)
            previous = signature
        done = (report['endpoint']['ready'] and
                all(v['ready'] for v in report['extension'].values()) and
                all(v['ready'] for v in report['style_eval3'].values()))
        if done or time.monotonic() >= deadline:
            break
        time.sleep(min(60, max(0, deadline-time.monotonic())))


if __name__ == '__main__':
    main()
