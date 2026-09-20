"""Score independent six-stage archives with the unchanged full-table scorers.

Registers a method in memory only; never mutates shared benchmark registries or
single-edit result boards. Prediction files are verified against shard hashes.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import tarfile
import time

ROOT = Path('/home/bc/nfsvfs/bc/data/runs/epr072_sixstage_compare_20260921')
NO_COT = Path('/home/bc/nfsvfs/bc/data/runs/epr072_sixstage_public_20260920/full')
COT = ROOT/'with_cot'
LOCAL = Path('/home/bc/data/runs/epr072_sixstage_compare_20260921/predictions')


def materialize(mode):
    source = (NO_COT if mode == 'nocot' else COT)/'artedit'
    summary = json.loads((source/'summary.json').read_text())
    if not summary['complete'] or summary['n'] != 400:
        raise ValueError('Refusing to score incomplete ArtEdit render')
    target = LOCAL/mode
    target.mkdir(parents=True, exist_ok=True)
    seen = set()
    for part in sorted(source.glob('part_*.json')):
        data = json.loads(part.read_text())
        archive = part.with_suffix('.tar')
        h = hashlib.sha256()
        with archive.open('rb') as f:
            for chunk in iter(lambda:f.read(1<<20), b''):
                h.update(chunk)
        assert h.hexdigest() == data['tar_sha256']
        with tarfile.open(archive) as pack:
            for row in data['rows']:
                sid = row['sample_id']
                assert Path(sid).name == sid and sid not in seen
                seen.add(sid)
                dest = target/(sid+'.png')
                if dest.exists() and hashlib.sha256(dest.read_bytes()).hexdigest() == row['prediction_sha256']:
                    continue
                blob = pack.extractfile(sid+'.png').read()
                assert hashlib.sha256(blob).hexdigest() == row['prediction_sha256']
                temp = dest.with_suffix('.partial')
                temp.write_bytes(blob)
                temp.replace(dest)
    if len(seen) != 400:
        raise ValueError('Archive coverage differs from summary')
    return target, seen


def scored_ids(path, method, key):
    import math
    if not path.exists():
        return set()
    rows = [json.loads(s) for s in path.read_text().splitlines() if s.strip()]
    return {r['sample_id'] for r in rows if r['method'] == method and
            isinstance(r.get(key), (int, float)) and math.isfinite(r[key])}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--mode', choices=['cot', 'nocot'], required=True)
    ap.add_argument('--lane', choices=['materialize', 'viescore', 'qalign', 'deqa', 'artimuse'], required=True)
    ap.add_argument('--device', default='cuda:0')
    args = ap.parse_args()
    pred, ids = materialize(args.mode)
    if args.lane == 'materialize':
        print(json.dumps(dict(mode=args.mode, n=len(ids), predictions=str(pred))))
        return
    method = 'ours_local_pix3200_six_'+args.mode
    out = ROOT/'scores'/args.mode
    out.mkdir(parents=True, exist_ok=True)
    if args.lane == 'viescore':
        from q3vl.whatb.pubbench import epr038b_metrics as M
        M.METHODS[method] = dict(dir=pred, instruction_field='instruction')
        from q3vl.whatb.pubbench import epr038b_viescore as V
        V.JUDGE_METHODS = [method]
        argv = ['--method', method, '--out-root', str(out), '--judge-model', 'gpt-5.6-terra',
                '--attempts', '6', '--judge-long-edge', '1024']
        path, key, call = out/'rows_viescore_en.jsonl', 'sc', lambda: V.main(argv)
    else:
        import torch
        torch.cuda.set_per_process_memory_fraction(.22)
        from q3vl.whatb.pubbench import epr038b_iaa as I
        I.METHODS[method] = dict(dir=pred)
        argv = ['--method', method, '--scorer', args.lane, '--out-root', str(out),
                '--device', args.device, '--batch', '1' if args.lane == 'artimuse' else '8']
        path, key, call = out/f'rows_iaa_{args.lane}_en.jsonl', 'score', lambda: I.main(argv)
    for attempt in range(4):
        if scored_ids(path, method, key) == ids:
            print(f'SCORE_COMPLETE {args.mode} {args.lane} n=400', flush=True)
            return
        call()
        if scored_ids(path, method, key) != ids:
            time.sleep(60)
    if scored_ids(path, method, key) != ids:
        raise RuntimeError('Scorer incomplete after four resumable passes')
    print(f'SCORE_COMPLETE {args.mode} {args.lane} n=400', flush=True)


if __name__ == '__main__':
    main()
