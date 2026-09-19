"""EPR-061-D2 step 2/3/5: cross-check the full-image codes, tabulate the capacity column,
and intersect the chain LUT ids with the 112 style_lut_unseen LUTs.

Cross-check: the 16,384-pixel process cache (``process20k_glut033_p16384_gpu1_v1``) stores
``z/prev/action/beta`` per stage for the chains it selected.  Solving the same closed form
on those 16,384 pixels gives ``W_16k``; this reports ``|W_16k - W_full|_F / |W_full|_F`` and
the rendered M of both codes on the cached pixels.  Nothing is written back into any cache.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import tarfile
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, '/home/bc/VeraRetouch')
from tools.epr061_cache import chain_common as C


def quantiles(values):
    values = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if not values.size:
        return None
    return dict(n=int(values.size), mean=float(values.mean()), p50=float(np.median(values)),
                p90=float(np.quantile(values, 0.9)), p99=float(np.quantile(values, 0.99)),
                max=float(values.max()))


def read_codes(tar_path, index_path):
    """-> (open file handle, {key -> (offset, size)}) for the D2 code tar."""
    db = sqlite3.connect(f'file:{index_path}?mode=ro', uri=True)
    rows = db.execute('SELECT key, offset, size FROM members').fetchall()
    db.close()
    return open(tar_path, 'rb'), {key: (offset, size) for key, offset, size in rows}


def member(handle, locate, key):
    offset, size = locate[key]
    handle.seek(offset)
    blob = handle.read(size)
    if len(blob) != size:
        raise OSError(f'short read for {key}')
    return np.frombuffer(blob, dtype=np.float32).reshape(2, 3, 260)


def process_rows(root, plan, split, wanted_keys, limit):
    """Stream decoded process-cache rows whose key is in ``wanted_keys`` (fixed pack order)."""
    from veraretouch_sprf.data import process_cache
    directory = Path(root) / 'data' / split
    parts = sorted(int(p.stem.split('-')[1]) for p in directory.glob('part-*.json'))
    produced = 0
    for part in parts:
        for row in process_cache.read_pack(directory, plan, split, part):
            if row['key'] not in wanted_keys:
                continue
            yield row
            produced += 1
            if produced >= limit:
                return
        if produced >= limit:
            return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=C.ABSOLUTE_CACHE_NAME)
    parser.add_argument('--out', default=str(C.OUT))
    parser.add_argument('--split', default='trainfull')
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--report', default='')
    args = parser.parse_args()
    out = Path(args.out)
    C.check_cache_semantics(out / f'{args.name}.index.sqlite', C.ABSOLUTE_CODE_SEMANTICS)
    began = time.monotonic()
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    torch.backends.cuda.matmul.allow_tf32 = False
    glut = C.Glut(C.GEOMETRY, device)

    handle, locate = read_codes(out / f'{args.name}.tar', out / f'{args.name}.index.sqlite')
    db = sqlite3.connect(f"file:{out / f'{args.name}.index.sqlite'}?mode=ro", uri=True)

    # ---- 3: capacity column straight off the index ------------------------------------
    capacity = {}
    for split, slot, *values in db.execute(
            'SELECT split, slot, count(*), avg(beta_coverage) FROM members GROUP BY split, slot'):
        capacity.setdefault(split, {})[slot] = dict(n=values[0], beta_coverage_mean=values[1])
    for column in ('m_all_mean', 'm_active_mean', 'm_uniform_all_mean', 'm_uniform_active_mean',
                   'm_identity_all_mean', 'm_identity_active_mean', 'code_fro',
                   'code_fro_uniform', 'splithalf_rel'):
        for split, slot in [(s, k) for s, slots in capacity.items() for k in slots]:
            values = [r[0] for r in db.execute(
                f'SELECT {column} FROM members WHERE split=? AND slot=?', (split, slot))]
            capacity[split][slot][column] = quantiles(values)
    empty = dict(db.execute('SELECT split, sum(empty_stage) FROM members GROUP BY split').fetchall())
    failures = [dict(chain_key=a, split=b, seq=c, reason=d)
                for a, b, c, d in db.execute('SELECT * FROM failures')]

    # ---- 5: LUT intersection with style_lut_unseen -------------------------------------
    from veraretouch_sprf.stylelocal.common import plan_for
    unseen = {row['lut_id'] for row in plan_for(C.STYLE_ROOT, 'style_lut_unseen')['records']}
    style_train = {row['lut_id'] for row in plan_for(C.STYLE_ROOT, 'style_train')['records']}
    chain_luts = {r[0] for r in db.execute('SELECT DISTINCT lut_id FROM members')}
    per_split = {}
    for split, in db.execute('SELECT DISTINCT split FROM members'):
        ids = {r[0] for r in db.execute('SELECT DISTINCT lut_id FROM members WHERE split=?', (split,))}
        hits = db.execute('SELECT count(*) FROM members WHERE split=? AND lut_id IN (%s)'
                          % ','.join('?' * len(unseen)), (split, *sorted(unseen))).fetchone()[0]
        per_split[split] = dict(distinct_lut_ids=len(ids), unseen_lut_ids=len(ids & unseen),
                                stage_members_on_unseen_luts=hits)
    lut_report = dict(style_lut_unseen=len(unseen), style_train=len(style_train),
                      chain_distinct=len(chain_luts),
                      chain_cap_unseen=len(chain_luts & unseen),
                      chain_cap_style_train=len(chain_luts & style_train),
                      unseen_also_in_style_train=len(unseen & style_train),
                      per_split=per_split,
                      unseen_ids_seen_in_chains=sorted(chain_luts & unseen))

    # ---- 2: full-image code vs 16,384-pixel process-cache code -------------------------
    plan = json.loads(C.PROCESS_PLAN.read_text())
    keys_in_codes = {k.rsplit('|', 1)[0] for k in locate}
    wanted = {r['key'] for r in plan['records'][args.split] if r['key'] in keys_in_codes}
    rows = []
    for row in process_rows(C.PROCESS_ROOT, plan, args.split, wanted, args.limit):
        key = row['key']
        z = row['z'].to(device).float()
        prev = row['prev'].to(device).float()
        action = row['action'].to(device).float()
        beta = row['beta'].to(device).float()
        slots = torch.tensor(C.COLOR_SLOTS, device=device)
        z, prev, action, beta = z[slots], prev[slots], action[slots], beta[slots]
        identity_gap = float((prev - (z + beta.unsqueeze(-1) * action)).abs().max())
        phi = glut.features(z)
        gram, rhs = C.absolute_statistics(phi, z, prev, beta)
        cached, empty_slots = glut.solve(gram, rhs)
        identity_code = torch.zeros_like(cached)
        identity_code[..., :, -4:-1] = torch.eye(3, device=device, dtype=cached.dtype)
        cached = torch.where(empty_slots[..., None, None], identity_code, cached)
        full = torch.from_numpy(np.stack([member(handle, locate, f'{key}|{slot}')
                                          for slot in C.COLOR_SLOTS])).to(device)
        full_codes, uniform_codes = full[:, 0], full[:, 1]
        scale = full_codes.norm(dim=(-1, -2)).clamp_min(1e-12)
        relative = ((cached - full_codes).norm(dim=(-1, -2)) / scale).cpu().numpy()
        support = (beta > 0).sum(-1).clamp_min(1)
        measures = {}
        for name, codes in (('full', full_codes), ('cached16k', cached),
                            ('uniform', uniform_codes), ('identity', None)):
            predicted = z if codes is None else z + beta.unsqueeze(-1) * (phi @ codes.transpose(-1, -2) - z)
            error = (predicted - prev).abs().amax(-1) * 255.0
            measures[name] = ((error.mean(-1)).cpu().numpy(),
                              ((error * (beta > 0)).sum(-1) / support).cpu().numpy())
        rows.append(dict(key=key, identity_gap=identity_gap,
                         relative={str(s): float(relative[i]) for i, s in enumerate(C.COLOR_SLOTS)},
                         m={n: {str(s): [float(a[i]), float(b[i])] for i, s in enumerate(C.COLOR_SLOTS)}
                            for n, (a, b) in measures.items()}))
    handle.close()
    compare = dict(n=len(rows), pixels=16384, source=str(C.PROCESS_ROOT), split=args.split,
                   plan=str(C.PROCESS_PLAN), plan_sha256=plan['sha256'],
                   identity_gap=quantiles([r['identity_gap'] for r in rows]))
    for slot in C.COLOR_SLOTS:
        compare[f'slot{slot}'] = dict(
            relative_code_diff=quantiles([r['relative'][str(slot)] for r in rows]),
            **{f'm_{name}_{scope}': quantiles([r['m'][name][str(slot)][position] for r in rows])
               for name in ('full', 'cached16k', 'uniform', 'identity')
               for position, scope in ((0, 'all16k'), (1, 'active16k'))})

    report = dict(task='EPR-061-D2 audit', name=args.name, geometry=str(C.GEOMETRY),
                  ridge=C.RIDGE, seconds=time.monotonic() - began,
                  capacity=capacity, empty_stage_members=empty,
                  failures=dict(n=len(failures), rows=failures[:50]),
                  lut_intersection=lut_report, process_cache_compare=compare)
    C.atomic_json(out / (args.report or f'{args.name}.audit.json'), report)
    print(json.dumps(dict(phase='audit_done', n_compare=len(rows),
                          seconds=time.monotonic() - began)), flush=True)


if __name__ == '__main__':
    main()
