"""EPR-061-D2 step 4: the two training manifests.

``stage1_style.json``  the existing 30k style cache (plan + journals + EPR-059 style codes)
                       plus the reserved slots D1's ``style_extra`` / ``local`` caches drop into.
``stage2_chain.json``  the 70k degradation chains: per-record journal location, the six CoT
                       segment spans, the per-stage code keys, the chain-rendering recipe and
                       the val128 / heldout1464 evaluation splits.

Both carry their own ``sha256`` (over every field but ``sha256``) and the statistics the
task card asks for.  Nothing outside ``OUT/manifests`` is written.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, '/home/bc/VeraRetouch')
from tools.epr061_cache import chain_common as C

STYLE_CODE_TAR = C.STYLE_CODES / 'codes_style_N64_r1e-2.tar'
STYLE_CODE_INDEX = C.STYLE_CODES / 'codes_style_N64_r1e-2.index.sqlite'


def file_sha256(path):
    import hashlib
    digest = hashlib.sha256()
    with open(path, 'rb') as stream:
        for block in iter(lambda: stream.read(1 << 22), b''):
            digest.update(block)
    return digest.hexdigest()


def stamp(payload):
    payload['sha256'] = C.digest_json({k: v for k, v in payload.items() if k != 'sha256'})
    return payload


def distribution(values):
    values = np.asarray(values, dtype=np.float64)
    if not values.size:
        return None
    return dict(n=int(values.size), mean=float(values.mean()), p05=float(np.quantile(values, .05)),
                p50=float(np.median(values)), p95=float(np.quantile(values, .95)),
                min=float(values.min()), max=float(values.max()))


# --------------------------------------------------------------------------- #
def stage1_style(out, hash_codes=True):
    from veraretouch_sprf.stylelocal.common import plan_for
    plan = plan_for(C.STYLE_ROOT, 'style_train')
    journals = sorted((C.STYLE_ROOT / 'journals' / 'style_train').glob('part-*.json'))
    lut_ids = sorted({row['lut_id'] for row in plan['records']})
    payload = dict(
        schema='epr061_stage1_style_v1', task='EPR-061-D2', built=time.time(),
        note=('列表顺序即拼接顺序；源内顺序 = 该缓存自己的 plan seq，不打乱。'
              'style_extra / local 两个源是 D1 的产物占位，D1 完成后由主 agent 填路径并合并。'),
        geometry=str(C.GEOMETRY), ridge=C.RIDGE, dims=260,
        sources=[
            dict(name='style30k', root=str(C.STYLE_ROOT), split='style_train', limit=0,
                 plan=str(C.STYLE_ROOT / 'style_train_plan.json'), plan_sha256=plan['sha256'],
                 journals=str(C.STYLE_ROOT / 'journals' / 'style_train'),
                 journal_parts=len(journals), part_size=plan['part_size'],
                 stage_token_id=plan['stage_token_id'],
                 codes=str(STYLE_CODE_TAR), codes_index=str(STYLE_CODE_INDEX),
                 codes_sha256=file_sha256(STYLE_CODE_TAR) if hash_codes else None,
                 n=len(plan['records']), distinct_lut_ids=len(lut_ids), ready=True),
            dict(name='style_extra', root='<FILL: D1 style_extra cache root>', split='style_train',
                 limit=0, codes='<FILL: D1 style_extra codes_*.tar (geometry_N64.pt)>',
                 ready=False, note='D1 交付后由主 agent 填入'),
            dict(name='local', root='<FILL: D1 local cache root>', split='<FILL: local train split>',
                 limit=0, codes='<FILL: D1 local codes_*.tar (geometry_N64.pt)>',
                 ready=False, note='D1 交付后由主 agent 填入'),
        ],
        statistics=dict(style30k=dict(n=len(plan['records']), distinct_lut_ids=len(lut_ids),
                                      instructions=len({r['instruction_sha256'] for r in plan['records']}),
                                      source_images=len({r['source_image_id'] for r in plan['records']}),
                                      majors=len({r['major'] for r in plan['records']}))))
    return stamp(payload)


# --------------------------------------------------------------------------- #
def chain_records(split, records, stage_ids, part_chains, families, held):
    """Journal location + the six generated segment spans, per chain."""
    from veraretouch_sprf.data.q3vl_text import spans_from_generated
    rows, part_index, journal = [], -1, None
    valid_count = 0
    for record in records:
        part = record['seq'] // part_chains
        if part != part_index:
            journal = json.loads((C.JOURNAL_ROOT / split / f'part-{part:06d}.json').read_text())
            part_index = part
        generated = journal['rows'][record['seq'] % part_chains]
        if (generated['key'], generated['seq']) != (record['key'], record['seq']):
            raise ValueError(f'Journal row order differs: {record["key"]}')
        ids = generated['token_ids']
        markers = [x for x in ids if x in stage_ids]
        spans, missing = spans_from_generated(None, ids, stage_ids)
        valid = (markers == list(stage_ids) and not missing
                 and generated['finish_reason'] != 'length'
                 and ids[-1] == 151645 and ids.count(151645) == 1)
        valid_count += int(valid)
        rows.append(dict(key=record['key'], seq=record['seq'], part=part,
                         row=record['seq'] % part_chains, n_tokens=len(ids),
                         spans=[list(s) if s else None for s in spans],
                         format_valid=bool(valid),
                         family_heldout=bool(held.intersection(families.get(record['key'], ())))))
    return rows, valid_count


def stage2_chain(out, name, hash_codes=True):
    """2026-09-17 用户裁定后的口径：trainfull 全量 + heldout1464 拼成一条训练流
    （heldout 排在 trainfull 之后，逐条带 ``split_origin``），val128 单列作次要监控。
    ``family_heldout`` 只标不过滤。核心评测走 ArtEdit-Bench，本 manifest 不为此改动。"""
    from veraretouch_sprf.langif.family_split import segment_families
    tar = Path(out) / f'{name}.tar'
    index = Path(out) / f'{name}.index.sqlite'
    C.check_cache_semantics(index, C.ABSOLUTE_CODE_SEMANTICS)
    db = sqlite3.connect(f'file:{index}?mode=ro', uri=True)
    held = set(json.loads(C.FAMILY_SPLIT.read_text())['heldout_families'])
    splits, records_out = {}, {}
    all_keys = []
    for split in C.SPLITS:
        plan, records = C.plan_records(split)
        have = {r[0] for r in db.execute('SELECT DISTINCT chain_key FROM members WHERE split=?', (split,))}
        if not have:
            continue
        records = [r for r in records if r['key'] in have]
        all_keys += [r['key'] for r in records]
        splits[split] = dict(plan=str(C.EXPORT / f'plan_{split}.json'), plan_sha256=plan['plan_sha256'],
                             stage_ids=plan['stage_ids'], part_chains=plan['part_chains'],
                             journals=str(C.JOURNAL_ROOT / split), n=len(records),
                             n_plan_records=len(plan['records']),
                             role='train' if split in C.TRAIN_SEQUENCE else 'eval_secondary',
                             selection='plan seq order, every record whose codes solved')
        records_out[split] = records
    families = segment_families(all_keys)

    per_split_rows, statistics = {}, {}
    for split, meta in splits.items():
        rows, valid = chain_records(split, records_out[split], meta['stage_ids'],
                                    meta['part_chains'], families, held)
        for row in rows:
            row['split_origin'] = split
        per_split_rows[split] = rows
        statistics[split] = split_statistics(db, split, rows)

    sequence = dict(
        train=[row for split in C.TRAIN_SEQUENCE for row in per_split_rows.get(split, [])],
        val=[row for split in C.EVAL_SPLITS for row in per_split_rows.get(split, [])])
    for name_, rows in sequence.items():
        statistics[f'{name_}_sequence'] = dict(
            n=len(rows), order=list(C.TRAIN_SEQUENCE) if name_ == 'train' else list(C.EVAL_SPLITS),
            by_origin={split: sum(r['split_origin'] == split for r in rows)
                       for split in dict.fromkeys(r['split_origin'] for r in rows)},
            format_valid=sum(r['format_valid'] for r in rows),
            family_heldout=sum(r['family_heldout'] for r in rows))

    ra = json.loads((C.BK_RUN / 'run_args.json').read_text())['config']
    meta = dict(db.execute('SELECT k, v FROM meta').fetchall())
    build_law = json.loads(meta['build_law'])
    payload = dict(
        schema='epr061_stage2_chain_v2', task='EPR-061-D2', built=time.time(),
        ruling=('2026-09-17 用户裁定：heldout1464 不再留作评测，全部链进阶段二训练'
                '（trainfull 全量 + heldout，heldout 排在 trainfull 之后）；'
                'val128 单独列出作次要监控；family_heldout 不过滤，只保留标记列；'
                '核心评测改为 ArtEdit-Bench（另一 agent 的 harness），本 manifest 不为此改动。'),
        geometry=str(C.GEOMETRY), ridge=C.RIDGE, dims=260,
        codes=dict(tar=str(tar), index=str(index),
                   code_semantics=C.ABSOLUTE_CODE_SEMANTICS,
                   support_semantics='recorded strength-scaled beta',
                   executor='z + beta * (phi(z) @ W.T - z)',
                   tar_sha256=file_sha256(tar) if hash_codes else None,
                   index_sha256=file_sha256(index) if hash_codes else None,
                   member='<split_origin>/<seq:06d>_<slot>.f32',
                   key='<chain_key>|<slot>',
                   payload='(2, 3, 260) float32 -- row 0 allpix beta^2-weighted, row 1 uniform 4096',
                   slots=list(C.COLOR_SLOTS),
                   slot_rule='text move j -> slot 5-j; slot 0 = subject/where (no colour code)'),
        chain_definition=dict(
            stage_order=list(C.STAGES), n_steps=6,
            asset_index=str(C.REFIT_ROOT / 'lean_index.json'),
            lut_bank=ra['data']['lut_bank_dir'],
            build_law=build_law,
            recipe=[
                "row = journal at lean_index[key]['dir']/pairs.jsonl offset off length len",
                "target = assets/<row['id']>.src.png (assets/ first, archive/ indexed tar otherwise)",
                "beta = train_stage0.alpha_fields(row, target).reshape(6, -1)   # already * calib.s",
                "states, actions = stage_targets.build_intermediates(target.reshape(-1,3), beta, row['luts'], lut_bank)",
                "stage k input z_k = states[k+1]; target prev_k = states[k]; prev_k = z_k + beta_k * action_k",
                "final state states[6] must match assets/<entry['asset']> within 1.1/255",
            ]),
        codes_meta={k: v for k, v in meta.items() if k not in ('subject_audit',)},
        splits=splits, train_order=list(C.TRAIN_SEQUENCE), eval_splits=list(C.EVAL_SPLITS),
        statistics=statistics, records=sequence)
    db.close()
    return stamp(payload)


def split_statistics(db, split, rows):
    coverage = [r[0] for r in db.execute('SELECT beta_coverage FROM members WHERE split=?', (split,))]
    lut_ids = {r[0] for r in db.execute('SELECT DISTINCT lut_id FROM members WHERE split=?', (split,))}
    return dict(
        n=len(rows), format_valid=sum(r['format_valid'] for r in rows),
        format_invalid=sum(not r['format_valid'] for r in rows),
        family_heldout=sum(r['family_heldout'] for r in rows),
        distinct_lut_ids=len(lut_ids),
        stage_members=db.execute('SELECT count(*) FROM members WHERE split=?', (split,)).fetchone()[0],
        empty_stages=db.execute('SELECT sum(empty_stage) FROM members WHERE split=?', (split,)).fetchone()[0],
        beta_coverage=distribution(coverage),
        beta_coverage_by_slot={str(slot): distribution([r[0] for r in db.execute(
            'SELECT beta_coverage FROM members WHERE split=? AND slot=?', (split, slot))])
            for slot in C.COLOR_SLOTS},
        strength=distribution([s for _, s in db.execute(
            'SELECT chain_key, strength FROM members WHERE split=? AND slot=1', (split,))]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--name', default=C.ABSOLUTE_CACHE_NAME)
    parser.add_argument('--out', default=str(C.OUT))
    parser.add_argument('--which', default='both', choices=('both', 'stage1', 'stage2'))
    parser.add_argument('--no-hash', action='store_true')
    args = parser.parse_args()
    destination = Path(args.out) / 'manifests'
    began = time.monotonic()
    written = {}
    if args.which in ('both', 'stage1'):
        payload = stage1_style(args.out, hash_codes=not args.no_hash)
        C.atomic_json(destination / 'stage1_style.json', payload)
        written['stage1_style.json'] = payload['sha256']
    if args.which in ('both', 'stage2'):
        payload = stage2_chain(args.out, args.name, hash_codes=not args.no_hash)
        C.atomic_json(destination / 'stage2_chain.json', payload)
        written['stage2_chain.json'] = payload['sha256']
        print(json.dumps(dict(phase='stage2_statistics', statistics=payload['statistics'])), flush=True)
    print(json.dumps(dict(phase='manifests', written=written,
                          seconds=time.monotonic() - began)), flush=True)


if __name__ == '__main__':
    main()
