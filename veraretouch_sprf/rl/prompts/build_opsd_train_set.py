#!/usr/bin/env python3
"""EPR-052：正式 OPSD 训练集——S2 快照 train 中按 sha1 规则抽 N 条 prompt，排除 held-out id 与 eval-only 键，
写 ms-swift jsonl（messages/images/teacher_prompt/key），y 图按 spec-5 几何重采样落盘；记录键表 sha256。
规则：sort train keys by sha1(f"{salt}:{key}") hex ascending, take first N。
"""
from __future__ import annotations
import argparse, hashlib, json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from PIL import Image
from veraretouch_sprf.data import cot_text as C
from veraretouch_sprf.models.vlm import q3vl_common as Q
from veraretouch_sprf.eval import guards
from veraretouch_sprf.rl.prompts.build_msswift_jsonl import teacher_prompt_text
from veraretouch_sprf import _paths as _P


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', default='/home/bc/data/runs/epr051_vlmsft/snap_sft2')
    ap.add_argument('--n', type=int, default=50000)
    ap.add_argument('--salt', default='epr052-opsd-train-v1')
    ap.add_argument('--heldout-ids', default=str(_P.STAGE0 / 'snapshot_newdata_v3.heldout_ids.json'))
    ap.add_argument('--out-dir', required=True)
    ap.add_argument('--path-map', default='/home/bc/data=/data')
    ap.add_argument('--workers', type=int, default=16)
    a = ap.parse_args()
    snap = Path(a.snapshot); out = Path(a.out_dir); (out / 'images').mkdir(parents=True, exist_ok=True)
    idx = json.load(open(snap / 'assets_index.json'))['index']
    train = json.load(open(snap / 'split_train_keys.json'))
    heldout = set(json.load(open(a.heldout_ids)))
    ev = guards.scan_eval_only()
    eval_keys = set(ev.get('keys') or []); eval_ids = set(ev.get('ids') or [])
    def _id(k): return k.split('|', 1)[0]
    pool = [k for k in train if _id(k) not in heldout and k not in eval_keys and _id(k) not in eval_ids]
    pick = sorted(pool, key=lambda k: hashlib.sha1(f'{a.salt}:{k}'.encode()).hexdigest())[: a.n]
    want = set(pick); recs = {}
    with open(snap / 'records.jsonl') as f:
        for ln in f:
            r = json.loads(ln)
            if r['key'] in want:
                recs[r['key']] = r
    assert len(recs) == len(pick), (len(recs), len(pick))
    src, dst = a.path_map.split('=', 1)

    def _one(k):
        r = recs[k]; instr, tier = C.instruction_for(r, k); gt = C.target_text(r)
        png_host = idx[k]['png']; out_png = out / 'images' / Path(png_host).name
        if not out_png.exists():
            img, g = Q.prepare_image_spec5(Image.open(png_host)); img.save(out_png); geom = (img.size[0], img.size[1], int(g.n_visual_tokens))
        else:
            im = Image.open(out_png); geom = (im.size[0], im.size[1], (im.size[0] // 32) * (im.size[1] // 32))
        return dict(messages=[dict(role='user', content='<image>' + instr)], images=[str(out_png).replace(src, dst, 1)],
                    teacher_prompt=teacher_prompt_text(instr, gt), key=k, instruction_tier=tier, gt_chars=len(gt), geom=geom)
    with ThreadPoolExecutor(a.workers) as ex:
        rows = list(ex.map(_one, pick))
    with open(out / 'opsd_train.jsonl', 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    keys_sha = hashlib.sha256(json.dumps(pick).encode()).hexdigest()
    (out / 'keys.json').write_text(json.dumps(pick))
    tiers = {}
    for r in rows: tiers[r['instruction_tier']] = tiers.get(r['instruction_tier'], 0) + 1
    meta = dict(n=len(rows), salt=a.salt, rule='sort train keys by sha1(salt:key) hex asc, first N',
                pool_after_exclusion=len(pool), train_universe=len(train), heldout_ids=len(heldout), eval_only_keys=len(eval_keys), eval_only_ids=len(eval_ids),
                keys_sha256=keys_sha, tiers=tiers, gt_chars_mean=sum(r['gt_chars'] for r in rows) / len(rows),
                geoms={}, jsonl=str(out / 'opsd_train.jsonl'))
    for r in rows:
        g = f"{r['geom'][0]}x{r['geom'][1]}"; meta['geoms'][g] = meta['geoms'].get(g, 0) + 1
    (out / 'META.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False)); print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()
