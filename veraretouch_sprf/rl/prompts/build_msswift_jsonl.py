#!/usr/bin/env python3
"""EPR-052 ENG-2：从 snap_sft2 抽 N 条 train 记录，写 ms-swift OPSD jsonl。

每行：messages=[{user: "<image>" + 逐样本指令}]（无 system；<image> 在指令前——SURVEY v2 §F.1），
images=[y 图路径]，teacher_prompt = "<image>" + 指令 + 参考段 + GT 六段 CoT + 过渡句（格式按
examples/train/rlhf/opsd/opsd_plugin.py），附加列 key / instruction_tier。
路径可用 --path-map 把宿主前缀换成容器前缀（默认 /home/bc/data -> /data）。
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from veraretouch_sprf.data import cot_text as C

TRANSITION_PROMPT = ('After understanding the reference grade and the rationale behind each move, '
                     'now articulate your own six-move reasoning for this photograph.')
REFERENCE_HEADER = 'Here is a reference grade for this photograph:'


def teacher_prompt_text(instr: str, gt: str) -> str:
    return f'<image>{instr}\n\n{REFERENCE_HEADER}\n{gt}\n\n{TRANSITION_PROMPT}'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--snapshot', default='/home/bc/data/runs/epr051_vlmsft/snap_sft2')
    ap.add_argument('--n', type=int, default=8)
    ap.add_argument('--salt', default='epr052-smoke-v1')
    ap.add_argument('--out', required=True)
    ap.add_argument('--path-map', default='/home/bc/data=/data')
    ap.add_argument('--spec5-dir', default='', help='若给出：把 y 图按 spec-5（短边 512、对齐 32、bicubic、EXIF）重采样后写到此目录并引用之（与 SFT 几何一致）')
    a = ap.parse_args()
    snap = Path(a.snapshot)
    idx = json.load(open(snap / 'assets_index.json'))['index']
    train = json.load(open(snap / 'split_train_keys.json'))
    pick = sorted(train, key=lambda k: hashlib.sha1(f'{a.salt}:{k}'.encode()).hexdigest())[: a.n]
    want = set(pick)
    recs = {}
    with open(snap / 'records.jsonl') as f:
        for ln in f:
            r = json.loads(ln)
            if r['key'] in want:
                recs[r['key']] = r
    assert len(recs) == len(pick), (len(recs), len(pick))
    src, dst = a.path_map.split('=', 1)
    rows = []
    for k in pick:
        r = recs[k]
        instr, tier = C.instruction_for(r, k)
        gt = C.target_text(r)
        png_host = idx[k]['png']
        geom = None
        if a.spec5_dir:
            from PIL import Image
            from veraretouch_sprf.models.vlm import q3vl_common as Q
            img, g = Q.prepare_image_spec5(Image.open(png_host))
            out_png = Path(a.spec5_dir) / Path(png_host).name
            out_png.parent.mkdir(parents=True, exist_ok=True)
            img.save(out_png)
            png_host = str(out_png)
            geom = dict(w=img.size[0], h=img.size[1], n_visual_tokens=int(g.n_visual_tokens))
        png = png_host.replace(src, dst, 1)
        rows.append(dict(messages=[dict(role='user', content='<image>' + instr)],
                         images=[png], teacher_prompt=teacher_prompt_text(instr, gt),
                         key=k, instruction_tier=tier, gt_chars=len(gt), geom=geom))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, 'w') as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(json.dumps(dict(n=len(rows), keys=pick, tiers=[r['instruction_tier'] for r in rows],
                          gt_chars=[r['gt_chars'] for r in rows], geom=[r['geom'] for r in rows], out=a.out), ensure_ascii=False, indent=1))


if __name__ == '__main__':
    main()
