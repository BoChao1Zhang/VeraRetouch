#!/usr/bin/env python3
"""EPR-052：从现有 OPSD 50k 集造副本，每行补 `assistant` = 该样本 GT 六段 CoT。

用途：GKD 的 DATASET 分支（`lmbda<1` 时约 1−λ 的批次）需要数据自带 assistant 段，否则标签全 -100、
`sft_alpha·CE` 与 JSD 都落在零监督批次上（见 NOTES N5）。除 messages 追加一条 assistant 外，
其余字段（images / teacher_prompt / key / instruction_tier / gt_chars / geom）与源文件**逐字相同**。
"""
from __future__ import annotations
import argparse, hashlib, json
from pathlib import Path
from veraretouch_sprf.data import cot_text as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='/home/bc/data/runs/epr052_rl/data/opsd50k/opsd_train.jsonl')
    ap.add_argument('--records', default='/home/bc/data/runs/epr051_vlmsft/snap_sft2/records.jsonl')
    ap.add_argument('--out-dir', default='/home/bc/data/runs/epr052_rl/data/opsd50k_assist')
    a = ap.parse_args()
    src_rows = [json.loads(l) for l in open(a.src)]
    want = {r['key'] for r in src_rows}
    gt = {}
    with open(a.records) as f:
        for ln in f:
            r = json.loads(ln)
            if r['key'] in want:
                gt[r['key']] = C.target_text(r)
    assert len(gt) == len(want), (len(gt), len(want))
    out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
    n_ok = 0
    with open(out / 'opsd_train_assist.jsonl', 'w') as f:
        for row in src_rows:
            new = dict(row)
            assert [m['role'] for m in row['messages']] == ['user'], row['key']
            new['messages'] = list(row['messages']) + [dict(role='assistant', content=gt[row['key']])]
            # teacher_prompt 不变（教师侧仍靠 build_teacher_view 替换最后一条 user）
            f.write(json.dumps(new, ensure_ascii=False) + '\n'); n_ok += 1
    keys = [r['key'] for r in src_rows]
    meta = dict(n=n_ok, src=a.src, src_keys_sha256=hashlib.sha256(json.dumps(keys).encode()).hexdigest(),
                out=str(out / 'opsd_train_assist.jsonl'),
                out_sha256=hashlib.sha256((out / 'opsd_train_assist.jsonl').read_bytes()).hexdigest(),
                assistant_source='veraretouch_sprf.data.cot_text.target_text(records.jsonl)',
                template_sha256=C.template_sha256(),
                gt_chars_mean=sum(len(gt[k]) for k in keys) / max(1, len(keys)),
                note='messages 追加 assistant=GT 六段 CoT；其余字段与源文件逐字相同')
    (out / 'META.json').write_text(json.dumps(meta, indent=1, ensure_ascii=False))
    print(json.dumps(meta, ensure_ascii=False))


if __name__ == '__main__':
    main()
