#!/usr/bin/env python3
"""EPR-052：把各诊断臂的 probe_segments.jsonl 汇总成对照表（markdown）。只列数字，不下结论。"""
from __future__ import annotations
import json, statistics as st, sys
from pathlib import Path

ROOT = Path('/home/bc/data/runs/epr052_rl')
ARMS = [
    ('baseline lr5e-6 无warmup(v1)', ROOT / 'opsd_full/opsd50k_gkd/probe_segments.jsonl', 'λ=1, API top-64'),
    ('arm1 lr1e-6+warmup(v1)', ROOT / 'arms/arm1_lr1e6/probe_segments.jsonl', 'λ=1'),
    ('arm2 本地全词表教师(v3)', ROOT / 'arms/arm2_localteacher/probe_segments.jsonl', 'arm1+全词表教师；PB8 OOM，仅 step0'),
    ('arm3 λ0.75零监督(v3)', ROOT / 'arms/arm3_lambda075/probe_segments.jsonl', 'arm1+λ0.75+sft_alpha0.1（数据无 assistant）'),
    ('arm3b λ0.75+CE(v4)', ROOT / 'arms/arm3b_lambda075_ce/probe_segments.jsonl', 'arm3+assistant=GT CoT；30 步'),
    ('arm4 lr3e-6(v7)', ROOT / 'arms/arm4_lr3e6/probe_segments.jsonl', 'arm1 基础上改 lr'),
    ('arm4b adamw_torch(v7)', ROOT / 'arms/arm4b_adamw_torch/probe_segments.jsonl', 'arm4+torch AdamW(bf16 状态)；20 步'),
    ('arm5 top_p0.9(v7)', ROOT / 'arms/arm5_topp09/probe_segments.jsonl', 'arm1 基础上改 top_p'),
    ('arm6 lr5e-6+warmup(v7)', ROOT / 'arms/arm6_lr5e6_warmup/probe_segments.jsonl', 'arm1 基础上改 lr（pilot 选定配置）'),
]


def load(p):
    if not p.exists():
        return []
    out = []
    for l in open(p):
        try:
            r = json.loads(l)
        except Exception:
            continue
        if str(r.get('probe','')).endswith('-loss') or str(r.get('probe','')).endswith('-log') or str(r.get('probe','')).endswith('-grad'):
            continue
        out.append(r)
    return out


def losses(p):
    if not p.exists():
        return []
    return [json.loads(l) for l in open(p) if '"v4-loss"' in l or '"v7-log"' in l]


def m(v):
    return None if not v else st.mean(v)


def f(x, n=4):
    return '—' if x is None else (f'{x:.{n}f}' if isinstance(x, float) else str(x))


rows = []
for name, path, desc in ARMS:
    R = load(path)
    if not R:
        rows.append((name, desc, '未产出', {})); continue
    sup = [r for r in R if r.get('n_valid', 0) > 0]          # 有监督步
    zero = [r for r in R if r.get('n_valid', 1) == 0]        # 零监督步
    d = [r['div_mean'] for r in sup if r.get('div_mean') is not None]
    six = [r['six_complete_rows'] for r in sup]; B = [r['batch'] for r in sup]
    cov = [r['cov_topk_mean'] for r in sup if r.get('cov_topk_mean') is not None]
    ent = [r['entropy_mean'] for r in sup if r.get('entropy_mean') is not None]
    trunc = sum(r.get('truncated_rows') or 0 for r in sup)
    fr = {}
    for r in sup:
        for x in (r.get('finish_reason') or []):
            fr[str(x)] = fr.get(str(x), 0) + 1
    tv = [r['train_vs_rollout_logp'] for r in sup if isinstance(r.get('train_vs_rollout_logp'), dict) and r['train_vs_rollout_logp'].get('n')]
    k3 = [r['k3'] for r in sup if isinstance(r.get('k3'), dict)]
    L = losses(path)
    stat = dict(
        steps=len(R), steps_sup=len(sup), steps_zero=len(zero),
        div_first10=m(d[:10]), div_last10=m(d[-10:]), div_min=min(d) if d else None, div_max=max(d) if d else None,
        six=f"{sum(six)}/{sum(B)}" if B else '—',
        six_seq=''.join(str(x) for x in six),
        cov=m(cov), ent_first10=m(ent[:10]), ent_last10=m(ent[-10:]),
        trunc=trunc, finish=fr,
        mem=max((r.get('cuda_max_reserved_gib') or 0) for r in R),
        tv_abs_mean=m([x['abs_mean'] for x in tv]), tv_abs_max=max([x['abs_max'] for x in tv]) if tv else None,
        tv_rows=sum(x.get('rows_matched', 0) for x in tv),
        k3_mean=m([x['mean'] for x in k3]), k3_neg=sum(x['n_negative'] for x in k3) if k3 else None,
        loss_ds=m([x['loss'] for x in L if 'DATASET' in str(x.get('data_source'))]),
        loss_st=m([x['loss'] for x in L if 'STUDENT' in str(x.get('data_source'))]),
        loss_nonfinite=sum(1 for x in L if not x.get('finite', True)) if L else None,
        seg_first10=[m([r['per_segment'][f'seg{k}']['mean'] for r in sup[:10] if r['per_segment'][f'seg{k}']['mean'] is not None]) for k in range(1, 7)],
        seg_last10=[m([r['per_segment'][f'seg{k}']['mean'] for r in sup[-10:] if r['per_segment'][f'seg{k}']['mean'] is not None]) for k in range(1, 7)],
    )
    rows.append((name, desc, '', stat))

print('| 臂 | 变量 | 步数(有监督/零监督) | 散度 前10→后10 | 六段有序行 | top-64 覆盖 | 熵 前10→后10 | 触顶 | gpu1 峰值 GiB |')
print('|---|---|---|---|---|---|---|---|---|')
for name, desc, note, s in rows:
    if note:
        print(f'| {name} | {desc} | {note} | — | — | — | — | — | — |'); continue
    print(f"| {name} | {desc} | {s['steps']}({s['steps_sup']}/{s['steps_zero']}) | {f(s['div_first10'])} → {f(s['div_last10'])} | {s['six']} | {f(s['cov'],5)} | {f(s['ent_first10'],3)} → {f(s['ent_last10'],3)} | {s['trunc']} | {f(s['mem'],1)} |")
print()
print('| 臂 | 分段散度 seg1..6（前10 / 后10） | finish_reason | 训推 logp 差 |Δ|均值/最大(行数) | k3 均值/负值数 | loss DATASET / STUDENT / 非有限 |')
print('|---|---|---|---|---|---|')
for name, desc, note, s in rows:
    if note:
        continue
    a = '/'.join(f(x, 3) for x in s['seg_first10']); b = '/'.join(f(x, 3) for x in s['seg_last10'])
    print(f"| {name} | {a} → {b} | {s['finish']} | {f(s['tv_abs_mean'],4)} / {f(s['tv_abs_max'],4)} ({s['tv_rows']}) | {f(s['k3_mean'],4)} / {s['k3_neg']} | {f(s['loss_ds'],4)} / {f(s['loss_st'],4)} / {s['loss_nonfinite']} |")
print()
for name, desc, note, s in rows:
    if not note:
        print(f"{name} 六段有序逐步序列: {s['six_seq']}")
