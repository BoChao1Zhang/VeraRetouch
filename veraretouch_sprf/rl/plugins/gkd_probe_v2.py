"""EPR-052 ENG-2 探针 v2（臂 2/3 起用；v1 = gkd_segment_probe.py 保持不动，臂 1 用的是 v1）。

在 v1（分段散度 / 阶段 token 序 / 显存）之上补 SURVEY_STRUCTURE_DEGEN §1 指出的四个诊断量：
  D1 覆盖率：学生采样 token 落在教师 top-64 内的比例（API 教师直接查 topk_indices；本地全词表教师则
     现取教师 top-64 做同口径统计，故臂 2 该列可与臂 1/3 直接比较，而不是恒 1）。
  D2 策略熵：学生全词表分布的逐 token 熵（均值/分位）。ms-swift GKD 训练器无此指标。
  D3 finish_reason 分布 + 补全长度分位（钩 GKDTrainer._rollout_samples）。
  D4 训推不匹配：vLLM rollout 返回的采样 token logprob 与训练前向 logprob 的逐 token 差（均值/最大/分位），
     以及教师-学生 logp 差 d 与 k3 = exp(d)−d−1 的统计（k3 数学上 ≥0，记录负值计数以查数值问题）。
     GKD 路径无任何 rollout 修正（gkd_trainer.py 内 rollout_importance 命中 0）。
损失值不改动：只读张量、只落盘。探针异常一律吞掉并记 probe_error（结果落盘先于可选阶段）。
"""
from __future__ import annotations

import json
import os
import time

import torch

from swift.rlhf_trainers import gkd_trainer as GT
from swift.rlhf_trainers.gkd_loss import TeacherOutput, extract_active, _align_vocab

STAGE_IDS = [int(x) for x in os.environ.get('VR_STAGE_IDS', '151669,151670,151671,151672,151673,151674').split(',')]
OUT = os.environ.get('VR_PROBE_OUT', 'probe_segments.jsonl')
COV_K = int(os.environ.get('VR_COV_K', '64'))
CHUNK = 512
_orig_jsd = GT.GKDTrainer._compute_jsd_loss
_roll: dict = {}


def _q(t, qs=(0.05, 0.5, 0.95)):
    if t.numel() == 0:
        return None
    v = torch.quantile(t.float(), torch.tensor(qs, device=t.device))
    return [round(float(x), 6) for x in v]


def _per_token_div(s, t, beta):
    out = []
    for a in range(0, s.shape[0], CHUNK):
        sl = torch.log_softmax(s[a:a + CHUNK], -1)
        tl = torch.log_softmax(t[a:a + CHUNK], -1)
        if beta == 0:
            d = (tl.exp() * (tl - sl)).sum(-1)
        elif beta == 1:
            d = (sl.exp() * (sl - tl)).sum(-1)
        else:
            b = torch.tensor(beta, dtype=sl.dtype, device=sl.device)
            m = torch.logsumexp(torch.stack([sl + torch.log1p(-b), tl + torch.log(b)]), 0)
            d = b * (tl.exp() * (tl - m)).sum(-1) + (1 - b) * (sl.exp() * (sl - m)).sum(-1)
        out.append(d)
    return torch.cat(out) if out else s.new_zeros(0)


def _entropy_and_logp(s_logits, tokens):
    """全词表学生分布的逐 token 熵，以及采样 token 的训练前向 logprob。"""
    ent, lp = [], []
    for a in range(0, s_logits.shape[0], CHUNK):
        sl = torch.log_softmax(s_logits[a:a + CHUNK].float(), -1)
        ent.append(-(sl.exp() * sl).sum(-1))
        lp.append(sl.gather(-1, tokens[a:a + CHUNK].view(-1, 1)).squeeze(-1))
    return torch.cat(ent), torch.cat(lp)


def _patched(self, student_logits, teacher_output: TeacherOutput, labels):
    rec = dict(t=time.time(), step=int(self.state.global_step), beta=float(self.beta),
               temperature=float(self.temperature), student_seq_len=int(labels.shape[1]),
               teacher_seq_len=int(teacher_output.labels.shape[1]), probe='v2')
    try:
        with torch.no_grad():
            sl = torch.roll(labels, -1, 1)
            tl = torch.roll(teacher_output.labels, -1, 1)
            to = TeacherOutput(full_logits=teacher_output.full_logits, topk_logprobs=teacher_output.topk_logprobs,
                               topk_indices=teacher_output.topk_indices, labels=tl)
            s_act, t_act, n = extract_active(student_logits, to, sl)
            rows_valid = (sl != -100)
            per_row_n = rows_valid.sum(1).tolist()
            toks = sl[rows_valid]
            is_stage = torch.zeros_like(toks, dtype=torch.bool)
            for sid in STAGE_IDS:
                is_stage |= toks == sid
            seg = torch.zeros_like(toks)
            orders, off = [], 0
            for nr in per_row_n:
                st = is_stage[off:off + nr]
                seg[off:off + nr] = (torch.cumsum(st.long(), 0) - st.long()).clamp(max=6)
                orders.append([STAGE_IDS.index(int(x)) + 1 for x in toks[off:off + nr][st].tolist()])
                off += nr
            rec.update(n_valid=int(n), batch=len(per_row_n), row_lens=per_row_n, stage_orders=orders,
                       six_complete_rows=sum(o == [1, 2, 3, 4, 5, 6] for o in orders),
                       six_complete=all(o == [1, 2, 3, 4, 5, 6] for o in orders))
            # ---- 散度（与 gkd_loss 同口径）+ D1 覆盖率 -------------------------- #
            if t_act.is_topk_mode:
                rec['support_mode'] = f'api_topk{t_act.topk_indices.shape[-1]}'
                s_sup = torch.gather(s_act.float(), -1, t_act.topk_indices)
                t_sup = t_act.topk_logprobs.float()
                cov = (t_act.topk_indices == toks.view(-1, 1)).any(-1).float()
                t_lp_sampled = torch.where(cov.bool(),
                                           torch.log_softmax(t_sup, -1).gather(
                                               -1, (t_act.topk_indices == toks.view(-1, 1)).float().argmax(-1, keepdim=True)).squeeze(-1),
                                           torch.full_like(cov, float('nan')))
            else:
                rec['support_mode'] = 'local_full_vocab'
                s_sup, t_full = _align_vocab(s_act.float(), t_act.full_logits.float())
                t_sup = t_full
                topk_idx = t_full.topk(min(COV_K, t_full.shape[-1]), dim=-1).indices
                cov = (topk_idx == toks.view(-1, 1)).any(-1).float()
                t_lp_sampled = torch.log_softmax(t_full, -1).gather(-1, toks.view(-1, 1)).squeeze(-1)
            d = _per_token_div(s_sup / self.temperature, t_sup / self.temperature, float(self.beta))
            rec['div_mean'] = float(d.mean()) if d.numel() else None
            rec['div_support'] = rec['support_mode']
            rec['cov_topk_mean'] = float(cov.mean()) if cov.numel() else None
            rec['cov_topk_by_stage_token'] = float(cov[is_stage].mean()) if is_stage.any() else None
            segs = {}
            for m in range(7):
                msk = seg == m
                nm = f'seg{m + 1 if m < 6 else "_tail"}'
                segs[nm] = dict(n=int(msk.sum()), mean=float(d[msk].mean()) if msk.any() else None,
                                cov=float(cov[msk].mean()) if msk.any() else None)
            rec['per_segment'] = segs
            # ---- D2 策略熵 + 训练前向 logprob ---------------------------------- #
            ent, s_lp = _entropy_and_logp(s_act, toks)
            rec['entropy_mean'] = float(ent.mean()); rec['entropy_q'] = _q(ent)
            rec['entropy_by_stage_token'] = float(ent[is_stage].mean()) if is_stage.any() else None
            # ---- D4 教师-学生 logp 差 d 与 k3 ---------------------------------- #
            fin = torch.isfinite(t_lp_sampled)
            if fin.any():
                dd = (t_lp_sampled[fin] - s_lp[fin])
                k3 = torch.expm1(dd) - dd
                rec['teacher_minus_student_logp'] = dict(mean=float(dd.mean()), q=_q(dd),
                                                         n=int(fin.sum()), n_uncovered=int((~fin).sum()))
                rec['k3'] = dict(mean=float(k3.mean()), min=float(k3.min()), n_negative=int((k3 < 0).sum()))
            # ---- D4 训推不匹配（vLLM rollout logprob vs 训练前向） --------------- #
            rl = _roll.get('logprobs') or []
            if rl and len(rl) == len(per_row_n):
                diffs, off, nmatch = [], 0, 0
                for i, nr in enumerate(per_row_n):
                    v = rl[i]
                    if v is not None and len(v) == nr:
                        diffs.append(s_lp[off:off + nr] - torch.tensor(v, device=s_lp.device, dtype=s_lp.dtype))
                        nmatch += 1
                    off += nr
                if diffs:
                    dv = torch.cat(diffs)
                    rec['train_vs_rollout_logp'] = dict(rows_matched=nmatch, rows=len(per_row_n), n=int(dv.numel()),
                                                        mean=float(dv.mean()), abs_mean=float(dv.abs().mean()),
                                                        abs_max=float(dv.abs().max()), q=_q(dv))
                else:
                    rec['train_vs_rollout_logp'] = dict(rows_matched=0, rows=len(per_row_n), note='length mismatch')
            else:
                rec['train_vs_rollout_logp'] = dict(note='no rollout logprobs stashed', have=len(rl), rows=len(per_row_n))
            # ---- D3 finish_reason / 长度 --------------------------------------- #
            if _roll.get('finish_reason') is not None:
                rec['finish_reason'] = _roll['finish_reason']
                rec['completion_len'] = _roll['lens']
                rec['completion_len_q'] = _q(torch.tensor(_roll['lens'], dtype=torch.float)) if _roll['lens'] else None
                rec['truncated_rows'] = sum(1 for f in _roll['finish_reason'] if f == 'length')
            rec['cuda_max_reserved_gib'] = torch.cuda.max_memory_reserved() / 2 ** 30
            rec['cuda_max_alloc_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
    except Exception as e:
        rec['probe_error'] = repr(e)
    with open(OUT, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    return _orig_jsd(self, student_logits, teacher_output, labels)


GT.GKDTrainer._compute_jsd_loss = _patched

_orig_roll = GT.GKDTrainer._rollout_samples


def _roll_patched(self, inputs):
    samples = _orig_roll(self, inputs)
    try:
        _roll['finish_reason'] = [getattr(s, 'finish_reason', None) for s in samples]
        _roll['lens'] = [len(s.response_token_ids) for s in samples]
        _roll['logprobs'] = [(list(s.rollout_logprobs) if getattr(s, 'rollout_logprobs', None) else None) for s in samples]
        _roll['data_source'] = str(getattr(self, '_data_source', None))
    except Exception as e:
        _roll['error'] = repr(e)
    return samples


GT.GKDTrainer._rollout_samples = _roll_patched
print(f'[gkd_probe_v2] patched _compute_jsd_loss + _rollout_samples; out={OUT} cov_k={COV_K}', flush=True)
