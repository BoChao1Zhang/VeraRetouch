"""EPR-052 ENG-2 探针 v3（臂 2/3 用；v1/v2 保持不动）。

= v2 的四个诊断量（top-K 覆盖率 / 策略熵 / finish_reason 与长度 / rollout-vs-训练 logprob 差与 k3），
但**全部在一次分块循环里算**，任何时刻只把 [chunk, V] 提升到 fp32。
v2 在本地全词表教师 + PB=8 下 OOM（94.84 GiB）：`_align_vocab(s_act.float(), t.full_logits.float())`
会一次性生成两份 [N,V] fp32（N≈13k、V=151,936 ⇒ 每份 ≈7.9 GB），是探针自身引入的显存。
损失值不改动：只读、只落盘；异常吞掉记 probe_error。
"""
from __future__ import annotations

import json
import os
import time

import torch

from swift.rlhf_trainers import gkd_trainer as GT
from swift.rlhf_trainers.gkd_loss import TeacherOutput, extract_active

STAGE_IDS = [int(x) for x in os.environ.get('VR_STAGE_IDS', '151669,151670,151671,151672,151673,151674').split(',')]
OUT = os.environ.get('VR_PROBE_OUT', 'probe_segments.jsonl')
COV_K = int(os.environ.get('VR_COV_K', '64'))
CHUNK = int(os.environ.get('VR_PROBE_CHUNK', '256'))
_orig_jsd = GT.GKDTrainer._compute_jsd_loss
_roll: dict = {}


def _q(t, qs=(0.05, 0.5, 0.95)):
    if t is None or t.numel() == 0:
        return None
    v = torch.quantile(t.float(), torch.tensor(qs, device=t.device))
    return [round(float(x), 6) for x in v]


def _scan(s_act, t_act, toks, temperature, beta):
    """一次分块扫描，返回逐 token 的 (散度, 覆盖, 学生熵, 学生logp, 教师logp@采样token)。"""
    N = s_act.shape[0]
    dev = s_act.device
    div = torch.empty(N, device=dev); cov = torch.empty(N, device=dev)
    ent = torch.empty(N, device=dev); s_lp = torch.empty(N, device=dev)
    t_lp = torch.full((N,), float('nan'), device=dev)
    topk_mode = t_act.is_topk_mode
    for a in range(0, N, CHUNK):
        b = min(a + CHUNK, N)
        s_full = s_act[a:b].float()                       # [c, V]（学生恒全词表）
        s_logn = torch.log_softmax(s_full, -1)
        ent[a:b] = -(s_logn.exp() * s_logn).sum(-1)
        s_lp[a:b] = s_logn.gather(-1, toks[a:b].view(-1, 1)).squeeze(-1)
        if topk_mode:
            idx = t_act.topk_indices[a:b]                 # [c, K]
            t_sup = t_act.topk_logprobs[a:b].float()
            s_sup = s_full.gather(-1, idx)
            hit = (idx == toks[a:b].view(-1, 1))
            cov[a:b] = hit.any(-1).float()
            tl_n = torch.log_softmax(t_sup / temperature, -1)
            pos = hit.float().argmax(-1, keepdim=True)
            t_lp[a:b] = torch.where(hit.any(-1), tl_n.gather(-1, pos).squeeze(-1), torch.full((b - a,), float('nan'), device=dev))
        else:
            t_full = t_act.full_logits[a:b].float()
            v = min(s_full.shape[-1], t_full.shape[-1])
            s_sup, t_sup = s_full[:, :v], t_full[:, :v]
            topk_idx = t_sup.topk(min(COV_K, v), dim=-1).indices
            cov[a:b] = (topk_idx == toks[a:b].view(-1, 1)).any(-1).float()
            t_lp[a:b] = torch.log_softmax(t_sup, -1).gather(-1, toks[a:b].view(-1, 1)).squeeze(-1)
        sl = torch.log_softmax(s_sup / temperature, -1)
        tl = torch.log_softmax(t_sup / temperature, -1)
        if beta == 0:
            d = (tl.exp() * (tl - sl)).sum(-1)
        elif beta == 1:
            d = (sl.exp() * (sl - tl)).sum(-1)
        else:
            bt = torch.tensor(beta, dtype=sl.dtype, device=dev)
            m = torch.logsumexp(torch.stack([sl + torch.log1p(-bt), tl + torch.log(bt)]), 0)
            d = bt * (tl.exp() * (tl - m)).sum(-1) + (1 - bt) * (sl.exp() * (sl - m)).sum(-1)
        div[a:b] = d
        del s_full, s_logn, s_sup, t_sup, sl, tl
    return div, cov, ent, s_lp, t_lp


def _patched(self, student_logits, teacher_output: TeacherOutput, labels):
    rec = dict(t=time.time(), step=int(self.state.global_step), beta=float(self.beta),
               temperature=float(self.temperature), student_seq_len=int(labels.shape[1]),
               teacher_seq_len=int(teacher_output.labels.shape[1]), probe='v3')
    try:
        with torch.no_grad():
            sl_lab = torch.roll(labels, -1, 1)
            tl_lab = torch.roll(teacher_output.labels, -1, 1)
            to = TeacherOutput(full_logits=teacher_output.full_logits, topk_logprobs=teacher_output.topk_logprobs,
                               topk_indices=teacher_output.topk_indices, labels=tl_lab)
            s_act, t_act, n = extract_active(student_logits, to, sl_lab)
            rows_valid = (sl_lab != -100)
            per_row_n = rows_valid.sum(1).tolist()
            toks = sl_lab[rows_valid]
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
                       six_complete=all(o == [1, 2, 3, 4, 5, 6] for o in orders),
                       support_mode=(f'api_topk{t_act.topk_indices.shape[-1]}' if t_act.is_topk_mode else 'local_full_vocab'))
            div, cov, ent, s_lp, t_lp = _scan(s_act, t_act, toks, float(self.temperature), float(self.beta))
            rec['div_mean'] = float(div.mean()) if div.numel() else None
            rec['div_support'] = rec['support_mode']
            rec['cov_topk_mean'] = float(cov.mean()); rec['cov_topk_k'] = COV_K
            rec['cov_topk_by_stage_token'] = float(cov[is_stage].mean()) if is_stage.any() else None
            rec['entropy_mean'] = float(ent.mean()); rec['entropy_q'] = _q(ent)
            rec['entropy_by_stage_token'] = float(ent[is_stage].mean()) if is_stage.any() else None
            segs = {}
            for m in range(7):
                msk = seg == m
                nm = f'seg{m + 1 if m < 6 else "_tail"}'
                segs[nm] = dict(n=int(msk.sum()), mean=float(div[msk].mean()) if msk.any() else None,
                                cov=float(cov[msk].mean()) if msk.any() else None,
                                ent=float(ent[msk].mean()) if msk.any() else None)
            rec['per_segment'] = segs
            fin = torch.isfinite(t_lp)
            if fin.any():
                dd = t_lp[fin] - s_lp[fin]
                k3 = torch.expm1(dd) - dd
                rec['teacher_minus_student_logp'] = dict(mean=float(dd.mean()), q=_q(dd), n=int(fin.sum()),
                                                         n_uncovered=int((~fin).sum()))
                rec['k3'] = dict(mean=float(k3.mean()), min=float(k3.min()), n_negative=int((k3 < 0).sum()))
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
                    rec['train_vs_rollout_logp'] = dict(rows_matched=0, rows=len(per_row_n), note='length mismatch',
                                                        rollout_lens=[len(v) if v else None for v in rl])
            else:
                rec['train_vs_rollout_logp'] = dict(note='no rollout logprobs stashed', have=len(rl), rows=len(per_row_n))
            if _roll.get('finish_reason') is not None:
                rec['finish_reason'] = _roll['finish_reason']; rec['completion_len'] = _roll['lens']
                rec['completion_len_q'] = _q(torch.tensor(_roll['lens'], dtype=torch.float)) if _roll['lens'] else None
                rec['truncated_rows'] = sum(1 for f in _roll['finish_reason'] if f == 'length')
                rec['data_source'] = _roll.get('data_source')
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
print(f'[gkd_probe_v3] patched; out={OUT} cov_k={COV_K} chunk={CHUNK}', flush=True)
