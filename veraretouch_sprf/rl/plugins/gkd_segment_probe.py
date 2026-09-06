"""EPR-052 ENG-2 探针插件（--external_plugins）：GKD 路径按六段分箱记录教师-学生逐 token 散度。

猴补丁 GKDTrainer._compute_jsd_loss：在调用原函数前，用与 gkd_loss 完全相同的 mask（extract_active）、
温度缩放与 β 公式（jsd_loss 循环体）算逐 token 散度 [N]，按目标 token 序列里 <vr_stage_m> 的位置分段
（段 m = 第 m-1 个阶段 token 之后到第 m 个阶段 token 含），落 jsonl（VR_PROBE_OUT）。不改损失值。
只处理 full-logits 教师（本地教师）；top-k/API 教师时只记长度与阶段 token 计数。
"""
from __future__ import annotations
import json, os, time
import torch
from swift.rlhf_trainers import gkd_trainer as GT
from swift.rlhf_trainers.gkd_loss import TeacherOutput, extract_active, _align_vocab

STAGE_IDS = [int(x) for x in os.environ.get('VR_STAGE_IDS', '151669,151670,151671,151672,151673,151674').split(',')]
OUT = os.environ.get('VR_PROBE_OUT', 'probe_segments.jsonl')
CHUNK = 512
_orig = GT.GKDTrainer._compute_jsd_loss


def _per_token_div(s, t, beta):
    out = []
    for a in range(0, s.shape[0], CHUNK):
        sl = torch.log_softmax(s[a:a + CHUNK], -1)
        tl = torch.log_softmax(t[a:a + CHUNK], -1)
        if beta == 0:
            d = (tl.exp() * (tl - sl)).sum(-1)                 # forward KL(T||S)
        elif beta == 1:
            d = (sl.exp() * (sl - tl)).sum(-1)                 # reverse KL(S||T)
        else:
            b = torch.tensor(beta, dtype=sl.dtype, device=sl.device)
            m = torch.logsumexp(torch.stack([sl + torch.log1p(-b), tl + torch.log(b)]), 0)
            d = b * (tl.exp() * (tl - m)).sum(-1) + (1 - b) * (sl.exp() * (sl - m)).sum(-1)
        out.append(d)
    return torch.cat(out) if out else s.new_zeros(0)


def _patched(self, student_logits, teacher_output: TeacherOutput, labels):
    rec = dict(t=time.time(), step=int(self.state.global_step), beta=float(self.beta), temperature=float(self.temperature),
               student_seq_len=int(labels.shape[1]), teacher_seq_len=int(teacher_output.labels.shape[1]))
    try:
        with torch.no_grad():
            sl = torch.roll(labels, -1, 1)
            tl = torch.roll(teacher_output.labels, -1, 1)
            to = TeacherOutput(full_logits=teacher_output.full_logits, topk_logprobs=teacher_output.topk_logprobs,
                               topk_indices=teacher_output.topk_indices, labels=tl)
            s_act, t_act, n = extract_active(student_logits, to, sl)
            # 逐序列（B>1）：按 labels 行切分有效 token
            rows_valid = (sl != -100)
            per_row_n = rows_valid.sum(1).tolist()
            toks = sl[rows_valid]
            is_stage = torch.zeros_like(toks, dtype=torch.bool)
            for sid in STAGE_IDS:
                is_stage |= toks == sid
            # 段号在每一行内独立累计
            seg = torch.zeros_like(toks)
            orders = []
            off = 0
            for nr in per_row_n:
                st = is_stage[off:off + nr]
                seg[off:off + nr] = (torch.cumsum(st.long(), 0) - st.long()).clamp(max=6)
                orders.append([STAGE_IDS.index(int(x)) + 1 for x in toks[off:off + nr][st].tolist()])
                off += nr
            rec.update(n_valid=int(n), batch=len(per_row_n), row_lens=per_row_n, stage_orders=orders,
                       six_complete_rows=sum(o == [1, 2, 3, 4, 5, 6] for o in orders),
                       six_complete=all(o == [1, 2, 3, 4, 5, 6] for o in orders))
            if t_act.is_topk_mode:
                # 教师 API top-k：与 gkd_loss 同口径——学生在 top-k 支撑上 gather 后 log_softmax
                s = torch.gather(s_act.float(), -1, t_act.topk_indices)
                tt = t_act.topk_logprobs.float()
                d = _per_token_div(s / self.temperature, tt / self.temperature, float(self.beta))
                rec['div_support'] = f'topk{t_act.topk_indices.shape[-1]}'
            else:
                s, tt = _align_vocab(s_act.float(), t_act.full_logits.float())
                d = _per_token_div(s / self.temperature, tt / self.temperature, float(self.beta))
                rec['div_support'] = 'full_vocab'
            rec['div_mean'] = float(d.mean()) if d.numel() else None
            segs = {}
            for m in range(7):
                msk = seg == m
                segs[f'seg{m + 1 if m < 6 else "_tail"}'] = dict(n=int(msk.sum()), mean=float(d[msk].mean()) if msk.any() else None,
                                                                sum=float(d[msk].sum()) if msk.any() else 0.0)
            rec['per_segment'] = segs
            rec['cuda_max_reserved_gib'] = torch.cuda.max_memory_reserved() / 2 ** 30
            rec['cuda_max_alloc_gib'] = torch.cuda.max_memory_allocated() / 2 ** 30
    except Exception as e:  # 探针失败不得毁掉训练
        rec['probe_error'] = repr(e)
    with open(OUT, 'a') as f:
        f.write(json.dumps(rec) + '\n')
    return _orig(self, student_logits, teacher_output, labels)


GT.GKDTrainer._compute_jsd_loss = _patched
print(f'[gkd_segment_probe] patched GKDTrainer._compute_jsd_loss; out={OUT} stage_ids={STAGE_IDS}', flush=True)


# --------------------------------------------------------------------------- #
# GRPO / OPD-RL 路径：k3 = exp(d)-d-1（d = log π_T − log π_S，采样 token 上）按段分箱
# --------------------------------------------------------------------------- #
try:
    from swift.rlhf_trainers import grpo_trainer as GRT
    _k3_stash: list = []
    _orig_k3 = GRT.compute_teacher_kl_per_token

    def _k3_patched(teacher_lp, policy_lp, completion_mask):
        k3 = _orig_k3(teacher_lp, policy_lp, completion_mask)
        try:
            d = (teacher_lp - policy_lp).masked_fill(~completion_mask.bool(), 0.0)
            _k3_stash.append(dict(k3=k3.detach().float().cpu(), k1=d.detach().float().cpu(), mask=completion_mask.detach().bool().cpu()))
        except Exception as e:
            _k3_stash.append(dict(error=repr(e)))
        return k3

    GRT.compute_teacher_kl_per_token = _k3_patched
    _orig_post = GRT.GRPOTrainer._postprocess_batch

    def _post_patched(self, samples, batch_encoded_inputs):
        _k3_stash.clear()
        ids = [list(s.response_token_ids) for s in samples]
        finish = [getattr(s, 'finish_reason', None) for s in samples]
        out = _orig_post(self, samples, batch_encoded_inputs)
        rec = dict(t=time.time(), step=int(self.state.global_step), path='grpo', n_samples=len(samples),
                   cuda_max_reserved_gib=torch.cuda.max_memory_reserved() / 2 ** 30,
                   cuda_max_alloc_gib=torch.cuda.max_memory_allocated() / 2 ** 30, finish_reason=finish)
        try:
            rows = []
            for st in _k3_stash:
                if 'error' in st:
                    raise RuntimeError(st['error'])
                for b in range(st['k3'].shape[0]):
                    m = st['mask'][b]
                    rows.append((st['k3'][b][m], st['k1'][b][m]))
            per = []
            for i, (k3r, k1r) in enumerate(rows):
                toks = torch.tensor(ids[i]) if i < len(ids) else torch.zeros(0, dtype=torch.long)
                n = min(len(toks), k3r.numel())
                is_stage = torch.zeros(n, dtype=torch.bool)
                for sid in STAGE_IDS:
                    is_stage |= toks[:n] == sid
                seg = (torch.cumsum(is_stage.long(), 0) - is_stage.long()).clamp(max=6)
                order = [STAGE_IDS.index(int(x)) + 1 for x in toks[:n][is_stage].tolist()]
                segs = {}
                for mm in range(7):
                    msk = seg == mm
                    segs[f'seg{mm + 1 if mm < 6 else "_tail"}'] = dict(n=int(msk.sum()), k3_mean=float(k3r[:n][msk].mean()) if msk.any() else None,
                                                                       k1_mean=float(k1r[:n][msk].mean()) if msk.any() else None)
                per.append(dict(n_tokens=int(k3r.numel()), n_ids=len(toks), aligned=(len(toks) == k3r.numel()),
                                k3_mean=float(k3r.mean()) if k3r.numel() else None, k1_mean=float(k1r.mean()) if k1r.numel() else None,
                                stage_order=order, six_complete=(order == [1, 2, 3, 4, 5, 6]), per_segment=segs, completion_ids=ids[i] if i < len(ids) else None))
            rec['samples'] = per
        except Exception as e:
            rec['probe_error'] = repr(e)
        with open(OUT, 'a') as f:
            f.write(json.dumps(rec) + '\n')
        return out

    GRT.GRPOTrainer._postprocess_batch = _post_patched
    print('[gkd_segment_probe] patched GRPOTrainer._postprocess_batch / compute_teacher_kl_per_token', flush=True)
except Exception as _e:  # grpo 模块不可用时不影响 GKD 路径
    print(f'[gkd_segment_probe] grpo hook skipped: {_e!r}', flush=True)
