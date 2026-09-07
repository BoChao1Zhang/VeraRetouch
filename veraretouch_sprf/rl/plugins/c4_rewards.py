"""EPR-052 C4：GRPO 双奖励 ORM 插件（latent 余弦 + 执行器 linf8）。

口径（预注册）
  R1 latent  : 生成文本 → 按 <vr_stage_m> 定位六段 → span mean-pool（当前策略权重的 last_hidden_state，
               与 dump_readout 同一口径）→ **冻结** S2F-B adapter → (6,128) slot 序 → 对 e*(该 key 的 slot 序目标)
               逐槽余弦后取均值。解析失败/缺段 ⇒ 该样本奖励 = MISS_LATENT（默认 0.0，即“余弦为 0”的下界）。
  R2 executor: 同一 latent（slot→chain 用 flip(0)）注入**冻结** BK-FULL（ckpt_last.pt）→ rollout → linf8 p50
               → 奖励 = **−linf8/23.0**（23.0 = held-out d6 identity 口径）。解析失败 ⇒ MISS_EXEC（默认 −1.0）。
  两项都只依赖同一次读出前向（缓存共享），不重复前向。
守卫
  A-inj（初始化时跑，失败即 die）：对 2 个键，注入 `edit_enc(inv_table[row])` 的路径与 oracle_lut 路径逐位相等。
  硬停：六段有序率**连续 2 个记录点 < 0.5** ⇒ 置 `trainer.control.should_training_stop = True` 并落盘原因。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from swift.rewards import ORM, orms
from swift.rlhf_trainers import grpo_trainer as GRT

from veraretouch_sprf.data import train_stage0 as T0
from veraretouch_sprf.data import stage_targets as ST
from veraretouch_sprf.data import archive_assets as AA
from veraretouch_sprf.data import cot_text as C
from veraretouch_sprf.data import q3vl_text as QT
from veraretouch_sprf.models import bk_load, edit_cond as EC
from veraretouch_sprf.models.vlm import q3vl_common as Q
from veraretouch_sprf.solver import stage_solver_bk as SS

RUN_BK = Path(os.environ.get('VR_BK_RUN', '/home/bc/data/runs/epr051_sprf/bkfull_adagn_ff_affhead'))
ADAPT = Path(os.environ.get('VR_ADAPT', '/home/bc/data/runs/epr051_vlmsft/adapt_s2fb/ckpt_epoch1'))
TARGETS = Path(os.environ.get('VR_TARGETS', '/home/bc/data/runs/epr051_vlmsft/targets_bkfull/target_latents_bkfull.pt'))
INV_INDEX = Path(os.environ.get('VR_INV_INDEX', '/home/bc/data/runs/epr051_sprf/lut_inv_cache/g17/index.json'))
ASSETS = Path(os.environ.get('VR_ASSETS', '/home/bc/data/runs/epr051_vlmsft/snap_sft2/assets_index.json'))
OUT = os.environ.get('VR_REWARD_OUT', '/home/bc/data/runs/epr052_rl/c4/rewards.jsonl')
N_PIX = int(os.environ.get('VR_N_PIX', '16384'))
IDENTITY_LINF8 = float(os.environ.get('VR_IDENTITY_LINF8', '23.0'))
MISS_LATENT = float(os.environ.get('VR_MISS_LATENT', '0.0'))
MISS_EXEC = float(os.environ.get('VR_MISS_EXEC', '-1.0'))
READOUT_CHUNK = int(os.environ.get('VR_READOUT_CHUNK', '4'))
STOP_THRESH = float(os.environ.get('VR_SIX_STOP', '0.5'))

_TRAINER: dict = {}
_orig_init = GRT.GRPOTrainer.__init__


from transformers import TrainerCallback  # noqa: E402


class _LogCb(TrainerCallback):
    """transformers 官方扩展点：记录每步 loss/grad_norm/lr（不包装任何损失路径函数，见 NOTES N10/N11）。"""

    def on_log(self, args, state, control, logs=None, **kw):
        try:
            rec = dict(t=time.time(), event='log', step=int(getattr(state, 'global_step', -1)))
            for k, v in (logs or {}).items():
                if isinstance(v, (int, float, str, bool)) or v is None:
                    rec[k] = v
            _log(rec)
        except Exception:
            pass


def _init_patched(self, *a, **k):
    _orig_init(self, *a, **k)
    _TRAINER['t'] = self
    try:
        self.add_callback(_LogCb())
    except Exception as e:
        print(f'[c4] on_log callback NOT added: {e!r}', flush=True)
    print('[c4] trainer handle captured', flush=True)


GRT.GRPOTrainer.__init__ = _init_patched


def _log(rec):
    with open(OUT, 'a') as f:
        f.write(json.dumps(rec, ensure_ascii=False) + '\n')


class _Ctx:
    """BK-FULL / adapter / 目标表 / 每键数据的惰性单例。"""

    def __init__(self):
        self.ready = False

    def build(self):
        if self.ready:
            return
        t0 = time.time()
        ra = json.loads((RUN_BK / 'run_args.json').read_text())
        self.cd = ra['config']
        self.n_steps = int(ra['data_law']['n_steps'])
        self.model = bk_load.load_bk_model(RUN_BK / 'run_args.json', RUN_BK / 'ckpt_last.pt', 'cuda:0')
        self.model.eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        self.inv = EC.InvLutSource(self.cd['edit']['inv_cache_dir'], int(self.cd['edit']['grid']))
        ST.install_compact_row(T0)
        T0.ASSET_MODE = self.cd['data']['asset_source']
        AA.install(T0)
        assert getattr(T0, '_SPRF_ARCHIVE_INSTALLED', False), 'X5: archive_assets 未装上'
        self.path_mode = self.cd['flow']['path_mode']
        self.nfe = int(self.cd['solver']['nfe_per_stage'])
        self.lo = float(self.cd['solver']['clamp_lo']); self.hi = float(self.cd['solver']['clamp_hi'])
        self.real_edit_enc = self.model.bk.core.edit_enc
        self.shard2feat = {}
        for e in ra['encoder']['files']:
            self.shard2feat.setdefault(Path(e['shard']).name, []).append(e['file'])
        self.index = json.load(open(ASSETS))['index']
        names = list(json.loads(INV_INDEX.read_text())['names'])
        self.row_of = {n: i for i, n in enumerate(names)}
        tl = torch.load(TARGETS, map_location='cuda:0')
        self.targets = tl['target_latents'].to('cuda:0')
        self.targets_sha = tl.get('tensor_sha256')
        ab = torch.load(ADAPT / 'adapter.pt', map_location='cpu')
        from veraretouch_sprf.models.vlm.adapter import Adapter
        self.adapter = Adapter(in_dim=ab['in_dim'], hidden=ab['hidden']).to('cuda:0')
        self.adapter.load_state_dict(ab['state_dict'])
        self.adapter.eval()
        for p in self.adapter.parameters():
            p.requires_grad_(False)
        self.keycache: dict = {}
        self.bound = False
        self.ready = True
        print(f'[c4] ctx built {time.time()-t0:.1f}s | targets {tuple(self.targets.shape)} sha {str(self.targets_sha)[:12]} '
              f'| adapter {ADAPT} | n_pix {N_PIX}', flush=True)
        self._a_inj_guard()

    def key_data(self, key):
        if key in self.keycache:
            return self.keycache[key]
        e = self.index[key]
        sid, dep = key.split('|'); depth = int(dep[1:])
        with open(Path(e['dir']) / 'pairs.jsonl', 'rb') as fh:
            fh.seek(e['off']); row = json.loads(fh.read(e['len']))
        assert row['id'] == sid, (row['id'], sid)
        x0, y = T0.load_pair(e['dir'], row, e['asset'])
        if not self.bound:
            law = T0.bind_build_config([{'shard': e['dir']}], self.cd['guard']['build_config_sha256_allowed'])
            assert int(law['n_steps']) == self.n_steps
            self.bound = True
        mask = T0.depth_mask(depth, self.n_steps)
        af = T0.alpha_fields(row, x0).reshape(self.n_steps, -1) * mask.unsqueeze(-1)
        npx = int(y.reshape(-1, 3).shape[0])
        idx = torch.arange(0, npx, max(1, npx // N_PIX))[:N_PIX]
        d = dict(
            xb=x0.reshape(-1, 3)[idx].to('cuda:0').unsqueeze(0),
            yb=y.reshape(-1, 3)[idx].to('cuda:0').unsqueeze(0),
            alphas=af[:, idx].to('cuda:0').unsqueeze(0),
            depth=torch.tensor([depth], device='cuda:0'), d_int=depth,
            sc=torch.tensor([float(row['calib']['s'])], device='cuda:0'),
            rows=self.inv.for_row(row, x0, mask).to('cuda:0'))
        d['union'] = ST.union_mask_of(d['alphas'])
        fk = T0.feature_key(dict(id=sid, depth=depth))
        feats = {}
        for f in self.shard2feat.get(e['shard'], []):
            blob = torch.load(f, map_location='cpu')['features']
            if fk in blob:
                feats[fk] = blob[fk]; break
        assert fk in feats, f'{key}: pooled c 不在 shard {e["shard"]} 的特征缓存里'
        d['c'] = T0.gather_feats(feats, [fk], 0, 'cuda:0')
        # e*：slot 序（m=1..6），chain k = cot_step_to_chain_k[m]
        m2k = e['cot_step_to_chain_k']
        d['estar'] = torch.stack([self.targets[self.row_of[e['chain'][int(m2k[str(m)])]['lut']]]
                                  for m in range(1, 7)], 0)
        self.keycache[key] = d
        return d

    @torch.no_grad()
    def roll(self, d, edits):
        base = self.model.cond.base(d['c'])
        out, _ = SS.rollout_with_metrics(
            self.model, base, d['yb'], ST.compose_alpha_hat(d['alphas']), d['alphas'], d['union'],
            d['depth'], d['sc'], self.path_mode, SS.stages_for(self.path_mode, d['d_int']), self.nfe,
            clamp_lo=self.lo, clamp_hi=self.hi, edits=edits)
        return out

    @torch.no_grad()
    def _a_inj_guard(self):
        keys = list(self.index)[:2]
        pre = []
        for k in keys:
            d = self.key_data(k)
            pre.append((k, d, self.roll(d, d['rows'].unsqueeze(0))))
        # 切到注入路径
        self.model.edit_descriptor = lambda src: src
        ident = nn.Identity()
        self.model.bk.core.edit_enc = ident
        object.__setattr__(self.model.cond, 'edit_enc', ident)
        self.model.edit_condition = 'predicted_lut'
        self.model.edit_contract = 'predicted_lut'
        for k, d, out_a in pre:
            lat = torch.stack([self.real_edit_enc(self.model.inv_table[d['rows'][m]].unsqueeze(0))[0].detach()
                               for m in range(d['rows'].shape[0])], 0)
            out_b = self.roll(d, lat.unsqueeze(0))
            dm = float((out_a - out_b).abs().max())
            if dm != 0.0:
                raise SystemExit(f'A-inj FAILED on {k}: 注入路径与 oracle_lut 路径差 {dm}（必须为 0）')
        print(f'[c4] A-inj PASS（{len(pre)} 键逐位相等）', flush=True)
        _log(dict(t=time.time(), event='a_inj_pass', keys=keys, n_pix=N_PIX,
                  targets_sha=self.targets_sha, identity_linf8=IDENTITY_LINF8))

    @torch.no_grad()
    def readout(self, rows):
        """rows: [(key, image_path, instruction, response_token_ids)] -> (N,6,128) slot 序 latent + ok 标志。"""
        tr = _TRAINER['t']
        proc = tr.template.processor
        tok = proc.tokenizer
        stage_ids = Q.stage_token_ids(tok)
        mm = Q.resolve_mm_model(tr.accelerator.unwrap_model(tr.model))
        dtype = next(mm.parameters()).dtype
        lat, ok = [], []
        from PIL import Image
        for i in range(0, len(rows), READOUT_CHUNK):
            for key, img_path, instr, gen in rows[i:i + READOUT_CHUNK]:
                try:
                    img = Image.open(img_path).convert('RGB')
                    enc = proc(text=[QT.prompt_text(instr)], images=[img], do_resize=False, return_tensors='pt')
                    ids = enc['input_ids'].to('cuda:0')
                    full = torch.cat([ids[0], torch.tensor(gen, device='cuda:0')]).unsqueeze(0)
                    o = mm(input_ids=full, attention_mask=torch.ones_like(full),
                           pixel_values=enc['pixel_values'].to('cuda:0', dtype),
                           image_grid_thw=enc['image_grid_thw'].to('cuda:0'), return_dict=True)
                    h = o.last_hidden_state.float()[0]
                    spans_rel, missing = QT.spans_from_generated(tok, list(gen), stage_ids, include_stage_token=False)
                    n_p = ids.shape[1]
                    spans_abs = [None if s is None else (n_p + s[0], n_p + s[1]) for s in spans_rel]
                    z = QT.span_pool(h, spans_abs)
                    lat.append(self.adapter(z.unsqueeze(0).to('cuda:0'))[0])
                    ok.append(len(missing) == 0)
                except Exception as e:  # 单样本失败不得毁掉整步
                    lat.append(torch.zeros(6, 128, device='cuda:0')); ok.append(False)
                    _log(dict(t=time.time(), event='readout_error', key=key, err=repr(e)[:200]))
        return torch.stack(lat, 0), ok


CTX = _Ctx()
_STEP: dict = {}


def _prep(completions, kwargs):
    """同一步内只算一次读出；返回 (lat (N,6,128), ok, keys)。"""
    tr = _TRAINER.get('t')
    step = int(getattr(getattr(tr, 'state', None), 'global_step', -1))
    rtid = kwargs.get('response_token_ids') or []
    sig = (step, len(completions), sum(len(x) for x in rtid))
    if _STEP.get('sig') == sig:
        return _STEP['lat'], _STEP['ok'], _STEP['keys']
    CTX.build()
    keys = kwargs.get('key') or []
    imgs = kwargs.get('images') or []
    msgs = kwargs.get('messages') or []
    rows = []
    for i in range(len(completions)):
        img = imgs[i][0] if isinstance(imgs[i], (list, tuple)) else imgs[i]
        if isinstance(img, dict):
            img = img.get('path') or img.get('bytes')
        instr = msgs[i][0]['content'] if msgs else ''
        if isinstance(instr, str) and instr.startswith('<image>'):
            instr = instr[len('<image>'):]
        rows.append((keys[i], img, instr, rtid[i]))
    lat, ok = CTX.readout(rows)
    # 结构指标 + 硬停
    import re as _re
    orders = [[int(m.group(1)) for m in _re.finditer(r'<vr_stage_(\d)>', c)] for c in completions]
    six = sum(1 for o in orders if o == [1, 2, 3, 4, 5, 6])
    fr = kwargs.get('finish_reason') or []
    trunc = sum(1 for f in fr if f == 'length')
    rate = six / max(1, len(completions))
    _STEP.update(sig=sig, lat=lat, ok=ok, keys=keys, six_rate=rate, trunc=trunc, step=step, orders=orders)
    hist = _STEP.setdefault('hist', [])
    hist.append(rate)
    if len(hist) >= 2 and hist[-1] < STOP_THRESH and hist[-2] < STOP_THRESH:
        if tr is not None and hasattr(tr, 'control'):
            tr.control.should_training_stop = True
        _log(dict(t=time.time(), step=step, event='HARD_STOP',
                  reason=f'六段有序率连续 2 点 < {STOP_THRESH}', last_two=hist[-2:]))
        print(f'[c4] HARD STOP: 六段有序率连续 2 点 < {STOP_THRESH}（{hist[-2:]}）', flush=True)
    return lat, ok, keys


class LatentCosReward(ORM):

    def __call__(self, completions, **kwargs):
        lat, ok, keys = _prep(completions, kwargs)
        out = []
        for i in range(len(completions)):
            if not ok[i]:
                out.append(MISS_LATENT); continue
            e = CTX.key_data(keys[i])['estar']
            cs = torch.nn.functional.cosine_similarity(lat[i].float(), e.float(), dim=-1)
            out.append(float(cs.mean()))
        _STEP['r_latent'] = out
        return out


class ExecutorLinf8Reward(ORM):

    @torch.no_grad()
    def __call__(self, completions, **kwargs):
        lat, ok, keys = _prep(completions, kwargs)
        out, lins = [], []
        for i in range(len(completions)):
            if not ok[i]:
                out.append(MISS_EXEC); lins.append(None); continue
            d = CTX.key_data(keys[i])
            x = CTX.roll(d, lat[i].flip(0).unsqueeze(0))     # slot -> chain
            lin = float(T0.err_stats(T0.linf8(x, d['xb']))['p50'])
            lins.append(lin); out.append(-lin / IDENTITY_LINF8)
        st = _STEP
        _log(dict(t=time.time(), step=st.get('step'), n=len(completions),
                  six_rate=st.get('six_rate'), trunc=st.get('trunc'),
                  r_latent=st.get('r_latent'), r_exec=out, linf8=lins,
                  n_ok=sum(1 for x in ok if x)))
        return out


orms['vr_latent_cos'] = LatentCosReward
orms['vr_executor_linf8'] = ExecutorLinf8Reward
print('[c4] rewards registered: vr_latent_cos, vr_executor_linf8', flush=True)
