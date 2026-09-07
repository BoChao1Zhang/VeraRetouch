"""EPR-052 C4：GRPO 双奖励插件（latent 余弦 + 执行器 linf8），供 `--external_plugins` 加载。

注册两个 ORM：
  `vr_latent`   生成 CoT → 按 <vr_stage_m> 定位六段 → span mean-pool（**当前策略**的隐状态）→ 冻结 adapter
                （adapt_s2fb/ckpt_epoch1/adapter.pt）→ 128 维 → 对该 key 的预建目标 e*（slot 序）算六槽平均余弦。
  `vr_executor` 同一 latent 注入**冻结 BK-FULL**（bkfull_adagn_ff_affhead/ckpt_last.pt）→ rollout → linf8 p50
                → 奖励 = −linf8 / 23.0（23.0 = held-out d6 的 identity p50，见 REPORT_STATUS §3.4 口径）。

预注册口径（失败样本）：
  * 某段缺失/解析不出 ⇒ 该槽余弦记 **0.0**（不是 −1），六槽平均因此被拉低；`n_missing` 逐步落盘。
  * 六段完全不可用（无法注入）⇒ 执行器奖励记 **−1.0**（= identity 水平，即「没比不改好」）。
读出口径差异（**必须并读**）：在线主判据 `dump_readout` 用「当前 ckpt 基座 + Stage-2 LoRA + adapter」；
本奖励用「当前策略（**无 LoRA**）+ 同一 adapter」。初始化时在 GT 文本上打印两者可比性所需的数字（见 `--vr-selftest`）。
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import torch
import torch.nn as nn

from swift.rewards import ORM, orms

from veraretouch_sprf.data import archive_assets as AA
from veraretouch_sprf.data import cot_text as C
from veraretouch_sprf.data import stage_targets as ST
from veraretouch_sprf.data import train_stage0 as T0
from veraretouch_sprf.models import bk_load
from veraretouch_sprf.models import edit_cond as EC
from veraretouch_sprf.models.vlm.adapter import Adapter
from veraretouch_sprf.solver import stage_solver_bk as SS

R_VLM = Path(os.environ.get('VR_RUNS_VLM', '/home/bc/data/runs/epr051_vlmsft'))
R_SPRF = Path(os.environ.get('VR_RUNS_SPRF', '/home/bc/data/runs/epr051_sprf'))
BK_RUN = R_SPRF / 'bkfull_adagn_ff_affhead'
ADAPT = Path(os.environ.get('VR_ADAPT', str(R_VLM / 'adapt_s2fb/ckpt_epoch1')))
IDENTITY_P50 = float(os.environ.get('VR_IDENTITY_P50', '23.0'))
N_PIX = int(os.environ.get('VR_REWARD_NPIX', '16384'))
OUT = os.environ.get('VR_REWARD_OUT', 'reward_log.jsonl')
STAGE_IDS_ENV = os.environ.get('VR_STAGE_IDS', '151669,151670,151671,151672,151673,151674')
STAGE_IDS = [int(x) for x in STAGE_IDS_ENV.split(',')]
MISSING_COS = float(os.environ.get('VR_MISSING_COS', '0.0'))
FAIL_EXEC = float(os.environ.get('VR_FAIL_EXEC', '-1.0'))

_S: dict = dict(trainer=None, bk=None, inv=None, cd=None, n_steps=6, adapter=None,
                targets=None, row_of=None, idx=None, keyctx={}, shard2feat=None,
                latcache={}, bound=False, feats={})


# --------------------------------------------------------------------------- #
# trainer 句柄：读出必须用**当前策略**权重，故不另加载一份模型
# --------------------------------------------------------------------------- #
def _install_trainer_hook():
    from swift.rlhf_trainers import grpo_trainer as GT
    orig = GT.GRPOTrainer.__init__

    def patched(self, *a, **k):
        orig(self, *a, **k)
        _S['trainer'] = self
        print('[c4_reward] trainer handle captured', flush=True)

    GT.GRPOTrainer.__init__ = patched


_install_trainer_hook()


def _lazy_init():
    if _S['bk'] is not None:
        return
    t0 = time.time()
    ra = json.loads((BK_RUN / 'run_args.json').read_text())
    _S['cd'] = cd = ra['config']
    _S['n_steps'] = int(ra['data_law']['n_steps'])
    m = bk_load.load_bk_model(BK_RUN / 'run_args.json', BK_RUN / 'ckpt_last.pt', 'cuda:0')
    m.eval()
    for p in m.parameters():
        p.requires_grad_(False)
    _S['bk'] = m
    _S['real_edit_enc'] = m.bk.core.edit_enc
    _S['inv'] = EC.InvLutSource(cd['edit']['inv_cache_dir'], int(cd['edit']['grid']))
    ST.install_compact_row(T0)
    T0.ASSET_MODE = cd['data']['asset_source']
    AA.install(T0)
    assert getattr(T0, '_SPRF_ARCHIVE_INSTALLED', False), 'archive_assets 未装上'
    _S['shard2feat'] = s2f = {}
    for e in ra['encoder']['files']:
        s2f.setdefault(Path(e['shard']).name, []).append(e['file'])
    _S['idx'] = json.load(open(R_VLM / 'snap_sft2/assets_index.json'))['index']
    tl = torch.load(R_VLM / 'targets_bkfull/target_latents_bkfull.pt', map_location='cuda:0')
    _S['targets'] = tl['target_latents']
    names = list(json.loads((Path(cd['edit']['inv_cache_dir']) / 'index.json').read_text())['names'])
    _S['row_of'] = {n: i for i, n in enumerate(names)}
    ab = torch.load(ADAPT / 'adapter.pt', map_location='cpu')
    ad = Adapter(in_dim=ab['in_dim'], hidden=ab['hidden']).to('cuda:0')
    ad.load_state_dict(ab['state_dict'])
    ad.eval()
    for p in ad.parameters():
        p.requires_grad_(False)
    _S['adapter'] = ad
    print(f'[c4_reward] init {time.time()-t0:.1f}s | bk inv_table {tuple(m.inv_table.shape)} '
          f'| targets {tuple(_S["targets"].shape)} | adapter in{ab["in_dim"]}/h{ab["hidden"]}', flush=True)


def _keyctx(key: str):
    """按 key 缓存执行器所需张量（journal 行 / 图 / α 场 / c 特征 / e* / chain 行号）。"""
    if key in _S['keyctx']:
        return _S['keyctx'][key]
    _lazy_init()
    e = _S['idx'][key]
    cd = _S['cd']
    n_steps = _S['n_steps']
    sid, dep = key.split('|')
    depth = int(dep[1:])
    with open(Path(e['dir']) / 'pairs.jsonl', 'rb') as fh:
        fh.seek(e['off'])
        row = json.loads(fh.read(e['len']))
    assert row['id'] == sid, (row['id'], sid)
    if not _S['bound']:
        law = T0.bind_build_config([{'shard': e['dir']}], cd['guard']['build_config_sha256_allowed'])
        assert int(law['n_steps']) == n_steps
        _S['bound'] = True
    x0, y = T0.load_pair(e['dir'], row, e['asset'])
    mask = T0.depth_mask(depth, n_steps)
    af = T0.alpha_fields(row, x0).reshape(n_steps, -1) * mask.unsqueeze(-1)
    npx = int(y.reshape(-1, 3).shape[0])
    sel = torch.arange(0, npx, max(1, npx // N_PIX))[:N_PIX]
    fk = T0.feature_key(dict(id=sid, depth=depth))
    if fk not in _S['feats']:
        for f in _S['shard2feat'].get(e['shard'], []):
            blob = torch.load(f, map_location='cpu')['features']
            if fk in blob:
                _S['feats'][fk] = blob[fk]
                break
    assert fk in _S['feats'], f'{key}: 特征缓存缺 {fk}'
    est = torch.stack([_S['targets'][_S['row_of'][e['chain'][e['cot_step_to_chain_k'][str(m)]]['lut']]]
                       for m in range(1, 7)], 0)          # slot 序 e*
    ctx = dict(xb=x0.reshape(-1, 3)[sel].to('cuda:0').unsqueeze(0),
               yb=y.reshape(-1, 3)[sel].to('cuda:0').unsqueeze(0),
               alphas=af[:, sel].to('cuda:0').unsqueeze(0),
               depth=torch.tensor([depth], device='cuda:0'),
               d_int=depth, sc=torch.tensor([float(row['calib']['s'])], device='cuda:0'),
               c=T0.gather_feats({fk: _S['feats'][fk]}, [fk], 0, 'cuda:0'),
               rows=_S['inv'].for_row(row, x0, mask).to('cuda:0'), estar=est)
    ctx['union'] = ST.union_mask_of(ctx['alphas'])
    _S['keyctx'][key] = ctx
    return ctx


@torch.no_grad()
def _rollout(ctx, edits):
    m = _S['bk']; cd = _S['cd']
    base = m.cond.base(ctx['c'])
    out, _ = SS.rollout_with_metrics(m, base, ctx['yb'], ST.compose_alpha_hat(ctx['alphas']),
                                     ctx['alphas'], ctx['union'], ctx['depth'], ctx['sc'],
                                     cd['flow']['path_mode'],
                                     SS.stages_for(cd['flow']['path_mode'], ctx['d_int']),
                                     int(cd['solver']['nfe_per_stage']),
                                     clamp_lo=float(cd['solver']['clamp_lo']),
                                     clamp_hi=float(cd['solver']['clamp_hi']), edits=edits)
    return out


def _switch_to_injection():
    """A-inj 守卫后把模型切到注入模式（edit_enc=Identity、contract=predicted_lut）。"""
    if _S.get('injected'):
        return
    m = _S['bk']
    guard_keys = list(_S['keyctx'])[:2]
    pre = []
    for k in guard_keys:
        ctx = _keyctx(k)
        pre.append((ctx, _rollout(ctx, ctx['rows'].unsqueeze(0))))
    fe = _S['real_edit_enc']
    m.edit_descriptor = lambda src: src
    ident = nn.Identity()
    m.bk.core.edit_enc = ident
    object.__setattr__(m.cond, 'edit_enc', ident)
    m.edit_condition = 'predicted_lut'
    m.edit_contract = 'predicted_lut'
    for ctx, out_a in pre:
        lat = torch.stack([fe(m.inv_table[ctx['rows'][i]].unsqueeze(0))[0].detach()
                           for i in range(ctx['rows'].shape[0])], 0)
        out_b = _rollout(ctx, lat.unsqueeze(0))
        d = float((out_a - out_b).abs().max())
        if d != 0.0:
            raise SystemExit(f'A-inj FAILED: 注入路径与 oracle_lut 路径差 {d}（必须为 0）')
    _S['injected'] = True
    print(f'[c4_reward] A-inj PASS（{len(pre)} 键逐位相同）', flush=True)


@torch.no_grad()
def _readout(rows) -> list:
    """rows = [(messages, images, response_token_ids)] -> [(6,128) slot 序 latent, missing 段号]"""
    tr = _S['trainer']
    assert tr is not None, 'trainer 句柄未捕获'
    from veraretouch_sprf.models.vlm import q3vl_common as Q
    model = tr.accelerator.unwrap_model(tr.model)
    mm = Q.resolve_mm_model(model)
    tmpl = tr.template
    dev = next(model.parameters()).device
    dt = next(model.parameters()).dtype
    out = []
    for msgs, imgs, rid in rows:
        prompt_msgs = [m for m in msgs if m['role'] != 'assistant']
        enc = tmpl.encode({'messages': prompt_msgs, 'images': imgs})
        pid = list(enc['input_ids'])
        ids = torch.tensor(pid + list(rid), device=dev).unsqueeze(0)
        kw = dict(input_ids=ids, attention_mask=torch.ones_like(ids), return_dict=True)
        if enc.get('pixel_values') is not None:
            kw['pixel_values'] = torch.as_tensor(enc['pixel_values']).to(dev, dt)
            kw['image_grid_thw'] = torch.as_tensor(enc['image_grid_thw']).reshape(-1, 3).to(dev)
        h = mm(**kw).last_hidden_state.float()[0]
        n_p = len(pid)
        pos = {}
        for j, t in enumerate(rid):
            if t in STAGE_IDS and STAGE_IDS.index(t) + 1 not in pos:
                pos[STAGE_IDS.index(t) + 1] = j
        order = [STAGE_IDS.index(t) + 1 for t in rid if t in STAGE_IDS]
        z, miss, prev = [], [], 0
        for m_ in range(1, 7):
            if m_ in pos and pos[m_] > prev:
                z.append(h[n_p + prev: n_p + pos[m_]].mean(0))
                prev = pos[m_] + 1
            else:
                z.append(torch.zeros(h.shape[-1], device=h.device))
                miss.append(m_)
        lat = _S['adapter'](torch.stack(z, 0).unsqueeze(0))[0]
        out.append((lat, miss, order, len(rid)))
    return out


def _latents_for(kwargs, completions):
    """按 response_token_ids 缓存 latent，两个 ORM 共用一次前向。"""
    _lazy_init()
    msgs = kwargs.get('messages'); imgs = kwargs.get('images'); rids = kwargs.get('response_token_ids')
    keys = kwargs.get('key')
    n = len(completions)
    sig = [hash(tuple(r)) if r else hash(completions[i]) for i, r in enumerate(rids)]
    todo = [i for i in range(n) if sig[i] not in _S['latcache']]
    if todo:
        rows = [(msgs[i], imgs[i] if imgs else None, rids[i]) for i in todo]
        res = _readout(rows)
        for i, r in zip(todo, res):
            _S['latcache'][sig[i]] = r
    if len(_S['latcache']) > 4096:
        _S['latcache'].clear()
    return [_S['latcache'][s] for s in sig], keys


class VRLatentReward(ORM):
    """六槽平均余弦（对预建 e*，slot 序）；缺段槽记 MISSING_COS。"""

    def __call__(self, completions, **kwargs):
        lats, keys = _latents_for(kwargs, completions)
        out = []
        rec = []
        for (lat, miss, order, ntok), key in zip(lats, keys):
            est = _keyctx(key)['estar']
            cs = torch.nn.functional.cosine_similarity(lat.float(), est.float(), dim=-1)
            for m_ in miss:
                cs[m_ - 1] = MISSING_COS
            out.append(float(cs.mean()))
            rec.append(dict(key=key, cos=float(cs.mean()), n_missing=len(miss),
                            six_ok=(order == [1, 2, 3, 4, 5, 6]), n_tok=ntok, order=order))
        with open(OUT, 'a') as f:
            f.write(json.dumps(dict(t=time.time(), orm='vr_latent', rewards=out,
                                    n_missing=[r['n_missing'] for r in rec],
                                    six_ok=[r['six_ok'] for r in rec], n_tok=[r['n_tok'] for r in rec],
                                    orders=[r['order'] for r in rec], keys=list(keys))) + '\n')
        return out


class VRExecutorReward(ORM):
    """−linf8_p50 / IDENTITY_P50；不可用样本记 FAIL_EXEC。"""

    def __call__(self, completions, **kwargs):
        lats, keys = _latents_for(kwargs, completions)
        _switch_to_injection()
        out, lins = [], []
        for (lat, miss, order, ntok), key in zip(lats, keys):
            if len(miss) == 6:
                out.append(FAIL_EXEC); lins.append(None); continue
            ctx = _keyctx(key)
            chain = lat.flip(0).to('cuda:0')          # slot 序 -> chain 序（A-lat 口径）
            o = _rollout(ctx, chain.unsqueeze(0))
            lin = float(T0.err_stats(T0.linf8(o, ctx['xb']))['p50'])
            lins.append(lin)
            out.append(-lin / IDENTITY_P50)
        with open(OUT, 'a') as f:
            f.write(json.dumps(dict(t=time.time(), orm='vr_executor', rewards=out, linf8=lins)) + '\n')
        return out


orms['vr_latent'] = VRLatentReward
orms['vr_executor'] = VRExecutorReward
print('[c4_reward] registered orms: vr_latent, vr_executor', flush=True)
