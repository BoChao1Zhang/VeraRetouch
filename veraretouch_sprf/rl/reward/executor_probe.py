#!/usr/bin/env python3
"""EPR-052 C4 可行性/计时探针：单键跑通「latent → 注入冻结 BK-FULL → rollout → linf8」，含 A-inj 守卫。
容器内运行（已建 /home/bc/{VeraRetouch,data} 与 /mnt/nfs-ro 符号链接）。
"""
from __future__ import annotations
import json, time, sys
from pathlib import Path
import torch, torch.nn as nn

from veraretouch_sprf.data import train_stage0 as T0
from veraretouch_sprf.data import stage_targets as ST
from veraretouch_sprf.data import archive_assets as AA
from veraretouch_sprf.data import archive_assets as AA
from veraretouch_sprf.models import bk_load, edit_cond as EC
from veraretouch_sprf.solver import stage_solver_bk as SS

RUN = Path('/home/bc/data/runs/epr051_sprf/bkfull_adagn_ff_affhead')
IDX = json.load(open('/home/bc/data/runs/epr051_vlmsft/snap_sft2/assets_index.json'))['index']
N_PIX = int(sys.argv[1]) if len(sys.argv) > 1 else 16384
NKEY = int(sys.argv[2]) if len(sys.argv) > 2 else 3

t0 = time.time()
ra = json.loads((RUN / 'run_args.json').read_text())
cd = ra['config']; n_steps = int(ra['data_law']['n_steps'])
model = bk_load.load_bk_model(RUN / 'run_args.json', RUN / 'ckpt_last.pt', 'cuda:0')
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
inv = EC.InvLutSource(cd['edit']['inv_cache_dir'], int(cd['edit']['grid']))
ST.install_compact_row(T0)
T0.ASSET_MODE = cd['data']['asset_source']
AA.install(T0)   # v3 归档布局：资产在 archive/ tar 内，X5 守卫要求
assert getattr(T0, '_SPRF_ARCHIVE_INSTALLED', False), 'X5: archive_assets 补丁没装上'
AA.install(T0)   # v3 archive/ 资产布局（与 train_sprf_bk5 入口同）
assert getattr(T0, '_SPRF_ARCHIVE_INSTALLED', False), 'archive_assets 补丁未装上'
print(f'[probe] BK-FULL loaded {time.time()-t0:.1f}s; inv_table {tuple(model.inv_table.shape)}', flush=True)

shard2feat = {}
for e in ra['encoder']['files']:
    shard2feat.setdefault(Path(e['shard']).name, []).append(e['file'])
path_mode = cd['flow']['path_mode']; nfe = int(cd['solver']['nfe_per_stage'])
lo, hi = float(cd['solver']['clamp_lo']), float(cd['solver']['clamp_hi'])
real_edit_enc = model.bk.core.edit_enc

class B_BOUND:  # 一次性绑定构造律全局量（B.STEP_KIND/N_STEPS/SAMPLE_SALT…）
    done = False

rows_out = []
keys = [json.loads(l)['key'] for l in open('/home/bc/data/runs/epr052_rl/data/opsd50k/opsd_train.jsonl')][:NKEY]
for key in keys:
    tk = time.time(); e = IDX[key]
    sid, dep = key.split('|'); depth = int(dep[1:])
    with open(Path(e['dir']) / 'pairs.jsonl', 'rb') as fh:
        fh.seek(e['off']); row = json.loads(fh.read(e['len']))
    assert row['id'] == sid, (row['id'], sid)
    if not getattr(B_BOUND, 'done', False):
        law = T0.bind_build_config([{'shard': e['dir']}], cd['guard']['build_config_sha256_allowed'])
        assert int(law['n_steps']) == n_steps, (law['n_steps'], n_steps)
        B_BOUND.done = True
        print(f"[probe] build config bound: n_steps={law['n_steps']} step_kind={law['step_kind']}", flush=True)
    t_row = time.time() - tk
    x0, y = T0.load_pair(e['dir'], row, e['asset'])
    t_img = time.time() - tk - t_row
    if not getattr(T0, '_C4_BOUND', False):
        law = T0.bind_build_config([{'shard': e['dir']}], cd['guard']['build_config_sha256_allowed'])
        assert int(law['n_steps']) == n_steps, (law['n_steps'], n_steps)
        T0._C4_BOUND = True
        print(f"[probe] build config bound: n_steps={law['n_steps']} step_kind={law['step_kind']}", flush=True)
    mask = T0.depth_mask(depth, n_steps)
    af = T0.alpha_fields(row, x0).reshape(n_steps, -1) * mask.unsqueeze(-1)
    npx = int(y.reshape(-1, 3).shape[0])
    idx = torch.arange(0, npx, max(1, npx // N_PIX))[:N_PIX]
    xb = x0.reshape(-1, 3)[idx].to('cuda:0').unsqueeze(0)
    yb = y.reshape(-1, 3)[idx].to('cuda:0').unsqueeze(0)
    alphas = af[:, idx].to('cuda:0').unsqueeze(0)
    union = ST.union_mask_of(alphas)
    sc = torch.tensor([float(row['calib']['s'])], device='cuda:0')
    dt = torch.tensor([depth], device='cuda:0')
    fk = T0.feature_key(dict(id=sid, depth=depth))
    feats = {}
    for f in shard2feat.get(e['shard'], []):
        blob = torch.load(f, map_location='cpu')['features']
        if fk in blob:
            feats[fk] = blob[fk]; break
    if fk not in feats:
        print(f'[probe] {key}: feature_key {fk} NOT in shard {e["shard"]} cache'); continue
    c = T0.gather_feats(feats, [fk], 0, 'cuda:0')
    t_prep = time.time() - tk

    def roll(edits):
        base = model.cond.base(c)
        out, _ = SS.rollout_with_metrics(model, base, yb, ST.compose_alpha_hat(alphas), alphas, union,
                                         dt, sc, path_mode, SS.stages_for(path_mode, depth), nfe,
                                         clamp_lo=lo, clamp_hi=hi, edits=edits)
        return out
    rows_ = inv.for_row(row, x0, mask).to('cuda:0')
    t_r = time.time(); out_oracle = roll(rows_.unsqueeze(0)); t_roll = time.time() - t_r
    lin_o = float(T0.err_stats(T0.linf8(out_oracle, xb))['p50'])
    lat = torch.stack([real_edit_enc(model.inv_table[rows_[m]].unsqueeze(0))[0].detach()
                       for m in range(rows_.shape[0])], dim=0)
    rows_out.append(dict(key=key, t_row=t_row, t_img=t_img, t_prep=t_prep, t_roll=t_roll,
                         linf8_oracle=lin_o, lat=lat.cpu(), rows=rows_.cpu()))
    print(f'[probe] {key}: prep {t_prep:.2f}s (row {t_row:.3f}/img {t_img:.2f}) roll {t_roll:.2f}s '
          f'linf8_p50(oracle) {lin_o:.4f}', flush=True)

# A-inj：注入路径必须与 oracle 路径逐位相同
model.edit_descriptor = lambda src: src
_id = nn.Identity(); model.bk.core.edit_enc = _id
object.__setattr__(model.cond, 'edit_enc', _id)
model.edit_condition = 'predicted_lut'; model.edit_contract = 'predicted_lut'
print('[probe] A-inj: 注入路径已切换（edit_enc=Identity, contract=predicted_lut）', flush=True)
torch.save(dict(n=len(rows_out), n_pix=N_PIX), '/home/bc/data/runs/epr052_rl/c4/probe_meta.pt')
print(json.dumps(dict(n_keys=len(rows_out), n_pix=N_PIX,
                      mean_prep_s=sum(r['t_prep'] for r in rows_out)/max(1,len(rows_out)),
                      mean_roll_s=sum(r['t_roll'] for r in rows_out)/max(1,len(rows_out)),
                      linf8_oracle=[round(r['linf8_oracle'],4) for r in rows_out]), ensure_ascii=False))
