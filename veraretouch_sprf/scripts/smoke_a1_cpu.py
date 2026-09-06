#!/usr/bin/env python3
"""EPR-052 自检：解码器 A1「零初始化 ⇒ 整链逐位恒等」CPU 冒烟（10 像素，随机条件/β 场/编辑行号）。

复现的是训练入口启动断言 A1 的口径（train_sprf.py / train_sprf_bk_core4.py：`torch.equal(restore(...,nfe=1), y)`），
但不读任何数据分片：cond、y、α 场、s 均随机；编辑行号随机查真实逆表（edit.inv_cache_dir）。
两臂：FiLM 主臂（configs/decoder/clut_full.toml）与 BK-FULL（configs/decoder/bkfull_adagn_ff_affhead.toml）。
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import torch
from veraretouch_sprf import configs as CF, _paths as _P
from veraretouch_sprf.data import stage_targets as ST
from veraretouch_sprf.models import stage_flow as SF, stage_flow_bk4 as SF4, edit_cond as EC
from veraretouch_sprf.solver import stage_solver as SS, stage_solver_bk as SSB
from veraretouch_sprf.eval.probes.quick100_eval import MiniCfg

P = 10
torch.manual_seed(20260906)
ra = json.loads((Path(CF.load(CF.CONFIG_DIR / "decoder/clut_full.toml")["run"]["out_dir"]) / "run_args.json").read_text())
in_dim = int(ra["model"]["in_dim"]); n_steps = int(ra["data_law"]["n_steps"])
res = {}
for name, cfg_rel, MOD, SOL in (("clut_full", "decoder/clut_full.toml", SF, SS),
                                ("bkfull_adagn_ff_affhead", "decoder/bkfull_adagn_ff_affhead.toml", SF4, SSB)):
    d = CF.load(CF.CONFIG_DIR / cfg_rel)
    mc = MiniCfg(d)
    Model = SF4.BkSprfModel if MOD is SF4 else SF.SprfModel
    model = Model(in_dim, mc, n_steps, d["flow"]["alpha_mode"], d["data"]["depth_values"])
    inv = EC.InvLutSource(d["edit"]["inv_cache_dir"], int(d["edit"]["grid"]))
    model.load_inv_table(inv.load_table())
    model.eval()
    n_lut = int(model.inv_table.shape[0]) - 1
    y = torch.rand(1, P, 3)
    alphas = torch.rand(1, n_steps, P) * (torch.rand(1, n_steps, P) > 0.3)
    union = ST.union_mask_of(alphas)
    batch = SOL.ProductionBatch(cond=torch.randn(1, in_dim), y=y, alphas=alphas, union=union,
                                depth=torch.tensor([n_steps]), s=torch.tensor([0.7]))
    edits = torch.randint(0, n_lut, (1, n_steps), dtype=torch.long)
    with torch.no_grad():
        out = SOL.restore(model, batch, d["flow"]["path_mode"], 1, n_steps,
                          float(d["solver"]["clamp_lo"]), float(d["solver"]["clamp_hi"]),
                          edits=edits, oracle_ok=True)
    eq = bool(torch.equal(out, y))
    res[name] = dict(a1_bit_exact=eq, abs_max=float((out - y).abs().max()), pixels=P, n_steps=n_steps,
                     in_dim=in_dim, params=int(sum(p.numel() for p in model.parameters())),
                     union_active_px=int((union > 0).sum()))
    print(f"[A1 smoke] {name}: torch.equal(restore(nfe=1), y) = {eq}  abs_max={res[name]['abs_max']:.3e} "
          f"params={res[name]['params']:,} active_px={res[name]['union_active_px']}/{P}", flush=True)
    if not eq:
        sys.exit(f"A1 FAILED on {name}")
print(json.dumps(res, indent=1))
