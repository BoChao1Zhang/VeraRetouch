#!/usr/bin/env python3
# 源自 experiments/prs/EPR-051_masked-restore-production/stage0/sprf/bk_load.py @ 2026-09-06（原件未入 git；原 sha256 见 veraretouch_sprf/PROVENANCE.md），逐字复制/仅改导入与路径解析（EPR-052）。
"""EPR-051 · BK 臂模型装载（分支文件，供 SFT/adapter 侧 `eval_vlmadapt` 的注入点与 A-inj 守卫使用）。

    from veraretouch_sprf.models.bk_load import load_bk_model
    model = load_bk_model(run_args_path, ckpt_path, device)   # -> stage_flow_bk*.BkSprfModel（eval 模式）

返回对象与 `stage_flow.SprfModel` 同名的四个接口原样保留：
  model.cond.edit_enc      编辑编码器 nn.Sequential(19652→512→512→128)（`model.cond` 即 backend，
                           其 `.edit_enc` 是 FullCondCore.edit_enc 的别名属性；ditblk 臂为 DitBlkBackend.edit_enc）
  model.edit_descriptor    (B,) 或 (B,K) long 逆表行号 -> (…, 19652) 描述子（SprfModel 原件）
  model.edit_condition     "inv_lut"
  model.inv_table          (N_lut+1, 19652) fp32 非持久 buffer（末行为全零 null 描述子，`model.edit_null_row`）
其余：`model.cond.base(c, edits_desc)`、`model(base, z, y, ahat, beta, alphas, lam, m, depth, s, None, edit_m)`、
`model.null_edit / roll_edit / stage_ids / features / param_counts`。

装载步骤（与训练入口 core*.main 逐字同源）：
  1. run_args.json → config_path → T0.Cfg（实验语义键全在 TOML）；
  2. 按 [bk] 选模型模块：adagn_ff / adagn_ff_affhead → stage_flow_bk4；canonfilm / adagn → stage_flow_bk3；
     ditblk 且有 bk.hidden → stage_flow_bk2；其余六臂 → stage_flow_bk；
  3. BkSprfModel(in_dim, cfg, n_steps, alpha_mode, depth_values)，其中 in_dim / n_steps / depth_values 取
     run_args.json 的 `model.in_dim` / `data_law.n_steps` / `config.data.depth_values`；
  4. 逆表：edit_cond.InvLutSource(cfg.edit.inv_cache_dir, cfg.edit.grid) → assert_fingerprint(build 工具 sha /
     build config sha / bank_dir) → model.load_inv_table(inv.load_table())；
  5. torch.load(ckpt)["model"] → load_state_dict(strict=True)；model.eval()。
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
from veraretouch_sprf import _paths as _P  # EPR-052：替代原 sys.path 注入块
_P.ensure_sys_path()
STAGE0 = _P.STAGE0
REPO = _P.REPO

from veraretouch_sprf.data import train_stage0 as T0                        # noqa: E402
from veraretouch_sprf.models import edit_cond as EC                           # noqa: E402
import epr050_build_degradation as B             # noqa: E402


def _pick_model_module(bk: dict):
    arm = bk.get("arm")
    if arm in ("adagn_ff", "adagn_ff_affhead"):
        from veraretouch_sprf.models import stage_flow_bk4 as SF
    elif arm in ("canonfilm", "adagn"):
        from veraretouch_sprf.models import stage_flow_bk3 as SF
    elif arm == "ditblk" and "hidden" in bk:
        import stage_flow_bk2 as SF
    else:
        from veraretouch_sprf.models import stage_flow_bk as SF
    return SF


def load_bk_model(run_args_path, ckpt_path, device="cuda:0", strict: bool = True,
                  verify_fingerprint: bool = True):
    ra = json.loads(Path(run_args_path).read_text())
    cfg_path = Path(ra["config_path"])
    if not cfg_path.is_file():
        raise FileNotFoundError(f"run_args 指向的 config 不存在: {cfg_path}")
    cfg = T0.Cfg(cfg_path)
    if ra["frozen_sha256"]["config"] != cfg.sha256:
        raise RuntimeError(f"config sha 漂移: run_args {ra['frozen_sha256']['config'][:12]} != 盘上 {cfg.sha256[:12]}")
    bk = cfg.d.get("bk") or {}
    SF = _pick_model_module(bk)
    in_dim = int(ra["model"]["in_dim"])
    n_steps = int(ra["data_law"]["n_steps"])
    depth_values = cfg.list_("data", "depth_values", int)
    alpha_mode = cfg.str_("flow", "alpha_mode")
    model = SF.BkSprfModel(in_dim, cfg, n_steps, alpha_mode, depth_values).to(device)
    espec = SF.edit_spec(cfg)
    if espec["condition"] != "inv_lut":
        raise RuntimeError("BK 臂均为 inv_lut(oracle_lut) 契约")
    inv = EC.InvLutSource(cfg.str_("edit", "inv_cache_dir"), espec["grid"])
    if verify_fingerprint:
        inv.assert_fingerprint(T0.sha256_file(Path(B.__file__)),
                               T0.sha256_file(Path(ra["data_law"]["config_path"])),
                               cfg.str_("data", "lut_bank_dir"))
    model.load_inv_table(inv.load_table())
    ck = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(ck["model"], strict=strict)
    model.eval()
    # `model.cond.edit_enc` 同名接口：backend 的编辑编码器在 `bk.core.edit_enc`（ditblk 为 `bk.edit_enc`）。
    # 用 object.__setattr__ 挂别名（不注册子模块 ⇒ state_dict / parameters() 不变，只是同一对象的第二个名字）。
    if not hasattr(model.cond, "edit_enc"):
        object.__setattr__(model.cond, "edit_enc", model.edit_encoder)
    assert model.cond.edit_enc is model.edit_encoder
    model.bk_ckpt_step = int(ck.get("step", -1))
    model.bk_ckpt_path = str(ckpt_path)
    return model


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-args", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--device", default="cuda:0")
    a = ap.parse_args()
    m = load_bk_model(a.run_args, a.ckpt, a.device)
    print(dict(arm=m.arm, ckpt_step=m.bk_ckpt_step, edit_condition=m.edit_condition,
               edit_enc=str(m.cond.edit_enc.__class__.__name__),
               edit_enc_params=sum(p.numel() for p in m.cond.edit_enc.parameters()),
               inv_table=tuple(m.inv_table.shape), edit_null_row=m.edit_null_row,
               params=m.param_counts()["total"]))
